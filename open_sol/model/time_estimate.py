"""Run-time estimate of a model (time_estimation_plan): per used mesh every physics' time recipe on a TimeContext (its
solver roles through their SolverType recipes, its array bytes from the memory estimate), multiplied by the study
counts (time steps, wavelengths, iterations), priced on the machine profile. CPU, GUI-thread safe."""
import itertools
import json
import math
from dataclasses import dataclass, field, replace

import numpy as np

from ..core.time_estimate import (HostStream, PythonCalls, SolverWork, TimeEstimate, TimeItem, Transfer, combine,
                                  format_seconds)
from ..core.time_profile import load_profile, profile_from_dict, profile_path
from . import meshing
from .geometry import (bounding_box, domain_entities, domain_extent, domain_size_z, expand_entities, resolve_geometry,
                       z_extent)
from .schema import get_registry, resolve_values, resolved_solvers, solver_backend
from .spec import outer_faces

# a coupled fixed point (stationary study, background stage) solves the physics on or downstream of a coupling cycle
# this many times (guess: depends on the coupling strength); the harmonic FDTD pulse runs this fraction of max_duration
FIXED_POINT_ITERATIONS_GUESS = 5
HARMONIC_PULSE_FRACTION_GUESS = 0.3
# Jacobi-PCG condition estimate (TimeContext.jacobi_conditioning): the Jacobi-scaled diffusion operator has largest
# eigenvalue ~2 (Gershgorin) and smallest ~ (pi / (2 L))^2 / (sum over axes of 2 <1 / h^2>), L the farthest distance
# from the window to the held set (sampled on a lattice of CONDITION_SAMPLES_PER_AXIS points per axis); a held set
# small against L lowers it by its capacitance in a ball of radius L around it (3D: 12 a / (pi^2 L), 2D: 8 / (pi^2
# ln(L / a)), a its half size). CG finite-termination cap: the product of the window cells of the axes with at least
# MODE_AXIS_MIN_CELLS (a short axis adds only a few well-separated mode clusters)
JACOBI_LARGEST_EIGENVALUE = 2.0
CONDITION_SAMPLES_PER_AXIS = 17
MODE_AXIS_MIN_CELLS = 16
# work around the study (calibrate from the selftest time_split): Python calls of one build stage (a memory item:
# domain, material array, physics array, feature; + validation and estimates) and of one run's set-up and output
# (summary, digests, model.json, log); host passes over the build's bytes and over the kept bytes in results.npz
BUILD_CALLS_PER_STAGE_GUESS = 2e4
BUILD_HOST_PASSES_GUESS = 8
OUTPUT_CALLS_GUESS = 1e5
OUTPUT_HOST_PASSES_GUESS = 3
# making a 'triangles' mesh (E3; UNCALIBRATED: set from romero e2c, 302k triangles of a mirror_x half window in a ~20 s
# 'domain' stage, priced at ~40 GB/s host stream; recalibrate from the triangulate log line "mesher ..., element
# tagging ..."): the built-in mesher (PSLG, refinement rounds, smoothing, layer ring, quality, mesh arrays; compute
# bound, so host-stream equivalent bytes) per meshed P1 node and refinement round (the former 1e7 per node over a
# 40-round guess; a round passes over every node: Delaunay update, edges, encroachment, size field) + Python calls per
# round; rounds a priori (mesher_rounds); gmsh per meshed node; element tagging per element and named domain object;
# the TriangleDomain set-up per element
MESHER_HOST_BYTES_PER_NODE_ROUND = 2.5e5
MESHER_CALLS_PER_ROUND = 2e4
GMSH_HOST_BYTES_PER_NODE = 1e6
TAGGING_HOST_BYTES_PER_ELEMENT = 200.0
DOMAIN_SETUP_HOST_BYTES_PER_ELEMENT = 150.0
# snapshots of stored fields on a triangle mesh (study._SnapshotRecorder -> field_access_triangles: a raster of 4 pixels
# per element within TRIANGLE_FIELD_PIXELS when the elements fit _SNAPSHOT_PIXELS, else _SNAPSHOT_PIXELS), host bytes
# counted from fem.sampling: the PointLocator buckets per element and bucket entry (+ the quicksort of the unique
# bucket * n_elements + owner keys), per pixel the bucket lookup and the P2 basis (TriangleDomain.located keeps both:
# once per mesh and run), per locate candidate the affine map, depths, gathers and the segmented argmax (two reduceat,
# the max repeat, the index mask; no sort); per read the raster points, their hash, evaluate and the float32 copy per
# pixel, the display triangulation per element
LOCATE_CANDIDATES_PER_PIXEL = 4.0     # bucket entries per bucket: an element's box covers ~(1 + 1.4)^2 mean buckets
LOCATOR_BYTES_PER_ELEMENT = 300.0
LOCATOR_BYTES_PER_BUCKET_ENTRY = 72.0
LOCATE_BYTES_PER_PIXEL = 536.0
LOCATE_BYTES_PER_CANDIDATE = 470.0
LOCATE_CALLS_PER_CHUNK = 40
READ_BYTES_PER_PIXEL = 420.0
TRIANGULATION_BYTES_PER_ELEMENT = 200.0
READ_CALLS = 30


@dataclass(frozen=True)
class StudyCounts:
    """The study of the estimate: type name, resolved settings and StudyType.time_counts (duration, harmonic_points,
    max_duration, eigenpairs, iterations, fixed_point_limit)."""
    type: str
    values: dict
    counts: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TimeContext:
    """Third argument of PhysicsType.time_estimate: the model around the physics, its resolved solvers, its own
    memory estimate (array bytes: the per-step traffic of an explicit engine), the run backend and the study."""
    spec: object
    physics_spec: object
    parameters: dict
    registry: object
    solvers: dict
    mesh: object
    memory: object                  # core.memory_estimate.MemoryEstimate of the physics and its features
    backend: str
    study: StudyCounts
    coupled: bool = False           # an enabled coupling reads or writes this physics
    fixed_point: bool = False       # on or downstream of a cycle of enabled couplings (fixed_point_physics)
    ordering_kept: bool = False     # a later study run on an unchanged mesh: cuDSS plans reuse the kept orderings

    @property
    def name(self) -> str:
        return self.physics_spec.name

    @property
    def time_dependent(self) -> bool:
        return self.study.type == 'time_dependent'

    def count(self, name, default=None):
        return self.study.counts.get(name, default)

    def material_sets(self, property_name) -> bool:
        """As meshing.MemoryContext.material_sets (a material sets the property to anything but 0 / None)."""
        return meshing.MemoryContext(self.spec, self.physics_spec, self.parameters, self.registry,
                                     ()).material_sets(property_name)

    def solver_work(self, role, **work) -> TimeEstimate:
        """The resolved `role` solver's recipe for SolverWork(**work) (physics and role filled in, backend = the run
        backend unless given: host-only solves pass 'cpu'; ordering_kept on a later run of one mesh unless given);
        'not estimated' when the role does not resolve or its type has no recipe."""
        resolved = self.solvers.get(role)
        recipe = getattr(getattr(resolved, 'solver_type', None), 'time_estimate', None)
        if recipe is None:
            return TimeEstimate(not_estimated=(f'{role} solver',))
        work = {'backend': self.backend, 'run_backend': self.backend, 'ordering_kept': self.ordering_kept, **work}
        return recipe(resolved.values, SolverWork(physics=self.name, role=role, **work))

    def static_solves(self) -> tuple:
        """(count, count_kind, limit) of the solves of a quasi-static physics in a stationary study or the
        background stage: 1 exact off any coupling cycle (one pass converges it), else the fixed point (a guess below
        the iteration limit)."""
        if not self.fixed_point:
            return 1.0, 'exact', None
        limit = self.count('fixed_point_limit')
        guess = FIXED_POINT_ITERATIONS_GUESS if limit is None else min(FIXED_POINT_ITERATIONS_GUESS, limit)
        return float(guess), 'guess', limit

    def jacobi_conditioning(self, held) -> tuple:
        """(condition number of the Jacobi-scaled diffusion operator, CG finite-termination cap) from where the physics
        holds its unknown (core.time_estimate.HeldValues); (0, 0) = unknown on polar / spherical grids. Material
        contrast and the axisymmetric r weighting are not counted (Jacobi scaling cancels contrast to first order)."""
        axes = _window_axes(self.spec, self.parameters, self.mesh)
        if axes is None:
            return 0.0, 0.0
        lengths = [length for length, _, _ in axes]
        widths = [1.0 / math.sqrt(mean_inverse_square) for _, mean_inverse_square, _ in axes]
        long_axes = [cells for _, _, cells in axes if cells >= MODE_AXIS_MIN_CELLS]
        modes = float(math.prod(long_axes) if long_axes else max(cells for _, _, cells in axes))
        boxes = _held_boxes(self.spec, self.parameters, self.physics_spec, held, lengths)
        if not boxes:   # the corner anchor: one window cell
            boxes = [(tuple(0.0 for _ in lengths), tuple(widths))]
        farthest = _farthest_distance(boxes, lengths)
        if farthest <= 0.0:
            return 1.0, modes
        half_size = 0.5 * max(max(max(high - low for low, high in zip(*box)) for box in boxes), min(widths))
        if len(axes) == 3:
            capacitance_factor = min(1.0, 12.0 * half_size / (math.pi ** 2 * farthest))
        elif len(axes) == 2:
            capacitance_factor = min(1.0, 8.0 / (math.pi ** 2 * math.log(max(farthest / half_size, math.e))))
        else:
            capacitance_factor = 1.0
        diagonal = sum(2.0 * mean_inverse_square for _, mean_inverse_square, _ in axes)
        smallest = (math.pi / (2.0 * farthest)) ** 2 * capacitance_factor / diagonal
        return max(JACOBI_LARGEST_EIGENVALUE / smallest, 1.0), modes

    def window_cells(self) -> int:
        """Physical-window cells of the physics' mesh (3D: voxels; triangles / tetrahedra: elements)."""
        mesh = self.mesh
        if mesh.kind != 'tensor':
            return int(mesh.n_cells)
        return math.prod(mesh.window_axis_cells)

    def longest_axis(self) -> int:
        """Window cells along the longest axis (3D: z included; triangles / tetrahedra: 2 x the longest side over the
        mean element size, as solvers.pcg.resolve_max_iterations)."""
        mesh = self.mesh
        if mesh.kind != 'tensor':
            sides = [high - low for low, high in zip(mesh.extent[0::2], mesh.extent[1::2])]
            mean_size = (max(math.prod(sides), 1e-300) / max(int(mesh.n_cells), 1)) ** (1.0 / len(sides))
            return max(int(2 * math.ceil(max(sides) / mean_size)), 1)
        return max(max(mesh.window_axis_cells), 1)


def _window_axes(spec, parameters, mesh):
    """[(window length [m], length-weighted mean of 1 / width^2 [1/m^2], cells)] per model axis (triangles /
    tetrahedra: the window box at the mean P1 node spacing); None on a polar / spherical grid."""
    if (spec.domain.coordinates or 'cartesian') != 'cartesian':
        return None
    dimensions = spec.dimensions
    if mesh.kind == 'tensor':
        sides = mesh.npml_sides
        axes = []
        for axis, widths in zip('xyz'[:dimensions], (mesh.dx_cells, mesh.dy_cells, mesh.dz_cells)):
            window = np.asarray(widths, dtype=np.float64)[sides.get(f'{axis}_lo', 0):
                                                          len(widths) - sides.get(f'{axis}_hi', 0)]
            length = float(window.sum())
            axes.append((length, float(np.sum(1.0 / window)) / length, len(window)))
        return axes
    x_min, x_max, y_min, y_max = domain_extent(spec, parameters)
    lengths = [x_max - x_min, y_max - y_min] + ([domain_size_z(spec, parameters)] if dimensions == 3 else [])
    spacing = (math.prod(lengths) / max(int(mesh.n_nodes), 1)) ** (1.0 / len(lengths))
    return [(length, 1.0 / spacing ** 2, max(int(round(length / spacing)), 1)) for length in lengths]


def _held_boxes(spec, parameters, physics_spec, held, lengths) -> list:
    """[(low corner, high corner)] [m] of the held sets in the window frame: outer faces as flat boxes, objects as their
    bounding boxes clipped to the window, 'all' as the window."""
    faces = outer_faces(spec)
    window = (tuple(0.0 for _ in lengths), tuple(lengths))

    def face_box(face):
        axis = 'xyz'.index(face[0])
        low, high = list(window[0]), list(window[1])
        if face.endswith('_lo'):
            high[axis] = 0.0
        else:
            low[axis] = lengths[axis]
        return tuple(low), tuple(high)

    shapes, boxes, freed = None, [], set()
    for feature in physics_spec.features:
        if not feature.enabled:
            continue
        if feature.type in held.edge_types:
            boxes += [face_box(face) for face in faces]
            continue
        if feature.type not in held.feature_types | held.free_types:
            continue
        try:
            entities = expand_entities(spec, feature.selection)
        except (ValueError, TypeError):   # reported by validation
            continue
        if feature.type in held.free_types:
            freed.update(name for name in entities if name in faces)
            continue
        for name in entities:
            if name in faces:
                boxes.append(face_box(name))
            elif name == 'all':
                boxes.append(window)
            else:
                shapes = resolve_geometry(spec, parameters, errors={}) if shapes is None else shapes
                box = _object_box(name, shapes, lengths)
                if box is not None:
                    boxes.append(box)
    if held.outer_faces_held:
        boxes += [face_box(face) for face in faces if face not in freed]
    return boxes


def _object_box(name, shapes, lengths):
    """Bounding box of a geometry object clipped to the window (3D: with its z extent); None when it does not resolve or
    lies outside."""
    try:
        x0, y0, x1, y1 = bounding_box(name, shapes)
        low, high = [x0, y0], [x1, y1]
        if len(lengths) == 3:
            z_lo, z_hi = z_extent(name, shapes)
            low.append(z_lo)
            high.append(z_hi)
    except (KeyError, ValueError):   # a failed object (reported by the geometry checks)
        return None
    low = [min(max(value, 0.0), length) for value, length in zip(low, lengths)]
    high = [min(max(value, 0.0), length) for value, length in zip(high, lengths)]
    if any(lo > hi for lo, hi in zip(low, high)):
        return None
    return tuple(low), tuple(high)


def _farthest_distance(boxes, lengths) -> float:
    """Largest distance [m] from a window lattice point (CONDITION_SAMPLES_PER_AXIS per axis) to the nearest held box."""
    samples = [np.linspace(0.0, length, CONDITION_SAMPLES_PER_AXIS) for length in lengths]
    points = np.array(list(itertools.product(*samples)), dtype=np.float64)          # (points, dimensions)
    low = np.array([box[0] for box in boxes], dtype=np.float64)[None]                  # (1, boxes, dimensions)
    high = np.array([box[1] for box in boxes], dtype=np.float64)[None]
    outside = np.maximum(np.maximum(low - points[:, None], points[:, None] - high), 0.0)
    return float(np.sqrt((outside ** 2).sum(axis=2)).min(axis=1).max())


def study_counts(spec, parameters, registry) -> StudyCounts:
    """StudyCounts of the spec's study (time_counts of its StudyType, {} without)."""
    study_type = registry.study_type(spec.study.type)
    settings = spec.study.settings if isinstance(spec.study.settings, dict) else {}
    values = resolve_values(study_type.params, settings, parameters)
    counts = study_type.time_counts(values) if study_type.time_counts is not None else {}
    return StudyCounts(study_type.name, values, counts)


def study_runs(spec, parameters, mesh_estimate=None) -> list:
    """[(label, run spec, run parameters, mesh estimate or None)] of the study runs: ('', spec, parameters,
    mesh_estimate) without a sweep, else one value spec per sweep value ('<parameter> = <value>', its parameters
    re-evaluated and no mesh estimate: a swept size changes the mesh); with the study setting parity_sectors every run
    splits into its 2^k sector runs ('<sector>' / '<parameter> = <value> [<sector>]', same mesh). Adaptation levels
    not counted; KeyError / ValueError as study.sweep_values_of."""
    if spec.study.sweep is None:
        return _sector_runs([('', spec, parameters, mesh_estimate)])
    from .expressions import evaluate_parameters
    from .study import sweep_value_spec, sweep_values_of   # loaded with the registry already
    parameter, values = sweep_values_of(spec)
    runs = []
    for value in values:
        value_spec = sweep_value_spec(spec, parameter, value)
        runs.append((f'{parameter} = {value:g}', value_spec, evaluate_parameters(value_spec.parameters), None))
    return _sector_runs(runs)


def estimated_runs(spec, parameters, registry=None, mesh_estimate=None) -> list:
    """study_runs with every run's {mesh name: estimate} filled in (one estimate_meshes per sweep value, shared by its
    parity-sector runs): pass it as runs= to estimate_time and estimate_overheads so the meshes are estimated once."""
    registry = registry if registry is not None else get_registry()
    by_parameters, filled = {}, []
    for label, run_spec, run_parameters, run_meshes in study_runs(spec, parameters, mesh_estimate):
        if not (isinstance(run_meshes, dict) and run_meshes):
            key = id(run_parameters)   # sector runs of one value share its parameters dict (alive in this loop)
            if key not in by_parameters:
                by_parameters[key] = meshing.estimate_meshes(run_spec, run_parameters, registry)
            run_meshes = by_parameters[key]
        filled.append((label, run_spec, run_parameters, run_meshes))
    return filled


def _sector_runs(runs) -> list:
    """runs with each run whose spec requests parity_sectors replaced by its sector runs (study_sectors.
    ordered_sector_specs, as run_parity_sectors builds them); a spec without a mirror face stays one run."""
    from .study import _parity_sectors_requested   # loaded with the registry already
    from .study_sectors import ordered_sector_specs
    registry = get_registry()
    expanded = []
    for label, run_spec, run_parameters, run_meshes in runs:
        try:
            sectors = ordered_sector_specs(run_spec) if _parity_sectors_requested(run_spec, registry) else []
        except ValueError:   # no mirror face: validation reports it
            sectors = []
        if not sectors:
            expanded.append((label, run_spec, run_parameters, run_meshes))
        for sector_label, sector_spec in sectors:
            expanded.append((f'{label} [{sector_label}]' if label else sector_label, sector_spec, run_parameters,
                             run_meshes))
    return expanded


def stored_profile(backend):
    """(MachineProfile or None, reason) of load_profile; an out-of-date file still loads, marked outdated (reason
    kept: the estimate text says '(outdated <backend> profile)'; only the time_profile_current selftest judges it)."""
    profile, reason = load_profile(backend)
    if profile is None:
        try:
            with open(profile_path(backend), encoding='utf-8') as handle:
                return replace(profile_from_dict(json.load(handle)), outdated=True), reason
        except (OSError, ValueError, KeyError, TypeError):   # missing / unreadable: no estimate
            pass
    return profile, reason


def estimate_time(spec, parameters, registry=None, mesh_estimate=None, profile=None, per_run=False, shapes=None,
                  runs=None):
    """(TimeEstimate, message) of running the spec on its backend: every physics' recipe (none: 'not estimated'),
    per-step work times the steps of the study, priced on `profile` (None: the stored machine profile of the
    backend). per_run: the spec as one run (the post-run check), else every study run (sweep values on their own
    mesh, parity sectors: study_runs), summed (items labelled '<run label>: ...', no counters), times the runs of a
    parameter estimation (estimation_runs: an upper bound). shapes: the caller's resolve_geometry of these parameters
    (per_run only; None = resolved where needed); runs: estimated_runs of the spec (None = study_runs here). (None,
    reason) without a profile; model errors raise as for estimate_memory."""
    registry = registry if registry is not None else get_registry()
    backend = solver_backend(spec)
    if profile is None:
        profile, reason = stored_profile(backend)
        if profile is None:
            return None, f'{reason}: run `python -m open_sol profile` to estimate run times'
    if per_run:
        estimate = _run_estimate(spec, parameters, registry, mesh_estimate, backend, shapes=shapes)
        return estimate.priced(profile, backend), ''
    runs = study_runs(spec, parameters, mesh_estimate) if runs is None else runs
    evaluations = estimation_runs(spec)
    if len(runs) == 1 and not runs[0][0]:
        estimate = _run_estimate(spec, parameters, registry, runs[0][3], backend)
        return _repeated(estimate, evaluations).priced(profile, backend), ''
    estimates, not_estimated, seen = [], [], []
    for run_label, run_spec, run_parameters, run_meshes in runs:
        # run_study keeps the cuDSS orderings: a later run on an unchanged mesh (same sector) plans on a kept one
        run_meshes = (run_meshes if isinstance(run_meshes, dict) and run_meshes
                      else meshing.estimate_meshes(run_spec, run_parameters, registry))
        sector = run_label.rpartition('[')[2] if '[' in run_label else ('' if spec.study.sweep is not None
                                                                        else run_label)
        key = (_mesh_signature(run_spec, run_parameters, run_meshes), sector)
        kept = key[0] is not None and key in seen
        seen.append(key)
        estimate = _run_estimate(run_spec, run_parameters, registry, run_meshes, backend, ordering_kept=kept)
        # a whole sweep (or sector set) is never compared with one run's counters
        estimates.append(replace(estimate, not_estimated=(), items=tuple(
            replace(item, label=f'{run_label}: {item.label}', counter=()) for item in estimate.items)))
        for label in estimate.not_estimated:   # once, not once per value
            if label not in not_estimated:
                not_estimated.append(label)
    return _repeated(replace(combine(estimates), not_estimated=tuple(not_estimated)), evaluations).priced(
        profile, backend), ''


def estimation_runs(spec) -> int:
    """Study runs of a parameter estimation (StudySpec.estimation): max_evaluations (an upper bound) + the best-point
    run; 1 without an estimation."""
    estimation = getattr(spec.study, 'estimation', None)
    if estimation is None:
        return 1
    try:
        return max(int(estimation.max_evaluations), 0) + 1
    except (TypeError, ValueError):   # reported by validation
        return 1


def _repeated(estimate, runs) -> TimeEstimate:
    """Copy of an unpriced estimate whose items are repeated `runs` times (labels 'estimation (<runs> runs): ...',
    exact counts become upper bounds, no counters); unchanged for runs <= 1."""
    if runs <= 1:
        return estimate
    return replace(estimate, items=tuple(
        replace(item, label=f'estimation ({runs} runs): {item.label}', count=item.count * runs,
                count_kind='bounded' if item.count_kind == 'exact' else item.count_kind,
                limit=None if item.limit is None else item.limit * runs, counter=(), seconds_counter=())
        for item in estimate.items))


def _mesh_signature(spec, parameters, meshes):
    """What fixes the run's operator patterns, per mesh: an element mesh's element_mesh_key, a tensor mesh's cell
    counts; None when a key does not evaluate (the run is then never taken as an unchanged mesh)."""
    signature, shapes = [], None
    try:
        for name, mesh in sorted(meshes.items()):
            kind = getattr(mesh, 'kind', 'tensor')
            if kind in ('triangles', 'tetrahedra'):
                shapes = resolve_geometry(spec, parameters, errors={}) if shapes is None else shapes
                key = meshing.element_mesh_key(mesh.sizing, meshing.element_window_size(spec, parameters, kind), shapes)
            else:
                key = tuple(getattr(mesh, attribute, None) for attribute in ('nx', 'ny', 'nz', 'n_cells'))
            signature.append((name, kind, key))
    except (AttributeError, KeyError, ValueError, TypeError):
        return None
    return tuple(signature)


def _run_estimate(spec, parameters, registry, mesh_estimate, backend, ordering_kept=False, shapes=None) -> TimeEstimate:
    """Unpriced TimeEstimate of one study run of the spec on mesh_estimate (None / {}: estimated here, on shapes when
    given); ordering_kept: a later run on an unchanged mesh (TimeContext.ordering_kept)."""
    meshes = (mesh_estimate if isinstance(mesh_estimate, dict) and mesh_estimate
              else meshing.estimate_meshes(spec, parameters, registry, shapes))
    memory_parts = {}
    meshing.estimate_memory(spec, parameters, registry, meshes, by_physics=memory_parts, shapes=shapes)
    study = study_counts(spec, parameters, registry)
    coupled = {name for coupling in spec.couplings if coupling.enabled for name in (coupling.source, coupling.target)}
    fixed_point = fixed_point_physics(spec.couplings, registry)
    parts = []
    for mesh_name, mesh in meshes.items():
        for physics_spec in spec.physics_on(mesh_name):
            parts.append(_physics_part(spec, parameters, registry, physics_spec, mesh, memory_parts, backend, study,
                                       physics_spec.name in coupled, physics_spec.name in fixed_point, ordering_kept))
    estimate = combine((_with_study_steps(parts, study), TimeEstimate(snapshot_read_items(spec, meshes, study))))
    if study.type == 'time_dependent':
        missing = tuple(f"coupling '{coupling.name}' transfers" for coupling in spec.couplings if coupling.enabled)
        estimate = combine((estimate, TimeEstimate(not_estimated=missing)))
    return estimate


def _snapshot_count(study) -> tuple:
    """(snapshots one run records, count kind): frequency_response one per frequency ('all') or the last, none with
    'none'; any other study its final one (a guess: time series and iterations may record more)."""
    setting = study.values.get('snapshots')
    if setting == 'none':
        return 0.0, 'exact'
    if study.type == 'frequency_response':
        return (float(study.counts.get('harmonic_points', 1) or 1) if setting == 'all' else 1.0), 'exact'
    return 1.0, 'guess'


def snapshot_read_items(spec, meshes, study) -> tuple:
    """Study-time TimeItems of the stored-field snapshots on triangle meshes (constants above): per mesh the point
    location of its raster once, per snapshot and stored field one read; tensor / tetrahedron meshes not counted."""
    from ..fem.sampling import POINT_CHUNK
    from .definition_helpers import SORT_PASS_BYTES
    from .field_access_triangles import TRIANGLE_FIELD_PIXELS
    from .study import _SNAPSHOT_PIXELS
    snapshots, kind = _snapshot_count(study)
    if not snapshots:
        return ()
    mesh_of = {physics_spec.name: name for name in meshes for physics_spec in spec.physics_on(name)}
    fields = {}
    for stored in spec.study.stored_fields:
        name = mesh_of.get(str(stored).partition('.')[0])
        if name is not None and getattr(meshes[name], 'kind', 'tensor') == 'triangles':
            fields[name] = fields.get(name, 0) + 1
    items = []
    for name, count in fields.items():
        elements = float(meshes[name].n_elements)
        low, high = TRIANGLE_FIELD_PIXELS
        pixels = min(high, max(low, 4.0 * elements)) if elements <= _SNAPSHOT_PIXELS else float(_SNAPSHOT_PIXELS)
        entries = LOCATE_CANDIDATES_PER_PIXEL * elements   # bucket entries (buckets ~ elements)
        locate = (LOCATOR_BYTES_PER_ELEMENT * elements
                  + entries * (LOCATOR_BYTES_PER_BUCKET_ENTRY + SORT_PASS_BYTES * math.log2(max(entries, 2.0)))
                  + LOCATE_BYTES_PER_PIXEL * pixels + LOCATE_CANDIDATES_PER_PIXEL * pixels * LOCATE_BYTES_PER_CANDIDATE)
        label = f"snapshots on mesh '{name}': "
        items += [TimeItem(f'{label}raster point location', 'mesh', 1.0, 'exact', (
                      HostStream(locate), (LOCATE_CALLS_PER_CHUNK * math.ceil(pixels / POINT_CHUNK), HostStream(1.0)))),
                  TimeItem(f'{label}field reads ({int(pixels):,} pixels)', 'read', snapshots * count, kind, (
                      HostStream(READ_BYTES_PER_PIXEL * pixels + TRIANGULATION_BYTES_PER_ELEMENT * elements),
                      (READ_CALLS, HostStream(1.0))))]
    return tuple(items)


def estimate_overheads(spec, parameters, registry=None, mesh_estimate=None, profile=None, save_arrays=True,
                       per_run=False, kept_mesh_keys=None, shapes=None, runs=None) -> dict:
    """{'build': TimeEstimate, 'output': TimeEstimate} of the work around the study, priced on `profile` (None: the
    stored machine profile; {} without one). Generic guesses from the memory estimate plus the triangle mesh items
    (mesh_build_items): physics work at build (port mode solves, kernel compiles) is not counted. A sweep builds once
    per value, once with reuse_build (on its own mesh; an unchanged element_mesh_key is meshed once: run_study keeps it) and
    writes once (the last value's arrays). per_run: the spec as one run (the builder's log); kept_mesh_keys: {mesh
    name: key} kept before the first run (meshing.kept_element_mesh_keys); shapes: the per_run spec's resolved geometry
    (None = resolved where needed); runs: estimated_runs of the spec (None = study_runs here)."""
    registry = registry if registry is not None else get_registry()
    backend = solver_backend(spec)
    if profile is None:
        profile, _ = stored_profile(backend)
        if profile is None:
            return {}
    build_items, kept_bytes, kept_keys = [], 0.0, dict(kept_mesh_keys or {})
    if per_run:
        runs = [('', spec, parameters, mesh_estimate)]
    elif runs is None:
        runs = study_runs(spec, parameters, mesh_estimate)
    if not per_run and spec.study.sweep is not None and getattr(spec.study.sweep, 'reuse_build', False):
        runs = runs[:1]   # reuse_build: the first value builds, later values only rerun
    for run_label, run_spec, run_parameters, run_meshes in runs:
        run_shapes = shapes if per_run else None
        meshes = (run_meshes if isinstance(run_meshes, dict) and run_meshes
                  else meshing.estimate_meshes(run_spec, run_parameters, registry, run_shapes))
        memory = meshing.estimate_memory(run_spec, run_parameters, registry, meshes, shapes=run_shapes)
        stages = float(len(memory.items) + len(memory.host_items) + 1)
        kept_bytes = float(memory.gpu_steady)   # the last run's arrays are the ones written
        prefix = f'{run_label}: ' if run_label else ''
        build_items += [
            TimeItem(f'{prefix}build stage', 'stage', stages, 'guess', (PythonCalls(BUILD_CALLS_PER_STAGE_GUESS),)),
            TimeItem(f'{prefix}build array pass', 'pass', float(BUILD_HOST_PASSES_GUESS), 'guess',
                     (HostStream(float(memory.gpu_build_peak + memory.host)),)),
            TimeItem(f'{prefix}build upload', 'upload', 1.0, 'exact', (Transfer(kept_bytes),))]
        build_items += mesh_build_items(run_spec, run_parameters, meshes, kept_keys, prefix, run_shapes)
    build = TimeEstimate(tuple(build_items))
    if not per_run:   # one build per estimation run (its mesh items too: an upper bound)
        build = _repeated(build, estimation_runs(spec))
    output_items = [TimeItem('run set-up and output', 'run', 1.0, 'guess', (PythonCalls(OUTPUT_CALLS_GUESS),)),
                    TimeItem('host copy of the kept arrays', 'copy', 1.0, 'exact', (Transfer(kept_bytes),))]
    if save_arrays:
        output_items.append(TimeItem('results.npz pass', 'pass', OUTPUT_HOST_PASSES_GUESS, 'guess',
                                     (HostStream(kept_bytes),)))
    return {'build': build.priced(profile, backend), 'output': TimeEstimate(tuple(output_items)).priced(profile, backend)}


def mesh_build_items(spec, parameters, meshes, kept_keys, prefix='', shapes=None) -> list:
    """TimeItems of making one run's element meshes, labelled "<prefix>mesh '<name>': ...": per 'triangles' estimate
    the mesher and element tagging (left out when kept_keys {mesh name: element_mesh_key} holds its key, as
    meshing.element_mesh reuses it; kept_keys updated in place) and the TriangleDomain set-up (every build); per swept
    'tetrahedra' estimate (sweep_axis 'z') only the footprint triangle mesher (same key rule). shapes: the resolved
    geometry (None = resolved here when needed)."""
    items = []
    for name, mesh in meshes.items():
        kind = getattr(mesh, 'kind', 'tensor')
        swept = kind == 'tetrahedra' and bool(getattr(mesh.sizing, 'sweep_axis', ''))
        if kind != 'triangles' and not swept:
            continue
        label = f"{prefix}mesh '{name}': "
        n_elements = float(mesh.n_elements)
        shapes = resolve_geometry(spec, parameters, errors={}) if shapes is None else shapes
        key = meshing.element_mesh_key(mesh.sizing, meshing.element_window_size(spec, parameters, kind), shapes)
        if kept_keys.get(name) != key:
            kept_keys[name] = key
            if swept:   # footprint nodes = swept nodes / layer planes (fem.prism_layers.swept_counts); no mirrors
                meshed_nodes = float(mesh.n_nodes) / max(len(mesh.sizing.layer_edges), 1)
            else:       # mirrors mesh a half / quarter
                meshed_nodes = float(mesh.n_nodes) / 2 ** len(mesh.sizing.mirror_axes)
            rounds = mesher_rounds(mesh.sizing, mesh.dx_min)
            mesher_ops = ((HostStream(GMSH_HOST_BYTES_PER_NODE * meshed_nodes),) if mesh.sizing.mesher == 'gmsh' else
                          (HostStream(MESHER_HOST_BYTES_PER_NODE_ROUND * rounds * meshed_nodes),
                           PythonCalls(rounds * MESHER_CALLS_PER_ROUND)))
            if swept:
                items.append(TimeItem(f'{label}footprint triangle mesher', 'mesh', 1.0, 'guess', mesher_ops))
                continue
            objects = sum(1 for entity in domain_entities(spec) if entity != 'all' and entity in shapes)
            items += [TimeItem(f'{label}triangle mesher', 'mesh', 1.0, 'guess', mesher_ops),
                      TimeItem(f'{label}mesh tagging', 'object', float(objects), 'exact',
                               (HostStream(TAGGING_HOST_BYTES_PER_ELEMENT * n_elements),))]
        if not swept:
            items.append(TimeItem(f'{label}TriangleDomain set-up', 'setup', 1.0, 'exact',
                                  (HostStream(DOMAIN_SETUP_HOST_BYTES_PER_ELEMENT * n_elements),)))
    return items


def mesher_rounds(sizing, dx_min) -> float:
    """Refinement rounds of the built-in triangle mesher: a round inserts the circumcentres of the triangles too large
    for h, at most halving their area, so the edge falls by >= sqrt 2 per round from h_max to the finest size (object
    sizes, size points, the estimate's dx_min; floored at h_min): 2 log2(h_max / h_finest) + 1."""
    sizes = [float(dx_min), *(float(size) for size in sizing.object_sizes.values())]
    if len(sizing.size_points):
        sizes.append(float(np.min(sizing.size_points[:, -1])))
    finest = max(min(sizes), float(sizing.h_min))
    return 2.0 * math.log2(max(float(sizing.h_max) / finest, 1.0)) + 1.0


def build_time_text(build_estimate, stage_seconds, domain_owners) -> str:
    """'Build estimate: domain estimated 18 s (guess; 6 s - 54 s), took 20.7 s; whole build estimated ..., took ...'
    from the priced per-run build part of estimate_overheads, the ledger's [(stage, s)] and {mesh name: its domain
    stage owner} (only meshes with mesh_build_items are listed)."""
    measured = {}
    for owner, seconds in stage_seconds:
        measured[owner] = measured.get(owner, 0.0) + float(seconds)

    def span(estimate):
        guess = format_seconds(estimate.seconds_guess)
        return (f'{guess} (guess; {format_seconds(estimate.seconds_low)} - {format_seconds(estimate.seconds_high)})'
                if estimate.has_guess else guess)

    parts = []
    for name, owner in domain_owners.items():
        items = tuple(item for item in build_estimate.items if item.label.startswith(f"mesh '{name}': "))
        if items:
            part = replace(build_estimate, items=items, not_estimated=())
            parts.append(f'{owner} estimated {span(part)}, took {format_seconds(measured.get(owner, 0.0))}')
    parts.append(f'whole build estimated {span(build_estimate)}, took {format_seconds(sum(measured.values()))}')
    return 'Build estimate: ' + '; '.join(parts)


def fixed_point_physics(couplings, registry=None) -> set:
    """Names of the physics a coupled fixed point re-solves: those on a cycle of enabled couplings (source -> target, a
    self-coupling and a type that reads its target, CouplingType.reads_target, included) and those downstream of one;
    every other physics converges in one pass."""
    targets = {}
    for coupling in couplings:
        if coupling.enabled:
            targets.setdefault(coupling.source, set()).add(coupling.target)
            if registry is not None and getattr(_coupling_type(registry, coupling.type), 'reads_target', False):
                targets.setdefault(coupling.target, set()).add(coupling.target)

    def downstream(start):
        reached, pending = set(), list(targets.get(start, ()))
        while pending:
            name = pending.pop()
            if name not in reached:
                reached.add(name)
                pending.extend(targets.get(name, ()))
        return reached

    reach = {name: downstream(name) for name in targets}
    on_cycle = {name for name, reached in reach.items() if name in reached}
    return on_cycle | {target for name in on_cycle for target in reach[name]}


def _coupling_type(registry, name):
    """The registered CouplingType `name`, None when unknown (reported by validation)."""
    try:
        return registry.coupling_type(name)
    except (KeyError, ValueError):
        return None


def _physics_part(spec, parameters, registry, physics_spec, mesh, memory_parts, backend, study, coupled, fixed_point,
                  ordering_kept=False):
    """(TimeEstimate labelled "physics '<name>'", every) of one physics; a missing or failing recipe -> not estimated."""
    label = f"physics '{physics_spec.name}'"
    try:
        physics_type = registry.physics_type(physics_spec.type)
    except KeyError:
        return TimeEstimate(not_estimated=(label,)), 1
    if physics_type.time_estimate is None or mesh.kind not in physics_type.grids:
        return TimeEstimate(not_estimated=(f'{label} ({physics_type.name} has no time recipe)',)), 1
    try:
        settings = meshing.physics_settings(physics_type, physics_spec, parameters)
        solvers = resolved_solvers(spec, registry, physics_spec, spec.study.type, parameters)
        context = TimeContext(spec, physics_spec, parameters, registry, solvers, mesh,
                              memory_parts.get(physics_spec.name), backend, study, coupled, fixed_point, ordering_kept)
        part = physics_type.time_estimate(settings, mesh, context).labelled(label)
    except Exception as exc:   # a recipe bug: never stops validation nor costs the other physics' estimates
        return TimeEstimate(not_estimated=(f'{label} (recipe failed: {exc})',)), 1
    every = physics_spec.every if isinstance(physics_spec.every, int) and physics_spec.every > 0 else 1
    return part, every


def _with_study_steps(parts, study) -> TimeEstimate:
    """Per-step items of every part turned into totals: time_dependent = round(duration / dt_sync) ticks (dt_sync the
    smallest dt_limit), a physics stepping every n ticks; frequency_domain = the harmonic FDTD pulse (a guess below
    max_duration); other studies run no steps. A stride-n physics steps at ticks 0, n, 2n, ... (ceil(ticks / n)) and
    runner.flush() steps every quasi-static physics once more."""
    finite = [part.dt_limit for part, _ in parts if math.isfinite(part.dt_limit)]
    if study.type == 'time_dependent' and finite and study.counts.get('duration'):
        dt_sync = min(finite)
        ticks = max(int(round(study.counts['duration'] / dt_sync)), 0)
        return combine(part.with_steps(-(-ticks // every) + (0 if math.isfinite(part.dt_limit) else 1), 'exact', every,
                                       ticks=ticks)
                       for part, every in parts)
    stepped = []
    for part, every in parts:
        if study.type == 'frequency_domain' and math.isfinite(part.dt_limit) and study.counts.get('max_duration'):
            limit = study.counts['max_duration'] / part.dt_limit
            stepped.append(part.with_steps(HARMONIC_PULSE_FRACTION_GUESS * limit, 'guess', steps_limit=limit,
                                           measured_by_ticks=False))
        else:
            stepped.append(part.with_steps(0, 'exact', measured_by_ticks=False))
    return combine(stepped)
