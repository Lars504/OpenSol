"""Solve flow of a model (GUI Solvers > Show solve flow): its build and study steps in run order, the solver every
physics role resolves to and the memory / run-time estimate items each step books. CPU only, GUI-thread safe.

Steps sit on rows (ROW_KINDS order; study steps take one row each) and over lanes: one lane slot per resolved solver
role of a physics (at least one), physics grouped by mesh, so a mesh / materials step spans its physics' lanes and the
model / couplings / study / output steps span every lane. Estimate items are matched to steps by their labels
(meshing.estimate_memory / model.time_estimate prefixes: "[mesh '<m>': ]physics '<p>': ", "feature '<p>.<f>': ",
"model: ", "subpixel smoothing: ", "conformal boundaries: ", "grid transfer ", "snapshots on mesh '<m>': ") and solver
time items to their role by the summary.json counter path ('solvers', physics, 'roles', role, ...). An item no rule
places goes to the 'unplaced' step, never dropped: a new label prefix needs a rule in _memory_step / _time_step."""
import math
import re
from dataclasses import dataclass, fields

from ..solvers.roles import BACKENDS, PRECISIONS, SOLVER_ROLES, resolved_precision
from .schema import resolved_solvers, solver_backend, solver_grid
from .time_estimate import estimation_runs, study_counts

# row kind -> row label (left margin of the chart); study rows are 'study' too, one per study step
ROW_KINDS = {'model': 'Model', 'mesh': 'Meshes', 'materials': 'Materials', 'solver': 'Solver jobs',
             'physics': 'Physics', 'couplings': 'Couplings', 'study': 'Study', 'output': 'Results',
             'unplaced': 'Unplaced'}
# MemoryEntry.group -> what the bytes are (details panel)
MEMORY_GROUPS = {
    'kept': 'device, kept from the build on',
    'run': 'device, allocated by the study and kept',
    'transient': 'device, freed when its phase ends (phases never overlap: the largest counts)',
    'external': 'device outside the memory pool (direct-solver factor), whole run',
    'external_transient': 'device outside the memory pool (direct-solver factor), one phase',
    'host': 'processor memory',
}
# TimeEntry.source -> where the item comes from
TIME_SOURCES = {'run': 'one study run (time_estimate per_run)', 'build': 'build overhead (estimate_overheads)',
                'output': 'output overhead (estimate_overheads)'}

# node names are identifiers (no quotes, no dots); the rest of a physics label: ': <item>' or ' (<reason>)'
_MESH_PREFIX = re.compile(r"^mesh '([^']+)': (.*)$", re.S)
_PHYSICS_PREFIX = re.compile(r"^physics '([^']+)'(.*)$", re.S)
_FEATURE_PREFIX = re.compile(r"^feature '([^'.]+)\.([^']+)': (.*)$", re.S)
_SNAPSHOT_PREFIX = re.compile(r"^snapshots on mesh '([^']+)': (.*)$", re.S)
_COUPLING_PREFIX = re.compile(r"^coupling '([^']+)'", re.S)
_MODEL_PREFIXES = ('model: ', 'subpixel smoothing: ', 'conformal boundaries: ')

_EXPLANATIONS = {
    'model': 'The model is validated on the processor, the parameters are evaluated, the geometry is resolved and '
             'the memory estimate is checked against the budgets; then the build starts on the worker thread '
             '(model.builder.build).',
    'mesh': 'make_domain builds this mesh: the tensor grid with its PML / ghost padding, or the element mesh (the '
            'mesher runs unless an equal mesh is still kept from an earlier build).',
    'materials': 'Static materials are painted into the MaterialGrid arrays of the physics types on this mesh; '
                 'state-dependent material updates, coupling transfer arrays and subpixel / conformal data are '
                 'allocated here.',
    'solver': "Build stage 'solvers': the role runtime is created before its physics (one memory-ledger entry per "
              'solver) and handed to the physics create. Its run-time items are counted here but the work runs in '
              'the study steps below.',
    'physics': 'The physics is created on its mesh with the solver runtimes above, its features are applied '
               '(sources, boundaries, monitors), dynamic material updates apply once, and finalize assembles its '
               'operators (some solvers factorise here).',
    'couplings': 'The couplings are added, runner.prepare runs and the build caches are released; the session is '
                 'then ready for the study.',
    'output': 'Snapshots of the stored fields, derived values and feature results are collected; a saved run writes '
              'summary.json and results.npz.',
    'unplaced': 'Estimate items whose label matches no step of this chart (solve_flow has no rule for their prefix). '
                'They still count in the totals.',
}


@dataclass(frozen=True)
class MemoryEntry:
    """One memory estimate item of a step: group (MEMORY_GROUPS key), label without the step's prefix, bytes, run
    phase (transient groups)."""
    group: str
    label: str
    bytes: int
    phase: str = ''


@dataclass(frozen=True)
class TimeEntry:
    """One priced run-time item of a step (core.time_estimate.TimeItem), its label without the step's prefix, its
    source (TIME_SOURCES key) and its guess seconds."""
    label: str
    item: object
    source: str
    seconds: float


@dataclass(frozen=True)
class FlowStep:
    """A box of the chart: row index, lane span [lane_start, lane_end), the steps it follows (arrows), title /
    subtitle / facts ((name, text)) / explanation and the estimate entries booked at it."""
    key: str
    kind: str
    row: int
    lane_start: float
    lane_end: float
    title: str
    subtitle: str = ''
    facts: tuple = ()
    explanation: str = ''
    after: tuple = ()
    memory: tuple = ()
    time: tuple = ()
    not_estimated: tuple = ()

    @property
    def memory_bytes(self) -> int:
        """Bytes of its entries (every group summed: a size indicator, not a peak)."""
        return sum(entry.bytes for entry in self.memory)

    @property
    def time_seconds(self) -> float:
        return sum(entry.seconds for entry in self.time)


@dataclass(frozen=True)
class FlowLoop:
    """A repetition around rows first_row..last_row (a sweep, refinement levels, parity sectors, an estimation)."""
    title: str
    text: str
    first_row: int
    last_row: int


@dataclass(frozen=True)
class FlowLink:
    """An enabled coupling between two physics steps (keys)."""
    source: str
    target: str
    label: str


@dataclass(frozen=True)
class SolveFlow:
    """The chart: steps, row labels (index = FlowStep.row), lane count, loops, coupling links, totals ((name, text))
    and notes (missing estimates, resolution failures)."""
    steps: tuple
    rows: tuple
    lanes: float
    loops: tuple = ()
    links: tuple = ()
    totals: tuple = ()
    notes: tuple = ()
    backend: str = 'gpu'

    def step(self, key):
        """The step `key`; KeyError when the chart has none."""
        for step in self.steps:
            if step.key == key:
                return step
        raise KeyError(f"solve flow has no step '{key}'")


def solve_flow(spec, registry, parameters, meshes, memory=None, plain_roles=frozenset(), run_time=None,
               overheads=None, study_time=None, time_message='') -> SolveFlow:
    """SolveFlow of the spec: meshes {used mesh name: estimate or None}; memory the (fitted) MemoryEstimate or None;
    plain_roles of meshing.fitted_memory_estimate; run_time the priced per_run TimeEstimate (counters kept: solver items
    go to their role) or None; overheads estimate_overheads(per_run=True) ({} / None without a profile); study_time the
    whole-study TimeEstimate (totals only); time_message why run_time is None."""
    backend = solver_backend(spec)
    notes = []
    builder = _Builder(spec, registry)
    builder.add_model_step()
    used = [mesh_spec.name for mesh_spec in spec.used_meshes()] if spec.physics else []
    lane = 0.0
    physics_lanes = {}
    for mesh_name in used:
        first = lane
        for physics_spec in spec.physics_on(mesh_name):
            roles, problem = _resolved_roles(spec, registry, physics_spec, parameters, plain_roles)
            if problem:
                notes.append(problem)
            width = float(max(len(roles), 1))
            physics_lanes[physics_spec.name] = (lane, lane + width, roles)
            lane += width
        if lane == first:
            lane += 1.0
        builder.add_mesh_steps(mesh_name, meshes.get(mesh_name), first, lane)
    lanes = max(lane, 1.0)
    builder.steps['model']['lane_end'] = lanes   # added before the lanes were known
    for mesh_name in used:
        for physics_spec in spec.physics_on(mesh_name):
            start, end, roles = physics_lanes[physics_spec.name]
            builder.add_physics_steps(physics_spec, mesh_name, start, end, roles, plain_roles)
    builder.add_couplings_step(lanes, [f"physics:{name}" for name in physics_lanes])
    builder.add_study_steps(parameters, lanes, run_time)
    builder.add_output_step(lanes)

    if memory is None:
        notes.append('Memory estimate not available (see the Validation tab): no step shows memory items.')
    else:
        builder.place_memory(memory)
    if run_time is None:
        notes.append(f'Run-time estimate not available: {time_message}' if time_message else
                     'Run-time estimate not available (see the Validation tab): no step shows time items.')
    else:
        builder.place_time(run_time.items, 'run')
        builder.place_not_estimated(run_time.not_estimated)
    for source in ('build', 'output'):
        estimate = (overheads or {}).get(source)
        if estimate is not None:
            builder.place_time(estimate.items, source)
            builder.place_not_estimated(estimate.not_estimated)
    builder.add_unplaced_step(lanes)
    if builder.bad_times:
        notes.append('Time items with a non-finite estimate (an estimate bug; counted as 0 s here): '
                     + '; '.join(builder.bad_times))
    links = tuple(FlowLink(f'physics:{coupling.source}', f'physics:{coupling.target}', coupling.name)
                  for coupling in spec.couplings if coupling.enabled
                  and coupling.source in physics_lanes and coupling.target in physics_lanes)
    loops = _loops(spec, builder)
    totals = _totals(spec, backend, memory, run_time, overheads, study_time)
    return SolveFlow(builder.finished_steps(), tuple(builder.rows), lanes, loops, links, totals, tuple(notes), backend)


def op_text(op) -> str:
    """'Spmv(rows=1.2e+05, nnz=6e+05)' of a core.time_estimate op (non-default fields only), or '3 x <op>' of a
    (multiplicity, op) pair."""
    if isinstance(op, tuple) and len(op) == 2:
        return f'{op[0]:.4g} x {op_text(op[1])}'
    parts = []
    for field_info in fields(op):
        value = getattr(op, field_info.name)
        if value == field_info.default:
            continue
        parts.append(f'{field_info.name}={value:.4g}' if isinstance(value, float) else f'{field_info.name}={value}')
    return f"{type(op).__name__}({', '.join(parts)})"


# --- construction ---------------------------------------------------------------------------------------------------

class _Builder:
    """Mutable step table of solve_flow (rows appended in run order; estimate entries collected per step key)."""

    def __init__(self, spec, registry):
        self.spec = spec
        self.registry = registry
        self.rows = []          # row labels
        self.row_of_kind = {}   # first row of each kind
        self.steps = {}         # key -> dict of FlowStep fields (entries appended until finished_steps)
        self.order = []
        self.mesh_of_physics = {}
        self.solver_keys = {}   # (physics, role) -> step key
        self.bad_times = []     # labels of time items whose seconds are not finite (an estimate bug)

    def _row(self, kind, label=None):
        if kind not in self.row_of_kind or kind == 'study':
            self.row_of_kind.setdefault(kind, len(self.rows))
            self.rows.append(label or ROW_KINDS[kind])
            return len(self.rows) - 1
        return self.row_of_kind[kind]

    def _add(self, key, kind, row, lanes, title, subtitle='', facts=(), after=(), explanation=None):
        if key in self.steps:
            raise ValueError(f"solve flow: duplicate step '{key}'")
        self.steps[key] = {'key': key, 'kind': kind, 'row': row, 'lane_start': float(lanes[0]),
                           'lane_end': float(lanes[1]), 'title': title, 'subtitle': subtitle, 'facts': tuple(facts),
                           'explanation': _EXPLANATIONS.get(kind, '') if explanation is None else explanation,
                           'after': tuple(after), 'memory': [], 'time': [], 'not_estimated': []}
        self.order.append(key)

    def finished_steps(self):
        return tuple(FlowStep(**{**self.steps[key], 'memory': tuple(self.steps[key]['memory']),
                                 'time': tuple(self.steps[key]['time']),
                                 'not_estimated': tuple(self.steps[key]['not_estimated'])}) for key in self.order)

    # steps

    def add_model_step(self):
        spec = self.spec
        facts = [('Physics', ', '.join(f'{physics.name} ({self._physics_label(physics)})' for physics in spec.physics)
                  or 'none'),
                 ('Parameters', str(len(spec.parameters))), ('Geometry objects', str(len(spec.geometry))),
                 ('Run on', BACKENDS.get(solver_backend(spec), solver_backend(spec)))]
        dimensions = getattr(spec, 'dimensions', None)
        subtitle = f'{dimensions}D model' if dimensions else ''
        self._add('model', 'model', self._row('model'), (0.0, 1.0), 'Check and prepare the model', subtitle, facts)

    def add_mesh_steps(self, mesh_name, estimate, start, end):
        mesh_spec = next((mesh for mesh in self.spec.meshes if mesh.name == mesh_name), None)
        kind = getattr(mesh_spec, 'kind', 'tensor')
        facts = [('Kind', str(kind))] + _mesh_facts(estimate)
        users = [physics.name for physics in self.spec.physics_on(mesh_name)]
        facts.append(('Used by', ', '.join(users) or 'nothing'))
        self._add(f'mesh:{mesh_name}', 'mesh', self._row('mesh'), (start, end), f"Mesh '{mesh_name}'",
                  _mesh_subtitle(kind, estimate), facts, after=('model',))
        self._add(f'materials:{mesh_name}', 'materials', self._row('materials'), (start, end),
                  f"Materials on '{mesh_name}'", f'{len(self.spec.materials)} materials', after=(f'mesh:{mesh_name}',))

    def add_physics_steps(self, physics_spec, mesh_name, start, end, roles, plain_roles):
        name = physics_spec.name
        self.mesh_of_physics[name] = mesh_name
        solver_row = self._row('solver')
        physics_row = self._row('physics')
        solver_keys = []
        for index, (role, resolved) in enumerate(roles.items()):
            key = f'solver:{name}:{role}'
            self.solver_keys[(name, role)] = key
            solver_keys.append(key)
            solver_type = resolved.solver_type
            precision = resolved_precision(solver_type, resolved.values)
            facts = [('Physics', name), ('Job', SOLVER_ROLES.get(role, role)),
                     ('Chosen by', f"solver node '{resolved.node}'" if resolved.node else 'Automatic'),
                     ('Solver type', f'{solver_type.label} ({solver_type.name})'),
                     ('Precision', PRECISIONS.get(precision, precision)),
                     ('Run on', BACKENDS.get(solver_backend(self.spec), solver_backend(self.spec))),
                     ('Grid', solver_grid(self.spec, physics_spec))]
            if (name, role) in plain_roles:
                facts.append(('Note', "this study's kept direct factor does not fit the memory budget, so the "
                                      'Automatic solver of the other studies runs instead'))
            facts += _settings_facts(solver_type, resolved.values)
            self._add(key, 'solver', solver_row, (start + index, start + index + 1), solver_type.label,
                      f"{name} · {SOLVER_ROLES.get(role, role).lower()}"
                      + ('' if resolved.node else ' · Automatic'), facts, after=(f'materials:{mesh_name}',))
        physics_type = self._physics_type(physics_spec)
        features = [feature for feature in physics_spec.features if feature.enabled]
        facts = [('Type', self._physics_label(physics_spec)), ('Mesh', mesh_name),
                 ('Steps', f'every {physics_spec.every} ticks') if physics_spec.every not in (1, None) else
                 ('Steps', 'every tick'),
                 ('Features', ', '.join(f'{feature.name} ({_feature_label(physics_type, feature)})'
                                        for feature in features) or 'none')]
        disabled = [feature.name for feature in physics_spec.features if not feature.enabled]
        if disabled:
            facts.append(('Disabled features', ', '.join(disabled)))
        if not self.spec.study.solves(name):
            facts.append(('Study role', 'background physics (not in the solved physics list)'))
        self._add(f'physics:{name}', 'physics', physics_row, (start, end), f'{name}',
                  f'{self._physics_label(physics_spec)} · {len(features)} features', facts,
                  after=tuple(solver_keys) or (f'materials:{mesh_name}',))

    def add_couplings_step(self, lanes, physics_keys):
        couplings = [coupling for coupling in self.spec.couplings if coupling.enabled]
        facts = [(coupling.name, f'{coupling.source} -> {coupling.target} ({coupling.type})') for coupling in couplings]
        disabled = [coupling.name for coupling in self.spec.couplings if not coupling.enabled]
        if disabled:
            facts.append(('Disabled', ', '.join(disabled)))
        self._add('couplings', 'couplings', self._row('couplings'), (0.0, lanes),
                  'Couplings and runner set-up', f'{len(couplings)} couplings' if couplings else 'no couplings',
                  facts, after=tuple(physics_keys) or ('model',))

    def add_study_steps(self, parameters, lanes, run_time):
        spec = self.spec
        try:
            study_type = self.registry.study_type(spec.study.type)
            counts = study_counts(spec, parameters, self.registry)
        except (KeyError, ValueError, TypeError) as exc:   # unknown type / bad settings: validation reports them
            study_type, counts, problem = None, None, str(exc)
        else:
            problem = ''
        label = study_type.label if study_type is not None else str(spec.study.type)
        count_facts = [(name, f'{value:.6g}' if isinstance(value, float) else str(value))
                       for name, value in sorted((counts.counts if counts is not None else {}).items())]
        steps = _study_steps(spec, study_type, counts, run_time, self.registry)
        if problem:
            steps = [(label, '', f'The study cannot be described: {problem}', [])]
        previous = 'couplings'
        for index, (title, subtitle, explanation, facts) in enumerate(steps):
            key = f'study:{index}'
            all_facts = list(facts) + ([('Study', label)] if index == 0 else [])
            if index == 0:
                all_facts += [(f'count: {name}', text) for name, text in count_facts]
            self._add(key, 'study', self._row('study', 'Study' if index == 0 else ''), (0.0, lanes), title, subtitle,
                      all_facts, after=(previous,), explanation=explanation)
            previous = key
        self.last_study = previous

    def add_output_step(self, lanes):
        study = self.spec.study
        stored = ', '.join(str(name) for name in study.stored_fields) or 'every field (none chosen)'
        facts = [('Stored fields', stored), ('Derived values', str(len(self.spec.derived_values))),
                 ('Analytic curves', str(len(self.spec.analytic_curves))),
                 ('Derived curves', str(len(self.spec.derived_curves)))]
        self._add('output', 'output', self._row('output'), (0.0, lanes), 'Results', 'snapshots, derived values, files',
                  facts, after=(self.last_study,))

    def add_unplaced_step(self, lanes):
        """The 'unplaced' step only when an entry landed there (filled by _target before)."""
        if 'unplaced' in self.steps:
            step = self.steps['unplaced']
            if step['row'] < 0:
                step['row'] = self._row('unplaced')
                step['lane_end'] = float(lanes)

    # estimate placement

    def place_memory(self, memory):
        groups = (('kept', ((label, size, '') for label, size in memory.items)),
                  ('run', ((label, size, '') for label, size in memory.run_items)),
                  ('transient', ((label, size, phase) for phase, label, size in memory.transient_items)),
                  ('external', ((label, size, '') for label, size in memory.external_items)),
                  ('external_transient', ((label, size, phase)
                                          for phase, label, size in memory.external_transient_items)),
                  ('host', ((label, size, '') for label, size in memory.host_items)))
        for group, entries in groups:
            for label, size, phase in entries:
                key, short = self._memory_step(label)
                self._target(key)['memory'].append(MemoryEntry(group, short, int(size),
                                                               self._memory_step(phase)[1] if phase else ''))

    def place_time(self, items, source):
        for item in items:
            key, short = self._time_step(item, source)
            seconds = item.counts()[1] * (item.seconds_per_unit or 0.0)
            if not math.isfinite(seconds):   # listed in the notes; kept out of the step sums
                self.bad_times.append(f'{item.label}: {item.count!r} x {item.seconds_per_unit!r} s')
                seconds = 0.0
            self._target(key)['time'].append(TimeEntry(short, item, source, seconds))

    def place_not_estimated(self, labels):
        for label in labels:
            match = _PHYSICS_PREFIX.match(label)
            coupling = _COUPLING_PREFIX.match(label)
            if match and f'physics:{match.group(1)}' in self.steps:
                key, short = f'physics:{match.group(1)}', _physics_rest(match, label)
            elif coupling:
                key, short = 'couplings', label
            else:
                key, short = 'unplaced', label
            self._target(key)['not_estimated'].append(short)

    def _target(self, key):
        """The step dict of key; the 'unplaced' step is created on first use (row set by add_unplaced_step)."""
        if key not in self.steps:
            if key != 'unplaced':
                raise KeyError(f"solve flow: no step '{key}'")
            self._add('unplaced', 'unplaced', -1, (0.0, 1.0), 'Unplaced estimate items',
                      'items no step of this chart matches')
        return self.steps[key]

    def _memory_step(self, label):
        """(step key, label without its prefix) of a memory item label."""
        mesh = None
        match = _MESH_PREFIX.match(label)
        if match and f'mesh:{match.group(1)}' in self.steps:
            mesh, label = match.group(1), match.group(2)
        match = _PHYSICS_PREFIX.match(label)
        if match and f'physics:{match.group(1)}' in self.steps:
            return f'physics:{match.group(1)}', _physics_rest(match, label)
        match = _FEATURE_PREFIX.match(label)
        if match and f'physics:{match.group(1)}' in self.steps:
            return f'physics:{match.group(1)}', f'feature {match.group(2)}: {match.group(3)}'
        if label.startswith(_MODEL_PREFIXES):
            mesh = mesh if mesh is not None else self._default_mesh()
            if mesh is not None and f'materials:{mesh}' in self.steps:
                return f'materials:{mesh}', label
        if label.startswith('grid transfer '):
            return 'couplings', label
        return 'unplaced', label if mesh is None else f"mesh '{mesh}': {label}"

    def _time_step(self, item, source):
        """(step key, label without its prefix) of a TimeItem."""
        label = item.label
        if source == 'output':
            return 'output', label
        match = _MESH_PREFIX.match(label)
        if match and f'mesh:{match.group(1)}' in self.steps:
            return f'mesh:{match.group(1)}', match.group(2)
        if source == 'build':
            return 'model', label
        match = _SNAPSHOT_PREFIX.match(label)
        if match:
            return 'output', label
        match = _PHYSICS_PREFIX.match(label)
        if match and f'physics:{match.group(1)}' in self.steps:
            physics, short = match.group(1), _physics_rest(match, label)
            path = tuple(item.counter or item.seconds_counter or ())
            if len(path) >= 4 and path[0] == 'solvers' and path[2] == 'roles' and path[1] == physics:
                key = self.solver_keys.get((physics, path[3]))
                if key is not None:
                    return key, short
            return f'physics:{physics}', short
        return 'unplaced', label

    def _default_mesh(self):
        return self.spec.meshes[0].name if self.spec.meshes else None

    def _physics_type(self, physics_spec):
        try:
            return self.registry.physics_type(physics_spec.type)
        except KeyError:
            return None

    def _physics_label(self, physics_spec):
        physics_type = self._physics_type(physics_spec)
        return physics_type.label if physics_type is not None else str(physics_spec.type)


def _physics_rest(match, label):
    """The item part of a matched "physics '<p>'..." label: after ': ', else the whole label (' (<reason>)')."""
    rest = match.group(2)
    return rest[2:] if rest.startswith(': ') and len(rest) > 2 else label


def _resolved_roles(spec, registry, physics_spec, parameters, plain_roles):
    """({role: ResolvedSolver}, problem text) of a physics in the spec's study ({} and the reason on failure)."""
    try:
        return resolved_solvers(spec, registry, physics_spec, spec.study.type, parameters, plain_roles), ''
    except KeyError as exc:   # unknown physics type: validation reports it
        return {}, f"physics '{physics_spec.name}': unknown type {exc}"
    except ValueError as exc:
        return {}, f'Solvers of {exc}'


def _feature_label(physics_type, feature):
    if physics_type is None:
        return str(feature.type)
    try:
        return physics_type.feature_type(feature.type).label
    except KeyError:
        return str(feature.type)


def _settings_facts(solver_type, values):
    """('setting: <label>', value) of every resolved solver value, in the type's Param order."""
    facts, seen = [], set()
    for param in solver_type.params:
        if param.name in values:
            seen.add(param.name)
            facts.append((f'setting: {param.label or param.name}', _value_text(values[param.name], param.unit)))
    for name in sorted(set(values) - seen):   # resolved extras (an auto back end's choice)
        facts.append((f'resolved: {name}', _value_text(values[name], '')))
    return facts


def _value_text(value, unit):
    if isinstance(value, float):
        text = f'{value:.6g}'
    else:
        text = str(value)
    return f'{text} {unit}' if unit and unit != '1' else text


def _mesh_subtitle(kind, estimate):
    if estimate is None:
        return f'{kind}, no estimate'
    if getattr(estimate, 'kind', 'tensor') == 'tensor':
        axes = [estimate.nx, estimate.ny] + ([estimate.nz] if getattr(estimate, 'z_edges', None) is not None else [])
        return f"{' x '.join(str(int(count)) for count in axes)} cells"
    elements = getattr(estimate, 'n_elements', getattr(estimate, 'n_cells', 0))
    built = '' if getattr(estimate, 'counts_built', False) else ' (estimated)'
    return f'{int(elements):,} elements{built}'


def _mesh_facts(estimate):
    if estimate is None:
        return [('Estimate', 'not available (see the Validation tab)')]
    facts = []
    if getattr(estimate, 'kind', 'tensor') == 'tensor':
        facts.append(('Cells (with padding)', f'{int(estimate.n_cells):,}'))
        facts.append(('Window cells', ' x '.join(str(int(count)) for count in estimate.window_axis_cells)))
        facts.append(('Cell size', f'{estimate.dx_min:.4g} - {estimate.dx_max:.4g} m'))
        if estimate.dt is not None:
            facts.append(('Courant time step', f'{estimate.dt:.4g} s'))
        if estimate.npml_sides:
            facts.append(('Padding cells', ', '.join(f'{face} {count}' for face, count in estimate.npml_sides.items()
                                                     if count)))
        if estimate.note:
            facts.append(('Note', estimate.note))
    else:
        for name in ('n_elements', 'n_nodes', 'n_cells'):
            value = getattr(estimate, name, None)
            if isinstance(value, (int, float)):
                facts.append((name.replace('n_', '').capitalize(), f'{int(value):,}'))
        facts.append(('Counts', 'from the built mesh' if getattr(estimate, 'counts_built', False)
                      else 'estimated from the size field (build the mesh for exact counts)'))
    return facts


# --- study steps ----------------------------------------------------------------------------------------------------

_BACKGROUND = ('Background stage', 'stationary solve of the physics the study does not solve itself',
               'Physics the study does not solve itself run as stationary background physics: solve_stationary over '
               'them, then every background coupling applies once (study._background_stage).')


def _study_steps(spec, study_type, counts, run_time, registry):
    """[(title, subtitle, explanation, facts)] of the study run (the study module's own order)."""
    if study_type is None:
        return [(str(spec.study.type), '', 'Unknown study type.', [])]
    name = study_type.name
    values = counts.counts if counts is not None else {}
    background = [_BACKGROUND] if study_type.background_study and _has_background(spec, registry, name) else []
    if name == 'time_dependent':
        duration = values.get('duration')
        dt = run_time.dt_limit if run_time is not None else math.inf
        ticks = (f'{int(round(duration / dt)):,} ticks of {dt:.4g} s' if duration and math.isfinite(dt) and dt > 0
                 else f'{duration:.4g} s' if duration else '')
        return [('Time loop', ticks,
                 'Chunks of about 0.1 s wall time: per tick the couplings apply, every physics steps (a physics with '
                 '"every n" each n-th tick; quasi-static physics re-solve when their inputs change), probes record '
                 'and snapshots are taken at the output times.', [('Duration', f'{duration:.6g} s')] if duration
                 else []),
                ('Flush', 'quasi-static physics solve once more',
                 'runner.flush(): pending quasi-static solves run once at the end, also after a cancel.', [])]
    if name == 'stationary':
        return [('Stationary solve', 'fixed point over every physics',
                 'solve_stationary: every physics solves in order and the couplings apply; with a coupling cycle the '
                 'loop repeats until the relative change is below the tolerance (fixed_point in the summary).', [])]
    if name == 'frequency_domain':
        points = values.get('harmonic_points')
        return background + [
            ('Pulsed harmonic physics', 'time-stepped physics only, if any',
             'Time-stepped harmonic physics (harmonic FDTD) run one broadband pulse until the window energy decays; '
             'no coupling applies during the pulse.', []),
            ('Wavelength loop', f'{points} wavelengths' if points else '',
             'Per wavelength solve_harmonic(omega) of every harmonic physics in solve order and one snapshot '
             "'lambda = <nm> nm'; the last wavelength stays live. A coupled scan (a coupling reads a harmonic "
             'physics) re-solves every physics per wavelength instead.', [])]
    if name == 'mode_analysis':
        return background + [('Mode solve', '', 'solve_modes() of every mode physics, then one snapshot per mode '
                                                "number ('mode k: n_eff = ...').", [])]
    if name == 'eigenfrequency':
        pairs = values.get('eigenpairs')
        return background + [
            ('Eigen solve', f'{pairs} resonances' if pairs else '',
             'Optional prestress solve, then solve_eigenfrequencies(n) (or the target / band search); the eigen role '
             'factorises the shifted pencil and runs the Krylov iterations.', []),
            ('Resonance snapshots', '', "select_resonance(k) and one snapshot per resonance ('resonance k: lambda = "
                                        "..., Q = ...'); resonance 1 stays live.", [])]
    if name == 'frequency_response':
        points = values.get('harmonic_points')
        return background + [('Frequency loop', f'{points} frequencies' if points else '',
                              'One driven solve per frequency (reduced / modal sweeps where chosen) and one snapshot '
                              'per frequency.', [])]
    if name == 'optimization':
        iterations = values.get('iterations')
        return [('Optimization loop', f'up to {iterations} iterations' if iterations else '',
                 'scipy L-BFGS-B over the design-region densities: per evaluation the densities are written to eps, '
                 'every FDFD / EME physics solves and its adjoint gives the sensitivities; one snapshot per accepted '
                 'iteration.', [])]
    if name == 'optomechanics':
        return background + [('Mechanical and optical solves', '',
                              'Mechanical eigenmodes and the optical modes / resonances of the physics the coupling '
                              'feature names, then the optomechanical coupling g0 per pair.', [])]
    if name == 'thermal_noise':
        return background + [('Thermal noise', '', 'Mode solve of the optics and the heat response that weighs the '
                                                   'thermal fluctuations into the noise spectra.', [])]
    if name == 'nonlinear_frequency_domain':
        return background + [('Harmonic balance', '', 'Newton iterations on the coupled harmonic channels until the '
                                                      'residual tolerance.', [])]
    return background + [(study_type.label, '', (study_type.summary or study_type.description)
                          + ' (The solve flow has no detailed step list for this study type.)', [])]


def _has_background(spec, registry, study_name):
    """True when a physics runs as background in the study (not listed as solved, or its type does not list it)."""
    for physics_spec in spec.physics:
        try:
            physics_type = registry.physics_type(physics_spec.type)
        except KeyError:
            continue
        if not spec.study.solves(physics_spec.name) or study_name not in physics_type.supported_studies:
            return True
    return False


# --- loops and totals -----------------------------------------------------------------------------------------------

def _loops(spec, builder):
    """FlowLoops of the repetitions around the build / study rows."""
    study = spec.study
    first_build = builder.row_of_kind.get('mesh', builder.row_of_kind['couplings'])
    first_study = builder.row_of_kind['study']
    last_study = builder.steps[builder.last_study]['row']
    loops = []
    settings = study.settings if isinstance(study.settings, dict) else {}
    if settings.get('parity_sectors'):
        loops.append(FlowLoop('Parity sectors', 'one build and run per symmetric / antisymmetric combination of the '
                                                'mirror faces, merged into one summary', first_build, last_study))
    if study.adaptation is not None:
        loops.append(FlowLoop('Mesh refinement', f'up to {study.adaptation.max_levels} refinement levels: the model '
                                                 'is rebuilt on each refined mesh', first_build, last_study))
    if study.sweep is not None:
        try:
            count = len(study.sweep.point_values())
        except (TypeError, ValueError):
            count = None
        reuse = getattr(study.sweep, 'reuse_build', False) is True
        what = f'{count} values of {study.sweep.parameter}' if count is not None else f'values of {study.sweep.parameter}'
        loops.append(FlowLoop('Parameter sweep', f'{what}: ' + ('one build, the study reruns per value' if reuse
                                                                else 'a build and a run per value'),
                              first_study if reuse else first_build, last_study))
    if study.estimation is not None:
        loops.append(FlowLoop('Parameter estimation', f'up to {estimation_runs(spec)} study runs of the search',
                              first_build, last_study))
    return tuple(loops)


def _totals(spec, backend, memory, run_time, overheads, study_time):
    """(name, text) rows of the estimate totals."""
    from ..core.gpu_memory import format_bytes   # GUI-thread safe (the only gpu_memory function the GUI calls)
    from ..core.time_estimate import format_seconds
    totals = []
    if memory is not None:
        if backend == 'cpu':
            totals.append(('Processor memory at the peak', format_bytes(memory.gpu_peak + memory.host)
                           + ' (device arrays live in processor memory on the cpu backend)'))
        else:
            totals.append(('Graphics card memory at the peak', format_bytes(memory.device_peak)))
            totals.append(('  kept from the build', format_bytes(memory.gpu_steady)))
            totals.append(('  build high-water mark', format_bytes(memory.gpu_build_peak)))
            totals.append(('  added by the study', format_bytes(memory.gpu_run)))
            if memory.external_peak:
                totals.append(('  direct-solver factors (outside the pool)', format_bytes(memory.external_peak)))
            if memory.host:
                totals.append(('Processor memory', format_bytes(memory.host)))

    def span(estimate):
        text = format_seconds(estimate.seconds_guess)
        if estimate.has_guess:
            text += f' ({format_seconds(estimate.seconds_low)} - {format_seconds(estimate.seconds_high)})'
        return text + estimate.outdated_text()
    if run_time is not None:
        totals.append(('Run time of one study run', span(run_time)))
    if study_time is not None:   # every sweep value / sector / estimation run (equals one run otherwise)
        totals.append(('Run time of the whole study', span(study_time)))
    for source, label in (('build', 'Build overhead'), ('output', 'Output overhead')):
        estimate = (overheads or {}).get(source)
        if estimate is not None:
            totals.append((label, span(estimate)))
    return tuple(totals)
