"""Shared helpers of the S7.8 mesh study (mesh_study_plan.md): mesh variants, check + run of each variant, one JSON
line per run in <out>/records/, read back with the last line per label winning."""
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from .. import api
from ..model.spec import MeshAxisSpec, MeshBandSpec, MeshSpec, RefinementSpec, SolverSpec

MODELS_DIR = Path(__file__).resolve().parent / 'models'
EXAMPLES_DIR = api.EXAMPLES_DIR
NM, MM = 1e-9, 1e-3
STUDY_SOLVER_NAME = 'mesh_study_solver'   # the solver node a variant adds (pcg / line_block parts)
COMPLETED = 'completed'


@dataclass
class StudyOptions:
    """Command-line options shared by the parts (defaults = mesh_study_plan.md)."""
    out_dir: Path = Path('mesh_study_out')
    force: bool = False            # re-run every recorded variant
    retry_failed: bool = False     # re-run recorded variants whose status is not completed
    fine: bool = False             # accuracy / line_block: add the slowest variants
    cases: tuple = ()              # --case names (empty = every case of every part)
    max_cells: float = 2e7         # skip variants with more cells (not recorded)
    max_wall_time: float | None = None   # per run [s]

    @property
    def records_dir(self):
        return Path(self.out_dir) / 'records'

    @property
    def runs_dir(self):
        return Path(self.out_dir) / 'runs'

    @property
    def tables_dir(self):
        return Path(self.out_dir) / 'tables'

    @property
    def figures_dir(self):
        return Path(self.out_dir) / 'figures'


@dataclass
class Variant:
    """One mesh variant: family, h = finest cell size [m], parameter overrides, mesh; optional solver_defaults flags,
    a solver node (type + settings) for `solver_role` of `solver_physics`, and swept values recorded with the run."""
    family: str
    h: float
    parameters: dict
    mesh: MeshSpec
    solver_defaults: dict = field(default_factory=dict)
    solver_type: str | None = None
    solver_settings: dict = field(default_factory=dict)
    solver_role: str = 'linear'
    solver_physics: tuple = ()
    sweep: dict = field(default_factory=dict)

    @property
    def label(self):
        extra = ''.join(f'_{key}{_label_value(value)}' for key, value in sorted(self.sweep.items()))
        return f'{self.family}_h{self.h:.4g}{extra}'


def _label_value(value):
    return f'{value:g}' if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)


# ---- meshes ----
def uniform_mesh(dx):
    return MeshSpec(mode='uniform', dx=dx)


def graded_mesh(dx_base, refinements=(), max_ratio=1.15, x_bands=(), y_bands=(), outer_dx=None, outer_ratio=None):
    """Graded mesh: base dx, object refinements [(object, dx, margin)] and S7.1 axis bands [(lo, hi, dx)] per axis
    (expressions allowed), optional outer grading."""
    axes = [MeshAxisSpec(axis, bands=[MeshBandSpec(lo, hi, dx) for lo, hi, dx in bands])
            for axis, bands in (('x', x_bands), ('y', y_bands)) if bands]
    return MeshSpec(mode='nonuniform', dx=dx_base, max_ratio=max_ratio,
                    refinements=[RefinementSpec(name, dx, margin) for name, dx, margin in refinements],
                    outer_dx=outer_dx, outer_ratio=outer_ratio, axes=axes)


def apply_variant(spec, variant):
    """Copy of spec with the variant's parameters, mesh, solver_defaults flags and solver node."""
    spec = spec.copy()
    for name, value in variant.parameters.items():
        spec.parameter(name).expression = value if isinstance(value, str) else repr(float(value))
    spec.meshes = [variant.mesh]   # the study's models are one-mesh models (physics mesh '')
    for name, value in variant.solver_defaults.items():
        if not hasattr(spec.solver_defaults, name):
            raise AttributeError(f'solver_defaults has no field {name!r}')
        setattr(spec.solver_defaults, name, value)
    if variant.solver_type is not None:
        spec.solvers = [node for node in spec.solvers if node.name != STUDY_SOLVER_NAME]
        spec.solvers.append(SolverSpec(STUDY_SOLVER_NAME, variant.solver_type, dict(variant.solver_settings)))
        for name in variant.solver_physics:
            spec.physics_by_name(name).solvers[variant.solver_role] = STUDY_SOLVER_NAME
        assigned = {name for physics in spec.physics for name in physics.solvers.values()}
        spec.solvers = [node for node in spec.solvers if node.name in assigned]   # replaced nodes: no unused warning
    return spec


def load_model(name):
    """ModelSpec of models/<name>.json."""
    return api.load(MODELS_DIR / f'{name}.json')


def load_example(relative_path):
    """ModelSpec of examples/<relative_path> without its sweep (needs a repository checkout)."""
    path = EXAMPLES_DIR / relative_path
    if not path.exists():
        raise FileNotFoundError(f'{path} not found: the mesh study reads examples/ from a repository checkout')
    spec = api.load(path)
    spec.study.sweep = None
    return spec


def selected_cases(options, part, names):
    """The names of this part that --case selects (no --case: all; a selection naming none of them: [] + a note)."""
    if not options.cases:
        return list(names)
    chosen = [name for name in names if name in options.cases]
    if not chosen:
        print(f"[{part}] --case names none of {', '.join(names)}: part skipped")
    return chosen


# ---- records ----
def record_file(options, part, case=None):
    return options.records_dir / (f'{part}_{case}.jsonl' if case else f'{part}.jsonl')


def read_records(path):
    """Records of a .jsonl file, the last line per label (earlier lines of a re-run label dropped)."""
    path = Path(path)
    if not path.exists():
        return []
    by_label = {}
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                by_label.pop(record['label'], None)
                by_label[record['label']] = record
    return list(by_label.values())


def append_record(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, default=_json_default) + '\n')


def labels_to_skip(path, options):
    """Labels already recorded that this call must not re-run (--force: none, --retry-failed: completed only)."""
    if options.force:
        return set()
    return {record['label'] for record in read_records(path)
            if not options.retry_failed or record.get('status') == COMPLETED}


def _json_default(value):
    if hasattr(value, 'item'):   # numpy scalars
        return value.item()
    return str(value)


# ---- runs ----
def run_variants(case, base_spec, variants, path, options):
    """Check + run every variant not yet recorded in path (one JSON line each, failures recorded too); returns the
    number of runs made."""
    skip = labels_to_skip(path, options)
    n_runs = 0
    for variant in variants:
        if variant.label in skip:
            print(f'[{case}] {variant.label}: already recorded, skipped')
            continue
        spec = apply_variant(base_spec, variant)
        report = api.check(spec)
        mesh = report.get('mesh') or {}
        base = {'case': case, 'family': variant.family, 'label': variant.label, 'h': variant.h,
                'parameters': variant.parameters, 'solver_defaults': variant.solver_defaults,
                'solver_type': variant.solver_type, 'solver_settings': variant.solver_settings,
                'sweep': variant.sweep, 'check_n_cells': mesh.get('n_cells'), 'mesh_uniform': mesh.get('uniform'),
                'mesh_note': mesh.get('note'), 'line_blocks': mesh.get('line_blocks'),
                'line_block_hint': mesh.get('line_block_hint'),
                'memory_estimate_gpu_peak': (report.get('memory_estimate') or {}).get('gpu_peak'),
                'check_issues': [f"[{issue['level']}] {issue['node']}: {issue['message']}"
                                 for level in ('error', 'warning', 'info') for issue in report['issues']
                                 if issue['level'] == level][:10]}   # infos: e.g. a physics ignoring subpixel
        if report['n_errors']:
            append_record(path, {**base, 'status': 'check_failed'})
            print(f"[{case}] {variant.label}: check failed: {base['check_issues'][0]}")
            continue
        if (mesh.get('n_cells') or 0) > options.max_cells:
            print(f"[{case}] {variant.label}: {mesh.get('n_cells'):,} cells > max cells, skipped (not recorded)")
            continue
        start = time.perf_counter()
        with api.run(spec, out_dir=options.runs_dir / case / variant.label, save_arrays=False,
                     max_wall_time=options.max_wall_time) as results:
            summary = results.summary()
        record = {**base, **summary_record(summary), 'script_seconds': time.perf_counter() - start}
        append_record(path, record)
        n_runs += 1
        print(f"[{case}] {variant.label}: {record['status']}, {record['n_cells']} cells, "
              f"{_number(record['study_seconds'])} s, derived {record['derived']}")
    return n_runs


def summary_record(summary):
    """The fields of a run's summary.json the analysis reads."""
    build = summary.get('build') or {}
    return {'status': summary.get('status'), 'message': summary.get('message'),
            'n_cells': build.get('n_cells'), 'nx': build.get('nx'), 'ny': build.get('ny'),
            'dx_min': build.get('dx_min'), 'dx_max': build.get('dx_max'),
            'wall_time': summary.get('wall_time'), 'study_seconds': summary.get('study_seconds'),
            'build_seconds': build.get('build_seconds'), 'gpu_bytes': build.get('session_gpu_bytes'),
            'derived': summary.get('derived') or {}, 'derived_reference': summary.get('derived_reference') or {},
            'derived_error': summary.get('derived_error') or {}, 'solvers': summary.get('solvers') or {},
            'warnings': list(summary.get('warnings') or [])[:5]}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _number(value):
    return 'n/a' if value is None else f'{value:.4g}' if isinstance(value, float) else str(value)
