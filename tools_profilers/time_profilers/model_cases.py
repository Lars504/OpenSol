"""Whole-model cases of the solver profiler: a model JSON file run end to end by api.run (validate, mesh, build incl.
assembly, study, output) with one solver choice assigned to one (physics, role), so a solver is judged by the run it
serves rather than by one isolated factor + solve. Targets (parent) and one timed run (child)."""
# A target = (model file, physics, role) whose role the model's study uses; its candidates come from solver_catalog
# like a synthetic problem's (physics type / role / study / grid kind of the model). The child assigns the choice as a
# new solver node (the settings of the node the physics already uses for that role kept when it is of the same type,
# the choice params overlaid), so every other physics and role runs as written. Columns: the model as written, or one
# per value of a --scan parameter (NAME=v1,v2: e.g. a mesh resolution parameter). Phases from summary.json: build =
# build_seconds_total (every build of a sweep / refinement summed), study = study_seconds, output = the rest of the
# api.run wall (validation, meshes / results written), total = the api.run wall. A repeat is a fresh api.run in the
# same child (libraries loaded, kernels compiled after the cold one; open_sol's mesh / ordering caches are scoped to
# one run_study, so nothing carries over).
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

PROFILED_NODE = 'profiled_solver'   # name of the solver node the child adds (suffixed when the model has one)


@dataclass(frozen=True, eq=False)
class ModelTarget:
    """One (model file, physics, role) of the profiler, with the ProblemClass fields the catalogue and the report
    read; sizes = the scan values (strings) or (None,) = the model as written."""
    name: str
    description: str
    physics_type: str
    role: str
    study: str
    grid: str
    dimensions: int
    size_text: str
    sizes: tuple
    quick_sizes: tuple
    path: str
    physics_name: str
    scan_parameter: str | None = None
    parameters: dict = field(default_factory=dict)
    kind: str = 'model'


def model_targets(path, physics_names=None, roles=None, scan=None, parameters=None) -> list:
    """[ModelTarget] of a model file: every physics (or those named) and every solver role its study uses (or those
    named); scan (NAME, (values...)) or None; parameters {name: expression} applied to every run."""
    from open_sol import api
    from open_sol.model.schema import get_registry, solver_grid
    spec = api.load(path)
    registry = get_registry()
    study = spec.study.type
    stem = Path(path).stem
    known = [physics.name for physics in spec.physics]
    unknown = [name for name in (physics_names or ()) if name not in known]
    if unknown:
        raise ValueError(f"{path}: no physics {unknown} (has: {', '.join(known)})")
    scan_name, scan_values = scan if scan else (None, (None,))
    targets = []
    for physics_spec in spec.physics:
        if physics_names and physics_spec.name not in physics_names:
            continue
        physics_type = registry.physics_type(physics_spec.type)
        for role, solver_role in physics_type.solver_roles.items():
            if not solver_role.used_in(study) or (roles and role not in roles):
                continue
            size_text = (f'parameter {scan_name}' if scan_name else 'the model as written') + (
                f"; overrides {parameters}" if parameters else '')
            targets.append(ModelTarget(
                f'{stem}/{physics_spec.name}.{role}',
                f'{spec.title or stem} ({path}): physics {physics_spec.name}, role {role}, whole api.run',
                physics_spec.type, role, study, solver_grid(spec, physics_spec), spec.dimensions, size_text,
                tuple(scan_values), tuple(scan_values), str(path), physics_spec.name, scan_name,
                dict(parameters or {})))
    return targets


def assign_solver(spec, physics_name, role, solver_type_name, values, registry) -> str:
    """Add a solver node of solver_type_name with the choice params of `values` (over the settings of the node the
    physics uses for `role` when it is of the same type) and assign it; returns the node name."""
    from open_sol.model.spec import SolverSpec
    from tools_profilers.time_profilers.solver_catalog import choice_names
    physics_spec = spec.physics_by_name(physics_name)
    nodes = {node.name: node for node in spec.solvers}
    current = nodes.get((physics_spec.solvers or {}).get(role) or '')
    settings = dict(current.settings) if current is not None and current.type == solver_type_name else {}
    solver_type = registry.solver_type(solver_type_name)
    settings.update({name: values[name] for name in choice_names(solver_type) if name in values})
    name, index = PROFILED_NODE, 1
    while name in nodes:
        index += 1
        name = f'{PROFILED_NODE}_{index}'
    spec.solvers.append(SolverSpec(name, solver_type_name, settings))
    physics_spec.solvers[role] = name
    return name


def _run_parameters(case) -> dict:
    parameters = dict(case.get('parameters') or {})
    if case.get('scan_parameter') and case.get('size') is not None:
        parameters[case['scan_parameter']] = str(case['size'])
    return parameters


def prepare(case):
    """(assembly fields, repeat()) of a model case: the registry and the model are loaded here (before the memory
    baseline); each repeat() is one api.run into its own folder under case['run_dir']."""
    from open_sol import api
    from open_sol.model.schema import get_registry
    registry = get_registry()
    api.load(case['path'])   # fails here (status error) for an unreadable / invalid file
    run_root = Path(case['run_dir'])
    run_root.mkdir(parents=True, exist_ok=True)
    counter = {'index': 0}

    def repeat():
        counter['index'] += 1
        return run_model(case, registry, run_root / f"repeat_{counter['index']}")
    return {'model_path': case['path'], 'run_dir': str(run_root)}, repeat


def _seconds(value):
    return float(value) if isinstance(value, (int, float)) else None


def run_model(case, registry, run_dir) -> dict:
    """Repeat record of one api.run of the case's model with its solver choice (status ok when the run completed)."""
    from open_sol import api
    spec = api.load(case['path'])
    node = assign_solver(spec, case['physics'], case['role'], case['solver_type'], case['values'], registry)
    started = time.perf_counter()
    results = api.run(spec, out_dir=run_dir, backend=case['backend'], parameters=_run_parameters(case) or None,
                      save_arrays=False)
    try:
        run_status = results.status
    finally:
        results.close()
    wall = time.perf_counter() - started
    summary_path = Path(run_dir) / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.is_file() else {}
    build_seconds, study_seconds = _seconds(summary.get('build_seconds_total')), _seconds(summary.get('study_seconds'))
    output = None if build_seconds is None or study_seconds is None else max(wall - build_seconds - study_seconds, 0.0)
    build = summary.get('build') or {}
    status = 'ok' if summary.get('status', run_status) == 'completed' else 'failed'
    details = {'run_status': summary.get('status', run_status), 'message': summary.get('message') or '',
               'errors': list(summary.get('errors') or [])[:3], 'node': node,
               'solvers': summary.get('solvers') or {}, 'solver_work': summary.get('solver_work') or {},
               'derived': summary.get('derived') or {}, 'derived_error': summary.get('derived_error') or {},
               'n_warnings': len(summary.get('warnings') or ()), 'n_cells': build.get('n_cells'),
               'gpu_peak_bytes': build.get('gpu_peak_bytes'), 'external_memory': summary.get('external_memory'),
               'summary_wall': _seconds(summary.get('wall_time')), 'run_dir': os.fspath(run_dir)}
    phases = {'build': build_seconds, 'study': study_seconds, 'output': output, 'total': wall}
    record = {'status': status, 'phases': phases, 'residual': None, 'tolerance': None, 'details': details}
    if status != 'ok':
        record['reason'] = details['message'] or '; '.join(details['errors']) or f"run status {details['run_status']}"
    return record
