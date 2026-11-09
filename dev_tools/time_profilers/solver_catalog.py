"""The CPU solver choices of a problem class, enumerated from open_sol itself: every registered SolverType that fills
the class's role, every combination of its choice settings (back_end / factorization / precision / preconditioner),
kept when roles.solver_misfit admits it for that physics type, role, study and grid kind on the cpu backend and the
SolverType's validate raises no error; host direct names whose wrapper is missing stay listed as unavailable."""
import itertools
from dataclasses import dataclass
from types import SimpleNamespace

VARIED_SETTINGS = ('back_end', 'factorization', 'precision', 'preconditioner')   # choice params enumerated
LIBRARY_SETTINGS = ('back_end', 'factorization')   # the setting naming a sparse factorisation library
# the reference of each role: host SuperLU (speed-ups and eigenvalue differences are against it)
BASELINES = {'linear': 'sparse_direct:scipy_splu', 'eigen': 'arpack_shift_invert:superlu'}


@dataclass(frozen=True)
class Candidate:
    """One solver choice of a problem class: label, SolverType name, resolved values, whether it is the cpu Automatic
    of the role on the grid kind, the baseline, and availability (reason: why not)."""
    label: str
    solver_type: str
    values: dict
    automatic: bool
    baseline: bool
    available: bool
    reason: str = ''


def label_of(type_name, values) -> str:
    """'sparse_direct:mumps', 'arpack_shift_invert:pardiso_mixed', 'pcg:double/multigrid', 'line_block:mixed',
    'banded'."""
    if type_name == 'sparse_direct':
        return f"sparse_direct:{values['back_end']}"
    if type_name == 'arpack_shift_invert':
        return f"arpack_shift_invert:{values['factorization']}"
    if type_name == 'pcg':
        return f"pcg:{values['precision']}/{values['preconditioner']}"
    if 'precision' in values:
        return f"{type_name}:{values['precision']}"
    return type_name


def _value_sets(solver_type):
    """Every combination of the VARIED_SETTINGS choice params of a SolverType, the other params at their defaults."""
    defaults = {param.name: param.default for param in solver_type.params}
    varied = [param for param in solver_type.params if param.kind == 'choice' and param.name in VARIED_SETTINGS]
    for combination in itertools.product(*(param.choices for param in varied)):
        values = dict(defaults)
        values.update(zip((param.name for param in varied), combination))
        yield values


def _cpu_value(values):
    """values with a library setting 'auto' replaced by what it resolves to on the cpu backend (host SuperLU)."""
    from open_sol.solvers.sparse_direct import expected_back_end
    resolved = dict(values)
    if resolved.get('back_end') == 'auto':
        resolved['back_end'] = expected_back_end('auto', 'cpu')
    if resolved.get('factorization') == 'auto':
        resolved['factorization'] = 'superlu'
    return resolved


def _automatic_label(registry, solver_role, grid) -> str:
    """Label of what Automatic resolves to on the cpu backend and grid kind (SolverRole.automatic; 'auto': SuperLU)."""
    type_name, settings = solver_role.automatic('cpu', grid)
    solver_type = registry.solver_type(type_name)
    values = {param.name: param.default for param in solver_type.params}
    values.update(settings)
    return label_of(type_name, _cpu_value(values))


def _validation_errors(solver_type, values, dimensions) -> str:
    """'; '-joined error messages of the SolverType's validate on a stand-in context ('' when none; a validate that
    needs more of the context than spec.dimensions gives no verdict)."""
    if solver_type.validate is None:
        return ''
    context = SimpleNamespace(spec=SimpleNamespace(dimensions=dimensions))
    try:
        issues = solver_type.validate(context, values, ('solver', 'profile'))
    except Exception:   # a validate reading more of the stand-in context: no verdict
        return ''
    return '; '.join(issue.message for issue in issues if issue.level == 'error')


def _unavailable(values) -> str:
    """host_direct.missing_text of a MUMPS / PARDISO setting whose wrapper is not installed (or PARDISO off x86-64)."""
    from open_sol.solvers.host_direct import HOST_DIRECT_BACK_ENDS, host_direct_available, missing_text
    for name in LIBRARY_SETTINGS:
        library = values.get(name)
        if library in HOST_DIRECT_BACK_ENDS and not host_direct_available(library):
            return missing_text(library)
    return ''


def cpu_candidates(problem_class):
    """([Candidate], [{'label', 'reason'}] excluded) of a problem class: the baseline first, then by label."""
    from open_sol.model.schema import get_registry
    from open_sol.solvers.roles import solver_misfit
    registry = get_registry()
    physics_type = registry.physics_type(problem_class.physics_type)
    role = problem_class.role
    solver_role = physics_type.solver_roles[role]
    automatic = _automatic_label(registry, solver_role, problem_class.grid)
    baseline = BASELINES[role]
    candidates, excluded, seen = [], [], set()
    for solver_type in registry.solver_types():
        if role not in solver_type.roles:
            continue
        for values in _value_sets(solver_type):
            if any(values.get(name) == 'auto' for name in LIBRARY_SETTINGS):
                continue   # 'auto' is host SuperLU on the cpu: the same run as scipy_splu / superlu
            label = label_of(solver_type.name, values)
            if label in seen:
                continue
            seen.add(label)
            misfit = solver_misfit(solver_type, values, physics_type, role, problem_class.study, 'cpu',
                                   problem_class.grid)
            invalid = misfit or _validation_errors(solver_type, values, problem_class.dimensions)
            missing = _unavailable(values)
            if invalid and not missing:
                excluded.append({'label': label, 'reason': invalid})
                continue
            if misfit:   # a misfit host direct name is excluded whether installed or not
                excluded.append({'label': label, 'reason': misfit})
                continue
            candidates.append(Candidate(label, solver_type.name, values, label == automatic, label == baseline,
                                        not missing, missing))
    candidates.sort(key=lambda candidate: (not candidate.baseline, candidate.label))
    return candidates, excluded
