"""The solver choices of a problem class (or model target) per backend, enumerated from open_sol itself: every
registered SolverType that fills the class's role, every combination of its choice settings (all of its 'choice'
params; a setting the problem class lists in ignored_settings is pinned to its resolved default instead), kept when
roles.solver_misfit admits it for that physics type, role, study and grid kind on the backend and the SolverType's
validate raises no error; host direct / cuDSS names whose package is missing, and every gpu choice without a usable
GPU, stay listed as unavailable."""
import dataclasses
import itertools
from dataclasses import dataclass
from types import SimpleNamespace

BACKENDS = ('cpu', 'gpu')
LIBRARY_SETTINGS = ('back_end', 'factorization')   # the setting naming a sparse factorisation library
# the reference of each role on the cpu backend: host SuperLU (speed-ups and eigenvalue differences are against it);
# a role without one (or whose SuperLU the role does not offer) takes its cpu Automatic
BASELINES = {'linear': 'sparse_direct:scipy_splu', 'eigen': 'arpack_shift_invert:superlu'}


@dataclass(frozen=True)
class Candidate:
    """One solver choice of a problem class on one backend: label (backend-free, what --solvers patterns match),
    SolverType name, resolved values, whether it is the Automatic of the role on that backend and grid kind, the
    baseline, and availability (reason: why not)."""
    label: str
    solver_type: str
    values: dict
    automatic: bool
    baseline: bool
    available: bool
    reason: str = ''
    backend: str = 'cpu'

    @property
    def row(self) -> str:
        """Report row / record key: the label, '<label> @gpu' on the gpu backend."""
        return self.label if self.backend == 'cpu' else f'{self.label} @{self.backend}'


def choice_names(solver_type) -> tuple:
    """Names of the 'choice' params of a SolverType (the settings the catalogue varies)."""
    return tuple(param.name for param in solver_type.params if param.kind == 'choice')


def _label_names(type_name, values) -> tuple:
    """Settings the short label of a type names, in order."""
    if type_name == 'sparse_direct':
        return ('back_end',)
    if type_name == 'arpack_shift_invert':
        return ('factorization',)
    if type_name in ('pcg', 'block_eigen'):
        return ('precision', 'preconditioner')
    return ('precision',) if 'precision' in values else ()


def label_of(type_name, values, varied=()) -> str:
    """'sparse_direct:mumps', 'arpack_shift_invert:pardiso_mixed', 'pcg:double/multigrid', 'block_eigen:mixed/jacobi',
    'line_block:mixed', 'banded'; a varied choice the short label does not name is appended as '/name=value'."""
    names = _label_names(type_name, values)
    label = f"{type_name}:{'/'.join(str(values[name]) for name in names)}" if names else type_name
    return label + ''.join(f'/{name}={values[name]}' for name in varied if name not in names)


def _value_sets(solver_type, pinned=()):
    """Every combination of the choice params of a SolverType but the pinned ones, the other params (pinned included)
    at their defaults."""
    defaults = {param.name: param.default for param in solver_type.params}
    varied = [param for param in solver_type.params if param.kind == 'choice' and param.name not in pinned]
    for combination in itertools.product(*(param.choices for param in varied)):
        values = dict(defaults)
        values.update(zip((param.name for param in varied), combination))
        yield values


def _pinned_values(values, pinned, solver_role, backend, grid) -> dict:
    """values with each pinned setting (ProblemClass.ignored_settings: the consumer ignores it) at what its default
    resolves to on the backend and grid kind ('auto' library settings: _resolved_value), so the one row left is labelled
    like the Automatic choice."""
    resolved = _resolved_value(values, solver_role, backend, grid)
    return {**values, **{name: resolved[name] for name in pinned}}


def _resolved_value(values, solver_role, backend, grid):
    """values with a library setting 'auto' replaced by what it resolves to on the backend and grid kind (gpu: the
    role's cuDSS back end when nvmath-python is installed, schema.auto_back_end_values; else host SuperLU)."""
    from open_sol.model.schema import auto_back_end_values
    from open_sol.solvers.sparse_direct import AUTO_BACK_END_KEY, expected_back_end
    resolved = dict(auto_back_end_values(values, solver_role, backend, grid))
    resolved.pop(AUTO_BACK_END_KEY, None)
    if resolved.get('back_end') == 'auto':
        resolved['back_end'] = expected_back_end('auto', backend)
    if resolved.get('factorization') == 'auto':
        resolved['factorization'] = 'superlu'
    return resolved


def _automatic_label(registry, solver_role, grid, backend) -> str:
    """Label of what Automatic resolves to on the backend and grid kind (SolverRole.automatic + its 'auto' library)."""
    type_name, settings = solver_role.automatic(backend, grid)
    solver_type = registry.solver_type(type_name)
    values = {param.name: param.default for param in solver_type.params}
    values.update(settings)
    return label_of(type_name, _resolved_value(values, solver_role, backend, grid), choice_names(solver_type))


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


def _gpu_missing() -> str:
    """Why the gpu backend cannot run here ('' when a CUDA device answers)."""
    from open_sol.core.backend import cupy_installed, gpu_available
    if not cupy_installed():
        return 'gpu backend: cupy is not installed'
    return '' if gpu_available() else 'gpu backend: no CUDA device answers'


def _unavailable(values, backend, gpu_missing) -> str:
    """Why a choice cannot run on this machine: no GPU for the gpu backend, a missing MUMPS / PARDISO wrapper
    (host_direct.missing_text; PARDISO off x86-64) or nvmath-python for a cuDSS back end ('' when it can)."""
    from open_sol.solvers.cudss_factor import cudss_available
    from open_sol.solvers.host_direct import HOST_DIRECT_BACK_ENDS, host_direct_available, missing_text
    if backend == 'gpu' and gpu_missing:
        return gpu_missing
    for name in LIBRARY_SETTINGS:
        library = values.get(name)
        if library in HOST_DIRECT_BACK_ENDS and not host_direct_available(library):
            return missing_text(library)
        if library in ('cudss', 'cudss_mixed') and not cudss_available():
            return f"{name} '{library}' needs nvmath-python with cuDSS (pip install nvmath-python[cu12])"
    return ''


def solver_candidates(target, backends=('cpu',)):
    """([Candidate], [{'label', 'backend', 'reason'}] excluded) of a problem class / model target (physics_type, role,
    study, grid, dimensions) on the backends: the baseline first, then by backend and label."""
    from open_sol.model.schema import get_registry
    from open_sol.solvers.roles import solver_misfit
    registry = get_registry()
    physics_type = registry.physics_type(target.physics_type)
    role = target.role
    solver_role = physics_type.solver_roles[role]
    gpu_missing = _gpu_missing() if 'gpu' in backends else ''
    candidates, excluded = [], []
    for backend in backends:
        automatic = _automatic_label(registry, solver_role, target.grid, backend)
        seen = set()
        for solver_type in registry.solver_types():
            if role not in solver_type.roles:
                continue
            pinned = tuple((getattr(target, 'ignored_settings', None) or {}).get(solver_type.name, ()))
            for values in _value_sets(solver_type, pinned):
                if pinned:
                    values = _pinned_values(values, pinned, solver_role, backend, target.grid)
                if any(values.get(name) == 'auto' for name in LIBRARY_SETTINGS):
                    continue   # 'auto' is one of the explicit library choices: the same run as that choice
                label = label_of(solver_type.name, values, choice_names(solver_type))
                if label in seen:
                    continue
                seen.add(label)
                misfit = solver_misfit(solver_type, values, physics_type, role, target.study, backend, target.grid)
                invalid = misfit or _validation_errors(solver_type, values, target.dimensions)
                missing = _unavailable(values, backend, gpu_missing)
                if invalid and not missing:
                    excluded.append({'label': label, 'backend': backend, 'reason': invalid})
                    continue
                if misfit:   # a misfit library name is excluded whether installed or not
                    excluded.append({'label': label, 'backend': backend, 'reason': misfit})
                    continue
                candidates.append(Candidate(label, solver_type.name, values, label == automatic,
                                            backend == 'cpu' and label == BASELINES.get(role), not missing, missing,
                                            backend))
    if not any(candidate.baseline for candidate in candidates):   # no SuperLU reference: the cpu Automatic
        candidates = [dataclasses.replace(candidate, baseline=candidate.backend == 'cpu' and candidate.automatic)
                      for candidate in candidates]
    candidates.sort(key=lambda candidate: (not candidate.baseline, BACKENDS.index(candidate.backend), candidate.label))
    return candidates, excluded
