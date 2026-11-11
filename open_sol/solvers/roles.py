"""Solver roles, SolverRole (what a physics hands out), SolverType (what a solver fills), ResolvedSolver, plus the one
fit rule `solver_misfit`. Pure Python: no cupy, no model imports (CPU, GUI-thread safe)."""
from collections.abc import Callable
from dataclasses import dataclass, field

# role -> GUI Add solver submenu text, in menu order
SOLVER_ROLES = {'stepper': 'Time stepping', 'linear': 'Linear systems', 'shifted': 'Frequency sweeps',
                'eigen': 'Eigenvalues'}
# operator class -> text used in misfit reasons
OPERATOR_CLASSES = {'spd': 'symmetric positive definite', 'complex_symmetric': 'complex symmetric',
                    'general': 'general (non-symmetric)'}
BACKENDS = {'gpu': 'GPU', 'cpu': 'CPU'}
# precision -> GUI text; storage / compute dtypes per role are the runtime's business (solver_plan S5)
PRECISIONS = {'single': 'single precision', 'double': 'double precision', 'mixed': 'mixed precision',
              'half_storage': 'half-precision storage'}
# solver_plan S7.0 adds kinds; a 'line' (1D model, one cell across: symmetry 2.20) and a 'tensor_3d' (a 3D model's
# tensor mesh: its own kind only so that grid_defaults can differ from 2D) are tensor meshes, so solvers and physics
# declaring 'tensor' fit them too (solver_misfit); 'triangles' / 'tetrahedra' = a MeshSpec of that kind (FEM, S7.5;
# igor3d_plan T4b)
GRID_KINDS = {'tensor': 'rectilinear (uniform or graded) mesh', 'line': '1D line of cells (a 1D model)',
              'tensor_3d': '3D rectilinear (uniform or graded) mesh',
              'triangles': 'triangle (finite element) mesh', 'tetrahedra': 'tetrahedron (finite element) mesh'}
TENSOR_GRID_KINDS = ('tensor', 'line', 'tensor_3d')   # fitted by a declared 'tensor'


def _require_keys(owner, what, values, vocabulary, allow_empty=False):
    """Raise ValueError unless `values` is a tuple of keys of `vocabulary` (non-empty unless allow_empty)."""
    if not isinstance(values, tuple) or (not values and not allow_empty) or not set(values) <= set(vocabulary):
        raise ValueError(f"{owner}: {what} must be a {'' if allow_empty else 'non-empty '}tuple of "
                         f"{', '.join(vocabulary)}, got {values!r}")


@dataclass(frozen=True)
class SolverRole:
    """One numerical job of a physics type: its operator class, the Automatic solver (type + settings), the studies it
    is used in (() = every study the physics runs in), the precisions the physics' own code accepts and, per solver
    setting it restricts, the values it accepts (setting_choices, e.g. the pcg preconditioner)."""
    operator: str                      # OPERATOR_CLASSES
    default_type: str                  # SolverType name used by Automatic
    default_settings: dict = field(default_factory=dict)
    studies: tuple = ()
    precisions: tuple = ('single', 'double')
    setting_choices: dict = field(default_factory=dict)   # {setting name: tuple of accepted values}; others: any
    # {backend: (type name, settings)}: Automatic on that backend when it differs (cpu diffusion: sparse_direct)
    backend_defaults: dict = field(default_factory=dict)
    # SolverType names this role accepts in a role they fill only on opt-in (SolverType.opt_in_roles: line_block linear)
    opt_in_types: tuple = ()
    # {grid kind: (type name, settings)}: Automatic on that grid kind on every backend, before backend_defaults (1D
    # diffusion: banded)
    grid_defaults: dict = field(default_factory=dict)
    # {grid kind: back end}: what a resolved sparse_direct back_end 'auto' becomes on the gpu backend when nvmath-python
    # is installed (schema.resolved_solvers; a cuDSS back end of setting_choices['back_end'], e.g. the frequency sweeps
    # of pressure_acoustics / membrane: cudss_mixed; an 'eigen' role: a factorization of setting_choices['factorization']);
    # other grids, the cpu or no nvmath keep 'auto' (host SuperLU)
    auto_back_ends: dict = field(default_factory=dict)
    # pcg preconditioner values the role runs at precision 'mixed' (solver_misfit; solid_mechanics_3d: its own
    # matrix-free multigrid too)
    mixed_preconditioners: tuple = ('jacobi',)

    def automatic(self, backend='gpu', grid='tensor') -> tuple:
        """(SolverType name, settings dict) Automatic resolves to on the backend and grid kind."""
        type_name, settings = self.grid_defaults.get(
            grid, self.backend_defaults.get(backend, (self.default_type, self.default_settings)))
        return type_name, dict(settings)

    def __post_init__(self):
        owner = f"SolverRole (default '{self.default_type}')"
        if self.operator not in OPERATOR_CLASSES:
            raise ValueError(f"{owner}: unknown operator class {self.operator!r}; valid: {', '.join(OPERATOR_CLASSES)}")
        _require_keys(owner, 'precisions', self.precisions, PRECISIONS)
        if not isinstance(self.studies, tuple):
            raise ValueError(f'{owner}: studies must be a tuple of study type names, got {self.studies!r}')
        for name, accepted in self.setting_choices.items():
            if not isinstance(accepted, tuple) or not accepted:
                raise ValueError(f'{owner}: setting_choices[{name!r}] must be a non-empty tuple, got {accepted!r}')
        _require_keys(owner, 'backend_defaults keys', tuple(self.backend_defaults), BACKENDS, allow_empty=True)
        _require_keys(owner, 'grid_defaults keys', tuple(self.grid_defaults), GRID_KINDS, allow_empty=True)
        _require_keys(owner, 'auto_back_ends keys', tuple(self.auto_back_ends), GRID_KINDS, allow_empty=True)
        key = 'back_end' if 'back_end' in self.setting_choices else 'factorization'
        offered = self.setting_choices.get(key, ())
        for grid, back_end in self.auto_back_ends.items():
            if back_end == 'auto' or back_end not in offered:
                raise ValueError(f"{owner}: auto_back_ends[{grid!r}] must be a {key} of setting_choices[{key!r}] "
                                 f"other than 'auto', got {back_end!r}")
        if not isinstance(self.opt_in_types, tuple):
            raise ValueError(f'{owner}: opt_in_types must be a tuple of SolverType names, got {self.opt_in_types!r}')

    def used_in(self, study) -> bool:
        """True when this role is used in the study type named `study`."""
        return not self.studies or study in self.studies


@dataclass(frozen=True)
class SolverType:
    """Self-describing solver: the roles it fills, its settings (model.schema Params) and what it handles; create /
    validate / memory_estimate are its callbacks (signatures below)."""
    name: str
    label: str
    roles: tuple                       # SOLVER_ROLES keys it can fill
    params: tuple = ()                 # model.schema.Param; a 'precision' choice Param when it offers several
    description: str = ''              # agents (describe)
    summary: str = ''                  # GUI text; '' -> description
    operators: tuple = ('spd', 'complex_symmetric', 'general')
    backends: tuple = ('gpu',)
    precisions: tuple = ('double',)
    grids: tuple = ('tensor',)
    create: Callable | None = None     # create(ctx, request: SolverRequest) -> runtime object of the role protocol
    validate: Callable | None = None   # validate(vctx, values, path) -> list[Issue] (machine availability included)
    # memory_estimate(values, request) -> core.memory_estimate.MemoryEstimate (CPU): only what the solver object
    # allocates itself; work vectors the physics sizes (pcg workspace) stay in the physics' estimator
    memory_estimate: Callable | None = None
    # time_estimate(values, work: core.time_estimate.SolverWork) -> core.time_estimate.TimeEstimate (CPU, unpriced):
    # ops per unit of its work (iterations, solves, factorisations) and its count guesses, never seconds
    time_estimate: Callable | None = None
    # backend_misfit(values, backend) -> '' or the reason a setting does not run on that backend (sparse_direct
    # back_end 'cusolver_qr' on the cpu); checked by solver_misfit after `backends`
    backend_misfit: Callable | None = None
    # roles (of `roles`) it fills only where the physics' SolverRole lists it in opt_in_types (line_block 'linear':
    # em_fdfd_ez / _hz only, not the pcg / sparse_direct linear roles of other physics)
    opt_in_roles: tuple = ()
    # {setting name: values} a role accepts only where its setting_choices[name] lists them (sparse_direct back_end
    # cudss / cudss_mixed: offered by the roles whose physics handle a cuDSS factor)
    opt_in_settings: dict = field(default_factory=dict)

    def __post_init__(self):
        owner = f"SolverType '{self.name}'"
        _require_keys(owner, 'roles', self.roles, SOLVER_ROLES)
        if not isinstance(self.opt_in_roles, tuple) or not set(self.opt_in_roles) <= set(self.roles):
            raise ValueError(f'{owner}: opt_in_roles must be a tuple of its roles {self.roles}, got {self.opt_in_roles!r}')
        _require_keys(owner, 'operators', self.operators, OPERATOR_CLASSES)
        _require_keys(owner, 'backends', self.backends, BACKENDS)
        _require_keys(owner, 'precisions', self.precisions, PRECISIONS)
        _require_keys(owner, 'grids', self.grids, GRID_KINDS)
        names = [param.name for param in self.params]
        if len(set(names)) != len(names):
            raise ValueError(f'{owner}: duplicate param names in {names}')
        if len(self.precisions) > 1 and 'precision' not in names:
            raise ValueError(f"{owner}: several precisions need a 'precision' param choosing among them")
        for name, opt_in in self.opt_in_settings.items():
            if name not in names or not isinstance(opt_in, tuple) or not opt_in:
                raise ValueError(f'{owner}: opt_in_settings[{name!r}] must name one of its params with a non-empty '
                                 f'tuple of values, got {opt_in!r}')


@dataclass(frozen=True)
class SolverRequest:
    """What the build hands SolverType.create: the (physics, role) pair, the role declaration, resolved settings and
    backend."""
    physics_name: str
    role: str
    solver_role: SolverRole
    values: dict
    backend: str = 'gpu'


@dataclass(frozen=True)
class ResolvedSolver:
    """What fills one (physics, role): the SolverType, its resolved values and the solver node (None = Automatic)."""
    solver_type: SolverType
    values: dict
    node: str | None = None


def resolved_precision(solver_type, values) -> str:
    """Precision a solver node runs at: its 'precision' setting, else the type's only precision."""
    return values.get('precision', solver_type.precisions[0])


def solver_misfit(solver_type, values, physics_type, role, study, backend, grid='tensor') -> str:
    """'' when solver_type (with its resolved values) can fill `role` of physics_type in the study on the backend and
    grid kind, else the GUI / validation reason (plain text, labels only)."""
    solver_role = getattr(physics_type, 'solver_roles', {}).get(role)
    physics_label = physics_type.label
    if solver_role is None:
        return f'{physics_label} has no {SOLVER_ROLES[role].lower()} job'
    if not solver_role.used_in(study):
        return f'{physics_label} does not use {SOLVER_ROLES[role].lower()} in this study'
    if role not in solver_type.roles:
        return f'{solver_type.label} does not do {SOLVER_ROLES[role].lower()}'
    if role in solver_type.opt_in_roles and solver_type.name not in solver_role.opt_in_types:
        return f'{solver_type.label} does not do the {SOLVER_ROLES[role].lower()} of {physics_label}'
    if solver_role.operator not in solver_type.operators:
        return (f'{solver_type.label} cannot solve the {OPERATOR_CLASSES[solver_role.operator]} systems of '
                f'{physics_label}')
    physics_backends = getattr(physics_type, 'backends', ('gpu',))
    if backend not in solver_type.backends or backend not in physics_backends:
        who = solver_type.label if backend not in solver_type.backends else physics_label
        return f'{who} does not run on the {BACKENDS[backend]}'
    if solver_type.backend_misfit is not None:
        reason = solver_type.backend_misfit(values, backend)
        if reason:
            return reason
    physics_grids = getattr(physics_type, 'grids', ('tensor',))

    def fits(grids):   # a line (one cell across) and a 3D tensor mesh are tensor meshes
        return grid in grids or (grid in TENSOR_GRID_KINDS and 'tensor' in grids)
    if not (fits(solver_type.grids) and fits(physics_grids)):
        who = solver_type.label if not fits(solver_type.grids) else physics_label
        return f'{who} does not support a {GRID_KINDS[grid]}'
    precision = resolved_precision(solver_type, values)
    if precision not in solver_type.precisions or precision not in solver_role.precisions:
        return f'{physics_label} cannot run its {SOLVER_ROLES[role].lower()} in {PRECISIONS[precision]}'
    if (precision == 'mixed' and 'preconditioner' in values
            and values['preconditioner'] not in solver_role.mixed_preconditioners):
        return (f"{physics_label} cannot use {solver_type.label} with preconditioner '{values['preconditioner']}' at "
                'mixed precision (the float32 copy of its operator runs with Jacobi only; set precision double or '
                'single)')
    for name, accepted in solver_role.setting_choices.items():
        if name in values and values[name] not in accepted:
            return (f"{physics_label} cannot use {solver_type.label} with {name} '{values[name]}' (accepts "
                    f"{', '.join(map(str, accepted))})")
    for name, opt_in in solver_type.opt_in_settings.items():
        if values.get(name) in opt_in and name not in solver_role.setting_choices:
            return f"{physics_label} does not offer {solver_type.label} with {name} '{values[name]}'"
    return ''
