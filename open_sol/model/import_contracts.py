"""Import contracts of the package (the layering and lazy-import rules, as data) and their AST check, run as the api
built-in selftest import_contracts. Standard library only; no package module is imported."""
import ast
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]   # the open_sol folder
ROOT_PACKAGE = __name__.split('.')[0]
INIT = '__init__'   # module names of packages end in it ('physics.em.__init__'); the root package is '__init__'
SCOPES = ('module', 'registration', 'any')
REGISTER_FUNCTION = 'register'   # definition modules' register(registry), run by get_registry
# Module names below the package (relative, dotted, packages as '<package>.__init__'); fnmatch patterns ('*' crosses
# dots). External targets are absolute module names ('scipy.sparse'); an external pattern also covers its submodules.
PHYSICS_RUNTIME_LAYERS = ('gui.*', 'api', 'io.*', 'viz.*', 'simulation', 'mesh_study.*', 'model.meshing',
                          'model.builder', 'model.build_masks', 'model.study', 'model.study_*', 'model.session',
                          'model.field_access*',
                          'model.validate', 'model.worker', 'model.templates', 'model.describe')
SELFTEST_MODULES = ('*.selftests*',)   # selftests, selftests_x, selftests3d (not selftest_helpers / _cache)
# (definition, selftests module) pairs imported in get_registry because the module holds templates (labels,
# descriptions, factories); every other selftests module loads only on registry.selftests()
TEMPLATE_SOURCES = (
    ('coupling.definitions_magnetic', 'physics.magnetic.selftests_coupled'),
    ('coupling.definitions_piezo', 'coupling.selftests_piezo'),
    ('coupling.definitions_pockels', 'coupling.selftests_pockels'),
    ('physics.acoustics.definition', 'physics.acoustics.selftests'),
    ('physics.em.definition', 'physics.em.selftests'),
    ('physics.em.definition', 'physics.em.selftests_effective_index'),
    ('physics.em.definition', 'physics.em.selftests_eigenfrequency'),
    ('physics.em.definition', 'physics.em.selftests_fdfd'),
    ('physics.em.definition', 'physics.em.selftests_harmonic_fdtd'),
    ('physics.em.definition', 'physics.em.selftests_hz'),
    ('physics.em.definition', 'physics.em.selftests_mode_ports'),
    ('physics.em.definition', 'physics.em.selftests_optimization'),
    ('physics.em.definition_bloch', 'physics.em.selftests_bloch'),
    ('physics.em.definition_bor', 'physics.em.selftests_bor'),
    ('physics.em.definition_modes', 'physics.em.selftests_carriers'),
    ('physics.em.definition_modes', 'physics.em.selftests_em_modes'),
    ('physics.em.definition_nonlinear', 'physics.em.selftests_nonlinear'),
    ('physics.em.definition_phase_change', 'physics.em.selftests_phase_change'),
    ('physics.em.definition_resonance', 'physics.em.selftests_resonance'),
    ('physics.em3d.definition', 'physics.em3d.selftests'),
    ('physics.em3d.definition', 'physics.em3d.selftests_bloch3d'),
    ('physics.em3d.definition', 'physics.em3d.selftests_fdfd3d'),
    ('physics.eme.definition', 'physics.eme.selftests'),
    ('physics.eme.definition', 'physics.eme.selftests3d'),
    ('physics.fluid.definition', 'physics.fluid.selftests'),
    ('physics.heat.definition', 'physics.heat.selftests'),
    ('physics.heat.definition', 'physics.heat.selftests_3d'),
    ('physics.heat.definition', 'physics.heat.selftests_thermal_noise'),
    ('physics.magnetic.definition3d', 'physics.magnetic.selftests_3d'),
    ('physics.magnetic.definition_ac', 'physics.magnetic.selftests_ac'),
    ('physics.magnetic.definition_magnetostatics', 'physics.magnetic.selftests'),
    ('physics.magnetic.definition_magnetostatics', 'physics.magnetic.selftests_fem'),
    ('physics.mechanics.definition', 'physics.mechanics.selftests'),
    ('physics.mechanics.definition3d', 'physics.mechanics.selftests_3d'),
    ('physics.mechanics.definition3d', 'physics.mechanics.selftests_bloch3d'),
    ('physics.mechanics.definition_optomechanics', 'physics.mechanics.selftests_optomechanics'),
    ('physics.membrane.definition', 'physics.membrane.selftests'),
    ('physics.acoustics.definition_harmonic', 'physics.acoustics.selftests_harmonic'),
    ('physics.nlfd.definition', 'physics.nlfd.selftests'),
    ('physics.nlfd.definition', 'physics.nlfd.selftests_adjoint'),
    ('physics.nlfd.definition', 'physics.nlfd.selftests_sheets'),
    ('physics.potential.definition', 'physics.potential.selftests'),
    ('physics.semiconductor.definition', 'physics.semiconductor.selftests'),
)


def tree(*names):
    """Patterns of each module or package name and every module below it."""
    return tuple(pattern for name in names for pattern in (name, f'{name}.*'))


@dataclass(frozen=True)
class Contract:
    """One import rule: modules matching `importers` (minus `skip`) import no internal `never` target and, with `only`
    set, no internal target outside it, nor an `external_never` module; `allowed` lists the exceptions."""
    name: str
    rule: str                   # the rule as a failure message reads it
    importers: tuple
    # 'module': imports that run at import time (outside function bodies); 'registration': those plus the ones inside
    # register() (both run in get_registry); 'any': all
    scope: str = 'any'
    only: tuple | None = None   # internal target patterns allowed (None = no whitelist)
    never: tuple = ()           # internal target patterns refused
    external_never: tuple = ()  # external module patterns refused
    skip: tuple = ()            # importer patterns exempt
    # (importer pattern, target pattern[, 'function']): exceptions, with 'function' only for imports inside functions
    allowed: tuple = ()

    def __post_init__(self):
        if self.scope not in SCOPES:
            raise ValueError(f"contract '{self.name}': scope must be one of {', '.join(SCOPES)}, got {self.scope!r}")
        for exception in self.allowed:
            if len(exception) not in (2, 3) or (len(exception) == 3 and exception[2] != 'function'):
                raise ValueError(f"contract '{self.name}': allowed entries are (importer, target[, 'function']), "
                                 f'got {exception!r}')


@dataclass(frozen=True)
class ImportEdge:
    """One import statement's target as seen from its module."""
    importer: str
    target: str
    internal: bool
    function: str   # name of the outermost function (or lambda) the import sits in; '' at module level
    line: int

    @property
    def in_function(self):
        return bool(self.function)


CONTRACTS = (
    Contract('cupy_module_level', 'cupy / cupyx are imported at module level only by core.backend (others: from '
             'core.backend import cp / cupyx_sparse / cupyx_ndimage), so the package imports without cupy',
             importers=('*',), scope='module', external_never=('cupy', 'cupyx'), skip=('core.backend',)),
    Contract('scipy_inside_functions', 'scipy is imported inside functions only', importers=('*',), scope='module',
             external_never=('scipy',)),
    Contract('direct_wrappers_lazy', 'python-mumps (mumps), pydiso and threadpoolctl are imported only inside '
             'functions of solvers.host_direct (optional packages; physics never reaches them)', importers=('*',),
             external_never=('mumps', 'pydiso', 'threadpoolctl'), skip=('solvers.host_direct',)),
    Contract('direct_wrappers_inside_functions', 'solvers.host_direct imports python-mumps (mumps), pydiso and '
             'threadpoolctl inside functions only, so open_sol imports without them',
             importers=('solvers.host_direct',), scope='module',
             external_never=('mumps', 'pydiso', 'threadpoolctl')),
    Contract('qt_only_in_gui', 'only gui imports PySide6 / pyqtgraph (api, CLI, describe, io and viz stay headless)',
             importers=('*',), external_never=('PySide6', 'pyqtgraph'), skip=tree('gui')),
    Contract('core_imports_core', 'core imports nothing outside core but its listed exceptions',
             importers=tree('core'), only=tree('core'),
             allowed=(('core.fv_diffusion', 'solvers.pcg'), ('core.fv_diffusion', 'solvers.sparse_direct', 'function'),
                      ('core.memory_estimate', 'solvers.pcg', 'function'),
                      ('core.memory_estimate', 'solvers.multigrid', 'function'),
                      ('core.memory_estimate', 'solvers.host_direct', 'function'),
                      ('core.mirror', 'model.geometry', 'function'),
                      ('core.time_profile', 'solvers.sparse_direct', 'function'),
                      ('core.time_profile', 'solvers.pcg', 'function'))),
    Contract('core_time_estimate_pure', 'core.time_estimate imports no package module', importers=('core.time_estimate',),
             only=()),
    Contract('solver_runtimes', 'solvers runtimes import nothing but solvers and core (never physics or model)',
             importers=tree('solvers'), only=tree('solvers', 'core'),
             skip=('solvers.definition', *SELFTEST_MODULES)),
    Contract('solver_runtimes_lazy_core', 'solvers runtimes import core inside functions only (numpy at module level)',
             importers=tree('solvers'), scope='module', never=tree('core'),
             skip=('solvers.definition', *SELFTEST_MODULES)),
    Contract('fem_below_model', 'fem never imports model, physics, coupling or the front ends', importers=tree('fem'),
             never=tree('model', 'physics', 'coupling', 'gui', 'io', 'viz', 'mesh_study') + ('api', 'simulation'),
             skip=SELFTEST_MODULES),
    Contract('fem_init_no_core', 'fem/__init__ never imports core', importers=('fem.__init__',), never=tree('core')),
    Contract('model_base', 'model expressions / schema / spec / geometry import only core, each other and '
             'solvers.roles at module level', scope='module',
             importers=('model.expressions', 'model.schema', 'model.spec', 'model.geometry'),
             only=tree('core') + ('model.expressions', 'model.schema', 'model.spec', 'model.geometry', 'solvers.roles')),
    Contract('model_analytic', 'model.analytic imports only model.expressions at module level (GUI-thread safe)',
             importers=('model.analytic',), scope='module', only=('model.expressions',)),
    Contract('builder_session_no_study', 'model.study imports the builder: builder / session (and the field_access '
             'modules) never import model.study', importers=('model.builder', 'model.session', 'model.field_access*'),
             never=('model.study', 'model.study_*')),
    Contract('field_access_below_session', 'model.field_access* never import model.session (Session imports them; '
             'they reach it through FieldAccess.view)', importers=('model.field_access*', 'model.field_data'),
             never=('model.session', 'model.builder')),
    Contract('build_masks_below_builder', 'model.build_masks never imports model.builder or session (BuildContext '
             'imports it; it reaches the context through MeshMasks.ctx)', importers=('model.build_masks',),
             never=('model.builder', 'model.session', 'model.study', 'model.study_*')),
    Contract('solver_hooks_pure', 'model.solver_hooks (the study hook Protocols; builder and studies import it) imports '
             'no package module', importers=('model.solver_hooks',), only=()),
    Contract('physics_below_runtime', 'physics modules never import the model runtime (meshing, builder, study, '
             'session, validate, worker) or the front ends; shared study rules go to core or model.definition_helpers',
             importers=tree('physics'), never=PHYSICS_RUNTIME_LAYERS, skip=SELFTEST_MODULES),
    Contract('coupling_definitions_below_runtime', 'coupling and solver definitions never import the model runtime',
             importers=('coupling.definitions', 'coupling.definitions_*', 'solvers.definition'),
             never=PHYSICS_RUNTIME_LAYERS),
    Contract('definition_helpers_below_runtime', 'model.definition_helpers / definition_resonance (imported by every '
             'physics definition) never import the model runtime, and open_sol.materials only inside functions',
             importers=('model.definition_helpers', 'model.definition_resonance'), never=PHYSICS_RUNTIME_LAYERS),
    Contract('definition_helpers_lazy_materials', 'model.definition_helpers / definition_resonance import '
             'open_sol.materials inside functions only (materials imports physics.em)', scope='module',
             importers=('model.definition_helpers', 'model.definition_resonance'), never=tree('materials')),
    Contract('physics_materials_lazy', 'physics imports open_sol.materials only inside functions (materials imports '
             'physics.em)', importers=tree('physics'), scope='module', never=tree('materials')),
    Contract('physics_init_no_definition', 'physics/<name>/__init__ never imports its definition modules',
             importers=('physics.*.__init__',), never=('physics.*.definition', 'physics.*.definition_*')),
    Contract('em_upstream', 'physics.em imports no other physics package (em3d, eme, nlfd, heat, ... import em)',
             importers=tree('physics.em'), never=('physics.*',), allowed=(('physics.em.*', 'physics.em.*'),)),
    Contract('mechanics_not_acoustics', 'physics.acoustics imports physics.mechanics, never the reverse',
             importers=tree('physics.mechanics'), never=tree('physics.acoustics')),
    Contract('magnetic_not_potential', 'physics.magnetic never imports physics.potential (its sigma_dc Params equal '
             'electric_currents field by field instead)', importers=tree('physics.magnetic'),
             never=tree('physics.potential')),
    Contract('fluid_acoustics_lazy', 'physics.fluid imports physics.acoustics inside functions only',
             importers=tree('physics.fluid'), scope='module', never=tree('physics.acoustics')),
    Contract('viz_below_model', 'viz never imports model (viz.scene_figure duck-types Scene3D) or the front ends',
             importers=tree('viz'), never=tree('model', 'gui') + ('api',)),
    Contract('viz_physics_lazy', 'viz imports physics inside functions only', importers=tree('viz'), scope='module',
             never=tree('physics')),
    Contract('gui_on_top', 'gui is imported by nothing outside gui', importers=('*',), never=tree('gui'),
             skip=tree('gui')),
    Contract('api_importers', 'api is imported only by the package root, __main__ and mesh_study (gui, '
             'model.selftests_mesh and model.selftests_time inside functions)', importers=('*',), never=('api',),
             skip=('__init__', '__main__', 'api') + tree('mesh_study'),
             allowed=(('gui.*', 'api', 'function'), ('model.selftests_mesh', 'api', 'function'),
                      ('model.selftests_time', 'api', 'function'))),
    Contract('mesh_study_on_demand', "mesh_study is imported only inside __main__'s mesh-study handler",
             importers=('*',), never=tree('mesh_study'), skip=tree('mesh_study'),
             allowed=(('__main__', 'mesh_study.*', 'function'),)),
    Contract('selftests_module_level', 'selftests modules import only core.constants, model.spec / schema / '
             'selftest_helpers and sibling selftests modules at module level (physics inside their checks)',
             importers=SELFTEST_MODULES, scope='module',
             only=('core.constants', 'model.spec', 'model.schema', 'model.selftest_helpers', *SELFTEST_MODULES)),
    Contract('selftests_lazy', 'get_registry imports no selftests module: definitions import their cases inside a '
             'selftest loader (registry.register_selftests), never at module level or in register(), except the '
             'TEMPLATE_SOURCES that hold templates', importers=('*',), scope='registration', never=SELFTEST_MODULES,
             skip=SELFTEST_MODULES, allowed=TEMPLATE_SOURCES),
    Contract('selftest_helpers', 'model.selftest_helpers imports only model.spec (solvers.cudss_factor inside '
             'skip_without_cudss, solvers.host_direct inside the host direct helpers)',
             importers=('model.selftest_helpers',), only=('model.spec',),
             allowed=(('model.selftest_helpers', 'solvers.cudss_factor', 'function'),
                      ('model.selftest_helpers', 'solvers.host_direct', 'function'))),
    Contract('coupling_runtime', 'coupling runtime modules import only coupling, core.constants and core.backend (and '
             'the listed E13 helpers)', importers=tree('coupling'),
             only=tree('coupling') + ('core.constants', 'core.backend'),
             skip=('coupling.definitions', 'coupling.definitions_*', *SELFTEST_MODULES),
             allowed=(('coupling.standard', 'physics.heat.coupling'),
                      ('coupling.slot_sum', 'physics.em.index_reference'),
                      ('coupling.index_terms', 'physics.em.index_reference'))),
    Contract('nlfd_tables_numpy_only', 'nlfd mixing / tensors / sheets import only core.constants and each other at '
             'module level (GUI thread)', scope='module',
             importers=('physics.nlfd.mixing', 'physics.nlfd.tensors', 'physics.nlfd.sheets'),
             only=('core.constants', 'physics.nlfd.mixing', 'physics.nlfd.tensors')),
    Contract('crystal_tensors_numpy_only', 'materials.crystal_tensors imports no package module at module level',
             importers=('materials.crystal_tensors',), scope='module', only=()),
)


def _matches(name, patterns):
    return any(fnmatchcase(name, pattern) for pattern in patterns)


def _module_files():
    """{module parts: (path, is_package)} of every .py file of the package."""
    modules = {}
    for path in sorted(PACKAGE_DIR.rglob('*.py')):
        parts = path.relative_to(PACKAGE_DIR).with_suffix('').parts
        if parts[-1] == INIT:
            modules[parts[:-1]] = (path, True)
        else:
            modules[parts] = (path, False)
    return modules


def _module_name(parts, modules):
    """Relative dotted name of module parts ('<package>.__init__' for a package, '__init__' for the root)."""
    entry = modules.get(parts)
    return '.'.join((*parts, INIT)) if entry is None or entry[1] else '.'.join(parts)


def _internal_targets(base, names, modules):
    """Module parts imported by `from <base> import names`: each name that is a module, else base itself once."""
    submodules = [(*base, name) for name in names if (*base, name) in modules]
    return submodules or [base]


def module_imports(parts, modules):
    """ImportEdges of one module (relative imports resolved; `from x import y` names y when y is a module)."""
    path, is_package = modules[parts]
    importer = _module_name(parts, modules)
    package = parts if is_package else parts[:-1]
    edges = []

    def add(target_parts, internal, node, function):
        target = _module_name(target_parts, modules) if internal else target_parts
        edges.append(ImportEdge(importer, target, internal, function, node.lineno))

    def visit(node, function):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Import):
                for alias in child.names:
                    top, _, rest = alias.name.partition('.')
                    if top == ROOT_PACKAGE:
                        add(tuple(rest.split('.')) if rest else (), True, child, function)
                    else:
                        add(alias.name, False, child, function)
            elif isinstance(child, ast.ImportFrom):
                names = [alias.name for alias in child.names]
                module_parts = tuple(child.module.split('.')) if child.module else ()
                if child.level:
                    if child.level - 1 > len(package):
                        continue   # beyond the root: Python raises at import
                    base = package[:len(package) - (child.level - 1)] + module_parts
                elif module_parts[:1] == (ROOT_PACKAGE,):
                    base = module_parts[1:]
                else:
                    add(child.module, False, child, function)
                    continue
                for target_parts in _internal_targets(base, names, modules):
                    add(target_parts, True, child, function)
            if not function and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, child.name)
            else:
                visit(child, function or ('<lambda>' if isinstance(child, ast.Lambda) else ''))

    visit(ast.parse(path.read_text(encoding='utf-8'), filename=str(path)), '')
    return edges


def _exception_index(contract, edge):
    """Index of the `allowed` entry covering edge, else None."""
    for index, (importer, target, *where) in enumerate(contract.allowed):
        if (_matches(edge.importer, (importer,)) and _matches(edge.target, (target,))
                and (not where or edge.in_function)):
            return index
    return None


def _breaks(contract, edge):
    """True when edge breaks the contract, ignoring its exceptions."""
    if contract.scope == 'module' and edge.in_function:
        return False
    if contract.scope == 'registration' and edge.function not in ('', REGISTER_FUNCTION):
        return False
    if not edge.internal:
        return any(edge.target == name or edge.target.startswith(f'{name}.') for name in contract.external_never)
    if _matches(edge.target, contract.never):
        return True
    return contract.only is not None and not _matches(edge.target, contract.only)


def check(contracts=CONTRACTS):
    """{'n_modules', 'n_imports', 'violations': {contract: ['importer:line -> target', ...]}, 'stale': [...]}; stale
    = importer patterns matching no module and exceptions no import used (a rule out of date with the code)."""
    modules = _module_files()
    names = [_module_name(parts, modules) for parts in modules]
    edges = [edge for parts in modules for edge in module_imports(parts, modules)]
    violations, stale = {}, []
    for contract in contracts:
        used = set()
        for edge in edges:
            if not _matches(edge.importer, contract.importers) or _matches(edge.importer, contract.skip):
                continue
            if not _breaks(contract, edge):
                continue
            index = _exception_index(contract, edge)
            if index is None:
                where = ' (inside a function)' if edge.in_function else ''
                violations.setdefault(contract.name, []).append(
                    f'{edge.importer}:{edge.line} -> {edge.target}{where}')
            else:
                used.add(index)
        if not any(_matches(name, contract.importers) for name in names):
            stale.append(f"{contract.name}: importers {', '.join(contract.importers)} match no module")
        stale.extend(f'{contract.name}: exception {exception!r} is not used by any import'
                     for index, exception in enumerate(contract.allowed) if index not in used)
    return {'n_modules': len(modules), 'n_imports': len(edges), 'violations': violations, 'stale': stale}


def rule_text(name, contracts=CONTRACTS):
    """The rule text of the contract called name."""
    return next(contract.rule for contract in contracts if contract.name == name)
