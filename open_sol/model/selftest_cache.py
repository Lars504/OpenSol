"""Selftest reuse cache (selftest_cache.json in the report directory): a case that passed or an example that ran to a
verdict is reused, not rerun, while every package code unit (top-level function, class, a file's module-level code)
its last run depended on is unchanged; the shared plumbing files of UNTRACKED_PATHS are no dependency."""
import ast
import functools
import gc
import hashlib
import importlib.metadata
import importlib.util
import os
import platform
import sys
import threading
import time
from pathlib import Path

from ..io.results_io import read_json, write_json

CACHE_FILE = 'selftest_cache.json'
SECTIONS = ('cases', 'examples')
# per section the record statuses kept for reuse: 'failed' and 'error' always rerun (a failure stays visible until fixed,
# even when the fix is in an UNTRACKED_PATHS file)
REUSABLE_STATUSES = {'cases': ('passed',), 'examples': ('passed', 'no_tolerance')}
MODULE_UNIT = '<module>'   # a file's module-level code: imports, constants, kernel sources, data classes
# Shared plumbing every build or run passes through (the runner, model building / validation / specs / estimates,
# material merging, results I/O, GUI): no dependency of any case, and the names its code uses are not followed, so an
# edit there reruns nothing. Run `selftest --reset` (e.g. overnight) after editing them. Package paths ('/', a trailing
# '/' = a folder); solvers, meshers, fem and physics stay tracked (a case depends on them only when it runs them).
UNTRACKED_PATHS = ('api.py', '__main__.py', 'model/builder.py', 'model/validate.py', 'model/schema.py',
                   'model/spec.py', 'model/session.py', 'model/study.py', 'model/study_adaptation.py', 'model/expressions.py', 'model/definition_helpers.py',
                   'model/sweep_reuse.py',
                   'model/describe.py', 'model/meshing.py', 'model/time_estimate.py', 'model/symmetry_check.py',
                   'core/memory_estimate.py', 'core/time_estimate.py', 'materials/__init__.py', 'io/', 'gui/')
_ROOT = os.path.normcase(str(Path(__file__).resolve().parents[1]))   # the open_sol package folder
_MAX_REEXPORTS = 8   # import hops followed to the module defining an imported name
_PACKAGE_PATHS = {}  # co_filename -> package path or None
_IMPORT_MAPS = {}    # module name -> _import_map


def _package_path(filename):
    """'core/constants.py' style path of a .py file inside the package (normcase), else None."""
    try:
        return _PACKAGE_PATHS[filename]
    except KeyError:
        pass
    full_path = os.path.normcase(os.path.abspath(filename))
    path = (os.path.relpath(full_path, _ROOT).replace(os.sep, '/')
            if full_path.startswith(_ROOT + os.sep) and full_path.endswith('.py') else None)
    _PACKAGE_PATHS[filename] = path
    return path


_OWN_PATH = _package_path(__file__)   # the tracer's own frames are no dependency
_UNTRACKED = tuple(os.path.normcase(path).replace(os.sep, '/') for path in UNTRACKED_PATHS)   # as _package_path


def _untracked(path):
    """True for the tracer's own file and the UNTRACKED_PATHS files / folders."""
    return path == _OWN_PATH or any(path == entry or (entry.endswith('/') and path.startswith(entry))
                                    for entry in _UNTRACKED)


def file_hash(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


class _FileUnits:
    """Code units of one package file: hashes {unit name: sha1} and refs {unit name: names its code uses}, both with
    MODULE_UNIT."""

    __slots__ = ('hashes', 'refs')

    def __init__(self, hashes, refs):
        self.hashes = hashes
        self.refs = refs


def _is_unit(node):
    """A top-level function, or an undecorated class with methods (a decorated / method-less class, e.g. a dataclass,
    belongs to MODULE_UNIT: its generated methods never run as package code, so the trace cannot see it used)."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return True
    return (isinstance(node, ast.ClassDef) and not node.decorator_list
            and any(isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) for item in node.body))


def _dump(node):
    """ast.dump of a node without docstrings (comments, blank lines, positions and docstrings do not count)."""
    for inner in ast.walk(node):
        body = getattr(inner, 'body', None)
        if (isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and body
                and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            inner.body = body[1:]
    return ast.dump(node)


def _used_names(node):
    """Names and attribute names in a node's code."""
    return {inner.id if isinstance(inner, ast.Name) else inner.attr
            for inner in ast.walk(node) if isinstance(inner, (ast.Name, ast.Attribute))}


def _sha1(text):
    return hashlib.sha1(text.encode('utf-8')).hexdigest()


def _file_units(full_path):
    """_FileUnits of a .py file; an unreadable or unparsable file is one MODULE_UNIT of its bytes."""
    try:
        tree = ast.parse(Path(full_path).read_text(encoding='utf-8'))
    except (OSError, SyntaxError, ValueError):
        try:
            digest = file_hash(full_path)
        except OSError:
            digest = None
        return _FileUnits({MODULE_UNIT: digest}, {MODULE_UNIT: frozenset()})
    body = tree.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]   # the module docstring
    dumps, refs, module_dumps, module_refs = {}, {}, [], set()
    for node in body:
        if _is_unit(node):
            dumps.setdefault(node.name, []).append(_dump(node))
            refs.setdefault(node.name, set()).update(_used_names(node))
        else:
            module_dumps.append(_dump(node))
            module_refs.update(_used_names(node))
    hashes = {name: _sha1('\n'.join(parts)) for name, parts in dumps.items()}
    hashes[MODULE_UNIT] = _sha1('\n'.join(module_dumps))
    refs = {name: frozenset(names) for name, names in refs.items()}
    refs[MODULE_UNIT] = frozenset(module_refs)
    return _FileUnits(hashes, refs)


@functools.cache
def package_units():
    """{package path: _FileUnits} of every .py file of the package, parsed once per process (the code its imports
    ran)."""
    units = {}
    for folder, subfolder_names, file_names in os.walk(_ROOT):
        subfolder_names[:] = [name for name in subfolder_names if name != '__pycache__' and not name.startswith('.')]
        for name in file_names:
            if name.endswith('.py'):
                full_path = os.path.join(folder, name)
                units[_package_path(full_path)] = _file_units(full_path)
    return units


def unit_hash(unit_id):
    """sha1 of a 'path::name' unit id in this process' snapshot; None when the file or unit does not exist."""
    path, _, name = unit_id.partition('::')
    file_units = package_units().get(path)
    return None if file_units is None else file_units.hashes.get(name)


def environment_key(backends, device_name):
    """What a reused record must share with this run: Python, numpy / scipy / cupy versions, backends, device."""
    def version(module_name):
        module = sys.modules.get(module_name)
        if module is not None:
            return getattr(module, '__version__', None)
        try:
            return importlib.metadata.version(module_name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {'python': platform.python_version(), 'numpy': version('numpy'), 'scipy': version('scipy'),
            'cupy': version('cupy'), 'backends': list(backends), 'device': device_name}


class DependencyTrace:
    """Context manager recording the code objects that run inside it (sys.settrace and threads started inside; call
    events only, no line events); units() -> the package code units they depend on. Unavailable under another tracer
    (debugger)."""

    def __init__(self):
        self.available = False
        self._codes = []

    def __enter__(self):
        self.available = sys.gettrace() is None
        if self.available:
            self._code_modules = code_modules = {}   # code object -> module name of its globals

            def trace_call(frame, event, arg):
                code = frame.f_code
                if code not in code_modules:
                    code_modules[code] = frame.f_globals.get('__name__')
                # None: no line events in this frame

            threading.settrace(trace_call)
            sys.settrace(trace_call)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.available:
            sys.settrace(None)
            threading.settrace(None)
            self._codes = list(self._code_modules.items())   # one C call: safe against a still-tracing thread
        return False

    def units(self):
        """Unit ids 'path::name' (None when tracing was unavailable): the units whose code ran, MODULE_UNIT and its
        references (kernel twin factories, feature callbacks) of every touched file, the units the run code names, and
        those that named units name in turn when they did not run themselves (memoised getters, numba factories).
        Code of the untracked files is skipped (no unit, its names not followed) and their units are dropped."""
        if not self.available:
            return None
        files = package_units()
        executed, named, touched = set(), set(), {}
        for code, module_name in self._codes:
            path = _package_path(code.co_filename)
            if path is None or _untracked(path):
                continue
            touched[path] = module_name
            top = code.co_qualname.partition('.')[0]
            file_units = files.get(path)
            executed.add((path, top if file_units is not None and top in file_units.hashes else MODULE_UNIT))
            named |= _unit_refs(path, module_name, frozenset(code.co_names + code.co_varnames + code.co_freevars))
        for path, module_name in touched.items():
            named.add((path, MODULE_UNIT))
            if path in files:
                named |= _unit_refs(path, module_name, files[path].refs[MODULE_UNIT])
        module_names = _module_names()
        second = set()
        for path, unit in named - executed:
            if unit != MODULE_UNIT and path in files and unit in files[path].refs and not _untracked(path):
                second |= _unit_refs(path, module_names.get(path), files[path].refs[unit])
        return {f'{path}::{unit}' for path, unit in executed | named | second if not _untracked(path)}


def _module_names():
    """{package path: module name} of the imported package modules."""
    names = {}
    for name, module in list(sys.modules.items()):
        file_name = getattr(module, '__file__', None)
        path = None if file_name is None else _package_path(file_name)
        if path is not None:
            names[path] = name
    return names


def _module_path(module_name):
    file_name = getattr(sys.modules.get(module_name), '__file__', None)
    return None if file_name is None else _package_path(file_name)


def _unit_refs(path, module_name, names):
    """(path, unit) pairs the names used by code of module_name (file path) refer to: its own units and the imported
    names followed to their defining package module."""
    file_units = package_units().get(path)
    found = set() if file_units is None else {(path, name) for name in names & file_units.hashes.keys()}
    imports = _import_map(module_name) if module_name else {}
    for name in names & imports.keys():
        found |= _imported_units(module_name, name, names)
    return found


def _imported_units(module_name, name, names):
    """(path, unit) pairs of an imported name: MODULE_UNIT of every package module on its re-export chain, plus the
    defining module's unit of that name, or for a module object the units of it named in `names`."""
    found = set()
    entry = _import_map(module_name).get(name)
    for _ in range(_MAX_REEXPORTS):
        if entry is None:
            break
        source, attribute = entry
        submodule = source if attribute is None else f'{source}.{attribute}'
        if attribute is None or submodule in sys.modules:   # a module object: names used on it are its attributes
            path = _module_path(submodule)
            file_units = None if path is None else package_units().get(path)
            if path is not None:
                found.add((path, MODULE_UNIT))
            if file_units is not None:
                found |= {(path, unit) for unit in names & file_units.hashes.keys()}
            break
        path = _module_path(source)
        if path is None:
            break
        found.add((path, MODULE_UNIT))
        file_units = package_units().get(path)
        if file_units is not None and attribute in file_units.hashes:
            found.add((path, attribute))
            break
        entry = _import_map(source).get(attribute)
    return found


def _import_map(module_name):
    """{local name: (absolute source module, imported name or None for `import x`)} of every import statement (any
    scope) in a package module's file; {} outside the package."""
    if module_name in _IMPORT_MAPS:
        return _IMPORT_MAPS[module_name]
    imports = {}
    module = sys.modules.get(module_name)
    file_name = getattr(module, '__file__', None)
    if file_name is not None and _package_path(file_name) is not None:
        try:
            tree = ast.parse(Path(file_name).read_text(encoding='utf-8'))
        except (OSError, SyntaxError, ValueError):   # unreadable now: the traced units themselves still count
            tree = None
        package = module.__package__ or module_name.rpartition('.')[0]
        for node in [] if tree is None else ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                try:
                    source = (importlib.util.resolve_name('.' * node.level + (node.module or ''), package)
                              if node.level else node.module)
                except (ImportError, ValueError):
                    continue
                for alias in node.names:
                    if alias.name != '*':
                        imports[alias.asname or alias.name] = (source, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    top_name = alias.name.partition('.')[0]
                    imports[alias.asname or top_name] = (alias.name if alias.asname else top_name, None)
    _IMPORT_MAPS[module_name] = imports
    return imports


class SelftestCache:
    """selftest_cache.json: per case / example the record of its last reusable run, its environment, an input
    hash (example model file) and the hashes of the package code units it depended on; reusable while all of them are
    unchanged. 'wall_times' keeps the last run time of every key whatever its status (progress estimates)."""

    def __init__(self, path, environment):
        self.path = Path(path)
        self.environment = environment
        package_units()   # the snapshot this run's records are compared and stored against
        self.entries = {section: {} for section in SECTIONS}
        self.wall_times = {section: {} for section in SECTIONS}
        if self.path.is_file():
            try:
                saved = read_json(self.path)
            except ValueError:   # a damaged cache file: everything runs and the file is rewritten
                saved = {}
            for section in SECTIONS:
                self.entries[section].update(saved.get(section) or {})
                self.wall_times[section].update((saved.get('wall_times') or {}).get(section) or {})
        # the parsed records (tens of MB of dicts), modules and registry live the whole selftest: out of the cyclic gc,
        # else every release_session gc.collect() walks them (~0.45 s per run); JSON data has no cycles to miss
        gc.freeze()

    def last_wall_time(self, section, key):
        """Seconds of the key's last run (else of its stored record), None when never timed."""
        seconds = self.wall_times[section].get(key)
        if seconds is None:
            seconds = (self.entries[section].get(key) or {}).get('record', {}).get('wall_time')
        return None if seconds is None else float(seconds)

    def record_wall_time(self, section, key, seconds):
        self.wall_times[section][key] = float(seconds)

    def reusable_record(self, section, key, input_hash=None):
        """The stored record (marked reused) when nothing it depends on changed, else None (also for an entry of the
        older file-level format)."""
        entry = self.entries[section].get(key)
        if (entry is None or entry.get('units') is None or entry['environment'] != self.environment
                or entry['input_hash'] != input_hash
                or entry['record'].get('status') not in REUSABLE_STATUSES[section]
                or any(unit_hash(unit_id) != digest for unit_id, digest in entry['units'].items())):
            return None
        return dict(entry['record'], reused=True, reused_from=entry['time'])

    def store(self, section, key, record, units, input_hash=None):
        """Keep a record of a REUSABLE_STATUSES status with its units (None: not traced) for reuse; drop the entry
        otherwise."""
        if units is None or record.get('status') not in REUSABLE_STATUSES[section]:
            self.entries[section].pop(key, None)
            return
        self.entries[section][key] = {
            'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'environment': self.environment, 'input_hash': input_hash,
            'units': {unit_id: unit_hash(unit_id) for unit_id in sorted(units)},   # None (new unit): stale next run
            'record': record}

    def write(self):
        write_json(self.path, {**self.entries, 'wall_times': self.wall_times})
