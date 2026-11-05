"""Safe arithmetic expressions over parameters, coordinates and physics fields (numpy / cupy generic)."""
import ast
import builtins
import collections
import difflib
import functools
import importlib
import keyword
import math
import unicodedata

import numpy as np

from ..core.constants import C0, ELEMENTARY_CHARGE, EPS0, HBAR, K_B, MU0, N_A, PLANCK


class ExpressionError(ValueError):
    """Invalid, unsafe or non-evaluable expression; the message names the offending part."""


def error_message(exc) -> str:
    """User-facing message of an exception (KeyError without the repr quotes)."""
    return str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc)


BUILTIN_CONSTANTS = {
    'pi': math.pi, 'c_const': float(C0), 'eps0_const': float(EPS0), 'mu0_const': float(MU0),
    'h_const': float(PLANCK), 'hbar_const': float(HBAR), 'e_const': float(ELEMENTARY_CHARGE),
    'kB_const': float(K_B), 'NA_const': float(N_A),
    'nm': 1e-9, 'um': 1e-6, 'mm': 1e-3, 'cm': 1e-2,
    'fs': 1e-15, 'ps': 1e-12, 'ns': 1e-9, 'us': 1e-6, 'ms': 1e-3,
    'THz': 1e12, 'GHz': 1e9, 'MHz': 1e6, 'kHz': 1e3,
}
CONSTANT_DESCRIPTIONS = {   # builtin constant -> meaning (describe('expressions'); describe checks the keys)
    'pi': 'ratio of circumference to diameter of a circle',
    'c_const': 'speed of light in vacuum [m/s]',
    'eps0_const': 'vacuum permittivity [F/m]',
    'mu0_const': 'vacuum permeability [H/m]',
    'h_const': 'Planck constant [J s]',
    'hbar_const': 'reduced Planck constant [J s]',
    'e_const': 'elementary charge [C]',
    'kB_const': 'Boltzmann constant [J/K]',
    'NA_const': 'Avogadro constant [1/mol]',
    'nm': 'length unit 1e-9 (m)',
    'um': 'length unit 1e-6 (m)',
    'mm': 'length unit 1e-3 (m)',
    'cm': 'length unit 1e-2 (m)',
    'fs': 'time unit 1e-15 (s)',
    'ps': 'time unit 1e-12 (s)',
    'ns': 'time unit 1e-9 (s)',
    'us': 'time unit 1e-6 (s)',
    'ms': 'time unit 1e-3 (s)',
    'THz': 'frequency unit 1e12 (Hz)',
    'GHz': 'frequency unit 1e9 (Hz)',
    'MHz': 'frequency unit 1e6 (Hz)',
    'kHz': 'frequency unit 1e3 (Hz)',
}

_ARRAY_FUNCTIONS = {   # expression function -> numpy / cupy attribute
    'sin': 'sin', 'cos': 'cos', 'tan': 'tan', 'asin': 'arcsin', 'acos': 'arccos', 'atan': 'arctan',
    'atan2': 'arctan2', 'sinh': 'sinh', 'cosh': 'cosh', 'tanh': 'tanh', 'exp': 'exp', 'log': 'log',
    'log10': 'log10', 'sqrt': 'sqrt', 'abs': 'abs', 'min': 'minimum', 'max': 'maximum', 'where': 'where',
    'floor': 'floor', 'ceil': 'ceil', 'sign': 'sign', 'pow': 'power', 'hypot': 'hypot',
}
# real special functions of one argument: scipy.special on numpy (imported on first call), cupyx.scipy.special on cupy
_SPECIAL_FUNCTIONS = ('j0', 'j1', 'y0', 'y1', 'i0', 'i1', 'k0', 'k1', 'erf', 'erfc')
FUNCTION_NAMES = frozenset(_ARRAY_FUNCTIONS) | frozenset(_SPECIAL_FUNCTIONS)
_VARIADIC_FUNCTIONS = frozenset({'min', 'max'})
# (min args, max args or None) in _ARRAY_FUNCTIONS then _SPECIAL_FUNCTIONS order (independent of the hash seed)
FUNCTION_ARITY = {name: (1, 1) for name in (*_ARRAY_FUNCTIONS, *_SPECIAL_FUNCTIONS)}
FUNCTION_ARITY.update(atan2=(2, 2), pow=(2, 2), hypot=(2, 2), where=(3, 3), min=(2, None), max=(2, None))
FUNCTION_DESCRIPTIONS = {   # function -> one-line signature and meaning (describe('expressions'))
    'sin': 'sin(x): sine of an angle x in rad',
    'cos': 'cos(x): cosine of an angle x in rad',
    'tan': 'tan(x): tangent of an angle x in rad',
    'asin': 'asin(x): inverse sine in rad, for x in [-1, 1]',
    'acos': 'acos(x): inverse cosine in rad, for x in [-1, 1]',
    'atan': 'atan(x): inverse tangent in rad, in (-pi/2, pi/2)',
    'atan2': 'atan2(y, x): angle of the point (x, y) in rad, in [-pi, pi], counter-clockwise from +x',
    'sinh': 'sinh(x): hyperbolic sine',
    'cosh': 'cosh(x): hyperbolic cosine',
    'tanh': 'tanh(x): hyperbolic tangent',
    'exp': 'exp(x): e to the power x',
    'log': 'log(x): natural logarithm, for x > 0',
    'log10': 'log10(x): base-10 logarithm, for x > 0',
    'sqrt': 'sqrt(x): square root, for x >= 0',
    'abs': 'abs(x): absolute value',
    'min': 'min(a, b, ...): element-wise minimum of 2 or more arguments',
    'max': 'max(a, b, ...): element-wise maximum of 2 or more arguments',
    'where': 'where(condition, a, b): element-wise a where condition is true (non-zero), else b; both a and b '
             'are evaluated everywhere',
    'floor': 'floor(x): largest whole number <= x (a float)',
    'ceil': 'ceil(x): smallest whole number >= x (a float)',
    'sign': 'sign(x): -1, 0 or 1 by the sign of x',
    'pow': 'pow(a, b): a to the power b, same as a**b',
    'hypot': 'hypot(a, b): sqrt(a**2 + b**2) without intermediate overflow',
    'j0': 'j0(x): Bessel function of the first kind J_0 (real x)',
    'j1': 'j1(x): Bessel function of the first kind J_1 (real x)',
    'y0': 'y0(x): Bessel function of the second kind Y_0, for x > 0 (-inf at 0, nan below)',
    'y1': 'y1(x): Bessel function of the second kind Y_1, for x > 0 (-inf at 0, nan below)',
    'i0': 'i0(x): modified Bessel function of the first kind I_0 (overflows to inf above x ~ 700)',
    'i1': 'i1(x): modified Bessel function of the first kind I_1 (overflows to inf above x ~ 700)',
    'k0': 'k0(x): modified Bessel function of the second kind K_0, for x > 0 (inf at 0, nan below)',
    'k1': 'k1(x): modified Bessel function of the second kind K_1, for x > 0 (inf at 0, nan below)',
    'erf': 'erf(x): error function (2/sqrt(pi)) * integral of exp(-u**2) from 0 to x, in (-1, 1)',
    'erfc': 'erfc(x): complementary error function 1 - erf(x), accurate where erf(x) is close to 1',
}

THREE_D_COORDINATE_NAMES = frozenset({'z'})      # 3D models only (validate rejects them in a 2D model)
# cell-centre coordinates [m] of field-dependent values (one place)
FIELD_COORDINATE_NAMES = frozenset({'x', 'y'}) | THREE_D_COORDINATE_NAMES
# simulated time [s]: derived values, state-dependent material properties and time_dependent feature params (0
# outside a time_dependent study; re-evaluated at the start of every runner step)
TIME_NAMES = frozenset({'t'})
COORDINATE_NAMES = FIELD_COORDINATE_NAMES | TIME_NAMES
RESERVED_NAMES = frozenset(BUILTIN_CONSTANTS) | FUNCTION_NAMES | COORDINATE_NAMES | frozenset({'value'})
# the parser NFKC-normalises the micro sign (U+00B5) to Greek mu (U+03BC): '5*µm' then means 5*um
_NAME_ALIASES = {'μm': 'um', 'μs': 'us'}


def is_reserved_name(name):
    """True for RESERVED_NAMES and the parser's aliases of them (non-ASCII, so kept out of RESERVED_NAMES, which
    describe lists): a parameter or alias named so would be read as the reserved name."""
    return name in RESERVED_NAMES or name in _NAME_ALIASES

_FUNCTION_SPELLINGS = {'ln': 'log'}   # common spelling -> expression function (unknown-function hint)
_SYNTAX_HINTS = {   # disallowed AST node / operator class -> actionable message
    'BitXor': "'^' is not allowed: use ** for powers",
    'BitAnd': "'&' is not allowed: use (a < x)*(x < b) for 'and'",
    'BitOr': "'|' is not allowed: use max(c1, c2) for 'or'",
    'Invert': "'~' is not allowed: use (c == 0) for 'not'",
    'Not': "'not' is not allowed: use (c == 0)",
    'BoolOp': "'and' / 'or' are not allowed: use (a < x)*(x < b) for 'and', max(c1, c2) for 'or', or where()",
    'IfExp': "'a if c else b' is not allowed: use where(c, a, b)",
    'MatMult': "'@' is not allowed: use * for products",
    'Attribute': 'attribute access (a.b) is not allowed: call functions by their plain name, e.g. sin(x)',
    'Subscript': 'indexing (a[i]) is not allowed',
}

MAX_EXPRESSION_LENGTH = 2000
# C-level imports (e.g. cupy compiling a kernel on first use) look up __import__ in the calling frame's builtins, so an
# empty dict raises KeyError '__import__'; the whitelist (no attributes, no free name outside the namespace) keeps it
# unreachable from expression text.
_EVAL_GLOBALS = {'__builtins__': {'__import__': builtins.__import__}}
_BINARY_OPERATORS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv)
_UNARY_OPERATORS = (ast.UAdd, ast.USub)
_COMPARISON_OPERATORS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)
_EVALUATION_ERRORS = (OverflowError, ZeroDivisionError, FloatingPointError, RecursionError, MemoryError,
                      TypeError, ValueError)


class Expression:
    """Compiled, whitelisted expression shared through the compile cache; `names` = free names (called functions
    and builtin constants excluded; an uncalled function name is a free name)."""

    __slots__ = ('text', 'names', '_code', '_constant_names', '_function_names')

    def __init__(self, text, names, code, constant_names, function_names):
        self.text = text
        self.names = names
        self._code = code
        self._constant_names = constant_names
        self._function_names = function_names

    def __repr__(self):
        return f'Expression({self.text!r})'

    def check_names(self, available_names, allowed=None):
        """Raise ExpressionError naming every free name not in `available_names`, with close matches among them;
        `allowed` (optional text) says what this value may use."""
        missing = sorted(self.names.difference(available_names))
        if missing:
            raise ExpressionError(_unknown_names_message(missing, available_names, allowed))

    def evaluate(self, namespace, xp=np, functions=None):
        """Scalar or array value; `namespace` maps free names to scalars/arrays (shadows builtin constants),
        functions run as `xp.*` (calls with scalar arguments only go through numpy and return floats); `functions`
        {name: callable} supplies the extra call names the expression was compiled with."""
        self.check_names(namespace)
        scope = {name: namespace[name] for name in self.names}
        for name in self._constant_names:
            scope[name] = namespace[name] if name in namespace else BUILTIN_CONSTANTS[name]
        table = _function_table(xp)
        for name in self._function_names:
            if name in table:
                scope[name] = table[name]
            elif functions is not None and name in functions:
                scope[name] = functions[name]
            else:
                raise ExpressionError(f'{name}() is not available here (derived-curve expressions only)')
        try:
            # numpy warns on NaN / inf where cupy is silent (e.g. the unused branch of where(x > 0, sqrt(x), 0)):
            # silent on both backends, non-finite values that are used are refused downstream
            with np.errstate(invalid='ignore', divide='ignore'):
                result = eval(self._code, _EVAL_GLOBALS, scope)
        except _EVALUATION_ERRORS as exc:
            raise ExpressionError(f'cannot evaluate {_quoted(self.text)}: {exc}') from exc
        if isinstance(result, complex) or getattr(getattr(result, 'dtype', None), 'kind', '') == 'c':
            raise ExpressionError(f'{_quoted(self.text)} gives complex values')
        return result


def compile_expression(text, functions=None):
    """Parse and whitelist-check `text` -> cached Expression (an Expression passes through unchanged); functions:
    extra call names {name: (min args, max args)} that evaluate(functions=) supplies; ExpressionError on bad syntax
    or disallowed nodes."""
    if isinstance(text, Expression):
        return text
    if not isinstance(text, str):
        raise ExpressionError(f'expression must be text, got {type(text).__name__}')
    return _compile_cached(text, tuple(sorted(functions.items())) if functions else ())


@functools.lru_cache(maxsize=4096)
def _compile_cached(text, extra_arity=()):
    """LRU-cached body of compile_expression (failures raise and are therefore not cached); extra_arity: sorted
    (name, (min, max)) pairs of the extra call names."""
    if len(text) > MAX_EXPRESSION_LENGTH:
        raise ExpressionError(f'expression longer than {MAX_EXPRESSION_LENGTH} characters')
    source = text.strip()
    if not source:
        raise ExpressionError('empty expression')
    free_names, constant_names, function_names = set(), set(), set()
    try:
        tree = ast.parse(source, mode='eval')
        _check_node(tree.body, free_names, constant_names, function_names, dict(extra_arity))
        code = compile(tree, '<expression>', 'eval')
    except ExpressionError:
        raise
    except SyntaxError as exc:
        raise ExpressionError(f'invalid expression {_quoted(source)}: {exc.msg}') from None
    except RecursionError:
        raise ExpressionError(f'expression {_quoted(source)} is nested too deeply') from None
    except (ValueError, TypeError, OverflowError, MemoryError) as exc:
        raise ExpressionError(f'invalid expression {_quoted(source)}: {exc}') from None
    both = sorted(free_names & function_names)   # one eval scope slot per name
    if both:
        raise ExpressionError(f"'{both[0]}' is used both as a name and as the function {both[0]}(...) in "
                              f'{_quoted(source)}; use it one way only')
    return Expression(source, frozenset(free_names), code, tuple(sorted(constant_names)),
                      tuple(sorted(function_names)))


def _check_node(node, free_names, constant_names, function_names, extra_arity):
    """Whitelist walk: raise on disallowed syntax, coerce int constants to float in place, collect names
    (extra_arity: {name: (min, max)} of call names allowed besides FUNCTION_NAMES)."""
    names = (free_names, constant_names, function_names, extra_arity)
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _BINARY_OPERATORS):
            raise ExpressionError(_not_allowed_message(node.op, 'operator'))
        _check_node(node.left, *names)
        _check_node(node.right, *names)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _UNARY_OPERATORS):
            raise ExpressionError(_not_allowed_message(node.op, 'operator'))
        _check_node(node.operand, *names)
    elif isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            raise ExpressionError('chained comparisons are not allowed')
        if not isinstance(node.ops[0], _COMPARISON_OPERATORS):
            raise ExpressionError(f'comparison {type(node.ops[0]).__name__} is not allowed')
        _check_node(node.left, *names)
        _check_node(node.comparators[0], *names)
    elif isinstance(node, ast.Call):
        if not (isinstance(node.func, ast.Name) and (node.func.id in FUNCTION_NAMES or node.func.id in extra_arity)):
            raise ExpressionError(_unknown_function_message(node.func, extra_arity))
        name = node.func.id
        if node.keywords:
            raise ExpressionError(f'{name}() takes positional arguments only')
        min_args, max_args = FUNCTION_ARITY[name] if name in FUNCTION_NAMES else extra_arity[name]
        n_args = len(node.args)
        if n_args < min_args or (max_args is not None and n_args > max_args):
            raise ExpressionError(f'{name}() takes {arity_text(min_args, max_args)}, got {n_args}')
        function_names.add(name)
        for argument in node.args:
            _check_node(argument, *names)
    elif isinstance(node, ast.Name):   # an uncalled function name is a free name (e.g. an xy series 'abs')
        node.id = _NAME_ALIASES.get(node.id, node.id)
        (constant_names if node.id in BUILTIN_CONSTANTS else free_names).add(node.id)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ExpressionError(f'constant {node.value!r} is not allowed')
        node.value = float(node.value)
    else:
        raise ExpressionError(_not_allowed_message(node, None))


def _not_allowed_message(node, category):
    """Hint for a disallowed node or operator (category 'operator' or None) from _SYNTAX_HINTS, else its class name."""
    name = type(node).__name__
    if name in _SYNTAX_HINTS:
        return _SYNTAX_HINTS[name]
    return f'operator {name} is not allowed' if category else f'{name} is not allowed in expressions'


def _unknown_function_message(func_node, extra_names=()):
    """Unknown call target -> message with a missing-'*' hint or close function names, plus the function list
    (extra_names: the extra call names of this expression)."""
    text = _quoted(ast.unparse(func_node))
    if not isinstance(func_node, ast.Name) or func_node.id in BUILTIN_CONSTANTS or func_node.id in COORDINATE_NAMES:
        return f"{text} is not a function: missing '*' before '('?"
    known = sorted(FUNCTION_NAMES.union(extra_names))
    close = ([_FUNCTION_SPELLINGS[func_node.id]] if func_node.id in _FUNCTION_SPELLINGS
             else difflib.get_close_matches(func_node.id, known, n=3))
    hint = f'did you mean {_or_list(close)}' if close else "or a missing '*' before '('"
    return f"unknown function {text} ({hint}?); functions: {', '.join(known)}"


def _or_list(names):
    """Quoted names joined as 'a', 'b' or 'c'."""
    quoted = [f"'{name}'" for name in names]
    return quoted[0] if len(quoted) == 1 else f"{', '.join(quoted[:-1])} or {quoted[-1]}"


def arity_text(min_args, max_args):
    """'1 argument', '2 arguments', '1 to 2 arguments' or 'at least 2 arguments' (errors and describe)."""
    if max_args is None:
        return f'at least {min_args} arguments'
    if min_args < max_args:
        return f'{min_args} to {max_args} arguments'
    return f"{min_args} argument{'' if min_args == 1 else 's'}"


def _quoted(text, limit=60):
    """Quoted, shortened expression text for messages."""
    return repr(text if len(text) <= limit else text[:limit - 3] + '...')


def _unknown_names_message(names, available_names=(), allowed=None):
    """Message listing unknown free names (sorted by the caller) with suggestions from `available_names`
    ('<physics>_<name>' qualified names first, else close spellings) and the optional `allowed` text."""
    listed = ', '.join(f"'{name}'" for name in names)
    message = f"unknown name{'s' if len(names) > 1 else ''} {listed}"
    candidates = sorted(name for name in available_names if isinstance(name, str))
    suggestions = []
    for name in names:
        matches = ([candidate for candidate in candidates if candidate.endswith(f'_{name}')]
                   or difflib.get_close_matches(name, candidates, n=3))
        if matches:
            suggestions.append(_or_list(matches) + (f" for '{name}'" if len(names) > 1 else ''))
    if suggestions:
        message += f"; did you mean {', '.join(suggestions)}?"
    functions = [name for name in names if name in FUNCTION_NAMES]
    if functions:
        message += f"; '{functions[0]}' is a function: call it as {functions[0]}(...)"
    return f'{message} ({allowed})' if allowed else message


@functools.lru_cache(maxsize=None)
def _function_table(xp):
    """Expression function name -> callable for array module `xp`."""
    table = {}
    for name, attribute in _ARRAY_FUNCTIONS.items():
        xp_function = getattr(xp, attribute)
        if name == 'where':   # cupy.where needs an array condition (a parameter comparison gives a bool)
            xp_function = lambda condition, if_true, if_false: xp.where(xp.asarray(condition), if_true, if_false)
        function = _dispatching(getattr(np, attribute), xp_function, xp.ndarray)
        table[name] = _variadic(function) if name in _VARIADIC_FUNCTIONS else function
    xp_special_module = 'scipy.special' if xp is np else 'cupyx.scipy.special'
    for name in _SPECIAL_FUNCTIONS:
        table[name] = _dispatching(_lazy_function('scipy.special', name), _lazy_function(xp_special_module, name),
                                   xp.ndarray)
    return table


def _lazy_function(module_name, name):
    """Callable importing module_name on first use and calling its function `name` (keeps scipy off the import)."""
    def call(*args):
        return getattr(importlib.import_module(module_name), name)(*args)
    return call


def _variadic(binary_function):
    """Pairwise reduction so min/max accept two or more arguments (each pair dispatched on its own)."""
    return lambda *args: functools.reduce(binary_function, args)


def _dispatching(numpy_function, xp_function, xp_array_type):
    """xp_function when any argument is an xp array, else numpy with 0-d results returned as Python scalars."""
    def call(*args):
        for argument in args:
            if isinstance(argument, xp_array_type):
                return xp_function(*args)
        result = numpy_function(*args)
        if isinstance(result, (np.ndarray, np.generic)) and result.ndim == 0:
            return result.item()
        return result
    return call


def _is_number(value):
    """int/float (numpy included), excluding bool."""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


def evaluate_scalar(value, parameters):
    """Number, expression text or Expression -> finite float over `parameters`; ExpressionError otherwise."""
    if _is_number(value):
        result, description = value, 'the number'
    elif isinstance(value, (str, Expression)):
        expression = compile_expression(value)
        result, description = expression.evaluate(parameters), _quoted(expression.text)
    else:
        raise ExpressionError(f'expected a number or expression, got {type(value).__name__}')
    if np.ndim(result) != 0:
        raise ExpressionError(f'{description} gives an array, expected a single number')
    try:
        number = float(result)
    except (OverflowError, TypeError, ValueError) as exc:   # e.g. an int beyond the float range
        raise ExpressionError(f'{description} is not a representable number: {exc}') from None
    if not math.isfinite(number):
        raise ExpressionError(f'{description} is not finite ({number})')
    return number


def referenced_names(value):
    """Free names of an expression (empty for numbers); ExpressionError on bad syntax or a non-numeric value."""
    if _is_number(value):
        return frozenset()
    if isinstance(value, Expression):
        return value.names
    if isinstance(value, str):
        return compile_expression(value).names
    raise ExpressionError(f'expected a number or expression, got {type(value).__name__}')


def is_identifier(name):
    """True for a non-keyword Python identifier that is NFKC-normalised (the parser rewrites other spellings,
    e.g. the micro sign, so such a name could never be referenced in an expression)."""
    return (isinstance(name, str) and name.isidentifier() and not keyword.iskeyword(name)
            and unicodedata.normalize('NFKC', name) == name)


def evaluate_parameters(parameter_specs, errors=None):
    """(name, expression) pairs or ParameterSpecs in any order -> {name: float} in input order, evaluated in
    dependency order. errors=dict -> partial mode: failures stored as {name: message}, never raises."""
    order, raw_values, dependencies, failures = [], {}, {}, {}
    dependent_failures = set()   # names failing only because a dependency failed
    for item in parameter_specs:
        name, raw = item if isinstance(item, (tuple, list)) else (item.name, item.expression)
        if not isinstance(name, str):   # malformed spec (e.g. a JSON list): keyed by its repr, fails as invalid
            name = repr(name)
        if name in raw_values or name in failures:
            failures[name] = 'defined more than once'
            raw_values.pop(name, None)
            dependencies.pop(name, None)
            continue
        order.append(name)
        if not is_identifier(name):
            failures[name] = f'{name!r} is not a valid name'
        elif is_reserved_name(name):
            failures[name] = f"'{name}' is a reserved name"
        else:
            try:
                dependencies[name] = referenced_names(raw)
                raw_values[name] = raw
            except ExpressionError as exc:
                failures[name] = str(exc)

    for name, names_used in dependencies.items():
        unknown = sorted(used for used in names_used if used not in dependencies and used not in failures)
        if unknown:
            failures[name] = _unknown_names_message(unknown, dependencies)

    # Kahn order: a parameter is evaluated (or fails with its first failed dependency) once all its pending
    # dependencies are done; names left waiting sit on or behind a cycle (below)
    values = {}
    pending = [name for name in order if name in dependencies and name not in failures]
    pending_set = set(pending)
    waiting = {name: {used for used in dependencies[name] if used in pending_set} for name in pending}
    dependents = {}
    for name in pending:
        for used in waiting[name]:
            dependents.setdefault(used, []).append(name)
    ready = collections.deque(name for name in pending if not waiting[name])
    while ready:
        name = ready.popleft()
        failed = sorted(used for used in dependencies[name] if used in failures)
        if failed:
            failures[name] = f"depends on '{failed[0]}', which has an error"
            dependent_failures.add(name)
        else:
            try:
                values[name] = evaluate_scalar(raw_values[name], values)
            except ExpressionError as exc:
                failures[name] = str(exc)
        for dependent in dependents.get(name, ()):
            waiting[dependent].discard(name)
            if not waiting[dependent]:
                ready.append(dependent)

    unresolved = {name for name in pending if name not in values and name not in failures}
    for name in sorted(unresolved):
        cycle = _dependency_cycle(name, dependencies, unresolved)
        if cycle[0] == name:
            failures[name] = f"circular dependency: {' -> '.join(cycle)}"
        else:   # reaches a cycle it is not part of
            failures[name] = f"depends on circular parameters {' -> '.join(cycle)}"
            dependent_failures.add(name)
    if errors is not None:
        errors.update(failures)
    else:
        failed_names = [name for name in order if name in failures]
        root_causes = [name for name in failed_names if name not in dependent_failures]
        if failed_names:
            name = (root_causes or failed_names)[0]
            raise ExpressionError(f"parameter '{name}': {failures[name]}")
    return {name: values[name] for name in order if name in values}


def _dependency_cycle(start, dependencies, unresolved):
    """Cycle [a, b, ..., a] reached from `start` by following its smallest unresolved dependency."""
    path = [start]
    while True:
        next_name = min(used for used in dependencies[path[-1]] if used in unresolved)
        if next_name in path:
            return path[path.index(next_name):] + [next_name]
        path.append(next_name)


def rename_in_expression(text, old, new):
    """Rename free name `old` to `new` (AST-aware, function names untouched) by splicing the source text, so the
    user's formatting is kept; text unchanged if parsing fails."""
    if not isinstance(text, str) or old == new:
        return text
    stripped = text.strip()
    try:
        tree = ast.parse(stripped, mode='eval')
    except (SyntaxError, ValueError, TypeError, RecursionError, MemoryError):
        return text
    called = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    name_spans = sorted(((node.lineno, node.col_offset, node.end_col_offset) for node in ast.walk(tree)
                         if isinstance(node, ast.Name) and node.id == old and id(node) not in called), reverse=True)
    if not name_spans:
        return text
    source_bytes = bytearray(stripped.encode('utf-8'))   # ast column offsets are UTF-8 byte offsets
    line_starts = [0]
    for line in bytes(source_bytes).splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))
    for lineno, start_col, end_col in name_spans:   # from the end backwards: earlier offsets stay valid
        line_start = line_starts[lineno - 1]
        source_bytes[line_start + start_col:line_start + end_col] = new.encode('utf-8')
    leading = text[:len(text) - len(text.lstrip())]
    trailing = text[len(text.rstrip()):]
    return leading + source_bytes.decode('utf-8') + trailing
