"""Small helpers shared by the selftests*.py modules (numpy and model.spec at import): the sub-check list and case
records (SubChecks, worst errors), relative errors, Results reads (solver roles, probe phasors), JSON lists,
single-wavelength values, uniform-mesh cell centres / edges and lines, float32-exact material constants, the exact pcg
solver node, the glass half-space block, the analytic symmetric-slab index and the numba vs numpy host-twin comparison
of the *_numba cases."""
import math

import numpy as np

from .spec import GeometrySpec, MaterialSpec, SolverDefaultsSpec, SolverSpec


# ---- case records ----
def checks_result(checks, **details):
    """Case record of {name: {'passed': bool, ...}}: measured = error = number of failed sub-checks, tolerance 0."""
    failed = [name for name, entry in checks.items() if not entry['passed']]
    return {'measured': float(len(failed)), 'expected': 0.0, 'error': float(len(failed)), 'tolerance': 0.0,
            'details': {'failed': failed, 'checks': dict(checks), **details}}


class SubChecks(dict):
    """Named pass / fail sub-checks of one case ({name: {'passed': bool, ...}}); `result(**details)` is the case
    record (checks_result). Pass the bound `add` to helper functions that fill it."""

    def add(self, name, passed, value=None, **details):
        """Record `name` as passed / failed with an optional JSON-able value and further detail entries."""
        self[name] = {'passed': bool(passed), 'value': value, **details}

    def add_error(self, name, error, tolerance, **details):
        """Pass when the error is finite and <= tolerance (NaN / inf fail)."""
        error = float(error)
        self[name] = {'passed': bool(math.isfinite(error) and error <= tolerance), 'error': error,
                      'tolerance': tolerance, **details}

    def add_bound(self, name, value, bound, at_least=False, **details):
        """Pass when the finite value is <= bound (>= bound with at_least)."""
        value = float(value)
        within = value >= bound if at_least else value <= bound
        self[name] = {'passed': bool(math.isfinite(value) and within), 'value': value,
                      ('lower_bound' if at_least else 'upper_bound'): bound, **details}

    def add_relative(self, name, value, expected, tolerance, **details):
        """add_error of relative_error of the real values (float() of each, so 'nan' strings read too), both recorded."""
        value, expected = float(value), float(expected)
        self.add_error(name, relative_error(value, expected), tolerance, value=value, expected=expected, **details)

    def add_raises(self, name, function, exceptions=ValueError):
        """Pass when function() raises one of `exceptions` (its message recorded)."""
        try:
            function()
        except exceptions as exc:
            self[name] = {'passed': True, 'message': str(exc)}
            return
        self[name] = {'passed': False, 'message': 'nothing raised'}

    @property
    def failed(self):
        """Names of the failed sub-checks, in insertion order."""
        return [name for name, entry in self.items() if not entry['passed']]

    def result(self, **details):
        return checks_result(self, **details)


def worst_error(errors):
    """Largest value of {item: error} (NaN propagates: a non-finite error fails the case; {} gives 0)."""
    return float(np.max(np.asarray(list(errors.values()), dtype=np.float64))) if errors else 0.0


def worst_item(errors):
    """(item, error) of the largest value of {item: error}, NaN counting as the largest."""
    item = max(errors, key=lambda name: math.inf if math.isnan(errors[name]) else errors[name])
    return item, errors[item]


def worst_result(errors, tolerance, **details):
    """Case record of the largest of {item: error} (NaN counts as the worst) against one tolerance."""
    item, error = worst_item(errors)
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': tolerance,
            'details': {'worst_item': item, 'errors': errors, **details}}


def scaled_worst_result(errors, tolerances, **details):
    """Case record of the worst error / own tolerance of {item: error} (NaN the worst): measured = error, tolerance 1."""
    item, scaled = worst_item({name: error / tolerances[name] for name, error in errors.items()})
    return {'measured': scaled, 'expected': 0.0, 'error': scaled, 'tolerance': 1.0,
            'details': {'worst_item': item, 'errors': errors, 'tolerances': tolerances, **details}}


def skip_result(reason):
    """A passing record of a case that compared nothing (details 'skipped': reason)."""
    return {'measured': 0.0, 'expected': 0.0, 'error': 0.0, 'tolerance': 1.0, 'details': {'skipped': reason}}


def skip_without_cudss():
    """skip_result when nvmath-python (cuDSS) is missing, else None."""
    from ..solvers.cudss_factor import cudss_available
    return None if cudss_available() else skip_result('nvmath-python (cuDSS) is not installed')


def no_runs():
    """specs() of a case that runs no model (check gets {})."""
    return {}


def cases_named(cases, *names):
    """Tuple of the SelfTestCases of `cases` with these names, in the given order (KeyError for an unknown name)."""
    by_name = {case.name: case for case in cases}
    return tuple(by_name[name] for name in names)


# ---- errors ----
def relative_error(measured, expected):
    """|measured - expected| / |expected| of two scalars (real or complex); the absolute difference at expected 0,
    NaN propagates."""
    difference = abs(measured - expected)
    return float(difference / abs(expected)) if expected != 0.0 else float(difference)


def max_relative_error(values, expected, scale=None):
    """max |values - expected| / scale (broadcast), scale None: max |expected|; a zero scale gives 0 for equal values,
    else inf (NaN propagates)."""
    values, expected = np.asarray(values), np.asarray(expected)
    difference = float(np.max(np.abs(values - expected)))
    scale = float(np.max(np.abs(expected))) if scale is None else float(scale)
    if scale == 0.0:
        return difference if difference == 0.0 or math.isnan(difference) else math.inf
    return difference / scale


def norm_relative_error(values, expected):
    """||values - expected||_2 / ||expected||_2 over all entries (inf for a zero reference and a nonzero difference)."""
    difference = float(np.linalg.norm(np.ravel(np.asarray(values) - np.asarray(expected))))
    scale = float(np.linalg.norm(np.ravel(expected)))
    if scale == 0.0:
        return difference if difference == 0.0 or math.isnan(difference) else math.inf
    return difference / scale


# ---- Results reads (only the calls a check may use) ----
def solver_role(results, physics, role='linear'):
    """The summary 'solvers' record {type, backend, ...} of one role of a physics ({} when missing)."""
    record = (results.summary().get('solvers') or {}).get(physics) or {}
    return (record.get('roles') or {}).get(role) or {}


def probe_phasor(results, physics, feature):
    """Complex phasor abs e^{j phase} of a field probe result of one solved wavelength / frequency."""
    series = results.result(physics, feature)['series']
    phase = single_value(series['phase'], f'{feature} phase')
    return single_value(series['abs'], f'{feature} abs') * complex(math.cos(phase), math.sin(phase))


def last_value(results, physics, feature):
    """Last entry of a feature result's 'values' as a float (NaN when empty)."""
    values = np.asarray(results.result(physics, feature)['values'], dtype=np.float64)
    return float(values[-1]) if values.size else math.nan


# ---- specs ----
def cpu_defaults():
    """SolverDefaultsSpec running on the cpu backend."""
    return SolverDefaultsSpec(backend='cpu')


def lithium_niobate_entry(euler_deg, plane='full'):
    """Crystal material entry of library lithium niobate rotated by the Euler angles [deg]."""
    return {'model': 'crystal', 'crystal': 'lithium_niobate', 'euler_deg': list(euler_deg), 'plane': plane}

PERIODIC_Y = {'bc_y_lo': 'periodic', 'bc_y_hi': 'periodic'}   # read-only: spread or copy it into settings
EXACT_CG_NAME = 'exact_cg'
EXACT_CG_TOLERANCE = 1e-12   # double-precision CG error ~1e-12 x the condition number << the case tolerances
# physics types whose 'linear' role is pcg (repeated as literals: selftests import no physics definitions)
_CG_PHYSICS_TYPES = frozenset({'heat', 'electric_currents', 'electrostatics', 'magnetostatics', 'magnetostatics_3d',
                               'solid_mechanics', 'solid_mechanics_3d'})


def with_exact_cg(spec, physics_names=None, tolerance=EXACT_CG_TOLERANCE, preconditioner=None):
    """spec with a double-precision pcg node 'exact_cg' at `tolerance` (preconditioner None: the node default jacobi;
    'multigrid' for large scalar diffusion grids) as the 'linear' solver of the named physics (None: every pcg-type
    physics of the spec; none: no node, an unused one is a warning); modified in place and returned."""
    names = [physics.name for physics in spec.physics if physics.type in _CG_PHYSICS_TYPES] \
        if physics_names is None else list(physics_names)
    if not names:
        return spec
    settings = {'precision': 'double', 'tolerance': tolerance}
    if preconditioner is not None:
        settings['preconditioner'] = preconditioner
    spec.solvers.append(SolverSpec(EXACT_CG_NAME, 'pcg', settings))
    for name in names:
        spec.physics_by_name(name).solvers['linear'] = EXACT_CG_NAME
    return spec


def float_list(values):
    """JSON-able list of Python floats."""
    return [float(value) for value in values]


def single_value(values, what):
    """The one float of a result array of a single solved wavelength; ValueError otherwise."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (1,):
        raise ValueError(f'{what}: expected one solved wavelength, got shape {values.shape}')
    return float(values[0])


def cell_centre(index, cell_size):
    """Coordinate [m] of the centre of window cell `index` (int or array) on a uniform mesh starting at 0."""
    return (index + 0.5) * cell_size


def cell_edges(centres, low_edge):
    """Cell edges [m] rebuilt from float64 centres (edge k+1 = 2 centre k - edge k), starting at low_edge."""
    edges = np.empty(centres.size + 1)
    edges[0] = low_edge
    for index, centre in enumerate(centres):
        edges[index + 1] = 2.0 * centre - edges[index]
    return edges


def vertical_line(name, x, y_start, y_end):
    """GeometrySpec line at x [m] from y_start to y_end."""
    return GeometrySpec(name, 'line', {'x0': x, 'y0': y_start, 'x1': x, 'y1': y_end})


def float32_value(value):
    """value rounded to float32 as a Python float: a material constant the float32 MaterialGrid stores exactly, so a
    float64 reference sees what the solver reads (else ~1e-8 relative misfits in exact cases)."""
    return float(np.float32(value))


def glass_half_space(x_start, size_x, size_y, eps_r):
    """(GeometrySpec 'half_space' covering x >= x_start over the full height, MaterialSpec 'glass' of eps_r on it)."""
    return (GeometrySpec('half_space', 'rectangle', {'x0': x_start, 'y0': 0.0, 'x1': size_x, 'y1': size_y}),
            MaterialSpec('glass', properties={'eps_r': eps_r}, selection=['half_space']))


def analytic_slab_index(k0, thickness, core_eps, cladding_eps, polarization, odd=False):
    """Real n_eff of the lossless symmetric slab: bisection on x = kappa d/2 of x tan x (even) or -x cot x (odd) =
    r gamma d/2, r = 1 (ez) or core_eps / cladding_eps (hz)."""
    half_thickness_k0 = 0.5 * thickness * k0
    v_number = half_thickness_k0 * math.sqrt(core_eps - cladding_eps)
    ratio = 1.0 if polarization == 'ez' else core_eps / cladding_eps
    lo, hi = (0.5 * math.pi, min(math.pi, v_number)) if odd else (0.0, min(0.5 * math.pi, v_number))
    for _ in range(100):
        x = 0.5 * (lo + hi)
        y = math.sqrt(v_number ** 2 - x ** 2)
        if (-x / math.tan(x) if odd else x * math.tan(x)) - ratio * y < 0.0:
            lo = x
        else:
            hi = x
    x = 0.5 * (lo + hi)
    return math.sqrt(core_eps - (x / half_thickness_k0) ** 2)


NUMBA_MISSING_DETAILS = {'numba': 'not installed (or OPEN_SOL_CPU_KERNELS=numpy): nothing compared'}


def _copied(arguments):
    """The kernel arguments with every numpy array copied (scalars kept)."""
    return [value.copy() if isinstance(value, np.ndarray) else value for value in arguments]


def host_twin_difference(numpy_twin, numba_twin, arguments) -> float:
    """Worst max |numba - numpy| / max |numpy| over the float arrays of `arguments` after one call of each host twin
    on its own copy (the in-place outputs; untouched inputs give 0); inf for a non-finite difference."""
    numpy_arguments, numba_arguments = _copied(arguments), _copied(arguments)
    numpy_twin(*numpy_arguments)
    numba_twin(*numba_arguments)
    worst = 0.0
    for reference, other in zip(numpy_arguments, numba_arguments):
        if not (isinstance(reference, np.ndarray) and reference.dtype.kind == 'f' and reference.size):
            continue
        reference = reference.astype(np.float64)
        difference = float(np.max(np.abs(np.asarray(other, dtype=np.float64) - reference)))
        peak = float(np.max(np.abs(reference)))
        if not (math.isfinite(difference) and math.isfinite(peak)):
            return math.inf
        worst = max(worst, difference / peak if peak > 0.0 else (0.0 if difference == 0.0 else math.inf))
    return worst


def host_twin_milliseconds(twin, arguments, n_calls) -> float:
    """Wall time per call [ms] of n_calls calls of a host twin on one copy of `arguments` (its outputs keep evolving)."""
    import time
    copies = _copied(arguments)
    start = time.perf_counter()
    for _ in range(int(n_calls)):
        twin(*copies)
    return 1e3 * (time.perf_counter() - start) / max(int(n_calls), 1)


def numba_twin_result(comparisons, timings, tolerance, numba_module):
    """Self-test result of numba vs numpy host twins: comparisons {label: (numpy, numba, arguments)} give error = the
    worst host_twin_difference; timings {label: (numpy, numba, arguments, n_numpy, n_numba)} ms per call of both
    (numba warmed up by one call) and the speed-up in details."""
    differences = {label: host_twin_difference(*entry) for label, entry in comparisons.items()}
    speeds = {}
    for label, (numpy_twin, numba_twin, arguments, n_numpy, n_numba) in timings.items():
        host_twin_milliseconds(numba_twin, arguments, 1)
        numpy_ms = host_twin_milliseconds(numpy_twin, arguments, n_numpy)
        numba_ms = host_twin_milliseconds(numba_twin, arguments, n_numba)
        speeds[label] = {'numpy_ms': numpy_ms, 'numba_ms': numba_ms,
                         'numba_speedup': numpy_ms / numba_ms if numba_ms > 0.0 else None}
    worst = max(differences.values()) if differences else 0.0
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': tolerance,
            'details': {'relative_differences': differences, 'ms_per_call': speeds,
                        'numba_threads': int(numba_module.get_num_threads())}}
