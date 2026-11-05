"""Self-tests of fem.line_fem (CPU, specs() empty, check({})): P1 / P2 / Hermite identities and exact solutions,
manufactured complex convergence rates, string modes and the clamped tensioned beam against its exact roots."""
import math

import numpy as np

from ..model.schema import SelfTestCase
from ..model.selftest_helpers import checks_result

_RTOL = 1e-10              # exact identities in float64 (first estimates)
_FAMILIES = (('lagrange', 1), ('lagrange', 2), ('hermite', 3))
_LINE_START, _LINE_END = 1e-6, 3e-6      # offset line [m] of the identity checks
_MANUFACTURED_LENGTH = 2e-6               # [m]
_MANUFACTURED_KAPPA = (2.0 + 3.0j) / _MANUFACTURED_LENGTH    # u = exp(kappa x)
_MANUFACTURED_TENSION = 1.0 + 0.4j        # a(x) = a0 (1 + x / L)
_MANUFACTURED_LOSS = (3.0 - 2.0j) / _MANUFACTURED_LENGTH ** 2
_MANUFACTURED_MESHES = {1: (20, 40), 2: (10, 20), 3: (10, 20)}   # polynomial degree -> element counts
_RATE_MARGIN = 0.15        # observed L2 rate >= degree + 1 - margin
# H-E1 variant A nanostring (hirsch2025 plan section 5): Si3N4, L 100 um, w = h = 100 nm, sigma 1 GPa
_BEAM_LENGTH = 100e-6
_BEAM_WIDTH = _BEAM_THICKNESS = 100e-9
_BEAM_YOUNG = 250e9
_BEAM_DENSITY = 3200.0
_BEAM_STRESS = 1e9
_BEAM_ELEMENTS = 4000      # 25 nm elements: the clamp boundary layer sqrt(EI / T) = 456 nm is resolved
_BEAM_PLAN_TARGETS_MHZ = (2.82113, 5.64400, 8.47036, 11.30194, 14.14048)
_MODE_COUNT = 5
_MODE_RTOL = 1e-6          # frequencies against the exact values (plan S3)
_PLAN_TARGET_RTOL = 1e-5   # exact roots against the plan's 6-digit table (rounding ~2e-6)
_STRING_TENSION = _BEAM_STRESS * _BEAM_THICKNESS        # N/m (per unit width)
_STRING_DENSITY = _BEAM_DENSITY * _BEAM_THICKNESS       # kg/m^2
_STRING_LOSS_FACTOR = 1e-3


def _graded_edges(start, end, count):
    """count + 1 smoothly graded edges on [start, end] (element sizes vary by about +-30 %)."""
    s = np.linspace(0.0, 1.0, count + 1)
    return start + (end - start) * (s + 0.3 * s * (1.0 - s))


def _adder(checks):
    def add(name, value, tolerance=_RTOL, at_least=False):
        value = float(value)
        passed = value >= tolerance if at_least else value <= tolerance
        checks[name] = {'value': value, 'tolerance': tolerance, 'passed': bool(passed)}
    return add


def _check_identities(_):
    from .line_fem import (assemble_line_bending, assemble_line_load, assemble_line_mass, assemble_line_matrix,
                           assemble_line_stiffness, assemble_point_load, assemble_point_matrix, element_means,
                           end_dofs, evaluate_line, interpolate_line, line_dofs, nodes_at, solve_line_system)
    checks = {}
    add = _adder(checks)
    rng = np.random.default_rng(7)
    start, end = _LINE_START, _LINE_END
    length = end - start
    edges = _graded_edges(start, end, 7)
    samples = np.sort(rng.uniform(start, end, 25))
    for family, order in _FAMILIES:
        tag = 'p1' if order == 1 else ('p2' if order == 2 else 'hermite')
        dofs = line_dofs(edges, family, order)
        one = interpolate_line(dofs, lambda x: np.ones_like(x), lambda x: np.zeros_like(x))
        linear = interpolate_line(dofs, lambda x: x, lambda x: np.ones_like(x))

        def complex_tension(x, _elements):
            return (1.0 + 0.3j) * (1.0 + x / length)
        stiffness = assemble_line_stiffness(dofs, complex_tension)
        scale = float(np.abs(stiffness.data).max())
        add(f'unconjugated_symmetry_{tag}', float(np.abs((stiffness - stiffness.T).data).max(initial=0.0)) / scale)
        add(f'constant_null_space_{tag}', float(np.abs(stiffness @ one).max()) / scale)
        mass = assemble_line_mass(dofs)
        add(f'mass_total_{tag}', abs(one @ mass @ one / length - 1.0))
        add(f'mass_moment_{tag}', abs(one @ mass @ linear / (0.5 * (end ** 2 - start ** 2)) - 1.0))
        add(f'linear_energy_{tag}', abs(linear @ assemble_line_stiffness(dofs) @ linear / length - 1.0))
        load = assemble_line_load(dofs, 2.5)
        add(f'line_load_{tag}', abs(load @ linear / (2.5 * 0.5 * (end ** 2 - start ** 2)) - 1.0))
        advection = assemble_line_matrix(dofs, 1.0, 0, 1)
        add(f'advection_{tag}', abs(one @ advection @ linear / length - 1.0))
        points = np.array([start + 0.3 * length, start + 0.71 * length])
        forces = np.array([1.5, -0.25])
        add(f'point_load_{tag}', abs(assemble_point_load(dofs, points, forces) @ linear - forces @ points)
            / float(np.abs(forces @ points)))
        add(f'point_matrix_{tag}', abs(linear @ assemble_point_matrix(dofs, points, forces) @ linear
                                       - forces @ points ** 2) / float(np.abs(forces @ points ** 2)))

        def polynomial(x):   # degree = order, on the reduced coordinate
            s = (x - start) / length
            return 1.0 - 2.0 * s + 3.0 * s ** 2 * (order >= 2) + 4.0 * s ** 3 * (order >= 3)

        def slope(x):
            s = (x - start) / length
            return (-2.0 + 6.0 * s * (order >= 2) + 12.0 * s ** 2 * (order >= 3)) / length

        def antiderivative(x):
            s = (x - start) / length
            return length * (s - s ** 2 + s ** 3 * (order >= 2) + s ** 4 * (order >= 3))
        coefficients = interpolate_line(dofs, polynomial, slope)
        add(f'interpolation_value_{tag}', np.abs(evaluate_line(dofs, coefficients, samples) - polynomial(samples)).max())
        add(f'interpolation_slope_{tag}', np.abs(evaluate_line(dofs, coefficients, samples, 1) - slope(samples)).max()
            * length)
        means = element_means(dofs, coefficients)
        exact_means = (antiderivative(edges[1:]) - antiderivative(edges[:-1])) / np.diff(edges)
        add(f'element_means_{tag}', np.abs(means - exact_means).max())
        add(f'outside_nan_{tag}', 0.0 if np.isnan(evaluate_line(dofs, coefficients, [end + 0.01 * length])[0]) else 1.0)
        # band solve of a complex non-symmetric operator against the dense solve
        operator = (stiffness + assemble_line_mass(dofs, (2.0 - 1.0j) / length ** 2)
                    + assemble_line_matrix(dofs, (0.5 + 0.2j) / length, 0, 1))
        fixed = np.concatenate([end_dofs(dofs, 'x_lo'), end_dofs(dofs, 'x_hi')])
        rhs = rng.standard_normal(dofs.n_dofs) + 1j * rng.standard_normal(dofs.n_dofs)
        solution = solve_line_system(operator, rhs, fixed, (1.0 + 2.0j, -0.5j))
        free = np.setdiff1d(np.arange(dofs.n_dofs), fixed)
        dense = operator.toarray()
        held = np.zeros(dofs.n_dofs, dtype=complex)
        held[fixed] = (1.0 + 2.0j, -0.5j)
        expected = held.copy()
        expected[free] = np.linalg.solve(dense[np.ix_(free, free)], (rhs - dense @ held)[free])
        add(f'band_solve_{tag}', np.abs(solution - expected).max() / np.abs(expected).max(), 1e-9)
        if family == 'lagrange':
            # string under a point force at an interior node: piecewise linear w = F x (L - a) / (T L), exact
            tension, force = 3.0, 2e-6
            node = 3
            position = edges[node]
            a = position - start
            string = assemble_line_stiffness(dofs, tension)
            held_ends = np.concatenate([end_dofs(dofs, 'x_lo'), end_dofs(dofs, 'x_hi')])
            deflection = solve_line_system(string, assemble_point_load(dofs, [position], [force]), held_ends)
            local = samples - start
            exact = np.where(local <= a, force * local * (length - a), force * a * (length - local)) / (tension * length)
            add(f'string_point_load_{tag}', np.abs(evaluate_line(dofs, deflection, samples) - exact).max()
                / (force * a * (length - a) / (tension * length)))
            add(f'nodes_at_{tag}', float(nodes_at(dofs, [position])[0] != node))
        else:
            add('bending_quadratic_hermite', abs(interpolate_line(dofs, lambda x: x ** 2, lambda x: 2 * x)
                                                 @ assemble_line_bending(dofs)
                                                 @ interpolate_line(dofs, lambda x: x ** 2, lambda x: 2 * x)
                                                 / (4.0 * length) - 1.0))
            # cantilever clamped at x_lo under a tip force P: w = P s^2 (3 L - s) / (6 EI), exact cubic
            rigidity, force = 2e-18, 1e-9
            beam = assemble_line_bending(dofs, rigidity)
            clamp = end_dofs(dofs, 'x_lo', value=True, slope=True)
            deflection = solve_line_system(beam, assemble_point_load(dofs, [end], [force]), clamp)
            local = samples - start
            exact = force * local ** 2 * (3.0 * length - local) / (6.0 * rigidity)
            tip = force * length ** 3 / (3.0 * rigidity)
            add('cantilever_deflection_hermite', np.abs(evaluate_line(dofs, deflection, samples) - exact).max() / tip)
            add('cantilever_tip_slope_hermite', abs(evaluate_line(dofs, deflection, [end], 1)[0]
                                                    / (force * length ** 2 / (2.0 * rigidity)) - 1.0))
            moment = assemble_point_load(dofs, [end], [force * length], test_derivative=1)
            rotated = solve_line_system(beam, moment, clamp)   # tip moment M: w = M s^2 / (2 EI)
            add('cantilever_moment_hermite', np.abs(evaluate_line(dofs, rotated, samples)
                                                    - force * length * local ** 2 / (2.0 * rigidity)).max() / tip)
    return checks_result(checks, elements=len(edges) - 1)


def _manufactured_error(family, order, count):
    """Relative L2 error of the complex manufactured solution u = exp(kappa x) of -(a u')' + b u = f on count
    graded elements, Dirichlet ends (value dofs only)."""
    from .line_fem import (assemble_line_load, assemble_line_mass, assemble_line_stiffness, end_dofs, evaluate_line,
                           line_dofs, solve_line_system)
    from .quadrature import gauss_line
    length, kappa = _MANUFACTURED_LENGTH, _MANUFACTURED_KAPPA

    def tension(x, _elements=None):
        return _MANUFACTURED_TENSION * (1.0 + x / length)

    def exact(x):
        return np.exp(kappa * x)

    def source(x, _elements=None):
        return (-_MANUFACTURED_TENSION * kappa * np.exp(kappa * x) * (1.0 / length + (1.0 + x / length) * kappa)
                + _MANUFACTURED_LOSS * np.exp(kappa * x))
    edges = _graded_edges(0.0, length, count)
    dofs = line_dofs(edges, family, order)
    operator = assemble_line_stiffness(dofs, tension) + assemble_line_mass(dofs, _MANUFACTURED_LOSS)
    load = assemble_line_load(dofs, source, degree=2 * order + 8)
    fixed = np.concatenate([end_dofs(dofs, 'x_lo'), end_dofs(dofs, 'x_hi')])
    solution = solve_line_system(operator, load, fixed, (exact(0.0), exact(length)))
    t, weights = gauss_line(2 * order + 10)
    widths = np.diff(edges)
    x = edges[:-1, None] + widths[:, None] * t[None, :]
    weighted = weights[None, :] * widths[:, None]
    error = evaluate_line(dofs, solution, x.ravel()).reshape(x.shape) - exact(x)
    return math.sqrt(float((weighted * np.abs(error) ** 2).sum()) / float((weighted * np.abs(exact(x)) ** 2).sum()))


def _check_manufactured(_):
    checks = {}
    add = _adder(checks)
    errors = {}
    for family, order in _FAMILIES:
        tag = 'p1' if order == 1 else ('p2' if order == 2 else 'hermite')
        coarse_count, fine_count = _MANUFACTURED_MESHES[order]
        coarse = _manufactured_error(family, order, coarse_count)
        fine = _manufactured_error(family, order, fine_count)
        errors[tag] = (coarse, fine)
        rate = math.log(coarse / fine) / math.log(fine_count / coarse_count)
        add(f'l2_rate_{tag}', rate, tolerance=order + 1 - _RATE_MARGIN, at_least=True)
    return checks_result(checks, l2_errors={tag: list(pair) for tag, pair in errors.items()})


def _clamped_free_dofs(dofs, slope):
    from .line_fem import end_dofs
    keep = np.ones(dofs.n_dofs, dtype=bool)
    for face in ('x_lo', 'x_hi'):
        keep[end_dofs(dofs, face, value=True, slope=slope)] = False
    return keep


def _check_string_modes(_):
    from .line_fem import assemble_line_mass, assemble_line_stiffness, line_dofs, line_eigenpairs
    checks = {}
    add = _adder(checks)
    length, tension, density = _BEAM_LENGTH, _STRING_TENSION, _STRING_DENSITY
    numbers = np.arange(1, _MODE_COUNT + 1)
    speed = math.sqrt(tension / density)
    exact = numbers * speed / (2.0 * length)
    details = {'exact_hz': exact.tolist()}
    # P1 on a uniform mesh: the exact discrete spectrum (6 / h^2) (1 - cos kh) / (2 + cos kh) T / rho (dense path)
    count = 100
    dofs = line_dofs(np.linspace(0.0, length, count + 1), 'lagrange', 1)
    eigenvalues, _ = line_eigenpairs(assemble_line_stiffness(dofs, tension), assemble_line_mass(dofs, density),
                                     _clamped_free_dofs(dofs, False), _MODE_COUNT)
    h = length / count
    phase = numbers * math.pi * h / length
    discrete = 6.0 / h ** 2 * (1.0 - np.cos(phase)) / (2.0 + np.cos(phase)) * tension / density
    add('p1_discrete_spectrum', np.abs(np.sort(eigenvalues.real) / discrete - 1.0).max())
    for family, order in (('lagrange', 2), ('hermite', 3)):
        tag = 'p2' if order == 2 else 'hermite'
        dofs = line_dofs(_graded_edges(0.0, length, 200), family, order)
        stiffness = assemble_line_stiffness(dofs, tension)
        mass = assemble_line_mass(dofs, density)
        keep = _clamped_free_dofs(dofs, False)
        eigenvalues, _ = line_eigenpairs(stiffness, mass, keep, _MODE_COUNT)
        frequencies = np.sqrt(np.sort(eigenvalues.real)) / (2.0 * math.pi)
        details[f'{tag}_hz'] = frequencies.tolist()
        add(f'string_frequencies_{tag}', np.abs(frequencies / exact - 1.0).max(), _MODE_RTOL)
        if order == 2:   # complex tension T (1 + i eta): lambda scales exactly by (1 + i eta) (ARPACK eigs path)
            lossy, _ = line_eigenpairs(assemble_line_stiffness(dofs, tension * (1.0 + 1j * _STRING_LOSS_FACTOR)),
                                       mass, keep, 3)
            lossy = lossy[np.argsort(lossy.real)]
            add('complex_tension_scaling_p2', np.abs(lossy / np.sort(eigenvalues.real)[:3]
                                                     - (1.0 + 1j * _STRING_LOSS_FACTOR)).max(), 1e-9)
    return checks_result(checks, **details)


def _beam_root_hz(number, rigidity, tension, line_density, length):
    """Exact f_n [Hz] of the clamped-clamped tensioned beam EI w'''' - T w'' = rho A omega^2 w (hirsch2025 plan
    section 4 reference algorithm): symmetric (odd n) / antisymmetric (even n) tanh forms, bracketed at (n +- 1/2)
    x the string f_1."""
    import scipy.optimize
    string_first = math.sqrt(tension / line_density) / (2.0 * length)
    half = 0.5 * length

    def mismatch(frequency):
        omega_squared = (2.0 * math.pi * frequency) ** 2
        root = math.sqrt(tension ** 2 + 4.0 * rigidity * line_density * omega_squared)
        k1 = math.sqrt((tension + root) / (2.0 * rigidity))
        k2 = math.sqrt((root - tension) / (2.0 * rigidity))
        if number % 2:
            return k2 * math.sin(k2 * half) + k1 * math.cos(k2 * half) * math.tanh(k1 * half)
        return k1 * math.sin(k2 * half) - k2 * math.cos(k2 * half) * math.tanh(k1 * half)
    return scipy.optimize.brentq(mismatch, (number - 0.5) * string_first, (number + 0.5) * string_first,
                                 xtol=1e-9, rtol=1e-15, maxiter=200)


def _check_tensioned_beam(_):
    from .line_fem import assemble_line_bending, assemble_line_mass, assemble_line_stiffness, line_dofs, line_eigenpairs
    checks = {}
    add = _adder(checks)
    area = _BEAM_WIDTH * _BEAM_THICKNESS
    rigidity = _BEAM_YOUNG * _BEAM_WIDTH * _BEAM_THICKNESS ** 3 / 12.0
    tension = _BEAM_STRESS * area
    line_density = _BEAM_DENSITY * area
    exact = np.array([_beam_root_hz(n, rigidity, tension, line_density, _BEAM_LENGTH)
                      for n in range(1, _MODE_COUNT + 1)])
    add('exact_roots_vs_plan_table', np.abs(exact / (np.array(_BEAM_PLAN_TARGETS_MHZ) * 1e6) - 1.0).max(),
        _PLAN_TARGET_RTOL)
    dofs = line_dofs(np.linspace(0.0, _BEAM_LENGTH, _BEAM_ELEMENTS + 1), 'hermite', 3)
    stiffness = assemble_line_bending(dofs, rigidity) + assemble_line_stiffness(dofs, tension)
    mass = assemble_line_mass(dofs, line_density)
    eigenvalues, _ = line_eigenpairs(stiffness, mass, _clamped_free_dofs(dofs, True), _MODE_COUNT)
    frequencies = np.sqrt(np.sort(eigenvalues.real)) / (2.0 * math.pi)
    add('hermite_clamped_frequencies', np.abs(frequencies / exact - 1.0).max(), _MODE_RTOL)
    boundary_layer = math.sqrt(rigidity / tension)
    return checks_result(checks, exact_hz=exact.tolist(), hermite_hz=frequencies.tolist(),
                         boundary_layer_m=boundary_layer, element_m=_BEAM_LENGTH / _BEAM_ELEMENTS)


LINE_FEM_SELFTEST_CASES = (
    SelfTestCase(
        name='fem_line_identities',
        label='FEM line: P1 / P2 / Hermite identities',
        description='CPU checks of fem.line_fem on a graded 7-element line offset from 0, P1, P2 and Hermite cubic: '
                    'complex stiffness unconjugated-symmetric with the constant in its null space; mass total and '
                    'first moment, linear energy, line load, first-derivative (advection) form and point load / '
                    'point matrix on linear fields; interpolation, slopes and element means of a degree-p polynomial '
                    'exact; NaN off the line; the band solve of a complex non-symmetric operator with complex held '
                    'ends against the dense solve; P1 / P2 string under a point force at a node (exact piecewise '
                    'linear); Hermite bending energy of x^2 and the clamped cantilever under a tip force / tip moment '
                    '(exact cubic / quadratic). 1e-10, the band solve 1e-9 (first estimates, not yet run). error = '
                    'number of failed '
                    'sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_identities,
    ),
    SelfTestCase(
        name='fem_line_manufactured',
        label='FEM line: complex manufactured convergence',
        description='-(a u\')\' + b u = f on a graded line with complex a = a0 (1 + x / L) (callable coefficient), '
                    'complex b, u = exp(kappa x), kappa L = 2 + 3i, complex Dirichlet ends (Hermite: value dofs, '
                    'slopes natural): the observed L2 rate between two meshes must be >= p + 1 - 0.15 (P1 2, P2 3, '
                    'Hermite 4; first estimates, not yet run). error = number of failed sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_manufactured,
    ),
    SelfTestCase(
        name='fem_line_string_modes',
        label='FEM line: string modes',
        description='Fixed-fixed string T = 100 N/m, rho_s = 3.2e-4 kg/m^2, L = 100 um, first five modes: P1 on 100 '
                    'uniform elements equals the discrete spectrum 6 / h^2 (1 - cos kh) / (2 + cos kh) T / rho_s '
                    '(1e-10, dense eigen path); P2 and Hermite (value dofs held, slopes free) on 200 graded elements '
                    'against f_n = n c / 2 L (1e-6, ARPACK shift-invert); P2 with complex tension T (1 + i eta), '
                    'eta = 1e-3: eigenvalues scale exactly by 1 + i eta (1e-9). error = number of failed sub-checks, '
                    'tolerance 0.',
        specs=lambda: {},
        check=_check_string_modes,
    ),
    SelfTestCase(
        name='fem_line_tensioned_beam',
        label='FEM line: clamped tensioned beam (Hermite)',
        description='H-E1 variant A nanostring as an Euler-Bernoulli beam (Si3N4 E 250 GPa, rho 3200, sigma 1 GPa, '
                    'L 100 um, w = h = 100 nm): Hermite cubic on 4000 uniform elements (25 nm, boundary layer 456 nm), '
                    'w = w\' = 0 at both ends, f_1..f_5 of EI w\'\'\'\' - T w\'\' = rho A omega^2 w against the exact '
                    'roots of the symmetric / antisymmetric tanh equations (hirsch2025 plan reference algorithm) to '
                    '1e-6; the roots against the plan table 2.82113 ... 14.14048 MHz to 1e-5 (6-digit rounding). '
                    'error = number of failed sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_tensioned_beam,
    ),
)
