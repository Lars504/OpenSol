"""Self-tests of fem.reduced_harmonic (CPU, specs() empty, check({})): a synthetic lossy P1 string pencil (complex
tension, mass damping, loads in omega^0, omega^1 and omega^-1): adaptive model vs direct solves, the sketched residual
estimate, modal bases with every mode (exact) and truncated (residual and verified error consistent)."""
import math

import numpy as np

from ..model.schema import SelfTestCase
from ..model.selftest_helpers import checks_result
from .selftests_harmonic import _checker

# unit string (T = rho = L = 1: c = 1 m/s, f_n ~ n / 2 Hz), clamped ends, P1 on interior nodes
_SMALL_UNKNOWNS = 200        # exact residual (below reduced_harmonic.EXACT_SKETCH_LIMIT)
_LARGE_UNKNOWNS = 3000       # sketched residual
_LOSS = 0.02                 # complex tension (1 + j eta): Q ~ 50
_MASS_DAMPING = 0.05         # C = beta M [1/s]
_BAND_HZ = (1.2, 4.3)        # modes 3 .. 8
_FREQUENCIES = 2000
_LOAD_POINTS = (0.3, 0.7)    # x [m] of the point loads (omega^0 and 10 / omega)
_TRAIN_TOLERANCE = 1e-9      # a decade above DEFLATION_TOLERANCE 1e-10: the residual floors near 1.5e-10 here
_ADAPTIVE_ERROR = 1e-7       # relative to the direct solution at every frequency (first estimate)
_MODAL_ERROR = 1e-9          # every mode in the basis
_TRUNCATED_MODES = 6
_RESIDUAL_MATCH = 1e-8       # exact sketch: estimate = true relative residual
_SKETCH_RATIO = 1.5          # sketched residual within this factor of the true one (512 rows)
_SKETCH_SAMPLES = 50


def _band_omegas():
    return 2.0 * math.pi * np.linspace(_BAND_HZ[0], _BAND_HZ[1], _FREQUENCIES)


def _string_system(n_unknowns):
    """(terms, loads) of the lossy unit string with n_unknowns interior P1 nodes."""
    from scipy.sparse import diags

    from .reduced_harmonic import HarmonicLoad, HarmonicTerm, PowerCoefficient
    h = 1.0 / (n_unknowns + 1)
    ones = np.ones(n_unknowns)
    stiffness = diags([-ones[1:], 2.0 * ones, -ones[1:]], [-1, 0, 1]).tocsr() * ((1.0 + 1j * _LOSS) / h)
    mass = diags([ones[1:], 4.0 * ones, ones[1:]], [-1, 0, 1]).tocsr() * (h / 6.0)
    terms = [HarmonicTerm(stiffness, PowerCoefficient(1.0, 0)),
             HarmonicTerm(_MASS_DAMPING * mass, PowerCoefficient(1j, 1)),
             HarmonicTerm(mass, PowerCoefficient(-1.0, 2))]
    points = []
    for x in _LOAD_POINTS:
        vector = np.zeros(n_unknowns, dtype=np.complex128)
        vector[int(round(x / h)) - 1] = 1.0
        points.append(vector)
    loads = [HarmonicLoad(points[0], PowerCoefficient(1.0, 0)),
             HarmonicLoad((mass @ ones).astype(np.complex128), PowerCoefficient(1j, 1)),
             HarmonicLoad(points[1], PowerCoefficient(10.0, -1))]
    return terms, loads


class _Splu:
    def __init__(self, matrix):
        import scipy.sparse.linalg
        self._factor = scipy.sparse.linalg.splu(matrix.tocsc().astype(np.complex128))

    def solve_host(self, rhs):
        return self._factor.solve(np.asarray(rhs, dtype=np.complex128))


def _operator(terms, omega):
    matrix = terms[0].coefficient(omega) * terms[0].matrix
    for term in terms[1:]:
        matrix = matrix + term.coefficient(omega) * term.matrix
    return matrix


def _factorizer(terms):
    return lambda omega: _Splu(_operator(terms, omega))


def _load(loads, omega):
    return sum(load.value(omega) for load in loads)


def _direct(terms, loads, omegas):
    """Complex (m, n) direct solutions."""
    return np.array([_Splu(_operator(terms, omega)).solve_host(_load(loads, omega)) for omega in omegas])


def _reduced(model, omegas):
    """Complex (m, n) reduced solutions V y."""
    return (model.basis @ model.reduced_coordinates(omegas).T).T


def _relative_errors(approximate, exact):
    return np.linalg.norm(approximate - exact, axis=1) / np.linalg.norm(exact, axis=1)


def _true_residuals(terms, loads, omegas, solutions):
    """Float64 (m,) ||b - A x|| / ||b|| per frequency."""
    values = []
    for omega, solution in zip(omegas, solutions):
        load = _load(loads, omega)
        values.append(np.linalg.norm(load - _operator(terms, omega) @ solution) / np.linalg.norm(load))
    return np.array(values)


def _check_adaptive(_):
    from .reduced_harmonic import ReducedHarmonicModel
    checks, add = _checker()
    omegas = _band_omegas()
    terms, loads = _string_system(_SMALL_UNKNOWNS)
    model = ReducedHarmonicModel(terms, loads, _factorizer(terms))
    info = model.train(omegas, tolerance=_TRAIN_TOLERANCE, moments=4, max_points=40)
    exact = _direct(terms, loads, omegas)
    errors = _relative_errors(_reduced(model, omegas), exact)
    add('adaptive_vs_direct', errors.max(), _ADAPTIVE_ERROR)
    add('adaptive_converged', 0.0 if info['converged'] else 1.0, 0.5)
    add('adaptive_factorizations', info['factorizations'], 41)
    worst = int(np.argmin(np.abs(omegas - 2.0 * math.pi * info['worst_frequency'])))
    add('verified_error_matches', abs(info['verified_error'] - errors[worst]), 1e-9)
    # sketched residual on a long string: a partly trained model (residuals well above round-off)
    large_terms, large_loads = _string_system(_LARGE_UNKNOWNS)
    large = ReducedHarmonicModel(large_terms, large_loads, _factorizer(large_terms))
    large.train(omegas, tolerance=1e-30, moments=1, max_points=4, verify=False)
    sample = omegas[np.linspace(0, omegas.size - 1, _SKETCH_SAMPLES).astype(int)]
    estimated = large.residuals(sample)
    true = _true_residuals(large_terms, large_loads, sample, _reduced(large, sample))
    add('sketch_ratio_log', np.abs(np.log(estimated / true)).max(), math.log(_SKETCH_RATIO))
    return checks_result(checks, adaptive_info=info, sketch_residual_range=[float(true.min()), float(true.max())])


def _check_modal(_):
    import scipy.linalg

    from .reduced_harmonic import ReducedHarmonicModel
    checks, add = _checker()
    omegas = _band_omegas()
    terms, loads = _string_system(_SMALL_UNKNOWNS)
    exact = _direct(terms, loads, omegas)
    eigenvalues, vectors = scipy.linalg.eig(terms[0].matrix.toarray(), terms[2].matrix.toarray())
    full = ReducedHarmonicModel(terms, loads, _factorizer(terms))
    full_info = full.fit_modal(vectors, omegas, tolerance=1e-6)
    add('all_modes_vs_direct', _relative_errors(_reduced(full, omegas), exact).max(), _MODAL_ERROR)
    add('all_modes_residual', full_info['max_residual'], _MODAL_ERROR)
    centre = 0.5 * (omegas.min() + omegas.max())
    nearest = np.argsort(np.abs(np.sqrt(eigenvalues.astype(np.complex128)) - centre))[:_TRUNCATED_MODES]
    truncated = ReducedHarmonicModel(terms, loads, _factorizer(terms))
    truncated_info = truncated.fit_modal(vectors[:, nearest], omegas, tolerance=1e-6)
    approximate = _reduced(truncated, omegas)
    errors = _relative_errors(approximate, exact)
    true = _true_residuals(terms, loads, omegas, approximate)
    estimated = truncated.residuals(omegas)
    add('truncated_is_approximate', errors.max(), 1e-6, at_least=True)
    add('truncated_residual_estimate', np.abs(estimated / true - 1.0).max(), _RESIDUAL_MATCH)
    worst = int(np.argmax(true))
    add('truncated_worst_frequency', abs(truncated_info['worst_frequency'] - omegas[worst] / (2.0 * math.pi)), 1e-9)
    add('truncated_verified_error', abs(truncated_info['verified_error'] / errors[worst] - 1.0), _RESIDUAL_MATCH)
    return checks_result(checks, all_modes_info=full_info, truncated_info=truncated_info,
                         truncated_max_error=float(errors.max()))


REDUCED_HARMONIC_SELFTEST_CASES = (
    SelfTestCase(
        name='fem_reduced_harmonic_adaptive',
        label='FEM reduced sweep: adaptive model and residual sketch',
        description=f'A lossy unit P1 string ({_SMALL_UNKNOWNS} unknowns, complex tension eta = {_LOSS:g}, mass damping, '
                    'loads in omega^0, j omega and 1 / omega) over 2000 frequencies holding six resonances: the '
                    f'adaptive reduced model (tolerance {_TRAIN_TOLERANCE:g}, 4 moments) against direct solves '
                    f'({_ADAPTIVE_ERROR:g} relative at every frequency), converged within 40 expansions, its verified '
                    f'error equal to the direct one; on a {_LARGE_UNKNOWNS}-unknown string the sketched residual '
                    f'estimate of a partly trained model within a factor {_SKETCH_RATIO:g} of the true residual. '
                    'error = number of failed sub-checks, tolerance 0 (first estimates).',
        specs=lambda: {},
        check=_check_adaptive,
    ),
    SelfTestCase(
        name='fem_reduced_harmonic_modal',
        label='FEM reduced sweep: modal bases (all modes, truncated)',
        description='The lossy string of fem_reduced_harmonic_adaptive: with every eigenvector of (K, M) plus the static '
                    f'correction the modal model equals the direct solves ({_MODAL_ERROR:g}); with the '
                    f'{_TRUNCATED_MODES} modes nearest the band centre the exact (identity sketch) residual estimate '
                    f'equals the true relative residual ({_RESIDUAL_MATCH:g}), the reported worst frequency is the true '
                    'one and its verified error equals the direct error there. error = number of failed sub-checks, '
                    'tolerance 0 (first estimates).',
        specs=lambda: {},
        check=_check_modal,
    ),
)
