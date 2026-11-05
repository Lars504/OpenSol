"""Self-tests of fem.resonance_search (CPU, specs() empty, check({})): two nearly equal P1 strings in one pencil (pairs
of resonances 1e-7 apart: one merge cluster, both kept) against their closed-form discrete frequencies: a counted band,
a lossy uncounted band and the union of two targets."""
import math

import numpy as np

from ..model.schema import SelfTestCase
from ..model.selftest_helpers import checks_result
from .selftests_harmonic import _checker

# unit strings (T = rho = L = 1 and T = 1 + _TENSION_SPLIT: f_k ~ k / 2 Hz), clamped ends, P1 on _NODES interior nodes;
# K = (T / h) tridiag(-1, 2, -1), M = (h / 6) tridiag(1, 4, 1): lambda_k = T (6 / h^2) (1 - cos t) / (2 + cos t), t = k
# pi / (_NODES + 1)
_NODES = 60
_TENSION_SPLIT = 1e-7        # relative: the pairs fall in one merge cluster (MERGE_TOLERANCE 1e-6) yet are distinct
_BAND_ORDERS = (3, 14)       # mode orders k inside the band (both strings: 2 x 12 resonances)
_PER_SHIFT = 4
_MAX_RESONANCES = 100
_LOSS = 1e-3                 # lossy band: K (1 + j eta), complex symmetric (no inertia count)
_TARGET_FACTORS = ((5, 1.01), (6, 0.99))   # targets: f_k x factor
_FREQUENCY_TOLERANCE = 1e-8  # relative, every resonance against the closed form (first estimate)


def _string_frequencies(tension):
    """Discrete Hz of every order k = 1 .. _NODES of a P1 string of the given tension."""
    h = 1.0 / (_NODES + 1)
    theta = np.arange(1, _NODES + 1) * math.pi / (_NODES + 1)
    eigenvalues = tension * 6.0 / h ** 2 * (1.0 - np.cos(theta)) / (2.0 + np.cos(theta))
    return np.sqrt(eigenvalues) / (2.0 * math.pi)


def _pencil(loss=0.0):
    """Block-diagonal (K, M) of the two strings (host CSR; complex with a loss)."""
    import scipy.sparse
    h = 1.0 / (_NODES + 1)
    ones = np.ones(_NODES)
    laplacian = scipy.sparse.diags([-ones[1:], 2.0 * ones, -ones[1:]], [-1, 0, 1]) / h
    mass = scipy.sparse.diags([ones[1:], 4.0 * ones, ones[1:]], [-1, 0, 1]) * (h / 6.0)
    stiffness = scipy.sparse.block_diag([laplacian, (1.0 + _TENSION_SPLIT) * laplacian]).tocsr()
    if loss:
        stiffness = (stiffness * (1.0 + 1j * loss)).tocsr()
    return stiffness, scipy.sparse.block_diag([mass, mass]).tocsr()


def _solver_pair(stiffness, mass, eigen_solver):
    """(solve_at, count_below) as the membrane builds them (count only for a real pencil)."""
    from .harmonic_resonances import solve_harmonic_resonances

    def solve_at(frequency, count):
        return solve_harmonic_resonances(stiffness, mass, frequency, count, eigen_solver=eigen_solver)

    def count_below(eigenvalue):
        inertia, _ = eigen_solver.count_below(stiffness, mass, eigenvalue)
        return None if inertia is None else inertia.n_below
    real = not np.iscomplexobj(stiffness.data)
    return solve_at, (count_below if real else None)


def _band_edges(frequencies):
    """(f_lo, f_hi) [Hz] at the middle of the gaps around the _BAND_ORDERS pairs."""
    first, last = _BAND_ORDERS
    low = 0.5 * (max(frequencies[:, first - 2]) + min(frequencies[:, first - 1]))
    high = 0.5 * (max(frequencies[:, last - 1]) + min(frequencies[:, last]))
    return low, high


def _frequency_error(measured, expected):
    """Largest relative deviation of sorted measured from sorted expected (inf on a count mismatch)."""
    measured, expected = np.sort(np.asarray(measured, float)), np.sort(np.asarray(expected, float))
    if measured.size != expected.size or not measured.size:
        return math.inf
    return float(np.max(np.abs(measured / expected - 1.0)))


def _check_search(_):
    from ..solvers.arpack_shift_invert import ArpackShiftInvertSolver
    from .resonance_search import solve_resonances_at_targets, solve_resonances_in_band
    checks, add = _checker()
    frequencies = np.stack([_string_frequencies(1.0), _string_frequencies(1.0 + _TENSION_SPLIT)])
    f_lo, f_hi = _band_edges(frequencies)
    first, last = _BAND_ORDERS
    expected_band = frequencies[:, first - 1:last].ravel()
    details = {'band_Hz': (f_lo, f_hi)}
    stiffness, mass = _pencil()
    solve_at, count_below = _solver_pair(stiffness, mass, ArpackShiftInvertSolver())
    band = solve_resonances_in_band(solve_at, f_lo, f_hi, _PER_SHIFT, _MAX_RESONANCES, count_below=count_below)
    measured = [resonance.frequency for resonance in band.resonances]
    add('band_frequencies', _frequency_error(measured, expected_band), _FREQUENCY_TOLERANCE)
    add('band_expected_count', abs((band.info['expected_count'] or 0) - expected_band.size), 0.0)
    add('band_frequency_order', float(any(np.diff(measured) < 0.0)), 0.0)
    details['band'] = {'measured_Hz': measured, 'shifts': band.info['shifts'], 'warnings': band.info['warnings']}
    lossy_stiffness, lossy_mass = _pencil(_LOSS)
    lossy_solve_at, lossy_count_below = _solver_pair(lossy_stiffness, lossy_mass, ArpackShiftInvertSolver())
    lossy = solve_resonances_in_band(lossy_solve_at, f_lo, f_hi, _PER_SHIFT, _MAX_RESONANCES,
                                     count_below=lossy_count_below)
    lossy_expected = expected_band * float(np.sqrt(1.0 + 1j * _LOSS).real)
    lossy_measured = [resonance.frequency for resonance in lossy.resonances]
    add('lossy_band_frequencies', _frequency_error(lossy_measured, lossy_expected), _FREQUENCY_TOLERANCE)
    add('lossy_band_uncounted', float(lossy.info['expected_count'] is not None), 0.0)
    details['lossy_band'] = {'measured_Hz': lossy_measured, 'shifts': lossy.info['shifts'],
                             'warnings': lossy.info['warnings']}
    targets = [factor * frequencies[0, order - 1] for order, factor in _TARGET_FACTORS]
    every = frequencies.ravel()
    kept = set()
    for target in targets:   # the _PER_SHIFT nearest of each target, union by identity
        kept.update(int(index) for index in np.argsort(np.abs(every - target))[:_PER_SHIFT])
    union = solve_resonances_at_targets(_solver_pair(stiffness, mass, ArpackShiftInvertSolver())[0], targets,
                                        _PER_SHIFT)
    union_measured = [resonance.frequency for resonance in union.resonances]
    add('targets_union', _frequency_error(union_measured, every[sorted(kept)]), _FREQUENCY_TOLERANCE)
    add('targets_shifts', abs(union.info['shifts'] - len(targets)), 0.0)
    details['targets'] = {'targets_Hz': targets, 'measured_Hz': union_measured,
                          'expected_Hz': sorted(float(every[index]) for index in kept)}
    return checks_result(checks, **details)


RESONANCE_SEARCH_SELFTEST_CASES = (
    SelfTestCase(
        name='fem_resonance_search',
        label='FEM resonance search: band and several targets',
        description=f'Two clamped unit P1 strings ({_NODES} interior nodes each, tensions 1 and 1 + {_TENSION_SPLIT:g}) '
                    'in one block-diagonal pencil: every order is a pair 1e-7 apart (one merge cluster, two independent '
                    'fields). fem.resonance_search on fem.harmonic_resonances shift solves (host SuperLU): the band '
                    f'around orders {_BAND_ORDERS[0]} .. {_BAND_ORDERS[1]} ({_PER_SHIFT} per shift) returns every '
                    f'closed-form discrete frequency once (relative {_FREQUENCY_TOLERANCE:g}) in frequency order, its '
                    'inertia count equals the pair count; the same band of the lossy pencil K (1 + j '
                    f'{_LOSS:g}) returns Re of the complex frequencies without a count; targets 1.01 f_5 and 0.99 f_6 '
                    f'return the union of the {_PER_SHIFT} nearest of each (shared modes once). error = number of '
                    'failed sub-checks, tolerance 0 (first estimates, not yet run).',
        specs=lambda: {},
        check=_check_search,
    ),
)
