"""Measurement of the measure-first cuDSS switches of open_sol.solvers.cudss_factor against today's nvmath path, on
real CudssFactorization objects: K3 REFACTORIZATION_PHASE (device-value refreshes on a kept plan), K7 RAW_SOLVE
(solves), K10 REORDERING_ALGORITHM 'AMD' on 2D operators (plan / fill / factorise / solve), plus K5 (a new factor
object adopting a parked plan vs planning on a kept ordering) and the K9 library-load stamps of this process.
Measurement only: the module constants are restored after each run. Run from any folder (gpu, nvmath):
python tools_profilers/time_profilers/cudss_phase_profile.py [--repeat N]
Results: printed table + out/cudss_phase_profile/results.json next to this script."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
OUT_DIR = Path(__file__).resolve().parent / 'out' / 'cudss_phase_profile'
LOSS = 0.02            # imaginary part of the complex Helmholtz shift (FDFD-like, complex symmetric LDL^T)
SHIFT_CELLS = 0.6      # k h of the Helmholtz shift
CONVECTION = 0.4       # upwind convection of the unsymmetric 2D operator (drift-diffusion-like LU)
REFRESH_STEP = 1e-3    # relative change of the shift per refresh (a sweep / Newton step: same pattern, new values)


def _line(cells):
    """1D second difference (Dirichlet) of unit spacing."""
    ones = np.ones(cells)
    return scipy.sparse.diags([-ones[:-1], 2.0 * ones, -ones[:-1]], [-1, 0, 1])


def _canonical(matrix):
    """Canonical CSR with int32 host index arrays (the pattern every refresh passes by identity)."""
    csr = matrix.tocsr()
    csr.sum_duplicates()
    csr.sort_indices()
    return csr.data, csr.indptr.astype(np.int32), csr.indices.astype(np.int32)


def cases(scale):
    """[(label, kind options, values(step) -> host values, indptr, indices)] of the measured operators."""
    found = []
    for cells in (128 * scale, 256 * scale):   # 2D complex symmetric Helmholtz (em_fdfd-like)
        laplacian = scipy.sparse.kron(_line(cells), scipy.sparse.identity(cells)) + scipy.sparse.kron(
            scipy.sparse.identity(cells), _line(cells))
        identity = scipy.sparse.identity(cells * cells)
        _, indptr, indices = _canonical(laplacian + identity)

        def values(step, laplacian=laplacian, identity=identity):
            shift = SHIFT_CELLS ** 2 * (1.0 + REFRESH_STEP * step) * (1.0 + 1j * LOSS)
            return _canonical(laplacian - shift * identity)[0]
        found.append((f'2D Helmholtz {cells}^2 (complex LDL^T)', {'real': False, 'symmetric': True}, values,
                      indptr, indices))
    cells = 128 * scale   # 2D real unsymmetric (LU)
    laplacian = scipy.sparse.kron(_line(cells), scipy.sparse.identity(cells)) + scipy.sparse.kron(
        scipy.sparse.identity(cells), _line(cells))
    upwind = scipy.sparse.kron(scipy.sparse.diags([-np.ones(cells - 1), np.ones(cells)], [-1, 0]),
                               scipy.sparse.identity(cells))
    _, indptr, indices = _canonical(laplacian + upwind)

    def general_values(step):
        return _canonical(laplacian + CONVECTION * (1.0 + REFRESH_STEP * step) * upwind)[0]
    found.append((f'2D convection {cells}^2 (real LU)', {'real': True}, general_values, indptr, indices))
    cells = 32 * scale   # 3D real SPD 7-point (Cholesky)
    line, identity = _line(cells), scipy.sparse.identity(cells)
    laplacian = (scipy.sparse.kron(scipy.sparse.kron(line, identity), identity)
                 + scipy.sparse.kron(scipy.sparse.kron(identity, line), identity)
                 + scipy.sparse.kron(scipy.sparse.kron(identity, identity), line))
    _, indptr, indices = _canonical(laplacian)

    def spd_values(step):
        return _canonical(laplacian * (1.0 + REFRESH_STEP * step))[0]
    found.append((f'3D Laplacian {cells}^3 (real Cholesky)', {'real': True, 'positive_definite': True}, spd_values,
                  indptr, indices))
    return found


def _sync():
    import cupy as cp
    cp.cuda.Device().synchronize()


def _relative_residual(values, indptr, indices, solution, rhs):
    """||b - A x|| / ||b|| on the host."""
    n = indptr.size - 1
    matrix = scipy.sparse.csr_matrix((values, indices, indptr), shape=(n, n))
    return float(np.linalg.norm(rhs - matrix @ solution) / np.linalg.norm(rhs))


def refresh_run(case, precision, refactorization, refreshes):
    """K3: plan once, then `refreshes` device-value refreshes (defer_check, each checked by one solve)."""
    import cupy as cp
    from open_sol.solvers import cudss_factor
    label, options, values, indptr, indices = case
    cudss_factor.REFACTORIZATION_PHASE = refactorization
    try:
        factor = cudss_factor.CudssFactorization.from_device_values(cp.asarray(values(0)), indptr, indices,
                                                                    precision=precision, **options)
        rhs = np.random.default_rng(1).standard_normal(indptr.size - 1)
        refresh_seconds, factorize_seconds, worst = [], [], 0.0
        for step in range(1, refreshes + 1):
            device_values = cp.asarray(values(step))
            _sync()
            started = time.perf_counter()
            factor.refactorize_device_values(device_values, indptr, indices, defer_check=True)
            _sync()
            refresh_seconds.append(time.perf_counter() - started)
            factorize_seconds.append(factor.factorize_seconds)
            solution = factor.solve_host(rhs)
            worst = max(worst, _relative_residual(values(step), indptr, indices, solution, rhs))
        record = {'case': label, 'precision': precision, 'refactorization_phase': refactorization,
                  'refresh_ms': 1e3 * float(np.median(refresh_seconds)),
                  'factorize_ms': 1e3 * float(np.median(factorize_seconds)), 'worst_residual': worst,
                  'raw_failed': sorted(factor._raw_failed), 'refinement_steps': factor.refinement_steps}
        factor.release()
        return record
    finally:
        cudss_factor.REFACTORIZATION_PHASE = False


def solve_run(case, precision, raw, solves):
    """K7: `solves` solves of one factor (timed together, one sync at the end)."""
    import cupy as cp
    from open_sol.solvers import cudss_factor
    label, options, values, indptr, indices = case
    cudss_factor.RAW_SOLVE = raw
    try:
        factor = cudss_factor.CudssFactorization.from_device_values(cp.asarray(values(0)), indptr, indices,
                                                                    precision=precision, **options)
        rhs_host = np.random.default_rng(2).standard_normal(indptr.size - 1)
        rhs = cp.asarray(rhs_host)
        factor.solve(rhs)   # warm-up (kernels, buffers)
        _sync()
        started = time.perf_counter()
        for _ in range(solves):
            solution = factor.solve(rhs)
        _sync()
        seconds = time.perf_counter() - started
        residual = _relative_residual(values(0), indptr, indices, cp.asnumpy(solution), rhs_host)
        record = {'case': label, 'precision': precision, 'raw_solve': raw, 'solve_ms': 1e3 * seconds / solves,
                  'residual': residual, 'raw_failed': sorted(factor._raw_failed)}
        factor.release()
        return record
    finally:
        cudss_factor.RAW_SOLVE = False


def ordering_run(case, algorithm):
    """K10: one fresh plan with REORDERING_ALGORITHM (None = cuDSS's default), its fill, factorise and solve."""
    import cupy as cp
    from open_sol.solvers import cudss_factor
    label, options, values, indptr, indices = case
    cudss_factor.REORDERING_ALGORITHM = algorithm
    try:
        factor = cudss_factor.CudssFactorization.from_device_values(cp.asarray(values(0)), indptr, indices,
                                                                    **options)
        rhs_host = np.random.default_rng(3).standard_normal(indptr.size - 1)
        _sync()
        started = time.perf_counter()
        solution = factor.solve(cp.asarray(rhs_host))
        _sync()
        record = {'case': label, 'reordering': algorithm or 'default', 'plan_ms': 1e3 * factor.plan_seconds,
                  'factorize_ms': 1e3 * factor.factorize_seconds, 'solve_ms': 1e3 * (time.perf_counter() - started),
                  'lu_entries': factor.lu_entries, 'permanent_bytes': factor.permanent_bytes,
                  'residual': _relative_residual(values(0), indptr, indices, cp.asnumpy(solution), rhs_host)}
        factor.release()
        return record
    finally:
        cudss_factor.REORDERING_ALGORITHM = None


def parked_run(case):
    """K5: inside cudss_orderings_kept, a second factor object of the pattern on a kept ordering (first alive) vs a
    third adopting the parked plan (both released before)."""
    import cupy as cp
    from open_sol.solvers import cudss_factor
    label, options, values, indptr, indices = case
    rhs_host = np.random.default_rng(4).standard_normal(indptr.size - 1)
    with cudss_factor.cudss_orderings_kept():
        first = cudss_factor.CudssFactorization.from_device_values(cp.asarray(values(0)), indptr, indices, **options)
        second = cudss_factor.CudssFactorization.from_device_values(cp.asarray(values(1)), indptr, indices, **options)
        kept_ordering = {'plan_ms': 1e3 * second.plan_seconds, 'reused_ordering': second.last_ordering_reused}
        first.release()
        second.release()
        third = cudss_factor.CudssFactorization.from_device_values(cp.asarray(values(2)), indptr, indices, **options)
        residual = _relative_residual(values(2), indptr, indices, third.solve_host(rhs_host), rhs_host)
        record = {'case': label, 'kept_ordering_plan_ms': kept_ordering['plan_ms'],
                  'kept_ordering_reused': kept_ordering['reused_ordering'], 'adopted_plan_ms': 1e3 * third.plan_seconds,
                  'adopted': third.last_plan_adopted, 'adopted_residual': residual,
                  'lu_entries': (first.lu_entries, third.lu_entries)}
        third.release()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--repeat', type=int, default=20, help='refreshes / solves per measurement')
    parser.add_argument('--scale', type=int, default=1, help='grid scale (1: 128^2 / 256^2 / 32^3)')
    arguments = parser.parse_args()
    from open_sol.solvers import cudss_factor
    cudss_factor.warm_cudss_library()
    measured = cases(arguments.scale)   # built while the library loads (K9 stamps below)
    first_label, first_options, first_values, first_indptr, first_indices = measured[0]
    import cupy as cp
    cold = cudss_factor.CudssFactorization.from_device_values(cp.asarray(first_values(0)), first_indptr, first_indices,
                                                              **first_options)   # cold start: not reported
    cold.release()
    timing = cudss_factor.cudss_library_timing()
    results = {'library_timing': {'load_seconds': (None if timing['loaded'] is None or timing['started'] is None
                                                   else timing['loaded'] - timing['started']),
                                  'join_wait_seconds': timing['join_wait']},
               'refresh': [], 'solve': [], 'ordering': [], 'parked': []}
    for case in measured:
        precisions = ('double', 'mixed') if not case[1].get('positive_definite') else ('double',)
        for precision in precisions:
            for refactorization in (False, True):
                results['refresh'].append(refresh_run(case, precision, refactorization, arguments.repeat))
                print(results['refresh'][-1], flush=True)
            for raw in (False, True):
                results['solve'].append(solve_run(case, precision, raw, arguments.repeat))
                print(results['solve'][-1], flush=True)
        if case[0].startswith('2D'):
            for algorithm in (None, 'AMD'):
                results['ordering'].append(ordering_run(case, algorithm))
                print(results['ordering'][-1], flush=True)
        results['parked'].append(parked_run(case))
        print(results['parked'][-1], flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'results.json').write_text(json.dumps(results, indent=1))
    print(f'written {OUT_DIR / "results.json"}')


if __name__ == '__main__':
    main()
