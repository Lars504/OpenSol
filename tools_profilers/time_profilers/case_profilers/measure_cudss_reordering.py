"""S6 measurement (speedup_plan.md): cuDSS plan / factorise / solve time, factor fill and peak memory per nvmath
DirectSolverReorderingAlg on matrices like open_sol's (3D P2 Helmholtz on Kuhn tetrahedra, 2D / 3D finite-difference
Helmholtz), with the DirectSolver options open_sol uses (blocking, single-threaded host planning, LDL^T on the lower
triangle for 'symmetric', LU for 'general'). Measurement only: open_sol's default ordering is not changed.
Run from any folder (gpu):
python measure_cudss_reordering.py [--sizes small|medium|large] [--repeat N]
Results: printed table + out/cudss_reordering/results.json next to this script"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse

REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY))
OUT_DIR = Path(__file__).resolve().parent / 'out' / 'cudss_reordering'

SIZES = {   # (P2 Kuhn cells per side, 3D finite-difference cells per side, 2D finite-difference cells per side)
    'small': ((8,), (24,), (128,)),
    'medium': ((10, 14), (32, 40), (256,)),
    'large': ((10, 14, 18), (32, 40, 50), (256, 512)),
}
WAVENUMBER_CELLS = 0.6   # k h of the Helmholtz shift (indefinite, like an acoustics / FDFD operator)


def tetrahedra_p2(cells):
    """Real symmetric indefinite K - k^2 M of P2 Lagrange on a Kuhn-split unit cube (open_sol fem.assembly3d)."""
    from open_sol.fem.assembly3d import assemble_lagrange_mass_3d, assemble_lagrange_stiffness_3d
    from open_sol.fem.dofs3d import lagrange_dofs_3d
    from open_sol.fem.mesh3d import structured_box_mesh
    edges = np.linspace(0.0, 1.0, cells + 1)
    dofs = lagrange_dofs_3d(structured_box_mesh(edges, edges, edges), 2)
    wavenumber = WAVENUMBER_CELLS * cells
    matrix = assemble_lagrange_stiffness_3d(dofs) - wavenumber ** 2 * assemble_lagrange_mass_3d(dofs)
    return f'tet P2 {cells}^3 cells', matrix.tocsr()


def finite_difference(cells, dimensions):
    """Real symmetric indefinite 5- / 7-point Laplacian - k^2 on a unit square / cube (Dirichlet sides)."""
    h = 1.0 / (cells + 1)
    line = scipy.sparse.diags([-np.ones(cells - 1), 2.0 * np.ones(cells), -np.ones(cells - 1)], [-1, 0, 1]) / h ** 2
    identity = scipy.sparse.identity(cells)
    if dimensions == 2:
        laplacian = scipy.sparse.kron(line, identity) + scipy.sparse.kron(identity, line)
    else:
        laplacian = (scipy.sparse.kron(scipy.sparse.kron(line, identity), identity)
                     + scipy.sparse.kron(scipy.sparse.kron(identity, line), identity)
                     + scipy.sparse.kron(scipy.sparse.kron(identity, identity), line))
    wavenumber = WAVENUMBER_CELLS / h
    matrix = (laplacian - wavenumber ** 2 * scipy.sparse.identity(laplacian.shape[0])).tocsr()
    return f'FD {dimensions}D {cells}^{dimensions}', matrix


def measure(matrix, kind, algorithm):
    """{plan_s, factorize_s, solve_s, lu_nnz, peak_bytes, residual} of one DirectSolver run (error text on refusal)."""
    import cupy as cp
    import cupyx.scipy.sparse
    from nvmath.sparse.advanced import (DirectSolver, DirectSolverMatrixType, DirectSolverMatrixViewType,
                                        DirectSolverReorderingAlg)
    full = cupyx.scipy.sparse.csr_matrix(matrix)
    stored = scipy.sparse.tril(matrix, format='csr') if kind == 'symmetric' else matrix
    stored.sort_indices()
    operator = cupyx.scipy.sparse.csr_matrix((cp.asarray(stored.data), cp.asarray(stored.indices.astype(np.int32)),
                                              cp.asarray(stored.indptr.astype(np.int32))), shape=stored.shape)
    rhs = cp.asarray(np.random.default_rng(0).standard_normal(matrix.shape[0]))
    options = {'blocking': True}
    if kind == 'symmetric':
        options.update(sparse_system_type=DirectSolverMatrixType.SYMMETRIC,
                       sparse_system_view=DirectSolverMatrixViewType.LOWER)
    solver = DirectSolver(operator, rhs, options=options)
    try:
        solver.plan_config.reordering_algorithm = getattr(DirectSolverReorderingAlg, algorithm)
        cp.cuda.Device().synchronize()
        started = time.perf_counter()
        solver.plan()
        cp.cuda.Device().synchronize()
        plan_seconds = time.perf_counter() - started
        peak = int(solver.plan_info.memory_estimates['peak_device_memory'])
        started = time.perf_counter()
        info = solver.factorize()
        cp.cuda.Device().synchronize()
        factor_seconds = time.perf_counter() - started
        started = time.perf_counter()
        solution = solver.solve()
        cp.cuda.Device().synchronize()
        solve_seconds = time.perf_counter() - started
        residual = float(cp.linalg.norm(full @ solution - rhs) / cp.linalg.norm(rhs))
        return {'plan_s': plan_seconds, 'factorize_s': factor_seconds, 'solve_s': solve_seconds,
                'lu_nnz': int(info.lu_nnz), 'peak_bytes': peak, 'residual': residual, 'status': int(info.info)}
    finally:
        solver.free()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--sizes', choices=sorted(SIZES), default='medium')
    parser.add_argument('--repeat', type=int, default=2, help='runs per (matrix, ordering); the best is reported')
    parser.add_argument('--algorithms', default='DEFAULT,NESTED_DISSECTION,AMD,COLAMD,BTF_COLAMD',
                        help='comma-separated DirectSolverReorderingAlg names')
    arguments = parser.parse_args()
    algorithms = [name.strip() for name in arguments.algorithms.split(',') if name.strip()]
    tet_cells, fd3_cells, fd2_cells = SIZES[arguments.sizes]
    matrices = ([tetrahedra_p2(cells) for cells in tet_cells] + [finite_difference(cells, 3) for cells in fd3_cells]
                + [finite_difference(cells, 2) for cells in fd2_cells])
    measure(matrices[0][1], 'symmetric', 'DEFAULT')   # cold start (library loads, first plan): not reported
    rows = []
    header = (f"{'matrix':<22} {'n':>8} {'nnz':>9} {'kind':<9} {'ordering':<18} {'plan s':>8} {'factor s':>9} "
              f"{'solve s':>8} {'L nnz / A nnz':>13} {'peak MiB':>9} {'residual':>9}")
    print(header)
    print('-' * len(header))
    for label, matrix in matrices:
        for kind in ('symmetric', 'general'):
            for algorithm in algorithms:
                best = None
                for _ in range(max(arguments.repeat, 1)):
                    try:
                        result = measure(matrix, kind, algorithm)
                    except Exception as error:   # refused ordering / kind: recorded, not fatal
                        result = {'error': f'{type(error).__name__}: {error}'[:160]}
                        break
                    if best is None or result['plan_s'] < best['plan_s']:
                        best = result
                result = best if best is not None else result
                row = {'matrix': label, 'n': int(matrix.shape[0]), 'nnz': int(matrix.nnz), 'kind': kind,
                       'ordering': algorithm, **result}
                rows.append(row)
                if 'error' in result:
                    print(f"{label:<22} {row['n']:>8} {row['nnz']:>9} {kind:<9} {algorithm:<18} {result['error']}")
                    continue
                print(f"{label:<22} {row['n']:>8} {row['nnz']:>9} {kind:<9} {algorithm:<18} {result['plan_s']:>8.3f} "
                      f"{result['factorize_s']:>9.3f} {result['solve_s']:>8.4f} {result['lu_nnz'] / matrix.nnz:>13.1f} "
                      f"{result['peak_bytes'] / 2 ** 20:>9.1f} {result['residual']:>9.1e}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'results.json').write_text(json.dumps(rows, indent=1))
    print(f'written {OUT_DIR / "results.json"}')


if __name__ == '__main__':
    main()
