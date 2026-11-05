"""Banded direct solver of 1D models (symmetry plan 2.20; SolverType 'banded', role 'linear'): the PCGSolver protocol
of core.fv_diffusion.DiffusionSystem on a window one cell across (unknowns in line order, so the operator is banded),
one host banded Cholesky (scipy.linalg) per assembly, exact solves. numpy at module level, scipy inside functions."""
import time

import numpy as np

from .sparse_direct import ExactSolveRecord, require_backend_operator

SYMMETRY_TOLERANCE = 1e-12   # max |A - A^T| / max |A| above which upper_bands refuses the operator (as cuDSS)


def upper_bands(matrix, max_bandwidth=None) -> np.ndarray:
    """(u + 1, n) float64 upper band storage (scipy.linalg cholesky_banded, lower=False) of a symmetric host scipy
    sparse matrix: ab[u + i - j, j] = a[i, j] for i <= j, u = the largest column offset of a nonzero entry. ValueError
    before any band allocation when u > max_bandwidth, or when the matrix is not symmetric (only the upper triangle
    would be read: an unsymmetric operator would be solved as its symmetrised version)."""
    csr = matrix.tocsr(copy=True)
    csr.sum_duplicates()   # duplicate entries (non-canonical CSR) sum
    coo = csr.tocoo()
    rows, cols, values = coo.row.astype(np.int64), coo.col.astype(np.int64), np.asarray(coo.data, dtype=np.float64)
    upper = (cols >= rows) & (values != 0.0)
    bandwidth = int((cols[upper] - rows[upper]).max()) if upper.any() else 0
    if max_bandwidth is not None and bandwidth > max_bandwidth:
        raise ValueError(f'upper_bands: bandwidth {bandwidth} > {max_bandwidth}')
    asymmetry = abs(csr - csr.T)
    largest = float(np.abs(values).max()) if values.size else 0.0
    if asymmetry.nnz and float(asymmetry.max()) > SYMMETRY_TOLERANCE * largest:
        raise ValueError(f'upper_bands: the operator is not symmetric (max |A - A^T| = {float(asymmetry.max()):.3g}, '
                         f'max |A| = {largest:.3g})')
    rows, cols, values = rows[upper], cols[upper], values[upper]
    n_columns = int(matrix.shape[0])
    flat = (bandwidth + rows - cols) * n_columns + cols   # entries unique after sum_duplicates: one bincount anyway
    return np.bincount(flat, weights=values, minlength=(bandwidth + 1) * n_columns).reshape(bandwidth + 1, n_columns)


class BandedSolver(ExactSolveRecord):
    """'banded' runtime: host banded Cholesky of the assembled symmetric positive definite operator per factorize,
    exact solves; gpu backend: one transfer of the CSR per factorisation and of rhs / solution per solve. Refuses an
    operator whose bandwidth exceeds max_bandwidth (a 2D window: use pcg or sparse_direct)."""

    def __init__(self, label=None, backend='gpu', max_bandwidth=8):
        if backend not in ('gpu', 'cpu'):
            raise ValueError(f"backend must be 'gpu' or 'cpu', got {backend!r}")
        self.backend = backend
        self.label = label if label is not None else (lambda: 'banded')
        self.max_bandwidth = int(max_bandwidth)
        self._init_exact_record()
        self.n_factorizations = 0
        self.total_factor_seconds = 0.0
        self.bandwidth = None
        self._factor = None
        self._host_operator = None

    def factorize(self, operator, preconditioner=None, n_unknowns=None) -> None:
        """Banded Cholesky of the assembled CSR (scipy on the cpu backend, cupyx on the gpu); preconditioner ignored;
        ValueError for a bandwidth above max_bandwidth or an operator that is not positive definite."""
        from scipy.linalg import LinAlgError, cholesky_banded
        require_backend_operator(self.label(), self.backend, operator)
        host_operator = operator if isinstance(operator.data, np.ndarray) else operator.get()
        start = time.perf_counter()
        try:
            bands = upper_bands(host_operator, self.max_bandwidth)
        except ValueError as exc:   # a bandwidth above max_bandwidth or an unsymmetric operator
            raise ValueError(f'{self.label()}: {exc}: the banded solver is for symmetric 1D operators (one cell '
                             f'across, bandwidth <= {self.max_bandwidth}); use a pcg or sparse_direct linear '
                             'solver') from exc
        try:
            self._factor = cholesky_banded(bands, lower=False)
        except LinAlgError as exc:
            raise ValueError(f'{self.label()}: the operator is not positive definite ({exc}); an unanchored or '
                             'non-conducting region?') from exc
        self.bandwidth = bands.shape[0] - 1
        self.total_factor_seconds += time.perf_counter() - start
        self.n_factorizations += 1
        self._host_operator = host_operator.tocsr() if self.compute_residual else None

    def solve(self, rhs, solution, reference=None) -> None:
        """solution[...] = A^-1 rhs (flat arrays of the backend); reference ignored (exact solve)."""
        from scipy.linalg import cho_solve_banded
        if self._factor is None:
            raise RuntimeError(f'{self.label()}: factorize before solve')
        host_rhs = self._host_rhs(rhs)
        host_solution = cho_solve_banded((self._factor, False), host_rhs)
        self._finish_host_solve(host_rhs, host_solution, solution, self._host_operator)

    def work_counters(self) -> dict:
        """Work so far (summary 'solvers' record of its (physics, role)); factor_seconds is host wall time."""
        return {'factorizations': self.n_factorizations, 'factor_seconds': self.total_factor_seconds,
                'solves': self.n_solves, 'bandwidth': self.bandwidth}


def time_estimate(values, work):
    """Run-time recipe of a banded role (core.time_estimate.SolverWork): host banded Cholesky (bandwidth 1-2 of a
    line) and solves, each with its transfers on the gpu; O(n) host work."""
    from ..core import time_estimate as te
    gpu = work.backend == 'gpu'
    factor = (te.HostStream(16.0 * work.nnz), te.PythonCalls(20)) + ((te.Transfer(12.0 * work.nnz),) if gpu else ())
    solve = (te.HostStream(48.0 * work.unknowns), te.PythonCalls(10)) + (
        (te.Transfer(16.0 * work.unknowns),) if gpu else ())
    return te.TimeEstimate((
        te.TimeItem('banded factorisations', 'factorisation', float(work.factorizations), work.factorizations_kind,
                    factor, per_step=work.factor_per_step, counter=work.counter('factorizations')),
        te.TimeItem('banded solves', 'solve', float(work.solves), work.solves_kind, solve, per_step=work.per_step,
                    counter=work.counter('solves'))))
