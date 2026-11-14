"""'batched_direct' linear-solver runtime: many systems on one sparsity pattern (a frequency sweep's A_k = sum_t c_kt T_t
plus a small per-system block at fixed pattern slots) factorised and solved together, on the GPU as one explicitly
batched cuDSS DirectSolver per (pattern, batch size, precision): planned once, its values refreshed in place for every
later batch; each system equilibrated and refined against its own float64 operator, and a system missing the accept
tolerance is the consumer's to re-solve on the per-system path (BatchedDirectSolver is a SparseDirectSolver). CPU
backend: the same calls as a loop of host SuperLU factors (parity, not speed). numpy at module level; cupy / cupyx /
scipy / nvmath inside functions."""
# Layout (gpu, one BatchPattern per consumer pattern; n unknowns, z pattern entries, T terms, B planned systems):
#   terms (T, z) complex128 and the row / column of each entry (int32), uploaded once per pattern;
#   values (N, z) complex128 = coefficients (N, T) @ terms (one GEMM) + the per-system extra blocks at their slots;
#   R_k = |diag A_k|^-1/2 (1 where the diagonal is 0): E_k = R_k A_k R_k, its lower triangle for a (complex)
#   symmetric pencil (cuDSS LDL^T, as cudss_factor) cast to the factor dtype into the B planned CSR data arrays;
#   rhs buffers (B, n) of the factor dtype, one per planned system; residuals r = b - A x in float64 by one SpMM of
#   the row selector S (n x z, ones) with (values * x[:, columns])^T (N x z complex128, the transient peak).
# The first batch of a pattern plans (host reordering per system, symbolic phase): nvmath explicit batching, a list
# of operators and a list of rhs. A batch shorter than the plan fills its pad slots with copies of its last system and
# zero rhs (no re-plan). Refinement: x += R E^-1 R r from x = 0, every system in every step (one batched solve), the
# better iterate kept per system; stop when every system is below REFINEMENT_TOLERANCE or none improved by STALL_RATIO.
# One host read per step (the per-system residual norms). Errors of the batched API (an nvmath without explicit
# batching, a plan over DEVICE_FRACTION of the free memory) raise BatchUnavailable: the consumer warns once and goes on
# per system; a system with a zero pivot shows up as a non-finite / large residual and is rejected, not raised.
import math
import time

import numpy as np

from .sparse_direct import SparseDirectSolver

PRECISIONS = ('mixed', 'double')
FACTOR_DTYPES = {'mixed': np.complex64, 'double': np.complex128}
REFINEMENT_TOLERANCE = 1e-12   # refine while some system has ||r|| / ||b|| above this
ACCEPT_TOLERANCE = 1e-9        # a system above this is rejected (re-solved per system by the consumer)
MAX_REFINEMENTS = {'mixed': 8, 'double': 2}   # correction solves after the first
STALL_RATIO = 0.5              # stop once no open system lowers its residual below this x the previous one
MAX_BATCH = 32                 # systems per batch (bounds the plan's host reordering and the run estimate)
MEMORY_FRACTION = 0.5          # share of the free device memory one batch may claim by the a-priori sizing
DEVICE_FRACTION = 0.8          # largest share of the free device memory cuDSS's peak estimate of a plan may claim
UNBUDGETED_BATCH_BYTES = 1 << 30   # sizing without a device memory reading (cpu: RAM unknown)
# run-time recipe: refinement solves per accepted system (mixed: unrefined complex64 ~1e-5 relative, ~3e-4 per step
# with the double residual -> 3 steps to 1e-12; double: one solve and one check step); uncalibrated guesses
SOLVES_PER_SYSTEM_GUESS = {'mixed': 4.0, 'double': 2.0}


class BatchUnavailable(RuntimeError):
    """The batched path cannot run here (no explicit batching in nvmath, the plan does not fit): solve per system."""


def systems_bytes(n_unknowns, entries, n_terms, precision, symmetric=True) -> int:
    """Device bytes per batched system besides its cuDSS factor (pool arrays of solve_batch): combined and scaled
    complex128 values, the factor-dtype copy, the residual product transient, the rhs buffer and the float64
    solution / residual / candidate / correction vectors."""
    factor_bytes = np.dtype(FACTOR_DTYPES[precision]).itemsize
    stored = (int(entries) + int(n_unknowns)) // 2 if symmetric else int(entries)
    return int(3 * int(entries) * 16 + stored * factor_bytes + int(n_unknowns) * (factor_bytes + 6 * 16))


def pattern_bytes(n_unknowns, entries, n_terms, symmetric=True) -> int:
    """Device bytes of a BatchPattern (kept per consumer pattern): complex128 terms, int32 rows / columns, the row
    selector, diagonal positions, the lower-triangle take and its int32 CSR arrays."""
    n, z = int(n_unknowns), int(entries)
    lower = (z + n) // 2 if symmetric else 0
    return int(int(n_terms) * z * 16 + 2 * z * 4 + z * (16 + 4) + (n + 1) * 4 + n * 8 + lower * (8 + 4) + (n + 1) * 4)


class BatchPattern:
    """Device state of one consumer pattern: canonical CSR (n x n, int32), T term value rows on it, the diagonal
    positions, the lower-triangle take (symmetric) and the row selector of the float64 residual product."""

    def __init__(self, indptr, indices, terms, symmetric, backend):
        indptr = np.asarray(indptr, dtype=np.int64)
        indices = np.asarray(indices, dtype=np.int64)
        self.n_unknowns = int(indptr.size - 1)
        self.n_entries = int(indices.size)
        self.symmetric = bool(symmetric)
        self.backend = backend
        self.source = (indptr, indices)   # host canonical pattern (consumers match it by identity of their arrays)
        terms = np.stack([np.asarray(term, dtype=np.complex128).ravel() for term in terms])
        if terms.shape[1] != self.n_entries:
            raise ValueError(f'batched_direct: each term needs {self.n_entries} pattern values, got {terms.shape[1]}')
        self.n_terms = int(terms.shape[0])
        rows = np.repeat(np.arange(self.n_unknowns, dtype=np.int64), np.diff(indptr))
        diagonal = np.full(self.n_unknowns, -1, dtype=np.int64)
        on_diagonal = np.flatnonzero(rows == indices)
        diagonal[rows[on_diagonal]] = on_diagonal
        self.has_diagonal = diagonal >= 0
        lower = np.flatnonzero(indices <= rows) if self.symmetric else None
        if backend == 'cpu':
            self.xp = np
            self.terms = terms
            self.rows, self.columns = rows, indices
            self.diagonal = np.where(self.has_diagonal, diagonal, 0)
            self.indptr, self.indices = indptr.astype(np.int32), indices.astype(np.int32)
            return
        import cupy as cp
        from cupyx.scipy import sparse as cpsp
        self.xp = cp
        self.terms = cp.asarray(terms)
        self.rows = cp.asarray(rows.astype(np.int32))
        self.columns = cp.asarray(indices.astype(np.int32))
        self.diagonal = cp.asarray(np.where(self.has_diagonal, diagonal, 0))
        self.diagonal_mask = cp.asarray(self.has_diagonal)
        self.indptr = cp.asarray(indptr.astype(np.int32))
        self.indices = self.columns
        # residual row sums: S (n x z) with S[i, e] = 1 for the entries e of row i (one SpMM per refinement step)
        self.selector = cpsp.csr_matrix((cp.ones(self.n_entries, dtype=cp.complex128),
                                         cp.arange(self.n_entries, dtype=cp.int32), self.indptr),
                                        shape=(self.n_unknowns, self.n_entries))
        if self.symmetric:
            counts = np.bincount(rows[lower], minlength=self.n_unknowns)
            self.lower = cp.asarray(lower)
            self.lower_indptr = cp.asarray(np.concatenate(([0], np.cumsum(counts))).astype(np.int32))
            self.lower_indices = cp.asarray(indices[lower].astype(np.int32))
        else:
            self.lower = None
            self.lower_indptr, self.lower_indices = self.indptr, self.indices

    def matches(self, indptr, indices) -> bool:
        """True when the consumer's pattern arrays are the ones this was built from (identity: never modify them)."""
        return self.source[0] is indptr and self.source[1] is indices or (
            np.array_equal(self.source[0], indptr) and np.array_equal(self.source[1], indices))


class _BatchPlan:
    """A planned explicitly batched cuDSS DirectSolver: B cupyx CSR operators sharing the factor pattern (their data
    arrays refreshed in place), B rhs buffers, its booked outside-pool bytes."""

    def __init__(self, pattern, n_systems, dtype):
        import cupy as cp
        from cupyx.scipy import sparse as cpsp
        from .cudss_factor import batched_direct_solver
        self.pattern = pattern
        self.n_systems = int(n_systems)
        self.dtype = np.dtype(dtype)
        n = pattern.n_unknowns
        stored = int(pattern.lower_indices.size)
        self.data = cp.zeros((self.n_systems, stored), dtype=self.dtype)
        self.data[:, :] = 1.0   # a non-singular placeholder until the first refresh (never solved)
        self.operators = [cpsp.csr_matrix((self.data[k], pattern.lower_indices, pattern.lower_indptr), shape=(n, n))
                          for k in range(self.n_systems)]
        self.buffers = cp.zeros((self.n_systems, n), dtype=self.dtype)
        self.rhs = [self.buffers[k] for k in range(self.n_systems)]
        self.permanent_bytes = 0
        self.solver = None
        started = time.perf_counter()
        try:
            self.solver = batched_direct_solver(self.operators, self.rhs,
                                                'symmetric' if pattern.symmetric else 'general')
            self.solver.plan()
        except Exception as error:   # an nvmath without explicit batching, a refused plan: per system from now on
            self.free()
            raise BatchUnavailable(f'cuDSS explicit batching unavailable ({type(error).__name__}: {error})') from None
        self.plan_seconds = time.perf_counter() - started
        permanent, peak = _plan_estimates(self.solver)
        self.permanent_bytes, self.peak_bytes = permanent, peak
        from .cudss_factor import book_external_bytes, free_device_bytes
        book_external_bytes(permanent, peak)
        free = free_device_bytes()
        if peak > DEVICE_FRACTION * free:
            self.free()
            raise MemoryError(f'batched cuDSS plan of {self.n_systems} systems needs ~{peak / 2**20:.0f} MiB, over '
                              f'{DEVICE_FRACTION:.0%} of the free device memory ({free / 2**20:.0f} MiB)')

    def matches(self, pattern, n_systems, dtype) -> bool:
        return self.pattern is pattern and self.n_systems >= int(n_systems) and self.dtype == np.dtype(dtype)

    def factorize(self, values) -> None:
        """Refresh every planned operator with values (m <= B, stored) and factorise (pads: copies of the last)."""
        m = values.shape[0]
        self.data[:m] = values
        if m < self.n_systems:
            self.data[m:] = values[m - 1]
        try:
            self.solver.factorize()
        except Exception as error:
            raise BatchUnavailable(f'cuDSS batched factorisation failed ({type(error).__name__}: {error})') from None

    def solve(self, scaled_rhs):
        """(m, n) solutions of E_k y_k = scaled_rhs[k] (factor dtype) for the first m systems; pads solve zero rhs."""
        import cupy as cp
        m = scaled_rhs.shape[0]
        self.buffers[:m] = scaled_rhs
        if m < self.n_systems:
            self.buffers[m:] = 0
        try:
            result = self.solver.solve()
        except Exception as error:
            raise BatchUnavailable(f'cuDSS batched solve failed ({type(error).__name__}: {error})') from None
        parts = list(result) if isinstance(result, (list, tuple)) else [result[k] for k in range(self.n_systems)]
        return cp.stack([cp.asarray(part).reshape(-1) for part in parts[:m]])

    def free(self) -> None:
        if self.solver is not None:
            try:
                self.solver.free()
            except Exception:   # interpreter shutdown: the CUDA context may already be gone
                pass
            self.solver = None
        if self.permanent_bytes:
            from .cudss_factor import release_external_bytes
            release_external_bytes(self.permanent_bytes)
            self.permanent_bytes = 0
        self.operators = self.rhs = []
        self.data = self.buffers = None


def _plan_estimates(solver) -> tuple:
    """(permanent, peak) device bytes of a planned batch from nvmath's plan_info (a per-system array is summed; 0
    when this nvmath reports none)."""
    try:
        estimates = solver.plan_info.memory_estimates
        return (int(np.sum(np.asarray(estimates['permanent_device_memory']))),
                int(np.sum(np.asarray(estimates['peak_device_memory']))))
    except Exception:
        return 0, 0


class BatchedDirectSolver(SparseDirectSolver):
    """'batched_direct' runtime of a 'linear' role: solve_batch factorises and solves up to batch_capacity systems of
    one BatchPattern together; everything else (and the consumers' re-solves of rejected systems) is the inherited
    SparseDirectSolver on cuDSS (gpu: cudss_mixed / cudss by precision, SuperLU after a refusal) or SuperLU (cpu)."""

    def __init__(self, precision='mixed', batch_size=0, label=None, backend='gpu'):
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {', '.join(PRECISIONS)}, got {precision!r}")
        if int(batch_size) < 0:
            raise ValueError(f'batch_size must be >= 0 (0 = automatic), got {batch_size}')
        gpu = backend == 'gpu'
        super().__init__(back_end=('cudss_mixed' if precision == 'mixed' else 'cudss') if gpu else 'scipy_splu',
                         label=label, backend=backend, fallback_back_end='scipy_splu' if gpu else None)
        self.precision = precision
        self.factor_dtype = np.dtype(FACTOR_DTYPES[precision] if gpu else np.complex128)
        self.batch_size = int(batch_size)
        self.unavailable = None   # text of the BatchUnavailable that stopped batching for good, or None
        self._plan = None
        self.n_batches = 0
        self.n_batched_systems = 0
        self.n_batch_plans = 0
        self.n_batch_rejected = 0
        self.n_batch_refinements = 0
        self.total_batch_seconds = 0.0
        self.total_batch_plan_seconds = 0.0
        self.batch_permanent_bytes = 0
        self.batch_peak_bytes = 0
        self.worst_batch_residual = 0.0

    # ---- patterns and sizing ----
    def pattern(self, indptr, indices, terms, symmetric) -> BatchPattern:
        """BatchPattern of a canonical host CSR pattern (n x n) and its T term value arrays (each (z,)); symmetric: a
        complex symmetric pencil (factored as LDL^T of its lower triangle). Keep it per consumer pattern."""
        return BatchPattern(indptr, indices, terms, symmetric, self.backend)

    def batch_capacity(self, pattern, n_systems) -> int:
        """Systems of the next batch: at most n_systems, batch_size (0: MAX_BATCH) and what MEMORY_FRACTION of the free
        memory holds by the a-priori sizing (pool arrays + the cuDSS factor estimate per system)."""
        limit = max(1, min(int(n_systems), self.batch_size or MAX_BATCH))
        from .cudss_factor import factor_device_bytes
        mixed = self.factor_dtype == np.complex64
        permanent, extra = factor_device_bytes(pattern.n_unknowns, pattern.n_entries, 16, 2, mixed=mixed,
                                               symmetric=pattern.symmetric)
        per_system = systems_bytes(pattern.n_unknowns, pattern.n_entries, pattern.n_terms, self.precision,
                                   pattern.symmetric) + permanent + extra
        if self.backend == 'cpu':
            from ..core.host_memory import available_bytes
            available = available_bytes()
            free = UNBUDGETED_BATCH_BYTES if available is None else available
        else:
            from ..core import gpu_memory
            room = gpu_memory.pool_room_bytes(release_cached=True)
            free = UNBUDGETED_BATCH_BYTES if room is None else room
        return max(1, min(limit, int(MEMORY_FRACTION * free // max(per_system, 1))))

    def release(self) -> None:
        """Free the kept batched plan (its outside-pool factor bytes); counters stay."""
        if self._plan is not None:
            self._plan.free()
            self._plan = None

    # ---- the batched solve ----
    def solve_batch(self, pattern, coefficients, rhs, extras=None) -> tuple:
        """(solutions complex128 host (N, n), relative residuals (N,), accepted (N,) bool) of A_k x_k = rhs[k] with A_k
        = sum_t coefficients[k, t] T_t (+ extras[k] = (slots int64, complex values) added at pattern slots, or None)
        for N <= batch_capacity systems; rejected systems (non-finite or above ACCEPT_TOLERANCE) are the caller's to
        re-solve. BatchUnavailable when the batched path cannot run (then never again on this runtime)."""
        if self.unavailable is not None:
            raise BatchUnavailable(self.unavailable)
        started = time.perf_counter()
        coefficients = np.asarray(coefficients, dtype=np.complex128)
        rhs = np.asarray(rhs, dtype=np.complex128)
        n_systems = coefficients.shape[0]
        if coefficients.shape != (n_systems, pattern.n_terms) or rhs.shape != (n_systems, pattern.n_unknowns):
            raise ValueError(f'{self.label()}: solve_batch needs coefficients ({n_systems}, {pattern.n_terms}) and rhs '
                             f'({n_systems}, {pattern.n_unknowns}), got {coefficients.shape} and {rhs.shape}')
        try:
            if self.backend == 'cpu':
                solutions, relative = self._host_batch(pattern, coefficients, rhs, extras)
            else:
                solutions, relative = self._device_batch(pattern, coefficients, rhs, extras)
        except (BatchUnavailable, MemoryError) as error:
            self.unavailable = str(error)
            self.release()
            raise BatchUnavailable(self.unavailable) from None
        accepted = np.isfinite(relative) & (relative <= ACCEPT_TOLERANCE)
        self.n_batches += 1
        self.n_batched_systems += n_systems
        self.n_batch_rejected += int(np.count_nonzero(~accepted))
        if accepted.any():
            self.worst_batch_residual = max(self.worst_batch_residual, float(relative[accepted].max()))
        self.total_batch_seconds += time.perf_counter() - started   # ends at the host read of the solutions
        return solutions, relative, accepted

    def _combined(self, pattern, coefficients, extras):
        """(N, z) complex128 values of the backend: coefficients @ terms + the extra blocks at their slots."""
        xp = pattern.xp
        values = xp.asarray(coefficients) @ pattern.terms
        for index, extra in enumerate(extras or ()):
            if extra is not None:
                slots, block = extra
                values[index, xp.asarray(np.asarray(slots, dtype=np.int64))] += xp.asarray(
                    np.asarray(block, dtype=np.complex128))
        return values

    def _host_batch(self, pattern, coefficients, rhs, extras):
        """CPU: one SuperLU per system of the batch (the same results as the gpu path, for parity)."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import splu
        values = self._combined(pattern, coefficients, extras)
        n = pattern.n_unknowns
        solutions = np.zeros(rhs.shape, dtype=np.complex128)
        relative = np.full(rhs.shape[0], np.inf)
        for index in range(rhs.shape[0]):
            matrix = csr_matrix((values[index], pattern.indices, pattern.indptr), shape=(n, n))
            try:
                solutions[index] = splu(matrix.tocsc()).solve(rhs[index])
            except RuntimeError:   # exactly singular: rejected (the consumer words it)
                continue
            norm = np.linalg.norm(rhs[index])
            residual = np.linalg.norm(rhs[index] - matrix @ solutions[index])
            relative[index] = residual / norm if norm > 0.0 else residual
        return solutions, relative

    def _device_batch(self, pattern, coefficients, rhs, extras):
        """GPU: combine, equilibrate, factorise all systems in one batched cuDSS call, refine every system against its
        float64 operator (one batched solve per step, one host read of the residual norms per step)."""
        import cupy as cp
        n_systems = rhs.shape[0]
        values = self._combined(pattern, coefficients, extras)
        diagonal = cp.abs(values[:, pattern.diagonal])
        diagonal = cp.where(pattern.diagonal_mask[None, :] & (diagonal > 0.0) & cp.isfinite(diagonal), diagonal, 1.0)
        scale = 1.0 / cp.sqrt(diagonal)                                   # (N, n) float64
        scaled = values * scale[:, pattern.rows] * scale[:, pattern.columns]
        stored = scaled if pattern.lower is None else scaled[:, pattern.lower]
        plan = self._planned(pattern, n_systems)
        plan.factorize(stored.astype(self.factor_dtype))
        del scaled, stored
        right = cp.asarray(rhs)
        rhs_norms = cp.linalg.norm(right, axis=1)
        rhs_norms = cp.where(rhs_norms > 0.0, rhs_norms, 1.0)
        solution = cp.zeros_like(right)
        residual = right
        relative = np.full(n_systems, np.inf)
        steps = 0
        for step in range(MAX_REFINEMENTS[self.precision] + 1):
            correction = self._corrections(plan, residual, scale)
            candidate = solution + correction
            candidate_residual = right - self._products(pattern, values, candidate)
            candidate_relative = cp.asnumpy(cp.linalg.norm(candidate_residual, axis=1) / rhs_norms)
            if step:
                steps += 1
            improved = ~(candidate_relative >= relative)   # NaN candidates never replace a finite iterate
            improved &= np.isfinite(candidate_relative) | ~np.isfinite(relative)
            keep = cp.asarray(improved)[:, None]
            progressing = (relative > REFINEMENT_TOLERANCE) & (candidate_relative < STALL_RATIO * relative)
            solution = cp.where(keep, candidate, solution)
            residual = cp.where(keep, candidate_residual, residual)
            relative = np.where(improved, candidate_relative, relative)
            if not np.any(relative > REFINEMENT_TOLERANCE) or (step and not progressing.any()):
                break
        self.n_batch_refinements += steps
        return cp.asnumpy(solution), relative

    def _corrections(self, plan, residual, scale):
        """R E^-1 R r of every system (rhs scaled to unit norm per system against float32 underflow)."""
        import cupy as cp
        scaled = residual * scale
        norms = cp.linalg.norm(scaled, axis=1)
        norms = cp.where(norms > 0.0, norms, 1.0)
        unit = (scaled / norms[:, None]).astype(self.factor_dtype)
        return plan.solve(unit).astype(cp.complex128) * (norms[:, None] * scale)

    @staticmethod
    def _products(pattern, values, vectors):
        """A_k x_k of every system, float64: S (values * x[:, columns])^T, one SpMM."""
        return (pattern.selector @ (values * vectors[:, pattern.columns]).T).T

    def _planned(self, pattern, n_systems) -> _BatchPlan:
        """The kept plan when it fits (same pattern and dtype, at least n_systems), else a fresh one (counted)."""
        if self._plan is not None and self._plan.matches(pattern, n_systems, self.factor_dtype):
            return self._plan
        self.release()
        self._plan = _BatchPlan(pattern, n_systems, self.factor_dtype)
        self.n_batch_plans += 1
        self.total_batch_plan_seconds += self._plan.plan_seconds
        self.batch_permanent_bytes, self.batch_peak_bytes = self._plan.permanent_bytes, self._plan.peak_bytes
        return self._plan

    def work_counters(self) -> dict:
        """The SparseDirectSolver counters (per-system factors: rejected systems, other solves) plus the batches."""
        return dict(super().work_counters(), batches=self.n_batches, batched_systems=self.n_batched_systems,
                    batch_plans=self.n_batch_plans, batch_rejected=self.n_batch_rejected,
                    batch_refinement_steps=self.n_batch_refinements, batch_seconds=self.total_batch_seconds,
                    batch_plan_seconds=self.total_batch_plan_seconds,
                    batch_permanent_bytes=self.batch_permanent_bytes, batch_peak_bytes=self.batch_peak_bytes,
                    worst_batch_residual=self.worst_batch_residual, batch_unavailable=self.unavailable or '')


def batch_external_bytes(n_unknowns, entries, n_systems, precision, symmetric=True) -> tuple:
    """(permanent, peak extra) outside-pool bytes of one batched plan of n_systems (cudss_factor.factor_device_bytes per
    system, 2D growth law): the memory estimators' external item."""
    from .cudss_factor import factor_device_bytes
    permanent, extra = factor_device_bytes(n_unknowns, entries, 16, 2, mixed=precision == 'mixed',
                                           symmetric=symmetric)
    return int(n_systems) * permanent, int(n_systems) * extra


def time_estimate(values, work):
    """Run-time recipe (core.time_estimate.SolverWork; work.factorizations = systems, work.solves = their solves):
    priced as the per-system cuDSS factorisations and SOLVES_PER_SYSTEM_GUESS solves of the precision (gpu; the batch's
    shared launches make the measured time lower: a deviation is the batching gain, not a lead on the recipe), host
    SuperLU on the cpu."""
    from . import sparse_direct
    precision = values.get('precision', 'mixed')
    if work.backend == 'gpu':
        back_end = 'cudss_mixed' if precision == 'mixed' else 'cudss'
    else:
        back_end = 'scipy_splu'
    return sparse_direct.time_estimate({'back_end': back_end}, work)


def batches_of(n_systems, capacity) -> int:
    """Batches of n_systems at capacity systems each."""
    return int(math.ceil(int(n_systems) / max(int(capacity), 1)))
