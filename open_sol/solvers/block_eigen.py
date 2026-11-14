"""'block_eigen' solver type runtime (role 'eigen'): the factor-free preconditioned block eigensolver (LOBPCG with soft
locking, row-block storage and chunked in-place updates) of the lowest eigenpairs of a symmetric pencil K x = lambda M
x on caller block callbacks; float64, or 'mixed' (a float32 phase, then float64 to the tolerance); counted. numpy only
at module level."""
import math
import time

import numpy as np

# Layout (module notes; solvers sub file "block_eigen"): one search block of b = wanted + guard rows. Three row stores
# S, KS = K S, MS = M S of (3 b, n) in the phase dtype hold the basis [X (b rows) | W (active rows) | P (active rows)]
# and its two products, every row a C-contiguous vector (what the physics' kernels take): 9 b n values, nothing else
# of that size. Per iteration (Knyazev 2001; implicit P and soft locking after Hetmaniuk-Lehoucq 2006 / Duersch et al.
# 2018):
#  1. R = KX - Lambda MX into the KS W slots (two block launches), ||R_i|| and ||MX_i|| by column chunks, ONE host read;
#     rows at the tolerance lock (they stay in X and the Rayleigh-Ritz, without W / P rows);
#  2. the active residual rows moved up (row copies, forward: a destination never precedes its source's reader), the
#     caller's preconditioner into the S W slots, the mask, one projection against X (W -= X (MX W^T): X is
#     M-orthonormal after every Rayleigh-Ritz), then the caller's K and M products of W into KS / MS;
#  3. the last P rows of the active set moved behind W (forward copies in S, KS, MS);
#  4. Rayleigh-Ritz on the m = b + 2 n_active rows: A = S KS^T, B = S MS^T (two GEMMs, ONE host read), on the host
#     diag(B)^-1/2 scaled, B's eigenvalues below DROP x the largest dropped (a dependent W / P direction), the lowest b
#     Ritz pairs;
#  5. X = C^T S and P = C_(W,P)^T S_(W,P) for S, KS and MS by column chunks of CHUNK_BYTES (each chunk read into the
#     scratch before it is written: in place, no second (b, n) block).
# Mixed: the float32 phase stops at max(tolerance, SINGLE_PHASE_TOLERANCE) or when the worst wanted residual stalls
# (STALL_ITERATIONS without a STALL_RATIO gain: the float32 rounding floor ~eps32 lambda_max / lambda of the products);
# X is cast to float64 (the float32 stores freed first) and LOBPCG continues there to the tolerance. Float32 halves the
# operator bandwidth and runs the dense block products (Gram matrices, projections, rotations) at the full float32 rate
# of consumer GPUs (float64 there: 1/32 - 1/64); the float64 phase starts near the answer.
# Problems with fewer free unknowns than DENSE_MAX_RATIO x b go to a dense eigh of the applied K and M (no LOBPCG).

DEFAULT_TOLERANCE = 1e-8       # relative residual ||K x - lambda M x|| / (|lambda| ||M x||) of every wanted pair
DEFAULT_MAX_ITERATIONS = 1000  # max_iterations 0 and no caller default
GUARD_FRACTION = 0.25          # guard vectors (auto) = max(MIN_GUARD_VECTORS, ceil(GUARD_FRACTION x wanted))
MIN_GUARD_VECTORS = 2
SINGLE_PHASE_TOLERANCE = 1e-4  # mixed: the float32 phase stops here at the latest (then float64)
STALL_ITERATIONS = 8           # iterations whose best worst-wanted residual must beat STALL_RATIO x the best before ...
STALL_RATIO = 0.9              # ... else the float32 phase ends (mixed), or the float64 one stops unconverged
DOUBLE_STALL_ITERATIONS = 40   # float64: a stall that long is the rounding floor (tolerance too tight)
CHUNK_BYTES = 8 * 2 ** 20      # scratch of one column chunk of a block product
# Rayleigh-Ritz basis drop (eigenvalues of the diag-scaled B below this x the largest): above the Gram rounding of the
# dtype (float32 GEMMs over ~1e6 entries: ~1e-6 relative entries), so a noise direction is never amplified
DROP = {np.dtype(np.float32): 1e-5, np.dtype(np.float64): 1e-12}
DENSE_MAX_RATIO = 6            # fewer free unknowns than this x the block: dense eigh
START_SEED = 23
PRECISIONS = ('mixed', 'double')
PRECONDITIONERS = ('multigrid', 'jacobi')
# time recipe guesses (a priori, never fitted): iterations per decade of the residual with a multigrid preconditioner,
# the float32 share of a mixed run's iterations
MULTIGRID_ITERATIONS_PER_DECADE = 6.0
MIXED_SINGLE_FRACTION = 0.6


def block_size(n_wanted, guard_vectors=0) -> int:
    """Rows of the search block: the wanted eigenpairs + guard vectors (0 = auto)."""
    n_wanted = max(int(n_wanted), 1)
    guard = int(guard_vectors) or max(MIN_GUARD_VECTORS, math.ceil(GUARD_FRACTION * n_wanted))
    return n_wanted + guard


def block_bytes(n_unknowns, n_wanted, guard_vectors=0, precision='mixed') -> int:
    """Device (or host on the cpu) bytes of one block_eigen solve: the three (3 b, n) row stores of the float64 phase
    (mixed: its float32 phase holds half, and the cast keeps the float32 X beside the float64 S once) + the chunk
    scratch."""
    rows = 3 * block_size(n_wanted, guard_vectors)
    stores = 3 * rows * 8 * int(n_unknowns)
    cast = (rows // 3) * 4 * int(n_unknowns) if precision == 'mixed' else 0
    return int(stores + cast + 2 * CHUNK_BYTES)


def _array_module(array):
    module = type(array).__module__
    if module.startswith('cupy'):
        import cupy
        return cupy
    return np


def _host(array):
    return array.get() if hasattr(array, 'get') else np.asarray(array)


def _chunks(n_unknowns, rows, itemsize):
    width = max(1024, CHUNK_BYTES // (int(itemsize) * max(int(rows), 1)))
    return [(start, min(start + width, n_unknowns)) for start in range(0, n_unknowns, width)]


class BlockEigenResult:
    """eigenvalues (k,) ascending, host vectors (n, k) (M-orthonormal, 0 off the mask), relative residuals (k,),
    converged, iterations (single, double) and the operator counts of one solve."""

    def __init__(self, eigenvalues, vectors, residuals, converged, single_iterations, double_iterations,
                 applications, seconds, method, warnings):
        self.eigenvalues = eigenvalues
        self.vectors = vectors
        self.residuals = residuals
        self.converged = converged
        self.single_iterations = single_iterations
        self.double_iterations = double_iterations
        self.applications = applications
        self.seconds = seconds
        self.method = method
        self.warnings = warnings

    @property
    def iterations(self) -> int:
        return self.single_iterations + self.double_iterations


class BlockEigenSolver:
    """'block_eigen' runtime: lowest(...) on block callbacks (module notes). Settings: tolerance (relative residual),
    guard_vectors (0 auto), max_iterations (0: the caller's default, else DEFAULT_MAX_ITERATIONS), precision 'mixed' /
    'double', preconditioner 'multigrid' / 'jacobi' (read by the physics that builds the preconditioner callback)."""

    def __init__(self, tolerance=DEFAULT_TOLERANCE, guard_vectors=0, max_iterations=0, precision='mixed',
                 preconditioner='multigrid', label=None, backend='gpu'):
        if precision not in PRECISIONS:
            raise ValueError(f"block_eigen: precision must be one of {', '.join(PRECISIONS)}, got {precision!r}")
        if preconditioner not in PRECONDITIONERS:
            raise ValueError(f"block_eigen: preconditioner must be one of {', '.join(PRECONDITIONERS)}, got "
                             f'{preconditioner!r}')
        self.tolerance = float(tolerance) if tolerance else DEFAULT_TOLERANCE
        self.guard_vectors = int(guard_vectors)
        self.max_iterations = int(max_iterations)
        self.precision = precision
        self.preconditioner = preconditioner
        self.backend = backend
        self._label = label
        self.n_eigen_solves = 0
        self.n_unconverged = 0
        self.n_dense_solves = 0
        self.total_iterations = 0
        self.single_iterations = 0
        self.double_iterations = 0
        self.operator_applications = 0      # K products (rows)
        self.mass_applications = 0          # M products (rows)
        self.preconditioner_applications = 0
        self.total_eigen_seconds = 0.0
        self.last_residual = 0.0
        self.last_block_size = 0
        self.last_warning = ''

    def label(self) -> str:
        label = self._label() if callable(self._label) else self._label
        return label or 'block_eigen solver'

    def resolved_max_iterations(self, default=0) -> int:
        return self.max_iterations or int(default) or DEFAULT_MAX_ITERATIONS

    # --- the solve ----------------------------------------------------------------------------------------------------

    def lowest(self, n_unknowns, n_wanted, stiffness, mass, precondition, xp, mask=None, default_max_iterations=0):
        """BlockEigenResult of the n_wanted lowest eigenpairs of K x = lambda M x on n_unknowns, K symmetric and M SPD
        on the masked unknowns. Callbacks take (rows_in, rows_out): C-contiguous (m, n) arrays of the backend `xp`
        (numpy / cupy), float32 or float64 (mixed: both, one dtype per call), rows_out = K rows_in / M rows_in / the
        preconditioner (an SPD approximation of K^-1, e.g. one multigrid V-cycle); mask: a backend bool (n,) array of
        the unknowns that take part (None: all), rows are 0 elsewhere and the callbacks keep them 0."""
        started = time.perf_counter()
        n_unknowns, n_wanted = int(n_unknowns), int(n_wanted)
        rows = block_size(n_wanted, self.guard_vectors)
        n_free = int(_host(mask.sum())) if mask is not None else n_unknowns
        if n_wanted < 1 or n_wanted >= n_free:
            raise ValueError(f'{self.label()}: {n_wanted} eigenpairs asked of {n_free} free unknowns')
        self.last_block_size = rows
        try:
            if n_free < DENSE_MAX_RATIO * rows:
                result = self._dense(n_unknowns, n_wanted, stiffness, mass, xp, mask)
            else:
                result = _Lobpcg(self, n_unknowns, n_wanted, rows, stiffness, mass, precondition, xp, mask,
                                 self.resolved_max_iterations(default_max_iterations)).run()
        finally:
            self.total_eigen_seconds += time.perf_counter() - started
            self.n_eigen_solves += 1
        result.seconds = time.perf_counter() - started
        self.total_iterations += result.iterations
        self.single_iterations += result.single_iterations
        self.double_iterations += result.double_iterations
        self.last_residual = float(np.max(result.residuals)) if np.size(result.residuals) else 0.0
        if not result.converged:
            self.n_unconverged += 1
            self.last_warning = result.warnings[-1] if result.warnings else 'not converged'
        return result

    def _dense(self, n_unknowns, n_wanted, stiffness, mass, xp, mask):
        """Small problems: K and M applied to the unit rows of the free unknowns (float64), scipy eigh of the dense
        free pencil."""
        import scipy.linalg
        free = (np.flatnonzero(_host(mask)) if mask is not None else np.arange(n_unknowns)).astype(np.int64)
        n_free = free.size
        dense_k = np.empty((n_free, n_free))
        dense_m = np.empty((n_free, n_free))
        step = max(1, CHUNK_BYTES // (8 * max(n_unknowns, 1)))
        free_device = xp.asarray(free)
        for start in range(0, n_free, step):
            stop = min(start + step, n_free)
            units = xp.zeros((stop - start, n_unknowns), dtype=np.float64)
            units[xp.arange(stop - start), free_device[start:stop]] = 1.0
            products = xp.empty_like(units)
            stiffness(units, products)
            dense_k[start:stop] = _host(products[:, free_device])
            mass(units, products)
            dense_m[start:stop] = _host(products[:, free_device])
        self.operator_applications += n_free
        self.mass_applications += n_free
        self.n_dense_solves += 1
        values, vectors = scipy.linalg.eigh(0.5 * (dense_k + dense_k.T), 0.5 * (dense_m + dense_m.T),
                                            subset_by_index=(0, n_wanted - 1))
        full = np.zeros((n_unknowns, n_wanted))
        full[free] = vectors
        return BlockEigenResult(values, full, np.zeros(n_wanted), True, 0, 0, n_free, 0.0, 'dense eigh', [])

    def work_counters(self) -> dict:
        """Work so far (summary 'solvers' record of its (physics, role)); eigen_seconds = host wall time of the solves."""
        return {'eigen_solves': self.n_eigen_solves, 'eigen_iterations': self.total_iterations,
                'eigen_single_iterations': self.single_iterations,
                'eigen_double_iterations': self.double_iterations,
                'operator_applications': self.operator_applications, 'mass_applications': self.mass_applications,
                'preconditioner_applications': self.preconditioner_applications,
                'eigen_seconds': self.total_eigen_seconds, 'eigen_unconverged': self.n_unconverged,
                'eigen_dense_solves': self.n_dense_solves, 'eigen_block_size': self.last_block_size,
                'eigen_last_residual': self.last_residual}


def _rayleigh_ritz(gram_a, gram_b, n_kept, drop):
    """(lowest n_kept Ritz values, coefficients C (m, n_kept) with C^T B C = I, basis directions kept) of the host
    Gram pair: diag(B)^-1/2 scaling, B's eigenvalues below drop x the largest dropped."""
    gram_a = 0.5 * (gram_a + gram_a.T)
    gram_b = 0.5 * (gram_b + gram_b.T)
    diagonal = np.sqrt(np.abs(np.diag(gram_b)))
    scale = np.where(diagonal > 0, 1.0 / np.where(diagonal > 0, diagonal, 1.0), 0.0)
    scaled_a = gram_a * scale[:, None] * scale[None, :]
    scaled_b = gram_b * scale[:, None] * scale[None, :]
    mu, unitary = np.linalg.eigh(scaled_b)
    keep = mu > drop * max(float(mu[-1]), 1e-300)
    if int(keep.sum()) < n_kept:
        raise RuntimeError(f'block_eigen: the Rayleigh-Ritz basis collapsed ({int(keep.sum())} independent directions '
                           f'for {n_kept} vectors): the operator or mass callbacks are not symmetric / definite')
    transform = unitary[:, keep] / np.sqrt(mu[keep])
    theta, vectors = np.linalg.eigh(transform.T @ scaled_a @ transform)
    return theta[:n_kept], scale[:, None] * (transform @ vectors[:, :n_kept]), int(keep.sum())


class _Lobpcg:
    """One lowest() run (module notes); the stores live only inside run()."""

    def __init__(self, owner, n_unknowns, n_wanted, rows, stiffness, mass, precondition, xp, mask, max_iterations):
        self.owner = owner
        self.n, self.k, self.b = n_unknowns, n_wanted, rows
        self.stiffness, self.mass, self.precondition = stiffness, mass, precondition
        self.xp, self.mask = xp, mask
        self.max_iterations = int(max_iterations)
        self.tolerance = owner.tolerance
        self.warnings = []
        self.store = None   # S, KS, MS
        self.theta = None   # host Ritz values (b,)
        self.have_p = False

    # --- stores ---------------------------------------------------------------------------------------------------

    def _allocate(self, dtype):
        xp = self.xp
        self.dtype = np.dtype(dtype)
        self.store = [xp.empty((3 * self.b, self.n), dtype=self.dtype) for _ in range(3)]
        self.drop = DROP[self.dtype]

    def _start(self):
        """Random start rows (seeded, masked) in S[:b]."""
        xp, b, n = self.xp, self.b, self.n
        basis = self.store[0]
        if xp is np:
            np.random.default_rng(START_SEED).standard_normal(dtype=self.dtype, out=basis[:b])
        else:   # row by row: no second (b, n) block
            generator = xp.random.RandomState(START_SEED)
            for row in range(b):
                basis[row] = generator.standard_normal(n, dtype=self.dtype)
        if self.mask is not None:
            xp.multiply(basis[:b], self.mask, out=basis[:b])

    def _apply(self, first, stop):
        """KS / MS rows [first, stop) of S by the callbacks."""
        basis, stiff, massed = self.store
        self.stiffness(basis[first:stop], stiff[first:stop])
        self.mass(basis[first:stop], massed[first:stop])
        self.owner.operator_applications += stop - first
        self.owner.mass_applications += stop - first

    def _gram_pair(self, m):
        """Host (A, B) = (S KS^T, S MS^T) of the first m rows, one read."""
        basis, stiff, massed = self.store
        pair = self.xp.empty((2, m, m), dtype=self.dtype)
        pair[0] = basis[:m] @ stiff[:m].T
        pair[1] = basis[:m] @ massed[:m].T
        pair = _host(pair).astype(np.float64)
        return pair[0], pair[1]

    def _rotate(self, coefficients, m):
        """X = C^T S[:m] into rows [0, b) and, with W / P rows, P = C_(b:)^T S[b:m] into rows [2 b, 3 b), for S, KS,
        MS by column chunks (each chunk read before it is written)."""
        xp, b = self.xp, self.b
        with_p = m > b
        rows_out = 2 * b if with_p else b
        stacked = np.zeros((rows_out, m))
        stacked[:b] = coefficients.T
        if with_p:
            stacked[b:, b:] = coefficients[b:].T
        stacked = xp.asarray(stacked, dtype=self.dtype)
        for start, stop in _chunks(self.n, rows_out + m, self.dtype.itemsize):
            for store in self.store:
                rotated = stacked @ store[:m, start:stop]
                store[:b, start:stop] = rotated[:b]
                if with_p:
                    store[2 * b:3 * b, start:stop] = rotated[b:]
        self.have_p = with_p

    def _residual_norms(self):
        """R = KX - Lambda MX into the KS W slots; host (||R_i||, ||MX_i||) by column chunks, one read."""
        xp, b = self.xp, self.b
        basis, stiff, massed = self.store
        residual = stiff[b:2 * b]
        theta = xp.asarray(self.theta, dtype=self.dtype)[:, None]
        xp.multiply(massed[:b], theta, out=residual)
        xp.subtract(stiff[:b], residual, out=residual)
        sums = xp.zeros((2, b), dtype=np.float64)
        for start, stop in _chunks(self.n, 2 * b, self.dtype.itemsize):
            part = residual[:, start:stop]
            sums[0] += xp.einsum('ij,ij->i', part, part)
            part = massed[:b, start:stop]
            sums[1] += xp.einsum('ij,ij->i', part, part)
        sums = np.sqrt(np.maximum(_host(sums), 0.0))
        return sums[0], sums[1]

    def _expand(self, active):
        """W rows (preconditioned active residuals, masked, projected against X) and their products; the last P rows
        of the active set moved behind them. Returns the basis rows m."""
        xp, b = self.xp, self.b
        basis, stiff, massed = self.store
        n_active = len(active)
        for row, source in enumerate(active):   # forward: row <= source
            if row != source:
                stiff[b + row] = stiff[b + source]
        self.precondition(stiff[b:b + n_active], basis[b:b + n_active])
        self.owner.preconditioner_applications += n_active
        work = basis[b:b + n_active]
        if self.mask is not None:
            xp.multiply(work, self.mask, out=work)
        projection = massed[:b] @ work.T   # X^T M W (b, n_active): X is M-orthonormal
        projection_rows = xp.ascontiguousarray(projection.T)
        for start, stop in _chunks(self.n, n_active + b, self.dtype.itemsize):
            work[:, start:stop] -= projection_rows @ basis[:b, start:stop]
        self._apply(b, b + n_active)
        m = b + n_active
        if self.have_p:
            for row, source in enumerate(active):   # P_i at 2 b + i -> b + n_active + row (forward)
                target, origin = m + row, 2 * b + source
                if target != origin:
                    for store in self.store:
                        store[target] = store[origin]
            m += n_active
        return m

    # --- phases ---------------------------------------------------------------------------------------------------

    def _ritz(self, m):
        """Rayleigh-Ritz on S[:m] and the rotation into X (and P)."""
        gram_a, gram_b = self._gram_pair(m)
        theta, coefficients, _ = _rayleigh_ritz(gram_a, gram_b, self.b, self.drop)
        self.theta = theta
        self._rotate(coefficients, m)

    def _phase(self, tolerance, iteration_limit, single):
        """LOBPCG iterations on the current stores to `tolerance` (wanted rows); (iterations, relative residuals,
        converged, stalled)."""
        b, k = self.b, self.k
        history = []
        stall_window = STALL_ITERATIONS if single else DOUBLE_STALL_ITERATIONS
        iterations = 0
        while True:
            residual_norms, mass_norms = self._residual_norms()
            relative = residual_norms / np.maximum(np.abs(self.theta) * mass_norms, 1e-300)
            worst = float(np.max(relative[:k]))
            if worst <= tolerance:
                return iterations, relative[:k], True, False
            history.append(worst)   # stalled: the best of the last window gained less than STALL_RATIO on the best before
            if len(history) > stall_window and min(history[-stall_window:]) > STALL_RATIO * min(history[:-stall_window]):
                return iterations, relative[:k], False, True
            if iterations >= iteration_limit:
                return iterations, relative[:k], False, False
            active = [int(row) for row in np.flatnonzero(relative > tolerance)]
            m = self._expand(active)
            self._ritz(m)
            iterations += 1

    def _initial(self):
        """KX, MX of S[:b] and the first Rayleigh-Ritz (X M-orthonormal)."""
        self.have_p = False
        self._apply(0, self.b)
        self._ritz(self.b)

    def _to_double(self):
        """Float32 stores -> float64 (X cast, products re-applied in float64; P dropped)."""
        single_x = self.store[0][:self.b].copy()
        self.store = None
        self._allocate(np.float64)
        self.store[0][:self.b] = single_x
        del single_x
        self._initial()

    def run(self) -> BlockEigenResult:
        owner = self.owner
        applications_before = owner.operator_applications
        mixed = owner.precision == 'mixed'
        self._allocate(np.float32 if mixed else np.float64)
        self._start()
        self._initial()
        single_iterations = double_iterations = 0
        limit = self.max_iterations
        converged = stalled = False
        relative = np.zeros(self.k)
        if mixed:
            single_tolerance = max(self.tolerance, SINGLE_PHASE_TOLERANCE)
            single_iterations, relative, converged, stalled = self._phase(single_tolerance, limit, single=True)
            converged = converged and single_tolerance <= self.tolerance
            if not converged:
                self._to_double()
        if not converged:
            double_iterations, relative, converged, stalled = self._phase(self.tolerance,
                                                                          max(limit - single_iterations, 1),
                                                                          single=False)
        if not converged:
            reason = ('the residual stalled (rounding floor: loosen the tolerance)' if stalled
                      else f'{limit} iterations were not enough (raise max_iterations, or use the multigrid '
                           'preconditioner)')
            self.warnings.append(f'{owner.label()}: block eigensolver stopped with relative residual '
                                 f'{float(np.max(relative)):.2e} above the tolerance {self.tolerance:.1e}: {reason}')
        vectors = _host(self.store[0][:self.k]).astype(np.float64, copy=False).T
        eigenvalues = np.asarray(self.theta[:self.k], dtype=np.float64)
        self.store = None
        method = f"LOBPCG ({'float32 + float64' if mixed else 'float64'}, block {self.b})"
        return BlockEigenResult(eigenvalues, vectors, np.asarray(relative, dtype=np.float64), converged,
                                single_iterations, double_iterations,
                                owner.operator_applications - applications_before, 0.0, method,
                                self.warnings)


def time_estimate(values, work):
    """Run-time recipe of a block_eigen role (core.time_estimate.SolverWork; solves = eigen solves of work.eigenpairs
    each): per iteration and active row one product_ops (the caller's K and M products) + preconditioner_ops, the block
    elementwise passes, the dense block products (DenseFlops; mixed: MIXED_SINGLE_FRACTION of the iterations float32)
    and two host reads (no seconds counter: eigen_seconds spans both phases). Iterations:
    MULTIGRID_ITERATIONS_PER_DECADE x decades of the tolerance with a caller-built multigrid (work.setup_ops), else sqrt(kappa) ln(2 / tol) / 2 of work.jacobi_condition, else
    pcg.CG_ITERATIONS_PER_AXIS_CELL x the longest axis (a guess, never fitted)."""
    from ..core import time_estimate as te
    from .pcg import CG_ITERATIONS_PER_AXIS_CELL
    n = float(work.unknowns)
    rows = block_size(max(int(work.eigenpairs), 1), int(values.get('guard_vectors', 0)))
    tolerance = float(values.get('tolerance', DEFAULT_TOLERANCE) or DEFAULT_TOLERANCE)
    mixed = values.get('precision', 'mixed') == 'mixed'
    if work.setup_ops:
        iterations = MULTIGRID_ITERATIONS_PER_DECADE * math.log10(1.0 / tolerance)
    elif work.jacobi_condition:
        iterations = 0.5 * math.sqrt(work.jacobi_condition) * math.log(2.0 / tolerance)
    else:
        iterations = CG_ITERATIONS_PER_AXIS_CELL * max(int(work.longest_axis), 1)
    m = 3.0 * rows
    # Gram pair 2 x 2 m^2, rotations 3 stores x 2 (2 b) m, projection 2 b^2 x 2, block passes ~ 6 b n values
    flops = (4.0 * m * m + 12.0 * rows * m + 4.0 * rows * rows) * n

    def per_iteration(single):
        item = 4 if single else 8
        return (tuple((float(rows), op) for op in work.product_ops + work.preconditioner_ops)
                + (te.Elementwise(6.0 * rows * item * n, 3.0 * rows * item * n, launches=4 * rows),
                   te.DenseFlops(flops, host=work.backend == 'cpu', single=single), te.Sync(2)))
    single_share = MIXED_SINGLE_FRACTION if mixed else 0.0
    phases = [(label, share, single) for label, share, single in
              (('block eigen iterations (float32)', single_share, True),
               ('block eigen iterations (float64)', 1.0 - single_share, False)) if share > 0.0]
    items = tuple(te.TimeItem(label, 'iteration', float(work.solves) * share * iterations, 'guess',
                              per_iteration(single), per_step=work.per_step,
                              counter=work.counter('eigen_single_iterations' if single else 'eigen_double_iterations'))
                  for label, share, single in phases)
    return te.TimeEstimate(items)
