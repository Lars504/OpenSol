"""'time_filter' eigen-solver runtime: resonances of an open or lossy frequency-domain operator without a
factorisation. The physics time-steps the same structure (FDTD) from a few broadband sources, accumulates a running DFT
of the whole grid at sampled frequencies of a band around the target (one complex accumulator per sampled frequency:
each one a filter (omega_j - H)^-1-like combination of the modes there), and this module's Rayleigh-Ritz step projects
the frequency-domain pencil A u = mu M u onto the span of those fields (plus polish steps that append the wanted Ritz
pairs' residuals), solves the small dense eigenproblem and reports residuals by sparse products only. numpy at module
level; cupy / scipy inside functions; the time stepping lives with the physics (physics.em.time_filter_eigen)."""
# Rayleigh-Ritz (dense part, any array module xp). Memory: the search basis is ONE block of rows (capacity, n)
# complex128 (a vector per row: the DFT accumulators' own layout, and row slices stay contiguous), allocated by the
# caller with room for the polish rows (basis_columns) and overwritten in place by its orthonormal basis Q; everything
# else is RITZ_CHUNK vectors at a time (operator products, projected columns, Ritz vectors, residuals) or m x m on the
# host. Nothing of size n x m is copied: no A Q block (the products are made again for every projection: sparse
# products are cheap beside the dense ones), no conj(Q) (Q^H X = conj(Q^T conj(X)) on the chunk X), and only the Ritz
# vectors the caller keeps (keep) are formed.
# Orthonormalisation in place: Gram matrix G = Q^H Q by row chunks, host eigh of the norm-scaled G, directions below
# RANK_TOLERANCE x the largest singular value (RANK_TOLERANCE^2 of the Gram eigenvalues) dropped (sampled frequencies
# far from every mode give nearly parallel fields), rows <- T^T rows by unknown chunks; done twice (the second pass
# removes the first's loss of orthogonality, ~ round-off x condition^2: the cut must stay above ~1e-8).
# Projection: with a symmetrizer D (D A, D M complex symmetric) bilinear, A_r = (D Q)^T A Q, M_r = (D Q)^T M Q (the
# stationary quotient of a complex symmetric pencil: eigenvalue error ~ the square of the subspace error), else
# Galerkin Q^H A Q, Q^H M Q; scipy.linalg.eig on the host (m x m). Polish step: append the M^-1-scaled residuals
# (A u - mu M u) / M of the wanted Ritz pairs not yet below CONVERGED_RESIDUAL (pinned entries 0; directions already
# in the span dropped, as is a step with none left) to the basis and project again: an unpreconditioned Davidson step,
# enough to remove the FDTD time-discretisation error ((omega dt)^2 / 24 of the mode frequency) and the PML-region
# mismatch of the filtered fields (FDTD applies its CPML stretch at every omega, the frequency-domain pencil freezes it
# at the target).
import math

import numpy as np

PRECISIONS = ('single', 'double')   # FDTD stepper precision (fields, coefficients); the projection is float64
# float32 stepping injects rounding every step, which leaks into every sampled field as ~1e-6 grid-scale noise (a
# window cannot remove it: each injection starts mid-run); the stationary quotient keeps omega near 1e-12 but the
# Laplacian lifts the noise ~2 / (k h)^2 in a residual (1e-4 - 1e-3: real modes fail SPURIOUS_RESIDUAL, each polish
# direction is mostly noise). The step cost is the complex128 DFT sums (16 B x sampled per cell), so double adds ~10 %.
DEFAULT_PRECISION = 'double'
DEFAULT_BANDWIDTH = 0.1     # half-width of the sampled band, relative to the target frequency
DEFAULT_SOURCES = 4         # random point sources of the broadband excitation
DEFAULT_POLISH_STEPS = 2
MIN_PERIODS = 100           # run length: at least this many target periods, and RESOLUTION_PERIODS / bandwidth
RESOLUTION_PERIODS = 10.0
MIN_SAMPLED_FREQUENCIES = 12   # sampled frequencies (0 = auto): max(this, 4 x wanted + 8), at most MAX_SAMPLED
SAMPLES_PER_WANTED = 4
MAX_SAMPLED_FREQUENCIES = 64
RANK_TOLERANCE = 1e-7       # basis directions below this x the largest singular value are dropped (Gram: squared)
FRESH_TOLERANCE = 1e-5      # a polish residual below this x its norm after projecting out the basis adds no row
RITZ_CHUNK = 8              # vectors per operator product / projection / Ritz-vector block
ORTHONORMAL_CHUNKS = 16     # the in-place basis rotation works on n / this unknowns at a time (>= MIN_ROTATE)
MIN_ROTATE_UNKNOWNS = 4096
CONVERGED_RESIDUAL = 1e-12  # a wanted pair below this relative residual adds no polish column (round-off only)
PULSE_BANDWIDTH_FACTOR = 3.0   # source pulse relative bandwidth = this x the sampled band half-width (<= 1)
SOURCE_SEED = 31            # deterministic source positions / amplitudes
RESIDUAL_WARNING = 1e-6     # a reported resonance above this relative residual is named in a warning
SPURIOUS_RESIDUAL = 1e-3    # an in-band Ritz pair above this relative residual is no mode of the span: dropped
GUARD_PAIRS = 2             # polished Ritz pairs beyond the requested resonances (wanted = n_resonances + this)


def wanted_pairs(n_resonances) -> int:
    """Ritz pairs each polish step refines: the requested resonances + GUARD_PAIRS."""
    return int(n_resonances) + GUARD_PAIRS


def basis_columns(n_resonances, sampled_frequencies=0, polish_steps=DEFAULT_POLISH_STEPS) -> int:
    """Largest basis of one solve (estimators): the sampled fields + polish_steps x wanted_pairs residual columns."""
    return sampled_count(n_resonances, sampled_frequencies) + int(polish_steps) * wanted_pairs(n_resonances)


def sampled_count(n_resonances, sampled_frequencies=0) -> int:
    """Sampled frequencies of a run: the node's value, or (0) max(MIN_SAMPLED_FREQUENCIES, SAMPLES_PER_WANTED x wanted +
    8) capped at MAX_SAMPLED_FREQUENCIES."""
    if int(sampled_frequencies) > 0:
        return int(sampled_frequencies)
    return int(min(MAX_SAMPLED_FREQUENCIES, max(MIN_SAMPLED_FREQUENCIES, SAMPLES_PER_WANTED * int(n_resonances) + 8)))


def run_periods(bandwidth, periods=0) -> float:
    """Target periods of the time stepping: the node's value, or (0) max(MIN_PERIODS, RESOLUTION_PERIODS / bandwidth)."""
    if float(periods) > 0.0:
        return float(periods)
    return float(max(MIN_PERIODS, RESOLUTION_PERIODS / float(bandwidth)))


def sampled_frequencies(target_frequency, bandwidth, count) -> np.ndarray:
    """count frequencies [Hz] evenly over target (1 +- bandwidth) (the target itself for count 1)."""
    if int(count) == 1:
        return np.array([float(target_frequency)])
    return float(target_frequency) * np.linspace(1.0 - float(bandwidth), 1.0 + float(bandwidth), int(count))


class TimeFilterEigenSolver:
    """'time_filter' runtime of an 'eigen' role: the settings the physics' time stepping reads (bandwidth, sampled
    frequencies, run periods, sources, stepper precision, polish steps), the Rayleigh-Ritz step (ritz_pairs) and the
    work record."""

    def __init__(self, bandwidth=DEFAULT_BANDWIDTH, sampled_frequencies=0, periods=0.0, sources=DEFAULT_SOURCES,
                 polish_steps=DEFAULT_POLISH_STEPS, precision=DEFAULT_PRECISION, label=None, backend='gpu'):
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {', '.join(PRECISIONS)}, got {precision!r}")
        if not (0.0 < float(bandwidth) < 1.0):
            raise ValueError(f'bandwidth must lie in (0, 1) (relative to the target), got {bandwidth!r}')
        if int(sampled_frequencies) < 0 or float(periods) < 0.0 or int(sources) < 1 or int(polish_steps) < 0:
            raise ValueError(f'need sampled_frequencies >= 0, periods >= 0, sources >= 1, polish_steps >= 0, got '
                             f'{sampled_frequencies}, {periods}, {sources}, {polish_steps}')
        if backend not in ('gpu', 'cpu'):
            raise ValueError(f"backend must be 'gpu' or 'cpu', got {backend!r}")
        self.bandwidth = float(bandwidth)
        self.sampled_frequencies = int(sampled_frequencies)
        self.periods = float(periods)
        self.sources = int(sources)
        self.polish_steps = int(polish_steps)
        self.precision = precision
        self.label = label if label is not None else (lambda: 'time_filter')
        self.backend = backend
        self.n_solves = 0
        self.n_time_steps = 0
        self.n_sampled = 0
        self.n_basis_columns = 0
        self.n_operator_applications = 0
        self.total_step_seconds = 0.0
        self.total_projection_seconds = 0.0
        self.last_residual = 0.0

    def count_run(self, steps, sampled, step_seconds) -> None:
        """Book one time stepping run (steps, sampled frequencies, wall seconds of the stepping)."""
        self.n_time_steps += int(steps)
        self.n_sampled += int(sampled)
        self.total_step_seconds += float(step_seconds)

    def ritz_pairs(self, apply_operator, mass, basis, wanted, target, symmetrizer=None, xp=np, columns=None,
                   keep=None) -> dict:
        """Rayleigh-Ritz of A u = mu M u on the span of the first `columns` rows of basis (xp (capacity, n) complex128
        C-contiguous, a vector per row; None: all rows) with polish_steps polish steps: apply_operator(V) -> A V for V
        (n, k) of the array module xp (k <= RITZ_CHUNK), mass = M diagonal (n,) xp (0 on pinned unknowns),
        symmetrizer D (n,) xp or None (module notes); wanted pairs = the `wanted` finite mu nearest `target` (mu of
        the target resonance, e.g. 0 for A = L + k_t^2 M) that each polish step refines. basis is OVERWRITTEN (its
        orthonormal basis; polish rows go into the spare rows, a basis without room is reallocated). keep(mu host
        (r,)) -> bool (r,): the Ritz pairs whose vectors are formed and returned (None: every finite one).
        Returns {mu (r,) host, vectors xp (r, n) rows (the kept Ritz pairs by |mu - target|), residuals (r,) host
        relative ||A u - mu M u|| / (||A u|| + |mu| ||M u||), columns (basis rank), applications (operator products,
        one per vector: every projection applies A to the whole basis again)}."""
        import time
        started = time.perf_counter()
        free = mass != 0
        basis[:, ~free] = 0
        rank = self._orthonormal_rows(basis, basis.shape[0] if columns is None else int(columns), xp)
        applications = 0
        mu = np.zeros(0, dtype=np.complex128)
        coefficients = np.zeros((0, 0), dtype=np.complex128)
        for step in range(self.polish_steps + 1):
            if rank == 0:
                break
            quotient = basis[:rank]
            mu, coefficients = self._project(quotient, apply_operator, mass, symmetrizer, xp)
            applications += rank
            order = np.argsort(np.where(np.isfinite(mu), np.abs(mu - target), np.inf))
            mu, coefficients = mu[order], coefficients[:, order]
            if step == self.polish_steps:
                break
            count = min(int(wanted), int(np.count_nonzero(np.isfinite(mu))))
            if count == 0:
                break
            extra, residuals = [], []
            for start in range(0, count, RITZ_CHUNK):   # the wanted pairs' residual vectors (count is small)
                chunk = slice(start, min(start + RITZ_CHUNK, count))
                _, residual_vectors, relative = self._ritz_block(quotient, coefficients[:, chunk], mu[chunk],
                                                                 apply_operator, mass, xp)
                applications += chunk.stop - chunk.start
                extra.append(residual_vectors)
                residuals.append(relative)
            residuals = np.concatenate(residuals)
            # a converged pair's residual is round-off outside every useful span: it would only grow the basis
            active = np.flatnonzero(residuals > CONVERGED_RESIDUAL)
            if active.size == 0:
                break
            safe_mass = xp.where(free, mass, 1.0)
            extra = xp.concatenate(extra, axis=1)[:, xp.asarray(active)]
            extra = xp.ascontiguousarray((xp.where(free[:, None], extra / safe_mass[:, None], 0)).T)   # (a, n) rows
            before = _host(xp.linalg.norm(extra, axis=1))
            for _ in range(2):   # two Gram-Schmidt passes against Q: E <- E - (Q^H E) Q without conj(Q)
                extra -= xp.conj(quotient @ xp.conj(extra).T).T @ quotient
            fresh = _host(xp.linalg.norm(extra, axis=1)) > FRESH_TOLERANCE * before   # not in span(Q) already
            extra = xp.ascontiguousarray(extra[xp.asarray(np.flatnonzero(fresh))])
            added = self._orthonormal_rows(extra, extra.shape[0], xp) if extra.shape[0] else 0
            if added == 0:
                break
            if rank + added > basis.shape[0]:   # no spare rows: one transient second block
                basis = xp.concatenate([basis[:rank], extra[:added]], axis=0)
            else:
                basis[rank:rank + added] = extra[:added]
            rank += added
            del extra, quotient
        quotient = basis[:rank]
        finite = np.isfinite(mu)
        chosen = np.flatnonzero(finite if keep is None else finite & np.asarray(keep(mu), dtype=bool))
        vectors = xp.empty((chosen.size, basis.shape[1]), dtype=np.complex128)
        residuals = np.zeros(chosen.size)
        for start in range(0, chosen.size, RITZ_CHUNK):
            chunk = chosen[start:start + RITZ_CHUNK]
            block, _, relative = self._ritz_block(quotient, coefficients[:, chunk], mu[chunk], apply_operator, mass,
                                                  xp)
            applications += chunk.size
            vectors[start:start + chunk.size] = block
            residuals[start:start + chunk.size] = relative
        self.n_operator_applications += applications
        self.n_basis_columns += rank
        self.total_projection_seconds += time.perf_counter() - started
        return {'mu': mu[chosen], 'vectors': vectors, 'residuals': residuals, 'columns': int(rank),
                'applications': applications}

    @staticmethod
    def _ritz_block(quotient, coefficients, mu, apply_operator, mass, xp):
        """(Ritz vectors (c, n) rows, residual vectors A u - mu M u (n, c), relative residuals (c,) host) of the c
        Ritz pairs with host coefficients (k, c) on the rows of quotient (k, n)."""
        block = xp.asarray(np.ascontiguousarray(coefficients.T)) @ quotient   # (c, n)
        products = apply_operator(block.T)
        massed = mass[:, None] * block.T
        device_mu = xp.asarray(mu)
        scale = xp.linalg.norm(products, axis=0) + xp.abs(device_mu) * xp.linalg.norm(massed, axis=0)
        residual_vectors = products - device_mu[None, :] * massed
        relative = _host(xp.linalg.norm(residual_vectors, axis=0) / xp.where(scale > 0, scale, 1.0))
        return block, residual_vectors, relative

    @staticmethod
    def _orthonormal_rows(rows, count, xp) -> int:
        """Overwrite the first rows of `rows` ((capacity, n) C-contiguous) with an orthonormal basis of the span of
        its first `count` rows; returns its rank (module notes: Gram + host eigh, twice, in place)."""
        n = rows.shape[1]
        step = max(MIN_ROTATE_UNKNOWNS, -(-n // ORTHONORMAL_CHUNKS))
        rank = int(count)
        for _ in range(2):
            if rank == 0:
                return 0
            block = rows[:rank]
            gram = np.empty((rank, rank), dtype=np.complex128)
            for start in range(0, rank, RITZ_CHUNK):   # G = conj(Q) Q^T (rows), conj on the chunk only
                stop = min(start + RITZ_CHUNK, rank)
                gram[start:stop] = _host(xp.conj(block[start:stop]) @ block.T)
            norms = np.sqrt(np.abs(np.diagonal(gram).real))
            nonzero = np.flatnonzero(norms > 0.0)
            if nonzero.size == 0:
                return 0
            scaled = gram[np.ix_(nonzero, nonzero)] / np.outer(norms[nonzero], norms[nonzero])
            values, directions = np.linalg.eigh(0.5 * (scaled + scaled.conj().T))
            kept = values > RANK_TOLERANCE ** 2 * float(values[-1])
            # new row i = sum_j rotation[i, j] old row j: an orthonormal combination, G = conj(R) G_old R^T = 1
            rotation = np.zeros((int(np.count_nonzero(kept)), rank), dtype=np.complex128)
            rotation[:, nonzero] = (directions[:, kept] / np.sqrt(values[kept])[None, :]).T / norms[nonzero][None, :]
            device_rotation = xp.asarray(rotation)
            new_rank = rotation.shape[0]
            for start in range(0, n, step):
                stop = min(start + step, n)
                rows[:new_rank, start:stop] = device_rotation @ rows[:rank, start:stop]
            rank = new_rank
        return rank

    @staticmethod
    def _project(quotient, apply_operator, mass, symmetrizer, xp):
        """(mu (k,) host, coefficients (k, k) host) of the pencil projected on the rows of quotient (k, n) (module
        notes), RITZ_CHUNK columns of the projected matrices per operator product."""
        from scipy.linalg import eig
        rank = quotient.shape[0]
        projected_a = np.empty((rank, rank), dtype=np.complex128)
        projected_m = np.empty((rank, rank), dtype=np.complex128)
        for start in range(0, rank, RITZ_CHUNK):
            chunk = slice(start, min(start + RITZ_CHUNK, rank))
            block = quotient[chunk].T                 # (n, c) view
            applied = apply_operator(block)
            massed = mass[:, None] * block
            if symmetrizer is None:                   # Q^H X = conj(Q^T conj(X))
                projected_a[:, chunk] = _host(xp.conj(quotient @ xp.conj(applied)))
                projected_m[:, chunk] = _host(xp.conj(quotient @ xp.conj(massed)))
            else:                                     # (D Q)^T X = Q^T (D X)
                projected_a[:, chunk] = _host(quotient @ (symmetrizer[:, None] * applied))
                projected_m[:, chunk] = _host(quotient @ (symmetrizer[:, None] * massed))
        mu, coefficients = eig(projected_a, projected_m)
        return np.asarray(mu, dtype=np.complex128), np.asarray(coefficients, dtype=np.complex128)

    def work_counters(self) -> dict:
        return {'eigen_solves': self.n_solves, 'time_steps': self.n_time_steps, 'sampled_frequencies': self.n_sampled,
                'basis_columns': self.n_basis_columns, 'operator_applications': self.n_operator_applications,
                'step_seconds': self.total_step_seconds, 'projection_seconds': self.total_projection_seconds,
                'eigen_last_residual': self.last_residual}


def _host(array):
    """Host numpy of a numpy / cupy array."""
    return array.get() if hasattr(array, 'get') else np.asarray(array)


def time_estimate(values, work):
    """Run-time recipe (core.time_estimate.SolverWork; work.unknowns = grid cells, work.factorizations = time steps
    of one run (the physics' run_periods / dt count: no factor here), work.solves = eigen solves, work.rhs = sampled
    frequencies, work.eigenpairs = the wanted (polished) Ritz pairs, wanted_pairs): the FDTD steps (elementwise passes
    of the fields, the DFT accumulation per sample), then per basis column (sampled + polish_steps x wanted, counter
    basis_columns) its operator products (one per projection: A Q is not kept), its orthonormalisation and its share
    of the projections and Ritz vectors (dense n x columns per column and projection)."""
    from ..core import time_estimate as te
    cells = float(work.unknowns)
    solves = float(work.solves or 1)
    steps = float(work.factorizations) * solves
    sampled = float(work.rhs or MIN_SAMPLED_FREQUENCIES)
    value_bytes = 4.0 if values.get('precision', DEFAULT_PRECISION) == 'single' else 8.0
    polish = float(values.get('polish_steps', DEFAULT_POLISH_STEPS))
    step_ops = (te.Elementwise(cells * value_bytes * 12.0, cells * value_bytes * 3.0, launches=12),
                te.Elementwise(cells * sampled * 16.0, cells * sampled * 16.0, launches=2))   # complex128 DFT sums
    columns = sampled + polish * float(work.eigenpairs or wanted_pairs(1))
    # per column: A q (CSR of work.nnz complex entries), QR / Gram-Schmidt against the basis, the projected rows and
    # its Ritz vector / product / residual columns in each of the (polish + 1) projections (complex: 8 real flops)
    column_ops = (te.Spmv(cells, float(work.nnz or 5.0 * cells), 16),) * int(polish + 1.0) + (
                  te.DenseFlops(8.0 * cells * columns * (2.0 + 3.0 * (polish + 1.0))),)
    return te.TimeEstimate((
        te.TimeItem('time-filter FDTD steps', 'step', steps, 'guess', step_ops, counter=work.counter('time_steps')),
        te.TimeItem('time-filter Rayleigh-Ritz', 'basis column', columns * solves, 'guess', column_ops,
                    counter=work.counter('basis_columns'))))


def periods_steps(periods, frequency, dt) -> int:
    """Time steps of `periods` periods of `frequency` [Hz] at step dt [s]."""
    return int(math.ceil(float(periods) / (float(frequency) * float(dt))))
