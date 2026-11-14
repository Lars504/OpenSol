"""'time_filter' eigen-solver runtime: resonances of an open or lossy frequency-domain operator without a
factorisation. The physics time-steps the same structure (FDTD) from a few broadband sources, accumulates a running DFT
of the whole grid at sampled frequencies of a band around the target (one complex accumulator per sampled frequency:
each one a filter (omega_j - H)^-1-like combination of the modes there), and this module's Rayleigh-Ritz step projects
the frequency-domain pencil A u = mu M u onto the span of those fields (plus polish steps that append the wanted Ritz
pairs' residuals), solves the small dense eigenproblem and reports residuals by sparse products only. numpy at module
level; cupy / scipy inside functions; the time stepping lives with the physics (physics.em.time_filter_eigen)."""
# Rayleigh-Ritz (dense part, any array module xp): the basis V (n, m) is orthonormalised by a QR and the rank-revealing
# SVD of its R (columns below RANK_TOLERANCE x the largest singular value dropped: sampled frequencies far from every
# mode give nearly parallel fields); with a symmetrizer D (D A, D M complex symmetric) the projection is bilinear,
# A_r = (D Q)^T A Q, M_r = (D Q)^T M Q (the stationary quotient of a complex symmetric pencil: eigenvalue error ~ the
# square of the subspace error), else Galerkin Q^H A Q, Q^H M Q; scipy.linalg.eig on the host (m x m). Polish step:
# append the M^-1-scaled residuals (A u - mu M u) / M of the wanted Ritz pairs (pinned rows 0) to the basis and project
# again: an unpreconditioned Davidson step, cheap (sparse products and m x m dense work) and enough to remove the FDTD
# time-discretisation error ((omega dt)^2 / 24 of the mode frequency) and the PML-region mismatch of the filtered fields
# (FDTD applies its CPML stretch at every omega, the frequency-domain pencil freezes it at the target).
import math

import numpy as np

PRECISIONS = ('single', 'double')   # FDTD stepper precision (fields, coefficients); the projection is float64
DEFAULT_BANDWIDTH = 0.1     # half-width of the sampled band, relative to the target frequency
DEFAULT_SOURCES = 4         # random point sources of the broadband excitation
DEFAULT_POLISH_STEPS = 2
MIN_PERIODS = 100           # run length: at least this many target periods, and RESOLUTION_PERIODS / bandwidth
RESOLUTION_PERIODS = 10.0
MIN_SAMPLED_FREQUENCIES = 12   # sampled frequencies (0 = auto): max(this, 4 x wanted + 8), at most MAX_SAMPLED
SAMPLES_PER_WANTED = 4
MAX_SAMPLED_FREQUENCIES = 64
RANK_TOLERANCE = 1e-10      # basis columns below this x the largest singular value are dropped
PULSE_BANDWIDTH_FACTOR = 3.0   # source pulse relative bandwidth = this x the sampled band half-width (<= 1)
SOURCE_SEED = 31            # deterministic source positions / amplitudes
RESIDUAL_WARNING = 1e-6     # a reported resonance above this relative residual is named in a warning


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
                 polish_steps=DEFAULT_POLISH_STEPS, precision='single', label=None, backend='gpu'):
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

    def ritz_pairs(self, apply_operator, mass, basis, wanted, target, symmetrizer=None, xp=np) -> dict:
        """Rayleigh-Ritz of A u = mu M u on span(basis) (xp (n, m)) with polish_steps polish steps: apply_operator(V)
        -> A V (n, k) of the array module xp, mass = M diagonal (n,) xp (0 on pinned rows), symmetrizer D (n,) xp or
        None (module notes); wanted pairs = the `wanted` finite mu nearest `target` (mu of the target resonance, e.g.
        0 for A = L + k_t^2 M) that each polish step refines. Returns {mu (r,) host, vectors xp (n, r) (every Ritz
        pair, wanted first by |mu - target|), residuals (r,) host relative ||A u - mu M u|| / (||A u|| + |mu| ||M u||),
        columns (basis rank), applications}."""
        import time
        started = time.perf_counter()
        free = mass != 0
        basis = xp.where(free[:, None], basis, 0)
        quotient = self._orthonormal(basis, xp)
        applied = apply_operator(quotient)
        applications = quotient.shape[1]
        mu, vectors, residuals, products = None, None, None, None
        for step in range(self.polish_steps + 1):
            mu, coefficients = self._project(quotient, applied, mass, symmetrizer, xp)
            order = np.argsort(np.where(np.isfinite(mu), np.abs(mu - target), np.inf))
            mu, coefficients = mu[order], coefficients[:, order]
            device_coefficients = xp.asarray(coefficients)
            vectors = quotient @ device_coefficients
            products = applied @ device_coefficients
            residual_vectors = products - xp.asarray(mu)[None, :] * (mass[:, None] * vectors)
            scale = (xp.linalg.norm(products, axis=0)
                     + xp.abs(xp.asarray(mu)) * xp.linalg.norm(mass[:, None] * vectors, axis=0))
            residuals = _host(xp.linalg.norm(residual_vectors, axis=0) / xp.where(scale > 0, scale, 1.0))
            if step == self.polish_steps:
                break
            count = min(int(wanted), int(np.count_nonzero(np.isfinite(mu))))
            if count < 1:
                break
            safe_mass = xp.where(free, mass, 1.0)
            extra = xp.where(free[:, None], residual_vectors[:, :count] / safe_mass[:, None], 0)
            before = xp.linalg.norm(extra, axis=0)
            extra = extra - quotient @ (quotient.conj().T @ extra)   # two Gram-Schmidt passes against Q
            extra = extra - quotient @ (quotient.conj().T @ extra)
            fresh = _host(xp.linalg.norm(extra, axis=0) > RANK_TOLERANCE ** 0.5 * before)   # not in span(Q) already
            added = self._orthonormal(extra[:, xp.asarray(np.flatnonzero(fresh))], xp)
            if added.shape[1] == 0:
                break
            quotient = xp.concatenate([quotient, added], axis=1)
            applied = xp.concatenate([applied, apply_operator(added)], axis=1)
            applications += added.shape[1]
        self.n_operator_applications += applications
        self.n_basis_columns += quotient.shape[1]
        self.total_projection_seconds += time.perf_counter() - started
        finite = np.isfinite(mu)
        return {'mu': mu[finite], 'vectors': vectors[:, xp.asarray(np.flatnonzero(finite))],
                'residuals': residuals[finite], 'columns': int(quotient.shape[1]), 'applications': applications}

    @staticmethod
    def _orthonormal(vectors, xp):
        """Orthonormal columns spanning vectors (n, k) up to RANK_TOLERANCE of the largest singular value of the
        column-normalised vectors (QR, then the SVD of R; zero columns dropped first)."""
        norms = _host(xp.linalg.norm(vectors, axis=0))
        nonzero = np.flatnonzero(norms > 0.0)
        if nonzero.size == 0:
            return vectors[:, :0]
        vectors = vectors[:, xp.asarray(nonzero)] / xp.asarray(norms[nonzero])[None, :]
        quotient, upper = xp.linalg.qr(vectors)
        left, singular, _ = np.linalg.svd(_host(upper))
        keep = singular > RANK_TOLERANCE * float(singular[0])
        return quotient @ xp.asarray(np.ascontiguousarray(left[:, keep]))

    @staticmethod
    def _project(quotient, applied, mass, symmetrizer, xp):
        """(mu (r,) host, coefficients (k, r) host) of the projected pencil (module notes)."""
        from scipy.linalg import eig
        massed = mass[:, None] * quotient
        left = quotient.conj() if symmetrizer is None else symmetrizer[:, None] * quotient
        projected_a = _host(left.T @ applied)
        projected_m = _host(left.T @ massed)
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
    frequencies): the FDTD steps (elementwise passes of the fields, the DFT accumulation per sample), then the
    projection (sparse products and the dense n x m^2 orthonormalisation, priced as products)."""
    from ..core import time_estimate as te
    cells = float(work.unknowns)
    steps = float(work.factorizations) * float(work.solves or 1)
    sampled = float(work.rhs or MIN_SAMPLED_FREQUENCIES)
    value_bytes = 4.0 if values.get('precision', 'single') == 'single' else 8.0
    polish = float(values.get('polish_steps', DEFAULT_POLISH_STEPS))
    step_ops = (te.Elementwise(cells * value_bytes * 12.0, cells * value_bytes * 3.0, launches=12),
                te.Elementwise(cells * sampled * 8.0, cells * sampled * 8.0, launches=2))
    columns = sampled * (1.0 + polish)
    projection = (te.Spmv(cells, 5.0 * cells, 16.0), te.DenseFlops(8.0 * cells * columns * columns))
    return te.TimeEstimate((
        te.TimeItem('time-filter FDTD steps', 'step', steps, 'guess', step_ops, counter=work.counter('time_steps')),
        te.TimeItem('time-filter Rayleigh-Ritz', 'solve', float(work.solves or 1), 'guess', projection,
                    counter=work.counter('basis_columns'))))


def periods_steps(periods, frequency, dt) -> int:
    """Time steps of `periods` periods of `frequency` [Hz] at step dt [s]."""
    return int(math.ceil(float(periods) / (float(frequency) * float(dt))))
