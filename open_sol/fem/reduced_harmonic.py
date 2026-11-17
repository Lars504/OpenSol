"""Reduced-order frequency sweeps of a linear harmonic system A(omega) x = b(omega) on its solved unknowns (experiment
plan A1 / A2): ReducedHarmonicModel = a Galerkin projection on an orthonormal basis grown by greedy moment matching at
expansion frequencies ('adaptive') or from given vectors plus a static correction ('modal'), with a sketched residual
estimate per frequency. Solver-neutral host numpy; the caller supplies the terms and the factorisation."""
# Contract (physics wire it through their own systems, e.g. fem.harmonic.HarmonicSystemBase.plan_reduced):
# - terms [HarmonicTerm(matrix, coefficient)]: A(omega) = sum coefficient(omega) matrix, scipy sparse (n, n) on the
#   solved (free) unknowns, real or complex; loads [HarmonicLoad(vector, coefficient)] or HarmonicLoad(vector_of=f) with
#   f(omega) -> complex (n,) (a non-polynomial load: no moments), b(omega) = their sum. A Dirichlet lift -A_fc(omega) g
#   is one load per term, with that term's coefficient. The vector_of loads are summed and evaluated once per omega
#   (cached with their sketch and projection; full vectors up to VARYING_LOAD_CACHE_BYTES): f depends on omega alone.
# - coefficients: PowerCoefficient(factor, power) = factor omega^power (K: (1, 0), j omega C: (1j, 1), -omega^2 M: (-1,
#   2), a resistive j b / omega: (1j, -1)), or any object with __call__(omega: float) -> complex, derivative(omega,
#   order) and analytic (False: snapshots only, moments 0).
# - factorize(omega) -> a factor of the whole A(omega) with solve_host(rhs (n,)) -> (n,); called once per expansion
#   frequency and once per verification; the model uses a factor only until its next call and never releases it (the
#   caller may refactorise in place: cuDSS plan reuse).
# Moments at omega0 (Taylor coefficients in omega - omega0, scaled by omega0^k): A0 xi_k = beta_k - sum_(j=1..k) alpha_j
# xi_(k-j), alpha_j = A^(j)(omega0) omega0^j / j!, beta_k likewise of b: back-solves on one factor. Basis: block classical
# Gram-Schmidt twice per vector, a vector deflated below DEFLATION_TOLERANCE of its norm. Galerkin: (V^H A(omega) V) y =
# V^H b(omega), x = V y (exact at the expansion frequencies up to round-off). Residual estimate ||S (b - A V y)|| / ||S b||,
# S a sparse sign embedding of SKETCH_ROWS rows (SKETCH_NONZEROS signs per unknown; the identity up to
# EXACT_SKETCH_LIMIT unknowns): stable near sharp peaks (no Gram-matrix cancellation, which floors a Gram estimate at
# ~1e-8 Q) and O(rows k) per frequency; a sketched norm lies within about +-20 % of the true one. The relative error is up
# to cond(A) x the relative residual (near a resonance of quality Q about Q x): the verification solve at the worst
# frequency reports the true error there.
import math
import time
from dataclasses import dataclass

import numpy as np

DEFLATION_TOLERANCE = 1e-10   # a new vector keeping less of its norm after orthogonalisation adds nothing
SKETCH_ROWS = 512
SKETCH_NONZEROS = 8
EXACT_SKETCH_LIMIT = 2048     # up to this many unknowns the residual is exact (identity sketch)
VARYING_LOAD_CACHE_BYTES = 512 * 2 ** 20   # full non-polynomial load vectors kept per frequency up to this total
# a term with max |A - A^H| below this x max |A| is Hermitian: its new reduced row is the conjugate of the new column
# (v^H A V = conj(V^H A v)), saving one sparse product and one (n, k) product per basis vector (real K / M / C)
HERMITIAN_RTOL = 1e-13
OMEGA_CHUNK = 64              # frequencies per batched reduced solve
SWEEP_METHODS = ('adaptive', 'modal')
_SKETCH_SEED = 11
_TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class PowerCoefficient:
    """factor * omega^power (omega [rad/s], any integer power) and its omega derivatives."""
    factor: complex = 1.0
    power: int = 0
    analytic = True   # class attribute: derivatives exist (moments allowed)

    def __call__(self, omega) -> complex:
        return complex(self.factor * float(omega) ** self.power)

    def derivative(self, omega, order) -> complex:
        """d^order / d omega^order of factor omega^power at omega."""
        falling = 1.0
        for step in range(int(order)):
            falling *= self.power - step
        if falling == 0.0:
            return 0.0j
        return complex(self.factor * falling * float(omega) ** (self.power - int(order)))


@dataclass(frozen=True, eq=False)
class HarmonicTerm:
    """coefficient(omega) x matrix: one term of A(omega) (scipy sparse (n, n), real or complex)."""
    matrix: object
    coefficient: object


@dataclass(frozen=True, eq=False)
class HarmonicLoad:
    """coefficient(omega) x vector (complex (n,)), or vector_of(omega) -> complex (n,) (non-polynomial: no moments)."""
    vector: object = None
    coefficient: object = PowerCoefficient()
    vector_of: object = None

    @property
    def analytic(self) -> bool:
        return self.vector_of is None and bool(getattr(self.coefficient, 'analytic', False))

    def value(self, omega) -> np.ndarray:
        if self.vector_of is not None:
            return np.asarray(self.vector_of(float(omega)), dtype=np.complex128).ravel()
        return self.coefficient(omega) * self.vector


class _Sketch:
    """Sparse sign embedding S (rows x n) of the residual estimate, one real CSR matrix; the identity for small n."""

    def __init__(self, n_unknowns):
        self.exact = n_unknowns <= EXACT_SKETCH_LIMIT
        self.rows = n_unknowns if self.exact else SKETCH_ROWS
        if not self.exact:
            from scipy import sparse
            generator = np.random.default_rng(_SKETCH_SEED)
            targets = generator.integers(0, SKETCH_ROWS, size=(SKETCH_NONZEROS, n_unknowns), dtype=np.int32)
            signs = (2 * generator.integers(0, 2, size=(SKETCH_NONZEROS, n_unknowns), dtype=np.int8) - 1)
            columns = np.broadcast_to(np.arange(n_unknowns, dtype=np.int32), targets.shape)
            weights = signs.astype(np.float64).ravel() / math.sqrt(SKETCH_NONZEROS)
            self.matrix = sparse.csr_matrix((weights, (targets.ravel(), columns.ravel())),
                                            shape=(SKETCH_ROWS, n_unknowns))   # duplicate targets sum

    def __call__(self, vector) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.complex128).ravel()
        if self.exact:
            return vector.copy()
        return np.asarray(self.matrix @ vector, dtype=np.complex128)


def _hermitian(matrix) -> bool:
    """True when a scipy sparse matrix equals its conjugate transpose to HERMITIAN_RTOL of its largest entry."""
    largest = float(abs(matrix).max()) if matrix.nnz else 0.0
    if largest == 0.0:
        return True
    difference = (matrix - matrix.conj().T).tocsr()
    return not difference.nnz or float(abs(difference).max()) <= HERMITIAN_RTOL * largest


def _checked_omegas(omegas) -> np.ndarray:
    omegas = np.atleast_1d(np.asarray(omegas, dtype=np.float64)).ravel()
    if not omegas.size:
        raise ValueError('a reduced sweep needs at least one angular frequency')
    if not (np.all(np.isfinite(omegas)) and np.all(omegas > 0.0)):
        raise ValueError('a reduced sweep needs finite angular frequencies > 0 rad/s')
    return omegas


class ReducedHarmonicModel:
    """Galerkin reduced model of A(omega) x = b(omega) (module header contract): grow the basis by expand(omega,
    moments) / add_vectors, then train / evaluate / fit_modal over a frequency list; solve(omega) = V y. info holds the
    last train / evaluate record."""

    def __init__(self, terms, loads, factorize, label=''):
        terms, loads = list(terms), list(loads)
        if not terms:
            raise ValueError('a reduced harmonic model needs at least one operator term')
        n_unknowns = int(terms[0].matrix.shape[0])
        for term in terms:
            if term.matrix.shape != (n_unknowns, n_unknowns):
                raise ValueError(f'reduced harmonic terms need one ({n_unknowns}, {n_unknowns}) shape, got '
                                 f'{term.matrix.shape}')
        self.n_unknowns = n_unknowns
        self.label = label
        self.factorize = factorize
        self.terms = [HarmonicTerm(term.matrix.tocsr(), term.coefficient) for term in terms]
        self._hermitian_terms = [_hermitian(term.matrix) for term in self.terms]   # row = conj(column) in _append
        self.loads = []
        for load in loads:
            if load.vector_of is None:
                vector = np.asarray(load.vector, dtype=np.complex128).ravel()
                if vector.size != n_unknowns:
                    raise ValueError(f'reduced harmonic load needs {n_unknowns} values, got {vector.size}')
                load = HarmonicLoad(vector, load.coefficient)
            self.loads.append(load)
        self.analytic = (all(getattr(term.coefficient, 'analytic', False) for term in self.terms)
                         and all(load.analytic for load in self.loads))
        self._sketch = _Sketch(n_unknowns)
        self._storage = np.zeros((n_unknowns, 0), dtype=np.complex128)
        self.basis_size = 0
        self._reduced_terms = [np.zeros((0, 0), dtype=np.complex128) for _ in self.terms]
        self._sketched_terms = [np.zeros((self._sketch.rows, 0), dtype=np.complex128) for _ in self.terms]
        self._fixed_loads = [index for index, load in enumerate(self.loads) if load.vector_of is None]
        self._reduced_loads = {index: np.zeros(0, dtype=np.complex128) for index in self._fixed_loads}
        self._sketched_loads = {index: self._sketch(self.loads[index].vector) for index in self._fixed_loads}
        self._varying_loads = [load for load in self.loads if load.vector_of is not None]
        self._varying_cache = {}   # omega -> [sum of vector_of loads (n,) or None past the budget, S b_v, V^H b_v]
        self.expansion_omegas = []
        self.factorizations = 0
        self.warnings = []
        self.info = None

    # ---- basis ----
    @property
    def basis(self) -> np.ndarray:
        """Complex (n, k) orthonormal basis V (a view)."""
        return self._storage[:, :self.basis_size]

    def add_vectors(self, vectors) -> int:
        """Append the columns of (n, m) vectors (or one (n,) vector) to the basis; returns how many were kept."""
        vectors = np.asarray(vectors, dtype=np.complex128)
        if vectors.ndim == 1:
            vectors = vectors[:, None]
        if vectors.shape[0] != self.n_unknowns:
            raise ValueError(f'basis vectors need {self.n_unknowns} rows, got {vectors.shape[0]}')
        return sum(self._append(vectors[:, column]) for column in range(vectors.shape[1]))

    def _append(self, vector) -> bool:
        """Orthonormalise vector against V (twice), deflate, extend the projected terms / loads and sketches."""
        vector = np.array(vector, dtype=np.complex128).ravel()
        norm = float(np.linalg.norm(vector))
        if not (math.isfinite(norm) and norm > 0.0):
            return False
        old = self.basis
        for _ in range(2):   # V^H w as conj(conj(w) @ V): no conjugated copy of V
            if self.basis_size:
                vector -= old @ np.conj(np.conj(vector) @ old)
        remaining = float(np.linalg.norm(vector))
        if remaining <= DEFLATION_TOLERANCE * norm:
            return False
        vector /= remaining
        size = self.basis_size
        if size == self._storage.shape[1]:
            grown = np.zeros((self.n_unknowns, max(16, 2 * size)), dtype=np.complex128)
            grown[:, :size] = self._storage[:, :size]
            self._storage = grown
        self._storage[:, size] = vector
        self.basis_size = size + 1
        basis = self.basis
        for index, term in enumerate(self.terms):
            image = term.matrix @ vector
            reduced = np.zeros((size + 1, size + 1), dtype=np.complex128)
            reduced[:size, :size] = self._reduced_terms[index]
            column = np.conj(np.conj(image) @ basis)                                    # V^H A_s v
            reduced[:, size] = column
            if self._hermitian_terms[index]:                                            # v^H A_s V_old = conj(V_old^H A_s v)
                reduced[size, :size] = np.conj(column[:size])
            else:
                reduced[size, :size] = (term.matrix.T @ np.conj(vector)) @ old        # v^H A_s V_old (A^T: a CSC view)
            self._reduced_terms[index] = reduced
            self._sketched_terms[index] = np.column_stack((self._sketched_terms[index], self._sketch(image)))
        for index in self._fixed_loads:
            self._reduced_loads[index] = np.append(self._reduced_loads[index], np.vdot(vector, self.loads[index].vector))
        return True

    # ---- operator / load evaluation ----
    def load(self, omega) -> np.ndarray:
        """Complex (n,) b(omega)."""
        total = np.zeros(self.n_unknowns, dtype=np.complex128)
        for index in self._fixed_loads:
            total += self.loads[index].value(omega)
        if self._varying_loads:
            varying = self._varying_entry(omega)[0]
            total += varying if varying is not None else self._varying_value(omega)
        return total

    def _varying_value(self, omega) -> np.ndarray:
        """Complex (n,) sum of the vector_of loads at omega (evaluated, never cached)."""
        total = np.zeros(self.n_unknowns, dtype=np.complex128)
        for load in self._varying_loads:
            total += load.value(omega)
        return total

    def _varying_entry(self, omega) -> list:
        """[b_v (n,) or None, S b_v (rows,), V^H b_v (k,)] of the vector_of loads at omega: evaluated once per omega
        (vector_of must depend on omega alone while the model lives), the projection extended as the basis grows."""
        omega = float(omega)
        entry = self._varying_cache.get(omega)
        vector = None
        if entry is None:
            vector = self._varying_value(omega)
            within_budget = (len(self._varying_cache) + 1) * vector.nbytes <= VARYING_LOAD_CACHE_BYTES
            entry = self._varying_cache[omega] = [vector if within_budget else None, self._sketch(vector),
                                                  np.zeros(0, dtype=np.complex128)]
        projected = entry[2]
        if projected.size < self.basis_size:
            if vector is None:
                vector = entry[0] if entry[0] is not None else self._varying_value(omega)
            new_columns = self.basis[:, projected.size:]
            entry[2] = np.concatenate((projected, np.conj(np.conj(vector) @ new_columns)))
        return entry

    def _apply_derivative(self, omega, order, scale, vector) -> np.ndarray:
        """alpha_order x vector: sum_s c_s^(order)(omega) scale^order / order! A_s vector."""
        weight = scale ** order / math.factorial(order)
        total = np.zeros(self.n_unknowns, dtype=np.complex128)
        for term in self.terms:
            coefficient = term.coefficient.derivative(omega, order) * weight
            if coefficient != 0.0:
                total += coefficient * (term.matrix @ vector)
        return total

    def _load_derivative(self, omega, order, scale) -> np.ndarray:
        """beta_order: sum_q d_q^(order)(omega) scale^order / order! b_q (order 0: b(omega))."""
        if order == 0:
            return self.load(omega)
        weight = scale ** order / math.factorial(order)
        total = np.zeros(self.n_unknowns, dtype=np.complex128)
        for load in self.loads:
            coefficient = load.coefficient.derivative(omega, order) * weight
            if coefficient != 0.0:
                total += coefficient * load.vector
        return total

    def expand(self, omega, moments=0, per_load=False) -> int:
        """Factorise A(omega) and add x(omega) plus `moments` Taylor moment vectors (per_load: A^-1 b_q of every load
        instead, the modal static correction); returns the number of vectors kept."""
        omega = float(_checked_omegas([omega])[0])
        moments = int(moments) if self.analytic else 0
        if moments < 0:
            raise ValueError(f'moments must be >= 0, got {moments}')
        factor = self.factorize(omega)
        self.factorizations += 1
        self.expansion_omegas.append(omega)
        if per_load:
            return sum(self._append(factor.solve_host(load.value(omega))) for load in self.loads)
        added, moment_vectors = 0, []
        for order in range(moments + 1):
            rhs = self._load_derivative(omega, order, omega)
            for step in range(1, order + 1):
                rhs -= self._apply_derivative(omega, step, omega, moment_vectors[order - step])
            moment = np.asarray(factor.solve_host(rhs), dtype=np.complex128)
            moment_vectors.append(moment)
            added += self._append(moment)
        return added

    # ---- reduced solves ----
    def _coefficients(self, coefficient, omegas) -> np.ndarray:
        return np.array([complex(coefficient(float(omega))) for omega in omegas], dtype=np.complex128)

    def _reduced_rhs(self, omegas) -> np.ndarray:
        """Complex (m, k) V^H b(omega) per frequency."""
        return self._load_rows(omegas, self.basis_size, self._reduced_loads, 2)

    def _sketched_rhs(self, omegas) -> np.ndarray:
        """Complex (m, rows) S b(omega) per frequency."""
        return self._load_rows(omegas, self._sketch.rows, self._sketched_loads, 1)

    def _load_rows(self, omegas, width, fixed_rows, entry) -> np.ndarray:
        """Complex (m, width) sum of the fixed loads' rows fixed_rows[index] x coefficient(omega) and of item `entry`
        of the vector_of loads' _varying_entry, per frequency."""
        rows = np.zeros((len(omegas), width), dtype=np.complex128)
        for index in self._fixed_loads:
            load = self.loads[index]
            rows += self._coefficients(load.coefficient, omegas)[:, None] * fixed_rows[index][None, :]
        if self._varying_loads:
            for row, omega in enumerate(omegas):
                rows[row] += self._varying_entry(omega)[entry]
        return rows

    def reduced_coordinates(self, omegas) -> np.ndarray:
        """Complex (m, k) Galerkin coordinates y at angular frequencies omegas [rad/s] (least squares where the reduced
        operator is singular)."""
        omegas = _checked_omegas(omegas)
        size = self.basis_size
        coordinates = np.zeros((omegas.size, size), dtype=np.complex128)
        if not size:
            return coordinates
        for start in range(0, omegas.size, OMEGA_CHUNK):
            chunk = omegas[start:start + OMEGA_CHUNK]
            operator = np.zeros((chunk.size, size, size), dtype=np.complex128)
            for term, reduced in zip(self.terms, self._reduced_terms):
                operator += self._coefficients(term.coefficient, chunk)[:, None, None] * reduced[None, :, :]
            rhs = self._reduced_rhs(chunk)
            try:
                coordinates[start:start + chunk.size] = np.linalg.solve(operator, rhs[..., None])[..., 0]
            except np.linalg.LinAlgError:
                for row in range(chunk.size):
                    coordinates[start + row] = np.linalg.lstsq(operator[row], rhs[row], rcond=None)[0]
        return coordinates

    def solve(self, omega) -> np.ndarray:
        """Complex (n,) reduced solution V y at angular frequency omega [rad/s]."""
        return self.basis @ self.reduced_coordinates([omega])[0]

    def residuals(self, omegas, coordinates=None) -> np.ndarray:
        """Float64 (m,) estimated relative residual ||b - A V y|| / ||b|| per angular frequency (0 where b = 0)."""
        omegas = _checked_omegas(omegas)
        if coordinates is None:
            coordinates = self.reduced_coordinates(omegas)
        result = np.zeros(omegas.size, dtype=np.float64)
        for start in range(0, omegas.size, OMEGA_CHUNK):
            chunk = omegas[start:start + OMEGA_CHUNK]
            sketched_load = self._sketched_rhs(chunk)                   # (m, rows)
            residual = -sketched_load
            block = coordinates[start:start + chunk.size]               # (m, k)
            if self.basis_size:
                for term, sketched in zip(self.terms, self._sketched_terms):
                    residual += self._coefficients(term.coefficient, chunk)[:, None] * (block @ sketched.T)
            load_norm = np.linalg.norm(sketched_load, axis=1)
            residual_norm = np.linalg.norm(residual, axis=1)
            result[start:start + chunk.size] = np.where(load_norm > 0.0, residual_norm / np.where(
                load_norm > 0.0, load_norm, 1.0), 0.0)
        return result

    def verify(self, omega) -> float:
        """True relative error ||V y - x|| / ||x|| at omega from one direct factorisation and solve (0 when x = 0)."""
        factor = self.factorize(float(omega))
        self.factorizations += 1
        exact = np.asarray(factor.solve_host(self.load(omega)), dtype=np.complex128)
        scale = float(np.linalg.norm(exact))
        if scale == 0.0:
            return 0.0
        return float(np.linalg.norm(self.solve(omega) - exact)) / scale

    # ---- sweeps ----
    def train(self, omegas, tolerance=1e-6, moments=4, max_points=40, verify=True) -> dict:
        """Greedy adaptive basis over the angular frequencies omegas [rad/s]: expansions at the ends and the middle,
        then at the worst estimated residual until it is <= tolerance or max_points expansions; returns info."""
        started = time.perf_counter()
        omegas = _checked_omegas(omegas)
        tolerance, moments, max_points = float(tolerance), int(moments), int(max_points)
        if not tolerance > 0.0 or moments < 0 or max_points < 1:
            raise ValueError(f'reduced sweep needs tolerance > 0, moments >= 0 and max_points >= 1, got {tolerance}, '
                             f'{moments}, {max_points}')
        if moments and not self.analytic:
            self.warnings.append('non-polynomial operator or load terms: expansions add snapshots only (moments 0)')
        middle = omegas[int(np.argmin(np.abs(omegas - 0.5 * (omegas.min() + omegas.max()))))]
        for omega in list(dict.fromkeys((float(omegas.min()), float(omegas.max()), float(middle))))[:max_points]:
            self.expand(omega, moments)
        converged = False
        while True:
            residuals = self.residuals(omegas)
            worst = int(np.argmax(residuals))
            if residuals[worst] <= tolerance:
                converged = True
                break
            if len(self.expansion_omegas) >= max_points:
                self.warnings.append(f'{max_points} expansion frequencies reached with an estimated relative residual '
                                     f'{residuals[worst]:.3g} > {tolerance:g}: raise rom_max_points or rom_moments')
                break
            if float(omegas[worst]) in self.expansion_omegas:
                self.warnings.append(f'the worst residual {residuals[worst]:.3g} sits on an expansion frequency (round-off '
                                     'limit of the reduced model): training stopped')
                break
            if not self.expand(omegas[worst], moments):
                self.warnings.append(f'the expansion at {omegas[worst] / _TWO_PI:.6g} Hz added no new direction: '
                                     'training stopped')
                break
        return self._finish('adaptive', omegas, residuals, tolerance, converged, verify, started)

    def evaluate(self, omegas, tolerance=1e-6, verify=True, method='modal') -> dict:
        """info of the current basis over omegas [rad/s] (no training): estimated residuals, converged = max <= tolerance,
        optional verification at the worst one."""
        started = time.perf_counter()
        omegas = _checked_omegas(omegas)
        residuals = self.residuals(omegas)
        converged = bool(residuals.max() <= float(tolerance))
        if not converged:
            self.warnings.append(f'estimated relative residual {residuals.max():.3g} > {float(tolerance):g}: the basis '
                                 'misses part of the response (more modes, or sweep_method adaptive)')
        return self._finish(method, omegas, residuals, float(tolerance), converged, verify, started)

    def fit_modal(self, mode_vectors, omegas, tolerance=1e-6, verify=True) -> dict:
        """Modal basis (plan A2): the (n, m) mode vectors plus A(omega_c)^-1 b_q of every load at the band centre omega_c
        (the static correction of the truncated modes), then evaluate over omegas [rad/s]."""
        omegas = _checked_omegas(omegas)
        kept = self.add_vectors(mode_vectors)
        if kept < np.asarray(mode_vectors).reshape(self.n_unknowns, -1).shape[1]:
            self.warnings.append(f'{kept} of the mode vectors kept (the others were linearly dependent)')
        self.expand(0.5 * (omegas.min() + omegas.max()), per_load=True)
        return self.evaluate(omegas, tolerance=tolerance, verify=verify, method='modal')

    def _finish(self, method, omegas, residuals, tolerance, converged, verify, started) -> dict:
        worst = int(np.argmax(residuals))
        verified_error = self.verify(omegas[worst]) if verify and self.basis_size else math.nan
        self.info = {'method': method, 'tolerance': tolerance, 'converged': bool(converged),
                     'basis_size': int(self.basis_size), 'factorizations': int(self.factorizations),
                     'expansion_frequencies': [omega / _TWO_PI for omega in self.expansion_omegas],
                     'n_frequencies': int(omegas.size), 'max_residual': float(residuals[worst]),
                     'worst_frequency': float(omegas[worst] / _TWO_PI), 'verified_error': float(verified_error),
                     'residual_sketch': 'exact' if self._sketch.exact else f'{self._sketch.rows} rows',
                     'seconds': time.perf_counter() - started, 'warnings': list(self.warnings)}
        return self.info


def sweep_text(info) -> str:
    """One log line of a reduced-sweep info dict."""
    method = info.get('method', '')
    expansions = info.get('expansion_frequencies') or []
    modes = f", {info['modes']} modes" if 'modes' in info else ''
    verified = info.get('verified_error', math.nan)
    verified_text = ('' if not math.isfinite(verified)
                     else f", verified error {verified:.3g} at {info['worst_frequency']:.6g} Hz")
    return (f"{method} reduced model{modes}: basis {info.get('basis_size', 0)}, {info.get('factorizations', 0)} "
            f"factorisations ({len(expansions)} expansion frequencies), max estimated residual "
            f"{info.get('max_residual', math.nan):.3g} (tolerance {info.get('tolerance', math.nan):g}){verified_text}, "
            f"{info.get('seconds', 0.0):.3g} s")
