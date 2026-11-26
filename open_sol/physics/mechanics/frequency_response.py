"""Driven time-harmonic response of the solid (frequency_response study): the
reduced pencil of (K + j L - omega^2 M) x = f with one complex symmetric factor per frequency (HarmonicSystem:
private host SuperLU, or a sparse_direct runtime's SuperLU / cuDSS LDL^T with plan reuse; or a fem.reduced_harmonic
model with precomputed record rows), the per-frequency electrode record and the admittance result of an electrode (with
its BVD fit and kappa_e); host numpy (scipy inside functions)."""
# Per-frequency terms (FrequencyTerm: coefficient(omega) X with a fixed X, e.g. a viscoelastic modulus group) join
# the pencil at fixed slots of its summed pattern, mapped once (factorize_pencil(extra=) on a pencil, the summed values
# on the host route), never as pencil terms of their own (no per-term device storage) and never re-assembled. Scale
# terms (SchurScaleTerm: the coupling B of mixed pressure rows) only add their Schur proxy to the diagonal d of the
# symmetric equilibration R = d^-1/2 the factorisations apply (E = R A R, x = R E^-1 R b), on every route, host SuperLU
# included (_factorize): A, its products, residual rows and every solution handed out stay unscaled.
# Convention exp(+j omega t) (as the EM frequency_domain study): losses enter as c -> c (1 + j eta) (mechanical loss
# factor) and eps -> eps (1 - j tan delta) (dielectric loss tangent), so L = [[eta K_uu, 0], [0, +tan delta K_phiphi]] on
# the coupled operator [[K_uu, K_uphi], [K_phiu, -K_phiphi]] (piezo.coupled_operator). Every load, prescribed
# displacement, electrode voltage and floating charge is the complex amplitude (phase 0) of the excitation. An electrode
# carries Q = -sum of its phi rows of A x (the discrete Gauss law of the complex D: displacement and loss current), its
# current is I = j omega Q and its admittance Y = I / V [S per m of depth; times the thickness setting in S].
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ...core.backend import array_module_of, to_host
from ...fem.assembly import congruence, pattern_sum
from .bvd import bvd_admittance, bvd_fit, external_coupling

ADMITTANCE_SERIES = ('abs', 're', 'im', 'phase', 'bvd_abs')
_PHASE_DEGREES = 180.0 / math.pi
# a support's reaction record component W = sum R u_held over its held dofs (the work of the support force on the
# prescribed displacement: torque x theta of a rigid rotation, omega Im(W) / 2 the mean power into the solid); its unit
# follows the record's force unit
REACTION_WORK = 'W'
_WORK_UNITS = {'N': 'J', 'N/m': 'J/m'}


@dataclass(frozen=True, eq=False)
class FrequencyTerm:
    """A host CSR X over every dof (the operator's numbering, before the column map; real, or complex symmetric where an
    elastic PML stretches it: the triangle solids' ring, like K and M then) entering A(omega) as
    coefficient(omega) X (complex): an omega-dependent modulus group (the excess G*_g - G0 (1 + j eta) on D_g, the
    displacement formulation; G*_g D_g in the mixed one) or a mixed pressure block (-1 / K*_g(omega) M_p,g)."""
    matrix: object
    coefficient: Callable[[float], complex]
    label: str = ''


@dataclass(frozen=True, eq=False)
class SchurScaleTerm:
    """Row-scale estimate only (never in A): the constraint block of a saddle point as a real host CSR (n, n) over every
    dof, B at its constrained rows (the mixed pressure rows) and the columns they couple (the displacements), nothing
    else stored. Per factorisation each constrained row i adds the diagonal Schur proxy s_i = sum_j B_ij^2 / d_j to the
    equilibration diagonal, d_j that of its coupled free column at omega (|diag K| + omega^2 |diag M| + sum_k |c_k|
    |diag X_k|), so every scaled row of R B R has sum_j |(R B R)_ij|^2 <= 1 whatever the mesh size, element shape or
    the viscous / inertial balance of d_j (a fixed modulus weight such as diag M_p / |G*| matches s_i only where the
    shear term dominates d_j: inertia-dominated cells, h >> the viscous depth, put it (h / delta)^2 too high)."""
    coupling: object


class HarmonicSystem:
    """The pencil (operator, loss, mass) of one assembly reduced once to the free dofs (G^T X G with G = column_map, None
    = identity) for the frequency solves; each solve factorises the complex symmetric operator + j loss - omega^2 mass
    of the free dofs: on a cuDSS back end from the term values kept on the device (sparse_direct.DevicePencil,
    factorize_pencil: LDL^T, the previous factor's plan reused), on MUMPS / PARDISO from host term values
    (sparse_direct.HostPencil, the analysis reused, no per-frequency asymmetry check), else by the runtime's
    factorize_host (host SuperLU; counted there) or, runtime None, a private host SuperLU. The summed pattern never depends on the
    values (fem.assembly.pattern_sum), so the cuDSS plan holds across frequencies. loads = [(reduced free dof, b)]: + j b
    / omega on that diagonal (a resistive electrode load, piezo.PiezoSystem). Direct solves with fields False after
    set_output_rows (no column map): x and A x only at those rows, the free solution kept (on the device after a cuDSS
    solve) for last_solution (R6). terms: FrequencyTerm list added as c_k(omega) X_k at slots of the summed pattern
    (module header; reduced sweeps refuse them); scale_terms: SchurScaleTerm list of the equilibration diagonal (also
    applied by host SuperLU: _factorize). Both empty: exactly the pencil above. set_functionals(W): every solve also
    gives last_functionals = W^H x, and solve_columns solves extra right-hand sides on the factor of the last direct
    solve (physics drives: one factorisation per frequency for every scan column); without W every path is unchanged."""

    def __init__(self, operator, loss, mass, free, column_map=None, runtime=None, previous=None, loads=(), terms=(),
                 scale_terms=()):
        import scipy.sparse
        self.operator, self.loss, self.mass = operator.tocsr(), None if loss is None else loss.tocsr(), mass.tocsr()
        self.column_map = column_map
        self.free_dofs = np.flatnonzero(np.asarray(free, dtype=bool))
        self._reduced = [self._reduced_free(matrix) for matrix in (self.operator, self.loss, self.mass)]
        self._terms = tuple(terms)
        self._term_matrices = tuple(term.matrix.tocsr() for term in self._terms)   # every dof (products, rows)
        self._reduced_terms = [self._reduced_free(matrix) for matrix in self._term_matrices]
        self._scale_terms = tuple(scale_terms)
        self._load = None
        if loads:
            position = np.full(int(self.operator.shape[0]), -1, dtype=np.int64)
            position[self.free_dofs] = np.arange(self.free_dofs.size)
            rows = position[np.array([dof for dof, _ in loads], dtype=np.int64)]
            if np.any(rows < 0):
                raise ValueError('a resistively loaded electrode has no free potential (every node held or inactive)')
            n_free = int(self.free_dofs.size)
            self._load = scipy.sparse.csr_matrix((np.array([value for _, value in loads], dtype=np.float64),
                                                  (rows, rows)), shape=(n_free, n_free))
        # the summed pattern once (pattern_sum's canonical CSR order depends on the patterns only): each term's values
        # aligned on it, so a frequency only combines value arrays (no COO round trip per solve)
        stiffness, loss, mass = self._reduced
        terms = {name: term for name, term in (('stiffness', stiffness), ('loss', loss), ('mass', mass),
                                               ('load', self._load)) if term is not None}
        zeros = {name: term * 0.0 for name, term in terms.items()}
        # the frequency terms' patterns join the summed one (their zeros); each term's (positions, values) on it once
        self._pattern = pattern_sum(list(zeros.values()) + [term * 0.0 for term in self._reduced_terms])
        # booked (estimate): per frequency-term entry int64 position + float64 value (16 B), host
        self._term_entries = [_pattern_positions(self._pattern, term) for term in self._reduced_terms]
        if self._reduced_terms:   # the pencil terms scattered on the united pattern (one search per term)
            aligned = {}
            for name, term in terms.items():
                positions, values = _pattern_positions(self._pattern, term)
                aligned[name] = np.zeros(self._pattern.nnz, dtype=values.dtype)
                aligned[name][positions] = values
        else:
            aligned = {name: pattern_sum([term if other == name else zeros[other] for other in terms]).data
                       for name, term in terms.items()}
        self._fixed_values = aligned['stiffness'].astype(np.complex128)
        if 'loss' in aligned:
            self._fixed_values += 1j * aligned['loss']
        self._mass_values = aligned['mass']
        self._load_values = aligned.get('load')
        # |diag K|, |diag M| of the free dofs: the row scale of every cuDSS factorisation (computed once)
        self._diagonals = (np.abs(stiffness.diagonal()), np.abs(mass.diagonal()))
        # per frequency term |diag X_k| and per Schur scale term B_ij^2 on the free dofs (equilibration; booked
        # (estimate): definition_viscoelastic 'mixed Schur row scale')
        self._term_diagonals = [np.abs(term.diagonal()) for term in self._reduced_terms]
        self._schur_squares = [_squared_entries(self._reduced_free(term.coupling)) for term in self._scale_terms]
        self._term_slots = None  # pencil route: (union slots U, [(index into U, values)] per term), mapped once
        self.runtime = runtime
        self._pencil = None      # sparse_direct DevicePencil (cuDSS) / HostPencil (MUMPS / PARDISO) of the terms, False
        #                          for the host route (made on the first factorize)
        self.factor = previous   # the last factor (a CudssFactorization is refactorised in place)
        self.factorizations = 0  # factorisations made (direct solves, reduced-model expansions and verification)
        self._planned = None     # (ReducedHarmonicModel, fixed, load, extra_rhs) of plan_reduced, or None
        self._rows = None        # fem.reduced_outputs.ReducedRows of set_output_rows, or None
        self._direct_rows = None   # (rows, columns, operator / mass / loss row blocks on columns) of direct solves
        self._kept = None        # (free solution, fixed) of the last direct rows-only solve (last_solution)
        self._fixed_terms = None   # (fixed, (K fixed, M fixed, L fixed) or None when fixed is 0) of _fixed_product
        self._free_position = np.full(self._n_reduced(), -1, dtype=np.int64)   # dof -> free index, -1 held
        self._free_position[self.free_dofs] = np.arange(self.free_dofs.size)
        # functionals W^H x (set_functionals; physics drives: the FDT readouts of brillouin_scattering); every branch
        # reading them is guarded by `self._functionals is not None`, so without them every solve is bit-identical
        self._functionals = None          # (k, n) complex host weights W over every dof, read-only (the caller's)
        self._device_functionals = None   # W at the free dofs on the device (cuDSS rows-only / column solves; booked)
        self._functional_block = None     # (model, conj(W P) V (k, basis), W^H fixed) of the reduced model
        self._factored_omega = None       # omega [rad/s] of the current factor (_factorize)
        self.last_functionals = None      # complex (k,) W^H x of the last solve (with functionals)

    def _reduced_free(self, matrix):
        if matrix is None:
            return None
        reduced = matrix if self.column_map is None else congruence(self.column_map, matrix)
        return reduced.tocsr()[self.free_dofs][:, self.free_dofs].tocsr()

    def _term_coefficients(self, omega) -> list:
        """c_k(omega) of the frequency terms (complex)."""
        return [complex(term.coefficient(omega)) for term in self._terms]

    def _product(self, omega, vector):
        """(operator + j loss - omega^2 mass + sum_k c_k X_k) vector over every dof."""
        product = _real_times(self.operator, vector) - (omega * omega) * _real_times(self.mass, vector)
        product = product if self.loss is None else product + 1j * _real_times(self.loss, vector)
        for coefficient, matrix in zip(self._term_coefficients(omega), self._term_matrices):
            product = product + coefficient * _real_times(matrix, vector)
        return product

    def _fixed_product(self, omega, fixed):
        """A(omega) fixed from K / M / L (and the frequency terms) times fixed, kept while the held values stay equal
        (0.0 when they are all 0)."""
        if self._fixed_terms is None or not np.array_equal(self._fixed_terms[0], fixed):
            terms = None if not fixed.any() else tuple(None if matrix is None else _real_times(matrix, fixed)
                                                       for matrix in (self.operator, self.mass, self.loss)
                                                       + self._term_matrices)
            self._fixed_terms = (fixed.copy(), terms)
        terms = self._fixed_terms[1]
        if terms is None:
            return 0.0
        operator_fixed, mass_fixed, loss_fixed = terms[:3]
        product = operator_fixed - (omega * omega) * mass_fixed
        product = product if loss_fixed is None else product + 1j * loss_fixed
        for coefficient, term_fixed in zip(self._term_coefficients(omega), terms[3:]):
            product = product + coefficient * term_fixed
        return product

    def _rows_product(self, omega, values):
        """A x over every dof, exact at the direct output rows and 0 elsewhere, from x at the row blocks' columns."""
        rows, _, (operator_rows, mass_rows, loss_rows, *term_rows) = self._direct_rows
        product = operator_rows @ values - (omega * omega) * (mass_rows @ values)
        if loss_rows is not None:
            product = product + 1j * (loss_rows @ values)
        for coefficient, block in zip(self._term_coefficients(omega), term_rows):
            product = product + coefficient * (block @ values)
        applied = np.zeros(self._n_reduced(), dtype=np.complex128)
        applied[rows] = product
        return applied

    def _equilibration_diagonal(self, omega) -> np.ndarray:
        """d = |diag K| + omega^2 |diag M| + sum_k |c_k| |diag X_k| over the free dofs, + on the constrained rows of
        each SchurScaleTerm its proxy sum_j B_ij^2 / d_j (one product per term; frequency or scale terms present);
        ValueError naming the dofs where d <= 0 (a pressure row coupled to no free displacement, without compliance)."""
        stiffness_diagonal, mass_diagonal = self._diagonals
        diagonal = stiffness_diagonal + (omega * omega) * mass_diagonal
        for coefficient, term_diagonal in zip(self._term_coefficients(omega), self._term_diagonals):
            diagonal = diagonal + abs(coefficient) * term_diagonal
        if self._schur_squares:   # B has no entry in a constrained column: the proxy reads the coupled columns only
            inverse = np.divide(1.0, diagonal, out=np.zeros_like(diagonal), where=diagonal > 0.0)
            diagonal = diagonal + sum(square @ inverse for square in self._schur_squares)
        bad = np.flatnonzero(~(diagonal > 0.0))
        if bad.size:
            raise ValueError(f'the harmonic operator has {bad.size} free row(s) without a positive equilibration '
                             f'diagonal (first free indices {bad[:5].tolist()}): a row with no stiffness, mass or scale '
                             'term (a pressure row needs its SchurScaleTerm and a free displacement to couple to)')
        return diagonal

    def _pencil_extra(self, omega):
        """(slots, values) of the frequency terms at omega in the pencil's stored layout (lower entries of a
        lower-stored one), the slots mapped once: union U of the terms' slots, each term's indices into U (unique within
        a term, so a fancy += sums a term once)."""
        if self._term_slots is None:
            pencil = self._pencil
            rows_of = np.repeat(np.arange(self._pattern.shape[0], dtype=np.int64), np.diff(self._pattern.indptr))
            columns_of = self._pattern.indices.astype(np.int64)
            mapped = []
            for positions, values in self._term_entries:
                rows, columns = rows_of[positions], columns_of[positions]
                if pencil.lower_stored:
                    keep = columns <= rows
                    rows, columns, values = rows[keep], columns[keep], values[keep]
                slots = pencil.slots(rows, columns)
                if slots is None:
                    raise RuntimeError('a frequency term entry is not in the pencil pattern (a bug: the summed '
                                       'pattern holds every term)')
                mapped.append((np.asarray(slots, dtype=np.int64), values))
            union = np.unique(np.concatenate([slots for slots, _ in mapped]))
            # booked (estimate): the union slots (int64) + per term index (int64) and kept values (float64), host
            self._term_slots = (union, [(np.searchsorted(union, slots), values) for slots, values in mapped])
        union, mapped = self._term_slots
        # booked (estimate): one complex128 value per union slot per frequency (uploaded with the slots on cuDSS)
        extra = np.zeros(union.size, dtype=np.complex128)
        for coefficient, (index, values) in zip(self._term_coefficients(omega), mapped):
            extra[index] += coefficient * values
        return union, extra

    def _factorize(self, omega):
        """Factor of the free-dof operator + j loss - omega^2 mass (+ j b / omega) (+ sum_k c_k X_k) at omega [rad/s]
        (counted)."""
        import scipy.sparse
        from ...solvers.sparse_direct import SparseDirectSolver, factorize_host
        coefficients = [1.0, -omega * omega] + ([] if self._load_values is None else [1j / omega])
        if self._pencil is None:
            terms = [self._fixed_values, self._mass_values] + ([] if self._load_values is None else [self._load_values])
            self._pencil = isinstance(self.runtime, SparseDirectSolver) and self.runtime.pencil(
                self._pattern.indptr, self._pattern.indices, terms, symmetric=True) or False
        # the scale of diag(K) + omega^2 diag(M) never cancels near a resonance (cuDSS equilibration)
        stiffness_diagonal, mass_diagonal = self._diagonals
        # With pressure rows (scale terms: the mixed saddle point) host SuperLU factors R A R too (scale_superlu, the
        # private one likewise; R the one every back end gets): its symmetric mode keeps a diagonal pivot only while
        # |a_jj| >= the pivot threshold x its column max, a comparison across units (N/m displacement diagonals, the
        # coupling B in m^2 in 3D, the pressure diagonal in m^3/Pa) that is meaningless unscaled
        # (examples/mechanics/bead_drag_water_frequency_response: 8915 row swaps, fill 105x, 53 s per factor, backward
        # error 1.7e-7; scaled: fill 5.2x, 0.37 s, 3.6e-16). Rows of one unit (every other system) keep SuperLU
        # unscaled, bit-identical.
        saddle = bool(self._scale_terms)
        if self._terms or saddle:   # + |c_k| |diag X_k| + the Schur proxy of the pressure rows (symmetric R A R)
            scale = (None if self.runtime is None and not saddle
                     else 1.0 / np.sqrt(self._equilibration_diagonal(omega)))
        else:
            scale = None if self.runtime is None else 1.0 / np.sqrt(stiffness_diagonal + (omega * omega) * mass_diagonal)
        if self._pencil:   # cuDSS: values on the device; MUMPS / PARDISO: host values on the kept analysis
            extra = self._pencil_extra(omega) if self._terms else None
            self.factor = self.runtime.factorize_pencil(self._pencil, coefficients, row_scale=scale,
                                                        previous=self.factor, extra=extra, scale_superlu=saddle)
        else:
            values = self._fixed_values - (omega * omega) * self._mass_values
            if self._load_values is not None:
                values += (1j / omega) * self._load_values
            for coefficient, (positions, term_values) in zip(self._term_coefficients(omega), self._term_entries):
                values[positions] += coefficient * term_values   # unique positions within a term
            matrix = scipy.sparse.csr_matrix((values, self._pattern.indices, self._pattern.indptr),
                                             shape=self._pattern.shape)
            if self.runtime is None:   # the private SuperLU: scaled with pressure rows only
                self.factor = factorize_host(matrix, row_scale=scale)
            else:
                self.factor = self.runtime.factorize_host(matrix, symmetric=True, row_scale=scale,
                                                          previous=self.factor, scale_superlu=saddle)
        self.factorizations += 1
        self._factored_omega = omega   # solve_columns reuses this factor only at this omega
        return self.factor

    def _n_reduced(self) -> int:
        """Unknowns of x_r (every dof without a column map, else its columns)."""
        return int(self.operator.shape[0] if self.column_map is None else self.column_map.shape[1])

    def _full(self, reduced_free_values, fixed) -> np.ndarray:
        """x = G x_r + fixed over every dof of free-dof values x_r."""
        reduced_solution = np.zeros(self._n_reduced(), dtype=np.complex128)
        reduced_solution[self.free_dofs] = reduced_free_values
        return (reduced_solution if self.column_map is None else self.column_map @ reduced_solution) + fixed

    def solve(self, omega, fixed, load, extra_rhs=(), fields=True, product_at_rows=False):
        """(x, A x): complex host vectors over every dof of (operator + j loss - omega^2 mass) x = load with x = G x_r +
        fixed, x_r solved on the free dofs; extra_rhs = [(reduced dof, value)] subtracted from the reduced rhs (a floating
        electrode's charge). After plan_reduced: x_r = V y of the model (fields False: x, A x exact only at the
        set_output_rows rows, 0 elsewhere: no V y). product_at_rows (after set_output_rows): A x exact only at those
        rows, 0 elsewhere, also with fields (no full products per frequency; without row blocks the full A x)."""
        fixed = np.asarray(fixed, dtype=np.complex128)
        if self._planned is not None:
            return self._reduced_solve(omega, fixed, load, extra_rhs, fields, product_at_rows)
        rhs = np.asarray(load, dtype=np.complex128) - self._fixed_product(omega, fixed)
        if self.column_map is not None:
            rhs = self.column_map.T @ rhs
        for dof, value in extra_rhs:
            rhs[dof] -= value
        self._factorize(omega)
        if not fields and self._direct_rows is not None:
            return self._rows_solve(omega, fixed, rhs[self.free_dofs])
        self._kept = None
        solution = self._full(self.factor.solve_host(rhs[self.free_dofs]), fixed)
        if self._functionals is not None:
            self.last_functionals = _adjoint_products(self._functionals, solution)
        if product_at_rows and self._direct_rows is not None:
            return solution, self._rows_product(omega, solution[self._direct_rows[1]])
        return solution, self._product(omega, solution)

    def _rows_solve(self, omega, fixed, free_rhs):
        """(x, A x) exact at the direct output rows, 0 elsewhere: the free solution stays where the factor solved it (a
        cuDSS factor: the device) and is kept; only its values at the row blocks' columns reach the host."""
        rows, columns, _ = self._direct_rows
        on_device = getattr(self.factor, 'back_end', None) == 'cudss'   # a CudssFactorization (both precisions)
        free_solution = self.factor.solve(free_rhs) if on_device else self.factor.solve_host(free_rhs)
        self._kept = (free_solution, fixed)
        if self._functionals is not None:   # from the kept free solution (on the device after a cuDSS solve)
            self.last_functionals = self._kept_functionals(free_solution, fixed, on_device)
        positions = self._free_position[columns]
        free = positions >= 0
        values = fixed[columns].copy()   # x = fixed + the free solution scattered (as _full)
        xp = array_module_of(free_solution)
        values[free] += to_host(free_solution[xp.asarray(positions[free])])
        solution = np.zeros(self._n_reduced(), dtype=np.complex128)
        solution[rows] = values[np.searchsorted(columns, rows)]
        return solution, self._rows_product(omega, values)

    @property
    def has_output_rows(self) -> bool:
        """set_output_rows was called (for the reduced model or the direct solves)."""
        return self._rows is not None or self._direct_rows is not None

    def last_solution(self):
        """Host x over every dof of the last direct rows-only solve (its kept free solution: one device transfer after a
        cuDSS solve, R6), None when the last solve was another kind."""
        if self._kept is None:
            return None
        free_solution, fixed = self._kept
        return self._full(np.asarray(to_host(free_solution)), fixed)

    # ---- functionals and extra columns (physics drives: brillouin_scattering's FDT readouts) ----
    @property
    def functionals(self):
        """The weights W of set_functionals (the caller's array), or None."""
        return self._functionals

    def set_functionals(self, weights) -> None:
        """Weights W (k, n) complex over every dof (the operator's numbering, before the column map; mixed: zero on the
        pressure rows reads the displacements only), or None to drop them: every later solve sets last_functionals =
        W^H x (direct: from x; rows-only: from the kept free solution, on the device after a cuDSS solve; reduced: from
        a precomputed (k, basis) block). The array is kept, not copied (pass a read-only one)."""
        self._device_functionals = None
        self._functional_block = None
        self.last_functionals = None
        if weights is None:
            self._functionals = None
            return
        weights = np.asarray(weights)
        if weights.ndim != 2 or weights.shape[1] != self.operator.shape[0] or weights.dtype != np.complex128:
            raise ValueError(f'set_functionals: weights must be complex128 (k, {self.operator.shape[0]}) over every '
                             f'dof, got {weights.dtype} {weights.shape}')
        self._functionals = weights

    def _free_weights(self, xp):
        """W at the free dofs (k, n_free) on the array module xp (the cuDSS device), uploaded row by row on first use
        and kept with this system (booked (estimate): k x n_free complex128 on the device; no column map)."""
        if self._device_functionals is None:
            if self.column_map is not None:
                raise RuntimeError('device functionals need the free dofs of every dof (no column map; a bug)')
            device = xp.empty((len(self._functionals), self.free_dofs.size), dtype=xp.complex128)
            for row in range(len(self._functionals)):   # one host row at a time (no (k, n_free) host copy)
                device[row] = xp.asarray(self._functionals[row, self.free_dofs])
            self._device_functionals = device
        return self._device_functionals

    def _kept_functionals(self, free_solution, fixed, on_device) -> np.ndarray:
        """W^H x of a rows-only solve (no column map) from its free solution (device or host) and the held values."""
        if on_device:
            xp = array_module_of(free_solution)
            values = np.asarray(to_host(xp.conj(self._free_weights(xp) @ xp.conj(free_solution))))
            return values + _adjoint_products(self._functionals, fixed) if fixed.any() else values
        solution = fixed.copy()
        solution[self.free_dofs] += free_solution
        return _adjoint_products(self._functionals, solution)

    def _reduced_functionals(self, model, coordinates, fixed) -> np.ndarray:
        """W^H x of the reduced model at its coordinates: conj(W P) V y + W^H fixed with P the prolongation of the free
        dofs (the column map included), the (k, basis) block made once per model."""
        import scipy.sparse
        block = self._functional_block
        if block is None or block[0] is not model:
            selection = scipy.sparse.csr_matrix((np.ones(self.free_dofs.size), (self.free_dofs, np.arange(
                self.free_dofs.size))), shape=(self._n_reduced(), self.free_dofs.size))
            prolongation = selection if self.column_map is None else (self.column_map @ selection).tocsr()
            projected = np.asarray(prolongation.T @ self._functionals.T)   # (n_free, k) = (W P)^T, P real
            constant = (_adjoint_products(self._functionals, fixed) if fixed.any()
                        else np.zeros(len(self._functionals), dtype=np.complex128))
            block = self._functional_block = (model, np.conj(projected).T @ model.basis, constant)
        return block[1] @ coordinates + block[2]

    def solve_columns(self, omega, loads, rows) -> np.ndarray:
        """Complex (m,) W[rows[s]]^H x_s of the extra right-hand sides loads (m, n) over every dof (held values 0, no
        extra_rhs), each solved on the factor of the last factorisation, which must be a direct solve at this omega
        [rad/s] (the column map and free set as solve; one solve per column: factor.solve_host, or on a cuDSS factor
        without column map factor.solve and a device dot with the kept free weights). RuntimeError without
        functionals, after plan_reduced or at another omega."""
        if self._functionals is None:
            raise RuntimeError('solve_columns needs set_functionals first (it returns W[rows]^H x of each column)')
        if self._planned is not None:
            raise RuntimeError('solve_columns after plan_reduced: a reduced model keeps no factor at each frequency '
                               '(extra columns need direct solves)')
        if self.factor is None or self._factored_omega != omega:
            raise RuntimeError(f'solve_columns at omega {omega!r} rad/s, but the factor is of omega '
                               f'{self._factored_omega!r}: solve the first column directly at this omega first')
        loads = np.asarray(loads)
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        n_dofs = int(self.operator.shape[0])
        if loads.shape != (rows.size, n_dofs):
            raise ValueError(f'solve_columns: loads {loads.shape} must be ({rows.size}, {n_dofs}): one per row index')
        if rows.size and (rows.min() < 0 or rows.max() >= len(self._functionals)):
            raise ValueError(f'solve_columns: row indices outside 0..{len(self._functionals) - 1}')
        on_device = getattr(self.factor, 'back_end', None) == 'cudss' and self.column_map is None
        values = np.empty(rows.size, dtype=np.complex128)
        device_values = None
        for index, (load, row) in enumerate(zip(loads, rows)):
            rhs = np.asarray(load, dtype=np.complex128)
            if self.column_map is not None:
                rhs = self.column_map.T @ rhs
            free_rhs = rhs[self.free_dofs]
            if on_device:   # the solution stays on the device: one dot there, one transfer of m values at the end
                free_solution = self.factor.solve(free_rhs)
                xp = array_module_of(free_solution)
                if device_values is None:
                    device_values = xp.empty(rows.size, dtype=xp.complex128)
                device_values[index] = xp.vdot(self._free_weights(xp)[row], free_solution)
            else:
                reduced = np.zeros(self._n_reduced(), dtype=np.complex128)
                reduced[self.free_dofs] = self.factor.solve_host(free_rhs)
                solution = reduced if self.column_map is None else self.column_map @ reduced
                values[index] = np.vdot(self._functionals[row], solution)
        if device_values is not None:
            values = np.asarray(to_host(device_values), dtype=np.complex128)
        return values

    # ---- reduced sweeps (fem.reduced_harmonic) ----
    @property
    def reduced_model(self):
        """The ReducedHarmonicModel solve() evaluates (plan_reduced), or None (one factorisation per solve)."""
        return None if self._planned is None else self._planned[0]

    def clear_reduced(self) -> None:
        """Drop the reduced model and its output rows: solve() factorises per frequency again (direct output rows
        stay)."""
        self._planned = None
        self._rows = None
        self._kept = None
        self._functional_block = None

    def reduced_problem(self, fixed, load, extra_rhs=()) -> tuple:
        """(HarmonicTerm list, HarmonicLoad list) of the free-dof problem of solve (fem.reduced_harmonic): K, j L, -omega^2
        M (+ j b / omega) and the rhs G^T (load - A(omega) fixed) - extra_rhs split by its powers of omega."""
        from ...fem.reduced_harmonic import HarmonicLoad, HarmonicTerm, PowerCoefficient
        self._refuse_reduced('a reduced')
        stiffness, loss, mass = self._reduced
        terms = [HarmonicTerm(stiffness, PowerCoefficient(1.0, 0))]
        if loss is not None:
            terms.append(HarmonicTerm(loss, PowerCoefficient(1j, 0)))
        terms.append(HarmonicTerm(mass, PowerCoefficient(-1.0, 2)))
        if self._load is not None:
            terms.append(HarmonicTerm(self._load, PowerCoefficient(1j, -1)))
        fixed = np.asarray(fixed, dtype=np.complex128)

        def reduced(vector):
            return np.array(vector if self.column_map is None else self.column_map.T @ vector, dtype=np.complex128)
        constant = reduced(np.asarray(load, dtype=np.complex128) - self.operator @ fixed)
        for dof, value in extra_rhs:
            constant[dof] -= value
        loads = [HarmonicLoad(constant[self.free_dofs], PowerCoefficient(1.0, 0))]
        if np.any(fixed != 0.0):
            loads.append(HarmonicLoad(reduced(self.mass @ fixed)[self.free_dofs], PowerCoefficient(1.0, 2)))
            if self.loss is not None:
                loads.append(HarmonicLoad(-reduced(self.loss @ fixed)[self.free_dofs], PowerCoefficient(1j, 0)))
        return terms, loads

    def plan_reduced(self, omegas, method, fixed, load, extra_rhs=(), *, tolerance=1e-6, moments=4, max_points=40,
                     modal_count=20, eigen_solver=None, verify=True, label='solid') -> dict:
        """Reduced model of solve's free-dof response over angular frequencies omegas [rad/s] for these fixed / load /
        extra_rhs: 'adaptive' (greedy moment matching) or 'modal' (modal_count modes of (K, M) nearest the band centre +
        the static correction; not with electrodes); returns the model's info."""
        from ...fem.harmonic_resonances import solve_harmonic_resonances
        from ...fem.reduced_harmonic import SWEEP_METHODS, ReducedHarmonicModel
        if method not in SWEEP_METHODS:
            raise ValueError(f"reduced sweep method must be one of {', '.join(SWEEP_METHODS)}, got {method!r}")
        self._refuse_reduced(f"{label}: sweep_method '{method}'")
        self.clear_reduced()
        omegas = np.atleast_1d(np.asarray(omegas, dtype=np.float64)).ravel()
        if not (omegas.size and np.all(np.isfinite(omegas)) and np.all(omegas > 0.0)):
            raise ValueError(f'{label}: a reduced sweep needs finite angular frequencies > 0 rad/s')
        if not self.free_dofs.size:
            raise ValueError(f'{label}: no free unknowns to reduce')
        fixed = np.array(fixed, dtype=np.complex128)
        load = np.array(load, dtype=np.complex128)
        terms, loads = self.reduced_problem(fixed, load, extra_rhs)
        model = ReducedHarmonicModel(terms, loads, self._factorize, label=label)
        if method == 'adaptive':
            info = model.train(omegas, tolerance=tolerance, moments=moments, max_points=max_points, verify=verify)
        else:
            if self.column_map is not None or self._load is not None:
                raise ValueError(f"{label}: sweep_method 'modal' needs a purely elastic pencil (piezoelectric electrodes "
                                 "couple a massless potential): use sweep_method 'adaptive'")
            stiffness, _, mass = self._reduced
            centre = 0.5 * (omegas.min() + omegas.max()) / (2.0 * math.pi)
            resonance_solve = solve_harmonic_resonances(stiffness, mass, centre, modal_count, eigen_solver=eigen_solver,
                                                        label='modal basis')
            if not resonance_solve.resonances:
                raise ValueError(f'{label}: the modal sweep found no mode near {centre:.6g} Hz')
            vectors = np.column_stack([np.asarray(resonance.field, dtype=np.complex128)
                                       for resonance in resonance_solve.resonances])
            info = model.fit_modal(vectors, omegas, tolerance=tolerance, verify=verify)
            info.update(modes=len(resonance_solve.resonances),
                        mode_frequencies=[resonance.frequency for resonance in resonance_solve.resonances],
                        eigen_seconds=float(resonance_solve.info['seconds']))
            info['warnings'] = list(info['warnings']) + list(resonance_solve.info['warnings'])
        self._planned = (model, fixed, load, [(int(dof), value) for dof, value in extra_rhs])
        return info

    def _refuse_reduced(self, what) -> None:
        """ValueError with frequency terms: a reduced model needs a pencil polynomial in omega (fem.reduced_harmonic
        HarmonicTerm powers), which the per-frequency terms are not (or not as stored: a zero-coefficient term)."""
        if self._terms:
            labels = ', '.join(term.label or 'unnamed' for term in self._terms[:3])
            more = f' and {len(self._terms) - 3} more' if len(self._terms) > 3 else ''
            raise ValueError(f"{what} sweep needs a pencil polynomial in omega, but this system has "
                             f"{len(self._terms)} per-frequency term(s) ({labels}{more}): frequency-dependent moduli "
                             "(viscosity, relaxation branches) and, at azimuthal_order 1, the pressure block of an "
                             "incompressible material of formulation mixed (kept as a zero term through the axis "
                             "ties) need sweep_method 'direct'")

    def set_output_rows(self, rows) -> None:
        """Rows (over every dof) where solve(fields=False) must give x and A x exactly (records: probes, supports,
        electrodes): after plan_reduced their L V blocks (fem.reduced_outputs.ReducedRows), else for the direct solves
        the operator / mass / loss rows on the columns they touch (no column map; with one, direct solves stay full)."""
        import scipy.sparse
        from ...fem.reduced_outputs import ReducedRows
        if self._planned is None:
            if self.column_map is None:
                rows = np.unique(np.asarray(rows, dtype=np.int64))
                blocks = [None if matrix is None else matrix[rows]
                          for matrix in (self.operator, self.mass, self.loss) + self._term_matrices]
                columns = np.unique(np.concatenate([rows] + [block.indices for block in blocks if block is not None]))
                self._direct_rows = (rows, columns, tuple(None if block is None else block[:, columns].tocsr()
                                                          for block in blocks))
            return
        model, fixed, _, _ = self._planned
        n_reduced = self._n_reduced()
        selection = scipy.sparse.csr_matrix((np.ones(self.free_dofs.size), (self.free_dofs, np.arange(
            self.free_dofs.size))), shape=(n_reduced, self.free_dofs.size))
        prolongation = selection if self.column_map is None else (self.column_map @ selection).tocsr()
        terms = [(self.operator, lambda omega: 1.0), (self.mass, lambda omega: -omega * omega)]
        if self.loss is not None:
            terms.append((self.loss, lambda omega: 1j))
        self._rows = ReducedRows(rows, prolongation, model.basis, fixed, terms)

    def _reduced_solve(self, omega, fixed, load, extra_rhs, fields, product_at_rows=False):
        model, planned_fixed, planned_load, planned_extra = self._planned
        if not (np.array_equal(planned_fixed, fixed) and np.array_equal(planned_load, np.asarray(load))
                and [(int(dof), value) for dof, value in extra_rhs] == planned_extra):
            raise RuntimeError('the held values, loads or charges changed after plan_reduced: plan again or call '
                               'clear_reduced()')
        coordinates = model.reduced_coordinates([omega])[0]
        if self._functionals is not None:
            self.last_functionals = self._reduced_functionals(model, coordinates, fixed)
        if not fields:
            if self._rows is None:
                raise RuntimeError('solve(fields=False) needs set_output_rows after plan_reduced')
            return self._rows.evaluate(omega, coordinates)
        solution = self._full(model.basis @ coordinates, fixed)
        if product_at_rows and self._rows is not None:
            return solution, self._rows.evaluate(omega, coordinates)[1]
        return solution, self._product(omega, solution)


def _squared_entries(matrix):
    """The CSR of |m_ij|^2 on a CSR's pattern, its data replaced in place (pass a matrix no one else holds)."""
    squared = matrix.tocsr()
    squared.data = np.abs(squared.data) ** 2
    return squared


def _adjoint_products(weights, vector) -> np.ndarray:
    """Complex (k,) W^H x = conj(W conj(x)) of dense weights (k, n) and a vector (n,) (no conjugated copy of W)."""
    return np.conj(weights @ np.conj(np.asarray(vector, dtype=np.complex128)))


def _real_times(matrix, vector):
    """matrix @ vector; a real sparse matrix times a complex vector as two real products (scipy's sparsetools would
    upcast the matrix values to a complex copy on every call)."""
    if np.iscomplexobj(matrix.data) or not np.iscomplexobj(vector):
        return matrix @ vector
    return matrix @ np.ascontiguousarray(vector.real) + 1j * (matrix @ np.ascontiguousarray(vector.imag))


def _pattern_positions(pattern, matrix) -> tuple:
    """(int64 positions in the canonical CSR pattern, values) of the entries of a same-shape sparse matrix (duplicates
    summed); ValueError when one is not in the pattern."""
    block = matrix.tocsr(copy=True)
    block.sum_duplicates()
    n_columns = int(pattern.shape[1])
    keys = np.repeat(np.arange(pattern.shape[0], dtype=np.int64), np.diff(pattern.indptr)) * n_columns \
        + pattern.indices.astype(np.int64)
    wanted = np.repeat(np.arange(block.shape[0], dtype=np.int64), np.diff(block.indptr)) * n_columns \
        + block.indices.astype(np.int64)
    if not wanted.size:
        return np.zeros(0, dtype=np.int64), np.asarray(block.data)
    where = np.minimum(np.searchsorted(keys, wanted), max(keys.size - 1, 0))
    if not keys.size or not np.array_equal(keys[where], wanted):
        raise ValueError('a harmonic term has entries outside the summed pattern')
    return where.astype(np.int64), np.asarray(block.data)


def solve_harmonic(operator, loss, mass, omega, free, fixed, load, column_map=None, extra_rhs=(), terms=(),
                   scale_terms=()):
    """(x, A x, factor host bytes) of one frequency by a private HarmonicSystem (one complex host SuperLU; arguments as
    HarmonicSystem / its solve)."""
    harmonic = HarmonicSystem(operator, loss, mass, free, column_map, terms=terms, scale_terms=scale_terms)
    solution, product = harmonic.solve(omega, fixed, load, extra_rhs)
    return solution, product, int(harmonic.factor.host_bytes or 0)


def reaction_table(record, name) -> dict:
    """'table' of a support's complex reaction per solved frequency from its core.harmonic_probe.HarmonicProbeRecord of
    force components in record.unit and the work REACTION_WORK (ascending f): columns frequency, <F>_re, <F>_im per
    component; extras frequency [Hz], components, support. ValueError before the first frequency."""
    frequencies, values = record.recorded()
    columns, units = ['frequency'], ['Hz']
    for component in record.components:
        columns += [f'{component}_re', f'{component}_im']
        unit = _WORK_UNITS[record.unit] if component == REACTION_WORK else record.unit
        units += [unit, unit]
    rows = [[float(frequency)] + [part for component in record.components
                                  for part in (float(values[component][index].real),
                                               float(values[component][index].imag))]
            for index, frequency in enumerate(frequencies)]
    return {'kind': 'table', 'columns': columns, 'units': units, 'rows': rows, 'frequency': frequencies.tolist(),
            'components': list(record.components), 'support': name}


@dataclass(frozen=True, eq=False)
class FrequencyPoint:
    """One solved frequency [Hz] with {electrode name: (V, Q)} complex amplitudes [V, C/m] (empty without piezo)."""
    frequency: float
    electrodes: dict


@dataclass(frozen=True, eq=False)
class AdmittanceMonitor:
    """Runtime object of an admittance feature: its electrode name ('' = the only driven voltage electrode), the BVD
    fit band [Hz] (None = the whole sweep) and the load impedance [Ohm] of kappa_e."""
    name: str
    electrode: str
    fit_min: float | None
    fit_max: float | None
    load_impedance: float = 50.0


def driven_electrode(monitor, electrodes, label):
    """Name of the monitored electrode: monitor.electrode, or the only voltage electrode with a non-zero voltage;
    ValueError otherwise."""
    voltage = [electrode for electrode in electrodes if electrode.connection == 'voltage']
    if monitor.electrode:
        match = [electrode for electrode in voltage if electrode.name == monitor.electrode]
        if not match:
            raise ValueError(f"{label()}: admittance '{monitor.name}' names electrode '{monitor.electrode}', which is "
                             f"no enabled voltage electrode; voltage electrodes: "
                             f"{', '.join(electrode.name for electrode in voltage) or 'none'}")
        return monitor.electrode
    driven = [electrode.name for electrode in voltage if electrode.voltage != 0.0]
    if len(driven) != 1:
        raise ValueError(f"{label()}: admittance '{monitor.name}' needs its electrode setting when {len(driven)} "
                         f"voltage electrodes have a non-zero voltage ({', '.join(driven) or 'none'})")
    return driven[0]


def admittance_result(monitor, electrode, points, thickness=None):
    """'xy' result of Y(f) = j omega Q / V of `electrode` over the solved FrequencyPoints (ascending f), times the
    thickness [m] (None: per m of depth; a 3D model passes 1.0, its charges are in C): series abs / re / im [S or S/m],
    phase [deg], bvd_abs (the fitted model, NaN without a fit); extras the BVD fit (bvd.FIT_KEYS, fit_warning when it
    failed), kappa_e / omega_C0_Z at monitor.load_impedance (NaN per m of depth: kappa_e_warning), load_impedance,
    bvd_units, electrode."""
    depth = 1.0 if thickness is None else float(thickness)
    points = sorted(points, key=lambda point: point.frequency)
    if not points:
        raise ValueError(f"admittance '{monitor.name}': no frequency was solved (a frequency_response study fills it)")
    frequencies = np.array([point.frequency for point in points], dtype=np.float64)
    admittance = np.empty(frequencies.size, dtype=np.complex128)
    for index, point in enumerate(points):
        voltage, charge = point.electrodes[electrode]
        if voltage == 0:
            raise ValueError(f"admittance '{monitor.name}': electrode '{electrode}' has voltage 0, so Y = I / V is "
                             'undefined: give the driven electrode a non-zero voltage')
        admittance[index] = 2j * math.pi * point.frequency * charge / voltage * depth
    fit = bvd_fit(frequencies, admittance, monitor.fit_min, monitor.fit_max)
    model = (np.abs(bvd_admittance(frequencies, fit)) if math.isfinite(fit['C0'])
             else np.full(frequencies.size, math.nan))
    per_depth = '' if thickness is not None else '/m'
    inverse_depth = '' if thickness is not None else ' m'
    unit = f'S{per_depth}'
    series = {'abs': np.abs(admittance), 're': admittance.real, 'im': admittance.imag,
              'phase': np.angle(admittance) * _PHASE_DEGREES, 'bvd_abs': model}
    units = {'abs': unit, 're': unit, 'im': unit, 'phase': 'deg', 'bvd_abs': unit}
    bvd_units = {'f_s': 'Hz', 'f_p': 'Hz', 'f_peak': 'Hz', 'f_dip': 'Hz', 'k_eff2': '1', 'Q_s': '1',
                 'tan_delta_0': '1', 'C0': f'F{per_depth}', 'C1': f'F{per_depth}', 'L1': f'H{inverse_depth}',
                 'R1': f'Ohm{inverse_depth}', 'fit_error': '1', 'fit_points': '1', 'kappa_e': 'rad/s',
                 'omega_C0_Z': '1', 'load_impedance': 'Ohm'}
    if thickness is None:   # L1 / C0 per m of depth: a load in Ohm has no meaning there
        coupling = {'kappa_e': math.nan, 'omega_C0_Z': math.nan,
                    'kappa_e_warning': 'kappa_e needs the admittance of a finite device: set the thickness setting '
                                       '(2D) so C0 and L1 are in F and H'}
    else:
        coupling = external_coupling(fit, monitor.load_impedance)
    return {'kind': 'xy', 'x': frequencies, 'x_label': 'Frequency', 'x_unit': 'Hz', 'y_label': 'Admittance',
            'series': series, 'series_units': units, 'electrode': electrode, 'bvd_units': bvd_units, **fit,
            **coupling, 'load_impedance': float(monitor.load_impedance)}
