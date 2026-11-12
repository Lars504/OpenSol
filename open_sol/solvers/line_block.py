"""'line_block' shifted-solver runtime (solver_plan Part II): batched block-tridiagonal direct solves of
(K + sum_i s_i M_i + sum_j t_j N_j) x = b, M_i diagonal, N_j sparse block-tridiagonal, on line-ordered grid unknowns: per
shift a block Thomas factorisation (stored block inverses), equilibrated, refined against the complex128 operator;
chunks of shifts (solve_shifted), a kept factor (factorize) or a kept batch of whole operators, one per shift
(factorize_operators: FDFD / BOR wavelengths, cross-section shift-invert operators); LineBlockFactorization = one
shift of a kept factor as a sparse_direct factor stand-in (shift-invert matvecs). Backend 'gpu' (cupy) or 'cpu' (numpy /
scipy, the same array code); numpy at module level, cupy inside functions."""
import math
import time

import numpy as np

# precision -> (work complex dtype, work real dtype); residuals are always complex128
PRECISION_DTYPES = {'mixed': (np.complex64, np.float32), 'double': (np.complex128, np.float64)}
REFINEMENT_TOLERANCE = 1e-9     # refine while some shift has ||r|| / ||b|| above this
ACCEPT_TOLERANCE = 1e-6         # a shift above this is rejected (the consumer re-solves it); keep above the fp32 floor
MAX_REFINEMENTS = 8
STALL_RATIO = 0.7               # stop refining once no open shift lowers its residual below this x the previous one
MEMORY_FRACTION = 0.6           # share of the free pool budget one chunk of shifts may take
MAX_CHUNK_SHIFTS = 64           # shifts factored together (keeps the run memory estimate bounded)
UNBUDGETED_CHUNK_BYTES = 1 << 30   # chunk bytes without a budget (gpu: set_budget never called; cpu: RAM unknown)
# block size up to which cp.linalg.inv runs batched; above it a loop of 2D inverses (cuBLAS *Batched is ~200x slower at
# b ~ 800, solver_plan Part II section 5); the crossover is not measured yet
BATCHED_INVERSE_MAX_BLOCK = 128


# run-time estimate: refinement sweeps per solve by precision (summary refinement_sweeps / factor_solves or shifts;
# em_modes_line_block: mixed 1, double 0)
REFINEMENT_SWEEPS_GUESS = {'mixed': 1.0, 'double': 0.0}
# run-time estimate: the loops over blocks are launch-bound on small blocks (a gpu cupy matmul costs the 'gemm' floor,
# ~80 us), so each block product / inverse is priced per call; elementwise / copy launches counted from _factor / _sweep
FACTOR_DENSE_CALLS_PER_BLOCK = 3   # two block products and one batched inverse
FACTOR_LAUNCHES_PER_BLOCK = 12
SWEEP_PRODUCTS_PER_BLOCK = 4       # forward coupling + inverse, backward coupling + inverse
SWEEP_LAUNCHES_PER_BLOCK = 7       # forward 4, backward 3
SWEEP_FIXED_LAUNCHES = 18          # per sweep: padding, scaling, residual product, norms
# kept factors (factorize, factorize_operators) also store G_k = D_k^-1 L_k and H_k = D_k^-1 U_k of the scaled system:
# their sweep is 3 products per block and no scale multiplies (loop: 5 launches per block); a single-rhs sweep on the
# gpu whose three stacks fit KEPT_SWEEP_KERNEL_MAX_BYTES per shift is one launch (one CUDA block per shift)
KEPT_BLOCK_STACKS = 3
KEPT_SWEEP_PRODUCTS_PER_BLOCK = 3
KEPT_SWEEP_LAUNCHES_PER_BLOCK = 5
KEPT_SWEEP_KERNEL_MAX_BYTES = 64 << 20  # one SM streams them: ~25 GB/s on an RTX 3050, ~2.5 ms at the cap
KEPT_SWEEP_THREADS = 256


# --- CPU helpers (no cupy: estimators) ----------------------------------------------------------------------------

def time_estimate(values, work):
    """Run-time recipe of a line_block role (core.time_estimate.SolverWork, solves = shifts, rhs per shift): per shift
    a block Thomas factorisation (~ 4 n_blocks b^3 complex flops), one sweep + refinement sweeps (b^2 per block and
    rhs) with their complex128 residual products; small blocks pay a per-call floor per block."""
    from ..core import time_estimate as te
    n_unknowns = float(work.unknowns)
    block = float(work.block_size or max(math.isqrt(int(n_unknowns)), 1))
    n_blocks = n_unknowns / block
    complex_flops = 4.0
    factor = ((FACTOR_DENSE_CALLS_PER_BLOCK * n_blocks,
               te.DenseFlops(complex_flops * 4.0 * block ** 3 / FACTOR_DENSE_CALLS_PER_BLOCK)),
              te.elementwise(n_blocks * block * block, 6, 3, 16, launches=int(FACTOR_LAUNCHES_PER_BLOCK * n_blocks)))
    sweep = _sweep_ops(n_unknowns, block, work.nnz, work.rhs)
    shifts = float(work.solves)
    sweeps = 1.0 + _refinement_sweeps(values.get('precision'))
    return te.TimeEstimate((
        te.TimeItem('line-block factorisations', 'shift', shifts, work.solves_kind, factor, per_step=work.per_step,
                    counter=work.counter('shifts')),
        te.TimeItem('line-block sweeps', 'sweep', shifts * sweeps, 'guess', sweep,
                    limit=shifts * (1 + MAX_REFINEMENTS), per_step=work.per_step)))


def _refinement_sweeps(precision):
    """REFINEMENT_SWEEPS_GUESS of a precision (None / unknown: the 'mixed' default of the SolverType)."""
    return REFINEMENT_SWEEPS_GUESS.get(precision, REFINEMENT_SWEEPS_GUESS['mixed'])


def _sweep_ops(n_unknowns, block, nnz, rhs):
    """Ops of one block Thomas sweep: per block SWEEP_PRODUCTS_PER_BLOCK (b, b) x (b, rhs) products (one call each) and
    SWEEP_LAUNCHES_PER_BLOCK small launches, the complex128 residual product and read-back."""
    from ..core import time_estimate as te
    n_blocks = n_unknowns / block
    launches = int(SWEEP_LAUNCHES_PER_BLOCK * n_blocks + SWEEP_FIXED_LAUNCHES)
    return ((max(SWEEP_PRODUCTS_PER_BLOCK * n_blocks - 3.0, 1.0), te.DenseFlops(8.0 * block ** 2 * rhs)),
            te.elementwise(launches * block * rhs, 2, 1, 16, launches=launches),
            te.Spmv(n_unknowns, max(nnz, 5.0 * n_unknowns), 16), te.Reduction(n_unknowns, 16, read_back=True),
            te.PythonCalls(4 * n_blocks))


def factor_solve_ops(n_unknowns, block_size=0, nnz=0.0, precision='mixed') -> tuple:
    """Ops of one solve on a kept factor (LineBlockFactorization.solve_host, e.g. a shift-invert application): host rhs
    to the device, one kept sweep + the precision's REFINEMENT_SWEEPS_GUESS refinement sweeps, the solution back."""
    from ..core import time_estimate as te
    n_unknowns = float(n_unknowns)
    block = float(block_size or max(math.isqrt(int(n_unknowns)), 1))
    sweeps = 1.0 + _refinement_sweeps(precision)
    pairs = (entry if isinstance(entry, tuple) else (1.0, entry)
             for entry in _kept_sweep_ops(n_unknowns, block, nnz, precision))
    return tuple((sweeps * multiplicity, op) for multiplicity, op in pairs) + (te.Transfer(2.0 * 16 * n_unknowns),)


def _kept_sweep_ops(n_unknowns, block, nnz, precision):
    """Ops of one single-rhs sweep on a kept factor: one kernel launch over the three block stacks when they fit
    KEPT_SWEEP_KERNEL_MAX_BYTES (gpu), else KEPT_SWEEP_PRODUCTS_PER_BLOCK calls per block; plus the fixed launches,
    the complex128 residual product and read-back."""
    from ..core import time_estimate as te
    n_blocks = n_unknowns / block
    complex_bytes, _ = _item_bytes(precision or 'mixed')
    if KEPT_BLOCK_STACKS * n_blocks * block * block * complex_bytes <= KEPT_SWEEP_KERNEL_MAX_BYTES:
        products = (1.0, te.DenseFlops(8.0 * block ** 2 * KEPT_SWEEP_PRODUCTS_PER_BLOCK * n_blocks))
        launches = SWEEP_FIXED_LAUNCHES + 1
        calls = 4.0
    else:
        products = (max(KEPT_SWEEP_PRODUCTS_PER_BLOCK * n_blocks - 2.0, 1.0), te.DenseFlops(8.0 * block ** 2))
        launches = int(KEPT_SWEEP_LAUNCHES_PER_BLOCK * n_blocks + SWEEP_FIXED_LAUNCHES)
        calls = 3.0 * n_blocks
    return (products, te.elementwise(launches * block, 2, 1, 16, launches=launches),
            te.Spmv(n_unknowns, max(nnz, 5.0 * n_unknowns), 16), te.Reduction(n_unknowns, 16, read_back=True),
            te.PythonCalls(calls))


def _item_bytes(precision):
    complex_dtype, real_dtype = PRECISION_DTYPES[precision]
    return np.dtype(complex_dtype).itemsize, np.dtype(real_dtype).itemsize


def operator_bytes(n_blocks, block_size, n_unknowns, n_entries, n_terms, precision, complex_operator=False,
                   n_matrix_terms=0, matrix_entries=0) -> int:
    """Device bytes kept by set_operator: the (n_blocks, 3, b, b) operator cache (+ a complex one and row norms per
    matrix term), padded row norms / pad identity / diagonal terms, the complex128 CSRs of the residual, the slot index."""
    complex_bytes, real_bytes = _item_bytes(precision)
    padded = int(n_blocks) * int(block_size)
    cache = 3 * padded * int(block_size) * (complex_bytes if complex_operator else real_bytes)
    matrix = int(n_matrix_terms) * (3 * padded * int(block_size) * complex_bytes + padded * 8)
    return (cache + matrix + padded * (8 + real_bytes) + int(n_terms) * int(n_unknowns) * 8
            + (int(n_entries) + int(matrix_entries)) * (16 + 4) + (int(n_matrix_terms) + 1) * (int(n_unknowns) + 1) * 4
            + int(n_unknowns) * 8)


def shift_bytes(n_blocks, block_size, n_unknowns, n_rhs, precision, matrix_terms=False, kept=False) -> int:
    """Device bytes per shift of a chunk (or of a kept factor): block inverses (kept: + the pre-multiplied couplings
    G_k, H_k), factor temporaries (+ the per-shift coupling blocks of matrix terms), padded shift diagonal and scales,
    sweep and complex128 refinement vectors."""
    complex_bytes, real_bytes = _item_bytes(precision)
    b, padded, n_rhs = int(block_size), int(n_blocks) * int(block_size), int(n_rhs)
    temporaries = (9 if matrix_terms else 6) * b * b * complex_bytes
    stacks = KEPT_BLOCK_STACKS if kept else 1
    return (stacks * padded * b * complex_bytes + temporaries + padded * (16 + complex_bytes + real_bytes)
            + 3 * padded * n_rhs * complex_bytes + ((7 if matrix_terms else 6) * n_rhs + 1) * int(n_unknowns) * 16)


def operator_shift_bytes(n_blocks, block_size, n_unknowns, n_entries, precision) -> int:
    """Device bytes per operator of a factorize_operators batch on top of shift_bytes: its (n_blocks, 3, b, b) work-dtype
    block cache, padded row norms and the complex128 residual CSR."""
    complex_bytes, _ = _item_bytes(precision)
    padded = int(n_blocks) * int(block_size)
    return (3 * padded * int(block_size) * complex_bytes + padded * 8 + int(n_entries) * (16 + 4)
            + (int(n_unknowns) + 1) * 4)


# --- layout -------------------------------------------------------------------------------------------------------

class LineBlockLayout:
    """Line ordering of n unknowns: block k = the unknowns of the k-th occupied grid line (ascending unknown order),
    padded to b = the largest block; slot[u] = k * b + position of unknown u in its block."""

    def __init__(self, line_ids, block_axis=None):
        line_ids = np.asarray(line_ids, dtype=np.int64).ravel()
        if line_ids.size == 0:
            raise ValueError('line_block layout needs at least one unknown')
        lines, block_of = np.unique(line_ids, return_inverse=True)
        sizes = np.bincount(block_of, minlength=lines.size)
        starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
        order = np.argsort(block_of, kind='stable')   # unknowns grouped by block: a permutation by construction
        position = np.empty(line_ids.size, dtype=np.int64)
        position[order] = np.arange(line_ids.size) - starts[block_of[order]]
        self.block_axis = block_axis     # grid axis inside a block (from_grid_cells), None when given line ids
        self.n_unknowns = int(line_ids.size)
        self.n_blocks = int(lines.size)
        self.block_size = int(sizes.max())
        self.sizes = sizes
        self.block_of = block_of.astype(np.int64)
        self.position = position
        self.slot = self.block_of * self.block_size + position

    @classmethod
    def from_grid_cells(cls, cell_index, grid_shape, block_axis=None):
        """Layout of one unknown per cell, cell_index flat i * ny + j on the (nx, ny) grid; block_axis = the grid axis
        inside a block (0: blocks are the cells of one y row j, 1: of one x column i), None = the cheaper n_blocks b^3."""
        cell_index = np.asarray(cell_index, dtype=np.int64).ravel()
        column, row = np.divmod(cell_index, int(grid_shape[1]))   # i (x), j (y)
        line_ids = {0: row, 1: column}
        if block_axis is None:
            def cost(axis):
                sizes = np.bincount(line_ids[axis])
                return np.count_nonzero(sizes) * float(sizes.max()) ** 3
            block_axis = min((1, 0), key=cost)
        if block_axis not in line_ids:
            raise ValueError(f'block_axis must be 0, 1 or None, got {block_axis!r}')
        return cls(line_ids[block_axis], block_axis)


# --- runtime object -----------------------------------------------------------------------------------------------

class _ShiftOperators:
    """Whole operators of a factorize_operators batch, one per shift: (n_c, n_blocks, 3, b, b) work-dtype block caches,
    (n_c, n_blocks, b) float64 padded row norms, complex128 backend CSRs (residuals)."""

    def __init__(self, cache, row_norms, device):
        self.cache = cache
        self.row_norms = row_norms
        self.device = device

    def select(self, part):
        """_ShiftOperators of the shifts in slice `part` (views)."""
        return _ShiftOperators(self.cache[part], self.row_norms[part], self.device[part])


class _Chunk:
    """Device factor state of a batch of shifts: diagonal shift (n_c, n) complex128, matrix-term coefficients (n_c,
    n_matrix) complex128 / work dtype (None without matrix terms), equilibration scales, block inverses, the kept
    factors' pre-multiplied couplings G_k / H_k (None in a solve_shifted chunk), and the per-shift _ShiftOperators of a
    factorize_operators batch (None: the bound operator family)."""

    def __init__(self, shift_flat, matrix_coefficients, matrix_work, scale, inverses=None, operators=None):
        self.shift_flat = shift_flat
        self.matrix_coefficients = matrix_coefficients
        self.matrix_work = matrix_work
        self.scale = scale
        self.inverses = inverses
        self.operators = operators
        self.lower = None   # kept: (n_c, n_blocks, b, b) G_k = D_k^-1 L_k (scaled system; G_0 = 0)
        self.upper = None   # kept: (n_c, n_blocks, b, b) H_k = D_k^-1 U_k (H_last = 0)

    def select(self, index):
        """_Chunk view of shift `index` alone (no copies)."""
        part = slice(index, index + 1)
        selected = _Chunk(None if self.shift_flat is None else self.shift_flat[part],
                          None if self.matrix_coefficients is None else self.matrix_coefficients[part],
                          None if self.matrix_work is None else self.matrix_work[part], self.scale[part],
                          self.inverses[part], None if self.operators is None else self.operators.select(part))
        if self.lower is not None:
            selected.lower, selected.upper = self.lower[part], self.upper[part]
        return selected


class LineBlockFactor:
    """Kept block inverses of a batch of shifts (LineBlockSolver.factorize): solve(rhs, index) = one sweep plus
    refinement against the complex128 operator of shift `index`; valid until its solver's operator changes."""

    def __init__(self, solver, chunk, n_shifts):
        self.solver = solver
        self.n_shifts = int(n_shifts)
        self._chunk = chunk
        self._serial = solver._operator_serial

    @property
    def valid(self) -> bool:
        """False once released or once the solver holds another operator."""
        return self._chunk is not None and self._serial == self.solver._operator_serial

    def solve(self, rhs, index=0):
        """(solution complex128 host (n,), relative residual, accepted) of shift `index` for a host rhs (n,);
        RuntimeError when the factor is no longer valid."""
        if not self.valid:
            raise RuntimeError(f'{self.solver.label()}: line_block factor used after release or set_operator')
        if not 0 <= int(index) < self.n_shifts:
            raise ValueError(f'shift index {index} outside the {self.n_shifts} factored shifts')
        return self.solver._factor_solve(self._chunk.select(int(index)), rhs)

    def release(self) -> None:
        """Drop the device inverses."""
        self._chunk = None


class RejectedLineBlockSolve(RuntimeError):
    """A kept line-block factor missed the solver's accept tolerance: the consumer re-solves with host SuperLU."""


class LineBlockFactorization:
    """sparse_direct.Factorization stand-in (solve_host, host_bytes, description) on shift `index` of a LineBlockFactor
    (shift-invert matvecs, driven solves): solve_host raises RejectedLineBlockSolve above the accept tolerance."""
    recyclable = False
    host_bytes = 0
    description = 'line-block'   # factor_kind / factor text of the eigen-solve infos

    def __init__(self, factor, index, label):
        self.factor = factor
        self.index = int(index)
        self.label = label

    @property
    def valid(self) -> bool:
        return self.factor.valid

    def solve_host(self, rhs):
        solution, relative, accepted = self.factor.solve(rhs, self.index)
        if not accepted:
            raise RejectedLineBlockSolve(f'{self.label}: line-block relative residual {relative:.3g} above the accept '
                                         f'tolerance {self.factor.solver.accept_tolerance:g}')
        return solution

    def release(self):
        """Release the whole LineBlockFactor (every shift of its batch)."""
        self.factor.release()


def batch_factorizations(solver, operators, layout, labels, owner=None):
    """[LineBlockFactorization] of one factorize_operators batch of host operators (each block-tridiagonal in
    `layout`), one per operator in order; they share one LineBlockFactor (release() of any frees the batch)."""
    factor = solver.factorize_operators(operators, layout, owner=owner)
    return [LineBlockFactorization(factor, index, label) for index, label in enumerate(labels)]


class LineBlockSolver:
    """Shifted solver of the 'line_block' type: set_operator(K, layout, diagonal terms M_i, matrix terms N_j) once,
    then solve_shifted yields chunks of solutions of (K + sum_i s_i M_i + sum_j t_j N_j) x = b, or factorize keeps the
    factor of a batch of shifts for repeated solves; keeps the work record."""

    def __init__(self, precision='mixed', tolerance=REFINEMENT_TOLERANCE, accept_tolerance=ACCEPT_TOLERANCE,
                 max_refinements=MAX_REFINEMENTS, max_chunk_shifts=MAX_CHUNK_SHIFTS, debug=False, label=None,
                 backend='gpu'):
        if precision not in PRECISION_DTYPES:
            raise ValueError(f"precision must be one of {', '.join(PRECISION_DTYPES)}, got {precision!r}")
        if not (tolerance > 0.0 and accept_tolerance > 0.0) or int(max_refinements) < 0 or int(max_chunk_shifts) < 1:
            raise ValueError(f'need tolerance, accept_tolerance > 0, max_refinements >= 0, max_chunk_shifts >= 1, got '
                             f'{tolerance}, {accept_tolerance}, {max_refinements}, {max_chunk_shifts}')
        if backend not in ('gpu', 'cpu'):
            raise ValueError(f"backend must be 'gpu' or 'cpu', got {backend!r}")
        self.backend = backend   # where the operator cache, factors and sweeps live; inputs / solutions stay host numpy
        self.precision = precision
        self.work_dtype, self.real_dtype = (np.dtype(dtype) for dtype in PRECISION_DTYPES[precision])
        self.tolerance = float(tolerance)
        self.accept_tolerance = float(accept_tolerance)
        self.max_refinements = int(max_refinements)
        self.max_chunk_shifts = int(max_chunk_shifts)
        self.debug = bool(debug)
        self.label = label if label is not None else (lambda: 'line_block')
        self.layout = None
        self.n_batches = 0
        self.n_shifts = 0
        self.n_refinement_sweeps = 0
        self.n_rejected = 0
        self.n_kept_factors = 0
        self.n_factor_solves = 0
        self.n_rejected_solves = 0
        self.n_batched_operators = 0   # operators factorised by factorize_operators (its batches count as kept factors)
        self.worst_accepted_residual = 0.0
        self.batch_log = []   # debug: per batch shifts, factor / solve seconds, residual before / after refinement
        self._operator_serial = 0   # bumped by every release: kept factors of an older operator become invalid
        self.release()

    def release(self) -> None:
        """Drop the device operator state (the work record stays; kept factors become invalid)."""
        self._operator_serial += 1
        self.owner = None              # set_operator's owner: consumers sharing one solver re-bind when it differs
        self._cache = None            # (n_blocks, 3, b, b): [k, 0] couples to k - 1, [k, 1] diagonal, [k, 2] to k + 1
        self._row_norms = None         # (n_blocks, b) float64 ||K row||_2, 0 on pads
        self._pad_identity = None      # (n_blocks, b) real work dtype, 1 on pads
        self._terms = None             # (n_diagonal_terms, n) float64 diagonals M_i
        self._matrix_caches = None     # (n_matrix_terms, n_blocks, 3, b, b) work dtype of the N_j, None without
        self._matrix_row_norms = None  # (n_matrix_terms, n_blocks, b) float64 ||N_j row||_2
        self._matrix_device = ()       # complex128 CSRs of the N_j (residuals)
        self._operator_device = None   # complex128 CSR of K (residuals)
        self._slot = None

    @property
    def n_terms(self) -> int:
        """Coefficient columns of the bound family: diagonal terms, then matrix terms."""
        return int(self._terms.shape[0]) + len(self._matrix_device)

    def set_operator(self, operator, layout, diagonal_terms, owner=None, matrix_terms=()) -> None:
        """Bind the host scipy sparse K (n x n, block-tridiagonal in `layout`: ValueError otherwise), the diagonal
        terms M_i ((n,) each, may be none) and the host sparse matrix terms N_j (block-tridiagonal like K) of the family
        (coefficient columns: the M_i, then the N_j); `owner` (kept as self.owner) tells sharing consumers whose it is."""
        from ..core.backend import sparse_module
        cp = self._xp()
        n_unknowns = layout.n_unknowns
        matrix, entries = self._checked_matrix(operator, layout, 'operator')
        checked_terms = [self._checked_matrix(term, layout, f'matrix term {index + 1}')
                         for index, term in enumerate(matrix_terms)]
        diagonal_terms = list(diagonal_terms)
        terms = (np.stack([np.asarray(term, dtype=np.float64).ravel() for term in diagonal_terms]) if diagonal_terms
                 else np.zeros((0, n_unknowns)))
        if terms.shape[1] != n_unknowns:
            raise ValueError(f'{self.label()}: diagonal terms must have {n_unknowns} values each, got {terms.shape[1]}')
        self.release()
        complex_operator = np.iscomplexobj(matrix.data)
        self._cache, self._row_norms = self._block_cache(entries, layout,
                                                         self.work_dtype if complex_operator else self.real_dtype)
        if checked_terms:
            # filled in place: separate caches stacked afterwards doubled the peak (freed parts too small to reuse)
            n_blocks, block_size = layout.n_blocks, layout.block_size
            self._matrix_caches = cp.empty((len(checked_terms), n_blocks, 3, block_size, block_size),
                                           dtype=self.work_dtype)
            self._matrix_row_norms = cp.empty((len(checked_terms), n_blocks, block_size), dtype=cp.float64)
            for index, (_, term_entries) in enumerate(checked_terms):
                _, self._matrix_row_norms[index] = self._block_cache(term_entries, layout, self.work_dtype,
                                                                     out=self._matrix_caches[index])
            self._matrix_device = tuple(sparse_module(self.backend).csr_matrix(term.astype(np.complex128))
                                        for term, _ in checked_terms)
        self._bind_layout(layout, terms)
        self._operator_device = sparse_module(self.backend).csr_matrix(matrix.astype(np.complex128))
        self.owner = owner

    def factorize_operators(self, operators, layout, owner=None) -> LineBlockFactor:
        """LineBlockFactor of one batch of whole host scipy operators A_s (each block-tridiagonal in `layout`, no shift
        family: e.g. one FDFD Helmholtz operator per wavelength), shift index s = operators[s]; replaces the bound
        operator (kept as owner). Size the batch with chunk_size(n, 1, layout, operator_shift_bytes(...))."""
        from ..core.backend import sparse_module
        cp = self._xp()
        matrices = [self._checked_matrix(operator, layout, f'operator {index + 1}')
                    for index, operator in enumerate(operators)]
        if not matrices:
            raise ValueError(f'{self.label()}: factorize_operators needs at least one operator')
        self.release()
        self._bind_layout(layout, np.zeros((0, layout.n_unknowns)))
        self.owner = owner
        n_chunk, n_blocks, block_size = len(matrices), layout.n_blocks, layout.block_size
        cache = cp.empty((n_chunk, n_blocks, 3, block_size, block_size), dtype=self.work_dtype)
        row_norms = cp.empty((n_chunk, n_blocks, block_size), dtype=cp.float64)
        for index, (_, entries) in enumerate(matrices):
            _, row_norms[index] = self._block_cache(entries, layout, self.work_dtype, out=cache[index])
        device = tuple(sparse_module(self.backend).csr_matrix(matrix.astype(np.complex128)) for matrix, _ in matrices)
        del matrices
        chunk = self._factor_chunk(cp.zeros((n_chunk, 0), dtype=cp.complex128),
                                   _ShiftOperators(cache, row_norms, device), kept=True)
        self.n_batches += 1
        self.n_shifts += n_chunk
        self.n_kept_factors += 1
        self.n_batched_operators += n_chunk
        return LineBlockFactor(self, chunk, n_chunk)

    def _bind_layout(self, layout, terms):
        """Layout state shared by set_operator and factorize_operators: pad identity, slot index, diagonal terms."""
        cp = self._xp()
        pad_identity = np.ones(layout.n_blocks * layout.block_size, dtype=self.real_dtype)
        pad_identity[layout.slot] = 0.0
        self._pad_identity = cp.asarray(pad_identity.reshape(layout.n_blocks, layout.block_size))
        self._terms = cp.asarray(terms)
        self._slot = cp.asarray(layout.slot)
        self.layout = layout

    def _checked_matrix(self, operator, layout, what):
        """(host CSR copy (duplicates summed), its COO entries) of an n x n operator; ValueError unless
        block-tridiagonal in `layout`."""
        matrix = operator.tocsr(copy=True)
        matrix.sum_duplicates()
        n_unknowns = layout.n_unknowns
        if matrix.shape != (n_unknowns, n_unknowns):
            raise ValueError(f'{self.label()}: {what} shape {matrix.shape} does not match the layout of '
                             f'{n_unknowns} unknowns')
        entries = matrix.tocoo()
        offsets = layout.block_of[entries.col.astype(np.int64)] - layout.block_of[entries.row.astype(np.int64)] + 1
        if offsets.size and (offsets.min() < 0 or offsets.max() > 2):
            raise ValueError(f'{self.label()}: the {what} couples unknowns {int(np.abs(offsets - 1).max())} lines '
                             'apart, so it is not block-tridiagonal in this line ordering (a wrap or long-range link '
                             'along the line axis: choose the other block axis)')
        return matrix, entries

    def _block_cache(self, entries, layout, dtype, out=None):
        """((n_blocks, 3, b, b) dtype block cache, (n_blocks, b) float64 padded row 2-norms) of a checked matrix's COO
        entries (_checked_matrix); `out`: a preallocated cache of that shape and dtype to fill."""
        cp = self._xp()
        rows, columns, values = entries.row.astype(np.int64), entries.col.astype(np.int64), entries.data
        block_rows = layout.block_of[rows]
        offsets = layout.block_of[columns] - block_rows + 1
        n_blocks, block_size = layout.n_blocks, layout.block_size
        if out is None:
            cache = cp.zeros((n_blocks, 3, block_size, block_size), dtype=dtype)
        else:
            cache = out
            cache[...] = 0
        cache[cp.asarray(block_rows), cp.asarray(offsets), cp.asarray(layout.position[rows]),
              cp.asarray(layout.position[columns])] = cp.asarray(values.astype(dtype))
        row_norms = np.sqrt(np.bincount(rows, weights=np.abs(values) ** 2, minlength=layout.n_unknowns))
        padded_norms = np.zeros(n_blocks * block_size)
        padded_norms[layout.slot] = row_norms
        return cache, cp.asarray(padded_norms.reshape(n_blocks, block_size))

    def _xp(self):
        """Array module of the backend (numpy or cupy)."""
        from ..core.backend import array_module
        return array_module(self.backend)

    def chunk_size(self, n_shifts, n_rhs, layout=None, extra_shift_bytes=0, kept=False) -> int:
        """Shifts per chunk (or per kept factor: kept=True, its coupling stacks counted): all, at most max_chunk_shifts
        and what MEMORY_FRACTION of the free memory holds (gpu: pool budget - pool used; cpu: the available RAM);
        layout + extra_shift_bytes (operator_shift_bytes): a factorize_operators batch, else the bound operator family."""
        if layout is None:
            layout, matrix_terms = self.layout, bool(self._matrix_device)
        else:
            matrix_terms = False
        per_shift = shift_bytes(layout.n_blocks, layout.block_size, layout.n_unknowns, n_rhs, self.precision,
                                matrix_terms=matrix_terms, kept=kept) + int(extra_shift_bytes)
        if self.backend == 'cpu':
            from ..core.host_memory import available_bytes
            available = available_bytes()
            free = UNBUDGETED_CHUNK_BYTES if available is None else available
        else:
            from ..core import gpu_memory
            budget = gpu_memory.budget_bytes()
            free = UNBUDGETED_CHUNK_BYTES if budget is None else max(budget - gpu_memory.pool_used_bytes(), 0)
        return max(1, min(int(n_shifts), self.max_chunk_shifts, int(MEMORY_FRACTION * free // per_shift)))

    def batch_size(self, n_operators, layout, n_entries) -> int:
        """Operators of one factorize_operators batch (chunk_size with operator_shift_bytes of n_entries CSR entries)."""
        return self.chunk_size(n_operators, 1, layout, operator_shift_bytes(layout.n_blocks, layout.block_size,
                                                                             layout.n_unknowns, n_entries,
                                                                             self.precision), kept=True)

    def _checked_coefficients(self, coefficients):
        """Host complex128 (n_shifts, n_terms) coefficients; RuntimeError before set_operator, ValueError on shape."""
        if self._cache is None:
            raise RuntimeError(f'{self.label()}: set_operator before solving')
        coefficients = np.asarray(coefficients, dtype=np.complex128)
        if coefficients.ndim != 2 or coefficients.shape[1] != self.n_terms or coefficients.shape[0] < 1:
            raise ValueError(f'{self.label()}: coefficients must be (n_shifts >= 1, {self.n_terms}), got '
                             f'{coefficients.shape}')
        return coefficients

    def solve_shifted(self, coefficients, rhs):
        """Yield (start, solutions (n_chunk, m, n) complex128 host, relative residuals (n_chunk,), accepted (n_chunk,)
        bool) per chunk of shifts: (K + sum_i coefficients[s, i] T_i) x = rhs[j] for every rhs row (rhs (m, n), the same
        for all shifts). Rejected shifts (residual above accept_tolerance or non-finite) are the caller's to re-solve."""
        from ..core.backend import to_host
        cp = self._xp()
        coefficients = self._checked_coefficients(coefficients)
        rhs = cp.asarray(rhs, dtype=cp.complex128)
        if rhs.ndim != 2 or rhs.shape[1] != self.layout.n_unknowns:
            raise ValueError(f'{self.label()}: rhs must be (n_rhs, {self.layout.n_unknowns}), got {rhs.shape}')
        rhs3 = rhs[None]
        rhs_norms = self._rhs_norms(rhs3)
        chunk = self.chunk_size(coefficients.shape[0], rhs.shape[0])
        for start in range(0, coefficients.shape[0], chunk):
            solutions, relative = self._solve_chunk(cp.asarray(coefficients[start:start + chunk]), rhs3, rhs_norms)
            accepted = np.isfinite(relative) & (relative <= self.accept_tolerance)
            self.n_batches += 1
            self.n_shifts += relative.size
            self.n_rejected += int(np.count_nonzero(~accepted))
            if accepted.any():
                self.worst_accepted_residual = max(self.worst_accepted_residual, float(relative[accepted].max()))
            host_solutions = to_host(solutions)
            del solutions   # release the chunk's device arrays before the next chunk
            yield start, host_solutions, relative, accepted

    def factorize(self, coefficients) -> LineBlockFactor:
        """LineBlockFactor keeping the block inverses of every shift of `coefficients` (n_shifts, n_terms) in one batch
        (size it with chunk_size(n_shifts, 1)); its solve(rhs, index) sweeps and refines against that shift."""
        cp = self._xp()
        coefficients = self._checked_coefficients(coefficients)
        chunk = self._factor_chunk(cp.asarray(coefficients), kept=True)
        self.n_batches += 1
        self.n_shifts += coefficients.shape[0]
        self.n_kept_factors += 1
        return LineBlockFactor(self, chunk, coefficients.shape[0])

    def work_counters(self) -> dict:
        """Machine-independent work so far (summary 'solvers' record of its (physics, role))."""
        return {'batches': self.n_batches, 'shifts': self.n_shifts, 'refinement_sweeps': self.n_refinement_sweeps,
                'rejected_shifts': self.n_rejected, 'kept_factors': self.n_kept_factors,
                'factor_solves': self.n_factor_solves, 'rejected_factor_solves': self.n_rejected_solves,
                'batched_operators': self.n_batched_operators,
                'worst_accepted_residual': self.worst_accepted_residual}

    # --- internals ------------------------------------------------------------------------------------------------

    def _rhs_norms(self, rhs3):
        """(n_c or 1, m) ||b||_2 of each rhs, 1 for b = 0 (absolute residual: x = 0)."""
        cp = self._xp()
        norms = cp.linalg.norm(rhs3, axis=2)
        return cp.where(norms > 0.0, norms, 1.0)

    def _factor_solve(self, chunk, rhs):
        """(host solution (n,), relative residual, accepted) of one single-shift chunk for a host rhs (n,)."""
        from ..core.backend import to_host
        cp = self._xp()
        rhs = np.asarray(rhs, dtype=np.complex128).ravel()
        if rhs.size != self.layout.n_unknowns:
            raise ValueError(f'{self.label()}: rhs must have {self.layout.n_unknowns} values, got {rhs.size}')
        rhs3 = cp.asarray(rhs.reshape(1, 1, -1))
        solution, relative, _, _ = self._refine(chunk, rhs3, self._rhs_norms(rhs3))
        value = float(relative[0])
        accepted = math.isfinite(value) and value <= self.accept_tolerance
        self.n_factor_solves += 1
        if accepted:
            self.worst_accepted_residual = max(self.worst_accepted_residual, value)
        else:
            self.n_rejected_solves += 1
        return to_host(solution[0, 0]), value, accepted

    def _solve_chunk(self, coefficients, rhs3, rhs_norms):
        """(solutions (n_c, m, n) complex128 of the backend, worst relative residual per shift (n_c,) host) of one chunk."""
        start_time = time.perf_counter()
        chunk = self._factor_chunk(coefficients)
        factor_seconds = self._elapsed(start_time)
        solution, relative, sweeps, initial_relative = self._refine(chunk, rhs3, rhs_norms)
        del chunk
        if self.debug:
            self.batch_log.append({'shifts': coefficients.shape[0], 'factor_seconds': factor_seconds,
                                   'solve_seconds': self._elapsed(start_time) - factor_seconds,
                                   'refinements': sweeps, 'initial_residual': float(np.nanmax(initial_relative)),
                                   'final_residual': float(np.nanmax(relative))})
        return solution, relative

    def _factor_chunk(self, coefficients, operators=None, kept=False):
        """_Chunk (equilibration scales, block inverses; kept: + G_k / H_k) of backend complex128 coefficients (n_c,
        n_terms); `operators`: the _ShiftOperators of a factorize_operators batch (coefficients (n_c, 0))."""
        cp = self._xp()
        layout = self.layout
        n_chunk = coefficients.shape[0]
        n_diagonal = int(self._terms.shape[0])
        if operators is not None:   # A_s itself in the operators: no diagonal shift to add (C6: no zero arrays)
            shift_flat = None
        elif n_diagonal:
            shift_flat = coefficients[:, :n_diagonal] @ self._terms   # (n_c, n) complex128 diagonal of sum_i s_i M_i
        else:
            shift_flat = cp.zeros((n_chunk, layout.n_unknowns), dtype=cp.complex128)
        matrix_coefficients = matrix_work = None
        if self._matrix_caches is not None:
            matrix_coefficients = cp.ascontiguousarray(coefficients[:, n_diagonal:])
            matrix_work = matrix_coefficients.astype(self.work_dtype)
        shift_padded = None
        if shift_flat is not None:
            shift_padded = cp.zeros((n_chunk, layout.n_blocks * layout.block_size), dtype=cp.complex128)
            shift_padded[:, self._slot] = shift_flat
            shift_padded = shift_padded.reshape(n_chunk, layout.n_blocks, layout.block_size)
        # equilibration s = 1 / sqrt(||K row|| + |shift| + sum_j |t_j| ||N_j row||), 1 on pads (and empty rows)
        row_norms = self._row_norms[None] if operators is None else operators.row_norms
        denominator = row_norms if shift_padded is None else row_norms + cp.abs(shift_padded)
        if matrix_coefficients is not None:
            denominator = denominator + cp.einsum('sj,jkb->skb', cp.abs(matrix_coefficients), self._matrix_row_norms)
        scale = (1.0 / cp.sqrt(cp.where(denominator > 0.0, denominator, 1.0))).astype(self.real_dtype)
        chunk = _Chunk(shift_flat, matrix_coefficients, matrix_work, scale, operators=operators)
        self._factor(chunk, None if shift_padded is None else shift_padded.astype(self.work_dtype), kept)
        return chunk

    def _refine(self, chunk, rhs3, rhs_norms):
        """(solutions (n_c, m, n) complex128, worst relative residual per shift (n_c,) host, sweeps, initial residuals)
        of rhs3 (n_c or 1, m, n): one sweep, then refinement while a shift is above tolerance and still improving."""
        cp = self._xp()
        n_chunk = chunk.scale.shape[0]
        right_sides = cp.broadcast_to(rhs3, (n_chunk,) + tuple(rhs3.shape[1:]))
        solution = self._from_padded(self._sweep(self._to_padded(right_sides), chunk))
        residual = self._residual(rhs3, solution, chunk)
        relative = self._relative(residual, rhs_norms)
        initial_relative = relative.copy()
        sweeps = 0
        while sweeps < self.max_refinements:
            open_shifts = relative >= self.tolerance   # NaN: not refined, rejected by the caller
            if not open_shifts.any():
                break
            candidate = solution + self._from_padded(self._sweep(self._to_padded(residual), chunk))
            candidate_residual = self._residual(rhs3, candidate, chunk)
            candidate_relative = self._relative(candidate_residual, rhs_norms)
            sweeps += 1
            improved = candidate_relative < relative
            keep = cp.asarray(improved)[:, None, None]
            solution = cp.where(keep, candidate, solution)
            residual = cp.where(keep, candidate_residual, residual)
            progressing = open_shifts & (candidate_relative < STALL_RATIO * relative)
            relative = np.where(improved, candidate_relative, relative)
            if not progressing.any():   # fp32 floor (or divergence) reached
                break
        self.n_refinement_sweeps += sweeps
        return solution, relative, sweeps, initial_relative

    def _elapsed(self, start_time):
        """Seconds since start_time (synchronised in debug mode only, else 0)."""
        if not self.debug:
            return 0.0
        from ..core.backend import synchronize
        synchronize(self.backend)
        return time.perf_counter() - start_time

    def _coupling(self, chunk, k, offset):
        """Block [k, offset] of the operator: the shared (b, b) K block, or (n_c, b, b) with the chunk's matrix terms or
        of its per-shift operators."""
        if chunk.operators is not None:
            return chunk.operators.cache[:, k, offset]
        block = self._cache[k, offset]
        if chunk.matrix_work is None:
            return block
        cp = self._xp()
        return block + cp.einsum('sj,jab->sab', chunk.matrix_work, self._matrix_caches[:, k, offset])

    def _factor(self, chunk, shift_padded, kept=False):
        """Block Thomas forward elimination of S (K + shift) S into the chunk: the inverses D_k^-1 (n_c, n_blocks, b,
        b); kept: also G_k = D_k^-1 L_k and H_k = D_k^-1 U_k (the coupling the elimination forms anyway); shift_padded
        None: no diagonal shift (a factorize_operators batch)."""
        cp = self._xp()
        scale = chunk.scale
        n_chunk, n_blocks, block_size = scale.shape
        diagonal = cp.arange(block_size)
        shape = (n_chunk, n_blocks, block_size, block_size)
        inverses = cp.empty(shape, dtype=self.work_dtype)
        kept_lower = cp.zeros(shape, dtype=self.work_dtype) if kept else None
        kept_upper = cp.zeros(shape, dtype=self.work_dtype) if kept else None
        coupling = None   # C_{k-1} = D_{k-1}^-1 A_{k-1,2}
        for k in range(n_blocks):
            block = cp.empty((n_chunk, block_size, block_size), dtype=self.work_dtype)
            block[...] = self._coupling(chunk, k, 1)
            block[:, diagonal, diagonal] += (self._pad_identity[k] if shift_padded is None
                                             else shift_padded[:, k] + self._pad_identity[k])
            block *= scale[:, k, :, None]
            block *= scale[:, k, None, :]
            lower = None
            if k > 0:
                lower = scale[:, k, :, None] * self._coupling(chunk, k, 0) * scale[:, k - 1, None, :]
                block -= cp.matmul(lower, coupling)
            inverses[:, k] = self._invert(block)
            if kept and lower is not None:
                kept_lower[:, k] = cp.matmul(inverses[:, k], lower)
            if k < n_blocks - 1:
                upper = scale[:, k, :, None] * self._coupling(chunk, k, 2) * scale[:, k + 1, None, :]
                coupling = cp.matmul(inverses[:, k], upper)
                if kept:
                    kept_upper[:, k] = coupling
        chunk.inverses, chunk.lower, chunk.upper = inverses, kept_lower, kept_upper

    def _invert(self, block):
        """Inverses of a (n_c, b, b) stack: batched for small b (always on the cpu: LAPACK per matrix), a loop of 2D
        inverses above BATCHED_INVERSE_MAX_BLOCK on the gpu."""
        cp = self._xp()
        if self.backend == 'cpu' or block.shape[-1] <= BATCHED_INVERSE_MAX_BLOCK:
            return cp.linalg.inv(block)
        return cp.stack([cp.linalg.inv(matrix) for matrix in block])

    def _sweep(self, rhs_padded, chunk):
        """x = S (S A S)^-1 S r for padded r (n_c, n_blocks, b, m): forward and backward block substitution (a kept
        factor: on its pre-multiplied couplings, _kept_sweep)."""
        if chunk.lower is not None:
            return self._kept_sweep(rhs_padded, chunk)
        cp = self._xp()
        inverses = chunk.inverses
        n_blocks = self.layout.n_blocks
        scale_columns = chunk.scale[..., None]
        solution = rhs_padded * scale_columns
        for k in range(n_blocks):
            block = solution[:, k]   # view: in-place updates skip the copy-back of `solution[:, k] -= ...`
            if k > 0:
                block -= scale_columns[:, k] * cp.matmul(self._coupling(chunk, k, 0),
                                                         scale_columns[:, k - 1] * solution[:, k - 1])
            block[...] = cp.matmul(inverses[:, k], block)
        for k in range(n_blocks - 2, -1, -1):
            coupled = scale_columns[:, k] * cp.matmul(self._coupling(chunk, k, 2),
                                                      scale_columns[:, k + 1] * solution[:, k + 1])
            block = solution[:, k]
            block -= cp.matmul(inverses[:, k], coupled)
        solution *= scale_columns
        return solution

    def _kept_sweep(self, rhs_padded, chunk):
        """The sweep on a kept factor: y = S r, y_k = D_k^-1 y_k - G_k y_(k-1) forward, y_k -= H_k y_(k+1) backward,
        x = S y; a single rhs on the gpu whose stacks fit KEPT_SWEEP_KERNEL_MAX_BYTES in one launch, else the loop."""
        cp = self._xp()
        scale_columns = chunk.scale[..., None]
        scaled = cp.ascontiguousarray(rhs_padded * scale_columns)
        solution = cp.empty_like(scaled)
        n_chunk, n_blocks, block_size, n_rhs = scaled.shape
        stack_bytes = KEPT_BLOCK_STACKS * n_blocks * block_size * block_size * np.dtype(self.work_dtype).itemsize
        arguments = (np.int32(n_blocks), np.int32(block_size), chunk.inverses, chunk.lower, chunk.upper, scaled,
                     solution)
        if self.backend == 'cpu' or (n_rhs == 1 and stack_bytes <= KEPT_SWEEP_KERNEL_MAX_BYTES
                                     and all(array.flags.c_contiguous for array in arguments[2:5])):
            _kept_sweep_twin()((n_chunk,), (KEPT_SWEEP_THREADS,), arguments)   # cpu: the numpy loop twin
        else:
            _kept_sweep_loop(*arguments)
        solution *= scale_columns
        return solution

    def _to_padded(self, flat):
        """(n_c, m, n) -> zero-padded (n_c, n_blocks, b, m) in the work dtype."""
        cp = self._xp()
        layout = self.layout
        n_chunk, n_rhs, _ = flat.shape
        padded = cp.zeros((n_chunk, n_rhs, layout.n_blocks * layout.block_size), dtype=self.work_dtype)
        padded[:, :, self._slot] = flat.astype(self.work_dtype)
        return cp.ascontiguousarray(padded.reshape(n_chunk, n_rhs, layout.n_blocks, layout.block_size)
                                    .transpose(0, 2, 3, 1))

    def _from_padded(self, padded):
        """(n_c, n_blocks, b, m) -> (n_c, m, n) complex128."""
        n_chunk, n_blocks, block_size, n_rhs = padded.shape
        flat = padded.transpose(0, 3, 1, 2).reshape(n_chunk, n_rhs, n_blocks * block_size)
        return flat[:, :, self._slot].astype(np.complex128)

    def _residual(self, rhs3, solution, chunk):
        """rhs - (K + diag(shift) + sum_j t_j N_j) x (a factorize_operators batch: rhs - A_s x) for every shift and rhs,
        complex128 (n_c, m, n)."""
        cp = self._xp()
        if chunk.operators is not None:
            return rhs3 - cp.stack([cp.asarray(operator @ cp.ascontiguousarray(solution[index].T)).T
                                    for index, operator in enumerate(chunk.operators.device)])
        n_chunk, n_rhs, n_unknowns = solution.shape
        columns = cp.ascontiguousarray(solution.reshape(n_chunk * n_rhs, n_unknowns).T)
        product = cp.asarray(self._operator_device @ columns).T.reshape(n_chunk, n_rhs, n_unknowns)
        for index, term in enumerate(self._matrix_device):
            product = product + (chunk.matrix_coefficients[:, index, None, None]
                                 * cp.asarray(term @ columns).T.reshape(n_chunk, n_rhs, n_unknowns))
        return rhs3 - product - chunk.shift_flat[:, None, :] * solution

    def _relative(self, residual, rhs_norms):
        """Worst ||r|| / ||b|| over the rhs of each shift, host float64 (n_c,)."""
        from ..core.backend import to_host
        cp = self._xp()
        return to_host((cp.linalg.norm(residual, axis=2) / rhs_norms).max(axis=1))


# --- kept-factor sweep (core.kernels raw twin 'solvers.line_block_kept_sweep') ---------------------------------------

# one CUDA block per shift, one warp per row (lanes over the columns: coalesced rows of the row-major blocks), the
# blocks of the line in sequence (__syncthreads between them); rhs / out (n_c, n_blocks, b) with one rhs
_KEPT_SWEEP_SOURCE = r'''
__device__ __forceinline__ void multiply_add(const cplx a, const cplx v, real& re, real& im) {
    re += a.x * v.x - a.y * v.y;
    im += a.x * v.y + a.y * v.x;
}

__device__ __forceinline__ void warp_sum(real& re, real& im) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        re += __shfl_down_sync(0xffffffffu, re, offset);
        im += __shfl_down_sync(0xffffffffu, im, offset);
    }
}

extern "C" __global__ void line_block_kept_sweep(int n_blocks, int b, const cplx* inverses, const cplx* lower,
                                                 const cplx* upper, const cplx* rhs, cplx* out) {
    const long long bb = (long long)b * b;
    const long long shift = blockIdx.x;
    inverses += shift * n_blocks * bb;
    lower += shift * n_blocks * bb;
    upper += shift * n_blocks * bb;
    rhs += shift * n_blocks * (long long)b;
    out += shift * n_blocks * (long long)b;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, n_warps = blockDim.x >> 5;
    for (int k = 0; k < n_blocks; ++k) {   // y_k = D_k^-1 r_k - G_k y_(k-1)
        const cplx* y = rhs + (long long)k * b;
        const cplx* previous = out + (long long)(k > 0 ? k - 1 : 0) * b;
        for (int i = warp; i < b; i += n_warps) {
            const cplx* inverse_row = inverses + k * bb + (long long)i * b;
            const cplx* lower_row = lower + k * bb + (long long)i * b;
            real re = 0, im = 0, lower_re = 0, lower_im = 0;
            for (int j = lane; j < b; j += 32) {
                multiply_add(inverse_row[j], y[j], re, im);
                if (k > 0) multiply_add(lower_row[j], previous[j], lower_re, lower_im);
            }
            re -= lower_re;
            im -= lower_im;
            warp_sum(re, im);
            if (lane == 0) { cplx value; value.x = re; value.y = im; out[(long long)k * b + i] = value; }
        }
        __syncthreads();
    }
    for (int k = n_blocks - 2; k >= 0; --k) {   // y_k -= H_k y_(k+1)
        const cplx* next = out + (long long)(k + 1) * b;
        for (int i = warp; i < b; i += n_warps) {
            const cplx* upper_row = upper + k * bb + (long long)i * b;
            real re = 0, im = 0;
            for (int j = lane; j < b; j += 32) multiply_add(upper_row[j], next[j], re, im);
            warp_sum(re, im);
            if (lane == 0) { out[(long long)k * b + i].x -= re; out[(long long)k * b + i].y -= im; }
        }
        __syncthreads();
    }
}
'''
_KEPT_SWEEP_TYPES = {np.dtype(np.complex64): ('float2', 'float'), np.dtype(np.complex128): ('double2', 'double')}
_KEPT_SWEEP_KERNELS = {}   # work dtype -> cupy RawKernel


def _kept_sweep_gpu(grid, block, arguments):
    """The CUDA sweep, compiled per work dtype (complex64 / complex128) on first use."""
    import cupy as cp
    dtype = np.dtype(arguments[2].dtype)
    kernel = _KEPT_SWEEP_KERNELS.get(dtype)
    if kernel is None:
        complex_type, real_type = _KEPT_SWEEP_TYPES[dtype]
        kernel = _KEPT_SWEEP_KERNELS[dtype] = cp.RawKernel(
            f'typedef {complex_type} cplx;\ntypedef {real_type} real;\n' + _KEPT_SWEEP_SOURCE, 'line_block_kept_sweep')
    kernel(grid, block, arguments)


def _kept_sweep_loop(n_blocks, block_size, inverses, lower, upper, rhs, out):
    """The kept sweep as a block loop on the arrays' module (the numpy twin; gpu stacks above the kernel cap, several
    rhs): rhs / out (n_c, n_blocks, b, m) scaled, 3 products per block."""
    from ..core.backend import array_module_of
    xp = array_module_of(out)
    for k in range(int(n_blocks)):
        out[:, k] = xp.matmul(inverses[:, k], rhs[:, k])
        if k > 0:
            out[:, k] -= xp.matmul(lower[:, k], out[:, k - 1])
    for k in range(int(n_blocks) - 2, -1, -1):
        out[:, k] -= xp.matmul(upper[:, k], out[:, k + 1])


def _kept_sweep_twin():
    """The registered raw twin (gpu kernel, numpy loop), registered on first use (core imports stay in functions)."""
    from ..core.kernels import REGISTRY, raw_twin
    found = REGISTRY.get('solvers.line_block_kept_sweep')
    return found if found is not None else raw_twin('solvers.line_block_kept_sweep', _kept_sweep_gpu, _kept_sweep_loop)
