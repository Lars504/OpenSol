"""KeptPreconditionerSolver: the 'linear' solver of a tensor saturation loop's DiffusionSystem that keeps the multigrid
hierarchy or host direct factor of an earlier loop iteration while the coefficients move little and solves each new
operator by PCG on it (fewer setups / factorisations; a missed solve rebuilds and repeats it)."""
import math
import time

from ...core.backend import array_module_of, to_host
from ...solvers.host_direct import quiet_openblas
from ...solvers.multigrid import MultigridPreconditioner
from ...solvers.pcg import PCGSolver
from ...solvers.sparse_direct import CUDSS_BACK_ENDS, DiffusionDirectSolver, SparseDirectSolver

# largest exp(max - min) of the change of ln c since the kept build: CG on the new operator with the kept
# preconditioner is at most sqrt(spread) slower (the face conductances are harmonic means of the cells' c, so their
# ratios stay in that range). A multigrid setup costs ~100 V-cycle iterations (4x a fresh solve's iterations is still
# cheaper); on the kept factor CG needs <= sqrt(3) / 2 ln(2 / 1e-10) ~ 21 solves, under the cap (a host factorisation
# costs ~20 solves, so a wider spread would refactorise anyway)
KEPT_SPREAD_LIMIT = {'multigrid': 16.0, 'direct': 3.0}
KEPT_DIRECT_TOLERANCE = 1e-10      # PCG on the kept factor (stands in for the exact solve), relative to ||b||
KEPT_DIRECT_MAX_ITERATIONS = 25    # a kept-factor solve missing it refactorises and solves exactly
KEPT_MULTIGRID_SLACK = 20          # cap of a kept-hierarchy solve: sqrt(spread limit) x the fresh solve's + this


def kept_preconditioner_solver(linear_solver):
    """The role runtime wrapped for a tensor saturation loop where keeping pays (pcg multigrid, sparse_direct), else
    unchanged (None, Jacobi / mixed pcg: their refresh is free; the AC assembly-only solver)."""
    if isinstance(linear_solver, SparseDirectSolver):
        return KeptPreconditionerSolver(DiffusionDirectSolver(linear_solver), 'direct')
    if (isinstance(linear_solver, PCGSolver) and linear_solver.preconditioner == 'multigrid'
            and linear_solver.precision != 'mixed'):
        return KeptPreconditionerSolver(linear_solver, 'multigrid')
    return linear_solver


class KeptPreconditionerSolver:
    """PCGSolver protocol of a DiffusionSystem over `inner` (a 'multigrid' PCGSolver or a DiffusionDirectSolver):
    factorize builds afresh unless the owner offered the loop state (offer_state) and it moved by at most
    KEPT_SPREAD_LIMIT since the kept build; then the new operator is solved by PCG on the kept hierarchy / factor."""

    _OWN = frozenset({'inner', 'kind', '_multigrid', '_kept_state', '_offer', '_bound_state', '_operator', '_kept',
                      '_fresh_iterations', '_kept_pcg', 'n_kept_solves', 'n_kept_fallbacks', 'kept_cg_iterations'})

    def __init__(self, inner, kind):
        self.inner = inner
        self.kind = kind                  # 'multigrid' or 'direct'
        self._multigrid = None            # multigrid: the hierarchy this wrapper owns (the pcg's own stays unbuilt)
        if kind == 'multigrid':
            self._multigrid = MultigridPreconditioner(inner.backend)
            inner.preconditioner_counters = self._multigrid.work_counters   # the role's summary counters
        self._kept_state = None           # loop state (float64 copy) the kept hierarchy / factor was built at
        self._offer = None                # (state, fresh) of the next assembly (offer_state), taken by factorize
        self._bound_state = None          # the state of the bound operator (None: no loop state)
        self._operator = None             # the bound operator (CSR of the backend)
        self._kept = False                # the bound operator is solved on the kept hierarchy / factor
        self._fresh_iterations = 0        # multigrid: CG iterations of the first solve after the kept build
        self._kept_pcg = None             # direct: the PCGSolver iterating on the kept factor
        self.n_kept_solves = 0            # solves on a kept hierarchy / factor (fallbacks included)
        self.n_kept_fallbacks = 0         # kept solves that missed and were repeated after a fresh build
        self.kept_cg_iterations = 0       # direct: PCG iterations of the kept-factor solves

    def __getattr__(self, name):          # tolerance, n_solves, last_iters, ... of the inner solver
        if name == 'inner':
            raise AttributeError(name)
        return getattr(self.inner, name)

    def __setattr__(self, name, value):   # label, compute_residual, tolerance, max_iterations land on the inner solver
        if name in self._OWN:
            object.__setattr__(self, name, value)
        else:
            setattr(self.inner, name, value)

    def bind_window(self, window_shape) -> None:
        """Window cell shape of the system (DiffusionSystem.prepare sets it only on a bare DiffusionDirectSolver): a
        3D SuperLU starts from nested dissection."""
        if self.kind == 'direct':
            self.inner.grid_shape = tuple(window_shape)

    def offer_state(self, state, fresh=False) -> None:
        """The loop state (ln of the coefficients up to a sign; device / host float64) of the next assembly: factorize
        keeps the earlier build while the state moved little since it; fresh (a loop's first assembly) builds afresh
        and keeps this state's build. Without an offer factorize builds afresh and keeps nothing."""
        self._offer = (state, bool(fresh))

    def work_counters(self) -> dict:
        """Kept solves, their fallbacks and (direct) PCG iterations (the physics adds them to its counters)."""
        counters = {'kept_preconditioner_solves': self.n_kept_solves,
                    'kept_preconditioner_fallbacks': self.n_kept_fallbacks}
        if self.kind == 'direct':
            counters['kept_factor_cg_iterations'] = self.kept_cg_iterations
        return counters

    # --- PCGSolver protocol -------------------------------------------------

    def factorize(self, operator, preconditioner=None, n_unknowns=None) -> None:
        """Bind the assembled CSR: keep the earlier hierarchy / factor when the offered state moved by at most
        KEPT_SPREAD_LIMIT since it was built (same operator size), else build afresh."""
        (state, fresh), self._offer = self._offer or (None, True), None
        previous, self._operator = self._operator, operator
        self._bound_state = state
        keep = (not fresh and state is not None and self._kept_state is not None and previous is not None
                and previous.shape == operator.shape and state.shape == self._kept_state.shape
                and not (self.kind == 'direct' and self.inner.runtime.back_end in CUDSS_BACK_ENDS)
                and _spread(state - self._kept_state) <= KEPT_SPREAD_LIMIT[self.kind])
        if keep:
            self._bind_kept(operator)
        else:
            self._build(operator)
        self._kept = keep

    def solve(self, rhs, solution, reference='rhs'):
        """Solve on the bound operator: a kept solve that misses its iteration cap rebuilds and repeats (exact on the
        direct kind); the first solve after a fresh multigrid build sets the cap of the kept ones."""
        if not self._kept:
            converged = self.inner.solve(rhs, solution, reference)
            if self.kind == 'multigrid':
                self._fresh_iterations = self.inner.last_iters
            return converged
        self.n_kept_solves += 1
        if self.kind == 'multigrid':
            return self._solve_kept_multigrid(rhs, solution, reference)
        return self._solve_kept_direct(rhs, solution, reference)

    # --- internals ------------------------------------------------------------

    def _build(self, operator) -> None:
        """Fresh hierarchy (its time counted in the pcg's factorize seconds) or factor of the operator; the bound state
        becomes the kept one."""
        state = self._bound_state
        self._kept_state = None if state is None else state.copy()
        if self.kind == 'direct':
            self.inner.factorize(operator)
            return
        start = time.perf_counter()
        self._multigrid.setup(operator)
        self.inner.total_factorize_seconds += time.perf_counter() - start
        self._bind_kept(operator)

    def _bind_kept(self, operator) -> None:
        """multigrid: the pcg binds the operator with the wrapper's hierarchy as its precondition callback; direct: a
        PCGSolver on the kept factor."""
        if self.kind == 'multigrid':
            _bind_callback(self.inner, operator, self._multigrid.precondition)
            return
        if self._kept_pcg is None:
            self._kept_pcg = PCGSolver('double', KEPT_DIRECT_TOLERANCE, KEPT_DIRECT_MAX_ITERATIONS,
                                       label=self.inner.label, backend=self.inner.backend)
        _bind_callback(self._kept_pcg, operator, self._apply_kept_factor)

    def _apply_kept_factor(self, residual, out) -> None:
        self.inner.solve(residual, out)

    def _solve_kept_multigrid(self, rhs, solution, reference):
        """CG on the kept hierarchy capped at sqrt(the spread limit) x the fresh solve's iterations +
        KEPT_MULTIGRID_SLACK; a miss is forgotten (solve count, warning) and repeated from its iterate after a fresh
        setup."""
        pcg = self.inner
        limit = pcg.max_iterations
        record = (pcg.n_solves, pcg.n_unconverged, pcg.last_warning)
        growth = math.sqrt(KEPT_SPREAD_LIMIT['multigrid'])
        pcg.max_iterations = min(limit, math.ceil(growth * self._fresh_iterations) + KEPT_MULTIGRID_SLACK)
        try:
            converged = pcg.solve(rhs, solution, reference)
        finally:
            pcg.max_iterations = limit
        if converged:
            return True
        pcg.n_solves, pcg.n_unconverged, pcg.last_warning = record
        return self._rebuild_and_solve(rhs, solution, reference)

    def _solve_kept_direct(self, rhs, solution, reference):
        """PCG on the kept factor to KEPT_DIRECT_TOLERANCE; the direct solver's record shows one solve with the PCG's
        iterations; a miss refactorises and solves exactly."""
        direct = self.inner
        solves = direct.n_solves
        with quiet_openblas(direct.runtime.back_end):   # a Krylov loop around a PARDISO factor (solvers sub file)
            converged = self._kept_pcg.solve(rhs, solution, 'rhs')
        self.kept_cg_iterations += self._kept_pcg.last_iters
        direct.n_solves = solves
        if not converged:
            return self._rebuild_and_solve(rhs, solution, reference)
        direct.n_solves = solves + 1
        direct.last_iters = self._kept_pcg.last_iters
        if direct.compute_residual:
            rhs_norm = float(array_module_of(rhs).linalg.norm(rhs))
            direct.last_residual = self._kept_pcg.relative_residual(rhs, solution, rhs_norm)
        return True

    def _rebuild_and_solve(self, rhs, solution, reference):
        """Fresh build of the bound operator (its state kept) and the solve on it, from the missed solve's iterate."""
        self.n_kept_fallbacks += 1
        self._build(self._operator)
        self._kept = False
        return self.solve(rhs, solution, reference)


def _bind_callback(pcg, operator, precondition) -> None:
    """pcg.factorize of a CSR with a precondition callback (accepted under the 'jacobi' setting), then the 'multigrid'
    setting: the check interval of an expensive preconditioner (a residual read every 1 - 2 iterations)."""
    pcg.preconditioner = 'jacobi'
    try:
        pcg.factorize(operator, precondition)
    finally:
        pcg.preconditioner = 'multigrid'


def _spread(change) -> float:
    """exp(max(change, 0) - min(change, 0)) of a device / host float64 change of ln c (one transfer)."""
    if change.size == 0:
        return 1.0
    xp = array_module_of(change)
    largest, smallest = (float(value) for value in to_host(xp.stack((change.max(), change.min()))))
    return math.exp(max(largest, 0.0) - min(smallest, 0.0))
