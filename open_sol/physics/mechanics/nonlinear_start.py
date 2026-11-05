"""Start states of the total-Lagrangian Newton of the nonlinear solids (2D triangles, 3D tetrahedra): a sweep seed of
another build, the last converged state of the same assembly or u = 0, the load ramp from that state, and the cold-ramp
fallback when a warm start does not converge (plan C2). numpy only."""
import numpy as np

COLD_START = 'cold start'
SEED_START = 'warm start (sweep seed)'
LAST_START = 'warm start (last solution)'
FAILED_START = 'warm start failed: cold ramp'
# what a failed warm start may raise inside the Newton loop (a non-finite update, a singular or rejected factor)
_WARM_FAILURES = (ArithmeticError, RuntimeError, ValueError)


class NonlinearStart:
    """Warm-start record of one nonlinear solid on its solver vectors (flat, solver numbering): `seed` (with_loads, u,
    external load, held values) of another build, taken once by a solve of the same kind; `last` the latest converged
    state of this assembly (the owner calls forget_last at every assembly: the solid / held sets may change)."""

    def __init__(self):
        self.seed = None
        self.last = None
        self.note = ''   # why the last sweep seed was ignored ('' = none)

    def set_seed(self, with_loads, displacement, external, held_values) -> None:
        self.seed = (bool(with_loads), *(np.array(part, dtype=np.float64).reshape(-1)
                                         for part in (displacement, external, held_values)))
        self.note = ''

    def forget_last(self) -> None:
        self.last = None

    def remember(self, with_loads, state, external, held_values) -> None:
        self.last = (bool(with_loads), *(np.array(part, dtype=np.float64).reshape(-1)
                                         for part in (state, external, held_values)))

    def candidate(self, with_loads, size):
        """(origin, u, external, held values) to start a solve of this kind from: the seed (taken), else the last
        converged state; None = a cold start. A seed of another size is dropped with a note."""
        for origin, entry in ((SEED_START, self.seed), (LAST_START, self.last)):
            if entry is None or entry[0] != bool(with_loads):
                continue
            if origin == SEED_START:
                self.seed = None
            if all(part.shape == (size,) for part in entry[1:]):
                return (origin,) + tuple(entry[1:])
            if origin == SEED_START:
                self.note = f'the sweep seed has {entry[1].size} unknowns, this solid {size}: ignored'
        return None


def ramp(newton, load_steps, propose_for, scale, state, start_load, start_held, external, target) -> tuple:
    """Newton (core.damped_iteration) on state in place over load_steps increments from (start_load, start_held) to
    (external, target); propose_for(load, held values) -> propose. -> (converged every step, warning or None)."""
    converged, warning = True, None
    for step in range(1, load_steps + 1):
        fraction = step / load_steps
        load = start_load + fraction * (external - start_load)
        held_values = start_held + fraction * (target - start_held)
        if not newton.run(state, propose_for(load, held_values), scale):
            converged, warning = False, f'load step {step} of {load_steps}: {newton.last_warning}'
    return converged, warning


def solve_from_start(start, with_loads, newton, load_steps, propose_for, scale, external, target,
                     on_cold_ramp=None, active=None) -> tuple:
    """(state (n,), (converged, warning, Newton iterations, origin text)): the Newton ramp from start.candidate (the
    warm ramp starts at its load and held values; active: bool (n,), a candidate's u zeroed off it), on failure
    (unconverged or raising) on_cold_ramp() and the cold ramp from u = 0 (the failed attempt's unconverged count is
    taken back); a converged state is remembered."""
    external = np.asarray(external, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    size = external.size
    candidate = start.candidate(with_loads, size)
    iterations_before = newton.total_iterations
    origin = COLD_START if not start.note else f'{COLD_START} ({start.note})'
    start.note = ''
    if candidate is not None:
        origin, initial, start_load, start_held = candidate
        state = initial.copy()
        if active is not None:   # a seed of another solid set on the same mesh: values off this solid never move
            state[~active] = 0.0
        unconverged_before = newton.n_unconverged
        try:
            converged, warning = ramp(newton, load_steps, propose_for, scale, state, start_load, start_held, external,
                                      target)
        except _WARM_FAILURES:
            converged = False
        if converged:
            start.remember(with_loads, state, external, target)
            return state, (True, None, newton.total_iterations - iterations_before, origin)
        newton.n_unconverged = unconverged_before   # the cold ramp decides convergence
        origin = FAILED_START
        if on_cold_ramp is not None:
            on_cold_ramp()
    state = np.zeros(size)
    zeros = np.zeros(size)
    converged, warning = ramp(newton, load_steps, propose_for, scale, state, zeros, zeros, external, target)
    if converged:
        start.remember(with_loads, state, external, target)
    return state, (converged, warning, newton.total_iterations - iterations_before, origin)
