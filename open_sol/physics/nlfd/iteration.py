"""Solve schedules of the NLFD-chi channels: the undepleted rank-ordered cascade, the pump-depleted block Gauss-Seidel
fixed point with under-relaxation, Anderson acceleration and source continuation (plan 5.1, 5.2 stage 3a), and the
pump-depleted Newton-Krylov solve (stage 3b)."""
# Pump-depleted fixed point x -> G(x): x = the ChannelFieldStore vector (what the nonlinear sources read), G = one sweep
# solving every channel in solve order with the newest stored fields (Kerr operator terms updated before their channel),
# each solve captured into the store. Mixing (relaxation, Anderson) writes the store only, never a channel solution, and
# the loop stops right after a sweep: the final store and every channel solution come from that sweep.
# Newton-Krylov: root of F(x) = x - G_J(x), G_J = every channel solved from the same stored x (block Jacobi, the
# channels' own factors as left preconditioner). The sources are polynomials in E and conj(E), so F is real-linear
# differentiable only: GMRES runs in the real inner product Re <u, v> on the complex vectors (= GMRES on the real 2N
# system), J v = v - dG_J(x)[v] exact (sources.NewtonLinearization: one solve per channel with a tangent source), per
# channel scaled by 1 / ||G_k|| of the step's first evaluation (re-weighted once a channel 0 there has grown: generated
# channels from x = 0); Eisenstat-Walker forcing, Armijo backtracking on the scaled ||F||. Kerr stays in the rhs (no lagged operator term): every solve reuses the channel factor.
import math
from dataclasses import dataclass

# Vectors are the run backend's (the store's module: cupy or numpy); every function takes the module of its inputs.
import numpy as np

from ...core.backend import array_module_of, to_host

_LSTSQ_RCOND = 1e-12   # Anderson least squares: singular values below this x the largest are dropped
ITERATION_METHODS = ('picard', 'newton')
GMRES_RESTART = 20           # Krylov vectors per GMRES cycle (each a store-sized complex128 device vector)
GMRES_MAX_ITERATIONS = 60    # Jacobian products per Newton step at most (each one solve per nonlinear channel)
NEWTON_BUFFERS = 12          # store-sized vectors of the Newton loop besides the Krylov basis (memory estimate)
_FORCING_INITIAL, _FORCING_MAX, _FORCING_GAMMA = 0.1, 0.5, 0.9   # Eisenstat-Walker choice 2 (eta_0, eta_max, gamma)
_ARMIJO = 1e-4               # sufficient decrease of the scaled ||F|| per unit step length
_MAX_BACKTRACKS = 8          # step halvings before a step is taken anyway


@dataclass(frozen=True)
class CascadeRecord:
    """Summary of one nonlinear solve (study log / summary.json 'nonlinear_solve')."""
    mode: str                 # 'undepleted' | 'pump_depleted'
    order: tuple              # channel names in solve order
    channel_solves: int       # FDFD solves (each one back-substitution, a recycled GMRES or a factorisation)
    sweeps: int               # passes over the channels (undepleted: 1; newton: Newton steps incl. the last check)
    converged: bool = True
    residual: float = 0.0     # last max over channels ||x_new - x|| / ||x_new|| (undepleted: 0)
    tolerance: float = 0.0
    residual_history: tuple = ()
    continuation_steps: int = 1
    method: str = ''          # pump_depleted: 'picard' | 'newton' ('' undepleted)
    gmres_iterations: int = 0     # newton: Jacobian products summed over the steps
    backtracks: int = 0           # newton: step halvings of the line search


@dataclass(frozen=True)
class PumpDepletedSettings:
    """Iteration settings of coupling_mode 'pump_depleted' (ValueError on invalid values)."""
    tolerance: float = 1e-8
    max_iterations: int = 50
    relaxation: float = 1.0
    anderson_depth: int = 5
    continuation_steps: int = 1
    iteration_method: str = 'picard'

    def __post_init__(self):
        if self.iteration_method not in ITERATION_METHODS:
            raise ValueError(f"iteration_method must be one of {', '.join(ITERATION_METHODS)}, got "
                             f"{self.iteration_method!r}")
        if not (math.isfinite(self.tolerance) and self.tolerance > 0.0):
            raise ValueError(f'tolerance must be finite and > 0, got {self.tolerance!r}')
        if not (0.0 < self.relaxation <= 1.0):
            raise ValueError(f'relaxation must be in (0, 1], got {self.relaxation!r}')
        for name in ('max_iterations', 'continuation_steps'):
            if int(getattr(self, name)) != getattr(self, name) or getattr(self, name) < 1:
                raise ValueError(f'{name} must be an integer >= 1, got {getattr(self, name)!r}')
        if int(self.anderson_depth) != self.anderson_depth or self.anderson_depth < 0:
            raise ValueError(f'anderson_depth must be an integer >= 0, got {self.anderson_depth!r}')


def solve_undepleted(channel_physics, channels, order, store, before_channel=None):
    """Solve every channel once in `order` (mixing.solve_order: sources rank below their targets), capturing each into
    `store`; before_channel(channel) runs before each solve (study progress; raising there cancels) -> CascadeRecord."""
    for index in order:
        if before_channel is not None:
            before_channel(channels[index])
        physics = channel_physics[index]
        physics.solve_harmonic(physics.omega)
        store.capture(index, physics)
    return CascadeRecord(mode='undepleted', order=tuple(channels[index].name for index in order),
                         channel_solves=len(order), sweeps=1)


class ScaledSource:
    """A linear channel source times `scale` (continuation ramp); every other attribute passes through."""

    def __init__(self, source, scale):
        self._source = source
        self.scale = float(scale)

    def __getattr__(self, name):
        return getattr(self._source, name)

    def add_to_rhs(self, rhs, physics, system):
        part = array_module_of(rhs).zeros_like(rhs)
        self._source.add_to_rhs(part, physics, system)
        part *= self.scale
        rhs += part

    def face_currents(self, physics, system):
        if not hasattr(self._source, 'face_currents'):
            return None, None
        return tuple(None if faces is None else self.scale * faces
                     for faces in self._source.face_currents(physics, system))


def _real_view(vector):
    """float64 view (real, imaginary parts interleaved) of a contiguous complex128 vector: Re <u, v> = view dot."""
    return vector.view(array_module_of(vector).float64)


def block_norms(vectors, blocks):
    """[[||vector[block]|| per channel block] per complex128 store-layout vector] as host floats, one device -> host
    transfer for all of them."""
    if not vectors or not blocks:
        return [[] for _ in vectors]
    xp = array_module_of(vectors[0])
    squares = []
    for vector in vectors:
        real = _real_view(xp.ascontiguousarray(vector))
        for block in blocks:
            part = real[2 * block.start:2 * block.stop]
            squares.append(part @ part)
    host = to_host(xp.stack(squares))
    count = len(blocks)
    return [[math.sqrt(max(float(value), 0.0)) for value in host[start:start + count]]
            for start in range(0, len(host), count)]


def channel_residual(change, values, blocks):
    """(max over channels ||change_k|| / ||values_k|| (0 for a channel staying 0), [||values_k||]) (one host sync)."""
    norms, differences = block_norms((values, change), blocks)
    ratios = [difference / size if size > 0.0 else (0.0 if difference == 0.0 else math.inf)
              for size, difference in zip(norms, differences)]
    return (max(ratios) if ratios else 0.0), norms


def block_weights(norms):
    """Per-channel weights 1 / ||g_k||; a zero channel takes the largest of them (1 when every channel is 0), never 1
    itself: fields are ~1e8 V/m, so weight 1 would make a generated channel dominate every scaled norm."""
    weights = [1.0 / size for size in norms if size > 0.0]
    fallback = max(weights) if weights else 1.0
    return [1.0 / size if size > 0.0 else fallback for size in norms]


def _set_continuation_sources(channel_physics, originals, step, steps, nonlinear_types):
    """Every channel's linear sources scaled by step / steps (the originals at the last step)."""
    for physics, sources in zip(channel_physics, originals):
        physics.sources = sources if step == steps else [
            source if isinstance(source, nonlinear_types) else ScaledSource(source, step / steps)
            for source in sources]


class AndersonMixer:
    """Anderson acceleration (Walker-Ni form, depth m; 0 = relaxed Picard) of the store fixed point, least squares
    weighted per channel by 1 / ||g_k|| of the first sweep, relaxation beta: x+ = x + beta f - (dX + beta dF) gamma."""

    def __init__(self, blocks, depth, relaxation):
        self.blocks = tuple(blocks)
        self.depth = int(depth)
        self.relaxation = float(relaxation)
        self._x = self._f = self._x_previous = self._f_previous = None
        self._dx = self._df = None    # (depth, N) ring buffers of differences
        self._count = self._next = 0
        self._weights = None

    def begin(self, values):
        """Copy the stored fields x before a sweep."""
        xp = array_module_of(values)
        if self._x is None:
            self._x = xp.empty_like(values)
        xp.copyto(self._x, values)

    def residual(self, values):
        """max over channels ||g_k - x_k|| / ||g_k|| of the swept fields g (0 for a channel staying 0); sets f."""
        xp = array_module_of(values)
        if self._f is None:
            self._f = xp.empty_like(values)
        xp.subtract(values, self._x, out=self._f)
        residual, norms = channel_residual(self._f, values, self.blocks)
        if self._weights is None:
            self._weights = block_weights(norms)
        return residual

    def update(self):
        """Next stored fields (new vector), or None to keep the swept fields (beta 1 without history)."""
        x, f = self._x, self._f
        xp = array_module_of(x)
        if self.depth > 0:
            if self._f_previous is not None:
                if self._df is None:
                    self._df = xp.empty((self.depth, x.size), dtype=x.dtype)
                    self._dx = xp.empty((self.depth, x.size), dtype=x.dtype)
                column = self._next % self.depth
                xp.subtract(f, self._f_previous, out=self._df[column])
                xp.subtract(x, self._x_previous, out=self._dx[column])
                self._next += 1
                self._count = min(self._count + 1, self.depth)
            # keep x, f as the previous pair; their old buffers take the next sweep's x, f
            self._x, self._x_previous = self._x_previous, x
            self._f, self._f_previous = self._f_previous, f
        if self._count == 0:
            return None if self.relaxation == 1.0 else x + self.relaxation * f
        df, dx = self._df[:self._count], self._dx[:self._count]
        gamma = xp.asarray(self._coefficients(df, f))
        # gamma is real: the history combination is two real GEMVs on the float64 views (no (depth, N) temporary)
        mixed = xp.multiply(f, self.relaxation)
        mixed += x
        mixed_real = _real_view(mixed)
        mixed_real -= gamma @ _real_view(dx)
        mixed_real -= (self.relaxation * gamma) @ _real_view(df)
        return mixed

    def _coefficients(self, df, f):
        """Real gamma of min || W (f - dF^T gamma) || (host normal equations of Re <u, v>, blockwise weighted Gram from
        real-view GEMMs, one transfer): the map conjugates fields (only real-linear), so complex gamma would mix
        differences by phases it does not respect."""
        xp = array_module_of(df)
        count = df.shape[0]
        df_real, f_real = _real_view(df), _real_view(f)
        gram = xp.zeros((count, count), dtype=xp.float64)
        rhs = xp.zeros(count, dtype=xp.float64)
        for block, weight in zip(self.blocks, self._weights):
            part = slice(2 * block.start, 2 * block.stop)
            columns = df_real if len(self.blocks) == 1 else xp.ascontiguousarray(df_real[:, part])
            gram += (weight ** 2) * (columns @ columns.T)
            rhs += (weight ** 2) * (columns @ f_real[part])
        system = to_host(xp.concatenate((gram.ravel(), rhs)))
        return np.linalg.lstsq(system[:count * count].reshape(count, count), system[count * count:],
                               rcond=_LSTSQ_RCOND)[0]


def solve_pump_depleted(channel_physics, channels, order, store, kerr_terms, settings, nonlinear_types,
                        before_channel=None, after_sweep=None):
    """Block Gauss-Seidel sweeps in `order` to the fixed point of every mixing term (store residual <= tolerance),
    per continuation step with the linear sources scaled by step / steps (warm-started); kerr_terms {index:
    KerrOperatorTerm} update before their channel's solve; channels record their monitors once at the end.
    before_channel(channel) before each solve, after_sweep(step, sweep, residual) after each sweep -> CascadeRecord;
    RuntimeError for non-finite fields."""
    originals = [list(physics.sources) for physics in channel_physics]
    history, solves, sweeps, residual, converged = [], 0, 0, math.inf, False
    steps = settings.continuation_steps
    try:
        for step in range(1, steps + 1):
            _set_continuation_sources(channel_physics, originals, step, steps, nonlinear_types)
            mixer = AndersonMixer(store.blocks, settings.anderson_depth, settings.relaxation)
            converged = False
            for sweep in range(1, settings.max_iterations + 1):
                mixer.begin(store.vector())
                for index in order:
                    if before_channel is not None:
                        before_channel(channels[index])
                    physics = channel_physics[index]
                    if index in kerr_terms:
                        kerr_terms[index].update(physics)
                    physics._solve(record_monitors=False)
                    store.capture(index, physics)
                    solves += 1
                sweeps += 1
                residual = mixer.residual(store.vector())
                history.append(residual)
                if after_sweep is not None:
                    after_sweep(step, sweep, residual)
                if not math.isfinite(residual):
                    raise RuntimeError(f'pump-depleted iteration: the channel fields became non-finite in sweep {sweep} '
                                       f'of continuation step {step}; lower the source amplitudes, set relaxation < 1 '
                                       'or use continuation_steps > 1')
                if residual <= settings.tolerance:
                    converged = True
                    break
                if sweep == settings.max_iterations:
                    break
                mixed = mixer.update()
                if mixed is not None:
                    store.write(mixed)
                del mixed
    finally:
        for physics, sources in zip(channel_physics, originals):
            physics.sources = sources
    for physics in channel_physics:
        physics.record_monitors()
    return CascadeRecord(mode='pump_depleted', order=tuple(channels[index].name for index in order),
                         channel_solves=solves, sweeps=sweeps, converged=converged, residual=float(residual),
                         tolerance=float(settings.tolerance), residual_history=tuple(float(value) for value in history),
                         continuation_steps=steps, method='picard')


def _real_dot(first, second):
    """Re <first, second> of two complex128 vectors (host float; real-view dot)."""
    xp = array_module_of(first)
    return float(_real_view(xp.ascontiguousarray(first)) @ _real_view(xp.ascontiguousarray(second)))


def real_gmres(apply, rhs, relative_tolerance, restart=GMRES_RESTART, max_iterations=GMRES_MAX_ITERATIONS):
    """Restarted GMRES from 0 for a real-linear `apply` (complex128 vectors of the rhs' module, real inner product Re
    <u, v> = dots of the float64 views, real Hessenberg; classical Gram-Schmidt twice, one host transfer per column) ->
    (solution, products, relative residual estimate)."""
    xp = array_module_of(rhs)
    rhs_norm = math.sqrt(max(_real_dot(rhs, rhs), 0.0))
    solution = xp.zeros_like(rhs)
    if rhs_norm == 0.0:
        return solution, 0, 0.0
    basis = xp.empty((restart + 1, rhs.size), dtype=xp.complex128)
    basis_real = _real_view(basis)   # (restart + 1, 2 N): Re <v_i, w> = basis_real[i] . w_real
    residual_vector, products, relative = rhs, 0, 1.0
    while products < max_iterations:
        beta = math.sqrt(max(_real_dot(residual_vector, residual_vector), 0.0))
        xp.multiply(residual_vector, 1.0 / beta, out=basis[0])
        hessenberg = np.zeros((restart + 1, restart))
        columns, finished, least_squares = 0, False, np.zeros(0)
        for column in range(restart):
            work = xp.ascontiguousarray(apply(basis[column]), dtype=xp.complex128)
            products += 1
            work_real = _real_view(work)
            projections = None
            for _ in range(2):   # CGS2: re-orthogonalise once (row-major basis: GEMVs without transpose copies)
                projection = basis_real[:column + 1] @ work_real   # Re <v_i, w>
                work_real -= projection @ basis_real[:column + 1]
                projections = projection if projections is None else projections + projection
            coefficients = to_host(xp.concatenate((projections, xp.reshape(work_real @ work_real, (1,)))))
            hessenberg[:column + 1, column] = coefficients[:-1]
            norm = math.sqrt(max(float(coefficients[-1]), 0.0))
            hessenberg[column + 1, column] = norm
            columns = column + 1
            target = np.zeros(columns + 1)
            target[0] = beta
            least_squares = np.linalg.lstsq(hessenberg[:columns + 1, :columns], target, rcond=None)[0]
            remainder = target - hessenberg[:columns + 1, :columns] @ least_squares
            relative = float(np.linalg.norm(remainder)) / rhs_norm
            finished = relative <= relative_tolerance or norm <= 1e-14 * beta or products >= max_iterations
            if finished:
                break
            xp.multiply(work, 1.0 / norm, out=basis[column + 1])
            del work, work_real
        # real coefficients: real GEMVs on the float64 views (no complex cast of y)
        solution_real = _real_view(solution)
        solution_real += xp.asarray(least_squares) @ basis_real[:columns]
        del solution_real
        if finished:
            break
        # restart: r = V_{m+1} (beta e1 - H y) (Arnoldi relation, no extra product)
        residual_vector = (xp.asarray(remainder) @ basis_real[:columns + 1]).view(xp.complex128)
    return solution, products, relative


def _scale_blocks(vector, blocks, factors):
    """vector[block] *= factor per channel block, in place; returns vector."""
    for block, factor in zip(blocks, factors):
        vector[block] *= factor
    return vector


def _scaled_norm(vector, blocks, weights):
    """sqrt(sum_k w_k^2 ||v_k||^2) (one host sync)."""
    return math.sqrt(sum((weight * size) ** 2 for size, weight in zip(block_norms((vector,), blocks)[0], weights)))


def _forcing(previous, norm, previous_norm, tolerance):
    """Eisenstat-Walker choice 2 forcing term with its safeguards and the oversolving floor 0.5 tol / ||F||."""
    if previous_norm is None:
        eta = _FORCING_INITIAL
    else:
        eta = _FORCING_GAMMA * (norm / previous_norm) ** 2
        safeguard = _FORCING_GAMMA * previous ** 2
        if safeguard > 0.1:
            eta = max(eta, safeguard)
    eta = min(_FORCING_MAX, eta)
    return min(_FORCING_MAX, max(eta, 0.5 * tolerance / norm)) if norm > 0.0 else eta


def solve_newton_krylov(channel_physics, channels, order, store, linearization, settings, nonlinear_types,
                        before_channel=None, after_sweep=None):
    """Newton-Krylov root of x = G_J(x) per continuation step (warm-started): GMRES on J d = G_J(x) - x preconditioned
    by the channel factors, Armijo backtracking; stops when max over channels ||G_k - x_k|| / ||G_k|| <= tolerance,
    then the store takes G_J(x) (store and channel solutions agree) and the monitors record once. before_channel /
    after_sweep(step, newton step, residual) as solve_pump_depleted -> CascadeRecord; RuntimeError when non-finite."""
    originals = [list(physics.sources) for physics in channel_physics]
    blocks = store.blocks
    counters = {'solves': 0}
    history, sweeps, products, backtracks, residual, converged = [], 0, 0, 0, math.inf, False
    steps = settings.continuation_steps

    def solve_channel(index):
        if before_channel is not None:
            before_channel(channels[index])
        channel_physics[index]._solve(record_monitors=False)
        counters['solves'] += 1

    def evaluate(x, out):
        """out = G_J(x): the store takes x, every channel is solved from it (channel solutions = G_J(x))."""
        store.write(x)
        for index in order:
            solve_channel(index)
            store.read(index, channel_physics[index], out)

    def jacobian_product(tangent):
        """J v = v - dG_J(x)[v] at the stored x (one solve per channel with an applied phasor)."""
        context = linearization.tangent_context(tangent)
        product = tangent.copy()
        derivative = store.xp.empty_like(tangent)
        for index in order:
            source = linearization.source(index, context)
            if source is None:
                continue
            physics = channel_physics[index]
            sources = physics.sources
            physics.sources = [source]
            try:
                solve_channel(index)
            finally:
                physics.sources = sources
            store.read(index, physics, derivative)
            product[blocks[index]] -= derivative[blocks[index]]
            del source
        return product

    try:
        for step in range(1, steps + 1):
            _set_continuation_sources(channel_physics, originals, step, steps, nonlinear_types)
            x = store.vector().copy()
            values = store.xp.empty_like(x)
            evaluate(x, values)
            update = values - x                                  # G_J(x) - x = -F(x)
            residual, norms = channel_residual(update, values, blocks)
            weights = block_weights(norms)
            zero_channels = [size == 0.0 for size in norms]
            norm, previous_norm, eta = _scaled_norm(update, blocks, weights), None, _FORCING_INITIAL
            converged = False
            for newton_step in range(1, settings.max_iterations + 1):
                if newton_step > 1:
                    residual, norms = channel_residual(update, values, blocks)
                sweeps += 1
                history.append(residual)
                if after_sweep is not None:
                    after_sweep(step, newton_step, residual)
                if not math.isfinite(residual):
                    raise RuntimeError(f'Newton-Krylov iteration: the channel fields became non-finite in Newton step '
                                       f'{newton_step} of continuation step {step}; lower the source amplitudes or use '
                                       'continuation_steps > 1')
                if residual <= settings.tolerance or newton_step == settings.max_iterations:
                    converged = residual <= settings.tolerance
                    break
                if any(zero and size > 0.0 for zero, size in zip(zero_channels, norms)):
                    # a channel 0 at the first evaluation (a generated channel from x = 0) has grown: re-weight by the
                    # current sizes, keeping the forcing ratio ||F_k|| / ||F_k-1|| of the old weights
                    ratio = norm / previous_norm
                    weights = block_weights(norms)
                    zero_channels = [size == 0.0 for size in norms]
                    norm = _scaled_norm(update, blocks, weights)
                    previous_norm = norm / ratio
                eta = _forcing(eta, norm, previous_norm, settings.tolerance)
                inverse = [1.0 / weight for weight in weights]

                def scaled_product(scaled):
                    return _scale_blocks(jacobian_product(_scale_blocks(scaled.copy(), blocks, inverse)), blocks,
                                         weights)

                scaled_step, step_products, _ = real_gmres(scaled_product,
                                                           _scale_blocks(update.copy(), blocks, weights), eta)
                products += step_products
                direction = _scale_blocks(scaled_step, blocks, inverse)
                del scaled_step
                length = 1.0
                for attempt in range(_MAX_BACKTRACKS + 1):
                    trial = x + length * direction
                    evaluate(trial, values)
                    trial_update = values - trial
                    trial_norm = _scaled_norm(trial_update, blocks, weights)
                    if trial_norm <= (1.0 - _ARMIJO * length) * norm or attempt == _MAX_BACKTRACKS:
                        break
                    backtracks += 1
                    length *= 0.5
                    del trial, trial_update
                previous_norm, norm = norm, trial_norm
                x, update = trial, trial_update
                del direction
            for index in order:   # the store takes G_J(x): every channel solution comes from the last evaluation
                store.capture(index, channel_physics[index])
            del x, values, update
    finally:
        for physics, sources in zip(channel_physics, originals):
            physics.sources = sources
    for physics in channel_physics:
        physics.record_monitors()
    return CascadeRecord(mode='pump_depleted', order=tuple(channels[index].name for index in order),
                         channel_solves=counters['solves'], sweeps=sweeps, converged=converged,
                         residual=float(residual), tolerance=float(settings.tolerance),
                         residual_history=tuple(float(value) for value in history), continuation_steps=steps,
                         method='newton', gmres_iterations=products, backtracks=backtracks)
