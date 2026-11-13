"""Host (numpy float64) drift-diffusion: Poisson + electron / hole continuity in scaled variables (u = psi / V_T, n / C_s,
p / C_s), Scharfetter-Gummel face fluxes, Newton with bias / generation continuation on a graph of control volumes and
faces (DriftDiffusionCore); DriftDiffusionSystem = the window cells of a tensor mesh (the triangle-mesh geometry is
drift_diffusion_fem.py); linear solves through the sparse direct back ends (solvers.sparse_direct)."""
import math

import numpy as np

from ...core.constants import ELEMENTARY_CHARGE, EPS0, K_B
from ...core.metric import CARTESIAN, window_metric
from ...solvers.cudss_factor import CUDSS_REFUSALS, CudssAccuracyError
from ...solvers.host_direct import HOST_DIRECT_BACK_ENDS, HOST_DIRECT_REFUSALS, HostDirectAccuracyError
from ...solvers.sparse_direct import CUDSS_BACK_ENDS, SparseDirectSolver

# back ends whose factor is kept per unknown count and refactorised through previous= (cuDSS plan / MUMPS / PARDISO
# analysis reuse); the others (scipy_splu, cusolver_qr) factorise afresh per Newton step
KEPT_FACTOR_BACK_ENDS = CUDSS_BACK_ENDS + HOST_DIRECT_BACK_ENDS
COMPONENTS = 3                   # unknowns per cell, interleaved: index 3 * cell + (0 u, 1 n, 2 p)
_SERIES_LIMIT = 1e-4              # |x| below which B(x) and B'(x) use their Taylor series
_MIN_STEP_FRACTION = 1.0 / 64.0   # continuation step (fraction of the first one) below which a continuation fails
_CARRIER_FLOOR = 1e-150           # scaled n, p floor of a Newton update (n exp(dn / n) never underflows to 0 / subnormal)
# secant predictor of a continuation step: carriers extrapolate in ln n (clipped: no overflow / underflow); Newton from
# the prediction gets this share of the iteration limit before the step restarts from the last converged state
_PREDICTOR_MAX_LOG_CHANGE = 200.0
_PREDICTED_NEWTON_SHARE = 0.5
# chord steps: once an update is <= _CHORD_START (scaled: V_T units, relative carrier change) the next Newton steps solve
# on the last factorised Jacobian (residual + one solve: no Jacobian values, no factorisation) while each update shrinks
# to <= _CHORD_CONTRACTION of the one before; an update that grows is dropped and the Jacobian factorised again
_CHORD_START = 1e-2
_CHORD_CONTRACTION = 0.25
# block layout of a Newton system's COO values: (row side, row component, column side, column component) per block, a
# side being the face's lower site, its upper site or every site; face blocks hold one value per face, site blocks one
# per site, concatenated in layout order (_PinnedPattern expands the scalar site graph from it)
_LO, _HI, _SITE = 0, 1, 2
_POISSON_LAYOUT = ((_LO, 0, _LO, 0), (_LO, 0, _HI, 0), (_HI, 0, _HI, 0), (_HI, 0, _LO, 0), (_SITE, 0, _SITE, 0))
_COUPLED_LAYOUT = (
    # Poisson faces (rows u_lo / u_hi) and site charge terms
    (_LO, 0, _LO, 0), (_LO, 0, _HI, 0), (_HI, 0, _HI, 0), (_HI, 0, _LO, 0), (_SITE, 0, _SITE, 1), (_SITE, 0, _SITE, 2),
    # electron flux term: + on row n_lo, - on row n_hi
    (_LO, 1, _LO, 1), (_LO, 1, _HI, 1), (_LO, 1, _HI, 0), (_LO, 1, _LO, 0),
    (_HI, 1, _LO, 1), (_HI, 1, _HI, 1), (_HI, 1, _HI, 0), (_HI, 1, _LO, 0),
    # hole flux term
    (_LO, 2, _LO, 2), (_LO, 2, _HI, 2), (_LO, 2, _HI, 0), (_LO, 2, _LO, 0),
    (_HI, 2, _LO, 2), (_HI, 2, _HI, 2), (_HI, 2, _HI, 0), (_HI, 2, _LO, 0),
    # recombination of both continuity rows; the potential diagonal (explicit zero) keeps every row's diagonal
    (_SITE, 1, _SITE, 1), (_SITE, 1, _SITE, 2), (_SITE, 2, _SITE, 1), (_SITE, 2, _SITE, 2), (_SITE, 0, _SITE, 0),
)


class ContinuationFailed(Exception):
    """Newton did not converge even on the smallest continuation step (message: where it stopped)."""


def bernoulli(x):
    """B(x) = x / (exp(x) - 1), B(0) = 1 (B(-x) = B(x) + x); overflow-safe for large |x|."""
    x = np.asarray(x, dtype=np.float64)
    small = np.abs(x) < _SERIES_LIMIT
    safe = np.where(small, 1.0, x)
    with np.errstate(over='ignore'):
        value = safe / np.expm1(safe)
    return np.where(small, 1.0 - 0.5 * x + x * x / 12.0, value)


def bernoulli_derivative(x, b):
    """dB/dx = B (1 - B - x) / x at x with b = B(x) (-1/2 + x / 6 near 0)."""
    x = np.asarray(x, dtype=np.float64)
    small = np.abs(x) < _SERIES_LIMIT
    safe = np.where(small, 1.0, x)
    return np.where(small, -0.5 + x / 6.0, b * (1.0 - b - safe) / safe)


def neutral_equilibrium(net_doping, n_intrinsic):
    """(u, n, p) of charge-neutral equilibrium (Fermi level 0): u = asinh(C / 2 n_i), n = n_i e^u, p = n_i e^-u, the
    minority density as n_i^2 / majority (no cancellation); zeros where n_intrinsic is 0."""
    semiconductor = n_intrinsic > 0.0
    n_i = np.where(semiconductor, n_intrinsic, 1.0)
    half = 0.5 * np.where(semiconductor, net_doping, 0.0)
    root = np.sqrt(half * half + n_i * n_i)
    majority = root + np.abs(half)
    minority = n_i * n_i / majority
    electrons = np.where(half >= 0.0, majority, minority)
    holes = np.where(half >= 0.0, minority, majority)
    u = np.arcsinh(half / n_i)
    return (np.where(semiconductor, u, 0.0), np.where(semiconductor, electrons, 0.0),
            np.where(semiconductor, holes, 0.0))


def _series_conductance(c_lo, c_hi, width_lo, width_hi, length):
    """Face conductance L / (w_lo / 2 c_lo + w_hi / 2 c_hi); 0 when either coefficient is 0."""
    denominator = width_lo * c_hi + width_hi * c_lo
    return np.where(denominator > 0.0, 2.0 * c_lo * c_hi * length / np.where(denominator > 0.0, denominator, 1.0), 0.0)


class DriftDiffusionCore:
    """Newton / continuation core of a drift-diffusion problem on a graph of sites (control volumes) and faces (lo, hi):
    held sites (ohmic contacts, gates), per-solve inputs and the last converged state; solve() continues from that
    state. Subclasses give the geometry (n_cells, face_lo / face_hi, cell_area) and set_inputs (see _finish_inputs)."""

    def __init__(self, label, tolerance=1e-8, max_newton_iterations=50, max_bias_step=0.1, linear_solver=None):
        self.label = label                                   # () -> message prefix, e.g. "drift_diffusion 'semi'"
        # solvers.sparse_direct.SparseDirectSolver of every Newton step (None: a private default one)
        self.linear_solver = SparseDirectSolver() if linear_solver is None else linear_solver
        self.tolerance = float(tolerance)
        self.max_newton_iterations = int(max_newton_iterations)
        self.max_bias_step = float(max_bias_step)
        self.contact_names, self.contact_voltages = [], []
        self.gate_names, self.gate_voltages, self.gate_offsets = [], [], []
        self.newton_iterations = 0
        self.continuation_steps = 0
        self.cold_starts = 0
        self.rejected_predictions = 0   # continuation steps whose secant prediction did not converge
        self.chord_steps = 0            # coupled Newton iterations solved on the last factorised Jacobian
        self.host_factor_bytes = None
        # KEPT_FACTOR_BACK_ENDS (cuDSS, MUMPS, PARDISO): the last factor per unknown count (Poisson n, coupled 3 n),
        # refactorised with previous= so its plan / analysis is reused while the pattern (the pinned set) holds; memory
        # outside the cupy pool / numpy (released by release_kept_factors)
        self._kept_factors = {}
        # _PinnedPattern per unknown count (Poisson n, coupled 3 n): CSR pattern + slot maps of the Newton system, kept
        # while the pinned set holds; both expand the scalar _SiteGraph (faces are fixed at construction: built once)
        self._patterns = {}
        self._graph = None
        self._inputs_ready = False
        self._converged = None   # physical state + the applied voltages / generation it was converged for
        self._state = None       # scaled (N, 3) state of the last solve

    # --- inputs -------------------------------------------------------------------

    def _finish_inputs(self):
        """Scale, equilibrium and pinned unknowns, once set_inputs has written the per-site arrays vt, semiconductor
        (bool), charge_fraction (semiconductor share of the control volume), n_intrinsic, doping (both 0 off the
        semiconductor), lifetime_n / _p, radiative, auger_n / _p, generation (per control volume), contact_cells /
        contact_index,
        gate_cells / gate_index, and the face arrays poisson_conductance, carrier_n, carrier_p."""
        self.scale = float(max(np.max(np.abs(self.doping)), np.max(self.n_intrinsic)))
        self.u_equilibrium, self.n_equilibrium, self.p_equilibrium = neutral_equilibrium(self.doping, self.n_intrinsic)
        self.pinned = np.zeros((self.n_cells, COMPONENTS), dtype=bool)
        self.pinned[self.contact_cells, :] = True
        self.pinned[self.gate_cells, 0] = True
        self.pinned[~self.semiconductor, 1:] = True
        self._inputs_ready = True

    def _qr_grid(self, components):
        """(nx, ny) index grid of the cusolver_qr back end (geometries on a grid override it)."""
        raise ValueError(f"{self.label()}: back_end 'cusolver_qr' needs a tensor mesh; use 'auto' or 'scipy_splu'")

    # --- solve ------------------------------------------------------------------

    def solve(self):
        """Converge the state for the current inputs: continuation from the last converged state (applied voltages and
        generation ramped from their previous values), else (or when that fails) from equilibrium."""
        if not self._inputs_ready:
            raise RuntimeError(f'{self.label()}: set_inputs before solve')
        end_voltages = np.array(self.contact_voltages + self.gate_voltages, dtype=np.float64)
        end_generation = self.generation
        if self._converged is not None:
            start_state = self._scaled_state(self._converged['psi'], self._converged['n'], self._converged['p'])
            try:
                state = self._continue(start_state, self._converged['voltages'], self._converged['generation'],
                                       end_voltages, end_generation)
                self._finish(state, end_voltages, end_generation)
                return
            except ContinuationFailed:
                pass   # restart from equilibrium below
        self.cold_starts += 1
        start_voltages = np.zeros_like(end_voltages)
        try:
            state = self._equilibrium_state()
        finally:
            self.release_kept_factors(self.n_cells)   # the Poisson factor serves the cold start only
        try:
            state = self._continue(state, start_voltages, np.zeros_like(end_generation), end_voltages, end_generation)
        except ContinuationFailed as exc:
            raise RuntimeError(f'{self.label()}: Newton did not converge ({exc}); try a smaller max_bias_step, a '
                               f'larger max_iterations (Newton iteration limit), a finer mesh near junctions or a smaller generation '
                               f'rate') from None
        self._finish(state, end_voltages, end_generation)

    def _finish(self, state, voltages, generation):
        """Keep the converged state (scaled, and physical for the next continuation)."""
        self._state = state
        self._converged = {'psi': state[:, 0] * self.vt, 'n': state[:, 1] * self.scale, 'p': state[:, 2] * self.scale,
                           'voltages': voltages.copy(), 'generation': generation.copy()}

    def converged_state(self):
        """Copy of the last converged state {'psi', 'n', 'p', 'generation'} (per site, physical) + 'voltages' (contacts
        then gates), or None before a solve."""
        return None if self._converged is None else {key: value.copy() for key, value in self._converged.items()}

    def seed_converged_state(self, converged) -> bool:
        """Continue the next solve from `converged` (a converged_state of another build on the same sites, contacts and
        gates); False, nothing kept, when a size differs."""
        sites = (self.n_cells,)
        held = (len(self.contact_names) + len(self.gate_names),)
        shapes = {'psi': sites, 'n': sites, 'p': sites, 'generation': sites, 'voltages': held}
        if not converged or any(np.shape(converged.get(key)) != shape for key, shape in shapes.items()):
            return False
        self._converged = {key: np.array(converged[key], dtype=np.float64) for key in shapes}
        return True

    def forget_converged_state(self) -> None:
        """The next solve starts cold from equilibrium (a doping change moves the equilibrium itself, so no
        continuation of the voltages / generation reaches it)."""
        self._converged = None

    def _scaled_state(self, psi, n, p):
        """(N, 3) scaled state of physical psi [V], n, p [1/m^3]; carriers reset to equilibrium where non-positive."""
        state = np.empty((self.n_cells, COMPONENTS))
        state[:, 0] = psi / self.vt
        semiconductor = self.semiconductor
        state[:, 1] = np.where(semiconductor & (n > 0.0), n, self.n_equilibrium) / self.scale
        state[:, 2] = np.where(semiconductor & (p > 0.0), p, self.p_equilibrium) / self.scale
        state[~semiconductor, 1:] = 0.0
        return state

    def _targets(self, voltages):
        """(N, 3) values of the pinned unknowns for the applied voltages (contacts first, then gates) [V]."""
        targets = np.zeros((self.n_cells, COMPONENTS))
        n_contacts = len(self.contact_names)
        contacts = self.contact_cells
        contact_voltage = voltages[:n_contacts][self.contact_index[contacts]] if n_contacts else 0.0
        targets[contacts, 0] = self.u_equilibrium[contacts] + contact_voltage / self.vt
        targets[contacts, 1] = self.n_equilibrium[contacts] / self.scale
        targets[contacts, 2] = self.p_equilibrium[contacts] / self.scale
        gates = self.gate_cells
        if self.gate_names:
            gate_index = self.gate_index[gates]
            offsets = np.asarray(self.gate_offsets)
            targets[gates, 0] = (voltages[n_contacts:][gate_index] - offsets[gate_index]) / self.vt
        return targets

    def _continue(self, state, start_voltages, start_generation, end_voltages, end_generation):
        """Newton along voltages / generation = start + s (end - start), s from 0 (state converged there) to 1; the
        step is at most max_bias_step in every applied voltage, halved on failure (ContinuationFailed below 1/64).
        From the second converged step Newton starts from the secant prediction of the last two (the bias change
        would otherwise reach the interior only through log-damped updates: ~10 iterations per V); a prediction that
        does not converge falls back to the last converged state and ends the predictions of this ramp."""
        voltage_change = float(np.max(np.abs(end_voltages - start_voltages))) if end_voltages.size else 0.0
        first_step = 1.0 if voltage_change <= self.max_bias_step else self.max_bias_step / voltage_change
        position, step = 0.0, first_step
        earlier = None   # (position, state) converged before `position`, for the secant prediction
        predict = True
        while position < 1.0:
            trial = min(1.0, position + step)
            voltages = start_voltages + trial * (end_voltages - start_voltages)
            generation = (start_generation + trial * (end_generation - start_generation)) / self.scale
            targets = self._targets(voltages)
            converged = False
            if predict and earlier is not None:
                converged, candidate = self._newton(
                    self._predicted_state(earlier, (position, state), trial), targets, generation,
                    max(1, int(_PREDICTED_NEWTON_SHARE * self.max_newton_iterations)))
                if not converged:
                    self.rejected_predictions += 1
                    predict = False
            if not converged:
                converged, candidate = self._newton(state, targets, generation)
            if converged:
                earlier = (position, state)
                position, state = trial, candidate
                self.continuation_steps += 1
                step = min(first_step, 2.0 * step)
                continue
            step *= 0.5
            if step < _MIN_STEP_FRACTION * first_step:
                raise ContinuationFailed(f'stuck at {100.0 * position:.3g} % of the ramp of the applied voltages and '
                                         f'generation (step {100.0 * step:.2g} %)')
        return state

    @staticmethod
    def _predicted_state(earlier, latest, position):
        """Scaled state at ramp `position` extrapolated from two converged (position, state) points: u linearly,
        carriers linearly in ln n / ln p where both points hold them (else the latest value), floored."""
        (earlier_position, earlier_state), (latest_position, latest_state) = earlier, latest
        ratio = (position - latest_position) / (latest_position - earlier_position)
        predicted = latest_state.copy()
        predicted[:, 0] += ratio * (latest_state[:, 0] - earlier_state[:, 0])
        carriers, earlier_carriers = latest_state[:, 1:], earlier_state[:, 1:]
        both = (carriers > 0.0) & (earlier_carriers > 0.0)
        log_change = ratio * (np.log(np.where(both, carriers, 1.0)) - np.log(np.where(both, earlier_carriers, 1.0)))
        growth = np.exp(np.clip(log_change, -_PREDICTOR_MAX_LOG_CHANGE, _PREDICTOR_MAX_LOG_CHANGE))
        predicted[:, 1:] = np.where(both, np.maximum(carriers * growth, _CARRIER_FLOOR), carriers)
        return predicted

    def _equilibrium_state(self):
        """Scaled equilibrium state at zero applied voltage (gates at -work_function_difference): nonlinear Poisson
        with Boltzmann carriers (Fermi level 0), Newton (chord steps near convergence) from the charge-neutral guess."""
        voltages = np.zeros(len(self.contact_names) + len(self.gate_names))
        targets = self._targets(voltages)[:, 0]
        pinned = self.pinned[:, 0]
        u = np.where(pinned, targets, self.u_equilibrium)
        n_intrinsic = self.n_intrinsic / self.scale   # 0 off the semiconductor
        chord, last_error = None, math.inf   # (factor, row norms) of the last factorised Jacobian; its update size
        for _ in range(self.max_newton_iterations):
            delta, chord_step = None, chord is not None and last_error <= _CHORD_START
            if chord_step:
                residual, _ = self._poisson_system(u, n_intrinsic, jacobian=False)
                residual[pinned] = u[pinned] - targets[pinned]
                delta = self._chord_solve(chord, -residual)
                chord_step = delta is not None
            if delta is None:
                chord = None   # dropped before the next factorisation: one SuperLU factor alive at a time
                residual, values = self._poisson_system(u, n_intrinsic)
                residual[pinned] = u[pinned] - targets[pinned]
                delta, chord = self._linear_solve(values, _POISSON_LAYOUT, pinned, -residual, 1)
            finite = bool(np.all(np.isfinite(delta)))
            error = float(np.max(np.abs(np.where(pinned, 0.0, delta)))) if finite else math.inf
            if chord_step:
                if error > last_error:   # the frozen Jacobian no longer contracts: drop the update, factorise again
                    chord = None
                    continue
                if error > _CHORD_CONTRACTION * last_error:
                    chord = None
            elif not finite:
                break
            u = u + np.where(pinned, delta, _log_damped(delta))
            last_error = error
            if error <= self.tolerance:
                boltzmann = np.where(self.semiconductor, u, 0.0)
                state = np.zeros((self.n_cells, COMPONENTS))
                state[:, 0] = u
                state[:, 1] = n_intrinsic * np.exp(boltzmann)
                state[:, 2] = n_intrinsic * np.exp(-boltzmann)
                return state
        raise RuntimeError(f'{self.label()}: the equilibrium (nonlinear Poisson) solve did not converge in '
                           f'{self.max_newton_iterations} Newton iterations')

    def _poisson_system(self, u, n_intrinsic, jacobian=True):
        """(residual, COO values in _POISSON_LAYOUT order) of the scaled equilibrium Poisson equation in u alone (n, p
        Boltzmann); values None without jacobian (a chord step)."""
        lo, hi = self.face_lo, self.face_hi
        g = self.poisson_conductance * self.vt / (ELEMENTARY_CHARGE * self.scale)
        flux = g * (u[lo] - u[hi])
        area = self.cell_area
        fraction = self.charge_fraction
        exp_u = np.exp(np.where(self.semiconductor, u, 0.0))
        electrons, holes = n_intrinsic * exp_u, n_intrinsic / exp_u
        residual = (_face_sum(lo, hi, flux, self.n_cells) / area
                    - fraction * (holes - electrons + self.doping / self.scale))
        if not jacobian:
            return residual, None
        g_lo, g_hi = g / area[lo], g / area[hi]
        return residual, self._layout_values(_POISSON_LAYOUT, (g_lo, -g_lo, g_hi, -g_hi, fraction * (electrons + holes)))

    def _layout_values(self, layout, blocks):
        """Concatenated COO values of `blocks` (an array or scalar per layout block: per face, or per site), written by
        slice into one buffer (no broadcast views: ~9 us each of 27 per coupled call)."""
        sizes = (self.face_lo.size, self.face_lo.size, self.n_cells)   # indexed by the row side _LO / _HI / _SITE
        values = np.empty(sum(sizes[block[0]] for block in layout))
        start = 0
        for block, block_values in zip(layout, blocks):
            stop = start + sizes[block[0]]
            values[start:stop] = block_values
            start = stop
        return values

    def _newton(self, state, targets, generation, max_iterations=None):
        """(converged, state) of Newton on the coupled system from `state`: log-damped potential updates, carrier
        updates n exp(dn / n) when negative (floored); converged when max |du| and max |dn| / (n + n_i) (p likewise)
        over the free unknowns are <= tolerance; a non-finite system or update is not converged. Chord steps (the last
        factor, no Jacobian) once an update is <= _CHORD_START. max_iterations: None = max_newton_iterations."""
        pinned = self.pinned
        n_intrinsic = (self.n_intrinsic / self.scale)[:, None]
        state = state.copy()
        state[pinned] = targets[pinned]
        chord, last_error = None, math.inf   # (factor, row norms) of the last factorised Jacobian; its update size
        for _ in range(self.max_newton_iterations if max_iterations is None else max_iterations):
            delta, chord_step = None, chord is not None and last_error <= _CHORD_START
            if chord_step:
                residual, _ = self._coupled_system(state, generation, jacobian=False)
                if not np.all(np.isfinite(residual)):
                    return False, state
                residual[pinned] = 0.0
                delta = self._chord_solve(chord, -residual.ravel())
                chord_step = delta is not None
            if delta is None:
                chord = None   # dropped before the next factorisation: one SuperLU factor alive at a time
                residual, values = self._coupled_system(state, generation)
                if not (np.all(np.isfinite(residual)) and np.all(np.isfinite(values))):
                    return False, state   # never factorise NaN / inf (SuperLU reports it as singular)
                residual[pinned] = 0.0
                delta, chord = self._linear_solve(values, _COUPLED_LAYOUT, pinned.ravel(), -residual.ravel(),
                                                  COMPONENTS)
            self.newton_iterations += 1
            self.chord_steps += int(chord_step)
            delta = delta.reshape(self.n_cells, COMPONENTS)
            if not np.all(np.isfinite(delta)):
                if chord_step:
                    chord = None
                    continue
                return False, state
            delta[pinned] = 0.0
            carriers = state[:, 1:]
            relative = delta[:, 1:] / np.where(carriers > 0.0, carriers, 1.0)
            error = max(float(np.max(np.abs(delta[:, 0]))),
                        float(np.max(np.abs(delta[:, 1:]) / (carriers + n_intrinsic + (carriers == 0.0)))))
            if chord_step:
                if error > last_error:   # the frozen Jacobian no longer contracts: drop the update, factorise again
                    chord = None
                    continue
                if error > _CHORD_CONTRACTION * last_error:
                    chord = None
            # np.where evaluates both branches: exp only sees the (selected) non-positive arguments
            carriers_updated = np.maximum(np.where(relative >= 0.0, carriers * (1.0 + relative),
                                                   carriers * np.exp(np.minimum(relative, 0.0))), _CARRIER_FLOOR)
            if not np.all(np.isfinite(carriers_updated)):
                if chord_step:
                    chord = None
                    continue
                return False, state
            state[:, 0] += _log_damped(delta[:, 0])
            state[:, 1:] = carriers_updated
            state[pinned] = targets[pinned]
            last_error = error
            if error <= self.tolerance:
                return True, state
        return False, state

    def _coupled_system(self, state, generation, jacobian=True):
        """(residual (N, 3), COO values in _COUPLED_LAYOUT order) of the scaled coupled system: Poisson rows (charge per
        control volume), electron / hole continuity rows (SG face terms + volume (R - G)); pinned rows are replaced by
        the caller. Values None without jacobian (a chord step)."""
        u, n, p = state[:, 0], state[:, 1], state[:, 2]
        lo, hi = self.face_lo, self.face_hi
        area = self.cell_area
        fraction = self.charge_fraction
        g = self.poisson_conductance * self.vt / (ELEMENTARY_CHARGE * self.scale)
        flux_u = g * (u[lo] - u[hi])
        delta = u[hi] - u[lo]
        b_plus = bernoulli(delta)
        b_minus = b_plus + delta
        n_lo, n_hi, p_lo, p_hi = n[lo], n[hi], p[lo], p[hi]
        flux_n = self.carrier_n * (n_lo * b_minus - n_hi * b_plus)
        flux_p = self.carrier_p * (p_lo * b_plus - p_hi * b_minus)
        recombination, dr_dn, dr_dp = self._recombination(n, p)
        source = area * (recombination - generation)

        residual = np.empty((self.n_cells, COMPONENTS))
        residual[:, 0] = _face_sum(lo, hi, flux_u, self.n_cells) / area - fraction * (p - n + self.doping / self.scale)
        residual[:, 1] = _face_sum(lo, hi, flux_n, self.n_cells) + source
        residual[:, 2] = _face_sum(lo, hi, flux_p, self.n_cells) + source
        if not jacobian:
            return residual, None

        db_plus = bernoulli_derivative(delta, b_plus)
        db_minus = bernoulli_derivative(-delta, b_minus)
        dn_dnlo, dn_dnhi = self.carrier_n * b_minus, -self.carrier_n * b_plus
        dn_duhi = -self.carrier_n * (n_lo * db_minus + n_hi * db_plus)
        dp_dplo, dp_dphi = self.carrier_p * b_plus, -self.carrier_p * b_minus
        dp_duhi = self.carrier_p * (p_lo * db_plus + p_hi * db_minus)
        g_lo, g_hi = g / area[lo], g / area[hi]
        volume_dr_dn, volume_dr_dp = area * dr_dn, area * dr_dp
        blocks = (   # _COUPLED_LAYOUT order
            g_lo, -g_lo, g_hi, -g_hi, fraction, -fraction,
            dn_dnlo, dn_dnhi, dn_duhi, -dn_duhi, -dn_dnlo, -dn_dnhi, -dn_duhi, dn_duhi,
            dp_dplo, dp_dphi, dp_duhi, -dp_duhi, -dp_dplo, -dp_dphi, -dp_duhi, dp_duhi,
            volume_dr_dn, volume_dr_dp, volume_dr_dn, volume_dr_dp, 0.0,
        )
        return residual, self._layout_values(_COUPLED_LAYOUT, blocks)

    def _recombination(self, n, p, weight=None):
        """(R / C_s, dR/dn, dR/dp) of SRH (mid-gap trap) + radiative + Auger recombination, R = (n p - n_i^2) (1 /
        (tau_p (n + n_i) + tau_n (p + n_i)) + B + C_n n + C_p p); scaled n, p in, times weight (default
        charge_fraction: per control volume). Physical float64 products (C ~1e-43 m^6/s stays normal)."""
        scale = self.scale
        electrons, holes, n_i = n * scale, p * scale, self.n_intrinsic
        excess = electrons * holes - n_i * n_i
        denominator = self.lifetime_p * (electrons + n_i) + self.lifetime_n * (holes + n_i)
        denominator = np.where(self.semiconductor, denominator, 1.0)
        auger = self.auger_n * electrons + self.auger_p * holes
        rate = 1.0 / denominator + self.radiative + auger
        weight = self.charge_fraction if weight is None else weight
        recombination = weight * excess * rate
        dr_dn = weight * (holes * rate - excess * self.lifetime_p / (denominator * denominator) + excess * self.auger_n)
        dr_dp = weight * (electrons * rate - excess * self.lifetime_n / (denominator * denominator)
                          + excess * self.auger_p)
        return recombination / scale, dr_dn, dr_dp

    def _site_graph(self):
        """_SiteGraph of the sites and faces (built on first use; faces are fixed at construction)."""
        if self._graph is None:
            self._graph = _SiteGraph(self.face_lo, self.face_hi, self.n_cells)
        return self._graph

    def _linear_solve(self, values, layout, pinned_rows, rhs, components):
        """(solution, chord) of the sparse system of COO values in `layout` order (pinned rows as identity rows) on the
        kept _PinnedPattern, rows L1-equilibrated, through the linear solver's back end (counted there): real cuDSS /
        MUMPS / PARDISO LU on KEPT_FACTOR_BACK_ENDS, else a real SuperLU (or cusolver_qr) solve; a cuDSS refusal of an
        'auto' runtime switches it to SuperLU (B1b) and the system is solved again there (a MUMPS / PARDISO refusal
        raises: no fallback). chord = (factor, row norms) for _chord_solve, None on cusolver_qr (QR keeps no factor)."""
        size = rhs.size
        pattern = self._patterns.get(size)
        if pattern is None or not np.array_equal(pattern.pinned_rows, pinned_rows):
            pattern = self._patterns[size] = _PinnedPattern(self._site_graph(), layout, components, pinned_rows)
        data = pattern.csr_values(values)
        row_norm = np.add.reduceat(np.abs(data), pattern.indptr[:-1])   # every row holds an entry
        row_norm = np.where(row_norm > 0.0, row_norm, 1.0)
        data /= row_norm[pattern.slot_rows]   # a take: np.repeat by per-row counts was ~2x slower
        rhs = rhs / row_norm
        try:
            try:
                solution, factor = self._factor_solve(pattern, data, rhs, components)
            except CUDSS_REFUSALS as exc:   # a deferred cuDSS check at solve_host (cudss_mixed on a reused plan)
                if not self.linear_solver.fall_back(exc, size):   # B1b: an 'auto' runtime -> SuperLU for good
                    raise
                self.release_kept_factors()
                solution, factor = self._factor_solve(pattern, data, rhs, components)
        except (CudssAccuracyError, HostDirectAccuracyError) as exc:   # probe missed after refinement: not a zero pivot
            raise ValueError(f'{self.label()}: {exc}') from exc
        except RuntimeError as exc:   # SuperLU 'Factor is exactly singular', Cudss- / HostDirectSingularError
            raise ValueError(f'{self.label()}: singular linear system ({exc}); every insulating region (n_intrinsic '
                             '0) must touch a semiconductor or a gate, and every semiconductor region an ohmic '
                             'contact') from exc
        back_end = self.linear_solver.back_end
        self.host_factor_bytes = None if back_end in CUDSS_BACK_ENDS else factor.host_bytes
        return solution, (None if back_end == 'cusolver_qr' else (factor, row_norm))

    @staticmethod
    def _chord_solve(chord, rhs):
        """Solution of the last factorised (equilibrated) Jacobian for a new right-hand side (a chord step: one solve,
        no assembly or factorisation); None when its factor is gone or refuses the solve (the caller factorises)."""
        factor, row_norm = chord
        if not getattr(factor, 'valid', True):
            return None
        try:
            return factor.solve_host(rhs / row_norm)
        except CUDSS_REFUSALS + HOST_DIRECT_REFUSALS:
            return None

    def _factor_solve(self, pattern, data, rhs, components):
        """(solution, factor) of the equilibrated system (CSR data on `pattern`) on the runtime's current back end: a
        real cuDSS / MUMPS / PARDISO LU (the kept factor of this size refactorised in place), a real host SuperLU or
        cusolver_qr on the grid."""
        size = rhs.size
        back_end = self.linear_solver.back_end
        if back_end in KEPT_FACTOR_BACK_ENDS:
            previous = self._kept_factors.pop(size, None)
            # values on the canonical pattern (identity-matched): no host CSR canonicalising / asymmetry check per step;
            # defer_check: a refactorisation on the kept plan / analysis skips its probe solve, the Newton solve below is
            # the check (an accuracy refusal then comes from solve_host: _linear_solve handles it as at the factorisation)
            factor = self.linear_solver.factorize_values(data, pattern.indptr, pattern.indices, real=True,
                                                         previous=previous, defer_check=True)
            if self.linear_solver.back_end in KEPT_FACTOR_BACK_ENDS:
                self._kept_factors[size] = factor
            else:   # B1b: factorize_values of an 'auto' runtime fell back to SuperLU: free every cuDSS factor
                if previous is not None:
                    previous.release()
                self.release_kept_factors()
            return factor.solve_host(rhs), factor
        if back_end == 'scipy_splu':
            factor = self.linear_solver.factorize_host(pattern.host_csr(data), real=True)
            return factor.solve_host(rhs), factor
        import cupy as cp
        import cupyx.scipy.sparse
        grid_x, grid_y = self._qr_grid(components)
        matrix = cupyx.scipy.sparse.csr_matrix(
            (cp.asarray(data), cp.asarray(pattern.indices), cp.asarray(pattern.indptr)), shape=(size, size))
        factor = self.linear_solver.factorize(matrix, grid_x, grid_y, False, False)
        return cp.asnumpy(factor.solve(cp.asarray(rhs))), factor

    def release_kept_factors(self, unknowns=None) -> None:
        """Free the kept cuDSS / MUMPS / PARDISO factor of the system of `unknowns` unknowns (n_cells: the Poisson one;
        None: all)."""
        sizes = list(self._kept_factors) if unknowns is None else [unknowns]
        for size in sizes:
            factor = self._kept_factors.pop(size, None)
            if factor is not None:
                factor.release()

    # --- outputs --------------------------------------------------------------------

    def _face_fluxes(self):
        """Physical SG face terms (T_n, T_p) [1/(m s) per unit z length; 1/s per ring in (r, z)] of the last state: the
        electron current out of the lower site is -q T_n, the hole current q T_p."""
        state = self._require_state()
        u, n, p = state[:, 0], state[:, 1] * self.scale, state[:, 2] * self.scale
        delta = u[self.face_hi] - u[self.face_lo]
        b_plus = bernoulli(delta)
        b_minus = b_plus + delta
        lo, hi = self.face_lo, self.face_hi
        return (self.carrier_n * (n[lo] * b_minus - n[hi] * b_plus),
                self.carrier_p * (p[lo] * b_plus - p[hi] * b_minus))

    def _quasi_fermi(self, psi, electrons, holes):
        """(phi_n, phi_p) [V] per site: psi -/+ V_T ln(n / n_i, p / n_i), 0 off the semiconductor."""
        semiconductor = self.semiconductor
        n_i = np.where(semiconductor, self.n_intrinsic, 1.0)
        phi_n = np.where(semiconductor, psi - self.vt * np.log(np.where(semiconductor, electrons, 1.0) / n_i), 0.0)
        phi_p = np.where(semiconductor, psi + self.vt * np.log(np.where(semiconductor, holes, 1.0) / n_i), 0.0)
        return phi_n, phi_p

    def contact_currents(self, index):
        """(electron, hole) current [A/m per unit z length; A of the whole ring in (r, z)] flowing out of ohmic contact
        `index` into the domain: the continuity reaction of its held sites (faces to other sites + their volume (R - G);
        the latter is 0 for tensor cells, which hold equilibrium and get no generation)."""
        state = self._require_state()
        flux_n, flux_p = self._face_fluxes()
        member = self.contact_cells & (self.contact_index == index)
        out_of_lo = member[self.face_lo] & ~member[self.face_hi]
        out_of_hi = member[self.face_hi] & ~member[self.face_lo]
        recombination, _, _ = self._recombination(state[:, 1], state[:, 2])   # per control volume, / C_s
        volume_source = float(np.sum(self.cell_area[member] * (recombination[member] * self.scale
                                                                - self.generation[member])))
        electron = -ELEMENTARY_CHARGE * (float(np.sum(flux_n[out_of_lo])) - float(np.sum(flux_n[out_of_hi]))
                                         + volume_source)
        hole = ELEMENTARY_CHARGE * (float(np.sum(flux_p[out_of_lo])) - float(np.sum(flux_p[out_of_hi]))
                                    + volume_source)
        return electron, hole

    def gate_charge(self, index):
        """Free charge [C/m per unit z length; C of the whole ring in (r, z)] on gate `index`: the D flux out of its
        sites (Gauss)."""
        state = self._require_state()
        psi = state[:, 0] * self.vt
        flux = self.poisson_conductance * (psi[self.face_lo] - psi[self.face_hi])
        member = self.gate_cells & (self.gate_index == index)
        out_of_lo = member[self.face_lo] & ~member[self.face_hi]
        out_of_hi = member[self.face_hi] & ~member[self.face_lo]
        return float(np.sum(flux[out_of_lo])) - float(np.sum(flux[out_of_hi]))

    def _require_state(self):
        if self._state is None:
            raise RuntimeError(f'{self.label()}: no solve yet')
        return self._state


class DriftDiffusionSystem(DriftDiffusionCore):
    """Drift-diffusion on the window cells of a tensor mesh (flat index i * wy + j; interior x faces then y faces): face
    geometry, held cells (ohmic contacts, gates), doping and set_inputs; the Newton core in DriftDiffusionCore."""

    def __init__(self, dx_window, dy_window, label, tolerance=1e-8, max_newton_iterations=50, max_bias_step=0.1,
                 axisymmetric=False, linear_solver=None):
        super().__init__(label, tolerance, max_newton_iterations, max_bias_step, linear_solver)
        self.dx = np.asarray(dx_window, dtype=np.float64)
        self.dy = np.asarray(dy_window, dtype=np.float64)
        self.shape = (self.dx.size, self.dy.size)
        self.n_cells = self.dx.size * self.dy.size
        # (r, z): x = r from the window's x_lo edge (the axis); face areas, widths and exact ring volumes from the
        # core.metric WindowMetric (as the DiffusionSystem)
        self.axisymmetric = bool(axisymmetric)
        self.metric = window_metric(self.dx, self.dy, CARTESIAN, self.axisymmetric)
        # cell area [m^2] per unit z length, or ring volume [m^3] when axisymmetric
        self.cell_area = self.metric.volume.full().ravel()
        self._build_faces()
        self.net_doping = np.zeros(self.n_cells)                         # N_D - N_A [1/m^3] of the doping features
        self.auger = np.zeros((2, self.n_cells))   # Auger C_n, C_p [m^6/s] of the auger_recombination features
        self.contact_index = np.full(self.n_cells, -1, dtype=np.int64)   # ohmic contact holding a cell, -1 none
        self.gate_index = np.full(self.n_cells, -1, dtype=np.int64)

    # --- setup ------------------------------------------------------------------

    def _qr_grid(self, components):
        """The interleaved unknowns as a (wx, components * wy) grid (a valid index order of cusolver_qr)."""
        wx, wy = self.shape
        return wx, components * wy

    def _build_faces(self):
        """Interior x faces then y faces: lower / upper cell, face length (ring face area when axisymmetric), the two
        cell widths along the normal."""
        wx, wy = self.shape
        index = np.arange(self.n_cells).reshape(wx, wy)
        self.n_x_faces = (wx - 1) * wy
        self.face_lo = np.concatenate([index[:-1, :].ravel(), index[:, :-1].ravel()])
        self.face_hi = np.concatenate([index[1:, :].ravel(), index[:, 1:].ravel()])
        metric = self.metric
        x_length, y_length = metric.x_length.full(), metric.y_length.full()   # physical cell widths
        self.face_length = np.concatenate([metric.x_face_area.full()[1:-1, :].ravel(),
                                           metric.y_face_area.full()[:, 1:-1].ravel()])
        self.face_width_lo = np.concatenate([x_length[:-1, :].ravel(), y_length[:, :-1].ravel()])
        self.face_width_hi = np.concatenate([x_length[1:, :].ravel(), y_length[:, 1:].ravel()])
        self.face_distance = 0.5 * (self.face_width_lo + self.face_width_hi)
        self.face_is_x = np.arange(self.face_lo.size) < self.n_x_faces

    def add_doping(self, net_doping):
        """Add N_D - N_A [1/m^3] (flat window array, 0 outside the feature)."""
        self.net_doping += np.asarray(net_doping, dtype=np.float64)

    def clear_doping(self):
        """Net doping 0 everywhere (DriftDiffusionPhysics.set_doping adds every feature again)."""
        self.net_doping[...] = 0.0

    def set_auger(self, mask, electron_coefficient, hole_coefficient):
        """Auger coefficients C_n, C_p [m^6/s] on the flat window mask (a later feature takes shared cells)."""
        self.auger[:, np.asarray(mask, dtype=bool)] = np.array([[electron_coefficient], [hole_coefficient]],
                                                               dtype=np.float64)

    def add_contact(self, name, mask, voltage) -> int:
        """Hold the semiconductor cells of the flat window mask as an ohmic contact at `voltage` [V]; returns its index
        (a later contact or gate takes shared cells)."""
        index = len(self.contact_names)
        self.contact_names.append(name)
        self.contact_voltages.append(float(voltage))
        self.contact_index[mask] = index
        self.gate_index[mask] = -1
        return index

    def add_gate(self, name, mask, voltage, work_function_difference) -> int:
        """Hold psi = voltage - work_function_difference [V] on the (insulating) cells of the flat window mask."""
        index = len(self.gate_names)
        self.gate_names.append(name)
        self.gate_voltages.append(float(voltage))
        self.gate_offsets.append(float(work_function_difference))
        self.gate_index[mask] = index
        self.contact_index[mask] = -1
        return index

    def set_inputs(self, materials, generation, thermal_voltage):
        """Per-solve inputs: materials {schema array: flat window float64}, generation [1/(m^3 s)] (flat) and V_T [V];
        ValueError for invalid material values or held regions."""
        self.vt = float(thermal_voltage)
        n_intrinsic = materials['n_intrinsic']
        semiconductor = n_intrinsic > 0.0
        if not semiconductor.any():
            raise ValueError(f'{self.label()}: no window cell has n_intrinsic > 0 (assign a semiconductor material)')
        for name in ('mobility_n', 'mobility_p', 'radiative_recombination'):
            if np.any(materials[name][semiconductor] < 0.0):
                raise ValueError(f'{self.label()}: {name} must be >= 0 in every semiconductor cell')
        for name in ('lifetime_n', 'lifetime_p'):
            if not np.all(materials[name][semiconductor] > 0.0):
                raise ValueError(f'{self.label()}: {name} must be > 0 in every semiconductor cell')
        self.semiconductor = semiconductor
        self.charge_fraction = semiconductor.astype(np.float64)   # a cell is wholly semiconductor or insulator
        self.n_intrinsic = np.where(semiconductor, n_intrinsic, 0.0)
        self.doping = np.where(semiconductor, self.net_doping, 0.0)
        self.contact_cells = (self.contact_index >= 0) & semiconductor
        self.gate_cells = self.gate_index >= 0
        for index, name in enumerate(self.contact_names):
            if not np.any(self.contact_cells & (self.contact_index == index)):
                raise ValueError(f"{self.label()}: ohmic contact '{name}' holds no semiconductor cell (n_intrinsic > 0); "
                                 'select cells of a semiconductor material')
        for index, name in enumerate(self.gate_names):
            if np.any(semiconductor & (self.gate_index == index)):
                raise ValueError(f"{self.label()}: gate '{name}' selects semiconductor cells (n_intrinsic > 0); a gate "
                                 'is an electrode on an insulator (use an ohmic contact on a semiconductor)')

        lo, hi = self.face_lo, self.face_hi
        length, width_lo, width_hi = self.face_length, self.face_width_lo, self.face_width_hi
        eps_lo = EPS0 * np.where(self.face_is_x, materials['eps_dc_xx'][lo], materials['eps_dc_yy'][lo])
        eps_hi = EPS0 * np.where(self.face_is_x, materials['eps_dc_xx'][hi], materials['eps_dc_yy'][hi])
        gate_lo, gate_hi = self.gate_cells[lo], self.gate_cells[hi]
        # gates are ideal conductors: their potential sits on their faces (free half cell only), as potential terminals
        self.poisson_conductance = np.where(
            gate_lo & ~gate_hi, 2.0 * eps_hi * length / width_hi,
            np.where(gate_hi & ~gate_lo, 2.0 * eps_lo * length / width_lo,
                     _series_conductance(eps_lo, eps_hi, width_lo, width_hi, length)))   # F/m (F per ring in (r, z))
        mobility_n = np.where(semiconductor, materials['mobility_n'], 0.0)
        mobility_p = np.where(semiconductor, materials['mobility_p'], 0.0)
        # SG face coefficients d = D_face L / h [m^2/s] (D = mu V_T, half-cell harmonic mean of the two mobilities)
        self.carrier_n = self.vt * _series_conductance(mobility_n[lo], mobility_n[hi], width_lo, width_hi, length)
        self.carrier_p = self.vt * _series_conductance(mobility_p[lo], mobility_p[hi], width_lo, width_hi, length)
        self.lifetime_n = np.where(semiconductor, materials['lifetime_n'], 1.0)
        self.lifetime_p = np.where(semiconductor, materials['lifetime_p'], 1.0)
        self.radiative = np.where(semiconductor, materials['radiative_recombination'], 0.0)
        self.auger_n, self.auger_p = np.where(semiconductor, self.auger, 0.0)
        self.generation = np.where(semiconductor & ~self.contact_cells, np.asarray(generation, dtype=np.float64), 0.0)
        self._finish_inputs()

    # --- outputs --------------------------------------------------------------------

    def outputs(self) -> dict:
        """{field: (wx, wy) float64} of the last solve: V, N, P, E, J (total, electron, hole), quasi-Fermi potentials,
        net doping, space charge and recombination (0 outside the free semiconductor cells)."""
        state = self._require_state()
        shape = self.shape
        psi = state[:, 0] * self.vt
        electrons, holes = state[:, 1] * self.scale, state[:, 2] * self.scale
        flux_n, flux_p = self._face_fluxes()
        lo, hi = self.face_lo, self.face_hi
        current_n = -ELEMENTARY_CHARGE * flux_n / self.face_length    # A/m^2 along +x / +y (lower -> upper cell)
        current_p = ELEMENTARY_CHARGE * flux_p / self.face_length
        field = -(psi[hi] - psi[lo]) / self.face_distance
        semiconductor = self.semiconductor
        recombination, _, _ = self._recombination(state[:, 1], state[:, 2])
        free = semiconductor & ~self.contact_cells
        phi_n, phi_p = self._quasi_fermi(psi, electrons, holes)
        outputs = {'V': psi, 'N': electrons, 'P': holes, 'phi_n': phi_n, 'phi_p': phi_p,
                   'net_doping': self.doping,
                   'space_charge': ELEMENTARY_CHARGE * np.where(semiconductor, holes - electrons + self.doping, 0.0),
                   'recombination': np.where(free, recombination * self.scale, 0.0)}
        outputs = {name: values.reshape(shape) for name, values in outputs.items()}
        outputs['E_x'], outputs['E_y'] = self._face_average(field)
        outputs['Jn_x'], outputs['Jn_y'] = self._face_average(current_n)
        outputs['Jp_x'], outputs['Jp_y'] = self._face_average(current_p)
        outputs['J_x'] = outputs['Jn_x'] + outputs['Jp_x']
        outputs['J_y'] = outputs['Jn_y'] + outputs['Jp_y']
        return outputs

    def _face_average(self, face_values):
        """Cell-centre (x, y) components: the mean of each cell's two faces along the axis (outer faces carry 0)."""
        wx, wy = self.shape
        x_faces = face_values[:self.n_x_faces].reshape(wx - 1, wy)
        y_faces = face_values[self.n_x_faces:].reshape(wx, wy - 1)
        centre_x = np.zeros((wx, wy))
        centre_x[:-1, :] += x_faces
        centre_x[1:, :] += x_faces
        centre_y = np.zeros((wx, wy))
        centre_y[:, :-1] += y_faces
        centre_y[:, 1:] += y_faces
        return 0.5 * centre_x, 0.5 * centre_y


def _face_sum(lo, hi, face_values, n_cells):
    """Per-cell sum of a face term entering the lower cell with + and the upper cell with -."""
    return np.bincount(lo, face_values, n_cells) - np.bincount(hi, face_values, n_cells)


def _log_damped(delta):
    """sign(d) log(1 + |d|): Newton potential update limited to logarithmic growth (in V_T units)."""
    return np.sign(delta) * np.log1p(np.abs(delta))


class _SiteGraph:
    """Scalar CSR pattern of the site graph (every site and its face neighbours, columns ascending): the position of
    each face's two entries and of each site's diagonal within their rows, which _PinnedPattern expands per unknown
    component. One face per pair of different sites."""

    def __init__(self, face_lo, face_hi, n_sites):
        lo, hi = np.asarray(face_lo, dtype=np.int64), np.asarray(face_hi, dtype=np.int64)
        n_faces = lo.size
        sites = np.arange(n_sites, dtype=np.int64)
        rows = np.concatenate([lo, hi, sites])
        keys = rows * n_sites + np.concatenate([hi, lo, sites])
        order = np.argsort(keys)
        if np.any(lo == hi) or np.any(np.diff(keys[order]) == 0):
            raise ValueError('drift_diffusion: every face must join two different sites, at most one face per pair')
        rank = np.empty(keys.size, dtype=np.int64)
        rank[order] = np.arange(keys.size)
        self.row_counts = np.bincount(rows, minlength=n_sites)   # face neighbours + 1 per site
        position = rank - (np.cumsum(self.row_counts) - self.row_counts)[rows]   # column rank within the row
        self.lo_hi, self.hi_lo = position[:n_faces], position[n_faces:2 * n_faces]
        self.diagonal = position[2 * n_faces:]
        self.face_lo, self.face_hi, self.n_sites = lo, hi, int(n_sites)


class _PinnedPattern:
    """Canonical CSR pattern (sorted, no duplicates, explicit zeros kept) of a Newton system with the pinned rows as
    identity rows, built once per pinned set by expanding the _SiteGraph per unknown component (no sort of the COO
    entries): the CSR slot of every COO value of the block layout (values of pinned rows go to the dummy slot nnz), so
    a Newton step only sums its values by one bincount. A free row (site s, component a) holds, per scalar column c of
    s in ascending order, the components the layout couples a to: of a face neighbour when c != s, of s itself when
    c == s."""

    def __init__(self, graph, layout, components, pinned_rows):
        self.pinned_rows = np.array(pinned_rows, dtype=bool)
        n_sites = graph.n_sites
        self.size = n_sites * components
        if any(_SITE in (side, column_side) and side != column_side for side, _, column_side, _ in layout):
            raise ValueError('_PinnedPattern: a site block couples a site with itself only')
        free = ~self.pinned_rows.reshape(n_sites, components)
        # column components of a face neighbour / of the own site per row component (ascending = CSR column order)
        neighbour = [sorted({b for side, a, column_side, b in layout if a == row and side != column_side})
                     for row in range(components)]
        own = [sorted({b for side, a, column_side, b in layout if a == row and side == column_side})
               for row in range(components)]
        neighbour_width = np.array([len(columns) for columns in neighbour], dtype=np.int64)
        own_width = np.array([len(columns) for columns in own], dtype=np.int64)
        degree = graph.row_counts - 1
        self.row_counts = np.where(free, degree[:, None] * neighbour_width + own_width, 1).ravel()
        if np.any(self.row_counts == 0):
            raise ValueError('_PinnedPattern: a free row without entries')
        indptr = np.concatenate([[0], np.cumsum(self.row_counts)])
        self.nnz = int(indptr[-1])
        start = indptr[:-1].reshape(n_sites, components)
        diagonal = graph.diagonal

        def neighbour_slots(row, sites, positions):
            """Slot of the first column of the face neighbour at scalar `positions` in the rows (sites, row)."""
            wider = (own_width[row] - neighbour_width[row]) * (positions > diagonal[sites])   # the own site before it
            return start[sites, row] + positions * neighbour_width[row] + wider

        def own_slots(row, sites):
            """Slot of the first own-site column in the rows (sites, row)."""
            return start[sites, row] + diagonal[sites] * neighbour_width[row]

        indices = np.empty(self.nnz, dtype=np.int64)
        face_sites = np.concatenate([graph.face_lo, graph.face_hi])
        face_columns = np.concatenate([graph.face_hi, graph.face_lo])
        face_positions = np.concatenate([graph.lo_hi, graph.hi_lo])
        all_sites = np.arange(n_sites)
        for row in range(components):
            kept = free[face_sites, row]
            base, columns = neighbour_slots(row, face_sites[kept], face_positions[kept]), face_columns[kept]
            for offset, component in enumerate(neighbour[row]):
                indices[base + offset] = components * columns + component
            sites = all_sites[free[:, row]]
            base = own_slots(row, sites)
            for offset, component in enumerate(own[row]):
                indices[base + offset] = components * sites + component
        pinned_index = np.flatnonzero(self.pinned_rows)
        self.pinned_slots = indptr[:-1][pinned_index]   # the only entry of each pinned row
        indices[self.pinned_slots] = pinned_index
        index_dtype = np.int32 if max(self.size, self.nnz) < 2 ** 31 else np.int64
        self.indices, self.indptr = indices.astype(index_dtype), indptr.astype(index_dtype)
        # row of every CSR slot (the L1 equilibration's take of the row norms)
        self.slot_rows = np.repeat(np.arange(self.size, dtype=index_dtype), self.row_counts)
        slots = []   # CSR slot of every COO value in layout order
        for side, row, column_side, component in layout:
            sites = all_sites if side == _SITE else (graph.face_lo if side == _LO else graph.face_hi)
            if side == column_side:
                slot = own_slots(row, sites) + own[row].index(component)
            else:
                positions = graph.lo_hi if side == _LO else graph.hi_lo
                slot = neighbour_slots(row, sites, positions) + neighbour[row].index(component)
            slots.append(np.where(free[sites, row], slot, self.nnz))
        self.slots = np.concatenate(slots)

    def csr_values(self, values):
        """Float64 CSR data (nnz,) of COO values in layout order (pinned rows: 1 on the diagonal)."""
        data = np.bincount(self.slots, values, self.nnz + 1)[:self.nnz]
        data[self.pinned_slots] = 1.0
        return data

    def host_csr(self, data):
        """Canonical scipy CSR of data on this pattern (the index arrays shared, never modified downstream)."""
        import scipy.sparse
        matrix = scipy.sparse.csr_matrix((data, self.indices, self.indptr), shape=(self.size, self.size))
        matrix.has_canonical_format = True
        return matrix


def thermal_voltage(temperature):
    """k_B T / q [V] at temperature [K]."""
    if not (math.isfinite(temperature) and temperature > 0.0):
        raise ValueError(f'temperature must be finite and > 0 K, got {temperature!r}')
    return K_B * temperature / ELEMENTARY_CHARGE
