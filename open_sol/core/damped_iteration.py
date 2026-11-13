"""Damped fixed-point / Newton loop shared by the nonlinear DiffusionSystem loops (electrostatics Poisson-Boltzmann,
magnetic B-H Picard; plans_overview decision 7), optionally Anderson-accelerated. Array-module agnostic (numpy or cupy
state; numpy for the host least squares)."""
import math
from dataclasses import dataclass

import numpy as np

from .backend import array_module_of

_LSTSQ_RCOND = 1e-10   # Anderson least squares: singular values below this x the largest are dropped


@dataclass(frozen=True)
class DampingSettings:
    """Knobs of one damped loop (physics settings, never a solver role): stop, iteration limit, relaxation, step cap,
    Anderson depth (0 = damped fixed point)."""
    tolerance: float = 1e-6          # converged when max |proposal - state| / scale <= tolerance
    max_iterations: int = 100
    relaxation: float = 1.0          # fraction of the proposed update applied, (0, 1]
    max_step: float = math.inf       # cap of the applied max |update| / scale per iteration
    min_relaxation: float = 1.0 / 64.0   # floor of the adaptive halving
    anderson_depth: int = 0          # > 0: Anderson mixing of the last depth iterates (fixed relaxation, no halving)

    def __post_init__(self):
        if not self.tolerance > 0.0:
            raise ValueError(f"damped iteration tolerance must be > 0, got {self.tolerance}")
        if int(self.max_iterations) < 1:
            raise ValueError(f"damped iteration max_iterations must be >= 1, got {self.max_iterations}")
        if not 0.0 < self.relaxation <= 1.0:
            raise ValueError(f"damped iteration relaxation must be in (0, 1], got {self.relaxation}")
        if not self.max_step > 0.0:
            raise ValueError(f"damped iteration max_step must be > 0, got {self.max_step}")
        if not 0.0 < self.min_relaxation <= self.relaxation:
            raise ValueError(f"damped iteration min_relaxation must be in (0, relaxation], got {self.min_relaxation}")
        if int(self.anderson_depth) != self.anderson_depth or self.anderson_depth < 0:
            raise ValueError(f"damped iteration anderson_depth must be an integer >= 0, got {self.anderson_depth!r}")


class AndersonHistory:
    """Walker-Ni Anderson mixing of depth m in the scaled norm: step = beta f - (dX + beta dF) gamma with gamma =
    argmin |(f - dF gamma) / scale|; differences of successive iterates in (m, N) ring buffers (also the stationary
    study's loop-back mixing, model.study)."""

    def __init__(self, depth):
        self.depth = int(depth)
        self._dx = self._df = self._weighted = None
        self._x_previous = self._f_previous = None
        self._count = self._next = 0
        self._weighted_scale = None   # the scale the kept weighted df rows were divided by (None: none kept)

    def _same_scale(self, scale):
        """True when scale is the one the kept weighted rows used: the same array object (never changed in place
        during a run), or an equal plain number."""
        kept = self._weighted_scale
        if kept is None:
            return False
        if hasattr(scale, 'shape') or hasattr(kept, 'shape'):
            return scale is kept
        return float(scale) == float(kept)

    def step(self, state, update, scale, relaxation):
        """Applied change of state for the undamped update f = proposal - state (one host transfer)."""
        if self._dx is None:
            flat = state.reshape(1, -1)
            self._dx = flat.repeat(self.depth, axis=0)   # contents overwritten before use
            self._df = flat.repeat(self.depth, axis=0)
            self._weighted = flat.repeat(self.depth + 1, axis=0)
            self._x_previous = state.copy()
        else:
            row = self._next % self.depth
            self._dx[row] = (state - self._x_previous).ravel()
            self._df[row] = (update - self._f_previous).ravel()
            self._next += 1
            self._count = min(self._count + 1, self.depth)
            self._x_previous[...] = state
        self._f_previous = update   # a new array per iteration (proposal - state)
        step = relaxation * update
        count = self._count
        if count == 0:
            return step
        if self._same_scale(scale):   # only the newest df row is new (the update slot is rewritten below)
            newest = (self._next - 1) % self.depth
            self._weighted[newest] = (self._df[newest].reshape(state.shape) / scale).ravel()
        else:
            shape = (count,) + state.shape
            self._weighted[:count] = (self._df[:count].reshape(shape) / scale).reshape(count, -1)
            self._weighted_scale = scale
        self._weighted[count] = (update / scale).ravel()
        products = np.asarray((self._weighted[:count + 1] @ self._weighted[:count].T).tolist())
        gamma = np.linalg.lstsq(products[:count], products[count], rcond=_LSTSQ_RCOND)[0]
        # step -= (dX + beta dF)^T gamma as two GEMVs over the ring rows (one small upload, no per-row loop)
        xp = array_module_of(self._dx)
        weights = xp.asarray(gamma, dtype=self._dx.dtype)
        step -= (weights @ self._dx[:count]).reshape(state.shape)
        step -= (weights @ self._df[:count]).reshape(state.shape) * relaxation
        return step


class DampedIteration:
    """Drives state <- state + factor (propose(state) - state) until max |update| / scale <= tolerance; the factor
    halves while the change grows and recovers after it shrinks (anderson_depth > 0: Anderson-mixed steps instead);
    keeps the convergence record and work counters."""

    def __init__(self, settings: DampingSettings, label):
        if not callable(label):
            raise TypeError(f"label must be a zero-argument callable, got {type(label).__name__}")
        self.settings = settings
        self.label = label
        self.n_runs = 0
        self.total_iterations = 0
        self.n_unconverged = 0
        self.last_iterations = 0
        self.last_change = math.nan       # max |update| / scale of the last proposal (undamped)
        self.last_converged = True
        self.last_warning = None          # message of the latest unconverged run (never cleared)

    def run(self, state, propose, scale) -> bool:
        """Iterate on `state` in place (propose(state) -> a new array of its shape, never a view of state); scale > 0
        a float or broadcastable array. Converged: state = the last proposal. Returns converged."""
        settings = self.settings
        relaxation = settings.relaxation
        previous_change = math.inf
        mixer = AndersonHistory(settings.anderson_depth) if settings.anderson_depth else None
        self.n_runs += 1
        self.last_converged = False
        for iteration in range(1, int(settings.max_iterations) + 1):
            proposal = propose(state)
            update = proposal - state
            change = float(abs(update / scale).max())   # the one host sync per iteration
            self.total_iterations += 1
            self.last_iterations = iteration
            self.last_change = change
            if not math.isfinite(change):
                raise FloatingPointError(f"{self.label()}: the damped iteration produced a non-finite update at "
                                         f"iteration {iteration}")
            if change <= settings.tolerance:
                state[...] = proposal
                self.last_converged = True
                return True
            if mixer is not None:   # Anderson: fixed relaxation, the step cap on the mixed step
                step = mixer.step(state, update, scale, relaxation)
                if math.isfinite(settings.max_step):
                    step_size = float(abs(step / scale).max())
                    if step_size > settings.max_step:
                        step *= settings.max_step / step_size
                state += step
                previous_change = change
                continue
            if change > previous_change:
                relaxation = max(settings.min_relaxation, 0.5 * relaxation)
            elif relaxation < settings.relaxation:
                relaxation = min(settings.relaxation, 2.0 * relaxation)
            factor = relaxation * min(1.0, settings.max_step / change)
            state += factor * update
            previous_change = change
        self.n_unconverged += 1
        self.last_warning = (f"{self.label()}: the damped iteration did not converge in {self.last_iterations} "
                             f"iterations (last change {self.last_change:.2e} of tolerance {settings.tolerance:.1e}); "
                             'the damped state is kept')
        return False

    def work_counters(self) -> dict:
        """Cumulative, machine-independent loop work (study summary 'solvers' physics counters)."""
        return {'nonlinear_runs': self.n_runs, 'nonlinear_iterations': self.total_iterations,
                'nonlinear_unconverged': self.n_unconverged}
