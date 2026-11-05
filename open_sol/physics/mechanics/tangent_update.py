"""Tangent refresh rule and Newton proposal of the total-Lagrangian solids (2D triangles, 3D tetrahedra): full Newton
or modified Newton that keeps the factorised tangent while the residual falls fast (plan G4). Tangent data of a device
TangentAssembler3D go to the system's device solve (plan G5). numpy only."""
import math

import numpy as np

TANGENT_UPDATES = ('every_iteration', 'when_slow')
# when_slow: the held factor is reused while max |free residual| fell below this fraction of the previous iteration's
SLOW_RESIDUAL_RATIO = 0.25


class TangentPolicy:
    """Factor bookkeeping of one nonlinear solid: 'every_iteration' refactorises the tangent per Newton iteration,
    'when_slow' keeps the held factor while each iteration cut the free residual below SLOW_RESIDUAL_RATIO."""

    def __init__(self, tangent_update='every_iteration'):
        if tangent_update not in TANGENT_UPDATES:
            raise ValueError(f"tangent_update must be one of {', '.join(TANGENT_UPDATES)}, got {tangent_update!r}")
        self.tangent_update = tangent_update
        self.n_factorizations = 0   # cumulative tangent factorisations (work counter tangent_factorizations)
        self.n_reuses = 0           # cumulative iterations solved on a held factor
        self.reset()

    def reset(self) -> None:
        """Drop the held factor (a new solve or a cold-ramp restart: the runtime may hold another factor by now)."""
        self.data = None            # tangent values (fixed-pattern data) of the held factor, None = none held
        self._previous = None       # max |free residual| of the previous iteration of this load step
        self._ratio = math.inf      # residual ratio of the latest iteration that had a previous one

    def start_step(self) -> None:
        """A new load step: its first iteration has no previous residual and keeps the last ratio."""
        self._previous = None

    def may_reuse(self) -> bool:
        """True when this iteration could solve on the held factor (decided after its residual by `reuse`)."""
        return self.tangent_update == 'when_slow' and self.data is not None

    def reuse(self, free_residual) -> bool:
        """Record max |free residual| of this iteration; True = solve on the held factor (when_slow, ratio small)."""
        norm = float(np.max(np.abs(free_residual))) if free_residual.size else 0.0
        previous = self._previous
        if previous is not None:
            self._ratio = norm / previous if previous > 0.0 else (0.0 if norm == 0.0 else math.inf)
        self._previous = norm
        return self.may_reuse() and self._ratio < SLOW_RESIDUAL_RATIO   # NaN ratios refactorise

    def counters(self) -> dict:
        return {'tangent_factorizations': self.n_factorizations, 'tangent_reuses': self.n_reuses}


def _device(assembler) -> bool:
    """A device TangentAssembler3D (plan G5): its tangent data live on the GPU."""
    return bool(getattr(assembler, 'device', False))


def _full_product(assembler, data, vector) -> np.ndarray:
    """Host full tangent of data times a host vector (on the device for device data)."""
    return assembler.full_product(data, vector) if _device(assembler) else assembler.full(data) @ vector


def _solve_tangent(system, assembler, data, rhs) -> np.ndarray:
    """Free-dof solution on a new factor of the tangent data: device data refresh the cuDSS factor on the device, host
    data hand the reduced host CSR over."""
    if _device(assembler):
        pattern = assembler.pattern
        return system.solve_tangent_device(assembler.reduced_values(data), pattern.reduced_indptr,
                                           pattern.reduced_indices, rhs)
    return system.solve_tangent_reduced(assembler.reduced(data), rhs)


def newton_proposer(policy, assembler, system, fixed_stress, held, free, scale, floor, latest):
    """propose_for(load, held_values) -> propose(current) of the Newton loop on the fixed-pattern tangent assembler:
    tangent + force (or the force alone on a reused factor), the held dofs lifted onto the step's values, one solve."""

    def propose_for(load, held_values):
        policy.start_step()

        def propose(current):
            data = None
            if policy.may_reuse():
                force = assembler.force(current, initial_stress=fixed_stress)
            else:
                force, data = assembler.force_and_tangent(current, initial_stress=fixed_stress)
            latest.update(state=current.copy(), force=force)
            residual = force - load
            reuse = policy.reuse(residual[free])
            if not reuse and data is None:   # the residual fell too slowly: the tangent at this state after all
                _, data = assembler.force_and_tangent(current, initial_stress=fixed_stress)
            correction = np.where(held, held_values - current, 0.0)   # the held dofs onto this step's values
            if correction.any():   # first iteration of a load step (a reused factor lifts with its own tangent)
                residual += _full_product(assembler, policy.data if reuse else data, correction)
            proposal = current + correction
            if reuse:
                policy.n_reuses += 1
                proposal[free] -= system.solve_last_tangent(residual[free])
            else:
                policy.data = None   # a failed factorisation leaves nothing reusable
                policy.n_factorizations += 1
                proposal[free] -= _solve_tangent(system, assembler, data, residual[free])
                policy.data = data if policy.tangent_update == 'when_slow' else None   # full Newton holds nothing
            scale[...] = max(floor, float(np.abs(proposal).max()))
            return proposal
        return propose
    return propose_for
