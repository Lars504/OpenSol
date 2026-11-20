"""Tangent refresh rule and Newton proposal of the total-Lagrangian solids (2D triangles, 3D tetrahedra): full Newton
or modified Newton that keeps the factorised tangent while the residual falls fast. Tangent data of a device
TangentAssembler3D go to the system's device solve. The full tangent data are dropped before each factorisation:
only its free-free values reach the factor, and modified Newton keeps the held-dof rows of the factor's tangent on the
host (`HeldRows`: enough for the lift onto a load step's held values, the tangent being symmetric). numpy only."""
import math

import numpy as np

TANGENT_UPDATES = ('every_iteration', 'when_slow')
# when_slow: the held factor is reused while max |free residual| fell below this fraction of the previous iteration's
SLOW_RESIDUAL_RATIO = 0.25


class TangentPolicy:
    """Factor bookkeeping of one nonlinear solid: 'every_iteration' refactorises the tangent per Newton iteration,
    'when_slow' keeps the held factor while each iteration cut the free residual below SLOW_RESIDUAL_RATIO."""

    def __init__(self, tangent_update='when_slow'):
        if tangent_update not in TANGENT_UPDATES:
            raise ValueError(f"tangent_update must be one of {', '.join(TANGENT_UPDATES)}, got {tangent_update!r}")
        self.tangent_update = tangent_update
        self.n_factorizations = 0   # cumulative tangent factorisations (work counter tangent_factorizations)
        self.n_reuses = 0           # cumulative iterations solved on a held factor
        self.reset()

    def reset(self) -> None:
        """Drop the held factor (a new solve or a cold-ramp restart: the runtime may hold another factor by now)."""
        self.data = None            # host HeldRows values of the held factor's tangent, None = no factor held
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
    """A device TangentAssembler3D: its tangent data live on the GPU."""
    return bool(getattr(assembler, 'device', False))


def _full_product(assembler, data, vector) -> np.ndarray:
    """Host full tangent of data times a host vector (on the device for device data)."""
    return assembler.full_product(data, vector) if _device(assembler) else assembler.full(data) @ vector


def _free_tangent(assembler, data):
    """Free-free part of tangent data, an own copy: device values on the host reduced pattern (device data) or the host
    reduced CSR, so the full data can be dropped before the factorisation."""
    return assembler.reduced_values(data) if _device(assembler) else assembler.reduced(data)


def _solve_tangent(system, assembler, free_tangent, rhs) -> np.ndarray:
    """Free-dof solution on a new factor of a _free_tangent: device values refresh the cuDSS factor on the device, a
    host reduced CSR is handed over."""
    if _device(assembler):
        pattern = assembler.pattern
        return system.solve_tangent_device(free_tangent, pattern.reduced_indptr, pattern.reduced_indices, rhs)
    return system.solve_tangent_reduced(free_tangent, rhs)


class HeldRows:
    """Rows of the held dofs of the assembler's fixed tangent pattern (host CSR index arrays, built once per Newton
    solve): `values(data)` takes the host values of those rows from tangent data (device data: gathered on the device
    by a kept device take, only the held entries cross), `lift(values, correction)` = T[:, held] correction[held] =
    rows^T correction[held] (T symmetric: the 'linear' factor vouches it), the full (n,) host product of a correction
    that is 0 off the held dofs."""

    def __init__(self, assembler, held):
        pattern = assembler.pattern
        self.n = int(pattern.n)
        self.rows = np.flatnonzero(np.asarray(held, dtype=bool))
        indptr = np.asarray(pattern.indptr, dtype=np.int64)
        starts = indptr[self.rows]
        self._counts = indptr[self.rows + 1] - starts   # entries per held row
        row_starts = np.zeros(len(self.rows) + 1, dtype=np.int64)
        np.cumsum(self._counts, out=row_starts[1:])
        index = np.int32 if int(pattern.nnz) < 2 ** 31 else np.int64
        take = (np.repeat(starts - row_starts[:-1], self._counts)
                + np.arange(int(row_starts[-1]), dtype=np.int64)).astype(index)
        self.columns = np.asarray(pattern.indices)[take]
        self._device = _device(assembler)
        self._xp = assembler.xp if self._device else np
        self._take = self._xp.asarray(take)   # device data: on the device (the held entries' only device array)

    def values(self, data) -> np.ndarray:
        """Host float64 values (held entries,) of the held rows of tangent data (host or device)."""
        if self._device:
            return self._xp.asnumpy(data[self._take])
        return data[self._take]

    def lift(self, values, correction) -> np.ndarray:
        """Host (n,) T correction for a correction that is 0 off the held dofs (values: `values` of T)."""
        weights = values * np.repeat(np.asarray(correction, dtype=np.float64)[self.rows], self._counts)
        return np.bincount(self.columns, weights=weights, minlength=self.n)


def newton_proposer(policy, assembler, system, fixed_stress, held, free, scale, floor, latest):
    """propose_for(load, held_values) -> propose(current) of the Newton loop on the fixed-pattern tangent assembler:
    tangent + force (or the force alone on a reused factor), the held dofs lifted onto the step's values, one solve.
    The full tangent data are dropped before the factorisation (its free-free copy and, modified Newton, the host
    HeldRows values live on)."""
    held_rows = HeldRows(assembler, held) if policy.tangent_update == 'when_slow' else None

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
            if not reuse:   # the held rows of a dead factor: freed before a new tangent is assembled
                policy.data = None
            if not reuse and data is None:   # the residual fell too slowly: the tangent at this state after all
                _, data = assembler.force_and_tangent(current, initial_stress=fixed_stress)
            correction = np.where(held, held_values - current, 0.0)   # the held dofs onto this step's values
            if correction.any():   # first iteration of a load step (a reused factor lifts with its own held rows)
                residual += (held_rows.lift(policy.data, correction) if reuse
                             else _full_product(assembler, data, correction))
            proposal = current + correction
            if reuse:
                policy.n_reuses += 1
                proposal[free] -= system.solve_last_tangent(residual[free])
            else:
                kept = None if held_rows is None else held_rows.values(data)   # full Newton holds nothing
                free_tangent = _free_tangent(assembler, data)
                data = None   # the full tangent never lives beside the factorisation
                policy.n_factorizations += 1
                proposal[free] -= _solve_tangent(system, assembler, free_tangent, residual[free])
                del free_tangent
                policy.data = kept   # set after a successful solve: a failed factorisation leaves nothing reusable
            scale[...] = max(floor, float(np.abs(proposal).max()))
            return proposal
        return propose
    return propose_for
