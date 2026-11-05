"""MultiphysicsRunner: steps named physics solvers in whole ticks of a shared dt_sync (per-solver stride) and
applies the couplings."""
import math

from .slot_sum import LOSS_FIELDS, SlotSum, slot_contributions, written_slots


class MultiphysicsRunner:
    """Steps named solvers in whole ticks of a shared dt_sync (per-solver stride), then applies couplings."""

    # State fields that need EM's J·E loss accounting to be live when a coupling reads them (ez jz_total is stepped
    # with any Drude pole, accounting or not, so it is not one).
    _LOSS_FIELDS = LOSS_FIELDS
    # Ticks per solve below which a `solves_every_step` quasi-static solver dominates the run time (as the heat
    # validation's MIN_EM_STEPS_PER_SOLVE).
    _MIN_TICKS_PER_SOLVE = 100

    def __init__(self):
        self.physics = {}
        self.strides = {}
        self.couplings = []
        self.t = 0.0
        self.tick = 0
        self._prepared = False
        self._dt_sync = math.inf
        self._schedule = ()
        self._coupling_schedule = ()
        self.prepare_warnings = []

    def add(self, name: str, solver, every: int = 1):
        if name in self.physics:
            raise ValueError(f"physics '{name}' is already attached")
        if int(every) < 1:
            raise ValueError(f"physics '{name}': every must be >= 1, got {every}")
        solver.name = name
        self.physics[name] = solver
        self.strides[name] = int(every)
        self._prepared = False

    def add_coupling(self, coupling):
        """Attach a coupling; E13 contributor terms (also those of a CouplingGroup) join the SlotSum of their (target,
        slot), created at the position of its first contributor. A slot has either SlotSum or overwriting writers."""
        for attr in ('apply', 'apply_now', 'flush', 'source_fields', 'every'):
            if not hasattr(coupling, attr):
                raise TypeError(f"coupling {coupling!r} lacks '{attr}' (coupling protocol: "
                                f"apply, apply_now, flush, source_fields, every)")
        contributions = slot_contributions(coupling)
        if not contributions:
            self._check_slot_writers(written_slots(coupling), coupling, sums=True)
            self.couplings.append(coupling)
        for term, every, after_source_step in contributions:
            self._slot_sum(term).add(term, every, after_source_step)
        self._prepared = False

    def _slot_sum(self, term):
        """The SlotSum of a contributor term's (target, slot), appended to couplings when new."""
        slot = (term.target, term.target_input)
        for coupling in self.couplings:
            if isinstance(coupling, SlotSum) and (coupling.target, coupling.target_input) == slot:
                return coupling
        self._check_slot_writers({slot}, term, sums=False)
        slot_sum = SlotSum(term.target, term.target_input, term.contribution)
        self.couplings.append(slot_sum)
        return slot_sum

    def _check_slot_writers(self, slots, coupling, sums):
        """ValueError when an attached SlotSum (sums) or overwriting coupling (not sums) also writes one of slots."""
        for other in self.couplings:
            overlap = sorted(slots & written_slots(other)) if isinstance(other, SlotSum) == sums else ()
            if overlap:
                target, slot = overlap[0]
                raise ValueError(f"coupling '{getattr(coupling, 'name', '')}' and coupling "
                                 f"'{getattr(other, 'name', '')}' both write slot '{slot}' of '{target}', one adding "
                                 'a contribution and one overwriting the slot; make both contributors or keep one')

    def prepare(self) -> list:
        """Cache dt_sync, validate strides and couplings, enable loss accounting read by couplings; returns (and keeps
        as prepare_warnings) the messages of _stride_cost_warnings and _resolve_couplings."""
        limits = {name: float(solver.dt_limit()) for name, solver in self.physics.items()}
        dt_sync = min(limits.values(), default=math.inf)
        if not dt_sync > 0:
            raise ValueError(f"dt_limit must be positive; got {limits}")
        for name, limit in limits.items():
            span = self.strides[name] * dt_sync
            if limit < span * (1.0 - 1e-9):
                raise ValueError(
                    f"physics '{name}' advances at most {limit:.4e} s per step, but every="
                    f"{self.strides[name]} makes it cover {span:.4e} s per step; use every=1 "
                    f"for it (sync dt = {dt_sync:.4e} s)")
        self.prepare_warnings = self._stride_cost_warnings(limits, dt_sync) + self._resolve_couplings()
        self._enable_loss_accounting()
        self._dt_sync = dt_sync
        self._schedule = tuple((solver, self.strides[name], self.strides[name] * dt_sync)
                               for name, solver in self.physics.items())
        # (coupling, every, phase): apply on ticks with (tick + 1 - phase) % every == 0, phase 1 = after_source_step.
        self._coupling_schedule = tuple(
            (coupling, int(coupling.every), int(bool(getattr(coupling, 'after_source_step', False))))
            for coupling in self.couplings)
        self._prepared = True
        return self.prepare_warnings

    def _stride_cost_warnings(self, limits: dict, dt_sync: float) -> list:
        """Messages for strides that waste solves: a span below 1e-3 of the solver's finite dt_limit, or a solver
        with `solves_every_step` (steady heat: no dirty check) stepped more often than every _MIN_TICKS_PER_SOLVE."""
        if not math.isfinite(dt_sync):
            return []
        messages = []
        for name, limit in limits.items():
            stride = self.strides[name]
            span = stride * dt_sync
            if math.isfinite(limit) and span < 1e-3 * limit:
                messages.append(
                    f"physics '{name}' steps by every * dt_sync = {span:.4e} s, below 1e-3 of its dt_limit "
                    f"{limit:.4e} s, so it solves every {stride} tick(s) for little progress; raise every (at most "
                    f"{int(limit / dt_sync)}, a few hundred is usually enough; a coupling reading it, e.g. "
                    f"em_heat_joule, needs the same every)")
            elif (not math.isfinite(limit) and stride < self._MIN_TICKS_PER_SOLVE
                  and getattr(self.physics[name], 'solves_every_step', False)):
                messages.append(
                    f"physics '{name}' re-solves at every step and every={stride} runs it every {stride} tick(s) "
                    f"of dt_sync = {dt_sync:.4e} s; one solve costs far more than a tick: set every to a few "
                    f"hundred (e.g. one heat solve per em_heat_joule window, the coupling with the same every)")
        return messages

    def _resolve_couplings(self) -> list:
        """Raise for a coupling naming an unattached physics, set every=None to the largest stride of its physics;
        messages for a coupling reading _LOSS_FIELDS whose every differs from its target's (window != solve)."""
        messages = []
        for coupling in self.couplings:
            label = getattr(coupling, 'name', '') or type(coupling).__name__
            sources = {physics_name for physics_name, _ in coupling.source_fields}
            targets = {term.target for term in getattr(coupling, 'terms', (coupling,)) if hasattr(term, 'target')}
            unknown = sorted((sources | targets) - set(self.physics))
            if unknown:
                raise ValueError(f"coupling '{label}' references unknown physics {unknown}; "
                                 f"attached: {list(self.physics)}")
            if getattr(coupling, 'every', 1) is None:
                if not sources | targets:
                    raise ValueError(f"coupling '{label}': every=None needs a source or target physics")
                coupling.every = max(self.strides[physics_name] for physics_name in sources | targets)
            every = getattr(coupling, 'every', None)
            loss_fields = sorted(field for _, field in coupling.source_fields if field in self._LOSS_FIELDS)
            for target in sorted(targets) if loss_fields and every is not None else ():
                if every != self.strides[target]:
                    messages.append(
                        f"coupling '{label}' reads {loss_fields} with every={every}, but its target '{target}' "
                        f"steps every {self.strides[target]} ticks: each transfer resets the absorbed-power window, "
                        f"so '{target}' gets a window that does not match its steps (CW: the {every}-tick J.E, not "
                        f"the mean; pass every=None or every={self.strides[target]})")
        return messages

    def _enable_loss_accounting(self):
        for coupling in self.couplings:
            for physics_name, field in coupling.source_fields:
                if field not in self._LOSS_FIELDS:
                    continue
                solver = self.physics[physics_name]   # attached: _resolve_couplings checked the names
                engine = getattr(solver, 'engine', solver)
                if not hasattr(engine, 'enable_q_accum'):
                    raise ValueError(f"coupling reads '{field}' from physics '{physics_name}', "
                                     f"which has no loss accounting (enable_q_accum)")
                engine.enable_q_accum = True

    @property
    def dt_sync(self) -> float:
        """min(dt_limit()) over the attached physics (inf when nothing steps in time)."""
        if self._prepared:
            return self._dt_sync
        return min((float(s.dt_limit()) for s in self.physics.values()), default=math.inf)

    def advance(self, n_ticks: int, t_end: float = None) -> int:
        """Run up to n_ticks whole ticks (stopping once t >= t_end - dt_sync/2); returns ticks run."""
        if not self._prepared:
            self.prepare()
        dt_sync = self._dt_sync
        if not math.isfinite(dt_sync):
            raise ValueError("nothing steps in time (every physics has an infinite dt_limit); "
                             "use a stationary study")
        stop_time = math.inf if t_end is None else t_end - 0.5 * dt_sync
        schedule = self._schedule
        coupling_schedule = self._coupling_schedule
        physics = self.physics
        n_done = 0
        while n_done < n_ticks and self.t < stop_time:
            tick = self.tick
            t = self.t
            for solver, stride, span in schedule:
                if tick % stride == 0:
                    solver.step(t, span)
            # the clock moves first: a time-dependent update (t of materials / features) reads the next step's start
            self.t = t + dt_sync
            self.tick = tick + 1
            for coupling, every, phase in coupling_schedule:
                if (tick + 1 - phase) % every == 0:
                    coupling.apply(physics)
            n_done += 1
        return n_done

    def flush(self):
        """End of a time-dependent run: flush the couplings that feed their target's next step (em_heat_joule merges
        the partial window with the last full one), one step(t, inf) of each quasi-static physics, then flush the
        after_source_step ones (material updates, heat -> EM) so they carry the final state."""
        after_source = [coupling for coupling in self.couplings if getattr(coupling, 'after_source_step', False)]
        for coupling in self.couplings:
            if coupling not in after_source:
                coupling.flush(self.physics)
        for solver in self.physics.values():
            if not math.isfinite(solver.dt_limit()):
                solver.step(self.t, math.inf)
        for coupling in after_source:
            coupling.flush(self.physics)

    def clear(self):
        """Drop every solver, stride, coupling and the cached schedule (e.g. when a Session closes)."""
        self.physics.clear()
        self.strides.clear()
        self.couplings.clear()
        self._schedule = ()
        self._coupling_schedule = ()
        self._prepared = False
        self._dt_sync = math.inf
