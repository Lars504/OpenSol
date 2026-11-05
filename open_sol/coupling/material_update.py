"""DynamicMaterialUpdate (a state- or time-dependent material property rewritten into MaterialGrid arrays under a mask)
and TimeDependentUpdate (feature values of the runner time t)."""
from .coupling_term import state_revisions


class DynamicMaterialUpdate:
    """Re-evaluates a state- or time-dependent material property and writes it into MaterialGrid arrays under a mask."""

    def __init__(self, materials, grid_properties: tuple, mask, evaluate, every: int = 1,
                 name: str = '', source_fields=frozenset(), time_dependent: bool = False, notify=None):
        if int(every) < 1:
            raise ValueError(f"DynamicMaterialUpdate every must be >= 1, got {every}")
        grid_properties = tuple(grid_properties)
        if not grid_properties:
            raise ValueError(f"DynamicMaterialUpdate '{name}' writes no material property")
        registered = materials.property_names()
        unknown = [prop for prop in grid_properties if prop not in registered]
        if unknown:
            raise AttributeError(f"DynamicMaterialUpdate '{name}': MaterialGrid has no properties "
                                 f"{unknown}; registered: {list(registered)}")
        self.materials = materials
        self.grid_properties = grid_properties
        self.mask = mask
        self.evaluate = evaluate
        self.every = int(every)
        self.name = name
        self.source_fields = frozenset(source_fields)
        self.time_dependent = bool(time_dependent)   # reads t: rewritten at every cadenced tick, sources clean or not
        self.notify = None if notify is None else frozenset(notify)   # physics told materials_changed (None: all)
        # reads physics with stride every: rewrite right after it steps; time only: before the next steps (the runner
        # clock already reads their start time)
        self.after_source_step = bool(self.source_fields)
        self._changed_names = frozenset(grid_properties)
        self._source_physics = tuple(sorted({physics_name for physics_name, _ in self.source_fields}))
        self._applied_revision = None   # source state_revisions at the last apply_now

    def apply(self, physics_by_name: dict):
        """Cadenced update (the runner picks the ticks); skipped while no source physics solved / stepped since the
        last update (a clean quasi-static source), unless the property reads t."""
        revision = state_revisions(physics_by_name, self._source_physics)
        if self.time_dependent or revision is None or revision != self._applied_revision:
            self.apply_now(physics_by_name)

    def apply_now(self, physics_by_name: dict):
        """Evaluate, assign to every grid property under the mask, then notify every solver's materials_changed."""
        self._applied_revision = state_revisions(physics_by_name, self._source_physics)
        value = self.transfer_value(physics_by_name)
        if value is not None:
            self.write(physics_by_name, value)

    def transfer_value(self, physics_by_name: dict):
        """The value apply_now would assign (None = nothing); may be a live array: copy before keeping it."""
        return self.evaluate(physics_by_name)

    def write(self, physics_by_name: dict, value):
        """Assign value to every grid property under the mask, then notify materials_changed of every solver (of the
        `notify` physics only: S7.2, those on this MaterialGrid's mesh)."""
        for prop in self.grid_properties:
            self.materials.assign(prop, self.mask, value)
        for name, solver in physics_by_name.items():
            hook = getattr(solver, 'materials_changed', None)
            if hook is not None and (self.notify is None or name in self.notify):
                hook(self._changed_names)

    def flush(self, physics_by_name: dict):
        """End-of-run update (runner.flush) = apply_now."""
        self.apply_now(physics_by_name)


class TimeDependentUpdate:
    """Feature value of time t: setter(value_at(t)) whenever the runner clock moved, before the target physics steps
    (every = its stride; the clock then reads the step start time). Coupling protocol, no source fields; the feature
    already set the value at the clock time of construction."""

    def __init__(self, name: str, target: str, value_at, setter, clock, every: int = 1):
        if int(every) < 1:
            raise ValueError(f"TimeDependentUpdate every must be >= 1, got {every}")
        self.name = name
        self.target = target              # physics whose feature this is (study code: a coupling target)
        self.value_at = value_at          # t [s] -> float
        self.setter = setter              # float -> None, the feature's own update of its solver
        self.clock = clock                # object with the runner time `t` [s] (the MultiphysicsRunner)
        self.every = int(every)
        self.after_source_step = False
        self.source_fields = frozenset()
        self._applied_time = float(clock.t)   # clock time of the last setter call

    def apply(self, physics_by_name: dict):
        self.apply_now(physics_by_name)

    def apply_now(self, physics_by_name: dict):
        """setter(value at the clock time), skipped when the clock did not move since the last call."""
        time = float(self.clock.t)
        if time != self._applied_time:
            self._applied_time = time
            self.setter(self.value_at(time))

    def flush(self, physics_by_name: dict):
        """End-of-run update (runner.flush) = apply_now: the value at the final time."""
        self.apply_now(physics_by_name)
