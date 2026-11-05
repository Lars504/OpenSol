"""PhysicsSolver ABC: the solver contract (prepare, step, dt_limit, state, inject_source) every physics implements."""
from abc import ABC, abstractmethod


class PhysicsSolver(ABC):

    name: str = "unnamed"
    material_schema: dict = {}
    coordinate_grids: tuple = ('cartesian',)   # Domain.coordinates grids it solves on (Simulation.add_physics checks)

    @abstractmethod
    def prepare(self, domain) -> None:
        ...

    @abstractmethod
    def step(self, t: float, dt: float) -> None:
        ...

    @abstractmethod
    def dt_limit(self) -> float:
        ...

    @abstractmethod
    def state(self) -> dict:
        ...

    def inject_source(self, name: str, array) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not accept external source '{name}'"
        )

    def finalize(self) -> None:
        """Called once through Simulation.finalize (model.builder after all features and the first state-dependent
        material evaluation; Simulation.run's first call in scripts)."""

    def materials_changed(self, names) -> None:
        """MaterialGrid arrays named in `names` were rewritten in place."""

    def scalars(self) -> dict:
        """{name: float} of the current solution (PhysicsType.scalar_outputs names; derived values read them as
        <physics>_<name>); a declared name missing here has no value in this run."""
        return {}
