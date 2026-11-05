"""Shared renderer of triangle-mesh physics (mesh plan 2.9): fem_field(component) -> (Lagrange DofMap, host float64
coefficients) from the physics' own fem_field (nodal P1 / P2 unknowns, P0 element fields); Session rasterises and
samples them. device_field returns the same coefficients as a float32 run-backend vector (not a window image)."""
import numpy as np

from ..core.renderer import PhysicsRenderer


class FEMFieldRenderer(PhysicsRenderer):
    """Renderer of a physics with fem_field(component) -> (DofMap, coefficients); field_infos per instance."""

    def __init__(self, physics, name, field_infos, time_label=None):
        self.physics = physics
        self.name = name
        self.field_infos = dict(field_infos)
        self._time_label = time_label

    def fem_field(self, component) -> tuple:
        """(Lagrange DofMap, host float64 coefficients) of a component (P0 maps: one value per element)."""
        self.field_info(component)
        dofs, coefficients = self.physics.fem_field(component)
        return dofs, np.asarray(coefficients, dtype=np.float64)

    def device_field(self, component):
        """The fem_field coefficients as a float32 run-backend vector (1D, dof order)."""
        _, coefficients = self.fem_field(component)
        return self.physics.domain.xp.asarray(coefficients, dtype=np.float32)

    def time_label(self):
        return self._time_label() if self._time_label is not None else None
