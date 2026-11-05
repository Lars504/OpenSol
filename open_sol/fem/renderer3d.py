"""Shared renderer of tetrahedron-mesh scalar physics (igor3d_plan T4b; the 2D fem.renderer.FEMFieldRenderer
analogue): sample(component, points) of the physics' lagrange_field(component) -> (DofMap3D, host coefficients) (nodal
P1 / P2 unknowns, P0 element fields), NaN outside the mesh: the Session's tetrahedron reads; device_field = the values
at the element centroids as a float32 run-backend (n_elements,) vector (not a window image)."""
import numpy as np

from ..core.renderer import PhysicsRenderer
from .sampling3d import centroid_values_3d, evaluate_lagrange_3d


class LagrangeSampleRenderer(PhysicsRenderer):
    """Renderer of a physics with `domain` (TetrahedronDomain) and lagrange_field(component) -> (DofMap3D,
    coefficients); field_infos per instance."""

    def __init__(self, physics, name, field_infos, time_label=None):
        self.physics = physics
        self.name = name
        self.field_infos = dict(field_infos)
        self._time_label = time_label

    def lagrange_field(self, component) -> tuple:
        """(DofMap3D, host float64 coefficients) of a component (P0 maps: one value per element)."""
        self.field_info(component)
        dofs, coefficients = self.physics.lagrange_field(component)
        return dofs, np.asarray(coefficients, dtype=np.float64)

    def sample(self, component, points) -> np.ndarray:
        """Host float64 (p,) values of `component` at points (p, 3) [m]; NaN outside the mesh (locations remembered by
        the domain)."""
        dofs, coefficients = self.lagrange_field(component)
        elements, reference = self.physics.domain.locate(points)
        values, _ = evaluate_lagrange_3d(dofs, coefficients, elements, reference, gradients=False)
        return values

    def device_field(self, component):
        """float32 run-backend (n_elements,) values at the element centroids."""
        dofs, coefficients = self.lagrange_field(component)
        return self.physics.domain.xp.asarray(centroid_values_3d(dofs, coefficients), dtype=np.float32)

    def time_label(self):
        return self._time_label() if self._time_label is not None else None
