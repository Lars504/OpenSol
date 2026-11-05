"""Heat conduction on a tetrahedron mesh (igor3d_plan T4b; the 3D analogue of solver_fem.py): FEMHeatPhysics3D, P1 /
P2 Lagrange on the conducting window elements (fem.diffusion3d), steady or implicit-Euler transient, diagonal k_xx /
k_yy / k_zz; the HeatPhysics field names, slot and feature API where they apply. State arrays are per element (D12: T
at the element centroid); the renderer samples T at points (fem.renderer3d)."""
import math

import numpy as np

from ...core.fv_diffusion import BCSide
from ...fem.diffusion3d import FEMDiffusionPhysicsBase3D, coefficient_tensor_3d
from ...fem.mesh3d import sliver_warnings
from ...fem.renderer3d import LagrangeSampleRenderer
from .renderer import HeatRenderer
from .solver_fem import ANCHOR_TEXT, REMEDY_TEXT, FEMHeatMixin


class FEMHeatPhysics3D(FEMHeatMixin, FEMDiffusionPhysicsBase3D):
    """Heat on a fem.domain3d.TetrahedronDomain: the rise T - T0 by Lagrange elements of `order` (2 quadratic, 1
    linear), steady or implicit-Euler transient; T, Q, Q_ext and heat_flux_abs per element (state), T at points
    (lagrange_field / the renderer)."""

    _operator_materials = frozenset({'k_xx', 'k_yy', 'k_zz'})

    def __init__(self, domain, T0: float = 293.15, order: int = 2, time_mode: str = 'stationary',
                 time_step: float = math.inf, linear_solver=None, require_anchor: bool = True):
        self._init_heat(T0, time_mode, time_step)
        self.finalize_warnings = []
        self._init_fem(domain, order, linear_solver, lambda: f"heat '{self.name}'", 'temperature',
                       ANCHOR_TEXT.format(boundary='face'), REMEDY_TEXT, require_anchors=require_anchor)
        self.prepare(domain)

    def finalize(self) -> None:
        """Re-record the thermo-optic reference indices, assemble (fails early on unanchored regions); a warning on
        sliver elements."""
        super().finalize()
        self.finalize_warnings = sliver_warnings(self.domain.mesh)

    def renderer(self):
        return LagrangeSampleRenderer(self, 'heat', HeatRenderer.field_infos, self._time_label)

    lagrange_field = FEMHeatMixin._field

    # --- interface API (the definition callbacks on tetrahedra) ----------------
    def add_surface_temperature(self, faces, T) -> int:
        """Dirichlet T [K] on mesh faces (an object interface); handle for set_surface_condition."""
        return self.add_surface_condition(faces, BCSide('dirichlet', value=T))

    def add_surface_convection(self, faces, h, T_ext) -> None:
        """Convective loss h (T - T_ext) per unit area on mesh faces."""
        self.add_surface_condition(faces, BCSide('robin', h=h, external=T_ext))

    def add_surface_flux(self, faces, flux) -> int:
        """Inward heat flux [W/m^2] on mesh faces (an interface: a sheet source, both sides take heat); handle for
        set_surface_flux."""
        return self.add_surface_condition(faces, BCSide('neumann', value=flux))

    def set_surface_flux(self, handle, flux) -> None:
        self.set_surface_condition(handle, BCSide('neumann', value=flux))

    def add_surface_radiation(self, faces, emissivity, ambient_temperature) -> None:
        """Surface-to-ambient radiation on mesh faces (an object interface)."""
        self._add_radiation_term(faces, emissivity, ambient_temperature)

    def _face_entities(self, face):
        return self.system.face_faces(face)

    def _entity_count(self) -> int:
        return self.domain.mesh.n_faces

    def _add_entity_condition(self, faces, condition) -> int:
        return self.system.add_surface_condition(faces, condition)

    def _set_entity_condition(self, handle, condition) -> None:
        self.system.set_surface_condition(handle, condition)

    def _entity_rise(self, faces) -> np.ndarray:
        """Rise at the face centroids (P2: -1/9 of each corner + 4/9 of each edge dof; P1: the corner mean)."""
        local = self.system.full.face_local_dofs(faces)   # corners, then (P2) the edges ab, ac, bc
        values = self.system.values[local]
        if self.system.order == 2:
            return (4.0 * values[:, 3:].sum(axis=1) - values[:, :3].sum(axis=1)) / 9.0
        return values.mean(axis=1)

    def add_point_heat_source_at(self, point, power: float) -> None:
        """Point heat source of total power P [W] at (x, y, z) [m]."""
        self.add_point_source_at(point, power)

    def _coefficient_tensor(self, materials):
        return coefficient_tensor_3d(materials.k_xx, materials.k_yy, materials.k_zz)
