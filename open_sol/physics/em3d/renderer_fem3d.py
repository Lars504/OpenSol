"""Renderer of em_fdfd_3d on a 'tetrahedra' mesh (igor3d_plan T4b): the FDFDRenderer3D field names sampled at points
(sample(component, points): Re / Im / modulus of E and H, e_abs, e_intensity, h_abs, w_e, eps_r per element; NaN
outside the mesh), the Session's tetrahedron reads; device_field = the values at the element centroids."""
import re
from dataclasses import replace

import numpy as np

from ...core.backend import to_host
from ...core.constants import EPS0
from ...core.renderer import PhysicsRenderer
from .renderer_fdfd3d import FDFDRenderer3D

_COMPONENT_INDEX = {'x': 0, 'y': 1, 'z': 2}
_PERMITTIVITY_OF = ('eps_xx', 'eps_yy', 'eps_r')   # array of the x / y / z component (eps_r when unallocated)


class FEMRenderer3D(PhysicsRenderer):
    """Point samples of a FEMResonancePhysics3D's selected resonance (zeros before a solve) and its eps_r."""

    name = 'em'
    field_infos = {name: replace(info, label=re.sub(r' \(\+[^)]*\)', '', info.label), staggered='')
                   for name, info in FDFDRenderer3D.field_infos.items()}

    def __init__(self, physics):
        self.em = physics
        self.domain = physics.domain

    def device_field(self, component):
        """float32 run-backend (n_elements,) values at the element centroids (not a window image)."""
        values = self.sample(component, self.domain.host_centroids(clamp_to_physical=False))
        return self.domain.xp.asarray(values, dtype=np.float32)

    def sample(self, component, points) -> np.ndarray:
        """Host float64 (p,) values of `component` at points (p, 3) [m]; NaN outside the mesh."""
        self.field_info(component)
        elements, reference = self.domain.locate(points)   # remembered per point set by the domain
        inside = elements >= 0
        element = np.maximum(elements, 0)
        if component == 'eps_r':
            return np.where(inside, self._host('eps_r')[element], np.nan)
        if self.em.selected_resonance is None:
            return np.where(inside, 0.0, np.nan)
        fields = self.em.fem.fields_located(elements, reference)
        if component in ('e_abs', 'e_intensity', 'h_abs'):
            intensity = np.sum(np.abs(fields['h' if component == 'h_abs' else 'e']) ** 2, axis=1)
            return intensity if component == 'e_intensity' else np.sqrt(intensity)
        if component == 'w_e':
            allocated = self.domain.materials.property_names()
            eps = np.stack([self._host(name if name in allocated else 'eps_r')[element] for name in _PERMITTIVITY_OF],
                           axis=1)
            return np.where(inside, 0.25 * EPS0 * np.sum(eps * np.abs(fields['e']) ** 2, axis=1), np.nan)
        name, _, part = component.rpartition('_')
        values = fields[name[0]][:, _COMPONENT_INDEX[name[1]]]
        return {'re': values.real, 'im': values.imag, 'abs': np.abs(values)}[part].astype(np.float64)

    def _host(self, name):
        return np.asarray(to_host(getattr(self.domain.materials, name)), dtype=np.float64)
