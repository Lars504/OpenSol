"""Renderer of NonlinearHarmonicPhysics: the selected channel's FDFD renderer fields (the static union of both
polarisations; the other polarisation's phasor fields read zeros), |P_NL| of that channel, the chi(2) poling map, the
steady-state free-carrier density and the physics-wide eps_sensitivity (cascade adjoint) and design_density."""
from dataclasses import replace

from ...core.constants import DTYPE
from ...core.mirror import plane_signs_text
from ...core.renderer import FieldInfo, PhysicsRenderer
from ..em.renderer import fdfd_union_field_infos
from .media import poling_map

# FDFD renderer fields of features the nonlinear physics does not offer (phase change; the cascade adjoint has no
# dJ/d eps_xy)
_OMITTED_FIELDS = frozenset({'crystallinity', 'eps_xy_sensitivity'})
_PRIMARY_FIELD = 'e_abs'   # polarisation-agnostic: meaningful in the snapshot of every channel
_MATERIAL_FIELDS = ('eps_r', 'eps_xx', 'eps_yy', 'sigma_e')
_NONLINEAR_FIELD_INFOS = {
    'nonlinear_polarization_abs': FieldInfo('nonlinear_polarization_abs', '|P_NL| (shown channel)', unit='C/m^2',
                                            cmap='inferno'),
    # a material map: the unfolded image half holds the reflected crystal with the same domain signs
    'poling': FieldInfo('poling', 'Poling sign (chi2 media)', unit='1', signed=True, cmap='RdBu_r', parity='even'),
    'carrier_density': FieldInfo('carrier_density', 'Free-carrier density (chi3 media)', unit='1/m^3',
                                 cmap='viridis'),
}


def _field_infos():
    """Union of the ez and hz FDFD renderer fields (minus _OMITTED_FIELDS, primary e_abs) + the nonlinear ones."""
    infos = fdfd_union_field_infos(_OMITTED_FIELDS, _PRIMARY_FIELD)
    infos.update(_NONLINEAR_FIELD_INFOS)
    return infos


NONLINEAR_FIELD_INFOS = _field_infos()   # = PhysicsType.output_fields of em_fdfd_nonlinear (same order)


class NonlinearRenderer(PhysicsRenderer):
    """Window-shaped float32 fields of the selected channel (NonlinearHarmonicPhysics.select_channel)."""

    name = 'nl'
    field_infos = NONLINEAR_FIELD_INFOS

    def __init__(self, nonlinear_physics):
        self.nl = nonlinear_physics
        self.domain = nonlinear_physics.domain
        self._channel_renderers = [physics.renderer() for physics in nonlinear_physics.channel_physics]

    def field_info(self, component):
        """The type's FieldInfo, carrying the selected channel's own mirror parity (plane_signs) where it differs from
        the physics' faces (an even generated channel under an antisymmetric face)."""
        info = super().field_info(component)
        config = self.nl.boundary_config
        signs = self.nl.plane_signs(self.nl.selected_channel)
        if any(sign != (-1 if getattr(config, face).value == 'antisymmetric' else 1) for face, sign in signs.items()):
            return replace(info, plane_signs=plane_signs_text(signs))
        return info

    def device_field(self, component: str):
        self.field_info(component)
        window = self.domain.physical_window()
        index = self.nl.channel_index(self.nl.selected_channel)
        if component == 'nonlinear_polarization_abs':
            return self.nl.polarization_magnitude(index)[window]
        if component == 'poling':
            return poling_map(self.nl.media, self.domain.materials.sigma_e.shape, self.domain.xp)[window]
        if component == 'carrier_density':
            return self.nl.carrier_density_map()[window]
        if component == 'eps_sensitivity':   # dJ/d eps of every channel's objectives (one cascade adjoint)
            return self.nl.eps_sensitivity().density[window]
        if component == 'design_density':
            return self.nl.design_density()[window]
        renderer = self._channel_renderers[index]
        if component in renderer.field_infos:
            return renderer.device_field(component)
        if component in _MATERIAL_FIELDS:
            return getattr(self.domain.materials, component)[window]
        shape = tuple(cells.stop - cells.start for cells in window)
        return self.domain.xp.zeros(shape, dtype=DTYPE)   # phasor field of the other polarisation: zero in this channel

    def time_label(self):
        index = self.nl.channel_index(self.nl.selected_channel)
        label = self._channel_renderers[index].time_label()
        return None if label is None else f'channel {self.nl.selected_channel}: {label}'
