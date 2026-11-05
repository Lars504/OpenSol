"""Renderer of EMEPhysics: the reconstructed field of the last solve over the window (the static union of the ez and
hz FDFD renderer fields: the other polarisation's phasor quartet reads zeros), its moduli, q_abs, the materials, the
adjoint eps_sensitivity and the design_density."""
import math
from dataclasses import replace

from ...core.constants import C0, DTYPE
from ...core.renderer import PhysicsRenderer
from ...core.wavelengths import wavelength_label
from ..em.renderer import fdfd_union_field_infos

# FDFD renderer fields EME does not have (phase change, the hz FDFD dJ/d eps_xy, the hz centred E_x / E_y parts)
_OMITTED_FIELDS = frozenset({'crystallinity', 'eps_xy_sensitivity', 'ex_re', 'ex_im', 'ey_re', 'ey_im'})
_PRIMARY_FIELD = 'e_abs'   # polarisation-agnostic, so one PhysicsType serves both polarisation settings
EME_MATERIAL_FIELDS = ('eps_r', 'eps_xx', 'eps_yy', 'sigma_e')
_SENSITIVITY_FIELD = 'eps_sensitivity'   # EME J is a power fraction: dJ/d eps per cell area in 1/m^2


def _field_infos():
    """Union of the ez and hz FDFD renderer fields minus _OMITTED_FIELDS, primary e_abs (eps_sensitivity in 1/m^2)."""
    infos = fdfd_union_field_infos(_OMITTED_FIELDS, _PRIMARY_FIELD)
    infos[_SENSITIVITY_FIELD] = replace(infos[_SENSITIVITY_FIELD], unit='1/m^2')
    return infos


EME_FIELD_INFOS = _field_infos()   # = PhysicsType.output_fields of em_eme (same order)


class EMERenderer(PhysicsRenderer):
    """Window-shaped float32 fields of the last EMEPhysics solve (zeros before it)."""

    name = 'eme'
    field_infos = EME_FIELD_INFOS

    def __init__(self, eme_physics):
        self.eme = eme_physics
        self.domain = eme_physics.domain

    def device_field(self, component: str):
        self.field_info(component)
        window = self.domain.physical_window()
        if component in EME_MATERIAL_FIELDS:
            return getattr(self.domain.materials, component)[window]
        if component == 'absorbed_power':
            return self.eme.q_abs[window]
        if component == _SENSITIVITY_FIELD:   # cascade adjoint of the objective features (adjoint.py)
            return self.eme.eps_sensitivity().density[window]
        if component == 'design_density':
            return self.eme.design_density()[window]
        fields = self.eme.field_arrays
        xp = self.domain.xp
        if component == 'e_abs':
            return xp.sqrt(fields['e_intensity'][window])
        if component in ('e_intensity', 'h_abs'):
            return fields[component][window]
        base, part = component.rsplit('_', 1)
        if base != self.eme.polarization:
            return xp.zeros(tuple(cells.stop - cells.start for cells in window), dtype=DTYPE)
        real, imaginary = fields['re'][window], fields['im'][window]
        if part == 're':
            return real
        if part == 'im':
            return imaginary
        if part == 'abs':
            return xp.sqrt(real * real + imaginary * imaginary)
        return xp.arctan2(imaginary, real)

    def time_label(self):
        omega = self.eme.solved_omega
        return None if omega is None else wavelength_label(2.0 * math.pi * C0 / omega)
