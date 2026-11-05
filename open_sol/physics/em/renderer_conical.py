"""Renderer of the conical FDFD (solvers/fdfd_conical.ConicalPhysics): window-shaped float32 centre fields of the shown
driven solve or resonance (Re and modulus per component, |E|, |E|^2, |H|, S_z), absorbed power and materials."""
from dataclasses import replace

from .renderer import _MODULUS_FIELD_INFOS, CentreFieldRenderer, EMRenderer, ModeRenderer, centre_component_infos

_COMPONENT_TEXTS = (('ex', 'E_x', 'V/m'), ('ey', 'E_y', 'V/m'), ('ez', 'E_z', 'V/m'),
                    ('hx', 'H_x', 'A/m'), ('hy', 'H_y', 'A/m'), ('hz', 'H_z', 'A/m'))


class ConicalRenderer(CentreFieldRenderer):
    """Window-shaped float32 fields of a ConicalPhysics at the cell centres (ConicalPhysics.fields) and its materials."""

    field_infos = {
        **centre_component_infos(_COMPONENT_TEXTS),
        'e_abs': _MODULUS_FIELD_INFOS['e_abs'],
        'e_intensity': replace(_MODULUS_FIELD_INFOS['e_intensity'], primary=True),
        'h_abs': _MODULUS_FIELD_INFOS['h_abs'],
        'sz': ModeRenderer.field_infos['sz'],
        'absorbed_power': EMRenderer.field_infos['absorbed_power'],
        'eps_r': ModeRenderer.field_infos['eps_r'],
        **{name: ModeRenderer.field_infos[name] for name in ('eps_xx', 'eps_yy', 'sigma_e')},
    }
