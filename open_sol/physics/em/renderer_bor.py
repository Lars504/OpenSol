"""Renderer of the body-of-revolution FDFD (solvers/fdfd_bor.BORPhysics): window-shaped float32 centre fields of the
shown driven solve or resonance (Re and modulus per component, |E|, |E|^2, |H|), absorbed power and materials."""
from dataclasses import replace

from ...core.renderer import FieldInfo
from .renderer import _MODULUS_FIELD_INFOS, CentreFieldRenderer, EMRenderer, centre_component_infos

_COMPONENT_TEXTS = (('er', 'E_r', 'V/m'), ('ephi', 'E_phi', 'V/m'), ('ez', 'E_z', 'V/m'),
                    ('hr', 'H_r', 'A/m'), ('hphi', 'H_phi', 'A/m'), ('hz', 'H_z', 'A/m'))


class BORRenderer(CentreFieldRenderer):
    """Window-shaped float32 fields of a BORPhysics at the cell centres (BORPhysics.fields) and its materials."""

    field_infos = {
        **centre_component_infos(_COMPONENT_TEXTS, parity=False),
        'e_abs': _MODULUS_FIELD_INFOS['e_abs'],
        'e_intensity': replace(_MODULUS_FIELD_INFOS['e_intensity'], primary=True),
        'h_abs': _MODULUS_FIELD_INFOS['h_abs'],
        **{name: EMRenderer.field_infos[name] for name in ('absorbed_power', 'sigma_e')},
        # permittivity arrays of the drawing-plane names: eps_r = out of plane (phi), eps_xx = r, eps_yy = z
        'eps_r': FieldInfo('eps_r', 'Relative permittivity phi phi', unit='1', cmap='YlOrBr'),
        'eps_xx': FieldInfo('eps_xx', 'Relative permittivity rr', unit='1', cmap='YlOrBr'),
        'eps_yy': FieldInfo('eps_yy', 'Relative permittivity zz (axial)', unit='1', cmap='YlOrBr'),
    }
