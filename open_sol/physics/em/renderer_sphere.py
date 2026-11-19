"""Renderer of the spherically symmetric FDFD (solvers/fdfd_sphere.SpherePhysics): window-shaped float32 radial
profiles of the shown solution (sphere-RMS amplitudes of E_r, E_t, H_r, H_t summed over every solved multipole term),
|E|, |E|^2, |H|, absorbed power and materials."""
from dataclasses import replace

from ...core.renderer import FieldInfo
from .renderer import _MODULUS_FIELD_INFOS, CentreFieldRenderer, EMRenderer

_RMS = 'rms over the sphere'


class SphereRenderer(CentreFieldRenderer):
    """Window-shaped (nx, 1) float32 cell-centre fields of a SpherePhysics (SpherePhysics.fields) and its materials."""

    material_names = ('eps_r', 'sigma_e')
    field_infos = {
        'er': FieldInfo('er', f'|E_r| ({_RMS})', unit='V/m', cmap='inferno'),
        'et': FieldInfo('et', f'|E_t| tangential ({_RMS})', unit='V/m', cmap='inferno'),
        'hr': FieldInfo('hr', f'|H_r| ({_RMS})', unit='A/m', cmap='viridis'),
        'ht': FieldInfo('ht', f'|H_t| tangential ({_RMS})', unit='A/m', cmap='viridis'),
        'e_abs': replace(_MODULUS_FIELD_INFOS['e_abs'], label=f'|E| ({_RMS})'),
        'e_intensity': replace(_MODULUS_FIELD_INFOS['e_intensity'], label='|E|^2 (mean over the sphere)',
                               primary=True),
        'h_abs': replace(_MODULUS_FIELD_INFOS['h_abs'], label=f'|H| ({_RMS})'),
        'absorbed_power': replace(EMRenderer.field_infos['absorbed_power'],
                                  label='Absorbed power density (mean over the sphere)'),
        'eps_r': EMRenderer.field_infos['eps_r'],
        'sigma_e': EMRenderer.field_infos['sigma_e'],
    }
