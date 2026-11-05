"""Renderer of EMEPhysics3D: the reconstructed E phasor of the last solve at the cell centres (complex64 3D arrays),
its moduli, q_abs, the materials and the adjoint eps_sensitivity; device_plane cuts one x / y / z plane without
materialising 3D derived fields."""
import math

from ...core.constants import C0, DTYPE
from ...core.renderer import FieldInfo, PhysicsRenderer
from ...core.wavelengths import wavelength_label
from ..em.renderer import _MODULUS_FIELD_INFOS

_AXES = ('x', 'y', 'z')
EME_3D_COMPONENTS = ('ex', 'ey', 'ez')
EME_3D_MATERIAL_FIELDS = {'eps_r': 'Relative permittivity (eps_zz)', 'eps_xx': 'Relative permittivity (eps_xx)',
                          'eps_yy': 'Relative permittivity (eps_yy)'}
_SENSITIVITY_FIELD = 'eps_sensitivity'


def _field_infos():
    """e_abs (primary), e_intensity, Re / Im of E_x, E_y, E_z (cell centres), absorbed_power, the permittivity arrays."""
    infos = {'e_abs': FieldInfo('e_abs', '|E|', unit='V/m', cmap='inferno', primary=True),
             'e_intensity': _MODULUS_FIELD_INFOS['e_intensity']}
    for component in EME_3D_COMPONENTS:
        for part, text in (('re', 'Re'), ('im', 'Im')):
            name = f'{component}_{part}'
            infos[name] = FieldInfo(name, f'{text} E_{component[1]} (phasor, centre)', unit='V/m', signed=True,
                                    cmap='RdBu_r', parity=f'polar_{component[1]}')
    infos['absorbed_power'] = FieldInfo('absorbed_power', 'Absorbed power density', unit='W/m^3', cmap='magma')
    for name, label in EME_3D_MATERIAL_FIELDS.items():
        infos[name] = FieldInfo(name, label, unit='1', cmap='YlOrBr')
    # dJ/d eps (eps_xx + eps_yy + eps_r terms) per cell volume of the adjoint_objective features (zeros without one)
    infos[_SENSITIVITY_FIELD] = FieldInfo(_SENSITIVITY_FIELD, 'Objective sensitivity dJ/d eps (isotropic)',
                                          unit='1/m^3', signed=True, cmap='RdBu_r', parity='even')
    return infos


EME_3D_FIELD_INFOS = _field_infos()   # = PhysicsType.output_fields of em_eme_3d (same order)


class EMERenderer3D(PhysicsRenderer):
    """Window-shaped (wx, wy, wz) float32 fields of the last EMEPhysics3D solve (zeros before it); device_plane(component,
    axis, k) = one window plane ('z' -> (wx, wy), 'y' -> (wx, wz), 'x' -> (wy, wz))."""

    name = 'eme'
    field_infos = EME_3D_FIELD_INFOS

    def __init__(self, eme_physics):
        self.eme = eme_physics
        self.domain = eme_physics.domain

    def device_field(self, component):
        return self._field(component, None)

    def device_plane(self, component, axis, index_window):
        """Window plane normal to axis 'x' | 'y' | 'z' at window cell index_window (the axis dropped)."""
        if axis not in _AXES:
            raise ValueError(f"cut plane axis must be one of {_AXES}, got {axis!r}")
        return self._field(component, (_AXES.index(axis), int(index_window)))

    def time_label(self):
        omega = self.eme.solved_omega
        return None if omega is None else wavelength_label(2.0 * math.pi * C0 / omega)

    def _field(self, component, cut):
        self.field_info(component)
        window = list(self.domain.physical_window())
        if cut is not None:
            axis, index = cut
            window[axis] = window[axis].start + index
        window = tuple(window)
        if component in EME_3D_MATERIAL_FIELDS:
            return getattr(self.domain.materials, component)[window]
        if component == 'absorbed_power':
            return self.eme.q_abs[window]
        if component == _SENSITIVITY_FIELD:   # cascade adjoint of the objective features (adjoint.py)
            return self.eme.eps_sensitivity().density[window]
        fields = self.eme.field_arrays
        if component in ('e_abs', 'e_intensity'):
            total = None
            for name in EME_3D_COMPONENTS:
                values = fields[name][window]
                squared = values.real * values.real + values.imag * values.imag
                total = squared if total is None else total + squared
            return (self.domain.xp.sqrt(total) if component == 'e_abs' else total).astype(DTYPE, copy=False)
        name, part = component.split('_')
        values = fields[name][window]
        return (values.real if part == 're' else values.imag).astype(DTYPE)
