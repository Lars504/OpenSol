"""Renderer of em_fdfd_3d: Re / Im / modulus of the six complex Yee components of the selected resonance (raw
samples), the centre moduli e_abs (primary), e_intensity, h_abs, the electric energy density and eps_r; device_plane
cuts one x / y / z plane without full-grid temporaries."""
from dataclasses import replace

import numpy as np

from ...core.constants import DTYPE, EPS0
from ...core.renderer import FieldInfo
from ..em.renderer import _MODULUS_FIELD_INFOS
from ..em.yee_grid import lower_face, lower_face_partner
from .mode_ports3d import axis_permittivity
from .renderer3d import CENTRE_AXES, RAW_LABELS, YeeRenderer3D, raw_parity

FDFD_3D_COMPONENTS = ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')


def _component_infos():
    """Re / Im / modulus FieldInfos of each raw component (E polar, H axial: core.mirror; staggered xy axes)."""
    infos = {}
    for name, label in RAW_LABELS.items():
        unit = 'V/m' if name[0] == 'e' else 'A/m'
        parity, staggered = raw_parity(name)
        for part, text, signed, cmap, part_parity in (('re', 'Re', True, 'RdBu_r', parity),
                                                      ('im', 'Im', True, 'RdBu_r', parity),
                                                      ('abs', '|.|', False, 'inferno', 'even')):
            infos[f'{name}_{part}'] = FieldInfo(f'{name}_{part}', f'{text} {label}', unit=unit, signed=signed,
                                                cmap=cmap, parity=part_parity, staggered=staggered)
    return infos


class FDFDRenderer3D(YeeRenderer3D):
    """Window-shaped float32 fields of an FDFDPhysics3D's selected resonance; device_plane(component, axis, k) = one
    window plane ('z' -> (wx, wy), 'y' -> (wx, wz), 'x' -> (wy, wz))."""

    field_infos = {
        **_component_infos(),
        'e_abs': replace(_MODULUS_FIELD_INFOS['e_abs'], primary=True),
        'e_intensity': _MODULUS_FIELD_INFOS['e_intensity'],
        'h_abs': _MODULUS_FIELD_INFOS['h_abs'],
        'w_e': FieldInfo('w_e', 'Electric energy density (time average)', unit='J/m^3', cmap='magma'),
        'eps_r': FieldInfo('eps_r', 'Relative permittivity (eps_zz)', unit='1', cmap='YlOrBr'),
    }

    def __init__(self, physics):
        super().__init__(physics)
        widths = physics.widths
        angles = physics.boundary_config.bloch_phase_angles(tuple(float(np.sum(axis)) for axis in widths))
        self._wrap_factors = tuple(complex(np.exp(1j * angle)) for angle in angles)   # the face one period down

    def _field(self, component, cut):
        self.field_info(component)
        xp = self.xp
        if component == 'eps_r':
            return self.domain.materials.eps_r[self._window(cut)]
        if component in ('e_abs', 'e_intensity', 'h_abs'):
            names = ('hx', 'hy', 'hz') if component == 'h_abs' else ('ex', 'ey', 'ez')
            values = sum(xp.abs(self._centre(name, cut)) ** 2 for name in names)
            return (values if component == 'e_intensity' else xp.sqrt(values)).astype(DTYPE)
        if component == 'w_e':
            window = self._window(cut)
            values = sum(axis_permittivity(self.domain.materials, axis)[window] * xp.abs(self._centre(name, cut)) ** 2
                         for axis, name in enumerate(('ex', 'ey', 'ez')))
            return (0.25 * EPS0 * values).astype(DTYPE)
        name, _, part = component.rpartition('_')
        values = self.em.fields[name][self._window(cut)]
        values = {'re': lambda: values.real, 'im': lambda: values.imag, 'abs': lambda: xp.abs(values)}[part]()
        return values.astype(DTYPE)

    def _centre(self, component, cut):
        """complex64 component averaged to the window cell centres (yee_grid.lower_face_partner: cell 0 of a
        non-periodic axis takes its own face, of a Bloch axis the wrapped face times e^{+j k L})."""
        field = self.em.fields[component]
        periodic = self.em.periodic
        axes = list(CENTRE_AXES[component])
        if cut is None:
            values, kept_axes = field, [0, 1, 2]
        else:
            axis, index = cut
            absolute = self.domain.physical_window()[axis].start + index
            key = [slice(None)] * 3
            key[axis] = absolute
            values = field[tuple(key)]
            if axis in axes:
                key[axis] = lower_face(absolute, periodic[axis])
                lower = field[tuple(key)]
                if absolute == 0 and periodic[axis]:
                    lower = lower * self._wrap_factors[axis]
                values = 0.5 * (values + lower)
                axes.remove(axis)
            kept_axes = [other for other in range(3) if other != axis]
        for axis in axes:
            values = 0.5 * (values + lower_face_partner(values, kept_axes.index(axis), periodic[axis], self.xp,
                                                        self._wrap_factors[axis]))
        window = self.domain.physical_window()
        return values[tuple(window[axis] for axis in kept_axes)]
