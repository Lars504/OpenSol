"""Renderer of em_fdtd_3d: raw Yee components, the shared EM moduli (e_abs, e_intensity, h_abs at the cell centres),
eps_r, absorbed_power and the harmonic FDTD phasors; device_plane cuts one x / y / z plane without materialising 3D
derived fields."""
import numpy as np

from ...core.constants import DTYPE
from ...core.renderer import FieldInfo, PhysicsRenderer
from ..em.renderer import _MODULUS_FIELD_INFOS
from .harmonic3d import COMPONENTS

_HARMONIC_STUDY = 'frequency_domain'
_AXES = ('x', 'y', 'z')
RAW_LABELS = {'ex': 'E_x (+x face)', 'ey': 'E_y (+y face)', 'ez': 'E_z (+z face)',
              'hx': 'H_x (+y +z edge)', 'hy': 'H_y (+x +z edge)', 'hz': 'H_z (+x +y edge)'}
# centre averaging axes of each raw component (E: along its own axis; H: along the two axes of its edge)
CENTRE_AXES = {'ex': (0,), 'ey': (1,), 'ez': (2,), 'hx': (1, 2), 'hy': (0, 2), 'hz': (0, 1)}


def _harmonic_infos():
    infos = {'e_abs_harmonic': FieldInfo('e_abs_harmonic', '|E| (phasor)', unit='V/m', cmap='inferno',
                                         requires_study=_HARMONIC_STUDY),
             'e_intensity_harmonic': FieldInfo('e_intensity_harmonic', '|E|^2 (phasor)', unit='V^2/m^2',
                                               cmap='inferno', requires_study=_HARMONIC_STUDY),
             'absorbed_power_harmonic': FieldInfo('absorbed_power_harmonic', 'Absorbed power density (time average)',
                                                  unit='W/m^3', cmap='magma', requires_study=_HARMONIC_STUDY)}
    for component in COMPONENTS:
        label = f'E_{component[1]}'
        parity = f'polar_{component[1]}'
        for part, text, unit, signed, cmap, part_parity in (
                ('re', f'Re {label}', 'V/m', True, 'RdBu_r', parity),
                ('im', f'Im {label}', 'V/m', True, 'RdBu_r', parity),
                ('abs', f'|{label}|', 'V/m', False, 'inferno', 'even'),
                ('phase', f'arg {label}', 'rad', True, 'twilight', f'phase_{parity}')):
            name = f'{component}_{part}'
            infos[name] = FieldInfo(name, f'{text} (phasor, centre)', unit=unit, signed=signed, cmap=cmap,
                                    requires_study=_HARMONIC_STUDY, parity=part_parity)
    return infos


def raw_parity(name):
    """(parity, staggered xy axes) of a raw Yee component (core.mirror): E polar, H axial; the hi-face axes of its
    sample in the xy cut (CENTRE_AXES without z)."""
    staggered = ''.join('xyz'[axis] for axis in CENTRE_AXES[name] if axis < 2)
    return f"{'polar' if name[0] == 'e' else 'axial'}_{name[1]}", staggered


class YeeRenderer3D(PhysicsRenderer):
    """Constructor, device_field / device_plane and window cut shared by the 3D Yee renderers (em_fdtd_3d,
    em_fdfd_3d); the class gives _field(component, cut)."""

    name = 'em'

    def __init__(self, physics):
        self.em = physics
        self.domain = physics.domain
        self.xp = physics.domain.xp

    def device_field(self, component):
        return self._field(component, None)

    def device_plane(self, component, axis, index_window):
        """Window plane normal to axis 'x' | 'y' | 'z' at window cell index_window (the axis dropped)."""
        if axis not in _AXES:
            raise ValueError(f"cut plane axis must be one of {_AXES}, got {axis!r}")
        return self._field(component, (_AXES.index(axis), int(index_window)))

    def _window(self, cut):
        """Window slices; with a cut, the cut axis an integer index (it drops from the indexed array)."""
        window = list(self.domain.physical_window())
        if cut is not None:
            axis, index = cut
            window[axis] = window[axis].start + index
        return tuple(window)


class EMRenderer3D(YeeRenderer3D):
    """Window-shaped (wx, wy, wz) float32 fields of an EMPhysics3D; device_plane(component, axis, k) = one window
    plane ('z' -> (wx, wy), 'y' -> (wx, wz), 'x' -> (wy, wz))."""

    field_infos = {
        **{name: FieldInfo(name, label, unit='V/m' if name[0] == 'e' else 'A/m', signed=True, cmap='RdBu_r',
                           primary=name == 'ez', parity=raw_parity(name)[0], staggered=raw_parity(name)[1])
           for name, label in RAW_LABELS.items()},
        **_MODULUS_FIELD_INFOS,
        'eps_r': FieldInfo('eps_r', 'Relative permittivity (eps_zz)', unit='1', cmap='YlOrBr'),   # the E_z array
        'absorbed_power': FieldInfo('absorbed_power', 'Absorbed power density', unit='W/m^3', cmap='magma'),
        **_harmonic_infos(),
    }
    harmonic_primary = 'e_abs_harmonic'

    def time_label(self):
        return f't = {float(self.em.engine.time) * 1e15:.1f} fs'

    def _field(self, component, cut):
        """Window field, or the window plane cut = (axis 0 | 1 | 2, window index) when given."""
        info = self.field_info(component)
        if info.requires_study:
            return self._harmonic_field(component, cut)
        window = self._window(cut)
        if component in RAW_LABELS:
            values = getattr(self.em.fields, component)[window]
            scale = self.em.engine.field_scale(component)
            if values.dtype != DTYPE or scale != 1.0:   # half_storage: float16 in stored units
                values = values.astype(DTYPE) * DTYPE(scale)
        elif component == 'eps_r':
            values = self.domain.materials.eps_r[window]
        elif component == 'absorbed_power':
            values = self._absorbed_power(window)
        elif component == 'h_abs':
            values = self.xp.sqrt(sum(self._centre(name, cut) ** 2 for name in ('hx', 'hy', 'hz')))
        else:
            values = sum(self._centre(name, cut) ** 2 for name in ('ex', 'ey', 'ez'))
            if component == 'e_abs':
                values = self.xp.sqrt(values)
        return values.astype(DTYPE, copy=False)

    def _absorbed_power(self, window):
        """Mean absorbed power density [W/m^3] of the current window, else the last completed one, else zeros."""
        engine = self.em.engine
        engine.deposit_pole_work()
        if engine.q_accum_time > 0.0:
            return engine.q_accum[window] * DTYPE(1.0 / float(engine.q_accum_time))
        if engine.last_q_accum is not None:
            return engine.last_q_accum[window] * DTYPE(1.0 / float(engine.last_q_accum_time))
        return self.xp.zeros(engine.material_id[window].shape, dtype=DTYPE)

    def _centre(self, component, cut):
        """float32 raw component averaged to the cell centres (cell 0 of a non-periodic axis takes its own face): one
        slice of the window plus the lo row per centred axis (a take only for a window starting at grid row 0), then one
        pair sum per centred axis."""
        xp = self.xp
        field = getattr(self.em.fields, component)
        window = list(self.domain.physical_window())
        if cut is not None:
            axis, index = cut
            window[axis] = slice(window[axis].start + index, window[axis].start + index + 1)
        periodic = self.em.engine.periodic
        centre_axes = CENTRE_AXES[component]
        box, takes = [], []
        for axis, cells in enumerate(window):
            if axis not in centre_axes:
                box.append(cells)
            elif cells.start > 0:
                box.append(slice(cells.start - 1, cells.stop))
            else:   # the lo face of cell 0: the wrapped last face (periodic) or its own face
                box.append(slice(None))
                lower = field.shape[axis] - 1 if periodic[axis] else 0
                takes.append((axis, np.concatenate(([lower], np.arange(cells.start, cells.stop)))))
        values = field[tuple(box)]
        for axis, rows in takes:
            values = xp.take(values, xp.asarray(rows), axis=axis)
        values = values.astype(DTYPE, copy=False)   # float16 storage summed in float32 (the sums below are new arrays)
        for axis in centre_axes:
            upper, lower = [slice(None)] * 3, [slice(None)] * 3
            upper[axis], lower[axis] = slice(1, None), slice(None, -1)
            values = values[tuple(upper)] + values[tuple(lower)]
        values *= DTYPE(self.em.engine.field_scale(component) / (2 ** len(centre_axes)))
        return values if cut is None else xp.squeeze(values, axis=cut[0])

    def _harmonic_field(self, component, cut):
        accumulator = self.em.harmonic
        if accumulator is None:
            raise ValueError(f"field '{component}' exists only in a frequency_domain study (harmonic FDTD)")
        if component == 'e_intensity_harmonic':
            return accumulator.e_intensity(cut)
        if component == 'e_abs_harmonic':
            return self.xp.sqrt(accumulator.e_intensity(cut))
        if component == 'absorbed_power_harmonic':
            return accumulator.absorbed_power(self.em, cut)
        name, _, part = component.rpartition('_')
        phasor = accumulator.phasor(name, cut)
        parts = {'re': lambda: phasor.real, 'im': lambda: phasor.imag, 'abs': lambda: self.xp.abs(phasor),
                 'phase': lambda: self.xp.angle(phasor)}
        return parts[part]().astype(DTYPE)
