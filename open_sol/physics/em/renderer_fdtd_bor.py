"""Renderer of the body-of-revolution FDTD (solvers/fdtd_bor.BORFDTDPhysics): window-shaped float32 instantaneous centre
fields of the standing wave, their moduli, materials, and the frequency_domain phasors (Re / modulus per component, |E|)
of the harmonic accumulator."""
from ...core.backend import array_module_of
from ...core.constants import DTYPE
from ...core.renderer import FieldInfo
from .body_of_revolution import yee_centres
from .renderer import _MODULUS_FIELD_INFOS, _EMRendererBase
from .renderer_bor import BORRenderer

_HARMONIC_STUDY = 'frequency_domain'   # FieldInfo.requires_study of the phasor fields
# (name, label, unit, azimuthal profile of the standing wave: fdtd_bor module header)
_COMPONENTS = (('er', 'E_r', 'V/m', 'cos'), ('ephi', 'E_phi', 'V/m', 'sin'), ('ez', 'E_z', 'V/m', 'cos'),
               ('hr', 'H_r', 'A/m', 'sin'), ('hphi', 'H_phi', 'A/m', 'cos'), ('hz', 'H_z', 'A/m', 'sin'))
_MATERIAL_NAMES = ('eps_r', 'eps_xx', 'eps_yy', 'sigma_e')


def _field_infos():
    """Instantaneous components and moduli, materials, then the frequency_domain phasor parts."""
    infos = {name: FieldInfo(name, f'{label} ({profile} m phi wave)', unit=unit, signed=True, cmap='RdBu_r')
             for name, label, unit, profile in _COMPONENTS}
    infos.update({'e_abs': _MODULUS_FIELD_INFOS['e_abs'],
                  'e_intensity': FieldInfo('e_intensity', '|E|^2', unit='V^2/m^2', cmap='inferno', primary=True),
                  'h_abs': _MODULUS_FIELD_INFOS['h_abs']})
    infos.update({name: BORRenderer.field_infos[name] for name in _MATERIAL_NAMES})
    for name, label, unit, _ in _COMPONENTS:
        infos[f'{name}_re'] = FieldInfo(f'{name}_re', f'Re {label} (phasor)', unit=unit, signed=True, cmap='RdBu_r',
                                        requires_study=_HARMONIC_STUDY)
        infos[f'{name}_abs'] = FieldInfo(f'{name}_abs', f'|{label}| (phasor)', unit=unit, cmap='inferno',
                                         requires_study=_HARMONIC_STUDY)
    infos['e_abs_harmonic'] = FieldInfo('e_abs_harmonic', '|E| (phasor)', unit='V/m', cmap='inferno',
                                        requires_study=_HARMONIC_STUDY)
    return infos


class BORFDTDRenderer(_EMRendererBase):
    """Window-shaped float32 fields of a BORFDTDPhysics at the cell centres (H half a step earlier), its materials and
    the harmonic phasors of a frequency_domain study (em_fdfd_bor convention, at the selected study wavelength)."""

    field_infos = _field_infos()
    harmonic_primary = 'e_abs_harmonic'   # stored by default in a frequency_domain study

    def device_field(self, component: str):
        info = self.field_info(component)
        window = self.domain.physical_window()
        if info.requires_study:
            return self._harmonic_field(component)[window]
        if component in _MATERIAL_NAMES:
            return getattr(self.domain.materials, component)[window]
        centres = {name: values[window] for name, values in self._centre_samples().items()}
        if component in centres:
            return centres[component]
        xp = array_module_of(centres['er'])
        if component == 'h_abs':
            return xp.sqrt(centres['hr'] ** 2 + centres['hphi'] ** 2 + centres['hz'] ** 2)
        intensity = centres['er'] ** 2 + centres['ephi'] ** 2 + centres['ez'] ** 2
        return xp.sqrt(intensity) if component == 'e_abs' else intensity

    def _harmonic_field(self, component):
        """Full-grid float32 phasor part of the selected study wavelength; ValueError outside a frequency_domain
        study or before its pulse ended."""
        harmonic = self.em.harmonic
        if harmonic is None:
            raise ValueError(f"field '{component}' exists only in a frequency_domain study (harmonic BOR FDTD)")
        key = ('harmonic', harmonic, harmonic.index, harmonic.normalized)   # the object, not its id (no reuse)
        if self._cached_key != key:   # one yee_centres of the phasors per selected wavelength, not per component
            self._cached_key, self._cached = key, yee_centres(harmonic.yee())[0]
        centres = self._cached
        xp = array_module_of(centres['er'])
        if component == 'e_abs_harmonic':
            return xp.sqrt(sum(xp.abs(centres[name]) ** 2 for name in ('er', 'ephi', 'ez'))).astype(DTYPE)
        name, _, part = component.rpartition('_')
        values = centres[name]
        return (values.real if part == 're' else xp.abs(values)).astype(DTYPE)

    _cached_key = None   # (kind, ...) of the centre dict in _cached: a snapshot reads every component of one state
    _cached = None

    def _centre_samples(self):
        """em.centre_samples() once per engine step (several components of one snapshot share it)."""
        engine = self.em.engine
        key = ('instant', engine, None if engine is None else engine.n_step)   # the object: a rebuilt engine misses
        if self._cached_key != key:
            self._cached_key, self._cached = key, self.em.centre_samples()
        return self._cached

    def time_label(self):
        engine = self.em.engine
        return None if engine is None else f't = {engine.time * 1e15:.1f} fs'
