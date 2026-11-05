"""EM renderers of window-shaped fields: EMRenderer / EMRendererHz (2D FDTD), FDFDRendererEz / _Hz (FDFD phasors) and
ModeRenderer (ModePhysics modes)."""
import math
from dataclasses import replace

from ...core.constants import C0, DTYPE
from ...core.renderer import FieldInfo, PhysicsRenderer
from ...core.wavelengths import wavelength_label
from .fields import collocate_to_centres
from .harmonic import face_mean_at_centres

# Polarisation-agnostic moduli on every EM renderer (at the cell centres): FDTD = instantaneous, FDFD = phasor modulus.
_MODULUS_FIELD_INFOS = {
    'e_abs': FieldInfo('e_abs', '|E|', unit='V/m', cmap='inferno'),
    'e_intensity': FieldInfo('e_intensity', '|E|^2', unit='V^2/m^2', cmap='inferno'),
    'h_abs': FieldInfo('h_abs', '|H|', unit='A/m', cmap='viridis'),
}
_HARMONIC_STUDY = 'frequency_domain'   # FieldInfo.requires_study of the harmonic FDTD fields (harmonic_fdtd.py)
# mirror parity (core.mirror) of the out-of-plane component: E_z polar, H_z axial
OUT_OF_PLANE_PARITY = {'ez': 'polar_z', 'hz': 'axial_z'}


def _phasor_parts(out_of_plane, label, unit, label_suffix=''):
    """(part, text, unit, signed, cmap, parity) of the Re / Im / modulus / argument fields of a phasor component."""
    parity = OUT_OF_PLANE_PARITY[out_of_plane]
    return (('re', f'Re {label}{label_suffix}', unit, True, 'RdBu_r', parity),
            ('im', f'Im {label}{label_suffix}', unit, True, 'RdBu_r', parity),
            ('abs', f'|{label}|{label_suffix}', unit, False, 'inferno', 'even'),
            ('phase', f'arg {label}{label_suffix}', 'rad', True, 'twilight', f'phase_{parity}'))


def _harmonic_fdtd_field_infos(out_of_plane, label, unit):
    """Re / Im / modulus / argument of the normalised harmonic FDTD phasor of `out_of_plane`, |E| and |E|^2 of the
    E phasor (ez: |E_z|, hz: in-plane) and the absorbed power density, all frequency_domain only."""
    infos = {f'{out_of_plane}_{part}': FieldInfo(f'{out_of_plane}_{part}', text, unit=part_unit, signed=signed,
                                                 cmap=cmap, requires_study=_HARMONIC_STUDY, parity=parity)
             for part, text, part_unit, signed, cmap, parity in _phasor_parts(out_of_plane, label, unit,
                                                                               ' (phasor)')}
    infos['e_abs_harmonic'] = FieldInfo('e_abs_harmonic', '|E| (phasor)', unit='V/m', cmap='inferno',
                                        requires_study=_HARMONIC_STUDY)
    infos['e_intensity_harmonic'] = FieldInfo('e_intensity_harmonic', '|E|^2 (phasor)', unit='V^2/m^2',
                                              cmap='inferno', requires_study=_HARMONIC_STUDY)
    infos['absorbed_power_harmonic'] = FieldInfo('absorbed_power_harmonic', 'Absorbed power density (time average)',
                                                 unit='W/m^3', cmap='magma', requires_study=_HARMONIC_STUDY)
    return infos


def _window_shape(window):
    """(mx, my) cell counts of a (slice_x, slice_y) physical window."""
    return tuple(cells.stop - cells.start for cells in window)


class _EMRendererBase(PhysicsRenderer):
    """Shared constructor of the EM renderers (FDTD EMPhysics or FDFDPhysics)."""

    name = "em"

    def __init__(self, em_physics):
        self.em = em_physics
        self.domain = em_physics.domain

    def _crystallinity(self, window):
        """Crystallinity of the phase_change_medium features (phase_change.py; later ones win on overlaps), 0 elsewhere."""
        if not self.em.phase_change_media:
            return self.domain.xp.zeros(_window_shape(window), dtype=DTYPE)
        total = self.domain.xp.zeros(self.domain.materials.sigma_e.shape, dtype=DTYPE)
        for medium in self.em.phase_change_media:
            medium.paint_crystallinity(total)
        return total[window]


class _FDTDRendererBase(_EMRendererBase):
    """Absorbed power window, harmonic (frequency_domain) fields and time label shared by both FDTD polarisations."""

    harmonic_primary = ''   # field a frequency_domain study stores by default (model.study._stored_field_keys)
    material_names = ()     # MaterialGrid arrays served as fields

    def device_field(self, component: str):
        """Window-shaped float32 field `component` (a double stepper's float64 fields rounded on output; float32 ones
        returned as they are, views included)."""
        return self._fdtd_field(component).astype(DTYPE, copy=False)

    def _fdtd_field(self, component):
        """Harmonic (frequency_domain) fields, materials, C5 media fields and absorbed power; the polarisation's own
        fields from _polarization_field."""
        if self.field_info(component).requires_study:
            return self._harmonic_field(component)
        window = self.domain.physical_window()
        if component in self.material_names:
            return getattr(self.domain.materials, component)[window]
        if component == 'carrier_density':
            return self._carrier_density(window)
        if component == 'population_inversion':
            return self._population_inversion(window)
        if component == 'crystallinity':
            return self._crystallinity(window)
        if component == 'absorbed_power':
            return self._absorbed_power(window)
        return self._polarization_field(component, window)

    def _polarization_field(self, component, window):
        raise NotImplementedError

    def _harmonic_field(self, component):
        """Window-shaped float32 harmonic field of the selected study wavelength; ValueError outside a frequency_domain
        study or before its pulse ended."""
        accumulator = self.em.harmonic
        if accumulator is None:
            raise ValueError(f"field '{component}' exists only in a frequency_domain study (harmonic FDTD)")
        if component == 'absorbed_power_harmonic':
            return accumulator.absorbed_power(self.em)
        if component == 'e_abs_harmonic':
            return accumulator.e_abs()
        if component == 'e_intensity_harmonic':
            return accumulator.e_intensity()
        out_of_plane, _, part = component.rpartition('_')
        return _phasor_part(accumulator.phasor(out_of_plane), (slice(None), slice(None)), part, self.domain.xp)

    def _absorbed_power(self, window):
        """Mean absorbed power density [W/m^3] of the current window, else the last completed one, else zeros."""
        engine = self.em.engine
        if engine.q_accum_time > 0.0:
            energy_density, window_seconds = engine.q_accum, float(engine.q_accum_time)
        elif engine.last_q_accum is not None:
            energy_density, window_seconds = engine.last_q_accum, float(engine.last_q_accum_time)
        else:   # q_accum may be None until accumulation is enabled
            return self.domain.xp.zeros(_window_shape(window), dtype=DTYPE)
        return energy_density[window] * DTYPE(1.0 / window_seconds)

    def _carrier_density(self, window):
        """Free-carrier density [1/m^3] of the nonlinear_medium features (fdtd_nonlinear), zeros without carriers."""
        nonlinear = self.em.engine.nonlinear
        if nonlinear is None:
            return self.domain.xp.zeros(_window_shape(window), dtype=DTYPE)
        return nonlinear.carrier_density()[window]

    def _population_inversion(self, window):
        """N2 - N1 [1/m^3] summed over the gain_medium regions (fdtd_gain), zeros without one."""
        engine = self.em.engine
        if not engine.gain_media:
            return self.domain.xp.zeros(_window_shape(window), dtype=DTYPE)
        total = self.domain.xp.zeros((engine.nx, engine.ny), dtype=DTYPE)
        for gain in engine.gain_media:
            total += gain.inversion((engine.nx, engine.ny))
        return total[window]

    def time_label(self):
        return f"t = {float(self.em.engine.time) * 1e15:.1f} fs"


# C5 media: TPA-generated carriers (zeros without a carrier-generating nonlinear_medium) and the gain_medium inversion
_CARRIER_DENSITY_INFO = FieldInfo('carrier_density', 'Free-carrier density', unit='1/m^3', cmap='viridis')
_INVERSION_INFO = FieldInfo('population_inversion', 'Population inversion N2 - N1', unit='1/m^3', signed=True,
                            cmap='RdBu_r', parity='even')
# phase_change_medium crystallinity X (0 amorphous, 1 crystalline; 0 outside the media), FDTD and FDFD renderers
_CRYSTALLINITY_INFO = FieldInfo('crystallinity', 'Crystallinity', unit='1', cmap='viridis')


class EMRenderer(_FDTDRendererBase):
    """Window-shaped float32 fields (hx / hy raw on their hi-y / hi-x faces, as in state(); h_abs collocated to the
    cell centres), materials and absorbed power of an EMPhysics."""

    field_infos = {
        'ez': FieldInfo('ez', 'E_z', unit='V/m', signed=True, cmap='RdBu_r', primary=True, parity='polar_z'),
        'hx': FieldInfo('hx', 'H_x (+y face)', unit='A/m', signed=True, cmap='RdBu_r', parity='axial_x',
                        staggered='y'),
        'hy': FieldInfo('hy', 'H_y (+x face)', unit='A/m', signed=True, cmap='RdBu_r', parity='axial_y',
                        staggered='x'),
        **_MODULUS_FIELD_INFOS,
        'eps_r': FieldInfo('eps_r', 'Relative permittivity', unit='1', cmap='YlOrBr'),
        'sigma_e': FieldInfo('sigma_e', 'Electric conductivity', unit='S/m', cmap='cividis'),
        'absorbed_power': FieldInfo('absorbed_power', 'Absorbed power density', unit='W/m^3', cmap='magma'),
        'carrier_density': _CARRIER_DENSITY_INFO,
        'population_inversion': _INVERSION_INFO,
        'crystallinity': _CRYSTALLINITY_INFO,
        **_harmonic_fdtd_field_infos('ez', 'E_z', 'V/m'),
    }
    harmonic_primary = 'ez_abs'
    material_names = ('eps_r', 'sigma_e')

    def _polarization_field(self, component, window):
        fields = self.em.fields
        if component in ('ez', 'hx', 'hy'):
            return getattr(fields, component)[window]
        if component == 'e_abs':
            return self.domain.xp.abs(fields.ez[window])
        if component == 'e_intensity':
            return fields.ez[window] ** 2
        if component == 'h_abs':   # H_x on hi-y faces averaged along y, H_y on hi-x faces along x
            config = self.em.boundary_config
            hx_centres = collocate_to_centres(fields.hx, 1, config.periodic_y)[window]
            hy_centres = collocate_to_centres(fields.hy, 0, config.periodic_x)[window]
            return self.domain.xp.sqrt(hx_centres ** 2 + hy_centres ** 2)
        raise ValueError(f"EMRenderer has no field '{component}'")


class EMRendererHz(_FDTDRendererBase):
    """Window-shaped float32 fields (Ex / Ey collocated to the cell centres), materials and absorbed power of an
    H_z-polarisation EMPhysics."""

    field_infos = {
        'hz': FieldInfo('hz', 'H_z', unit='A/m', signed=True, cmap='RdBu_r', primary=True, parity='axial_z'),
        'ex': FieldInfo('ex', 'E_x', unit='V/m', signed=True, cmap='RdBu_r', parity='polar_x'),
        'ey': FieldInfo('ey', 'E_y', unit='V/m', signed=True, cmap='RdBu_r', parity='polar_y'),
        **_MODULUS_FIELD_INFOS,
        'eps_xx': FieldInfo('eps_xx', 'Relative permittivity xx', unit='1', cmap='YlOrBr'),
        'eps_yy': FieldInfo('eps_yy', 'Relative permittivity yy', unit='1', cmap='YlOrBr'),
        'sigma_e': FieldInfo('sigma_e', 'Electric conductivity', unit='S/m', cmap='cividis'),
        'absorbed_power': FieldInfo('absorbed_power', 'Absorbed power density', unit='W/m^3', cmap='magma'),
        'carrier_density': _CARRIER_DENSITY_INFO,
        'population_inversion': _INVERSION_INFO,
        'crystallinity': _CRYSTALLINITY_INFO,
        **_harmonic_fdtd_field_infos('hz', 'H_z', 'A/m'),
    }
    harmonic_primary = 'hz_abs'
    material_names = ('eps_xx', 'eps_yy', 'sigma_e')

    def _polarization_field(self, component, window):
        fields = self.em.fields
        if component == 'hz':
            return fields.hz[window]
        if component in ('ex', 'ey'):
            return self._centred_e(component)[window]
        if component in ('e_abs', 'e_intensity'):
            e_intensity = self._centred_e('ex')[window] ** 2 + self._centred_e('ey')[window] ** 2
            return self.domain.xp.sqrt(e_intensity) if component == 'e_abs' else e_intensity
        if component == 'h_abs':
            return self.domain.xp.abs(fields.hz[window])
        raise ValueError(f"EMRendererHz has no field '{component}'")

    def _centred_e(self, component):
        """Full-grid 'ex' (hi-y faces, averaged along y) or 'ey' (hi-x faces, along x) at the cell centres."""
        config = self.em.boundary_config
        if component == 'ex':
            return collocate_to_centres(self.em.fields.ex, 1, config.periodic_y)
        return collocate_to_centres(self.em.fields.ey, 0, config.periodic_x)


# ---- FDFD: phasor fields of the last solve, zeros before the first one ----------------------------------------

def _phasor_part(solution, window, part, xp):
    """float32 window array of 're' | 'im' | 'abs' | 'phase' [rad] of a complex128 phasor field (None = not solved)."""
    if solution is None:
        return xp.zeros(_window_shape(window), dtype=DTYPE)
    values = solution[window]
    if part == 're':
        return values.real.astype(DTYPE)
    if part == 'im':
        return values.imag.astype(DTYPE)
    if part == 'abs':
        return xp.abs(values).astype(DTYPE, copy=False)
    return xp.angle(values).astype(DTYPE, copy=False)


def _wavelength_label(solved_omega):
    """'lambda = 1550 nm' of the last solve's angular frequency [rad/s], None before the first solve."""
    if solved_omega is None:
        return None
    return wavelength_label(2.0 * math.pi * C0 / solved_omega)


def _phasor_field_infos(out_of_plane, label, unit):
    """Re / Im / modulus / argument FieldInfos of the out-of-plane phasor ('ez' / 'hz'); Re is the primary field."""
    return {f'{out_of_plane}_{part}': FieldInfo(f'{out_of_plane}_{part}', text, unit=part_unit, signed=signed,
                                                cmap=cmap, primary=part == 're', parity=parity)
            for part, text, part_unit, signed, cmap, parity in _phasor_parts(out_of_plane, label, unit)}


# dJ/d eps_r per area of the FDFD objective features (adjoint.EpsSensitivity.density; zeros without one)
_SENSITIVITY_FIELD_INFO = FieldInfo('eps_sensitivity', 'Objective sensitivity dJ/d eps_r', unit='W/m^3', signed=True,
                                    cmap='RdBu_r', parity='even')
# projected density rho_bar of the FDFD design regions (design_density.DesignRegion; 0 outside them)
_DESIGN_DENSITY_FIELD_INFO = FieldInfo('design_density', 'Design density', unit='1', cmap='YlOrBr')
# hz off-diagonal permittivity (material eps_xy + the eps_xy_update change; FDFDPhysics.eps_xy). Its true mirror image
# is -eps_xy on any plane, which no core.mirror parity expresses: 'even' (a model with eps_xy has no mirror symmetry).
_EPS_XY_FIELD_INFO = FieldInfo('eps_xy', 'Relative permittivity xy (off-diagonal)', unit='1', signed=True,
                               cmap='RdBu_r', parity='even')
# hz dJ/d eps_xy per area at fixed eps_xx / eps_yy (adjoint.EpsSensitivity.off_diagonal_density; zeros without an
# objective). Its mirror image is odd on every plane (as eps_xy), which no parity expresses: 'even', so an unfolded
# mirror model shows the image half with the wrong sign.
_EPS_XY_SENSITIVITY_FIELD_INFO = FieldInfo('eps_xy_sensitivity', 'Objective sensitivity dJ/d eps_xy', unit='W/m^3',
                                           signed=True, cmap='RdBu_r', parity='even')


# hz E_x / E_y phasor parts at the cell centres (complex faces averaged first, as the moduli): interface sampling (igor
# 1.8) projects them onto object normals
_IN_PLANE_E_PART_INFOS = {
    f'e{axis}_{part}': FieldInfo(f'e{axis}_{part}', f'{text} E_{axis} (phasor, cell centres)', unit='V/m', signed=True,
                                 cmap='RdBu_r', parity=f'polar_{axis}')
    for axis in ('x', 'y') for part, text in (('re', 'Re'), ('im', 'Im'))}


def _in_plane_centres(fdfd_physics, window):
    """(F_x, F_y) complex window arrays of the in-plane face phasors (ez H, hz E) averaged onto the cell centres;
    None before the first solve."""
    if fdfd_physics.solution is None:
        return None
    xp = fdfd_physics.domain.xp
    config = fdfd_physics.boundary_config
    wrap_x, wrap_y = fdfd_physics.lower_wrap_factors   # Bloch phase of the wrapped lo face (1 otherwise)
    x_component_faces, y_component_faces = fdfd_physics.face_fields()   # F_x on hi-y faces, F_y on hi-x faces
    return (face_mean_at_centres(x_component_faces, 1, config.periodic_y, xp, wrap_factor=wrap_y)[window],
            face_mean_at_centres(y_component_faces, 0, config.periodic_x, xp, wrap_factor=wrap_x)[window])


def _in_plane_intensity(fdfd_physics, window):
    """|F_x|^2 + |F_y|^2 of the in-plane face phasors (ez H, hz E) with each complex face phasor averaged onto the
    cell centres before the modulus (zeros before the first solve)."""
    xp = fdfd_physics.domain.xp
    centres = _in_plane_centres(fdfd_physics, window)
    if centres is None:
        return xp.zeros(_window_shape(window), dtype=DTYPE)
    return xp.abs(centres[0]) ** 2 + xp.abs(centres[1]) ** 2


class _FDFDRendererBase(_EMRendererBase):
    """Shared device_field of the FDFD renderers: the out-of-plane phasor gives its own modulus, the in-plane face
    phasors (collocated to the centres) the other one."""

    out_of_plane = ''        # 'ez' | 'hz'
    material_names = ()

    def device_field(self, component: str):
        self.field_info(component)
        window = self.domain.physical_window()
        xp = self.domain.xp
        if component in self.material_names:
            return getattr(self.domain.materials, component)[window]
        if component == 'absorbed_power':
            return self.em.state()['q_abs'][window]
        if component == 'eps_sensitivity':   # adjoint gradient of the objective features (one transposed solve)
            return self.em.eps_sensitivity().density[window]
        if component == 'design_density':
            return self.em.design_density()[window]
        if component == 'crystallinity':
            return self._crystallinity(window)
        if component == 'eps_xy':
            return self.em.eps_xy()[window]
        if component == 'eps_xy_sensitivity':   # from the same transposed solve as eps_sensitivity
            density = self.em.eps_sensitivity().off_diagonal_density
            return xp.zeros(_window_shape(window), dtype=DTYPE) if density is None else density[window]
        if component in _MODULUS_FIELD_INFOS:
            if component[0] == self.out_of_plane[0]:   # modulus of the out-of-plane phasor ('e' of ez, 'h' of hz)
                modulus = _phasor_part(self.em.solution, window, 'abs', xp)
                return modulus ** 2 if component == 'e_intensity' else modulus
            intensity = _in_plane_intensity(self.em, window)
            return (intensity if component == 'e_intensity' else xp.sqrt(intensity)).astype(DTYPE)
        if component in _IN_PLANE_E_PART_INFOS:   # hz only (field_info checked it)
            centres = _in_plane_centres(self.em, window)
            if centres is None:
                return xp.zeros(_window_shape(window), dtype=DTYPE)
            values = centres[0 if component[1] == 'x' else 1]
            return (values.real if component.endswith('_re') else values.imag).astype(DTYPE)
        return _phasor_part(self.em.solution, window, component[len(self.out_of_plane) + 1:], xp)

    def time_label(self):
        return _wavelength_label(self.em.solved_omega)


class FDFDRendererEz(_FDFDRendererBase):
    """Window-shaped float32 E_z phasor parts, phasor moduli (|H| from complex faces collocated to the centres),
    materials and absorbed power (q_abs) of an 'ez' FDFDPhysics."""

    out_of_plane = 'ez'
    material_names = ('eps_r', 'sigma_e')
    field_infos = {**_phasor_field_infos('ez', 'E_z', 'V/m'), **_MODULUS_FIELD_INFOS,
                   **{name: EMRenderer.field_infos[name] for name in (*material_names, 'absorbed_power')},
                   'eps_sensitivity': _SENSITIVITY_FIELD_INFO, 'design_density': _DESIGN_DENSITY_FIELD_INFO,
                   'crystallinity': _CRYSTALLINITY_INFO}


class FDFDRendererHz(_FDFDRendererBase):
    """Window-shaped float32 H_z phasor parts, E_x / E_y phasor parts and moduli (|E|) from complex faces collocated
    to the centres, materials and absorbed power (q_abs) of an 'hz' FDFDPhysics."""

    out_of_plane = 'hz'
    material_names = ('eps_xx', 'eps_yy', 'sigma_e')
    field_infos = {**_phasor_field_infos('hz', 'H_z', 'A/m'), **_IN_PLANE_E_PART_INFOS, **_MODULUS_FIELD_INFOS,
                   **{name: EMRendererHz.field_infos[name] for name in (*material_names, 'absorbed_power')},
                   'eps_sensitivity': _SENSITIVITY_FIELD_INFO, 'design_density': _DESIGN_DENSITY_FIELD_INFO,
                   'crystallinity': _CRYSTALLINITY_INFO, 'eps_xy': _EPS_XY_FIELD_INFO,
                   'eps_xy_sensitivity': _EPS_XY_SENSITIVITY_FIELD_INFO}


def fdfd_union_field_infos(omitted, primary) -> dict:
    """Union of the ez and hz FDFD renderer FieldInfos (ez first) minus the names in omitted, `primary` the primary
    field (polarisation-agnostic renderers: EME, NLFD)."""
    infos = {}
    for renderer_class in (FDFDRendererEz, FDFDRendererHz):
        for name, info in renderer_class.field_infos.items():
            if name not in omitted and name not in infos:
                infos[name] = replace(info, primary=name == primary)
    return infos


# ---- cross-section modes: the selected mode of a ModePhysics, zeros before the first solve ----------------------

def _mode_component_info(name, label, unit):
    """Signed FieldInfo of the real part of one mode field component (E polar, H axial: core.mirror)."""
    return FieldInfo(name, f'Re {label}', unit=unit, signed=True, cmap='RdBu_r',
                     parity=f"{'polar' if name[0] == 'e' else 'axial'}_{name[1]}")


def centre_component_infos(texts, parity=True) -> dict:
    """Re (signed; parity: the mirror parity of an E / H component) and modulus (<name>_abs) FieldInfos of centre
    phasor components [(name, label, unit)], Re first."""
    infos = {name: (_mode_component_info(name, label, unit) if parity
                    else FieldInfo(name, f'Re {label}', unit=unit, signed=True, cmap='RdBu_r'))
             for name, label, unit in texts}
    infos.update({f'{name}_abs': FieldInfo(f'{name}_abs', f'|{label}|', unit=unit, cmap='inferno')
                  for name, label, unit in texts})
    return infos


class CentreFieldRenderer(_EMRendererBase):
    """Window-shaped float32 centre fields a physics keeps in a dict (attribute `centre_fields`), its materials, e_abs
    from e_intensity and absorbed_power from 'q_abs'; time label = the physics' selection_label (ModeRenderer,
    BORRenderer, ConicalRenderer)."""

    material_names = ('eps_r', 'eps_xx', 'eps_yy', 'sigma_e')
    centre_fields = 'fields'

    def device_field(self, component: str):
        self.field_info(component)
        window = self.domain.physical_window()
        if component in self.material_names:
            return getattr(self.domain.materials, component)[window]
        fields = getattr(self.em, self.centre_fields)
        if component == 'absorbed_power':
            return fields['q_abs'][window]
        if component == 'e_abs':
            return self.domain.xp.sqrt(fields['e_intensity'][window])
        return fields[component][window]

    def time_label(self):
        return self.em.selection_label()


class ModeRenderer(CentreFieldRenderer):
    """Window-shaped float32 fields of the selected mode of a ModePhysics at the cell centres (Re of the six
    components, moduli, S_z; ModePhysics.mode_fields) and its materials."""

    centre_fields = 'mode_fields'
    field_infos = {
        'ex': _mode_component_info('ex', 'E_x', 'V/m'), 'ey': _mode_component_info('ey', 'E_y', 'V/m'),
        'ez': _mode_component_info('ez', 'E_z', 'V/m'), 'hx': _mode_component_info('hx', 'H_x', 'A/m'),
        'hy': _mode_component_info('hy', 'H_y', 'A/m'), 'hz': _mode_component_info('hz', 'H_z', 'A/m'),
        'e_abs': _MODULUS_FIELD_INFOS['e_abs'],
        'e_intensity': replace(_MODULUS_FIELD_INFOS['e_intensity'], primary=True),
        'h_abs': _MODULUS_FIELD_INFOS['h_abs'],
        'sz': FieldInfo('sz', 'S_z (power flow along z)', unit='W/m^2', signed=True, cmap='RdBu_r', parity='even'),
        'eps_r': FieldInfo('eps_r', 'Relative permittivity zz', unit='1', cmap='YlOrBr'),
        **{name: EMRendererHz.field_infos[name] for name in ('eps_xx', 'eps_yy', 'sigma_e')},
    }
