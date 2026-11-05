"""AcousticRenderer: window-shaped pressure, stress and velocity fields of the elastic_waves AcousticPhysics."""
from ...core.backend import array_module_of
from ...core.renderer import FieldInfo, PhysicsRenderer


def _lower_face_values(face_values, axis, periodic):
    """Each cell's lo-face value along `axis`: the previous face, the wrapped last one (periodic) or 0 (the slip wall)."""
    shifted = array_module_of(face_values).roll(face_values, 1, axis=axis)
    if not periodic:
        if axis == 0:
            shifted[0, :] = 0.0
        else:
            shifted[:, 0] = 0.0
    return shifted


class AcousticRenderer(PhysicsRenderer):
    """Window-shaped float32 pressure, stresses and velocities of an AcousticPhysics (raw staggered samples, |v| at
    the cell centres, physical units also under half_storage); stt only in an axisymmetric physics."""

    name = 'acoustics'
    # Mirror parity (core.mirror, symmetry 2.5): a 'rigid' face on an EM mirror plane is the even mirror (v_n = 0,
    # shear 0 on the window face: AcousticPhysics._pins), so v is polar, sxy axial_z (odd on both planes), the normal
    # stresses even; v / sxy are raw hi-face / corner samples (staggered).
    field_infos = {
        'pressure': FieldInfo('pressure', 'Pressure', unit='Pa', signed=True, cmap='RdBu_r', primary=True,
                              parity='even'),
        'sxx': FieldInfo('sxx', 'Normal stress xx', unit='Pa', signed=True, cmap='RdBu_r', parity='even'),
        'syy': FieldInfo('syy', 'Normal stress yy', unit='Pa', signed=True, cmap='RdBu_r', parity='even'),
        'sxy': FieldInfo('sxy', 'Shear stress xy', unit='Pa', signed=True, cmap='RdBu_r', parity='axial_z',
                         staggered='xy'),
        'stt': FieldInfo('stt', 'Hoop stress (axisymmetric)', unit='Pa', signed=True, cmap='RdBu_r', parity='even'),
        'vx': FieldInfo('vx', 'Particle velocity x', unit='m/s', signed=True, cmap='RdBu_r', parity='polar_x',
                        staggered='x'),
        'vy': FieldInfo('vy', 'Particle velocity y', unit='m/s', signed=True, cmap='RdBu_r', parity='polar_y',
                        staggered='y'),
        'v_abs': FieldInfo('v_abs', 'Particle speed |v|', unit='m/s', cmap='inferno'),
        'elastic_strain_xx': FieldInfo('elastic_strain_xx', 'Elastic strain xx', unit='1', signed=True, cmap='RdBu_r',
                                       parity='even'),
        'elastic_strain_yy': FieldInfo('elastic_strain_yy', 'Elastic strain yy', unit='1', signed=True, cmap='RdBu_r',
                                       parity='even'),
    }

    def __init__(self, acoustic_physics):
        self.acoustics = acoustic_physics
        if not acoustic_physics.axisymmetric:   # per instance: Session.available_fields reads it
            self.field_infos = {name: info for name, info in type(self).field_infos.items() if name != 'stt'}

    def device_field(self, component):
        self.field_info(component)
        physics = self.acoustics
        window = physics.window
        if component == 'pressure':
            return physics.pressure()[window]
        if component in ('elastic_strain_xx', 'elastic_strain_yy'):
            strain_xx, strain_yy = physics.elastic_strain()
            return (strain_xx if component == 'elastic_strain_xx' else strain_yy)[window]
        if component != 'v_abs':
            return physics.physical_array(component)[window]
        vx, vy = physics.physical_array('vx'), physics.physical_array('vy')
        vx_centre = 0.5 * (vx + _lower_face_values(vx, 0, physics.periodic_x))
        vy_centre = 0.5 * (vy + _lower_face_values(vy, 1, physics.periodic_y))
        return array_module_of(vx).sqrt(vx_centre * vx_centre + vy_centre * vy_centre)[window]

    def time_label(self):
        return f't = {self.acoustics.time:.6g} s'
