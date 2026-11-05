"""Membrane field infos (the w phasor as real components, tension, areal density) and MembraneLineRenderer, the
window-shaped (nx, 1) renderer of a 1D membrane; triangle meshes use fem.renderer.FEMFieldRenderer."""
from ...core.renderer import FieldInfo, PhysicsRenderer

# Mirror parities (core.mirror): w takes the plane sign of the membrane's own face feature there (FACE_FEATURE_SIGNS:
# symmetry / free +1, antisymmetry / held -1); the tension is even; tension_xy flips under either mirror whatever the
# sign of w, which no parity kind expresses: it stays undeclared (unfolding it raises).
MEMBRANE_FIELD_INFOS = {
    'w_re': FieldInfo('w_re', 'Out-of-plane displacement (real part)', unit='m', signed=True, cmap='RdBu_r',
                      primary=True, parity='scalar'),
    'w_im': FieldInfo('w_im', 'Out-of-plane displacement (imaginary part)', unit='m', signed=True, cmap='RdBu_r',
                      parity='scalar'),
    'w_abs': FieldInfo('w_abs', 'Out-of-plane displacement amplitude', unit='m', cmap='viridis'),
    'w_phase': FieldInfo('w_phase', 'Out-of-plane displacement phase', unit='rad', signed=True, cmap='twilight',
                         parity='phase_scalar'),
    'tension_xx': FieldInfo('tension_xx', 'Tension xx', unit='N/m', signed=True, cmap='RdBu_r', parity='even'),
    'tension_yy': FieldInfo('tension_yy', 'Tension yy', unit='N/m', signed=True, cmap='RdBu_r', parity='even'),
    'tension_xy': FieldInfo('tension_xy', 'Tension xy', unit='N/m', signed=True, cmap='RdBu_r'),
    'areal_density': FieldInfo('areal_density', 'Areal density', unit='kg/m^2', cmap='viridis'),
}
STATE_FIELD_NAMES = ('w_re', 'w_im', 'w_abs')   # per element / cell (D12): the phasor at the centroid / cell centre
PHASOR_FIELD_NAMES = ('w_re', 'w_im', 'w_abs', 'w_phase')


def shown_label(physics):
    """'resonance k: f = ... Hz', 'f = ... Hz' or 'not solved' of a membrane physics (renderer time label)."""
    if physics.selected_resonance is not None:
        resonance = physics.eigen_solve.resonances[physics.selected_resonance - 1]
        return f'resonance {physics.selected_resonance}: {resonance.text()}'
    if physics.response_frequency is not None:
        return f'f = {physics.response_frequency:.6g} Hz'
    return 'not solved'


class MembraneLineRenderer(PhysicsRenderer):
    """Window-shaped float32 (nx, 1) fields of a 1D membrane (MembraneLinePhysics.window_field)."""

    name = 'membrane'
    field_infos = MEMBRANE_FIELD_INFOS

    def __init__(self, physics):
        self.physics = physics

    def device_field(self, component):
        self.field_info(component)
        return self.physics.window_field(component)

    def time_label(self):
        return shown_label(self.physics)
