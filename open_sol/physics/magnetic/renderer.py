"""Window-shaped magnetic fields: MagneticRenderer (magnetostatics), MagneticRenderer3D, MagneticACRenderer (phasors)
and their FieldInfo tables."""
from ...core.renderer import FieldInfo, PhysicsRenderer

# One vocabulary for both geometries: x = r and y = z in axisymmetric models (labels name both).
# Mirror parity (core.mirror): A and J are polar out-of-plane components, B and H axial in-plane ones; the plane sign
# is the physics' own face kind there (core.mirror.FACE_FEATURE_SIGNS: symmetry +1 = A even, insulation -1 = A odd).
MAGNETIC_FIELD_INFOS = {
    'B_abs': FieldInfo('B_abs', 'Magnetic flux density magnitude', unit='T', cmap='viridis', primary=True),
    'B_x': FieldInfo('B_x', 'Magnetic flux density x (axisymmetric: r)', unit='T', signed=True, cmap='RdBu_r',
                     parity='axial_x'),
    'B_y': FieldInfo('B_y', 'Magnetic flux density y (axisymmetric: z)', unit='T', signed=True, cmap='RdBu_r',
                     parity='axial_y'),
    'A': FieldInfo('A', 'Magnetic vector potential (z; axisymmetric: phi)', unit='T m', signed=True, cmap='RdBu_r',
                   parity='polar_z'),
    'H_x': FieldInfo('H_x', 'Magnetic field x (axisymmetric: r)', unit='A/m', signed=True, cmap='RdBu_r',
                     parity='axial_x'),
    'H_y': FieldInfo('H_y', 'Magnetic field y (axisymmetric: z)', unit='A/m', signed=True, cmap='RdBu_r',
                     parity='axial_y'),
    'w_mag': FieldInfo('w_mag', 'Magnetic energy density', unit='J/m^3', cmap='magma'),
    'J_coupled': FieldInfo('J_coupled', 'Coupled current density', unit='A/m^2', signed=True, cmap='RdBu_r',
                           parity='polar_z'),
    'J_applied': FieldInfo('J_applied', 'Applied current density', unit='A/m^2', signed=True, cmap='RdBu_r',
                           parity='polar_z'),
    'mu_dc_xx': FieldInfo('mu_dc_xx', 'Static relative permeability xx', unit='1', cmap='cividis'),
    'mu_dc_yy': FieldInfo('mu_dc_yy', 'Static relative permeability yy', unit='1', cmap='cividis'),
}


def _phasor_infos(name, label, unit, parity):
    """(re, im) FieldInfos of one phasor component (both parts share its mirror parity)."""
    return (FieldInfo(f'{name}_re', f'{label}, real part', unit=unit, signed=True, cmap='RdBu_r', parity=parity),
            FieldInfo(f'{name}_im', f'{label}, imaginary part', unit=unit, signed=True, cmap='RdBu_r', parity=parity))


# magnetic_fields_ac: exp(+j omega t) phasors (peak amplitudes); magnitudes are |phasor|, losses and w_mag time averages.
MAGNETIC_AC_FIELD_INFOS = {info.name: info for info in (
    FieldInfo('B_abs', 'Magnetic flux density amplitude', unit='T', cmap='viridis', primary=True),
    *_phasor_infos('B_x', 'Magnetic flux density x (axisymmetric: r)', 'T', 'axial_x'),
    *_phasor_infos('B_y', 'Magnetic flux density y (axisymmetric: z)', 'T', 'axial_y'),
    FieldInfo('A_abs', 'Magnetic vector potential amplitude', unit='T m', cmap='viridis'),
    *_phasor_infos('A', 'Magnetic vector potential (z; axisymmetric: phi)', 'T m', 'polar_z'),
    FieldInfo('J_total_abs', 'Total current density amplitude', unit='A/m^2', cmap='inferno'),
    *_phasor_infos('J_total', 'Total current density', 'A/m^2', 'polar_z'),
    *_phasor_infos('J_conduction', 'Conduction current density (eddy and solid conductors)', 'A/m^2', 'polar_z'),
    FieldInfo('q_eddy', 'Ohmic loss density (time average)', unit='W/m^3', cmap='inferno'),
    FieldInfo('q_magnetic', 'Magnetic loss density (time average)', unit='W/m^3', cmap='inferno'),
    FieldInfo('w_mag', 'Magnetic energy density (time average)', unit='J/m^3', cmap='magma'),
    MAGNETIC_FIELD_INFOS['J_coupled'],
    *_phasor_infos('J_applied', 'Applied current density', 'A/m^2', 'polar_z'),
    MAGNETIC_FIELD_INFOS['mu_dc_xx'],
    MAGNETIC_FIELD_INFOS['mu_dc_yy'],
    FieldInfo('sigma_dc_xx', 'Static conductivity', unit='S/m', cmap='cividis'),
)}


# magnetostatics_3d: phi_m = Omega, the reduced scalar potential (H = T - grad Omega, T the coil fields). core.mirror's
# plane sign is phi_m's own (FACE_FEATURE_SIGNS: insulation +1 = even, symmetry -1 = odd), so B and H, its gradient,
# are polar and J = curl H axial.
MAGNETIC_3D_FIELD_INFOS = {info.name: info for info in (
    FieldInfo('B_abs', 'Magnetic flux density magnitude', unit='T', cmap='viridis', primary=True),
    *(FieldInfo(f'B_{axis}', f'Magnetic flux density {axis}', unit='T', signed=True, cmap='RdBu_r',
                parity=f'polar_{axis}') for axis in 'xyz'),
    FieldInfo('phi_m', 'Magnetic scalar potential', unit='A', signed=True, cmap='RdBu_r', parity='scalar'),
    *(FieldInfo(f'H_{axis}', f'Magnetic field {axis}', unit='A/m', signed=True, cmap='RdBu_r',
                parity=f'polar_{axis}') for axis in 'xyz'),
    MAGNETIC_FIELD_INFOS['w_mag'],
    *(FieldInfo(f'J_{axis}', f'Coil current density {axis}', unit='A/m^2', signed=True, cmap='RdBu_r',
                parity=f'axial_{axis}') for axis in 'xyz'),
    FieldInfo('mu_eff', 'Effective relative permeability', unit='1', cmap='cividis'),
    MAGNETIC_FIELD_INFOS['mu_dc_xx'],
    MAGNETIC_FIELD_INFOS['mu_dc_yy'],
    FieldInfo('mu_dc_zz', 'Static relative permeability zz', unit='1', cmap='cividis'),
)}


class MagneticRenderer(PhysicsRenderer):
    """Window-shaped float32 fields of a MagneticPhysics: A, B, H, w_mag, current densities and the permeabilities
    (MagneticACRenderer: the phasor fields)."""

    name = 'magnetic'
    field_infos = MAGNETIC_FIELD_INFOS

    def __init__(self, physics):
        self.physics = physics

    def device_field(self, component: str):
        self.field_info(component)
        physics = self.physics
        if component.startswith(('mu_dc_', 'sigma_dc_')):
            return getattr(physics.domain.materials, component)[physics.window]
        return getattr(physics.fields, component)[physics.window]

    def time_label(self):
        return f"solve {self.physics.n_solves}"


class MagneticRenderer3D(MagneticRenderer):
    """Window-shaped (wx, wy, wz) float32 fields of a MagneticPhysics3D (J reads 0 in a model without coils, mu_eff the
    mean of mu_dc_xx / _yy / _zz in a model without saturation)."""

    field_infos = MAGNETIC_3D_FIELD_INFOS

    def device_field(self, component: str):
        physics = self.physics
        fields = physics.fields
        if component.startswith('J_') and not fields.has_current_density:
            self.field_info(component)
            return physics.xp.zeros(physics.system.window_shape, dtype=fields.B_abs.dtype)
        if component == 'mu_eff' and not fields.has_effective_permeability:
            materials = physics.domain.materials
            window = physics.window
            return (materials.mu_dc_xx[window] + materials.mu_dc_yy[window] + materials.mu_dc_zz[window]) / 3.0
        return super().device_field(component)


class MagneticACRenderer(MagneticRenderer):
    """Window-shaped float32 phasor fields of a MagneticACPhysics."""

    field_infos = MAGNETIC_AC_FIELD_INFOS

    def time_label(self):
        return f"f = {self.physics.frequency:.4g} Hz, solve {self.physics.n_solves}"
