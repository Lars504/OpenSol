"""FluidRenderer: window-shaped velocity, pressure, vorticity, body-force and viscosity fields of the laminar_flow
FluidPhysics (lattice Boltzmann)."""
from ...core.renderer import FieldInfo, PhysicsRenderer


class FluidRenderer(PhysicsRenderer):
    """Window-shaped float32 fields of a FluidPhysics: velocity (u = flow velocity here, displacement in solid_mechanics)
    and its magnitude, pressure, vorticity, coupled body forces, viscosity; parities of the even (slip-wall) mirror of a
    model's EM mirror faces (core.mirror)."""

    name = 'fluid'
    field_infos = {
        'u_abs': FieldInfo('u_abs', 'Flow speed |u|', unit='m/s', cmap='viridis', primary=True),
        'u_x': FieldInfo('u_x', 'Velocity x', unit='m/s', signed=True, cmap='RdBu_r', parity='polar_x'),
        'u_y': FieldInfo('u_y', 'Velocity y', unit='m/s', signed=True, cmap='RdBu_r', parity='polar_y'),
        'pressure': FieldInfo('pressure', 'Pressure', unit='Pa', signed=True, cmap='RdBu_r', parity='even'),
        'vorticity': FieldInfo('vorticity', 'Vorticity', unit='1/s', signed=True, cmap='RdBu_r', parity='axial_z'),
        'body_force_x': FieldInfo('body_force_x', 'Coupled body force x', unit='N/m^3', signed=True, cmap='RdBu_r',
                                  parity='polar_x'),
        'body_force_y': FieldInfo('body_force_y', 'Coupled body force y', unit='N/m^3', signed=True, cmap='RdBu_r',
                                  parity='polar_y'),
        'viscosity': FieldInfo('viscosity', 'Dynamic viscosity', unit='Pa s', cmap='cividis'),
    }

    def __init__(self, physics):
        self.physics = physics

    def device_field(self, component: str):
        self.field_info(component)
        physics = self.physics
        if component == 'vorticity':
            return physics.vorticity_window()
        if component == 'viscosity':
            return physics.domain.materials.viscosity[physics.window]
        return getattr(physics.fields, component)[physics.window]

    def time_label(self):
        return f'lattice step {self.physics.total_steps}'
