"""HeatRenderer: window-shaped temperature, temperature rise, heat source and heat-flux fields of a HeatPhysics."""
from ...core.renderer import FieldInfo, PhysicsRenderer


class HeatRenderer(PhysicsRenderer):
    """Window-shaped float32 temperature, heat-source and heat-flux fields of a HeatPhysics."""

    name = "heat"
    field_infos = {
        'T': FieldInfo('T', 'Temperature', unit='K', cmap='inferno', primary=True),
        'T_rise': FieldInfo('T_rise', 'Temperature rise', unit='K', cmap='inferno'),
        'Q': FieldInfo('Q', 'Coupled heat source', unit='W/m^3', cmap='magma'),
        'Q_ext': FieldInfo('Q_ext', 'Applied heat source', unit='W/m^3', cmap='magma'),
        'heat_flux_abs': FieldInfo('heat_flux_abs', 'Conductive heat flux magnitude', unit='W/m^2', cmap='viridis'),
    }

    def __init__(self, heat_physics):
        self.heat = heat_physics

    def device_field(self, component: str):
        self.field_info(component)
        heat = self.heat
        window = heat.window
        if component == 'T':
            return heat.fields.T[window]
        if component == 'T_rise':
            return heat.temperature_rise()
        if component == 'Q':
            return heat.fields.Q[window]
        if component == 'Q_ext':
            return heat.fields.Q_ext[window]
        return heat.heat_flux_magnitude()

    def time_label(self):
        heat = self.heat
        if heat.time_mode == 'transient':
            return f"t = {heat.time:.6g} s"
        return f"solve {heat.n_solves}"
