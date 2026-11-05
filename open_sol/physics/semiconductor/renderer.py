"""SemiconductorRenderer: window-shaped potential, carrier density, current, quasi-Fermi, doping and rate fields of a
DriftDiffusionPhysics."""
from ...core.renderer import FieldInfo, PhysicsRenderer

# parities: the insulating (even) mirror of a model's EM mirror faces (core.mirror); vectors are centre face means
SEMICONDUCTOR_FIELD_INFOS = {
    'V': FieldInfo('V', 'Electric potential', unit='V', signed=True, cmap='RdBu_r', primary=True,
                   parity='scalar'),
    'N': FieldInfo('N', 'Electron density', unit='1/m^3', cmap='viridis'),
    'P': FieldInfo('P', 'Hole density', unit='1/m^3', cmap='viridis'),
    'E_x': FieldInfo('E_x', 'Electric field x', unit='V/m', signed=True, cmap='RdBu_r', parity='polar_x'),
    'E_y': FieldInfo('E_y', 'Electric field y', unit='V/m', signed=True, cmap='RdBu_r', parity='polar_y'),
    'J_x': FieldInfo('J_x', 'Current density x', unit='A/m^2', signed=True, cmap='RdBu_r', parity='polar_x'),
    'J_y': FieldInfo('J_y', 'Current density y', unit='A/m^2', signed=True, cmap='RdBu_r', parity='polar_y'),
    'Jn_x': FieldInfo('Jn_x', 'Electron current density x', unit='A/m^2', signed=True, cmap='RdBu_r',
                      parity='polar_x'),
    'Jn_y': FieldInfo('Jn_y', 'Electron current density y', unit='A/m^2', signed=True, cmap='RdBu_r',
                      parity='polar_y'),
    'Jp_x': FieldInfo('Jp_x', 'Hole current density x', unit='A/m^2', signed=True, cmap='RdBu_r', parity='polar_x'),
    'Jp_y': FieldInfo('Jp_y', 'Hole current density y', unit='A/m^2', signed=True, cmap='RdBu_r', parity='polar_y'),
    'phi_n': FieldInfo('phi_n', 'Electron quasi-Fermi potential', unit='V', signed=True, cmap='RdBu_r',
                       parity='scalar'),
    'phi_p': FieldInfo('phi_p', 'Hole quasi-Fermi potential', unit='V', signed=True, cmap='RdBu_r',
                       parity='scalar'),
    'net_doping': FieldInfo('net_doping', 'Net doping', unit='1/m^3', signed=True, cmap='RdBu_r', parity='even'),
    'space_charge': FieldInfo('space_charge', 'Space charge density', unit='C/m^3', signed=True, cmap='RdBu_r',
                              parity='scalar'),
    'recombination': FieldInfo('recombination', 'Net recombination rate', unit='1/(m^3 s)', signed=True,
                               cmap='RdBu_r', parity='even'),
    'generation': FieldInfo('generation', 'Coupled generation rate', unit='1/(m^3 s)', cmap='magma'),
    'generation_ext': FieldInfo('generation_ext', 'Applied generation rate', unit='1/(m^3 s)', cmap='magma'),
}


class SemiconductorRenderer(PhysicsRenderer):
    """Window-shaped float32 fields of a DriftDiffusionPhysics (its `fields` dict)."""

    name = 'semiconductor'
    field_infos = SEMICONDUCTOR_FIELD_INFOS

    def __init__(self, physics):
        self.physics = physics

    def device_field(self, component: str):
        self.field_info(component)
        return self.physics.fields[component][self.physics.window]

    def time_label(self):
        return f"solve {self.physics.n_solves}"
