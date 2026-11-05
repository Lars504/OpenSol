"""CurrentsRenderer / ElectrostaticsRenderer: window-shaped V, E, flux, source, Joule and ion-concentration fields
of a PotentialPhysics (electric_currents / electrostatics); THREE_D_FIELD_INFOS, the z components on tetrahedra."""
from ...core.renderer import FieldInfo, PhysicsRenderer, polar_field_infos
from .electrolyte import ION_CONCENTRATION_FIELDS

_ION_CONCENTRATION_INFOS = {   # computed on demand from V (PotentialPhysics.ion_concentration), not in state()
    name: FieldInfo(name, f'Ion concentration, species {index}', unit='mol/L', cmap='viridis')
    for index, name in enumerate(ION_CONCENTRATION_FIELDS, start=1)}

_POTENTIAL_INFOS = {
    # parity (core.mirror): scalars follow the face parity, cell-centre vectors are polar
    'V': FieldInfo('V', 'Electric potential', unit='V', signed=True, cmap='RdBu_r', primary=True, parity='scalar'),
    'E_x': FieldInfo('E_x', 'Electric field x', unit='V/m', signed=True, cmap='RdBu_r', parity='polar_x'),
    'E_y': FieldInfo('E_y', 'Electric field y', unit='V/m', signed=True, cmap='RdBu_r', parity='polar_y'),
}


# The z components of a tetrahedron-mesh run (FEMPotentialPhysics3D, 3D models only): output fields beside the 2D
# vocabulary, never listed by a 2D renderer.
_E_Z_INFO = FieldInfo('E_z', 'Electric field z', unit='V/m', signed=True, cmap='RdBu_r', parity='polar_z')
THREE_D_FIELD_INFOS = {
    'currents': {
        'E_z': _E_Z_INFO,
        'J_z': FieldInfo('J_z', 'Current density z', unit='A/m^2', signed=True, cmap='RdBu_r', parity='polar_z'),
        'sigma_dc_zz': FieldInfo('sigma_dc_zz', 'DC conductivity zz', unit='S/m', cmap='cividis')},
    'electrostatics': {
        'E_z': _E_Z_INFO,
        'D_z': FieldInfo('D_z', 'Electric displacement z', unit='C/m^2', signed=True, cmap='RdBu_r',
                         parity='polar_z'),
        'eps_dc_zz': FieldInfo('eps_dc_zz', 'DC relative permittivity zz', unit='1', cmap='YlOrBr')},
}


class _PotentialRenderer(PhysicsRenderer):
    """Window-shaped float32 fields of a PotentialPhysics: V, E, flux, sources, q_joule, the xx / yy coefficients and
    (electrostatics) the ion concentrations c_ion_k of the species present when the renderer is made."""

    def __init__(self, physics):
        self.physics = physics
        n_species = getattr(physics, 'ion_species_count', 0)
        infos = {name: info for name, info in type(self).field_infos.items()   # per instance: available_fields
                 if name not in ION_CONCENTRATION_FIELDS[n_species:]}
        coordinates = getattr(physics.domain, 'coordinates', None)
        if coordinates is not None and coordinates.polar:   # E / J / D and the xx / yy coefficients: grid frame
            infos = polar_field_infos(infos, physics.axisymmetric)
        if infos != type(self).field_infos:
            self.field_infos = infos
        spec = physics.spec
        self._field_attributes = {   # renderer component -> PotentialFields2D attribute (others: MaterialGrid arrays)
            'V': 'V', 'E_x': 'E_x', 'E_y': 'E_y',
            spec.flux_names[0]: 'flux_x', spec.flux_names[1]: 'flux_y',
            spec.source_field: 'source', f'{spec.source_field}_ext': 'source_ext',
            'q_joule': 'q_joule', 'rho_ion': 'rho_ion',
        }

    def device_field(self, component: str):
        self.field_info(component)
        physics = self.physics
        window = physics.window
        attribute = self._field_attributes.get(component)
        if attribute is not None:
            return getattr(physics.fields, attribute)[window]
        if component in ION_CONCENTRATION_FIELDS:
            return physics.ion_concentration(ION_CONCENTRATION_FIELDS.index(component))
        return getattr(physics.domain.materials, component)[window]

    def time_label(self):
        return f"solve {self.physics.n_solves}"


class CurrentsRenderer(_PotentialRenderer):
    """Fields of an electric-currents PotentialPhysics (J = sigma_dc E, Joule heating)."""

    name = "currents"
    field_infos = {
        **_POTENTIAL_INFOS,
        'J_x': FieldInfo('J_x', 'Current density x', unit='A/m^2', signed=True, cmap='RdBu_r', parity='polar_x'),
        'J_y': FieldInfo('J_y', 'Current density y', unit='A/m^2', signed=True, cmap='RdBu_r', parity='polar_y'),
        'q_joule': FieldInfo('q_joule', 'Joule heating density', unit='W/m^3', cmap='magma'),
        'current_source': FieldInfo('current_source', 'Coupled current source', unit='A/m^3', signed=True,
                                    cmap='RdBu_r', parity='scalar'),
        'current_source_ext': FieldInfo('current_source_ext', 'Applied current source', unit='A/m^3', signed=True,
                                        cmap='RdBu_r', parity='scalar'),
        'sigma_dc_xx': FieldInfo('sigma_dc_xx', 'DC conductivity xx', unit='S/m', cmap='cividis'),
        'sigma_dc_yy': FieldInfo('sigma_dc_yy', 'DC conductivity yy', unit='S/m', cmap='cividis'),
    }


class ElectrostaticsRenderer(_PotentialRenderer):
    """Fields of an electrostatics PotentialPhysics (D = eps0 eps_dc E)."""

    name = "electrostatics"
    field_infos = {
        **_POTENTIAL_INFOS,
        'D_x': FieldInfo('D_x', 'Electric displacement x', unit='C/m^2', signed=True, cmap='RdBu_r', parity='polar_x'),
        'D_y': FieldInfo('D_y', 'Electric displacement y', unit='C/m^2', signed=True, cmap='RdBu_r', parity='polar_y'),
        'rho_free': FieldInfo('rho_free', 'Coupled charge density', unit='C/m^3', signed=True, cmap='RdBu_r',
                              parity='scalar'),
        'rho_free_ext': FieldInfo('rho_free_ext', 'Applied charge density', unit='C/m^3', signed=True, cmap='RdBu_r',
                                  parity='scalar'),
        'rho_ion': FieldInfo('rho_ion', 'Ionic space charge', unit='C/m^3', signed=True, cmap='RdBu_r',
                             parity='scalar'),
        **_ION_CONCENTRATION_INFOS,
        'eps_dc_xx': FieldInfo('eps_dc_xx', 'DC relative permittivity xx', unit='1', cmap='YlOrBr'),
        'eps_dc_yy': FieldInfo('eps_dc_yy', 'DC relative permittivity yy', unit='1', cmap='YlOrBr'),
    }
