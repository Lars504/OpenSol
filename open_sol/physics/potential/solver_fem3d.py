"""Electric currents / electrostatics on a tetrahedron mesh (piezo_3d_plan stage 6; the 3D analogue of solver_fem.py):
FEMPotentialPhysics3D, P1 / P2 Lagrange V on the window elements (fem.diffusion3d), diagonal sigma_dc / eps_dc tensors;
the PotentialPhysics field names, slot, held regions and scalar outputs where they apply (no electrolytes, driven
terminals, Stern layers or mirror planes). State arrays are per element (D12: V at the centroid, E / J / D element
means, q_joule the element dissipation density); the renderer samples V at points (fem.renderer3d)."""
from dataclasses import dataclass

from ...core.fv_diffusion import BCSide
from ...fem.diffusion import element_means
from ...fem.diffusion3d import FEMDiffusionPhysicsBase3D, coefficient_tensor_3d
from ...fem.mesh3d import sliver_warnings
from ...fem.renderer3d import LagrangeSampleRenderer
from .electrolyte import ION_CONCENTRATION_FIELDS
from .renderer import THREE_D_FIELD_INFOS
from .solver import REMEDY_TEXT, RENDERERS
from .solver_fem import FEMPotentialMixin

# diagonal coefficient arrays (xx, yy, zz) of the operator and the z flux name per kind
DIAGONAL_ARRAYS = {'currents': ('sigma_dc_xx', 'sigma_dc_yy', 'sigma_dc_zz'),
                   'electrostatics': ('eps_dc_xx', 'eps_dc_yy', 'eps_dc_zz')}
_FLUX_Z_NAMES = {'currents': 'J_z', 'electrostatics': 'D_z'}


@dataclass(frozen=True, eq=False)
class FEMHeldRegion3D:
    """Runtime object of a voltage / terminal feature on tetrahedra: held outer faces, object interface faces (int64
    mesh faces) and / or held elements (host bool mask) at one voltage."""
    name: str
    voltage: float
    faces: tuple
    surfaces: object = None       # int64 mesh faces of held object interfaces, or None
    elements: object = None       # host bool per element of a terminal, or None
    stern_capacitance: float = 0.0   # always 0 on tetrahedra (read by solver._face_held_by)


class FEMPotentialPhysics3D(FEMPotentialMixin, FEMDiffusionPhysicsBase3D):
    """3D quasi-static potential V on a fem.domain3d.TetrahedronDomain (`kind` 'currents' / 'electrostatics'): one solve
    per change of operator or sources; held fluxes from the reactions of the un-eliminated system."""

    _axes = 'xyz'

    def __init__(self, domain, kind: str = 'currents', order: int = 2, linear_solver=None):
        self._init_potential(kind)
        self.diagonal_arrays = self._operator_arrays = DIAGONAL_ARRAYS[kind]
        self.flux_names = (*self.spec.flux_names, _FLUX_Z_NAMES[kind])
        self.finalize_warnings = []
        self._init_fem(domain, order, linear_solver, lambda: f"{self.spec.label} '{self.name}'", 'potential',
                       self.spec.anchor_text, REMEDY_TEXT)
        self.prepare(domain)

    def finalize(self) -> None:
        """Assemble (fails early on unanchored regions); a warning on sliver elements."""
        self.rebuild_operator()
        self.finalize_warnings = sliver_warnings(self.domain.mesh)

    def state(self) -> dict:
        """The 2D state names (V, E_x / E_y, the x / y fluxes, sources, coefficients, q_joule / rho_ion) plus E_z, the z
        flux and the zz coefficient."""
        fields, spec = self.fields, self.spec
        materials = self.domain.materials
        state = {'V': fields['V'], spec.source_field: fields['source'],
                 f'{spec.source_field}_ext': fields['source_ext']}
        state.update({name: fields[name] for name in ('E_x', 'E_y', 'E_z')})
        state.update({flux_name: fields[f'flux_{axis}'] for flux_name, axis in zip(self.flux_names, 'xyz')})
        for array_name in self.diagonal_arrays:
            state[array_name] = getattr(materials, array_name)
        for name in ('q_joule', 'rho_ion'):
            if name in fields:
                state[name] = fields[name]
        return state

    def renderer(self):
        """LagrangeSampleRenderer of the 2D field vocabulary (no ion concentrations) plus the z components."""
        infos = {name: info for name, info in RENDERERS[self.kind].field_infos.items()
                 if name not in ION_CONCENTRATION_FIELDS}
        infos.update(THREE_D_FIELD_INFOS[self.kind])
        return LagrangeSampleRenderer(self, RENDERERS[self.kind].name, infos, lambda: f'solve {self.n_solves}')

    # --- potential API (the definition callbacks on tetrahedra) --------------
    @property
    def flux_unit(self) -> str:
        """Unit of held_flux: whole totals in 3D (A, C)."""
        return self.spec.ring_flux_unit

    def add_surface_potential(self, faces, V) -> None:
        """V [V] on mesh faces (an object interface: Dirichlet)."""
        self.add_surface_condition(faces, BCSide('dirichlet', value=V))

    def add_surface_flux(self, faces, value) -> int:
        """Inward current density / surface charge on mesh faces (an interface: a sheet source feeding both sides);
        handle for set_surface_flux."""
        return self.add_surface_condition(faces, BCSide('neumann', value=value))

    def set_surface_flux(self, handle, value) -> None:
        self.set_surface_condition(handle, BCSide('neumann', value=value))

    def scalars(self) -> dict:
        """current_<feature> [A] or charge_<feature> [C] of every kept region; {} before a solve."""
        return self._held_scalars() if self.system.n_solves else {}

    def held_flux(self, region: FEMHeldRegion3D) -> float:
        """Flux [A | C] leaving the held faces / interface faces / elements of `region` into the domain (> 0 out): outer
        faces still holding its voltage (their Dirichlet reaction), plus the reactions of its interface and element
        dofs."""
        return self._held_reaction(region)

    def _held_dofs(self, region) -> list:
        full = self.system.full
        dofs = []
        if region.surfaces is not None:
            dofs.append(full.face_dofs(region.surfaces))
        if region.elements is not None:
            dofs.append(full.element_dofs_of(region.elements))
        return dofs

    def rebuild_operator(self) -> None:
        """Assemble the diagonal coefficient tensor (sigma_dc or eps0 eps_dc), eliminate the held dofs, factorise."""
        materials = self.domain.materials
        self._tensor = self.spec.coefficient_scale * coefficient_tensor_3d(
            *(getattr(materials, name) for name in self.diagonal_arrays))
        self.system.assemble(self._tensor)
        self._source_dirty = True

    def lagrange_field(self, component) -> tuple:
        """(DofMap3D, host coefficients) of a renderer component: V nodal, the rest per element."""
        if component == 'V':
            return self.nodal_field()
        spec = self.spec
        attributes = {'E_x': 'E_x', 'E_y': 'E_y', 'E_z': 'E_z', spec.source_field: 'source',
                      f'{spec.source_field}_ext': 'source_ext', 'q_joule': 'q_joule', 'rho_ion': 'rho_ion',
                      **{flux_name: f'flux_{axis}' for flux_name, axis in zip(self.flux_names, 'xyz')}}
        attribute = attributes.get(component)
        if attribute is not None and attribute in self.fields:
            return self.element_field(self.fields[attribute])
        if component in self.diagonal_arrays:
            return self.element_field(getattr(self.domain.materials, component))
        raise KeyError(f"{spec.label} '{self.name}' has no field '{component}'")

    # --- internals ------------------------------------------------------------
    def _publish(self):
        """Per-element state (_publish_fields: V, E, flux, q_joule)."""
        integrals = self.system.element_integrals(self._tensor)
        self._publish_fields(integrals, element_means(integrals)[1])
