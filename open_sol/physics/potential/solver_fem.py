"""Electric currents / electrostatics on a triangle mesh (mesh plan 2.7, stage T4): FEMPotentialPhysics, P1 / P2
Lagrange V on the window (fem.diffusion), planar or axisymmetric (r, z); the PotentialPhysics field names, slots, held
regions, scalar outputs and linear electrolytes where they apply (no nonlinear electrolytes, driven terminals or
azimuthal orders). State arrays are per element (D12: V at the element centroid, E / J / D element means, q_joule the
element dissipation density, rho_ion from the element mean V); FEMPotentialMixin is the physics shared with
FEMPotentialPhysics3D."""
import math
from dataclasses import dataclass

import numpy as np

from ...core.backend import to_host
from ...core.constants import DTYPE
from ...core.fv_diffusion import BCSide
from ...core.mirror import touch_factor
from ...fem.diffusion import FEMDiffusionPhysicsBase, coefficient_tensor, element_means
from ...fem.indicators import diffusion_indicator
from ...fem.mesher import MERGE_TOLERANCE
from ...fem.renderer import FEMFieldRenderer
from .electrolyte import ION_CONCENTRATION_FIELDS, number_densities, screening_coefficient, species_concentration
from .solver import ELECTROLYTE_SCALAR_PREFIXES, KINDS, REMEDY_TEXT, RENDERERS, IonicMedium, _face_held_by



class FEMPotentialMixin:
    """Potential physics over a FEMDiffusionPhysicsBase[3D] (FEMPotentialPhysics, FEMPotentialPhysics3D): kind, fields,
    steps, sources, held regions and their reactions; a subclass gives _axes, _operator_arrays (after _init_potential)
    and _held_dofs(region)."""

    name = 'currents'
    _probe_field = 'V'
    _axes = 'xy'   # E / flux components of the per-element fields

    def _init_potential(self, kind) -> None:
        """kind 'currents' / 'electrostatics' and its schema (before _init_fem)."""
        if kind not in KINDS:
            raise ValueError(f'kind must be one of {list(KINDS)}, got {kind!r}')
        self.kind = kind
        self.spec = KINDS[kind]
        self.material_schema = dict(self.spec.schema)
        self.slot_names = (self.spec.slot,)
        self.held_regions = []
        self._source_dirty = True
        self._tensor = None   # host (n_elements, d, d) coefficient of the last assembly

    # --- PhysicsSolver contract ---------------------------------------------
    def prepare(self, domain) -> None:
        """Register the coefficient arrays and allocate the per-element fields; assembly is deferred."""
        self.domain = domain
        domain.materials.add_properties(self.material_schema)
        xp = self.xp
        count = domain.mesh.n_elements
        vectors = [f'{kind}_{axis}' for kind in ('E', 'flux') for axis in self._axes]
        self.fields = {name: xp.zeros(count, dtype=DTYPE) for name in ('V', *vectors, 'source', 'source_ext')}
        if self.spec.with_joule:
            self.fields['q_joule'] = xp.zeros(count, dtype=DTYPE)
        if self.spec.with_ions:   # tetrahedra: no electrolytes, rho_ion stays 0 (the state vocabulary of the type)
            self.fields['rho_ion'] = xp.zeros(count, dtype=DTYPE)
        self._source_dirty = True

    def materials_changed(self, names) -> None:
        if set(self._operator_arrays) & set(names):
            self.system.operator_dirty = True

    def step(self, t: float = 0.0, dt: float = math.inf) -> None:
        """Re-solve only when the operator or a source changed (always at the first call); probes sample after a
        re-solve and at every dt = inf call (as PotentialPhysics)."""
        t = 0.0 if t is None else float(t)
        system = self.system
        re_solve = system.operator_dirty or self._source_dirty or system.n_solves == 0
        if re_solve:
            if system.operator_dirty:
                self.rebuild_operator()
            system.solve(self._source())
            self._publish()
            self._source_dirty = False
        if re_solve or dt is None or not math.isfinite(dt):
            self._record_probes(t, replace_same_time=t > 0.0)

    def solve_coupled_source(self, time: float) -> None:
        self.step(float(time), math.inf)

    def reset_records(self) -> None:
        """RerunSolver (reused sweep build): clear the probe series; the next run re-solves when an input changed."""
        for probe, *_ in self.probes:
            probe.clear()

    def dt_limit(self) -> float:
        return math.inf

    def inject_source(self, name: str, array) -> None:
        if name != self.spec.slot:
            raise ValueError(f"{self.spec.label} '{self.name}' accepts source '{self.spec.slot}'; got '{name}'")
        self.xp.copyto(self.fields['source'], self.xp.asarray(array, dtype=DTYPE))
        self._source_dirty = True

    # --- potential API (the definition callbacks) -----------------------------
    @property
    def state_revision(self) -> int:
        return self.system.n_solves

    def add_fixed_potential(self, mask, V, waveform=None, inside=None) -> None:
        """Hold V [V] (a number) on every dof of the masked elements (a terminal); no waveforms on element meshes."""
        if waveform is not None:
            raise ValueError(f"{self.spec.label} '{self.name}': driven terminals (waveform) need a 2D tensor mesh")
        self.add_fixed_value(mask, V)

    def add_source(self, mask, density) -> None:
        """Add a volumetric source density (number or per-element array) to source_ext on the masked elements."""
        self._add_masked(self.fields['source_ext'], mask, density)
        self._source_dirty = True

    def add_held_region(self, region):
        self.held_regions.append(region)
        return region

    def _held_scalars(self) -> dict:
        """current_<feature> or charge_<feature> [flux_unit] of every kept region."""
        prefix = 'current_' if self.kind == 'currents' else 'charge_'
        return {f'{prefix}{region.name}': self.held_flux(region) for region in self.held_regions}

    def _held_reaction(self, region) -> float:
        """Flux leaving the held faces / outline or interface / elements of `region` into the domain (> 0 out): outer
        faces still holding its voltage BC (their face_flux) plus the reaction of its _held_dofs (per radian on (r,
        z))."""
        system = self.system
        if system.n_solves == 0:
            raise RuntimeError(f"{self.spec.label} '{self.name}': the flux of '{region.name}' needs a solve first")
        flux = 0.0
        for face in region.faces:
            if _face_held_by(self.bcs[face], region):
                flux += system.face_flux(face)
        dofs = self._held_dofs(region)
        if dofs:
            flux += system.reaction(np.unique(np.concatenate(dofs)))
        return float(flux)

    def warm_start_fields(self) -> dict:
        return {'values': self.system.values.copy(), 'source': np.array(to_host(self.fields['source']))}

    def seed_warm_start(self, fields) -> tuple:
        self.system.values[...] = fields['values']
        self.xp.copyto(self.fields['source'], self.xp.asarray(fields['source'], dtype=DTYPE))
        self._source_dirty = True
        self._values_changed()
        return ('V', self.spec.source_field)

    # --- internals ------------------------------------------------------------
    def _source(self):
        """Host float64 source_ext + source per element."""
        return (np.asarray(to_host(self.fields['source_ext']), dtype=np.float64)
                + np.asarray(to_host(self.fields['source']), dtype=np.float64))

    def _publish_fields(self, integrals, gradient):
        """V at the centroids, E = -grad V and flux = c E (element means), q_joule = the element dissipation density
        (its integral = the stiffness energy, so V I balances exactly)."""
        xp = self.xp
        fields = self.fields
        electric = -gradient
        flux = np.einsum('mcd,md->mc', self._tensor, electric)
        fields['V'][...] = xp.asarray(self.system.centroid_values(), dtype=DTYPE)
        for axis_index, axis in enumerate(self._axes):
            fields[f'E_{axis}'][...] = xp.asarray(electric[:, axis_index], dtype=DTYPE)
            fields[f'flux_{axis}'][...] = xp.asarray(flux[:, axis_index], dtype=DTYPE)
        if 'q_joule' in fields:
            measure = integrals['measure']
            density = np.where(measure > 0.0, integrals['dissipation'] / np.where(measure > 0.0, measure, 1.0), 0.0)
            fields['q_joule'][...] = xp.asarray(density, dtype=DTYPE)

    def _rhs_changed(self):
        self._source_dirty = True

    def _values_changed(self):
        self.fields['V'][...] = self.xp.asarray(self.system.centroid_values(), dtype=DTYPE)


@dataclass(frozen=True, eq=False)
class FEMHeldRegion:
    """Runtime object of a voltage / terminal feature on triangles: held outer faces, outline edges (int64) and / or
    held elements (host bool mask) at one voltage; faces behind a Stern layer when stern_capacitance > 0."""
    name: str
    voltage: float
    faces: tuple
    edges: object = None          # int64 mesh edges of held object outlines, or None
    elements: object = None       # host bool per element of a terminal, or None
    stern_capacitance: float = 0.0


class FEMPotentialPhysics(FEMPotentialMixin, FEMDiffusionPhysicsBase):
    """2D quasi-static potential V on a fem.domain.TriangleDomain (`kind` 'currents' / 'electrostatics'): one solve per
    change of operator or sources; held fluxes from the reactions of the un-eliminated system."""

    mirror_faces = frozenset()

    def __init__(self, domain, kind: str = 'currents', order: int = 2, axisymmetric: bool = False,
                 linear_solver=None):
        self._init_potential(kind)
        self._operator_arrays = self.spec.coefficient_arrays
        self._init_fem(domain, order, linear_solver, axisymmetric, lambda: f"{self.spec.label} '{self.name}'",
                       'potential', self.spec.anchor_text, REMEDY_TEXT)
        self.electrolytes = []   # ElectrolyteRegions of add_electrolyte_region, reported by scalars()
        self.ionic_media = []    # IonicMedium per add_electrolyte, mask_window = host bool per element
        self.runtime_warnings = []
        self._mean_potential = np.zeros(domain.mesh.n_elements)   # element mean V of the last solve (ionic terms)
        self.prepare(domain)

    def finalize(self) -> None:
        self.rebuild_operator()

    def state(self) -> dict:
        fields, spec = self.fields, self.spec
        materials = self.domain.materials
        state = {'V': fields['V'], 'E_x': fields['E_x'], 'E_y': fields['E_y'],
                 spec.flux_names[0]: fields['flux_x'], spec.flux_names[1]: fields['flux_y'],
                 spec.source_field: fields['source'], f'{spec.source_field}_ext': fields['source_ext']}
        for array_name in spec.coefficient_arrays:
            state[array_name] = getattr(materials, array_name)
        for name in ('q_joule', 'rho_ion'):
            if name in fields:
                state[name] = fields[name]
        return state

    def renderer(self):
        """FEMFieldRenderer listing c_ion_1 .. up to ion_species_count (made after the build, as PotentialPhysics')."""
        infos = {name: info for name, info in RENDERERS[self.kind].field_infos.items()
                 if name not in ION_CONCENTRATION_FIELDS[self.ion_species_count:]}
        return FEMFieldRenderer(self, RENDERERS[self.kind].name, infos, lambda: f'solve {self.n_solves}')

    # --- potential API (the definition callbacks on triangles) ---------------
    @property
    def waveforms(self) -> tuple:
        return ()

    def set_mirror_faces(self, faces) -> None:
        """Mirror planes of the model on the window edge: held fluxes join their images (x 2 per touched face)."""
        self.mirror_faces = frozenset(faces)

    def add_edge_potential(self, edges, V, stern_capacitance: float = 0.0) -> None:
        """V [V] on mesh edges (an object outline: Dirichlet; no Stern layer on outlines)."""
        if stern_capacitance > 0.0:
            raise ValueError('a Stern layer applies to outer faces only')
        self.add_edge_condition(edges, BCSide('dirichlet', value=V))

    def add_edge_flux(self, edges, value) -> int:
        """Inward current density / surface charge on mesh edges (a line source on an outline); handle for
        set_edge_flux."""
        return self.add_edge_condition(edges, BCSide('neumann', value=value))

    def set_edge_flux(self, handle, value) -> None:
        self.set_edge_condition(handle, BCSide('neumann', value=value))

    # --- electrolytes (electrostatics, model 'linear' only) -------------------
    def add_electrolyte(self, mask, concentrations, valences, temperature: float = 298.15,
                        bulk_potential: float = 0.0, model: str = 'linear') -> IonicMedium:
        """Debye-Hueckel ion species on the masked elements: the loss k (V - V_bulk), none on held elements, overlapping
        electrolytes add; model 'nonlinear' needs a tensor mesh (ValueError)."""
        label = f"{self.spec.label} '{self.name}'"
        if not self.spec.with_ions:
            raise ValueError(f'{label} has no electrolytes (electrostatics only)')
        if model != 'linear':
            raise ValueError(f"{label}: electrolyte model {model!r} needs a tensor mesh (rectangular cells); on "
                             f"triangles use model 'linear'")
        concentrations = tuple(float(value) for value in concentrations)
        valences = tuple(float(value) for value in valences)
        if (not concentrations or len(concentrations) != len(valences) or 0.0 in valences
                or not all(math.isfinite(value) and value >= 0.0 for value in concentrations)):
            raise ValueError(f'electrolyte needs one non-zero valence per finite concentration >= 0, got '
                             f'{concentrations} and {valences}')
        if not float(temperature) > 0.0:
            raise ValueError(f'electrolyte temperature must be > 0 K, got {temperature}')
        medium = IonicMedium(self.element_mask(mask).copy(), model, concentrations, number_densities(concentrations),
                             valences, float(temperature), float(bulk_potential),
                             screening_coefficient(concentrations, valences, temperature))
        self.ionic_media.append(medium)
        self.system.operator_dirty = True
        return medium

    def add_electrolyte_region(self, region):
        """Keep an electrolyte feature's ElectrolyteRegion for scalars(); returns it."""
        self.electrolytes.append(region)
        return region

    @property
    def ion_species_count(self) -> int:
        """Longest species list of the electrolytes (0 without one): the renderer lists c_ion_1 .. up to it."""
        return max((len(medium.valences) for medium in self.ionic_media), default=0)

    def mean_relative_permittivity(self, mask) -> float:
        """Area-weighted mean of (eps_dc_xx + eps_dc_yy) / 2 over the masked elements (NaN for an empty mask)."""
        mask = self.element_mask(mask)
        materials = self.domain.materials
        permittivity = 0.5 * (np.asarray(to_host(materials.eps_dc_xx), dtype=np.float64)
                              + np.asarray(to_host(materials.eps_dc_yy), dtype=np.float64))
        areas = self.domain.element_areas(self.system.geometry_order)
        total = float(areas[mask].sum())
        return float((areas * permittivity)[mask].sum()) / total if total > 0.0 else math.nan

    def _held_elements(self) -> np.ndarray:
        """Host bool per element of every held region (terminals: ion-free conductors)."""
        held = np.zeros(self.system.mesh.n_elements, dtype=bool)
        for mask, _, _ in self.system.held:
            held |= mask
        return held

    def _ionic_loss(self) -> tuple:
        """(k, k V_bulk) per element summed over the electrolytes, 0 on held elements."""
        count = self.system.mesh.n_elements
        coefficient, drive = np.zeros(count), np.zeros(count)
        for medium in self.ionic_media:
            coefficient += np.where(medium.mask_window, medium.coefficient, 0.0)
            drive += np.where(medium.mask_window, medium.coefficient * medium.bulk_potential, 0.0)
        held = self._held_elements()
        coefficient[held] = 0.0
        drive[held] = 0.0
        return coefficient, drive

    def ion_concentration(self, species_index: int) -> np.ndarray:
        """Host float64 concentration [mol/L] per element of species `species_index` (0-based) of every electrolyte at
        the element mean V (linear: exact element means); 0 on held and unsolved elements and without ions."""
        total = np.zeros(self.system.mesh.n_elements)
        for medium in self.ionic_media:
            if species_index < len(medium.valences):
                concentration = species_concentration(
                    self._mean_potential, medium.bulk_potential, medium.concentrations[species_index],
                    medium.valences[species_index], medium.temperature, medium.model, np)
                total += np.where(medium.mask_window, concentration, 0.0)
        total[self._held_elements() | ~self._solved_elements()] = 0.0
        return total

    def _solved_elements(self) -> np.ndarray:
        """Host bool per element of the last assembly's active elements."""
        solved = np.zeros(self.system.mesh.n_elements, dtype=bool)
        if self.system.dofs is not None:
            solved[self.system.dofs.elements] = True
        return solved

    @property
    def flux_unit(self) -> str:
        """Unit of held_flux: per unit z length (A/m, C/m), whole rings when axisymmetric (A, C)."""
        return self.spec.ring_flux_unit if self.axisymmetric else self.spec.flux_unit

    def scalars(self) -> dict:
        """current_<feature> or charge_<feature> [flux_unit] of every kept region, ionic_strength_<feature> [mol/L] /
        debye_length_<feature> [m] of every electrolyte; {} before a solve."""
        if self.system.n_solves == 0:
            return {}
        scalars = self._held_scalars()
        for region in self.electrolytes:
            scalars.update({f'{name_prefix}{region.name}': getattr(region, attribute)
                            for name_prefix, attribute in ELECTROLYTE_SCALAR_PREFIXES.items()})
        return scalars

    def held_flux(self, region: FEMHeldRegion) -> float:
        """Flux [flux_unit] leaving the held faces / outline / elements of `region` into the domain (> 0 out): faces
        still holding its voltage BC (Dirichlet reaction or the Stern-layer Robin flux), plus the reactions of its
        outline and element dofs; whole rings when axisymmetric; x mirror_factor(region)."""
        flux = self._held_reaction(region)
        if self.axisymmetric:
            flux *= self.ring_angle
        return flux * self.mirror_factor(region)

    def _held_dofs(self, region) -> list:
        system = self.system
        dofs = []
        if region.edges is not None:
            dofs.append(system.dofs.edge_dofs(region.edges))
        if region.elements is not None:
            dofs.append(system.region_dofs(region.elements))
        return dofs

    def mirror_factor(self, region: FEMHeldRegion) -> float:
        """core.mirror.touch_factor of the mirror faces `region` touches: a held face on or meeting the plane, held
        outline / element nodes on the window edge there."""
        touched = set()
        mesh = self.system.mesh
        x0, y0, x1, y1 = mesh.window
        tolerance = MERGE_TOLERANCE * max(x1 - x0, y1 - y0)
        nodes = []
        if region.edges is not None:
            nodes.append(mesh.edges[region.edges].ravel())
        if region.elements is not None:
            nodes.append(mesh.triangles[region.elements].ravel())
        points = mesh.points[np.concatenate(nodes)] if nodes else np.zeros((0, 2))
        for mirror in self.mirror_faces:
            opposite = mirror[0] + ('_hi' if mirror.endswith('lo') else '_lo')
            axis = 0 if mirror[0] == 'x' else 1
            line = {'x_lo': x0, 'x_hi': x1, 'y_lo': y0, 'y_hi': y1}[mirror]
            if any(face != opposite for face in region.faces) or bool(
                    (np.abs(points[:, axis] - line) <= tolerance).any()):
                touched.add(mirror)
        return touch_factor(self.mirror_faces, touched)

    def rebuild_operator(self) -> None:
        """Assemble the coefficient tensor (sigma_dc or eps0 eps_dc) and the ionic loss, eliminate the held dofs,
        factorise."""
        materials = self.domain.materials
        names = self.spec.coefficient_arrays
        self._tensor = self.spec.coefficient_scale * coefficient_tensor(*(getattr(materials, name) for name in names))
        if self.ionic_media:   # the whole loss term is potential's: rewritten from the media and current held regions
            coefficient, drive = self._ionic_loss()
            inside = coefficient > 0.0
            self.system.set_loss(inside, coefficient, np.where(inside, drive / np.where(inside, coefficient, 1.0), 0.0))
        self.system.assemble(self._tensor)
        self._source_dirty = True

    def fem_field(self, component) -> tuple:
        """(DofMap, host coefficients) of a renderer component: V nodal, the rest per element."""
        if component == 'V':
            return self.nodal_field()
        spec = self.spec
        attributes = {'E_x': 'E_x', 'E_y': 'E_y', spec.flux_names[0]: 'flux_x', spec.flux_names[1]: 'flux_y',
                      spec.source_field: 'source', f'{spec.source_field}_ext': 'source_ext',
                      'q_joule': 'q_joule', 'rho_ion': 'rho_ion'}
        attribute = attributes.get(component)
        if attribute is not None and attribute in self.fields:
            return self.element_field(self.fields[attribute])
        if component in spec.coefficient_arrays:
            return self.element_field(getattr(self.domain.materials, component))
        if component in ION_CONCENTRATION_FIELDS[:self.ion_species_count]:
            return self.element_field(self.ion_concentration(ION_CONCENTRATION_FIELDS.index(component)))
        raise KeyError(f"{spec.label} '{self.name}' has no field '{component}'")

    def error_indicator(self):
        """eta^2 per element (float64 (n_elements,)) of the last solve for mesh refinement (S7.7, mesh plan A3):
        fem.indicators.diffusion_indicator with the sources source_ext + source."""
        return diffusion_indicator(self.system, self._tensor, self._source())

    # --- internals ------------------------------------------------------------
    def _publish(self):
        """Per-element state (_publish_fields) and rho_ion = -k (mean V - V_bulk) (its integral = minus the sum of every
        held charge: the reactions see the same loss term)."""
        integrals = self.system.element_integrals(self._tensor)
        mean_potential, gradient = element_means(integrals)
        self._mean_potential = mean_potential
        if self.ionic_media:
            coefficient, drive = self._ionic_loss()
            charge = np.where(integrals['measure'] > 0.0, drive - coefficient * mean_potential, 0.0)
            self.fields['rho_ion'][...] = self.xp.asarray(charge, dtype=DTYPE)
        self._publish_fields(integrals, gradient)
