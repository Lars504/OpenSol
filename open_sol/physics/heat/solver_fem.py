"""Heat conduction on a triangle mesh (mesh plan 2.7, stage T4): FEMHeatPhysics, P1 / P2 Lagrange on the conducting
window elements (fem.diffusion), steady or implicit-Euler transient, planar or axisymmetric (r, z); the HeatPhysics
field names, slots and feature API where they apply. State arrays are per element (D12: T at the element centroid);
FEMHeatMixin is the heat physics shared with FEMHeatPhysics3D."""
import math

import numpy as np

from ...core.backend import to_host
from ...core.constants import DTYPE, STEFAN_BOLTZMANN
from ...core.fv_diffusion import BCSide
from ...fem.diffusion import EdgeCondition, FEMDiffusionPhysicsBase, coefficient_tensor, element_means
from ...fem.indicators import diffusion_indicator
from ...fem.renderer import FEMFieldRenderer
from ..em.index_reference import record_reference_index
from .fields import heat_state
from .materials import THERMAL_SCHEMA
from .renderer import HeatRenderer
from .solver import RADIATION_MAX_ITERATIONS, RADIATION_TOLERANCE

_CAPACITY_MATERIALS = frozenset({'rho', 'cp'})
ANCHOR_TEXT = 'temperature / convection {boundary}, fixed-temperature domain or volumetric loss'
REMEDY_TEXT = 'Anchor every conducting region, or solve in transient mode with rho and cp set there.'


class FEMHeatMixin:
    """Heat physics over a FEMDiffusionPhysicsBase[3D] (FEMHeatPhysics, FEMHeatPhysics3D): T0 handling, steps, state,
    sources, radiation and per-element fields; a subclass gives _operator_materials, _coefficient_tensor(materials) and
    the boundary-entity hooks of radiation (_entity_count, _add_entity_condition, _set_entity_condition,
    _entity_rise)."""

    name = 'heat'
    material_schema = THERMAL_SCHEMA
    solves_every_step = True   # no dirty check (as HeatPhysics)
    slot_names = ('volumetric_power',)
    _probe_field = 'T'
    _operator_materials = frozenset()

    def _init_heat(self, T0, time_mode, time_step) -> None:
        """Checked time settings and the reference temperature (before _init_fem)."""
        if time_mode not in ('stationary', 'transient'):
            raise ValueError(f"time_mode must be 'stationary' or 'transient', got {time_mode!r}")
        if time_mode == 'transient' and not (math.isfinite(time_step) and time_step > 0.0):
            raise ValueError(f'transient heat needs a finite time_step > 0, got {time_step}')
        self.T0 = float(T0)
        self.reference_value = self.T0
        self.time_mode = time_mode
        self.time_step = float(time_step)
        self.time = 0.0
        self._assembled_time_step = None
        self._capacity_stale = False
        self._tensor = None        # host (n_elements, d, d) k of the last assembly
        self._flux_stale = False   # heat_flux_abs is computed on read (_refresh_flux), not after every step
        self._flux_tensor = None
        # radiation terms: [boundary entities, condition handle, eps sigma, T_amb, T of the last linearisation]
        self._radiation = []
        self.radiation_iterations = 0   # steady re-linearisations solved (work counter)
        self.runtime_warnings = []      # append-only study log (an unconverged radiation loop)

    # --- PhysicsSolver contract ---------------------------------------------
    def prepare(self, domain) -> None:
        """Register the thermal schema, record the thermo-optic reference indices, allocate the per-element state."""
        self.domain = domain
        materials = domain.materials
        materials.add_properties(self.material_schema)
        record_reference_index(materials)
        xp = self.xp
        count = domain.mesh.n_elements
        self.T = xp.full(count, self.T0, dtype=DTYPE)
        self.Q = xp.zeros(count, dtype=DTYPE)
        self.Q_ext = xp.zeros(count, dtype=DTYPE)
        self._heat_flux = xp.zeros(count, dtype=DTYPE)
        self._T0_device = xp.asarray(self.T0, dtype=DTYPE)

    def finalize(self) -> None:
        """Re-record the thermo-optic reference indices and assemble (fails early on unanchored regions)."""
        record_reference_index(self.domain.materials)
        self.rebuild_operator()

    def materials_changed(self, names) -> None:
        names = set(names)
        if self._operator_materials & names:
            self.system.operator_dirty = True
        if _CAPACITY_MATERIALS & names:
            self._capacity_stale = True

    def step(self, t: float = 0.0, dt: float = math.inf) -> None:
        """Steady solve, or (transient mode, finite dt) one implicit-Euler step over [t, t + dt] (T, `time` and probe
        samples then belong to t + dt)."""
        t = 0.0 if t is None else float(t)
        transient = self.time_mode == 'transient' and dt is not None and math.isfinite(dt)
        time_step = float(dt) if transient else math.inf
        if not time_step > 0.0:
            raise ValueError(f"heat '{self.name}': transient step needs dt > 0, got {dt}")
        # radiation: Newton-linearised at the current T (transient: lagged one step); steady: re-linearised and
        # re-solved until the radiating entities' T settles (each re-linearisation re-assembles: h changes)
        iterations = RADIATION_MAX_ITERATIONS if (self._radiation and not transient) else 1
        for iteration in range(iterations):
            if self._radiation:
                change = self._linearise_radiation()
                if iteration and change <= RADIATION_TOLERANCE:
                    break
                if iteration:
                    self.radiation_iterations += 1
            if (self.system.operator_dirty or time_step != self._assembled_time_step
                    or (transient and self._capacity_stale)):
                self._assemble(time_step)
            self.system.solve(self._source())
        else:
            if iterations > 1:
                self.runtime_warnings.append(
                    f"heat '{self.name}': the radiation re-linearisation did not converge in {iterations} solves "
                    f'(largest relative temperature change on radiating boundaries above {RADIATION_TOLERANCE:g}); '
                    'the last solve is kept')
        self._publish()
        record_time = t + time_step if transient else t
        if transient:
            self.time = record_time
        self._record_probes(record_time)

    def reset_records(self) -> None:
        """RerunSolver (reused sweep build): clear the probe series; the next run re-solves from the last field."""
        for probe, *_ in self.probes:
            probe.clear()

    def dt_limit(self) -> float:
        return self.time_step if self.time_mode == 'transient' else math.inf

    def state(self) -> dict:
        return heat_state(self.T, self._T0_device, self.Q, self.Q_ext, self.domain.materials)

    def inject_source(self, name: str, array) -> None:
        if name != 'volumetric_power':
            raise ValueError(f"heat '{self.name}' accepts source 'volumetric_power'; got '{name}'")
        self.xp.copyto(self.Q, self.xp.asarray(array, dtype=DTYPE))

    # --- heat API (the definition callbacks) ----------------------------------
    def add_fixed_temperature(self, mask, T, inside=None) -> None:
        """Hold T [K] (a number) on every dof of the masked elements (inside: unused, the mesh conforms)."""
        self.add_fixed_value(mask, T)

    def add_heat_source(self, mask, Q) -> None:
        """Add Q [W/m^3] (number or per-element array) to Q_ext on the masked elements."""
        self._add_masked(self.Q_ext, mask, Q)

    def set_initial_temperature(self, mask, T) -> None:
        """Write T [K] (a number) on the dofs of the masked elements."""
        self.set_initial_value(mask, T)

    def rebuild_operator(self) -> None:
        """Re-assemble at the last step length (before any step: time_step when transient)."""
        time_step = self._assembled_time_step
        if time_step is None:
            time_step = self.time_step if self.time_mode == 'transient' else math.inf
        self._assemble(time_step)

    def temperature_rise(self):
        """T - T0 [K] per element (centroid), float32 run backend."""
        return (self.T - self._T0_device).astype(DTYPE)

    def heat_flux_magnitude(self):
        """|k grad T| [W/m^2] per element (element mean of the flux vector), float32 run backend."""
        self._refresh_flux()
        return self._heat_flux

    def _field(self, component) -> tuple:
        """(DofMap, host coefficients) of a renderer component: T / T_rise nodal, the rest per element (fem_field on
        triangles, lagrange_field on tetrahedra)."""
        if component == 'T':
            return self.nodal_field(self.T0)
        if component == 'T_rise':
            return self.nodal_field()
        if component in ('Q', 'Q_ext'):
            return self.element_field(getattr(self, component))
        if component == 'heat_flux_abs':
            return self.element_field(self.heat_flux_magnitude())
        raise KeyError(f"heat '{self.name}' has no field '{component}'")

    def warm_start_fields(self) -> dict:
        """Host copies of the unknown and the coupled Q, to seed a build on the same mesh (sweep warm start)."""
        return {'values': self.system.values.copy(), 'Q': np.array(to_host(self.Q))}

    def seed_warm_start(self, fields) -> tuple:
        self.system.values[...] = fields['values']
        self.xp.copyto(self.Q, self.xp.asarray(fields['Q'], dtype=DTYPE))
        self._publish()
        return ('T', 'Q')

    # --- radiation (feature radiation on triangles / tetrahedra) ------------------
    def add_face_radiation(self, face, emissivity, ambient_temperature) -> None:
        """Surface-to-ambient radiation eps sigma (T^4 - T_amb^4) on an outer face (its window edges / faces)."""
        if face not in self._faces:
            raise ValueError(f'unknown face {face!r}; expected one of {self._faces}')
        self._add_radiation_term(self._face_entities(face), emissivity, ambient_temperature)

    def _add_radiation_term(self, entities, emissivity, ambient_temperature) -> None:
        """Register one radiation term on mesh edges (2D) / faces (3D): a Robin condition whose per-entity h and
        external are the Newton linearisation at the entity's T, re-linearised by every step."""
        if not (0.0 <= emissivity <= 1.0 and ambient_temperature > 0.0):
            raise ValueError(f"heat '{self.name}': radiation needs 0 <= emissivity <= 1 and T_amb > 0 K, got "
                             f'{emissivity}, {ambient_temperature}')
        entities = np.unique(np.asarray(entities, dtype=np.int64))
        if not entities.size or emissivity == 0.0:
            return
        handle = self._add_entity_condition(entities, EdgeCondition('robin'))
        self._radiation.append([entities, handle, float(emissivity) * STEFAN_BOLTZMANN, float(ambient_temperature),
                                None])
        self._linearise_radiation()

    def _linearise_radiation(self) -> float:
        """Every radiation term's Robin condition at the current T: loss c (T^4 - T_amb^4) ~ q_k + g (T - T_k), g = 4 c
        T_k^3 = h, external = T_k - q_k / g (per entity, in the rise); returns the largest change of a radiating
        entity's T since its previous linearisation over the largest T (inf at the first)."""
        change = 0.0
        count = self._entity_count()
        for term in self._radiation:
            entities, handle, coefficient, ambient, linearised_at = term
            temperature = self.T0 + self._entity_rise(entities)
            cube = temperature * temperature * temperature
            slope = 4.0 * coefficient * cube
            loss = coefficient * (cube * temperature - ambient ** 4)
            h, external = np.zeros(count), np.zeros(count)
            h[entities] = slope
            external[entities] = np.where(slope > 0.0, temperature - loss / np.where(slope > 0.0, slope, 1.0)
                                          - self.T0, 0.0)
            self._set_entity_condition(handle, EdgeCondition('robin', h=h, external=external))
            term[4] = temperature
            if linearised_at is None:
                change = math.inf
            elif change < math.inf:
                change = max(change, float(np.max(np.abs(temperature - linearised_at)) / np.max(temperature)))
        self._rhs_changed()
        return change

    # --- internals ------------------------------------------------------------
    def _source(self):
        """Host float64 Q_ext + Q per element."""
        return np.asarray(to_host(self.Q_ext), dtype=np.float64) + np.asarray(to_host(self.Q), dtype=np.float64)

    def _assemble(self, time_step):
        materials = self.domain.materials
        tensor = self._coefficient_tensor(materials)
        capacity = None
        if math.isfinite(time_step):
            capacity = (np.asarray(to_host(materials.rho), dtype=np.float64)
                        * np.asarray(to_host(materials.cp), dtype=np.float64) / time_step)
        self.system.assemble(tensor, capacity)
        self._tensor = tensor
        self._assembled_time_step = time_step
        self._capacity_stale = False

    def _publish(self):
        """Per-element state from the solution: T at the centroids; |k grad T| only marked stale (computed on read)."""
        self._values_changed()
        self._flux_stale = True
        self._flux_tensor = self._tensor   # k of this solve (a re-assembly before the read must not change it)

    def _refresh_flux(self):
        """|k grad T| element means of the last solution into _heat_flux when a solve made them stale."""
        if not self._flux_stale:
            return
        self._flux_stale = False
        if self._flux_tensor is not None:
            _, gradient = element_means(self.system.element_integrals())
            flux = np.einsum('mcd,md->mc', self._flux_tensor, gradient)
            self._heat_flux[...] = self.xp.asarray(np.linalg.norm(flux, axis=1), dtype=DTYPE)

    def _values_changed(self):
        self.T[...] = self.xp.asarray(self.T0 + self.system.centroid_values(), dtype=DTYPE)

    def _time_label(self):
        return f't = {self.time:.6g} s' if self.time_mode == 'transient' else f'solve {self.n_solves}'


class FEMHeatPhysics(FEMHeatMixin, FEMDiffusionPhysicsBase):
    """Heat on a fem.domain.TriangleDomain: the rise T - T0 by Lagrange elements of `order` (2 quadratic, 1 linear),
    steady or implicit-Euler transient; T, Q, Q_ext and heat_flux_abs per element (state), T nodal (fem_field)."""

    _operator_materials = frozenset({'k_xx', 'k_yy', 'k_xy'})

    def __init__(self, domain, T0: float = 293.15, order: int = 2, time_mode: str = 'stationary',
                 time_step: float = math.inf, axisymmetric: bool = False, linear_solver=None,
                 require_anchor: bool = True):
        self._init_heat(T0, time_mode, time_step)
        self._init_fem(domain, order, linear_solver, axisymmetric, lambda: f"heat '{self.name}'", 'temperature',
                       ANCHOR_TEXT.format(boundary='boundary'), REMEDY_TEXT, require_anchors=require_anchor)
        self.prepare(domain)

    def renderer(self):
        return FEMFieldRenderer(self, 'heat', HeatRenderer.field_infos, self._time_label)

    fem_field = FEMHeatMixin._field

    # --- outline API (the definition callbacks on triangles) -------------------
    def add_edge_temperature(self, edges, T) -> int:
        """Dirichlet T [K] on mesh edges (an object outline); handle for set_edge_condition."""
        return self.add_edge_condition(edges, BCSide('dirichlet', value=T))

    def add_edge_convection(self, edges, h, T_ext) -> None:
        """Convective loss h (T - T_ext) per unit outline length on mesh edges."""
        self.add_edge_condition(edges, BCSide('robin', h=h, external=T_ext))

    def add_edge_flux(self, edges, flux) -> int:
        """Inward heat flux [W/m^2] on mesh edges (an outline: a line source, both sides take heat); handle for
        set_edge_flux."""
        return self.add_edge_condition(edges, BCSide('neumann', value=flux))

    def set_edge_flux(self, handle, flux) -> None:
        self.set_edge_condition(handle, BCSide('neumann', value=flux))

    def add_edge_radiation(self, edges, emissivity, ambient_temperature) -> None:
        """Surface-to-ambient radiation on mesh edges (an object outline: both sides radiate where both conduct)."""
        self._add_radiation_term(edges, emissivity, ambient_temperature)

    def _face_entities(self, face):
        return self.system.face_edges(face)

    def _entity_count(self) -> int:
        return self.domain.mesh.n_edges

    def _add_entity_condition(self, edges, condition) -> int:
        return self.system.add_edge_condition(edges, condition)

    def _set_entity_condition(self, handle, condition) -> None:
        self.system.set_edge_condition(handle, condition)

    def _entity_rise(self, edges) -> np.ndarray:
        """Rise at the edge midpoints (P2: the midside dof; P1: the mean of the end nodes)."""
        mesh, values = self.domain.mesh, self.system.values
        if self.system.order == 2:
            return values[mesh.n_points + edges]
        return values[mesh.edges[edges]].mean(axis=1)

    def add_point_heat_source_node(self, node: int, line_power: float) -> None:
        """Point heat source at a mesh node: P [W/m] planar, the ring's total power [W] axisymmetric."""
        self.add_point_source_node(node, line_power)

    def error_indicator(self):
        """eta^2 per element (float64 (n_elements,)) of the last solve for mesh refinement (S7.7, mesh plan A3):
        fem.indicators.diffusion_indicator with the heat sources Q_ext + Q (steady residual)."""
        return diffusion_indicator(self.system, self._tensor, self._source())

    def _coefficient_tensor(self, materials):
        return coefficient_tensor(materials.k_xx, materials.k_yy, materials.k_xy)
