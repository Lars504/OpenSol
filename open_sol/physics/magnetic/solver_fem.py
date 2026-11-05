"""2D magnetostatics on a triangle mesh (magnetic plan section 10, stage 5): MagneticFEMPhysics, Lagrange P1 / P2 A
(planar A_z, axisymmetric A_phi) on the window elements over system_fem.MagneticFEMSystem; the MagneticPhysics feature
API (faces, held regions, external field, sources, stranded coils, B-H tables, probes), scalars and the field names.
State arrays are per element (D12: A at the centroid, B / H / w_mag element means); display fields by fem_field (A
nodal, B / H averaged to the nodes). Saturation by exact-tangent Newton (nonlinear_method 'newton')."""
import math
from dataclasses import dataclass

import numpy as np

from ...core.backend import to_host
from ...core.constants import DTYPE, FACES, MU0
from ...core.monitors import FieldProbe
from ...core.solver_base import PhysicsSolver
from ...fem.indicators import CHUNK_ITEMS, add_edge_terms, edge_sides, lagrange_at
from ...fem.renderer import FEMFieldRenderer
from .materials import MAGNETIC_SCHEMA, PERMEABILITY_ARRAYS, REMANENCE_ARRAYS, SATURATION_ARRAY
from .renderer import MAGNETIC_FIELD_INFOS
from .saturation import (TRIANGLE_NONLINEAR_METHODS, bh_curve, refuse_curve_conflicts, saturation_curve_terms,
                         tabulated_curve_terms)
from .system_fem import MagneticFEMSystem

DISCRETIZATIONS = {'quadratic': 2, 'linear': 1}   # setting discretization -> Lagrange order (mesh plan D5)
ELEMENT_FIELDS = ('A', 'B_x', 'B_y', 'B_abs', 'H_x', 'H_y', 'w_mag', 'J_coupled', 'J_applied')
NODAL_FIELDS = ('B_x', 'B_y', 'H_x', 'H_y')
_ANCHOR_TEXT = 'magnetic insulation face, held potential or external field'
_REMEDY_TEXT = 'Keep magnetic insulation (the default) on at least one face, or add an external field.'


@dataclass(frozen=True, eq=False)
class FEMCoilRegion:
    """Runtime object of a stranded coil on triangles: host bool window elements, turns, current [A] (> 0 along +z /
    +phi), cross-section area [m^2] (the r-z area when axisymmetric)."""
    name: str
    elements: object
    turns: int
    current: float
    area: float


class ElementFields:
    """Per-element float32 arrays (n_elements,) named `names` on the domain's array module (the state() fields; the
    coupling transforms read them as on the tensor field classes)."""

    def __init__(self, names, n_elements, xp):
        for name in names:
            setattr(self, name, xp.zeros(int(n_elements), dtype=DTYPE))


def _host(array) -> np.ndarray:
    return np.asarray(to_host(array), dtype=np.float64)


class TriangleMagneticBase(PhysicsSolver):
    """What MagneticFEMPhysics and MagneticACFEMPhysics share on a triangle mesh: faces, node probes, element masks,
    the current-density slot and the axisymmetric ring; subclasses set type_label, _probe_component, `system`,
    `fields` (J_coupled) and `_axisymmetric`."""

    name = 'magnetic'
    coordinate_grids = ('cartesian',)
    conformal_locator = None     # moot on triangles: the elements follow the drawn outlines
    _probe_component = 'A'

    def _label(self):
        return f"{self.type_label} '{self.name}'"

    def inject_source(self, name: str, array) -> None:
        if name != 'current_density':
            raise ValueError(f"{self._label()} accepts source 'current_density'; got '{name}'")
        values = self.xp.asarray(array, dtype=DTYPE).reshape(-1)
        if values.size != self.fields.J_coupled.size:
            raise ValueError(f"{self._label()}: source 'current_density' needs one value per element "
                             f'({self.fields.J_coupled.size}), got {values.size}')
        self.xp.copyto(self.fields.J_coupled, values)
        self._source_dirty = True

    def set_conformal_boundaries(self, locator) -> None:
        """solver_defaults.conformal_boundaries: nothing to do (the mesh conforms to the drawn outlines)."""

    @property
    def axisymmetric(self) -> bool:
        return self._axisymmetric

    @property
    def ring_angle(self) -> float:
        return 2.0 * math.pi

    def set_face_bc(self, face: str, bc) -> None:
        """Dirichlet A = 0 (magnetic insulation) or Neumann 0 (symmetry) on an outer face's window edge; axisymmetric
        x_lo is the axis (A_phi = 0) and cannot be changed."""
        if face not in FACES:
            raise ValueError(f'unknown face {face!r}; expected one of {FACES}')
        if self._axisymmetric and face == 'x_lo':
            raise ValueError(f'{self._label()}: x_lo is the symmetry axis in axisymmetric geometry (A_phi = 0 there); '
                             'its condition is fixed')
        if bc.value != 0.0 or bc.kind not in ('dirichlet', 'neumann'):
            raise ValueError(f'{self._label()}: a face takes dirichlet 0 (insulation) or neumann 0 (symmetry) on a '
                             f'triangle mesh, got {bc.kind} {bc.value!r}')
        self.system.set_face(face, 'dirichlet' if bc.kind == 'dirichlet' else 'natural')

    def add_probe_node(self, node: int) -> FieldProbe:
        """Record _probe_component (A; AC: |A|) at a mesh node (a point object's node) whenever probes record."""
        node = int(node)
        if not 0 <= node < self.domain.mesh.n_points:
            raise ValueError(f'vector potential probe node {node} is not a mesh node')
        probe = FieldProbe(node, 0, self._probe_component)
        self.probes.append((probe, node))
        return probe

    def field_dofs(self, name):
        """fem DofMap of state field or source slot `name` (the transfer contract): every one is per element (P0)."""
        if name not in self.state() and name != 'current_density':
            raise ValueError(f"{self._label()} has no state field or source slot '{name}'")
        return self.system.element_map

    def _element_mask(self, mask, what):
        """Host bool (n_elements,) of a build mask, window elements only."""
        mask = np.asarray(to_host(mask), dtype=bool).reshape(-1)
        if mask.size != self.domain.mesh.n_elements:
            raise ValueError(f'{self._label()} {what}: mask has {mask.size} values, expected one per element '
                             f'({self.domain.mesh.n_elements})')
        return mask & self.system.window_elements

    def _kelly_indicator(self, compliance):
        """eta^2 per element: Kelly jumps h_E compliance int_E |[H . t]|^2 ds of the tangential H (the subclass's
        _edge_tangential_field, real or complex) over edges between window elements (1/2 per side) and on symmetry
        faces (one-sided); r weights axisymmetric."""
        system, mesh = self.system, self.domain.mesh
        window = system.window_elements
        natural = np.zeros(mesh.n_edges, dtype=bool)
        for face, kind in system.face_kinds.items():
            if kind == 'natural' and system.external_field is None and not (self._axisymmetric and face == 'x_lo'):
                natural[system.face_edges(face)] = True
        eta_squared = np.zeros(mesh.n_elements)
        candidates = np.unique(mesh.element_edges[window])
        for start in range(0, candidates.size, CHUNK_ITEMS):
            sides = edge_sides(mesh, candidates[start:start + CHUNK_ITEMS], window, system.geometry_order)
            keep = sides.two_sided | natural[sides.edges]
            if not keep.any():
                continue
            sides = sides.select(keep)
            jump = self._edge_tangential_field(sides, 0, np.arange(len(sides.edges)))
            two_sided = np.flatnonzero(sides.two_sided)
            if two_sided.size:
                jump[two_sided] -= self._edge_tangential_field(sides, 1, two_sided)
            weights = sides.weights * (sides.points[..., 0] if self._axisymmetric else 1.0)
            add_edge_terms(eta_squared, sides, np.sum(np.abs(jump) ** 2 * weights, axis=1), compliance)
        return eta_squared


class MagneticFEMPhysics(TriangleMagneticBase):
    """2D magnetostatics -curl(H(curl A)) = J on a fem.domain.TriangleDomain (window elements; the PML layer ring holds
    the window field at the clamped position): one linear solve per change, Newton with an energy line search when a
    material saturates; per unit z length (planar) or whole rings (axisymmetric, integrals per radian x 2 pi)."""

    type_label = 'magnetostatics'
    material_schema = dict(MAGNETIC_SCHEMA)

    def __init__(self, domain, axisymmetric: bool = False, discretization: str = 'quadratic', linear_solver=None,
                 nonlinear_tolerance: float = 1e-6, nonlinear_max_iterations: int = 100,
                 nonlinear_method: str = 'newton'):
        if getattr(domain, 'grid_kind', 'tensor') != 'triangles':
            raise ValueError(f'{self.type_label}: MagneticFEMPhysics needs a triangle mesh (fem TriangleDomain); a '
                             'tensor grid uses MagneticPhysics')
        if discretization not in DISCRETIZATIONS:
            raise ValueError(f"discretization must be one of {', '.join(DISCRETIZATIONS)}, got {discretization!r}")
        nonlinear_method = 'newton' if nonlinear_method == 'auto' else nonlinear_method
        if nonlinear_method not in TRIANGLE_NONLINEAR_METHODS:
            raise ValueError(f"{self.type_label}: nonlinear_method on a triangle mesh must be 'auto' or 'newton', got "
                             f'{nonlinear_method!r} (anderson / picard run on tensor meshes)')
        self._axisymmetric = bool(axisymmetric)
        self.order = DISCRETIZATIONS[discretization]
        self.linear_solver = linear_solver
        self.nonlinear_tolerance = float(nonlinear_tolerance)
        self.nonlinear_max_iterations = int(nonlinear_max_iterations)
        self.runtime_warnings = []   # append-only (unconverged Newton loops); the studies log new ones
        self.system = None
        self.prepare(domain)

    # --- PhysicsSolver contract ---------------------------------------------
    def prepare(self, domain) -> None:
        """Register the permeability arrays, allocate the system and the per-element fields (clears faces, held
        regions, sources, coils, curves and probes); assembly is deferred."""
        self.domain = domain
        self.xp = domain.xp
        domain.materials.add_properties(self.material_schema)
        self.system = MagneticFEMSystem(domain.mesh, self.order, self._axisymmetric, self._label, self.linear_solver,
                                        backend=domain.backend.name, anchor_text=_ANCHOR_TEXT,
                                        remedy_text=_REMEDY_TEXT)
        self.fields = ElementFields(ELEMENT_FIELDS, domain.mesh.n_elements, self.xp)
        self.coils = []
        self.probes = []             # (FieldProbe, node)
        self._bh_curves = []         # (host bool element mask, saturation.BHCurve), later wins
        self._nodal = {}             # nodal display values of the last solve
        self._materials_dirty = True
        self._source_dirty = True

    def finalize(self) -> None:
        """Materials, constraints and (without saturating elements) the factorised operator: fails early on a
        non-positive permeability or a floating region."""
        self._set_materials()
        self.system.prepare_solve()

    def materials_changed(self, names) -> None:
        names = set(names)
        if names & (PERMEABILITY_ARRAYS | set(REMANENCE_ARRAYS) | {SATURATION_ARRAY}):
            self._materials_dirty = True

    def step(self, t: float = 0.0, dt: float = math.inf) -> None:
        """Re-solve only when materials, constraints or a source changed (always at the first call); probes sample after
        a re-solve and at every dt = inf call."""
        t = 0.0 if t is None else float(t)
        system = self.system
        re_solve = self._materials_dirty or system.operator_dirty or self._source_dirty or system.n_solves == 0
        if re_solve:
            if self._materials_dirty:
                self._set_materials()
            density = _host(self.fields.J_coupled) + _host(self.fields.J_applied)
            if not system.solve(density, self.nonlinear_tolerance, self.nonlinear_max_iterations):
                self.runtime_warnings.append(system.last_warning)
            self._publish()
            self._source_dirty = False
        if re_solve or dt is None or not math.isfinite(dt):
            for probe, node in self.probes:
                probe.record_value(np.float32(system.values[node]), t, t > 0.0)

    def dt_limit(self) -> float:
        return math.inf

    def state(self) -> dict:
        state = {name: getattr(self.fields, name) for name in ELEMENT_FIELDS}
        for array_name in self.material_schema:
            state[array_name] = getattr(self.domain.materials, array_name)
        return state

    def renderer(self):
        return FEMFieldRenderer(self, 'magnetic', MAGNETIC_FIELD_INFOS, lambda: f'solve {self.n_solves}')

    # --- properties and records -------------------------------------------------
    @property
    def n_solves(self) -> int:
        return self.system.n_solves

    @property
    def state_revision(self) -> int:
        return self.system.n_solves

    def scalars(self) -> dict:
        """energy [J/m planar, J axisymmetric], per coil <name>_flux_linkage / <name>_inductance; {} before a solve."""
        if self.system.n_solves == 0:
            return {}
        values = {'energy': self.field_energy()}
        for coil in self.coils:
            result = self.coil_results(coil)
            values[f'{coil.name}_flux_linkage'] = result['flux_linkage']
            values[f'{coil.name}_inductance'] = result['inductance']
        return values

    def work_counters(self) -> dict:
        """The Newton loop's counters (nonlinear_runs / _iterations / _unconverged / _backtracks) once it ran, else {};
        the factorisations are on the 'linear' runtime."""
        system = self.system
        if not system.n_runs:
            return {}
        return {'nonlinear_runs': system.n_runs, 'nonlinear_iterations': system.total_iterations,
                'nonlinear_unconverged': system.n_unconverged, 'nonlinear_backtracks': system.total_backtracks}

    # --- magnetics API (the definition callbacks on triangles) ---------------
    def set_external_field(self, B_x: float, B_y: float) -> None:
        """Uniform applied flux density [T]: the window-edge dofs hold A = B_x y - B_y x (axisymmetric A_phi = B_y r /
        2, B_x must be 0); wins over the face conditions."""
        self.system.set_external_field(B_x, B_y)

    def add_fixed_potential(self, mask, value) -> None:
        """Hold planar A_z [T m] / axisymmetric psi = r A_phi [T m^2] (a number) on every dof of the masked elements."""
        self.system.add_held(self._element_mask(mask, 'fixed potential'), float(value))

    def add_source(self, mask, density) -> None:
        """Add a current density J_z / J_phi [A/m^2] (number or per-element array) to J_applied on the masked
        elements."""
        xp = self.xp
        self.fields.J_applied += xp.where(xp.asarray(mask), xp.asarray(density, dtype=DTYPE), DTYPE(0.0)).astype(
            DTYPE, copy=False)
        self._source_dirty = True

    def add_coil(self, name: str, mask, turns: int, current: float) -> FEMCoilRegion:
        """Stranded coil: J = turns * current / area on the masked window elements (area = their x-y / r-z area)."""
        if int(turns) < 1:
            raise ValueError(f"coil '{name}': turns must be >= 1, got {turns}")
        elements = self._element_mask(mask, f"coil '{name}'")
        _, area = self.system.region_integrals(elements)
        if not area > 0.0:
            raise ValueError(f"coil '{name}' selects no window element")
        coil = FEMCoilRegion(name, elements, int(turns), float(current), area)
        density = np.where(elements, coil.turns * coil.current / area, 0.0)
        self.fields.J_applied += self.xp.asarray(density, dtype=DTYPE)
        self._source_dirty = True
        self.coils.append(coil)
        return coil

    def add_bh_curve(self, mask, rows) -> bool:
        """Tabulated anhysteretic B-H curve (saturation.bh_curve rules) on the masked window elements, isotropic, mu_dc
        ignored there; a later curve wins shared elements. False when the mask covers no window element."""
        curve = bh_curve(rows)
        elements = self._element_mask(mask, 'B-H curve')
        if not elements.any():
            return False
        self._bh_curves.append((elements, curve))
        self._materials_dirty = True
        return True

    def coil_results(self, coil: FEMCoilRegion) -> dict:
        """current, turns, flux_linkage [Wb/m planar, Wb axisymmetric] = turns / area * int A dA (x 2 pi r), inductance
        = flux_linkage / current [H/m, H] (nan at zero current)."""
        if self.system.n_solves == 0:
            raise RuntimeError(f"{self._label()}: coil '{coil.name}' needs a solve first")
        integral, _ = self.system.region_integrals(coil.elements)
        linkage = coil.turns * (self.ring_angle if self._axisymmetric else 1.0) * integral / coil.area
        inductance = linkage / coil.current if coil.current != 0.0 else float('nan')
        return {'current': coil.current, 'turns': coil.turns, 'flux_linkage': linkage, 'inductance': inductance}

    def field_energy(self) -> float:
        """Stored energy int w(B) dV of the window [J/m planar, J axisymmetric]: the integral of H dB (secant and true
        energy agree for linear materials); with remanence the energy of B alone."""
        return self.system.stored_energy() * (self.ring_angle if self._axisymmetric else 1.0)

    def rebuild_operator(self) -> None:
        self._materials_dirty = True

    def warm_start_fields(self) -> dict:
        """Host copies of the nodal A and the coupled source, to seed a build on the same mesh."""
        return {'values': self.system.values.copy(), 'source': np.array(to_host(self.fields.J_coupled))}

    def seed_warm_start(self, fields) -> tuple:
        """Write warm_start_fields() of another build (A is the Newton start); returns the slots seeded."""
        self.system.values[...] = fields['values']
        self.system.started = True
        self.xp.copyto(self.fields.J_coupled, self.xp.asarray(fields['source'], dtype=DTYPE))
        self._source_dirty = True
        return ('A', 'J_coupled')

    # --- display fields -----------------------------------------------------------
    def fem_field(self, component) -> tuple:
        """(DofMap, host coefficients): A nodal, B / H nodal averages, B_abs = |B| of those, the rest per element."""
        system = self.system
        if component == 'A':
            return system.full, system.values
        if component in NODAL_FIELDS or component == 'B_abs':
            if not self._nodal:
                return system.full, np.zeros(system.full.n_dofs)
            if component == 'B_abs':
                return system.full, np.hypot(self._nodal['B_x'], self._nodal['B_y'])
            return system.full, self._nodal[component]
        if component in ELEMENT_FIELDS:
            return system.element_map, _host(getattr(self.fields, component))
        if component in self.material_schema:
            return system.element_map, _host(getattr(self.domain.materials, component))
        raise KeyError(f"{self._label()} has no field '{component}'")

    def error_indicator(self):
        """eta^2 per element (float64 (n_elements,)) of the last solve for mesh refinement (S7.7, mesh plan A3): Kelly
        jumps h_E / nu int_E [H . t]^2 ds of the tangential H over edges between window elements (1/2 per side) and
        H . t on symmetry faces (one-sided), nu the element's mean reluctivity; r weights axisymmetric."""
        system = self.system
        if system.n_solves == 0:
            raise ValueError(f'{self._label()}: the error indicator needs a solve')
        return self._kelly_indicator(np.where(system.window_elements, 2.0 / system.reluctivity.sum(axis=1), 0.0))

    # --- internals ------------------------------------------------------------------
    def _edge_tangential_field(self, sides, side, rows):
        """(k, q) H . t (t = the normal out of side 0 turned by +90 deg) at the edge points of `rows` from the element
        on `side`."""
        system = self.system
        elements = sides.elements[rows, side]
        values, gradients = lagrange_at(system.full, system.values, elements, sides.reference_points(side, rows))
        if self._axisymmetric:
            flux = np.stack([-gradients[..., 1], gradients[..., 0] + values / sides.points[rows][..., 0]], axis=-1)
        else:
            flux = np.stack([gradients[..., 1], -gradients[..., 0]], axis=-1)
        field, _, _ = system.constitutive(elements, flux, tangent=False)
        if system.remanence is not None:
            field = field - (system.reluctivity[elements] * system.remanence[elements])[:, None, :]
        normal = sides.normals[rows]
        return field[..., 1] * normal[..., 0] - field[..., 0] * normal[..., 1]

    def _set_materials(self):
        """Per-axis nu = 1 / (mu0 mu_dc) of the window (checked finite, > 0), the remanence and the saturating curves:
        the J_s curve (isotropic mu_dc required), then the B-H tables (later wins); a saturating element with a
        remanence, or with both a J_s and a table, raises."""
        materials = self.domain.materials
        window = self.system.window_elements
        mu_xx, mu_yy = _host(materials.mu_dc_xx), _host(materials.mu_dc_yy)
        valid = np.isfinite(mu_xx) & np.isfinite(mu_yy) & (mu_xx > 0.0) & (mu_yy > 0.0)
        if (window & ~valid).any():
            raise ValueError(f'{self._label()}: mu_dc_xx / mu_dc_yy must be finite and > 0 in every window element')
        mu_xx, mu_yy = np.where(window, mu_xx, 1.0), np.where(window, mu_yy, 1.0)
        reluctivity = np.stack([1.0 / (MU0 * mu_xx), 1.0 / (MU0 * mu_yy)], axis=1)
        names = materials.property_names()
        remanence = None
        if any(name in names for name in REMANENCE_ARRAYS):
            remanence = np.stack([np.where(window, _host(getattr(materials, name)), 0.0) if name in names
                                  else np.zeros(mu_xx.size) for name in REMANENCE_ARRAYS], axis=1)
        curves, curve_index = [], np.full(mu_xx.size, -1, dtype=np.int64)
        saturating = np.zeros(mu_xx.size, dtype=bool)
        if SATURATION_ARRAY in names:
            polarization = _host(materials.saturation_polarization)
            saturating = window & (polarization > 0.0)
            if saturating.any():
                if (saturating & (np.abs(mu_xx - mu_yy) > 1e-12 * np.abs(mu_xx))).any():
                    raise ValueError(f'{self._label()}: saturation_polarization needs an isotropic mu_dc on a '
                                     'triangle mesh (the B-space curve has no per-axis energy): mu_dc_xx = mu_dc_yy')
                curves.append(lambda magnitude, elements: saturation_curve_terms(
                    magnitude, mu_xx[elements][:, None], polarization[elements][:, None], np))
                curve_index[saturating] = 0
        for mask, curve in self._bh_curves:
            refuse_curve_conflicts(self._label(), 'element', curve_and_polarization=bool((mask & saturating).any()))
            curve_index[mask] = len(curves)
            curves.append(lambda magnitude, elements, curve=curve: tabulated_curve_terms(magnitude, curve, np))
        refuse_curve_conflicts(self._label(), 'element', saturating_remanence=remanence is not None and bool(
            ((curve_index >= 0) & (remanence != 0.0).any(axis=1)).any()))
        self.system.set_materials(reluctivity, remanence, curves, curve_index)
        self._materials_dirty = False

    def _publish(self):
        """Per-element state (A centroid, B / H / w_mag element means, B_abs = |mean B|) and the nodal display values."""
        xp = self.xp
        means = self.system.element_means()
        fields = self.fields
        flux, field = means['B'], means['H']
        for name, values in (('A', means['A']), ('B_x', flux[:, 0]), ('B_y', flux[:, 1]),
                             ('B_abs', np.hypot(flux[:, 0], flux[:, 1])), ('H_x', field[:, 0]), ('H_y', field[:, 1]),
                             ('w_mag', means['w'])):
            xp.copyto(getattr(fields, name), xp.asarray(values, dtype=DTYPE))
        self._nodal = self.system.nodal_fields()
