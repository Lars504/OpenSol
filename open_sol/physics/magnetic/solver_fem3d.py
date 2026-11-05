"""magnetostatics_3d on a tetrahedron mesh (the FEM analogue of solver3d.py): the reduced scalar potential u = mu0 phi_m
by Lagrange P1 / P2 (fem.diffusion3d) with C = mu_dc per axis; the drive w = mu0 T + B_r / mu_dc enters as the gradient
load int grad v . mu w (coil T as Nedelec-1 edge integrals, coils3d), symmetry (u = 0) / insulation faces, ideal-iron
held elements, a uniform external field on the window faces' dofs, a corner anchor, a saturation loop on ln mu per
element; energy and coil flux linkage by exact quadrature. State per element (D12), phi_m nodal."""
import math
from dataclasses import dataclass

import numpy as np

from ...core.backend import to_host
from ...core.constants import DTYPE, FACES_3D, MU0
from ...core.damped_iteration import DampedIteration, DampingSettings
from ...core.monitors import FieldProbe
from ...fem.assembly3d import assemble_lagrange_gradient_load_3d, rule_values_3d
from ...fem.diffusion3d import FEMDiffusionPhysicsBase3D, coefficient_tensor_3d
from ...fem.dofs3d import nedelec_dofs_3d
from ...fem.mesh3d import sliver_warnings
from ...fem.renderer3d import LagrangeSampleRenderer
from ...fem.sampling3d import evaluate_lagrange_3d, evaluate_nedelec_3d
from .coils3d import coil_edge_drive_per_ampere
from .fields import (MAGNETIC_3D_CURRENT_NAMES, MAGNETIC_3D_FIELD_NAMES, MAGNETIC_3D_PERMEABILITY_NAME,
                     MAGNETIC_3D_PROBE_COMPONENTS)
from .materials import MAGNETIC_3D_SCHEMA, PERMEABILITY_3D_ARRAYS, REMANENCE_3D_ARRAYS, SATURATION_ARRAY
from .renderer import MAGNETIC_3D_FIELD_INFOS
from .saturation import (ANDERSON_DEPTH, TENSOR_NONLINEAR_METHODS, bh_curve, refuse_curve_conflicts,
                         saturation_permeability, tabulated_reluctivity)

RULE_DEGREE = 2          # exact for every product here: affine elements, P0 mu, linear B (P2 grad u, Nedelec-1 T)
POST_CHUNK = 20_000      # elements per post-processing / load block (bounds the (m, q, 10, 3) P2 temporaries)
COIL_FIELD_CACHE_BYTES = 256 * 2 ** 20   # total-T rule values kept across saturation iterations (host), at most
_ANCHOR_TEXT = 'symmetry face, ideal iron region, external field or the corner anchor'
_REMEDY_TEXT = 'Give every window element mu_dc > 0.'
_PERMEABILITY_NAMES = ('mu_dc_xx', 'mu_dc_yy', 'mu_dc_zz')
_AXES = 'xyz'
_ELEMENT_FIELDS = ('B_x', 'B_y', 'B_z', 'B_abs', 'H_x', 'H_y', 'H_z', 'w_mag')


@dataclass(frozen=True, eq=False)
class CoilRegionFEM3D:
    """Stranded coil on tetrahedra: axis index, turns, current [A] (> 0 counter-clockwise seen from +axis), axial
    length h [m] and the host float64 Nedelec-1 dofs of T per ampere (n_edges,) [1]."""
    name: str
    axis_index: int
    turns: int
    current: float
    axial_length: float
    per_ampere: np.ndarray


class MagneticFEMPhysics3D(FEMDiffusionPhysicsBase3D):
    """magnetostatics_3d on a fem.domain3d.TetrahedronDomain: div(mu (w - grad u)) = 0 by Lagrange elements of `order`
    (2 quadratic, 1 linear); one direct solve per change (a saturation loop when an element saturates); B = mu (w -
    grad u), H = (B - B_r) / (mu0 mu) per element (means), 0 in held elements."""

    name = 'magnetic'
    type_label = 'magnetostatics_3d'
    material_schema = MAGNETIC_3D_SCHEMA

    def __init__(self, domain, order: int = 2, linear_solver=None, nonlinear_tolerance: float = 1e-6,
                 nonlinear_max_iterations: int = 100, nonlinear_relaxation: float = 0.5,
                 nonlinear_method: str = 'auto'):
        self._init_fem(domain, order, linear_solver, lambda: f"{self.type_label} '{self.name}'",
                       'magnetic scalar potential', _ANCHOR_TEXT, _REMEDY_TEXT)
        nonlinear_method = 'anderson' if nonlinear_method == 'auto' else nonlinear_method
        if nonlinear_method not in TENSOR_NONLINEAR_METHODS:
            raise ValueError(f"{self.type_label}: nonlinear_method must be 'auto' or one of {TENSOR_NONLINEAR_METHODS}, "
                             f'got {nonlinear_method!r}')
        self.picard = DampedIteration(
            DampingSettings(nonlinear_tolerance, nonlinear_max_iterations, nonlinear_relaxation,
                            anderson_depth=ANDERSON_DEPTH if nonlinear_method == 'anderson' else 0),
            label=lambda: f"{self.type_label} '{self.name}' saturation")
        self.runtime_warnings = []      # append-only (unconverged saturation loops)
        self.finalize_warnings = []
        mesh = domain.mesh
        self.coils = []
        self.edge_map = nedelec_dofs_3d(mesh, 1)   # coil T dofs (line integrals per edge)
        self._held = np.zeros(mesh.n_elements, dtype=bool)
        self._external = None           # (dof ids, values) of the applied field on the window faces
        self._external_handle = None
        self._anchor_handle = None
        self._anchor_dof = None
        self._bh_curves = []            # (element mask, saturation.BHCurve), later wins
        self._nonlinear = None          # int64 saturating elements (J_s > 0 or a curve); None: no loop
        self._saturation = None         # float64 J_s of the saturating elements
        self._log_permeability_state = None   # float64 ln mu_r per saturating element (warm start); None = mu_dc
        self._saturation_stale = True
        self._permeability = None       # host float64 (n_elements, 3) mu_dc of the last full assembly
        self._assembled = None          # host float64 (n_elements, 3) assembled mu (secant where saturating)
        self._source_dirty = True
        self._values = {}               # host float64 per-element fields (_ELEMENT_FIELDS, J_*, 'B_rms')
        self._energy = math.nan
        self._linkages = {}             # coil name -> flux linkage [Wb] of the last solve
        # total-T rule values per element chunk (mesh- and current-constant): {'key': (element set, currents), chunk
        # key -> (m, q, 3)}; cleared at every full assembly (the active / held sets may change)
        self._coil_field_cache = {}
        self.prepare(domain)

    # --- PhysicsSolver contract ---------------------------------------------
    def prepare(self, domain) -> None:
        """Register the permeability arrays, allocate the per-element float32 state; assembly is deferred."""
        self.domain = domain
        domain.materials.add_properties(self.material_schema)
        count = domain.mesh.n_elements
        xp = self.xp
        self._state = {name: xp.zeros(count, dtype=DTYPE) for name in MAGNETIC_3D_FIELD_NAMES}
        self._values = {name: np.zeros(count) for name in (*_ELEMENT_FIELDS, 'B_rms')}
        self._source_dirty = True
        self._saturation_stale = True
        self._log_permeability_state = None

    def finalize(self) -> None:
        """Assemble once (fails early on a non-positive permeability or an unanchored region); warns on slivers."""
        self._assemble()
        self.finalize_warnings = sliver_warnings(self.domain.mesh)

    def materials_changed(self, names) -> None:
        names = set(names)
        if PERMEABILITY_3D_ARRAYS & names:
            self.system.operator_dirty = True
        if names.intersection(REMANENCE_3D_ARRAYS):
            self._source_dirty = True
        if SATURATION_ARRAY in names:
            self._saturation_stale = True
            self.system.operator_dirty = True

    def step(self, t: float = 0.0, dt: float = math.inf) -> None:
        """Re-solve only when the operator or a drive changed (always at the first call; a saturation loop when an
        element saturates); probes sample after a re-solve and at every dt = inf call."""
        t = 0.0 if t is None else float(t)
        system = self.system
        re_solve = system.operator_dirty or self._source_dirty or system.n_solves == 0
        if re_solve:
            if self._saturation_stale:
                self._refresh_saturation()
            if self._nonlinear is not None:
                self._solve_saturated()
            else:
                if system.operator_dirty:
                    self._assemble()
                self._solve_once()
            self._source_dirty = False
        if re_solve or dt is None or not math.isfinite(dt):
            self._record_probes(t, replace_same_time=t > 0.0)

    def dt_limit(self) -> float:
        return math.inf

    def state(self) -> dict:
        state = dict(self._state)
        for array_name in self.material_schema:
            state[array_name] = getattr(self.domain.materials, array_name)
        return state

    def inject_source(self, name: str, array) -> None:
        raise ValueError(f"{self.type_label} '{self.name}' has no source slots; got '{name}'")

    def renderer(self):
        return LagrangeSampleRenderer(self, 'magnetic', MAGNETIC_3D_FIELD_INFOS, self._time_label)

    def scalars(self) -> dict:
        """energy [J], per coil <name>_flux_linkage [Wb] / <name>_inductance [H]; {} before a solve."""
        if self.system.n_solves == 0:
            return {}
        values = {'energy': self._energy}
        for coil in self.coils:
            result = self.coil_results(coil)
            values[f'{coil.name}_flux_linkage'] = result['flux_linkage']
            values[f'{coil.name}_inductance'] = result['inductance']
        return values

    def work_counters(self) -> dict:
        """The saturation loop's counters once it ran, else {}."""
        return self.picard.work_counters() if self.picard.n_runs else {}

    # --- magnetics API (set_face_bc: BCSide('neumann', 0) = insulation, the default; ('dirichlet', 0) = symmetry) ---
    def add_ideal_iron(self, mask, potential=0.0) -> None:
        """Hold Omega = potential [A] (a number; u = mu0 Omega) on every dof of the masked elements: ideal iron (H = 0
        inside, B normal to its surface); B and H read 0 there."""
        element_mask = self.element_mask(mask) & self.system.window_elements
        self.system.add_held(element_mask, MU0 * float(potential))
        self._held |= element_mask
        self._source_dirty = True

    def set_external_field(self, B_x: float, B_y: float, B_z: float) -> None:
        """Uniform applied flux density [T]: u = -B . r (r from the window corner) on every dof of the six window
        faces, re-imposed at each assembly so it wins dofs shared with face features and held elements."""
        system = self.system
        faces = np.unique(np.concatenate([system.face_faces(face) for face in FACES_3D]))
        ids = system.full.face_dofs(faces)
        offsets = system.full.dof_points()[ids] - np.asarray(self.domain.mesh.window[:3], dtype=np.float64)
        self._external = (ids, -(offsets @ np.array([B_x, B_y, B_z], dtype=np.float64)))
        if self._external_handle is None:
            self._external_handle = system.add_fixed_dofs(*self._external)
        self._source_dirty = True

    def add_coil(self, name: str, mask, axis: str, turns: int, current: float) -> CoilRegionFEM3D:
        """Stranded coil around axis 'x' / 'y' / 'z' on the masked winding elements (closed around a hole in every
        section normal to the axis): T = turns current / h * s along the axis as Nedelec-1 edge integrals (coils3d),
        J = curl T per element."""
        if int(turns) < 1:
            raise ValueError(f"coil '{name}': turns must be >= 1, got {turns}")
        if axis not in _AXES:
            raise ValueError(f"coil '{name}': axis must be 'x', 'y' or 'z', got {axis!r}")
        winding = self.element_mask(mask) & self.system.window_elements
        per_ampere, axial_length = coil_edge_drive_per_ampere(self.domain.mesh, winding, axis, int(turns),
                                                              f"coil '{name}'", self.domain.locate)
        coil = CoilRegionFEM3D(name, _AXES.index(axis), int(turns), float(current), axial_length, per_ampere)
        self.coils.append(coil)
        self._write_current_density()
        self._source_dirty = True
        return coil

    def add_bh_curve(self, mask, rows) -> bool:
        """Tabulated B-H curve [(H [A/m], B [T]), ...] on the masked window elements (isotropic, mu_dc ignored there;
        a later curve wins shared elements). False when the mask covers no window element."""
        curve = bh_curve(rows, np)
        element_mask = self.element_mask(mask) & self.system.window_elements
        if not element_mask.any():
            return False
        self._bh_curves.append((element_mask, curve))
        self._log_permeability_state = None
        self._saturation_stale = True
        self.system.operator_dirty = True
        return True

    def coil_results(self, coil: CoilRegionFEM3D, flows=None) -> dict:
        """current, turns, flux_linkage [Wb] = int T / I . B dV over the free elements (with an external field, ideal
        iron at a potential or a symmetry face up to their boundary term), inductance = flux_linkage / current [H]
        (nan at zero current), energy [J] (field_energy); flows: unused (the tensor signature)."""
        if self.system.n_solves == 0:
            raise RuntimeError(f"{self.type_label} '{self.name}': coil '{coil.name}' needs a solve first")
        linkage = self._linkages[coil.name]
        inductance = linkage / coil.current if coil.current != 0.0 else float('nan')
        return {'current': coil.current, 'turns': coil.turns, 'flux_linkage': linkage, 'inductance': inductance,
                'energy': self._energy}

    def field_energy(self, flows=None) -> float:
        """Energy [J] = int sum_c B_c^2 / (2 mu0 mu_c) dV over the free elements (held excluded; with remanence the
        energy of B alone; saturating: the secant value)."""
        return self._energy

    def add_probe_at(self, point, component: str = 'B_abs') -> FieldProbe:
        """Record B_abs [T] (|B| evaluated at the point) or phi_m [A] at (x, y, z) [m] whenever step records."""
        if component not in MAGNETIC_3D_PROBE_COMPONENTS:
            raise ValueError(f'{self.type_label} probe component must be one of {MAGNETIC_3D_PROBE_COMPONENTS}, got '
                             f'{component!r}')
        element, reference = self._locate_point(point, 'magnetic probe')
        probe = FieldProbe(element, 0, component)
        self.probes.append((probe, element, reference))
        return probe

    def rebuild_operator(self) -> None:
        self._assemble()

    def lagrange_field(self, component) -> tuple:
        """(DofMap3D, host coefficients) of a renderer component: phi_m nodal, the rest per element."""
        system = self.system
        if component == 'phi_m':
            return system.full, system.values / MU0
        if component in self._values:
            return self.element_field(self._values[component])
        if component in MAGNETIC_3D_CURRENT_NAMES:   # zeros before a coil
            return self.element_field(np.zeros(self.domain.mesh.n_elements))
        if component == MAGNETIC_3D_PERMEABILITY_NAME:
            if self._nonlinear is not None and self._assembled is not None:
                return self.element_field(self._assembled.mean(axis=1))
            materials = self.domain.materials
            return self.element_field(sum(np.asarray(to_host(getattr(materials, name)), dtype=np.float64)
                                          for name in _PERMEABILITY_NAMES) / 3.0)
        if component in _PERMEABILITY_NAMES:
            return self.element_field(getattr(self.domain.materials, component))
        raise KeyError(f"{self.type_label} '{self.name}' has no field '{component}'")

    def warm_start_fields(self) -> dict:
        """Host copy of the unknown u, to seed a build on the same mesh."""
        return {'values': self.system.values.copy()}

    def seed_warm_start(self, fields) -> tuple:
        self.system.values[...] = fields['values']
        self._state['phi_m'][...] = self.xp.asarray(self.system.centroid_values() / MU0, dtype=DTYPE)
        self._source_dirty = True
        return ('phi_m',)

    # --- internals ------------------------------------------------------------
    def _rhs_changed(self) -> None:
        self._source_dirty = True

    def _values_changed(self) -> None:
        self._source_dirty = True

    def _time_label(self):
        return f'solve {self.n_solves}'

    def _assemble(self, full=True):
        """Assembly with C = mu_dc per axis (exp of the loop state in saturating elements); full=False (later
        saturation iterations) skips the anchor / external-field update and the mu_dc read."""
        if full:
            self._update_constraints()
            self._read_permeability()
            self._coil_field_cache = {}
        coefficients = self._permeability.copy()
        if self._log_permeability_state is not None:
            coefficients[self._nonlinear] = np.exp(self._log_permeability_state)[:, None]
        self.system.assemble(coefficient_tensor_3d(*coefficients.T))
        self._assembled = coefficients

    def _update_constraints(self):
        """Re-impose the external field (wins shared dofs) and hold the window node nearest the window's lower corner
        at u = 0 when nothing else anchors (no symmetry face, held element or applied field)."""
        system = self.system
        if self._external is not None:
            system.set_fixed_dofs(self._external_handle, *self._external)
        needed = (self._external is None and not self._held.any()
                  and all(self.bcs[face].kind == 'neumann' for face in FACES_3D))
        if self._anchor_handle is None and not needed:
            return
        if self._anchor_dof is None:
            mesh = self.domain.mesh
            nodes = np.unique(mesh.tets[system.window_elements])
            distance = np.linalg.norm(mesh.points[nodes] - np.asarray(mesh.window[:3]), axis=1)
            self._anchor_dof = int(nodes[np.argmin(distance)])
        ids = [self._anchor_dof] if needed else []
        if self._anchor_handle is None:
            self._anchor_handle = system.add_fixed_dofs(ids, 0.0)
        else:
            system.set_fixed_dofs(self._anchor_handle, ids, 0.0)

    def _read_permeability(self):
        """Host float64 (n_elements, 3) mu_dc, checked finite and > 0 on the window elements."""
        materials = self.domain.materials
        permeability = np.stack([np.asarray(to_host(getattr(materials, name)), dtype=np.float64)
                                 for name in _PERMEABILITY_NAMES], axis=1)
        window = permeability[self.system.window_elements]
        if not (np.isfinite(window).all() and (window > 0.0).all()):
            raise ValueError(f"{self.type_label} '{self.name}': mu_dc_xx / mu_dc_yy / mu_dc_zz must be finite and > 0 "
                             'in every window element')
        self._permeability = permeability

    def _host_material(self, name):
        """Host float64 per-element material array, None when no material allocated it."""
        materials = self.domain.materials
        if name not in materials.property_names():
            return None
        return np.asarray(to_host(getattr(materials, name)), dtype=np.float64)

    def _remanence(self):
        """Host float64 (n_elements, 3) B_r, None without a remanence array."""
        components = [self._host_material(name) for name in REMANENCE_3D_ARRAYS]
        if all(component is None for component in components):
            return None
        count = self.domain.mesh.n_elements
        return np.stack([np.zeros(count) if component is None else component for component in components], axis=1)

    def _refresh_saturation(self):
        """The saturating window elements (J_s > 0 or a B-H curve; None: no loop) and their J_s; a J_s element with mu_dc
        differing between axes, a curve element with J_s > 0 or a saturating element with a remanence raises; the
        compact loop state is dropped when the saturating set changes or vanishes."""
        self._saturation_stale = False
        previous_elements = self._nonlinear
        self._nonlinear = None
        self._saturation = None
        window = self.system.window_elements
        saturation = self._host_material(SATURATION_ARRAY)
        saturating = np.zeros_like(window) if saturation is None else (saturation > 0.0) & window
        if saturating.any():
            permeability = np.stack([np.asarray(to_host(getattr(self.domain.materials, name)), dtype=np.float64)
                                     for name in _PERMEABILITY_NAMES], axis=1)[saturating]
            if ((permeability[:, 1] != permeability[:, 0]) | (permeability[:, 2] != permeability[:, 0])).any():
                raise ValueError(f"{self.type_label} '{self.name}': an element with saturation_polarization > 0 has "
                                 'different mu_dc_xx / mu_dc_yy / mu_dc_zz (the saturating curve is isotropic: give '
                                 'one mu_dc)')
        tabulated = np.zeros_like(window)
        for element_mask, _ in self._bh_curves:
            tabulated |= element_mask
        refuse_curve_conflicts(f"{self.type_label} '{self.name}'", 'element',
                               curve_and_polarization=bool((tabulated & saturating).any()))
        nonlinear = saturating | tabulated
        if not nonlinear.any():
            self._log_permeability_state = None   # back to the linear mu_dc (_assemble reads the state alone)
            return
        remanence = self._remanence()
        refuse_curve_conflicts(f"{self.type_label} '{self.name}'", 'element', saturating_remanence=(
            remanence is not None and bool((nonlinear & (remanence != 0.0).any(axis=1)).any())))
        self._nonlinear = np.flatnonzero(nonlinear)
        if previous_elements is None or not np.array_equal(self._nonlinear, previous_elements):
            self._log_permeability_state = None   # the compact state indexes the old set: restart at H = 0
        self._saturation = (np.zeros(self._nonlinear.size) if saturation is None
                            else np.where(saturating, saturation, 0.0)[self._nonlinear])
        if MAGNETIC_3D_PERMEABILITY_NAME not in self._state:
            self._state[MAGNETIC_3D_PERMEABILITY_NAME] = self.xp.zeros(self.domain.mesh.n_elements, dtype=DTYPE)

    def _solve_saturated(self):
        """Fixed point on ln mu_r of the saturating elements (core.damped_iteration): each iteration assembles with mu
        = exp(state) there (the first one fully), refactors, solves and proposes ln of the curves' secant mu_r at the
        element |H| = B_rms / (mu0 mu_r), H-based for both curves; scale 1. The first state is the H = 0 limit."""
        first = [True]
        if self._log_permeability_state is None:
            self._read_permeability()
            self._log_permeability_state = self._saturated_log_permeability(np.zeros(self._nonlinear.size))
        state = self._log_permeability_state

        def propose(log_permeability):   # the state itself, read by _assemble through the attribute
            self._assemble(full=first[0])
            first[0] = False
            self._solve_once(linkages=False)   # the coil linkages of the last solve only, after the loop
            field_strength = self._values['B_rms'][self._nonlinear] / (MU0 * np.exp(log_permeability))
            return self._saturated_log_permeability(field_strength)

        if not self.picard.run(state, propose, 1.0):
            self.runtime_warnings.append(self.picard.last_warning)
        self._update_linkages()

    def _saturated_log_permeability(self, field_strength):
        """ln of the secant mu_r of the saturating elements at |H| = field_strength (J_s curve with mu_dc_xx, each B-H
        table on its elements)."""
        elements = self._nonlinear
        secant = saturation_permeability(field_strength, self._permeability[elements, 0], self._saturation, np)
        for element_mask, curve in self._bh_curves:
            secant = np.where(element_mask[elements],
                              1.0 / (MU0 * tabulated_reluctivity(field_strength, curve, np)), secant)
        return np.log(secant)

    def _coil_field(self, coil_dofs, chunk):
        """Total-T rule values (m, q, 3) [A/m] of an element chunk, kept across saturation iterations while the coil
        currents hold (up to COIL_FIELD_CACHE_BYTES; full assemblies clear it)."""
        cache = self._coil_field_cache
        currents = tuple((id(coil), coil.current) for coil in self.coils)
        if cache.get('currents') != currents:
            cache.clear()
            cache.update(currents=currents, bytes=0)
        key = (int(chunk[0]), int(chunk[-1]), int(chunk.size))   # sorted subsets of one element set: unique
        field = cache.get(key)
        if field is None:
            field, _, _ = rule_values_3d(self.edge_map, coil_dofs, chunk, RULE_DEGREE)
            if cache['bytes'] + field.nbytes <= COIL_FIELD_CACHE_BYTES:
                cache[key] = field
                cache['bytes'] += field.nbytes
        return field

    def _coil_dofs(self):
        """Nedelec-1 dofs of the total T [A] (sum of current x T per ampere: line integrals add over axes)."""
        return sum(coil.current * coil.per_ampere for coil in self.coils)

    def _solve_once(self, linkages=True):
        """One solve on the assembled operator with the drive's gradient load, then the fields and scalars (linkages
        False: the coil linkages wait for _update_linkages)."""
        self.system.solve(None, extra_load=self._drive_load())
        self._update_fields(linkages)

    def _drive_load(self):
        """int grad v . (mu0 mu T + B_r) dV over the active elements (None without a coil or remanence)."""
        remanence = self._remanence()
        if not self.coils and remanence is None:
            return None
        system = self.system
        elements = np.flatnonzero(system.active)
        if not self.coils:   # B_r constant per element
            return assemble_lagrange_gradient_load_3d(system.full, remanence, elements=elements, degree=RULE_DEGREE)
        coil_dofs = self._coil_dofs()
        load = np.zeros(system.full.n_dofs)
        for start in range(0, elements.size, POST_CHUNK):
            chunk = elements[start:start + POST_CHUNK]
            coil_field = self._coil_field(coil_dofs, chunk)
            vector = MU0 * self._assembled[chunk][:, None, :] * coil_field
            if remanence is not None:
                vector = vector + remanence[chunk][:, None, :]
            load += assemble_lagrange_gradient_load_3d(system.full, vector, elements=chunk, degree=RULE_DEGREE)
        return load

    def _chunk_flux(self, chunk, coil_dofs, remanence) -> tuple:
        """(B (m, q, 3) = mu (mu0 T - grad u) + B_r, rule weights (m, q), mu (m, 3), B_r (m, 3)) of the last solve on
        an element chunk."""
        system = self.system
        _, gradient, weights = rule_values_3d(system.full, system.values, chunk, RULE_DEGREE)
        mu = self._assembled[chunk]   # (m, 3)
        flux = -mu[:, None, :] * gradient
        if coil_dofs is not None:
            flux += MU0 * mu[:, None, :] * self._coil_field(coil_dofs, chunk)
        residual = np.zeros((chunk.size, 3)) if remanence is None else remanence[chunk]
        flux += residual[:, None, :]
        return flux, weights, mu, residual

    def _field_chunks(self):
        """Yield (chunk, B, weights, mu, B_r) per POST_CHUNK block of the free elements (active, not held)."""
        remanence = self._remanence()
        coil_dofs = self._coil_dofs() if self.coils else None
        elements = np.flatnonzero(self.system.active & ~self._held)
        for start in range(0, elements.size, POST_CHUNK):
            chunk = elements[start:start + POST_CHUNK]
            yield (chunk, *self._chunk_flux(chunk, coil_dofs, remanence))

    def _coil_linkages(self, chunk, flux, weights, linkages) -> None:
        """linkages[coil] += int T / I . B dV over the chunk (one Nedelec evaluation per coil)."""
        for coil in self.coils:
            per_ampere, _, _ = rule_values_3d(self.edge_map, coil.per_ampere, chunk, RULE_DEGREE)
            linkages[coil.name] += float(np.einsum('mqc,mqc,mq->', per_ampere, flux, weights))

    def _update_linkages(self) -> None:
        """Each coil's linkage int T / I . B dV of the last solve (after a saturation loop that skipped them)."""
        linkages = {coil.name: 0.0 for coil in self.coils}
        if self.coils:
            for chunk, flux, weights, _, _ in self._field_chunks():
                self._coil_linkages(chunk, flux, weights, linkages)
        self._linkages = linkages

    def _update_fields(self, linkages=True):
        """Per free element at the rule points: B = mu (mu0 T - grad u) + B_r; element means of B, H = (B - B_r) / (mu0
        mu), w_mag = B^2 / (2 mu0 mu), B_abs = |mean B|, B_rms; the energy and (linkages) each coil's linkage int T /
        I . B dV; phi_m at the centroids; 0 in held / shell elements. Copies the float32 state."""
        system = self.system
        count = self.domain.mesh.n_elements
        values = {name: np.zeros(count) for name in (*_ELEMENT_FIELDS, 'B_rms')}
        energy = 0.0
        coil_linkages = {coil.name: 0.0 for coil in self.coils}
        for chunk, flux, weights, mu, residual in self._field_chunks():
            volume = weights.sum(axis=1)
            mean_flux = np.einsum('mqc,mq->mc', flux, weights) / volume[:, None]
            field_strength = (mean_flux - residual) / (MU0 * mu)
            for axis, name in enumerate(_AXES):
                values[f'B_{name}'][chunk] = mean_flux[:, axis]
                values[f'H_{name}'][chunk] = field_strength[:, axis]
            values['B_abs'][chunk] = np.linalg.norm(mean_flux, axis=1)
            # B^2 / (2 mu0 mu): the density of the energy scalar below (as 2D; B . H / 2 differs inside magnets)
            values['w_mag'][chunk] = np.einsum('mqc,mqc,mc,mq->m', flux, flux, 0.5 / (MU0 * mu), weights) / volume
            values['B_rms'][chunk] = np.sqrt(np.einsum('mqc,mqc,mq->m', flux, flux, weights) / volume)
            energy += float(np.einsum('mqc,mqc,mc,mq->', flux, flux, 0.5 / (MU0 * mu), weights))
            if linkages:
                self._coil_linkages(chunk, flux, weights, coil_linkages)
        current_density = {name: self._values[name] for name in MAGNETIC_3D_CURRENT_NAMES if name in self._values}
        self._values = {**values, **current_density}
        self._energy = energy
        self._linkages = coil_linkages   # zeros until _update_linkages when skipped
        xp = self.xp
        self._state['phi_m'][...] = xp.asarray(system.centroid_values() / MU0, dtype=DTYPE)
        for name in _ELEMENT_FIELDS:
            self._state[name][...] = xp.asarray(values[name], dtype=DTYPE)
        if MAGNETIC_3D_PERMEABILITY_NAME in self._state:
            self._state[MAGNETIC_3D_PERMEABILITY_NAME][...] = xp.asarray(self._assembled.mean(axis=1), dtype=DTYPE)

    def _write_current_density(self):
        """J = curl T [A/m^2] per window element (constant on Nedelec-1), host float64 and float32 state."""
        system = self.system
        count = self.domain.mesh.n_elements
        density = np.zeros((count, 3))
        elements = np.flatnonzero(system.window_elements)
        coil_dofs = self._coil_dofs()
        for start in range(0, elements.size, POST_CHUNK):
            chunk = elements[start:start + POST_CHUNK]
            _, curls, _ = rule_values_3d(self.edge_map, coil_dofs, chunk, RULE_DEGREE)
            density[chunk] = curls[:, 0, :]
        xp = self.xp
        for axis, name in enumerate(MAGNETIC_3D_CURRENT_NAMES):
            self._values[name] = density[:, axis]
            self._state[name] = xp.asarray(density[:, axis], dtype=DTYPE)

    def _record_probes(self, time, replace_same_time=False) -> None:
        """B_abs = |mu (mu0 T - grad u) + B_r| at the point (0 in a held element) or phi_m = u / mu0."""
        if not self.probes:
            return
        system = self.system
        remanence = self._remanence()
        coil_dofs = self._coil_dofs() if self.coils else None
        for probe, element, reference in self.probes:
            elements, point = np.array([element]), reference[None, :]
            potential, gradient = evaluate_lagrange_3d(system.full, system.values, elements, point)
            if probe.field_component == 'phi_m':
                value = potential[0] / MU0
            elif self._held[element] or self._assembled is None:
                value = 0.0
            else:
                mu = self._assembled[element]
                flux = -mu * gradient[0]
                if coil_dofs is not None:
                    flux = flux + MU0 * mu * evaluate_nedelec_3d(self.edge_map, coil_dofs, elements, point)[0][0]
                if remanence is not None:
                    flux = flux + remanence[element]
                value = float(np.linalg.norm(flux))
            probe.record_value(np.float32(value), time, replace_same_time)
