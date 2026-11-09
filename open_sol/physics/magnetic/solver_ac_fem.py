"""magnetic_fields_ac on a triangle mesh (MagneticACFEMPhysics): complex Lagrange P1 / P2 A (planar A_z, axisymmetric
A_phi) of curl(nu curl A) + j omega sigma A = J in B-space on the window elements, on the curl quadrature and the
constraints of system_fem.MagneticFEMSystem; solid conductors as bordered circuit rows (circuit.py), one complex host
factor per operator through the 'linear' sparse_direct role. State per element (D12), display by fem_field."""
import math
from dataclasses import dataclass

import numpy as np

from ...core.backend import to_host
from ...core.constants import DTYPE, MU0
from ...fem.diffusion import floating_region_count
from ...fem.elements import lagrange_basis
from ...fem.indicators import lagrange_at
from ...fem.magnetostatics import (assemble_curl_tangent, block_load, block_mass, element_integrals,
                                   quadrature_flux_density, quadrature_integral, quadrature_values)
from ...fem.mesher import MERGE_TOLERANCE
from ...fem.renderer import FEMFieldRenderer
from ...solvers.sparse_direct import CUDSS_BACK_ENDS, SparseDirectSolver
from .circuit import (COIL_RESULT_KEYS, ConductorRow, RecycledFactor, bordered_system,
                      coil_drive, coil_result_values, stranded_resistance, triangle_size_issue)
from .fields import AC_FIELD_NAMES
from .materials import CONDUCTIVITY_ARRAYS, MAGNETIC_AC_SCHEMA, PERMEABILITY_ARRAYS
from .renderer import MAGNETIC_AC_FIELD_INFOS
from .solver_fem import DISCRETIZATIONS, ElementFields, TriangleMagneticBase, _host
from .system_fem import CENTROID, MagneticFEMSystem

NODAL_FIELDS = ('A_re', 'A_im', 'A_abs', 'B_x_re', 'B_x_im', 'B_y_re', 'B_y_im', 'B_abs')   # fem_field: nodal values
_ANCHOR_TEXT = 'magnetic insulation face, held potential or external field'


@dataclass(frozen=True, eq=False)
class FEMACCoilRegion:
    """Runtime object of an AC coil on triangles: host bool window elements, 'stranded' / 'solid', drive, complex value
    (current [A]; voltage V/m planar, V axisymmetric), turns, fill factor and cross-section area (r-z area) [m^2]."""
    name: str
    elements: np.ndarray
    conductor: str
    drive: str
    value: complex
    turns: int
    fill_factor: float
    area: float


@dataclass(frozen=True, eq=False)
class _FEMHostSystem:
    """Host arrays of one assembly: free / fixed window dofs and fixed values, the eliminated complex matrix and the
    lifting of the fixed values, conductor rows with their full window load g (current evaluation), per-element
    complex reluctivities, conductivities (raw and eddy: 0 in held and stranded elements) and the held mask."""
    free: np.ndarray
    fixed: np.ndarray
    fixed_values: np.ndarray
    matrix_free: object
    lift: np.ndarray
    conductors: tuple
    conductor_loads: dict
    nu_xx: np.ndarray
    nu_yy: np.ndarray
    sigma: np.ndarray
    sigma_eddy: np.ndarray
    held: np.ndarray


class MagneticACFEMPhysics(TriangleMagneticBase):
    """curl(nu curl A) + j omega sigma A = J at one frequency on a fem.domain.TriangleDomain (window elements; the PML
    layer ring holds the window field at the clamped position): host complex factor of the eliminated, bordered
    system, GMRES on the old factor after an operator change; per unit z length or whole rings (x 2 pi)."""

    type_label = 'magnetic_fields_ac'
    material_schema = dict(MAGNETIC_AC_SCHEMA)
    _probe_component = 'A_abs'

    def __init__(self, domain, frequency: float, axisymmetric: bool = False, discretization: str = 'quadratic',
                 host_solver=None):
        if getattr(domain, 'grid_kind', 'tensor') != 'triangles':
            raise ValueError(f'{self.type_label}: MagneticACFEMPhysics needs a triangle mesh (fem TriangleDomain); a '
                             'tensor grid uses MagneticACPhysics')
        if not (math.isfinite(frequency) and frequency > 0.0):
            raise ValueError(f'{self.type_label} needs a finite frequency > 0 Hz, got {frequency}')
        if discretization not in DISCRETIZATIONS:
            raise ValueError(f"discretization must be one of {', '.join(DISCRETIZATIONS)}, got {discretization!r}")
        self.frequency = float(frequency)
        self.omega = 2.0 * math.pi * self.frequency
        self._axisymmetric = bool(axisymmetric)
        self.order = DISCRETIZATIONS[discretization]
        self._owns_host_solver = host_solver is None
        self.host_solver = SparseDirectSolver(backend=domain.backend.name) if host_solver is None else host_solver
        self._factor = RecycledFactor(self.host_solver)
        self.system = None
        self.prepare(domain)

    # --- PhysicsSolver contract ---------------------------------------------
    def prepare(self, domain) -> None:
        """Register the permeability / conductivity arrays, allocate the constraint system and the per-element fields
        (clears faces, held regions, sources, coils and probes); assembly is deferred."""
        self.domain = domain
        self.xp = domain.xp
        domain.materials.add_properties(self.material_schema)
        mesh = domain.mesh
        # constraints and curl quadrature only: its real 'linear' wrapper is never factorised
        self.system = MagneticFEMSystem(mesh, self.order, self._axisymmetric, self._label, self.host_solver,
                                        backend=domain.backend.name, anchor_text=_ANCHOR_TEXT)
        self._measure = self.system.element_measure   # int dV per element (per radian when axisymmetric)
        self._area = self.system.element_area         # plain x-y / r-z area
        x0, y0, x1, y1 = mesh.window
        self._axis_tolerance = MERGE_TOLERANCE * max(x1 - x0, y1 - y0)
        self.fields = ElementFields(AC_FIELD_NAMES, mesh.n_elements, self.xp)
        self.coils = []
        self.probes = []             # (FieldProbe, node)
        self.values = np.zeros(self.system.full.n_dofs, dtype=np.complex128)   # full-mesh A of the last solve
        self._potential = None       # window dof values of the last solve
        self._host = None
        self._host_dirty = True
        self._source_dirty = True
        self._operator_revision = 0
        self._conductor_drives = {}
        self._nodal_flux = None      # full-mesh (n_dofs, 2) complex nodal B
        self._element_potential_integrals = None   # int A dV per element (linkage)
        self._power = 0.0            # int B* . nu B dV of the active elements (per radian when axisymmetric)
        self._eddy_power = 0.0       # int q_eddy dV (per radian)
        self._n_harmonic_solves = 0
        self._kept_curl = None       # (nu_xx, nu_yy, free dofs, curl CSR) of the last host build

    def finalize(self) -> None:
        """Assemble the host system early: a non-positive permeability, a floating region or an oversized system fails
        before the first solve."""
        self._build_host()

    def materials_changed(self, names) -> None:
        if set(names) & (PERMEABILITY_ARRAYS | CONDUCTIVITY_ARRAYS | {'mu_dc_loss'}):
            self._host_dirty = True

    def step(self, t: float = 0.0, dt: float = math.inf) -> None:
        """Re-solve only when materials, constraints or a source changed (always at the first call); probes sample after
        a re-solve and at every dt = inf call."""
        t = 0.0 if t is None else float(t)
        re_solve = (self._host_dirty or self.system.operator_dirty or self._source_dirty
                    or self._n_harmonic_solves == 0)
        if re_solve:
            if self._host_dirty or self.system.operator_dirty:
                self._build_host()
            self._solve()
            self._publish()
            self._source_dirty = False
        if re_solve or dt is None or not math.isfinite(dt):
            for probe, node in self.probes:
                probe.record_value(np.float32(abs(self.values[node])), t, t > 0.0)

    def dt_limit(self) -> float:
        return math.inf

    def state(self) -> dict:
        state = {name: getattr(self.fields, name) for name in AC_FIELD_NAMES}
        for array_name in self.material_schema:
            state[array_name] = getattr(self.domain.materials, array_name)
        return state

    def renderer(self):
        return FEMFieldRenderer(self, 'magnetic', MAGNETIC_AC_FIELD_INFOS,
                                lambda: f'f = {self.frequency:.4g} Hz, solve {self.n_solves}')

    # --- properties and records -------------------------------------------------
    @property
    def n_solves(self) -> int:
        return self._n_harmonic_solves

    @property
    def host_factor_bytes(self):
        """Host L + U of the last factorisation (Simulation.memory_report), None before one."""
        return self._factor.factor_bytes

    def _ring(self) -> float:
        return 2.0 * math.pi if self._axisymmetric else 1.0

    def scalars(self) -> dict:
        """energy [J/m, J], eddy_loss / magnetic_loss [W/m, W] (time averages), per coil COIL_RESULT_KEYS; {} before a
        solve."""
        if self._n_harmonic_solves == 0:
            return {}
        ring = self._ring()
        values = {'energy': 0.25 * ring * self._power.real, 'eddy_loss': ring * self._eddy_power,
                  'magnetic_loss': 0.5 * self.omega * ring * self._power.imag}
        for coil in self.coils:
            result = self.coil_results(coil)
            values.update({f'{coil.name}_{key}': result[key] for key in COIL_RESULT_KEYS})
        return values

    def work_counters(self) -> dict:
        """Solves; factorisations / recycled GMRES are the 'linear' role's record, or added here for a private
        (script) solver."""
        counters = {'solves': self._n_harmonic_solves}
        if self._owns_host_solver:
            counters.update(self.host_solver.work_counters())
        return counters

    def solve_summary(self):
        """One log line: back end, unknowns, circuit rows, solves, factorisations and recycled solves."""
        if self._host is None:
            return None
        n_rows = sum(1 for conductor in self._host.conductors if conductor.drive == 'current')
        solver = self.host_solver
        return (f'{solver.back_end} P{self.order} triangles at {self.frequency:.4g} Hz, {self._host.free.size:,} '
                f'unknowns + {n_rows} circuit rows, {self._n_harmonic_solves} solves, '
                f'{solver.n_factorizations} factorisations, {solver.n_recycled_solves} recycled '
                f'({solver.total_recycled_iterations} GMRES iterations)')

    # --- magnetics API (the definition callbacks on triangles) ---------------
    def set_external_field(self, B_x: float, B_y: float) -> None:
        """Uniform applied flux density amplitude [T] at phase 0 on the window-edge dofs (as magnetostatics)."""
        self.system.set_external_field(B_x, B_y)

    def add_fixed_potential(self, mask, value) -> None:
        """Hold planar A_z [T m] / axisymmetric psi = r A_phi [T m^2] (phase 0) on every dof of the masked elements; no
        eddy current flows there."""
        self.system.add_held(self._element_mask(mask, 'fixed potential'), float(value))
        self._host_dirty = True

    def add_source(self, mask, density) -> None:
        """Add a current density amplitude J_z / J_phi [A/m^2] at phase 0 (number or per-element array) to J_applied."""
        xp = self.xp
        self.fields.J_applied_re += xp.where(xp.asarray(mask), xp.asarray(density, dtype=DTYPE), DTYPE(0.0)).astype(
            DTYPE, copy=False)
        self._source_dirty = True

    def add_coil(self, name: str, mask, current: float = 1.0, phase: float = 0.0, turns: int = 1,
                 conductor: str = 'stranded', drive: str = 'current', voltage: float = 0.0,
                 fill_factor: float = 1.0) -> FEMACCoilRegion:
        """Stranded coil (J = turns * I / area, I = current e^(j phase deg)) or solid conductor (circuit row: drive
        'current' prescribes I, 'voltage' the loop voltage, V/m planar, V axisymmetric), as on tensor meshes."""
        drive, value = coil_drive(name, conductor, drive, current, voltage, phase, turns, fill_factor)
        elements = self._element_mask(mask, f"coil '{name}'")
        area = float(self._area[elements].sum())
        if not area > 0.0:
            raise ValueError(f"coil '{name}' selects no window element")
        coil = FEMACCoilRegion(name, elements, conductor, drive, value, int(turns), float(fill_factor), area)
        if conductor == 'stranded':
            density = coil.turns * value / area
            self.fields.J_applied_re += self.xp.asarray(np.where(elements, density.real, 0.0), dtype=DTYPE)
            self.fields.J_applied_im += self.xp.asarray(np.where(elements, density.imag, 0.0), dtype=DTYPE)
        self.coils.append(coil)
        self._host_dirty = True   # stranded elements lose their eddy currents, solid ones add a circuit row
        self._source_dirty = True
        return coil

    def coil_results(self, coil: FEMACCoilRegion) -> dict:
        """Complex current and loop voltage (stranded: R_dc I + j omega flux linkage; solid: the circuit unknown and
        I = v G - j omega g . a), Z = V / I -> resistance, inductance, power; planar per unit length, axisymmetric
        whole rings (as MagneticACPhysics.coil_results)."""
        if self._n_harmonic_solves == 0:
            raise RuntimeError(f"{self._label()}: coil '{coil.name}' needs a solve first")
        host = self._host
        ring = self._ring()
        elements = coil.elements
        if coil.conductor == 'stranded':
            current = coil.value
            linkage = coil.turns * ring * complex(self._element_potential_integrals[elements].sum()) / coil.area
            sigma = host.sigma[elements]
            path = ring * self._measure[elements]   # x-y area, or 2 pi r over the r-z area
            resistance_dc = stranded_resistance(coil.turns, coil.fill_factor, coil.area, path, sigma)
            voltage = resistance_dc * current + 1j * self.omega * linkage
        else:
            row = next(conductor for conductor in host.conductors if conductor.name == coil.name)
            drive_value = self._conductor_drives[coil.name]
            current = complex(drive_value * row.conductance
                              - 1j * self.omega * (host.conductor_loads[coil.name] @ self._potential))
            voltage = ring * drive_value
        return coil_result_values(current, voltage, self.omega, coil.conductor, coil.drive)

    def field_energy(self) -> float:
        """Time-averaged magnetic energy (1/4) Re int B* . nu B dV [J/m planar, J axisymmetric]."""
        return self.scalars().get('energy', math.nan)

    def warm_start_fields(self) -> dict:
        """Host copy of the coupled source (the direct solve needs no starting unknown)."""
        return {'source': np.array(to_host(self.fields.J_coupled))}

    def seed_warm_start(self, fields) -> tuple:
        self.xp.copyto(self.fields.J_coupled, self.xp.asarray(fields['source'], dtype=DTYPE))
        self._source_dirty = True
        return ('J_coupled',)

    # --- display fields -----------------------------------------------------------
    def fem_field(self, component) -> tuple:
        """(DofMap, host coefficients): A and B (re / im, magnitudes) nodal (B averaged to the nodes), the rest per
        element."""
        system = self.system
        if component in NODAL_FIELDS:
            if component.startswith('A_'):
                values = self.values
            elif self._nodal_flux is None:
                return system.full, np.zeros(system.full.n_dofs)
            elif component == 'B_abs':
                return system.full, np.sqrt(np.sum(np.abs(self._nodal_flux) ** 2, axis=1))
            else:
                values = self._nodal_flux[:, 0 if component.startswith('B_x') else 1]
            part = component.rsplit('_', 1)[1]
            return system.full, {'re': values.real, 'im': values.imag, 'abs': np.abs(values)}[part]
        if component in AC_FIELD_NAMES:
            return system.element_map, _host(getattr(self.fields, component))
        if component in self.material_schema:
            return system.element_map, _host(getattr(self.domain.materials, component))
        raise KeyError(f"{self._label()} has no field '{component}'")

    def error_indicator(self):
        """eta^2 per element (float64 (n_elements,)) of the last solve for mesh refinement (S7.7): Kelly jumps h_E /
        |nu| int_E |[H . t]|^2 ds of the complex tangential H over edges between window elements (1/2 per side) and on
        symmetry faces (one-sided); r weights axisymmetric."""
        if self._n_harmonic_solves == 0:
            raise ValueError(f'{self._label()}: the error indicator needs a solve')
        host = self._host
        return self._kelly_indicator(np.where(self.system.window_elements,
                                              2.0 / (np.abs(host.nu_xx) + np.abs(host.nu_yy)), 0.0))

    # --- internals ------------------------------------------------------------------
    def _edge_tangential_field(self, sides, side, rows):
        """(k, q) complex H . t (t = the normal out of side 0 turned by +90 deg) at the edge points of `rows` from the
        element on `side`."""
        host = self._host
        elements = sides.elements[rows, side]
        values, gradients = lagrange_at(self.system.full, self.values, elements, sides.reference_points(side, rows))
        if self._axisymmetric:
            flux_x, flux_y = -gradients[..., 1], gradients[..., 0] + values / sides.points[rows][..., 0]
        else:
            flux_x, flux_y = gradients[..., 1], -gradients[..., 0]
        normal = sides.normals[rows]
        return (host.nu_yy[elements][:, None] * flux_y * normal[..., 0]
                - host.nu_xx[elements][:, None] * flux_x * normal[..., 1])

    def _complex_reluctivities(self):
        """Per-element (nu_xx, nu_yy) = 1 / (mu0 (mu_dc - j mu_dc_loss)) of the window (1 / mu0 off it)."""
        materials = self.domain.materials
        window = self.system.window_elements
        mu_xx, mu_yy = _host(materials.mu_dc_xx), _host(materials.mu_dc_yy)
        valid = np.isfinite(mu_xx) & np.isfinite(mu_yy) & (mu_xx > 0.0) & (mu_yy > 0.0)
        if (window & ~valid).any():
            raise ValueError(f'{self._label()}: mu_dc_xx / mu_dc_yy must be finite and > 0 in every window element')
        loss = np.zeros(mu_xx.size)
        if 'mu_dc_loss' in materials.property_names():
            loss = _host(materials.mu_dc_loss)
            if not np.all(np.isfinite(loss[window]) & (loss[window] >= 0.0)):
                raise ValueError(f'{self._label()}: mu_dc_loss must be finite and >= 0 in every window element')
        mu_xx, mu_yy, loss = (np.where(window, value, default) for value, default in ((mu_xx, 1.0), (mu_yy, 1.0),
                                                                                      (loss, 0.0)))
        return 1.0 / (MU0 * (mu_xx - 1j * loss)), 1.0 / (MU0 * (mu_yy - 1j * loss))

    def _conductivity(self):
        """Per-element sigma_dc seen by the out-of-plane current (isotropic in-plane components required), 0 off the
        window."""
        materials = self.domain.materials
        window = self.system.window_elements
        sigma_xx, sigma_yy = _host(materials.sigma_dc_xx), _host(materials.sigma_dc_yy)
        if not np.all(np.isfinite(sigma_xx[window]) & (sigma_xx[window] >= 0.0)):
            raise ValueError(f"{self._label()}: sigma_dc must be finite and >= 0 in every window element")
        if not np.array_equal(sigma_xx[window], sigma_yy[window]):
            raise NotImplementedError(f'{self._label()}: the eddy current flows out of the plane, so it needs '
                                      'sigma_dc_xx = sigma_dc_yy (an anisotropic sigma_dc has no defined out-of-plane '
                                      'value here): use the isotropic sigma_dc')
        return np.where(window, sigma_xx, 0.0)

    def _point_values(self, per_element, radius_power=0):
        """Per block (m, q) point values of a per-element array, divided by r^radius_power at the points."""
        return [np.broadcast_to(per_element[block.elements][:, None], block.weights.shape)
                / block.points[..., 0] ** radius_power for block in self.system.blocks]

    def _build_host(self):
        """Complex operator K(nu) + j omega M(sigma_eddy) on the curl quadrature, the fixed-dof elimination and the
        solid-conductor rows (g = int sigma phi / r dV, G = int sigma / r dA)."""
        system = self.system
        free, fixed, fixed_values = system.constraints()
        n_dofs = system.dofs.n_dofs
        nu_xx, nu_yy = self._complex_reluctivities()
        sigma = self._conductivity()
        held = system.held_elements()
        stranded = np.zeros(held.size, dtype=bool)
        for coil in self.coils:
            if coil.conductor == 'stranded':
                stranded |= coil.elements
        sigma_eddy = np.where(held | stranded, 0.0, sigma)
        n_rows = sum(1 for coil in self.coils if coil.conductor == 'solid' and coil.drive == 'current')
        on_cudss = self.host_solver.back_end in CUDSS_BACK_ENDS   # cuDSS: its own estimate after planning guards it
        issue = None if on_cudss else triangle_size_issue(free.size + n_rows, self.order, self.host_solver.back_end)
        if issue is not None and issue[0] == 'error':
            raise ValueError(f'{self._label()}: {issue[1]}')
        kept = self._kept_curl   # (nu_xx, nu_yy, free, curl): a sigma-only change (sigma(T) coupling) reuses it
        if (kept is not None and np.array_equal(kept[0], nu_xx) and np.array_equal(kept[1], nu_yy)
                and np.array_equal(kept[2], free)):
            curl = kept[3]
        else:
            tangents = []
            for block in system.blocks:
                tangent = np.zeros(block.weights.shape + (2, 2), dtype=np.complex128)
                tangent[..., 0, 0] = nu_xx[block.elements][:, None]
                tangent[..., 1, 1] = nu_yy[block.elements][:, None]
                tangents.append(tangent)
            curl = assemble_curl_tangent(system.blocks, tangents, n_dofs).tocsr()
            if not self._axisymmetric:   # the A / r term anchors every axisymmetric region
                floating = floating_region_count(curl.real[free][:, free].tocsr())
                if floating:
                    raise ValueError(f'{self._label()}: {floating} connected region(s) of the vector potential touch '
                                     f'no {_ANCHOR_TEXT}, so the system is singular. Keep magnetic insulation (the '
                                     'default) on at least one face, or add an external field.')
            self._kept_curl = (np.array(nu_xx), np.array(nu_yy), np.array(free), curl)
        operator = (curl + 1j * self.omega * block_mass(system.blocks, sigma_eddy, n_dofs)).tocsr()
        rows = operator[free]
        conductors, loads = [], {}
        for coil in self.coils:
            if coil.conductor != 'solid':
                continue
            conducting = np.where(coil.elements, sigma_eddy, 0.0)
            if not conducting.any():
                raise ValueError(f"{self._label()}: solid conductor '{coil.name}' has no conducting free element "
                                 '(sigma_dc > 0)')
            radius_power = int(self._axisymmetric)   # E = v / r - j omega A_phi axisymmetric
            if self._axisymmetric:
                corners = self.domain.mesh.points[self.domain.mesh.triangles[conducting > 0.0]][..., 0]
                if (corners <= self._axis_tolerance).any():
                    raise NotImplementedError(
                        f"{self._label()}: solid conductor '{coil.name}' touches the axis, where its applied field v / "
                        'r is singular (no finite loop voltage drives a ring of zero radius): keep the conductor off '
                        'the axis or make it a stranded coil')
            load = block_load(system.blocks, self._point_values(conducting, radius_power), n_dofs)
            conductance = quadrature_integral(system.blocks, self._point_values(conducting, 2 * radius_power))
            free_load = load[free]
            positions = np.flatnonzero(free_load != 0.0)
            value = coil.value if coil.drive == 'current' or not self._axisymmetric else coil.value / (2.0 * math.pi)
            conductors.append(ConductorRow(coil.name, positions, free_load[positions], coil.drive, value,
                                           float(conductance), complex(load[fixed] @ fixed_values)))
            loads[coil.name] = load
        self._host = _FEMHostSystem(free, fixed, fixed_values, rows[:, free].tocsr(), rows[:, fixed] @ fixed_values,
                                    tuple(conductors), loads, nu_xx, nu_yy, sigma, sigma_eddy, held)
        self._operator_revision += 1
        self._host_dirty = False

    def _solve(self):
        """One complex solve of the bordered system (RecycledFactor); stores the window dof values, the full-mesh A
        and the solid conductors' v_k."""
        host = self._host
        system = self.system
        density = np.where(system.window_elements,
                           _host(self.fields.J_coupled) + _host(self.fields.J_applied_re)
                           + 1j * _host(self.fields.J_applied_im), 0.0)
        loads = block_load(system.blocks, self._point_values(density), system.dofs.n_dofs)
        matrix, rhs = bordered_system(host.matrix_free, loads[host.free] - host.lift, host.conductors, self.omega)
        solution = self._factor.solve(matrix, rhs, self._operator_revision)
        n_free = host.free.size
        potential = np.empty(system.dofs.n_dofs, dtype=np.complex128)
        potential[host.fixed] = host.fixed_values
        potential[host.free] = solution[:n_free]
        border = iter(solution[n_free:])
        self._conductor_drives = {conductor.name: complex(next(border)) if conductor.drive == 'current'
                                  else conductor.value for conductor in host.conductors}
        self._potential = potential
        self.values = system.full_values(potential)
        self._n_harmonic_solves += 1

    def _publish(self):
        """Per-element phasor state (A at the centroid, element means of B, E, the loss and energy densities), the
        power / loss integrals and the nodal B of the display."""
        host = self._host
        system = self.system
        blocks = system.blocks
        count = self.domain.mesh.n_elements
        window = system.window_elements
        active = np.where(window & ~host.held, 1.0, 0.0)   # held elements: B of the interpolated psi / r only
        drive = np.zeros(count, dtype=np.complex128)
        for coil in self.coils:
            if coil.name in self._conductor_drives:
                drive[coil.elements] = self._conductor_drives[coil.name]
        flux = quadrature_flux_density(blocks, self._potential)
        point_potential = quadrature_values(blocks, self._potential)
        applied = self._point_values(drive, int(self._axisymmetric))
        field_E = [field - 1j * self.omega * value for field, value in zip(applied, point_potential)]
        q_eddy, q_magnetic, w_mag, power = [], [], [], []
        for block, block_flux, block_E in zip(blocks, flux, field_E):
            elements = block.elements
            weight = active[elements][:, None]
            flux_x2, flux_y2 = np.abs(block_flux[..., 0]) ** 2, np.abs(block_flux[..., 1]) ** 2
            nu_xx, nu_yy = host.nu_xx[elements][:, None], host.nu_yy[elements][:, None]
            q_eddy.append(0.5 * host.sigma_eddy[elements][:, None] * np.abs(block_E) ** 2)
            q_magnetic.append(0.5 * self.omega * weight * (nu_xx.imag * flux_x2 + nu_yy.imag * flux_y2))
            w_mag.append(0.25 * weight * (nu_xx.real * flux_x2 + nu_yy.real * flux_y2))
            power.append(weight * (nu_xx * flux_x2 + nu_yy * flux_y2))
        self._power = complex(quadrature_integral(blocks, power))
        self._eddy_power = quadrature_integral(blocks, q_eddy)
        self._element_potential_integrals = element_integrals(blocks, point_potential, count)
        safe = np.where(self._measure > 0.0, self._measure, 1.0)
        mean_flux = element_integrals(blocks, flux, count) / safe[:, None]
        mean_flux *= active[:, None]
        J_conduction = host.sigma_eddy * element_integrals(blocks, field_E, count) / safe
        centroid = np.where(window, self.values[system.full.element_dofs] @ lagrange_basis(self.order, CENTROID)[0][0],
                            0.0)
        J_total = (_host(self.fields.J_coupled) + _host(self.fields.J_applied_re) + 1j * _host(self.fields.J_applied_im)
                   + J_conduction)
        values = {'A_re': centroid.real, 'A_im': centroid.imag, 'A_abs': np.abs(centroid),
                  'B_x_re': mean_flux[:, 0].real, 'B_x_im': mean_flux[:, 0].imag,
                  'B_y_re': mean_flux[:, 1].real, 'B_y_im': mean_flux[:, 1].imag,
                  'B_abs': np.sqrt(np.sum(np.abs(mean_flux) ** 2, axis=1)),
                  'J_conduction_re': J_conduction.real, 'J_conduction_im': J_conduction.imag,
                  'J_total_re': J_total.real, 'J_total_im': J_total.imag, 'J_total_abs': np.abs(J_total),
                  'q_eddy': element_integrals(blocks, q_eddy, count) / safe,
                  'q_magnetic': element_integrals(blocks, q_magnetic, count) / safe,
                  'w_mag': element_integrals(blocks, w_mag, count) / safe}
        xp = self.xp
        for name, array in values.items():
            xp.copyto(getattr(self.fields, name), xp.asarray(array, dtype=DTYPE))
        self._nodal_flux = system.nodal_flux_density(self._potential)
