"""Time-harmonic (eddy-current) 2D magnetic fields: MagneticPhysics' window, held cells, external-field ring and
anchor check, solved on the host as one complex SuperLU system K(nu) + j omega sigma A / r (+ solid-conductor rows);
exp(+j omega t) phasors, peak amplitudes. Planar A_z or axisymmetric psi = r A_phi on the planar operator. Fields on
the Domain's xp (gpu or cpu backend)."""
import math
from dataclasses import dataclass

import numpy as np

from ...core.backend import to_host

from ...core.constants import DTYPE, FACES, MU0
from ...core.fv_diffusion import edge_extension_views
from ...core.gpu_memory import format_bytes
from ...solvers.host_direct import HOST_DIRECT_BACK_ENDS, factor_label
from ...solvers.sparse_direct import CUDSS_BACK_ENDS, SparseDirectSolver, size_issue
from .ac_operator import HostOperator, assemble_operator, face_fluxes_at_centres, face_power
from .circuit import (COIL_RESULT_KEYS, ConductorRow, RecycledFactor, bordered_system,
                      coil_drive, coil_result_values, stranded_resistance)
from .fields import AC_FIELD_NAMES, MagneticFieldsAC2D
from .materials import CONDUCTIVITY_ARRAYS, MAGNETIC_AC_SCHEMA, PERMEABILITY_ARRAYS
from .renderer import MagneticACRenderer
from .solver import MagneticPhysics

_UNEXTENDED_FIELDS = frozenset({'A_re', 'J_coupled', 'J_applied_re', 'J_applied_im'})   # A_re: DiffusionPhysicsBase


@dataclass(frozen=True, eq=False)
class ACCoilRegion:
    """Runtime object of an AC coil: host bool window mask, 'stranded' (J = turns I / area, no eddy currents in its
    cells) or 'solid' (one circuit row), drive, complex drive value (current [A]; voltage V/m planar, V axisymmetric),
    turns, fill factor (stranded resistance) and cross-section area [m^2]."""
    name: str
    mask_window: np.ndarray
    conductor: str
    drive: str
    value: complex
    turns: int
    fill_factor: float
    area: float


class _AssemblyOnlySolver:
    """Linear solver stand-in of the real DiffusionSystem assembly (anchor check, held cells): float64, never solves,
    so no PCG workspace or device CSR product is built for the host complex solve."""

    precision = 'double'
    tolerance, max_iterations = 1e-8, 1
    n_solves, last_iters, last_converged, last_residual, last_warning, n_unconverged = 0, 0, True, 0.0, None, 0

    def __init__(self, backend):
        self.backend = backend
        self.label = lambda: 'magnetic_fields_ac'
        self.compute_residual = False

    def factorize(self, operator, inverse_diagonal) -> None:
        """Nothing to factorise: the host SuperLU of the complex system is the solve."""

    def solve(self, rhs, solution, reference) -> None:
        raise RuntimeError(f'{self.label()}: the real assembly is never solved (host complex solve only)')


@dataclass(frozen=True, eq=False)
class _HostSystem:
    """Host arrays of one assembly: operator, held mask / values, free / held flat indices, eliminated matrix and its
    held + boundary drive, conductor rows, reluctivities, conductivities (raw and eddy: 0 in held and stranded cells),
    r_c column (1 planar) and cell areas dx dy."""
    operator: object
    held: np.ndarray
    held_index: np.ndarray
    held_values: np.ndarray
    free_index: np.ndarray
    matrix_free: object
    held_drive: np.ndarray
    conductors: tuple
    nu_xx: np.ndarray
    nu_yy: np.ndarray
    sigma: np.ndarray
    sigma_eddy: np.ndarray
    radius: np.ndarray
    cell_area: np.ndarray


class MagneticACPhysics(MagneticPhysics):
    """curl(nu curl A) + j omega sigma A = J at one frequency (magnetic_fields_ac): host SuperLU of the eliminated window
    system, refactorised only when a recycled-factor GMRES misses (operator change); fields copied to the run backend."""

    type_label = 'magnetic_fields_ac'
    _base_material_schema = MAGNETIC_AC_SCHEMA
    _probe_field = 'A_abs'
    _probe_label = 'vector potential amplitude probe'

    def __init__(self, domain, frequency: float, axisymmetric: bool = False, host_solver=None):
        if not (math.isfinite(frequency) and frequency > 0.0):
            raise ValueError(f'magnetic_fields_ac needs a finite frequency > 0 Hz, got {frequency}')
        self.frequency = float(frequency)
        self.omega = 2.0 * math.pi * self.frequency
        # the build's 'linear' sparse_direct solver (ctx.solvers; host factorisations only), else a private one whose
        # counters work_counters reports
        self._owns_host_solver = host_solver is None
        self.host_solver = SparseDirectSolver(backend=domain.backend.name) if host_solver is None else host_solver
        self._factor = RecycledFactor(self.host_solver)
        self._operator_revision = 0
        self._host_dirty = True
        self._host = None
        self._solution = None            # host complex window unknown of the last solve
        self._conductor_drives = {}      # solid conductor -> v_k of the last solve
        self._q_eddy = None
        self._n_harmonic_solves = 0
        # the real DiffusionSystem assembly only checks anchors and holds the held cells: no PCG behind it (the anchor
        # check then runs at once on the gpu too, never deferred to a CG solve that never comes)
        super().__init__(domain, axisymmetric=axisymmetric, precision='float64',
                         linear_solver=_AssemblyOnlySolver(domain.backend.name))

    # --- PhysicsSolver contract ---------------------------------------------

    @property
    def n_solves(self) -> int:
        return self._n_harmonic_solves

    @property
    def host_factor_bytes(self):
        """SuperLU L + U of the last factorisation (Simulation.memory_report), None before one."""
        return self._factor.factor_bytes

    def prepare(self, domain) -> None:
        """Register the permeability / conductivity arrays, prepare the window system and allocate the phasor fields."""
        self.domain = domain
        self.xp = domain.xp   # numpy (cpu backend) or cupy: fields, sources and the inherited helpers
        domain.materials.add_properties(self.material_schema)
        system = self.system
        system.prepare(domain)
        nx, ny = domain.grid.nx, domain.grid.ny
        if self.fields is None or (self.fields.nx, self.fields.ny) != (nx, ny):
            self.fields = MagneticFieldsAC2D(nx, ny, xp=self.xp)
        if min(system.window_shape) < 3:
            raise ValueError(f"magnetic_fields_ac '{self.name}' needs at least 3 x 3 window cells, got "
                             f"{system.window_shape}")
        window = system.window
        self._source_window = self.fields.J_coupled[window]
        self._applied_source_window = self.fields.J_applied_re[window]
        self._applied_imag_window = self.fields.J_applied_im[window]
        x_range, y_range = self._bind_window(self.fields.A_re)
        self._derived_pml_copies = [
            pair for field_name in AC_FIELD_NAMES if field_name not in _UNEXTENDED_FIELDS
            for pair in edge_extension_views(getattr(self.fields, field_name), x_range, y_range)]
        self._bind_geometry()
        self._source_dirty = True
        self._host_dirty = True
        self._kept_operator = None   # (nu_xx, nu_yy, held, BC key, HostOperator without the sigma diagonal)

    def materials_changed(self, names) -> None:
        names = set(names)
        if PERMEABILITY_ARRAYS & names:
            self.system.operator_dirty = True
        if names & (CONDUCTIVITY_ARRAYS | {'mu_dc_loss'}):
            self._host_dirty = True

    def set_frequency(self, frequency) -> None:
        """New frequency [Hz] (a reused sweep build): the host system is rebuilt on the kept K(nu) stencil and the
        RecycledFactor solves by GMRES on the old factor or refactorises on its plan."""
        frequency = float(frequency)
        if not (math.isfinite(frequency) and frequency > 0.0):
            raise ValueError(f'magnetic_fields_ac needs a finite frequency > 0 Hz, got {frequency}')
        if frequency != self.frequency:
            self.frequency = frequency
            self.omega = 2.0 * math.pi * frequency
            self._host_dirty = True

    def reset_records(self) -> None:
        """RerunSolver (reused sweep build): clear the probe series; the next step re-solves when an input changed."""
        for probe in self.probes:
            probe.clear()

    def step(self, t: float = 0.0, dt: float = math.inf) -> None:
        """Re-solve only when the operator, a material or a source changed (always at the first call); probes as
        MagneticPhysics."""
        t = 0.0 if t is None else float(t)
        re_solve = (self.system.operator_dirty or self._host_dirty or self._source_dirty
                    or self._n_harmonic_solves == 0)
        if re_solve:
            if self.system.operator_dirty:
                self._assemble()
            if self._host_dirty:
                self._build_host_system()
            self._solve_harmonic()
            self._update_fields()
            self._source_dirty = False
        if re_solve or dt is None or not math.isfinite(dt):
            for probe in self.probes:
                probe.record(self.fields, t, replace_same_time=t > 0.0)

    def state(self) -> dict:
        materials = self.domain.materials
        state = {name: getattr(self.fields, name) for name in AC_FIELD_NAMES}
        for array_name in self.material_schema:
            state[array_name] = getattr(materials, array_name)
        return state

    def renderer(self):
        return MagneticACRenderer(self)

    def scalars(self) -> dict:
        """energy [J/m, J], eddy_loss / magnetic_loss [W/m, W] (time averages), per coil COIL_RESULT_KEYS; {} before a
        solve."""
        if self._n_harmonic_solves == 0:
            return {}
        power = face_power(self._host.operator, self._solution, self._host.held)
        ring = 2.0 * math.pi if self._axisymmetric else 1.0
        values = {'energy': 0.25 * ring * power.real, 'eddy_loss': float(np.sum(self._q_eddy * self._cell_volume())),
                  'magnetic_loss': 0.5 * self.omega * ring * power.imag}
        for coil in self.coils:
            result = self.coil_results(coil)
            values.update({f'{coil.name}_{key}': result[key] for key in COIL_RESULT_KEYS})
        return values

    def work_counters(self) -> dict:
        """Solves (summary 'solvers' physics counters); factorisations / recycled GMRES are the 'linear' role's record,
        or added here for a private (script) solver."""
        counters = {'solves': self._n_harmonic_solves}
        if self._owns_host_solver:
            counters.update(self.host_solver.work_counters())
        return counters

    def solve_summary(self):
        """One log line: unknowns, circuit rows, solves, factorisations, recycled solves and the last factor size."""
        if self._host is None:
            return None
        n_rows = sum(1 for conductor in self._host.conductors if conductor.drive == 'current')
        factor = '' if self.host_factor_bytes is None else f', last factor {format_bytes(self.host_factor_bytes)}'
        solver = self.host_solver
        library = factor_label(solver.back_end)   # SuperLU, MUMPS, PARDISO (host) or cuDSS
        where = '' if solver.back_end in CUDSS_BACK_ENDS else 'host '
        return (f'{where}{library} at {self.frequency:.4g} Hz, {self._host.free_index.size:,} unknowns + {n_rows} '
                f'circuit rows, {self._n_harmonic_solves} solves, {solver.n_factorizations} factorisations, '
                f'{solver.n_recycled_solves} recycled ({solver.total_recycled_iterations} GMRES iterations){factor}')

    # --- magnetics API ------------------------------------------------------------

    def add_coil(self, name: str, mask, current: float = 1.0, phase: float = 0.0, turns: int = 1,
                 conductor: str = 'stranded', drive: str = 'current', voltage: float = 0.0,
                 fill_factor: float = 1.0) -> ACCoilRegion:
        """Stranded coil (J = turns * I / area, I = current e^(j phase deg), always current-driven) or solid conductor
        (circuit row: drive 'current' prescribes I, 'voltage' the loop voltage, V/m planar, V axisymmetric)."""
        drive, value = coil_drive(name, conductor, drive, current, voltage, phase, turns, fill_factor)
        system = self.system
        mask_window = system.window_mask(mask, f"coil '{name}'")
        mask_host = np.array(to_host(mask_window))   # a copy: to_host returns the live array on the cpu
        area = float(np.sum(np.outer(system.dx_window, system.dy_window)[mask_host]))
        if not area > 0.0:
            raise ValueError(f"coil '{name}' selects no window cell")
        coil = ACCoilRegion(name, mask_host, conductor, drive, value, int(turns), float(fill_factor), area)
        if conductor == 'stranded':
            density = coil.turns * value / area
            self._applied_source_window += self.xp.where(mask_window, DTYPE(density.real), DTYPE(0.0))
            self._applied_imag_window += self.xp.where(mask_window, DTYPE(density.imag), DTYPE(0.0))
        self.coils.append(coil)
        self._host_dirty = True   # stranded cells lose their eddy currents, solid ones add a circuit row
        self._source_dirty = True
        return coil

    def coil_results(self, coil: ACCoilRegion) -> dict:
        """Complex current and loop voltage (stranded: R I + j omega flux linkage, R from the coil cells' sigma_dc and
        fill_factor; solid: the circuit unknown), Z = V / I -> resistance Re Z, inductance Im Z / omega, power
        Re(V I*) / 2; planar per unit length (A, V/m, Ohm/m, H/m, W/m), axisymmetric whole rings (A, V, Ohm, H, W)."""
        if self._n_harmonic_solves == 0:
            raise RuntimeError(f"magnetic_fields_ac '{self.name}': coil '{coil.name}' needs a solve first")
        host = self._host
        mask = coil.mask_window
        ring = 2.0 * math.pi if self._axisymmetric else 1.0
        unknown = self._solution
        if coil.conductor == 'stranded':
            current = coil.value
            linkage = coil.turns * ring * complex(np.sum(unknown[mask] * host.cell_area[mask])) / coil.area
            sigma = host.sigma[mask]
            path = host.cell_area[mask] * (ring * np.broadcast_to(host.radius, unknown.shape)[mask])
            resistance_dc = stranded_resistance(coil.turns, coil.fill_factor, coil.area, path, sigma)
            voltage = resistance_dc * current + 1j * self.omega * linkage
        else:
            drive_value = self._conductor_drives[coil.name]
            weights = (host.sigma_eddy * host.cell_area / host.radius)[mask]
            current = complex(np.sum(weights * (drive_value - 1j * self.omega * unknown[mask])))
            voltage = ring * drive_value
        return coil_result_values(current, voltage, self.omega, coil.conductor, coil.drive)

    def field_energy(self) -> float:
        """Time-averaged magnetic energy (1/4) Re(u^H K u) of the free cells [J/m planar, J axisymmetric]."""
        return self.scalars().get('energy', math.nan)

    def warm_start_fields(self) -> dict:
        """Host copy of the coupled source (the direct solve needs no starting unknown)."""
        return {'source': np.array(to_host(self._source_window))}

    def seed_warm_start(self, fields) -> tuple:
        self.xp.copyto(self._source_window, self.xp.asarray(fields['source'], dtype=DTYPE))
        self._source_dirty = True
        return ('J_coupled',)

    # --- internals ------------------------------------------------------------

    def _assemble(self):
        super()._assemble()   # real operator: mu_dc check, held cells, anchor check
        self._host_dirty = True

    def _cell_volume(self):
        """Host cell volumes: dx dy planar, ring 2 pi r_c dr dz axisymmetric."""
        host = self._host
        return host.cell_area * (2.0 * math.pi * host.radius if self._axisymmetric else 1.0)

    def _complex_reluctivities(self):
        """Host (nu_xx, nu_yy) = 1 / (mu0 (mu_dc - j mu_dc_loss)) of the window (mu_dc_loss only when allocated)."""
        materials = self.domain.materials
        window = self.system.window
        mu_xx = to_host(materials.mu_dc_xx[window]).astype(np.float64)
        mu_yy = to_host(materials.mu_dc_yy[window]).astype(np.float64)
        loss = 0.0
        if 'mu_dc_loss' in materials.property_names():
            loss = to_host(materials.mu_dc_loss[window]).astype(np.float64)
            if not np.all(np.isfinite(loss) & (loss >= 0.0)):
                raise ValueError(f"magnetic_fields_ac '{self.name}': mu_dc_loss must be finite and >= 0 in every "
                                 'window cell')
        return 1.0 / (MU0 * (mu_xx - 1j * loss)), 1.0 / (MU0 * (mu_yy - 1j * loss))

    def _conductivity(self):
        """Host sigma_dc of the window seen by the out-of-plane current (isotropic in-plane components required)."""
        materials = self.domain.materials
        window = self.system.window
        sigma_xx = to_host(materials.sigma_dc_xx[window]).astype(np.float64)
        sigma_yy = to_host(materials.sigma_dc_yy[window]).astype(np.float64)
        if not np.all(np.isfinite(sigma_xx) & (sigma_xx >= 0.0)):
            raise ValueError(f"magnetic_fields_ac '{self.name}': sigma_dc must be finite and >= 0 in every window cell")
        if not np.array_equal(sigma_xx, sigma_yy):
            raise NotImplementedError(f"magnetic_fields_ac '{self.name}': the eddy current flows out of the plane, so "
                                      'it needs sigma_dc_xx = sigma_dc_yy (an anisotropic sigma_dc has no defined '
                                      'out-of-plane value here): use the isotropic sigma_dc')
        return sigma_xx

    def _build_host_system(self):
        """Host operator K(nu) / r-form + j omega sigma A / r, held-cell elimination and the conductor rows."""
        system = self.system
        # MUMPS / PARDISO keep the SuperLU limits (uncalibrated), the text naming the library; cuDSS: SuperLU's rule
        back_end = self.host_solver.back_end
        rule = back_end if back_end in HOST_DIRECT_BACK_ENDS else 'scipy_splu'
        issue = size_issue(rule, *system.window_shape, False, False)
        if issue is not None and issue[0] == 'error':
            raise ValueError(f"magnetic_fields_ac '{self.name}': {issue[1]}")
        dx, dy = system.dx_window, system.dy_window
        nu_xx, nu_yy = self._complex_reluctivities()
        sigma = self._conductivity()
        radius = (self._x_centres if self._axisymmetric else np.ones_like(self._x_centres))[:, None]
        held = np.array(to_host(system.fixed_user))   # a copy (the live mask on the cpu)
        stranded = np.zeros(system.window_shape, dtype=bool)
        for coil in self.coils:
            if coil.conductor == 'stranded':
                stranded |= coil.mask_window
        sigma_eddy = np.where(stranded | held, 0.0, sigma)
        cell_area = np.outer(dx, dy)
        sigma_weight = sigma_eddy * cell_area / radius   # sigma A / r [c]
        bcs = {face: getattr(self.bcs, face) for face in FACES}   # axisymmetric x_lo: conductance_factor 2 (axis)
        operator = self._operator_with(nu_xx, nu_yy, held, bcs, radius, 1j * self.omega * sigma_weight)
        held_flat = held.ravel()
        free_index = np.flatnonzero(~held_flat)
        held_index = np.flatnonzero(held_flat)
        held_values = to_host(system.fixed_user_values).astype(np.float64).ravel()[held_index]
        rows = operator.matrix[free_index]
        positions = np.full(held_flat.size, -1)
        positions[free_index] = np.arange(free_index.size)
        conductors = []
        for coil in self.coils:
            if coil.conductor != 'solid':
                continue
            cells = np.flatnonzero(coil.mask_window.ravel() & ~held_flat)
            weights = sigma_weight.ravel()[cells]
            if not np.sum(weights) > 0.0:
                raise ValueError(f"magnetic_fields_ac '{self.name}': solid conductor '{coil.name}' has no conducting "
                                 'free cell (sigma_dc > 0)')
            value = coil.value if coil.drive == 'current' or not self._axisymmetric else coil.value / (2.0 * math.pi)
            conductors.append(ConductorRow(coil.name, positions[cells], weights, coil.drive, value,
                                           float(np.sum(weights))))
        self._host = _HostSystem(
            operator, held, held_index, held_values, free_index, rows[:, free_index].tocsr(),
            operator.boundary_rhs.ravel()[free_index] - rows[:, held_index] @ held_values, tuple(conductors),
            nu_xx, nu_yy, sigma, sigma_eddy, radius, cell_area)
        self._operator_revision += 1
        self._host_dirty = False

    def _operator_with(self, nu_xx, nu_yy, held, bcs, radius, diagonal_extra):
        """HostOperator K(nu) + diag(diagonal_extra): the face conductances and stencil CSR kept while nu, the held
        cells and the BCs stay (a sigma-only change, e.g. sigma_dc(heat_T), re-adds only the j omega sigma diagonal)."""
        bc_key = tuple((face, bc.kind, bc.value, bc.h, bc.external, bc.conductance_factor) for face, bc in bcs.items())
        kept = self._kept_operator
        if not (kept is not None and kept[3] == bc_key and np.array_equal(kept[0], nu_xx)
                and np.array_equal(kept[1], nu_yy) and np.array_equal(kept[2], held)):
            system = self.system
            base = assemble_operator(nu_yy / radius, nu_xx / radius, system.dx_window, system.dy_window, held, bcs,
                                     np.zeros(held.shape))
            kept = self._kept_operator = (nu_xx, nu_yy, held, bc_key, base)
        base = kept[4]
        # every diagonal entry is stored (the stencil's COO lists each cell): a new data array on the same index arrays
        matrix = base.matrix.copy()
        matrix.setdiag(matrix.diagonal() + np.asarray(diagonal_extra).ravel())
        return HostOperator(matrix, base.boundary_rhs, base.conductance_x, base.conductance_y, base.face_terms)

    def _solve_harmonic(self):
        """One complex solve: direct with the current factorisation, recycled GMRES after an operator change,
        refactorised when that misses; stores the window unknown and the solid conductors' v_k."""
        host = self._host
        source = ((to_host(self._source_window).astype(np.float64)
                   + to_host(self._applied_source_window).astype(np.float64)
                   + 1j * to_host(self._applied_imag_window).astype(np.float64)) * host.cell_area)
        matrix, rhs = bordered_system(host.matrix_free, host.held_drive + source.ravel()[host.free_index],
                                      host.conductors, self.omega)
        solution = self._factor.solve(matrix, rhs, self._operator_revision)
        n_free = host.free_index.size
        unknown = np.zeros(host.held.size, dtype=np.complex128)
        unknown[host.held_index] = host.held_values
        unknown[host.free_index] = solution[:n_free]
        self._solution = unknown.reshape(host.held.shape)
        border = iter(solution[n_free:])
        self._conductor_drives = {conductor.name: complex(next(border)) if conductor.drive == 'current'
                                  else conductor.value for conductor in host.conductors}
        self._n_harmonic_solves += 1

    def _update_fields(self):
        """Phasor fields from the solution (host float64, one transfer each): B = H / nu from the face fluxes, E = (v_k -
        j omega u) / r, J_conduction = sigma_eddy E, q_eddy = sigma_eddy |E|^2 / 2, q_magnetic = omega Im(nu) |B|^2 / 2,
        w_mag = Re(nu) |B|^2 / 4; edge-extended into the PML ring."""
        host = self._host
        system = self.system
        unknown = self._solution
        held = host.held
        flux_x, flux_y = face_fluxes_at_centres(host.operator, unknown, system.dx_window, system.dy_window, held)
        sign = -1.0 if self._axisymmetric else 1.0   # flux = (H_y, -H_x) planar, (-H_z, H_r) axisymmetric
        B_x = -sign * flux_y / host.nu_xx
        B_y = sign * flux_x / host.nu_yy
        drive = np.zeros(unknown.shape, dtype=np.complex128)
        for coil in self.coils:
            if coil.name in self._conductor_drives:
                drive[coil.mask_window] = self._conductor_drives[coil.name]
        field_E = np.where(held, 0.0, (drive - 1j * self.omega * unknown) / host.radius)
        J_conduction = host.sigma_eddy * field_E
        self._q_eddy = 0.5 * host.sigma_eddy * np.abs(field_E) ** 2
        J_total = (to_host(self._source_window) + to_host(self._applied_source_window)
                   + 1j * to_host(self._applied_imag_window) + J_conduction)
        B_x2, B_y2 = np.abs(B_x) ** 2, np.abs(B_y) ** 2
        potential = unknown / host.radius
        values = {'A_re': potential.real, 'A_im': potential.imag, 'A_abs': np.abs(potential),
                  'B_x_re': B_x.real, 'B_x_im': B_x.imag, 'B_y_re': B_y.real, 'B_y_im': B_y.imag,
                  'B_abs': np.sqrt(B_x2 + B_y2),
                  'J_conduction_re': J_conduction.real, 'J_conduction_im': J_conduction.imag,
                  'J_total_re': J_total.real, 'J_total_im': J_total.imag, 'J_total_abs': np.abs(J_total),
                  'q_eddy': self._q_eddy,
                  'q_magnetic': 0.5 * self.omega * (host.nu_xx.imag * B_x2 + host.nu_yy.imag * B_y2),
                  'w_mag': 0.25 * (host.nu_xx.real * B_x2 + host.nu_yy.real * B_y2)}
        window = system.window
        xp = self.xp
        for name, array in values.items():
            getattr(self.fields, name)[window] = xp.asarray(array.astype(np.float32))
        self._extend_into_pml()
        for destination, source in self._derived_pml_copies:
            xp.copyto(destination, source)

    def _indicator_flux(self):
        """error_indicator's flux: the complex host face fluxes of the last phasor solve (|jump|^2 per face)."""
        if self._host is None or self._solution is None:
            raise ValueError(f"magnetic_fields_ac '{self.name}' has no phasor solve yet")
        system = self.system
        return face_fluxes_at_centres(self._host.operator, self._solution, system.dx_window, system.dy_window,
                                      self._host.held)
