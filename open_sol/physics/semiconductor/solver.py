"""DriftDiffusionPhysics: the PhysicsSolver wrapper of the host drift-diffusion core (materials, doping, ohmic contacts,
gates, generation); device float32 output fields, edge-extended into the PML ring."""
import math
import time
from dataclasses import dataclass

import numpy as np

from ...core.backend import to_host
from ...core.constants import DTYPE
from ...core.fv_diffusion import edge_extension_views
from ...core.gpu_memory import format_bytes
from ...core.solver_base import PhysicsSolver
from ...solvers.host_direct import HOST_DIRECT_BACK_ENDS
from ...solvers.sparse_direct import SparseDirectSolver, size_issue
from .drift_diffusion import COMPONENTS, DriftDiffusionSystem, thermal_voltage
from .renderer import SEMICONDUCTOR_FIELD_INFOS, SemiconductorRenderer

# MaterialGrid arrays of the drift-diffusion vocabulary (eps_dc_xx / _yy shared with electrostatics, same default).
SEMICONDUCTOR_SCHEMA = {
    'eps_dc_xx': 1.0, 'eps_dc_yy': 1.0,
    'n_intrinsic': 0.0, 'mobility_n': 0.0, 'mobility_p': 0.0,
    'lifetime_n': 1e-6, 'lifetime_p': 1e-6, 'radiative_recombination': 0.0,
}
# Material property -> MaterialGrid arrays it writes (isotropic entries expand to their components).
SEMICONDUCTOR_PROPERTY_ARRAYS = {
    'eps_dc': ('eps_dc_xx', 'eps_dc_yy'), 'eps_dc_xx': ('eps_dc_xx',), 'eps_dc_yy': ('eps_dc_yy',),
    'n_intrinsic': ('n_intrinsic',), 'mobility_n': ('mobility_n',), 'mobility_p': ('mobility_p',),
    'lifetime': ('lifetime_n', 'lifetime_p'), 'lifetime_n': ('lifetime_n',), 'lifetime_p': ('lifetime_p',),
    'radiative_recombination': ('radiative_recombination',),
}
GENERATION_SLOT = 'generation'
CONTACT_CURRENT_PREFIX = 'current_'   # scalar current_<contact> [A/m; A when axisymmetric] (E5)
GATE_CHARGE_PREFIX = 'charge_'        # scalar charge_<gate> [C/m; C when axisymmetric] (E5)
SOLVED_FIELDS = tuple(name for name in SEMICONDUCTOR_FIELD_INFOS if not name.startswith('generation'))
_SMALLER_MODEL = 'coarsen the mesh or shrink the domain'


def linear_size_issue(back_end, window_cells_x, window_cells_y):
    """sparse_direct.size_issue of the coupled system (3 unknowns per window cell; cusolver_qr sized as a (wx, 3 wy)
    grid, the interleaved index order; MUMPS / PARDISO take the SuperLU rule, uncalibrated); None, ('warning', text) or
    ('error', text). CPU only."""
    if back_end in ('scipy_splu',) + HOST_DIRECT_BACK_ENDS:
        return size_issue(back_end, window_cells_x, window_cells_y, False, False, unknowns_per_cell=COMPONENTS,
                          remedy=_SMALLER_MODEL)
    return size_issue(back_end, window_cells_x, COMPONENTS * window_cells_y, False, False)


@dataclass(frozen=True, eq=False)
class ContactRegion:
    """Runtime object of an ohmic_contact feature."""
    name: str
    voltage: float
    index: int


@dataclass(frozen=True, eq=False)
class GateRegion:
    """Runtime object of a gate feature."""
    name: str
    voltage: float
    work_function_difference: float
    index: int


class DriftDiffusionPhysics(PhysicsSolver):
    """2D stationary drift-diffusion (Poisson + electron / hole continuity) on the physical window, planar or
    axisymmetric (r, z); re-solves (Newton with continuation) only after a material, doping or generation change."""

    name = 'semiconductor'

    def __init__(self, domain, temperature=300.0, linear_solver='auto', tolerance=1e-8, max_newton_iterations=50,
                 max_bias_step=0.1, axisymmetric=False):
        if not (tolerance > 0.0 and int(max_newton_iterations) >= 1 and max_bias_step > 0.0):
            raise ValueError(f'need tolerance > 0, max_newton_iterations >= 1 and max_bias_step > 0, got {tolerance}, '
                             f'{max_newton_iterations}, {max_bias_step}')
        self.axisymmetric = bool(axisymmetric)   # (r, z): x = r from the window's x_lo edge; Session ring-weights on it
        self.material_schema = dict(SEMICONDUCTOR_SCHEMA)
        self.temperature = float(temperature)
        self.thermal_voltage = thermal_voltage(self.temperature)
        # the build's 'linear' sparse_direct solver (ctx.solvers); a back_end choice (scripts) makes a private one,
        # whose counters work_counters then reports
        self._owns_linear_solver = not isinstance(linear_solver, SparseDirectSolver)
        self.linear_solver = (SparseDirectSolver(back_end=linear_solver, backend=domain.backend.name)
                              if self._owns_linear_solver else linear_solver)
        self._finalized = False
        self._newton_settings = (float(tolerance), int(max_newton_iterations), float(max_bias_step))
        self.contacts = []
        self.gates = []
        self.n_solves = 0
        self.fields = None
        self._dirty = True
        self._solve_seconds = 0.0
        self.prepare(domain)

    # --- PhysicsSolver contract ---------------------------------------------

    def prepare(self, domain) -> None:
        """Register the material arrays, allocate the full-grid fields and the window system (features come later)."""
        self.domain = domain
        self.xp = domain.xp   # numpy (cpu backend) or cupy: the output fields; the system is host float64 anyway
        domain.materials.add_properties(self.material_schema)
        nx, ny = domain.grid.nx, domain.grid.ny
        window_x, window_y = domain.physical_window()
        x_start, x_stop, _ = window_x.indices(nx)
        y_start, y_stop, _ = window_y.indices(ny)
        self.window = (slice(x_start, x_stop), slice(y_start, y_stop))
        self.fields = {name: self.xp.zeros((nx, ny), dtype=DTYPE) for name in SEMICONDUCTOR_FIELD_INFOS}
        dx_cells, dy_cells = domain.cell_widths()
        tolerance, max_iterations, max_bias_step = self._newton_settings
        self.system = DriftDiffusionSystem(np.asarray(dx_cells)[x_start:x_stop], np.asarray(dy_cells)[y_start:y_stop],
                                           label=lambda: f"drift_diffusion '{self.name}'", tolerance=tolerance,
                                           max_newton_iterations=max_iterations, max_bias_step=max_bias_step,
                                           axisymmetric=self.axisymmetric, linear_solver=self.linear_solver)
        self._pml_copies = [pair for name in SOLVED_FIELDS
                            for pair in edge_extension_views(self.fields[name], (x_start, x_stop), (y_start, y_stop))]
        self._dirty = True

    def finalize(self) -> None:
        """Resolve the linear back end (ValueError for 'scipy_splu' without scipy, or a window over its size limit)."""
        issue = linear_size_issue(self.linear_solver.back_end, *self.system.shape)
        if issue is not None and issue[0] == 'error':
            raise ValueError(issue[1])
        self._finalized = True
        self._dirty = True

    def materials_changed(self, names) -> None:
        if not set(self.material_schema).isdisjoint(names):
            self._dirty = True

    def step(self, t: float = 0.0, dt: float = math.inf) -> None:
        """Solve when an input changed since the last solve (always at the first call)."""
        if self._dirty or self.n_solves == 0:
            self._solve()

    def dt_limit(self) -> float:
        return math.inf

    def state(self) -> dict:
        """Full-grid float32 fields (window values edge-extended into the PML ring; generation slots as given)."""
        return dict(self.fields)

    def inject_source(self, name: str, array) -> None:
        if name != GENERATION_SLOT:
            raise ValueError(f"drift_diffusion '{self.name}' accepts source '{GENERATION_SLOT}'; got '{name}'")
        self.xp.copyto(self.fields[GENERATION_SLOT], self.xp.asarray(array, dtype=DTYPE))
        self._dirty = True

    def renderer(self):
        return SemiconductorRenderer(self)

    # --- feature API ------------------------------------------------------------

    def add_doping(self, mask, donors, acceptors) -> None:
        """Add donors - acceptors [1/m^3] (scalars or full-grid arrays) on the window cells under mask."""
        mask_window = self._host_window_mask(mask, 'doping')
        net = self._host_window_values(donors) - self._host_window_values(acceptors)
        self.system.add_doping(np.where(mask_window, net, 0.0).ravel())
        self._dirty = True

    def set_auger_recombination(self, mask, electron_coefficient, hole_coefficient) -> None:
        """Auger coefficients C_n, C_p [m^6/s] (host float64 scalars, never the float32 MaterialGrid) under mask."""
        if not (electron_coefficient >= 0.0 and hole_coefficient >= 0.0):
            raise ValueError(f"drift_diffusion '{self.name}': Auger coefficients must be >= 0, got "
                             f'{electron_coefficient}, {hole_coefficient}')
        self.system.set_auger(self._host_window_mask(mask, 'auger_recombination').ravel(), float(electron_coefficient),
                              float(hole_coefficient))
        self._dirty = True

    def add_generation(self, mask, rate) -> None:
        """Add an applied generation rate [1/(m^3 s)] (scalar or full-grid array) to generation_ext under mask."""
        mask_window = self._device_window_mask(mask)
        rate = self.xp.asarray(rate, dtype=DTYPE)
        rate_window = rate if rate.ndim == 0 else rate[self.window]
        self.fields['generation_ext'][self.window] += self.xp.where(mask_window, rate_window, DTYPE(0.0))
        self._dirty = True

    def add_ohmic_contact(self, name, mask, voltage) -> ContactRegion:
        """Hold the semiconductor cells under mask at neutral equilibrium shifted by `voltage` [V]."""
        index = self.system.add_contact(name, self._host_window_mask(mask, 'ohmic contact').ravel(), voltage)
        region = ContactRegion(name, float(voltage), index)
        self.contacts.append(region)
        self._dirty = True
        return region

    def add_gate(self, name, mask, voltage, work_function_difference=0.0) -> GateRegion:
        """Hold psi = voltage - work_function_difference [V] on the insulating cells under mask."""
        index = self.system.add_gate(name, self._host_window_mask(mask, 'gate').ravel(), voltage,
                                     work_function_difference)
        region = GateRegion(name, float(voltage), float(work_function_difference), index)
        self.gates.append(region)
        self._dirty = True
        return region

    def set_contact_voltage(self, region: ContactRegion, voltage) -> None:
        """New applied voltage [V] of an ohmic contact (a reused sweep build); the next solve continues from the last
        converged state."""
        self.system.contact_voltages[region.index] = float(voltage)
        self._dirty = True

    def set_gate_voltage(self, region: GateRegion, voltage) -> None:
        """New applied voltage [V] of a gate (a reused sweep build)."""
        self.system.gate_voltages[region.index] = float(voltage)
        self._dirty = True

    def contact_voltage(self, region: ContactRegion) -> float:
        """The contact's current applied voltage [V] (the feature value, or the last set_contact_voltage)."""
        return self.system.contact_voltages[region.index]

    def gate_voltage(self, region: GateRegion) -> float:
        return self.system.gate_voltages[region.index]

    def reset_records(self) -> None:
        """RerunSolver (reused sweep build): the next step re-solves; the last converged state stays the continuation
        start (one ramp of the sweep step instead of a cold solve)."""
        self._dirty = True

    def contact_currents(self, region: ContactRegion):
        """(electron, hole) current [current_unit] out of the contact into the domain, from the last solve."""
        self._require_solve(region.name)
        return self.system.contact_currents(region.index)

    def gate_charge(self, region: GateRegion) -> float:
        """Free charge [charge_unit] on the gate (its outward D flux) from the last solve."""
        self._require_solve(region.name)
        return self.system.gate_charge(region.index)

    @property
    def current_unit(self) -> str:
        """Unit of contact currents: 'A/m' per unit z length, 'A' of the whole ring when axisymmetric."""
        return 'A' if self.axisymmetric else 'A/m'

    @property
    def charge_unit(self) -> str:
        """Unit of gate charges: 'C/m' per unit z length, 'C' of the whole ring when axisymmetric."""
        return 'C' if self.axisymmetric else 'C/m'

    # --- reporting ----------------------------------------------------------------

    def scalars(self) -> dict:
        """current_<contact> (electron + hole, current_unit) and charge_<gate> (charge_unit) of the last solve; {}
        before one."""
        if self.n_solves == 0:
            return {}
        values = {f'{CONTACT_CURRENT_PREFIX}{region.name}': float(sum(self.system.contact_currents(region.index)))
                  for region in self.contacts}
        values.update({f'{GATE_CHARGE_PREFIX}{region.name}': float(self.system.gate_charge(region.index))
                       for region in self.gates})
        return values

    def warm_start_fields(self) -> dict:
        """Host copy of the last converged state (study warm_start: the next sweep value continues from it); {} before
        a solve."""
        return self.system.converged_state() or {}

    def seed_warm_start(self, fields) -> tuple:
        """Continue the next solve from warm_start_fields() of another build (same sites, contacts and gates; else a
        cold start); seeds no state slot (the fields come from the solve)."""
        if self.system.seed_converged_state(fields):
            self._dirty = True
        return ()

    @property
    def host_factor_bytes(self):
        """Host bytes of the last host factorisation (SuperLU, MUMPS or PARDISO; None before one, with cusolver_qr or
        cuDSS)."""
        return self.system.host_factor_bytes

    def work_counters(self) -> dict:
        """Newton work (summary 'solvers' physics counters); the factorisations are the 'linear' role's record, or added
        here for a private (script) solver."""
        system = self.system
        counters = {'newton_iterations': system.newton_iterations, 'continuation_steps': system.continuation_steps,
                    'cold_starts': system.cold_starts, 'rejected_predictions': system.rejected_predictions,
                    'chord_steps': system.chord_steps, 'solve_seconds': self._solve_seconds}
        if self._owns_linear_solver:
            counters.update(self.linear_solver.work_counters())
        return counters

    def solve_summary(self) -> str:
        """One log line: back end, unknowns, Newton work and the last factor size (cuDSS: factorisations and plans)."""
        counters = self.linear_solver.work_counters()
        factor = '' if self.host_factor_bytes is None else f', last factor {format_bytes(self.host_factor_bytes)}'
        if 'cudss_plans' in counters:
            factor = (f", {counters['factorizations']} cuDSS factorisations on {counters['cudss_plans']} plan(s), "
                      f"last factor {format_bytes(counters['cudss_factor_device_bytes'])} on the GPU")
        back_end = counters['back_end'] or self.linear_solver.back_end_choice
        return (f'{back_end}, {COMPONENTS * self.system.n_cells:,} unknowns, '
                f'{self.n_solves} solves, {self.system.newton_iterations} Newton iterations, '
                f'{self.system.continuation_steps} continuation steps{factor}')

    # --- internals ------------------------------------------------------------

    def _solve(self):
        if not self._finalized:
            self.finalize()
        start = time.perf_counter()
        window = self.window
        materials = self.domain.materials
        host_materials = {name: to_host(getattr(materials, name)[window]).astype(np.float64).ravel()
                          for name in self.material_schema}
        generation = (self.fields['generation'][window].astype(np.float64)
                      + self.fields['generation_ext'][window].astype(np.float64))
        self.system.set_inputs(host_materials, to_host(generation).ravel(), self.thermal_voltage)
        self.system.solve()
        for name, values in self.system.outputs().items():
            self.fields[name][window] = self.xp.asarray(values, dtype=DTYPE)
        for destination, source in self._pml_copies:
            self.xp.copyto(destination, source)
        self.n_solves += 1
        self._dirty = False
        self._solve_seconds += time.perf_counter() - start

    def _host_window_mask(self, mask, label):
        """numpy bool window view of a full-grid mask; ValueError for another shape."""
        return np.array(to_host(self._device_window_mask(mask, label)))

    def _device_window_mask(self, mask, label='mask'):
        """bool window mask on the run backend of a full-grid mask; ValueError for another shape."""
        mask = self.xp.asarray(mask)
        grid_shape = (self.domain.grid.nx, self.domain.grid.ny)
        if mask.shape != grid_shape:
            raise ValueError(f'{label}: mask must be a full-grid {grid_shape} boolean array, got shape {mask.shape}')
        return mask[self.window].astype(np.bool_)

    def _host_window_values(self, value):
        """float64 window array of a full-grid array, or the float of a scalar."""
        if getattr(value, 'ndim', 0):
            return to_host(value[self.window]).astype(np.float64)
        return float(value)

    def _require_solve(self, feature_name):
        if self.n_solves == 0:
            raise RuntimeError(f"drift_diffusion '{self.name}': the result of '{feature_name}' needs a solve first")
