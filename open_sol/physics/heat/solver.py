"""HeatPhysics: anisotropic heat conduction (planar, axisymmetric, polar / spherical or 3D), steady or implicit-Euler
transient, on core.fv_diffusion's DiffusionSystem."""
import math

import numpy as np

from ...core.backend import cp, to_host
from ...core.constants import DTYPE, STEFAN_BOLTZMANN
from ...core.cut_cells import locate_held_edges
from ...core.fv_diffusion import DiffusionPhysicsBase, DiffusionSystem
from ...solvers.line_block import LineBlockSolver
from ...solvers.sparse_direct import SparseDirectSolver

from .fields import TemperatureField2D, TemperatureField3D, heat_state
from .boundaries import HeatBoundaryConfig
from ..em.index_reference import record_reference_index
from .materials import THERMAL_SCHEMA
from .renderer import HeatRenderer
from .thermal_noise import HarmonicHeatOperator, HarmonicParticleOperator, solve_noise_monitor


# MaterialGrid arrays the assembled conduction operator depends on (k_zz only on a 3D domain); rho / cp enter only the
# transient capacity term, replaced by DiffusionSystem.refresh_capacity (steady heat never reads them).
_OPERATOR_MATERIALS = frozenset({'k_xx', 'k_yy', 'k_zz', 'k_xy'})
_CAPACITY_MATERIALS = frozenset({'rho', 'cp'})

# Anchor message parts of the DiffusionSystem singular-system check.
_ANCHOR_TEXT = 'temperature/convection boundary, fixed-temperature cell or volumetric loss'
_REMEDY_TEXT = 'Anchor every conducting region, or solve in transient mode with rho and cp set there.'
# Surface-to-ambient radiation (Newton-linearised at the radiating cells' T): re-linearisations of a steady solve, and
# its convergence = the largest change of a radiating cell's T between two linearisations over the largest T
RADIATION_MAX_ITERATIONS = 50
RADIATION_TOLERANCE = 1e-7

_rise_to_temperature = cp.ElementwiseKernel(
    'T rise, float64 reference_temperature',
    'float32 temperature',
    'temperature = (float)(reference_temperature + (double)rise)',
    'heat_rise_to_temperature')

_temperature_to_rise = cp.ElementwiseKernel(
    'float32 temperature, float64 reference_temperature',
    'T rise',
    'rise = (T)((double)temperature - reference_temperature)',
    'heat_temperature_to_rise')


def _rise_to_temperature_host(rise, reference_temperature, temperature):
    """numpy twin of _rise_to_temperature (float64 sum, float32 result)."""
    np.copyto(temperature, (reference_temperature + rise.astype(np.float64)).astype(DTYPE))


def _temperature_to_rise_host(temperature, reference_temperature, rise):
    """numpy twin of _temperature_to_rise."""
    np.copyto(rise, (temperature.astype(np.float64) - reference_temperature).astype(rise.dtype))


class HeatPhysics(DiffusionPhysicsBase):
    """Anisotropic heat conduction on the physical window (2D planar, axisymmetric (r, z) with x = r, polar (r, phi) /
    spherical (r, theta) on a polar Domain, or 3D on a Domain3D): steady or implicit-Euler transient, a DiffusionSystem
    (`system`) solving the rise T - T0 by its linear_solver (None: Jacobi-PCG; model Automatic: multigrid-PCG on the gpu,
    host SuperLU on the cpu) (T float32); (r, z) azimuthal_order m != 0: the rise is the amplitude of (T - T0) cos(m phi)."""

    name = "heat"
    material_schema = THERMAL_SCHEMA
    conformal_locator = None   # model.boundary_locator.GeometryLocator of set_conformal_boundaries (cut-cell held edges)
    solves_every_step = True   # no dirty check: the runner warns when a steady heat solve runs at (nearly) every tick
    _probe_field = 'T'
    _probe_label = 'temperature probe'

    def __init__(self, domain, bcs: HeatBoundaryConfig = None,
                 T0: float = 293.15, tol: float = 1e-8, max_iter: int = 5000,
                 compute_residual: bool = False,
                 precision: str = 'float64', time_mode: str = 'stationary',
                 time_step: float = math.inf, axisymmetric: bool = False, require_anchor: bool = True,
                 linear_solver=None, shifted_solver=None, azimuthal_order: int = 0):
        if bcs is not None and not isinstance(bcs, HeatBoundaryConfig):
            raise TypeError(f"bcs must be a HeatBoundaryConfig, got {type(bcs).__name__}")
        if time_mode not in ('stationary', 'transient'):
            raise ValueError(f"time_mode must be 'stationary' or 'transient', got {time_mode!r}")
        if time_mode == 'transient' and not (math.isfinite(time_step) and time_step > 0.0):
            raise ValueError(f"transient heat needs a finite time_step > 0, got {time_step}")
        # precision, tol and max_iter are validated by the DiffusionSystem below (a linear_solver replaces them)

        system = DiffusionSystem(
            domain, precision, float(tol), int(max_iter),
            label=lambda: f"heat '{self.name}'", unknown_name='temperature',
            anchor_text=_ANCHOR_TEXT, remedy_text=_REMEDY_TEXT, compute_residual=compute_residual,
            axisymmetric=axisymmetric, linear_solver=linear_solver, azimuthal_order=azimuthal_order)
        system.require_anchors = bool(require_anchor)   # False only for the thermal noise solve (never CG-solved)
        self._init_diffusion(domain, bcs if bcs is not None else HeatBoundaryConfig(), system)
        self.T0 = float(T0)
        self.time_mode = time_mode
        self.time_step = float(time_step)
        self.time = 0.0
        self.noise_monitors = []        # thermal_noise.ThermalNoiseMonitor of the thermorefractive / particle noise
        # 'shifted' solver of the noise shifts: solvers.sparse_direct (host SuperLU each; None = a private one) or a
        # solvers.line_block.LineBlockSolver shared by every noise operator
        self.shifted_solver = shifted_solver
        # radiation terms: [flat window cells, radiating area, eps sigma, T_amb, applied conductance, applied drive,
        # T of the linearisation] (device arrays of the system dtype; float64 values)
        self._radiation = []
        self.radiation_iterations = 0   # steady re-linearisations solved (work counter)
        self.runtime_warnings = []      # append-only study log (an unconverged radiation loop)

        self.prepare(domain)

    # --- PhysicsSolver contract ---------------------------------------------

    def prepare(self, domain) -> None:
        """Register the thermal schema, allocate the system and the T window views, record the thermo-optic
        reference indices (again at finalize: Simulation.run's first call or the build); assembly is deferred."""
        self.domain = domain
        materials = domain.materials
        materials.add_properties(self.material_schema)
        record_reference_index(materials)

        grid_shape = tuple(materials.shape)
        self.xp = domain.xp   # numpy (cpu backend) or cupy
        if self.fields is None or self.fields.T.shape != grid_shape:
            field_class = TemperatureField3D if len(grid_shape) == 3 else TemperatureField2D
            self.fields = field_class(*grid_shape, T0=self.T0, xp=self.xp)

        self.system.prepare(domain)
        window = self.system.window
        self._T_window = self.fields.T[window]
        self._Q_window = self.fields.Q[window]
        self._applied_source_window = self.fields.Q_ext[window]
        self._bind_window(self.fields.T)
        self._T0_device = self.xp.asarray(self.T0, dtype=DTYPE)
        on_cpu = self.system.backend == 'cpu'
        self._rise_to_temperature = _rise_to_temperature_host if on_cpu else _rise_to_temperature
        self._temperature_to_rise = _temperature_to_rise_host if on_cpu else _temperature_to_rise

        self._assembled_time_step = None
        self._capacity_stale = False   # rho / cp changed since the last assembly (only a transient step reads them)
        self._theta_stale = True

    def finalize(self) -> None:
        """Re-record the thermo-optic reference indices (n0_zz, plus n0_xx / n0_yy when eps_xx / eps_yy exist) from
        the current (unheated) permittivity and assemble (fails early on singular setups)."""
        record_reference_index(self.domain.materials)
        self.rebuild_operator()

    def materials_changed(self, names) -> None:
        names = set(names)
        if _OPERATOR_MATERIALS & names:
            # values only (k(T)): the re-assembly tests the c > 0 pattern instead of the full anchor check
            self.system.mark_coefficients_changed()
        if _CAPACITY_MATERIALS & names:
            self._capacity_stale = True
        if 'k_xy' in names:
            self.system.cross_term_dirty = True   # kxy_active re-decided at the next assembly

    def step(self, t: float = 0.0, dt: float = math.inf) -> None:
        """Steady solve, or (transient mode, finite dt) one implicit-Euler step over [t, t + dt]: T, `time` and the
        probe samples then belong to t + dt, up to (every-1) dt_sync ahead of the runner time."""
        t = 0.0 if t is None else float(t)
        transient = (self.time_mode == 'transient' and dt is not None and math.isfinite(dt))
        solve_time_step = float(dt) if transient else math.inf
        if not solve_time_step > 0.0:
            raise ValueError(f"heat '{self.name}': transient step needs dt > 0, got {dt}")

        system = self.system
        capacity_changed = transient and (self._capacity_stale or solve_time_step != self._assembled_time_step)
        # only rho / cp or the step length changed: replace C/dt (False: no transient assembly yet, or the cells it
        # pins / anchors changed)
        if (capacity_changed and not system.operator_dirty
                and system.refresh_capacity(self._capacity_over_dt(solve_time_step))):
            self._assembled_time_step = solve_time_step
            self._capacity_stale = False
        # radiation: Newton-linearised at the current T (transient: lagged one step); steady: re-linearised and
        # re-solved until the radiating cells' T settles
        iterations = RADIATION_MAX_ITERATIONS if (self._radiation and not transient) else 1
        for iteration in range(iterations):
            if self._radiation:
                change = self._linearise_radiation()
                if iteration and change <= RADIATION_TOLERANCE:
                    break
                if iteration:
                    self.radiation_iterations += 1
            if (self._radiation and system.operator_dirty and solve_time_step == self._assembled_time_step
                    and not (transient and self._capacity_stale)):
                system.refresh_diagonal()   # only the losses changed: no re-assembly when it succeeds
            if (system.operator_dirty or solve_time_step != self._assembled_time_step
                    or (transient and self._capacity_stale)):
                self._assemble(solve_time_step)   # also resyncs a stale rise from T
            system.build_rhs(self._Q_window, self._applied_source_window)
            system.solve()
        else:
            if iterations > 1:
                self.runtime_warnings.append(
                    f"heat '{self.name}': the radiation re-linearisation did not converge in {iterations} solves "
                    '(largest relative temperature change on radiating cells above '
                    f'{RADIATION_TOLERANCE:g}); the last solve is kept')
        self._rise_to_temperature(system.unknown2d, self.T0, self._T_window)
        self._extend_into_pml()

        record_time = t + solve_time_step if transient else t
        if transient:
            self.time = record_time
        for probe in self.probes:
            probe.record(self.fields, record_time)

    def reset_records(self) -> None:
        """RerunSolver (reused sweep build): clear the probe series; the next run re-solves from the last field."""
        for probe in self.probes:
            probe.clear()

    def dt_limit(self) -> float:
        return self.time_step if self.time_mode == 'transient' else math.inf

    def state(self) -> dict:
        fields = self.fields
        return heat_state(fields.T, self._T0_device, fields.Q, fields.Q_ext, self.domain.materials)

    def inject_source(self, name: str, array) -> None:
        if name != 'volumetric_power':
            raise ValueError(
                f"HeatPhysics accepts source 'volumetric_power'; got '{name}'")
        self.xp.copyto(self.fields.Q, self.xp.asarray(array, dtype=DTYPE))

    def renderer(self):
        return HeatRenderer(self)

    # --- heat-specific API (shared settings, set_face_bc, add_boundary_flux, probes: DiffusionPhysicsBase) ---

    def add_fixed_temperature(self, mask, T, inside=None) -> None:
        """Pin T [K] (scalar or full-grid array) on window cells under mask; also writes T. inside(x, y) -> bool (host,
        the drawn held region; 2D) puts the held edge at its drawn position (core.cut_cells cut cells)."""
        system = self.system
        mask_window = system.window_mask(mask, 'fixed temperature')
        temperature = system.window_values(T, 'fixed temperature', system.dtype)
        system.set_fixed(mask_window, temperature - self.T0)
        if inside is not None:
            locate_held_edges(system, mask_window, inside)
        self.xp.copyto(self._T_window, temperature, where=mask_window)
        self._temperature_changed()

    def set_conformal_boundaries(self, locator):
        """solver_defaults.conformal_boundaries: held features pass locator.inside(entities) (model.boundary_locator,
        duck-typed) to add_fixed_temperature, perimeter heat_flux / convection take the drawn outline, k = 0 walls
        become cut cells at the first assembly (re-located when the conducting cells change); None = staircased."""
        self.conformal_locator = locator
        self._set_insulated_wall_locator(locator)

    def add_surface_loss(self, mask, conductance, T_ext) -> None:
        """Add a perimeter loss of total conductance [W/(m K)] per cell (full-grid array; axisymmetric: whole ring W/K)
        * (T - T_ext) on window cells under mask (cut-cell convection; volume fractions never rescale it)."""
        system = self.system
        mask_window = system.window_mask(mask, 'surface loss')
        dtype = system.dtype
        conductance_window = system.window_values(conductance, 'surface loss conductance', dtype)
        T_ext_window = system.window_values(T_ext, 'surface loss T_ext', dtype)
        system.add_surface_loss(mask_window, conductance_window, T_ext_window, self.T0)

    def add_volumetric_loss(self, mask, h_volumetric, T_ext) -> None:
        """Add a loss h_volumetric [W/(m^3 K)] * (T - T_ext) on window cells under mask (scalars or full-grid arrays)."""
        if np.isscalar(h_volumetric) and h_volumetric < 0.0:
            raise ValueError(f"h_volumetric must be >= 0, got {h_volumetric}")
        system = self.system
        mask_window = system.window_mask(mask, 'volumetric loss')
        dtype = system.dtype
        h_window = system.window_values(h_volumetric, 'volumetric loss h', dtype)
        T_ext_window = system.window_values(T_ext, 'volumetric loss T_ext', dtype)
        system.add_diagonal_loss(mask_window, h_window, T_ext_window, self.T0)

    def add_radiation(self, weights, emissivity, ambient_temperature) -> None:
        """Surface-to-ambient radiation eps sigma (T^4 - T_amb^4) per radiating area of the window cells with weights
        > 0 (full-grid radiating area / cell volume [1/m], as definition_helpers.perimeter_flux_weights)."""
        system = self.system
        weight = self.xp.asarray(weights, dtype=system.dtype)[system.window].ravel()
        cells = self.xp.flatnonzero(weight > 0)
        self._add_radiating_cells(cells, weight[cells] * system.geometric_cell_area.ravel()[cells], emissivity,
                                  ambient_temperature)

    def add_face_radiation(self, face, emissivity, ambient_temperature) -> None:
        """Surface-to-ambient radiation on an outer face (its area on the boundary cells; axis faces have none)."""
        system = self.system
        axis = 'xyz'.index(face[0])
        last = face.endswith('hi')
        shape = system.window_shape
        if system.dimensions == 3:
            separable = system.metric_3d.face_area[axis]
            factors = [separable.along_x, separable.along_y, separable.along_z]
        else:
            separable = (system.metric.x_face_area, system.metric.y_face_area)[axis]
            factors = [separable.along_x, separable.along_y]
        factors[axis] = factors[axis][[-1 if last else 0]]   # the face itself (n + 1 entries along the axis)
        area = factors[0]
        for factor in factors[1:]:
            area = np.multiply.outer(area, factor)
        index = [np.arange(count) for count in shape]
        index[axis] = np.array([shape[axis] - 1 if last else 0])
        cells = np.ravel_multi_index(np.ix_(*index), shape).ravel()
        area = area.ravel()
        keep = area > 0.0
        self._add_radiating_cells(self.xp.asarray(cells[keep]), self.xp.asarray(area[keep], dtype=system.dtype),
                                  emissivity, ambient_temperature)

    def _add_radiating_cells(self, cells, area, emissivity, ambient_temperature):
        """Register one radiation term (flat window cells, radiating area [m^2 per m; m^2 in 3D; whole ring W/K
        weights axisymmetric]) and linearise it at the current T."""
        if not (0.0 <= emissivity <= 1.0 and ambient_temperature > 0.0):
            raise ValueError(f"heat '{self.name}': radiation needs 0 <= emissivity <= 1 and T_amb > 0 K, got "
                             f'{emissivity}, {ambient_temperature}')
        if not int(cells.size) or emissivity == 0.0:
            return
        zeros = self.xp.zeros(cells.shape, dtype=self.system.dtype)
        self._radiation.append([cells, area, float(emissivity) * STEFAN_BOLTZMANN, float(ambient_temperature),
                                zeros, zeros.copy(), None])
        self._linearise_radiation()

    def _linearise_radiation(self) -> float:
        """Newton linearisation of every radiation term at the current T (loss A c (T^4 - T_amb^4) ~ A (q_k + g (T -
        T_k)), g = 4 c T_k^3) written as a cell loss on the rise; returns the largest change of a radiating cell's T
        since its previous linearisation over the largest T (inf at the first)."""
        xp = self.xp
        system = self.system
        rise = self._current_rise().reshape(-1)
        change = 0.0
        for term in self._radiation:
            cells, area, coefficient, ambient, conductance_old, drive_old, linearised_at = term
            temperature = rise[cells].astype(np.float64) + self.T0
            cube = temperature * temperature * temperature
            slope = 4.0 * coefficient * cube
            loss = coefficient * (cube * temperature - ambient ** 4)
            conductance = (area * slope).astype(system.dtype)
            drive = (area * (slope * (temperature - self.T0) - loss)).astype(system.dtype)
            system.add_cell_loss(cells, conductance - conductance_old, drive - drive_old)
            term[4], term[5], term[6] = conductance, drive, temperature
            if linearised_at is None:
                change = math.inf
            elif change < math.inf:   # one host sync per term (accepted: a nonlinear loop, core sub file)
                change = max(change, float(xp.max(xp.abs(temperature - linearised_at)) / xp.max(temperature)))
        return change

    def add_heat_source(self, mask, Q) -> None:
        """Add Q [W/m^3] (scalar or full-grid array) to Q_ext on window cells under mask; 3D: also an (nx, ny, 1)
        layer broadcast along z (eval_field of x, y) or a BoxValues (added on its box: DiffusionSystem.add_masked)."""
        system = self.system
        system.add_masked(self._applied_source_window, system.window_mask(mask, 'heat source'), Q, 'heat source Q')

    def add_point_heat_source(self, ix: int, iy: int, line_power: float, iz: int = None) -> None:
        """Add a line source P [W/m] in full-grid cell (ix, iy): Q_ext += P / cell area; axisymmetric: P [W] is the
        ring's total power, Q_ext += P / ring volume (on the axis cell a point source; at an order m != 0 the order-m
        part of a point source P at phi = 0); 3D (iz given): P [W] is a point source, Q_ext += P / (dx_i dy_j dz_k)."""
        system = self.system
        system.require_window_cell(ix, iy, 'point heat source', iz)
        cell = (ix, iy) if iz is None else (ix, iy, iz)
        self.fields.Q_ext[cell] += DTYPE(float(line_power) / system.cell_volume(*cell))

    def add_point_heat_source_at(self, x: float, y: float, line_power: float, z: float = None) -> None:
        """add_point_heat_source in the cell containing physical (x, y) [m] (3D: (x, y, z), P in W)."""
        position = self._physical_position(x, y, z)
        self.domain.validate_physical_pos(*position, 'Point heat source')
        ix, iy, *iz = self.domain.pos_to_cell(*position)
        self.add_point_heat_source(ix, iy, line_power, *iz)

    def set_initial_temperature(self, mask, T) -> None:
        """Write T [K] (scalar or full-grid array; 3D also a layer or a BoxValues) on window cells under mask."""
        system = self.system
        system.copy_masked(self._T_window, system.window_mask(mask, 'initial temperature'), T, 'initial temperature')
        self._temperature_changed()

    def rebuild_operator(self) -> None:
        """Re-assemble operator, fixed-cell elimination, anchor check and the linear solver's factor / preconditioner."""
        if self._assembled_time_step is None:
            time_step = self.time_step if self.time_mode == 'transient' else math.inf
        else:
            time_step = self._assembled_time_step
        self._assemble(time_step)

    def temperature_rise(self):
        """T - T0 [K] over the window as float32, from the solve-precision rise."""
        return self._current_rise().astype(DTYPE)

    def warm_start_fields(self) -> dict:
        """Host copies of the window rise T - T0 and coupled Q, to seed a build on the same mesh (sweep warm start)."""
        return {'rise': np.array(to_host(self._current_rise())), 'Q': np.array(to_host(self._Q_window))}

    def seed_warm_start(self, fields) -> tuple:
        """Write warm_start_fields() of another build (rise on free cells, Q); returns the state slots now seeded."""
        self._current_rise()   # resync a stale rise first: the seed must not be overwritten by the next assembly
        self.system.seed_unknown(fields['rise'])
        self._rise_to_temperature(self.system.unknown2d, self.T0, self._T_window)
        self._extend_into_pml()
        self.xp.copyto(self._Q_window, self.xp.asarray(fields['Q'], dtype=DTYPE))
        return ('T', 'Q')

    def add_thermal_noise(self, monitor):
        """Register a thermal_noise.ThermalNoiseMonitor (thermal_noise study hook solve_thermal_noise fills it)."""
        if self.system.dimensions != 2 or not self.system.metric.cartesian_planar:
            raise ValueError(f"heat '{self.name}': thermal noise needs a planar 2D heat physics on a Cartesian grid")
        self.noise_monitors.append(monitor)
        return monitor

    def solve_thermal_noise(self, physics_by_name, frequencies, progress=None) -> None:
        """Thermal_noise study hook: spectra of every noise monitor at `frequencies` [Hz] (physics_by_name: the runner's
        physics, mode physics already solved); progress(done, total) per (frequency, axial sample) shift of each
        monitor; the line-block device state is released afterwards."""
        operators = {}
        shifted_solver = self.shifted_solver if self.shifted_solver is not None else SparseDirectSolver()

        def operator_for(particle=None):
            """The HarmonicHeatOperator, or with particle = (cells, diffusivity, number density) a
            HarmonicParticleOperator; one per distinct input (built on first use)."""
            if particle is None:
                key = 'heat'
                if key not in operators:
                    operators[key] = HarmonicHeatOperator(self, shifted_solver)
            else:
                cells, diffusivity, number_density = particle
                key = (cells.tobytes(), float(diffusivity), float(number_density))
                if key not in operators:
                    operators[key] = HarmonicParticleOperator(self, cells, diffusivity, number_density,
                                                              shifted_solver)
            return operators[key]

        try:
            for monitor in self.noise_monitors:
                solve_noise_monitor(self, monitor, physics_by_name, frequencies, operator_for, progress)
        finally:
            if isinstance(shifted_solver, LineBlockSolver):
                shifted_solver.release()   # device state only: the work record stays

    def heat_flux_magnitude(self):
        """Conductive heat-flux magnitude [W/m^2] at window cell centres, float32 (axis faces + the k_xy cross flux; 0
        in fixed-temperature cells; 3D: over x, y, z); re-assembles first when k, BCs or held cells changed."""
        self._refresh_post_processing()
        flux_components = self.system.flux_at_centres()
        return self.xp.sqrt(sum(component * component for component in flux_components)).astype(DTYPE)

    def error_indicator(self):
        """S7.7 hook: the Kelly heat-flux jump eta^2 parts (2, *window) of DiffusionPhysicsBase, after the re-assembly /
        rise resync of heat_flux_magnitude."""
        self._refresh_post_processing()
        return super().error_indicator()

    # --- internals ------------------------------------------------------------

    def _refresh_post_processing(self):
        """Re-assemble when k, BCs or held cells changed (also resyncs a stale rise), else resync a stale rise."""
        if self.system.operator_dirty:
            self.rebuild_operator()
        elif self._theta_stale:
            self._sync_rise_from_temperature()

    def _assemble(self, time_step):
        """Assemble the system with the window k arrays (+ rho cp area / dt for a finite time_step)."""
        if self._theta_stale:
            self._sync_rise_from_temperature()
        materials = self.domain.materials
        system = self.system
        window = system.window
        capacity_over_dt = self._capacity_over_dt(time_step) if math.isfinite(time_step) else None
        c_zz = materials.k_zz[window] if system.dimensions == 3 else None
        system.assemble(materials.k_xx[window], materials.k_yy[window], materials.k_xy[window],
                        self.bcs, self.T0, capacity_over_dt, c_zz=c_zz)
        self._assembled_time_step = time_step
        self._capacity_stale = False

    def _capacity_over_dt(self, time_step):
        """rho cp x cell area / time_step [W/K] on the window (solve dtype)."""
        materials = self.domain.materials
        system = self.system
        window = system.window
        dtype = system.dtype
        heat_capacity = materials.rho[window].astype(dtype) * materials.cp[window].astype(dtype) * system.cell_area
        return heat_capacity / time_step

    def _sync_rise_from_temperature(self):
        self._temperature_to_rise(self._T_window, self.T0, self.system.unknown2d)
        self._theta_stale = False

    def _current_rise(self):
        if self._theta_stale:
            self._sync_rise_from_temperature()
        return self.system.unknown2d

    def _temperature_changed(self):
        # T was written directly: resync the rise, re-pin empty cells, refresh the PML ring.
        self._theta_stale = True
        self.system.operator_dirty = True
        self._extend_into_pml()
