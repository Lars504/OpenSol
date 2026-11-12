"""Incompressible laminar flow on the physical window by a D2Q9 lattice-Boltzmann method (TRT, He-Luo, Guo forcing):
steady (march to a steady state per solve) or transient (one lattice step per runner step); arrays on the Domain's
array module, kernels core.kernels twins (gpu and cpu backends)."""
import math
import warnings
from dataclasses import dataclass

import numpy as np

from ...core.backend import array_module_of, to_host
from ...core.constants import DTYPE, FACES
from ...core.fv_diffusion import edge_extension_views
from ...core.half_storage import half_bits
from ...core.monitors import FieldProbe
from ...core.solver_base import PhysicsSolver
from .lattice import (BLOCK_SIZE, FACE_KIND_NAMES, FACE_PERIODIC, FACE_PRESSURE, FACE_SLIP, FACE_VELOCITY, FACE_WALL,
                      MAGIC_LAMBDA, N_DIRECTIONS, POPULATION_DTYPES, POPULATION_SCALE, STORE_BOUNDARY, STORE_LATTICE,
                      STORE_PUBLISHED, equilibrium_populations, force_to_lattice, step_kernel, store_populations,
                      to_physical)
from .bouzidi import wall_link_fractions
from .renderer import FluidRenderer

FLUID_SCHEMA = {'viscosity': 0.0, 'rho': 0.0}   # viscosity 0 = solid (no-slip obstacle); rho shared with heat
FLUID_PROPERTY_ARRAYS = {'viscosity': ('viscosity',), 'rho': ('rho',)}
FORCE_SLOTS = ('body_force_x', 'body_force_y')
PROBE_QUANTITIES = {'u_abs': 'm/s', 'u_x': 'm/s', 'u_y': 'm/s', 'pressure': 'Pa'}
TIME_MODES = ('stationary', 'transient')
MACH_WARNING = 0.2           # lattice Mach number sqrt(3) |u_lat| above which accuracy degrades (runtime warning)
MACH_LIMIT = 0.7             # ... above which the run fails (the lattice equilibrium no longer holds)
MIN_TAU_OFFSET = 1e-3        # tau+ - 1/2 below which a cell's relaxation degenerates (run fails)
TAU_OFFSET_WARNING = 0.02    # ... below which low-viscosity cells lose stability (runtime warning)
FLOAT32_STALL_CHANGE = 1e-5  # relative velocity change per check interval that stopped decreasing below this is
                             # float32 rounding noise (~1e-7..1e-6 of the peak): steady, whatever the tolerance
MIN_CHECK_DECAY = 0.05       # decay exponent per steady check interval below which the interval doubles (the old
                             # fixed interval: 5 % of the slowest decay), so a check resolves the decay above the noise
STEP_GRAPHS = True           # gpu: runs of steps without a host read replay from CUDA graphs (_fixed_block)
GRAPH_STEPS = 32             # lattice steps per graph (even: the population buffers end where they started)
_MAX_STEP_GRAPHS = 4         # graph keys kept (stale ones belong to replaced wall tables / relaxation / force args)
_AXIS_FACES = {'x': ('x_lo', 'x_hi'), 'y': ('y_lo', 'y_hi')}
_INWARD_NORMAL = {'x_lo': (1.0, 0.0), 'x_hi': (-1.0, 0.0), 'y_lo': (0.0, 1.0), 'y_hi': (0.0, -1.0)}


def steady_march_steps(window_shape, shear_free_counts, periodic_axes, relaxation_time, viscosity_ratio, max_steps):
    """(first check interval, largest check interval, step limit) of a steady march. Worst-case decay L^2 / (pi^2 nu_lat):
    L the longest window side, doubled with exactly one shear-free face (slip / pressure), halved on a periodic axis
    (slowest wavenumber 2 pi / L). The first interval max(100, 3.5 L) lets sound cross; max_steps 0 = auto, 30 decays."""
    decay_cells = max(cells / 2.0 if periodic else (2 * cells if n_shear_free == 1 else cells)
                      for cells, n_shear_free, periodic in zip(window_shape, shear_free_counts, periodic_axes))
    slowest_lattice_viscosity = (relaxation_time - 0.5) * viscosity_ratio / 3.0   # viscosity_ratio = nu_min / nu_max
    decay_steps = decay_cells ** 2 / (math.pi ** 2 * slowest_lattice_viscosity)
    first_interval = max(100, int(3.5 * max(window_shape)))
    largest_interval = max(first_interval, int(MIN_CHECK_DECAY * decay_steps))
    return first_interval, largest_interval, int(max_steps) or max(10 * largest_interval, int(30.0 * decay_steps))


def decay_per_check(change_ratio, length_ratio):
    """Decay factor over the latest check interval of one exponential mode from consecutive changes c_k / c_(k-1) over
    intervals of length ratio T_k / T_(k-1); None when the changes do not decay."""
    if not 0.0 < change_ratio < length_ratio:
        return None
    if length_ratio == 1.0:
        return change_ratio
    # c_k / c_(k-1) = x (1 - x^m) / (1 - x), x the decay over the previous interval: rises from 0 to m on (0, 1)
    low, high = 0.0, 1.0
    for _ in range(60):
        previous_decay = 0.5 * (low + high)
        if previous_decay * (1.0 - previous_decay ** length_ratio) / (1.0 - previous_decay) < change_ratio:
            low = previous_decay
        else:
            high = previous_decay
    return (0.5 * (low + high)) ** length_ratio


def march_steps_estimate(decay_steps, decay_times, first_interval, largest_interval, step_limit):
    """Lattice steps of _march_to_steady on one mode decaying as exp(-steps / decay_steps): checks from first_interval,
    doubled up to largest_interval while one decays less than MIN_CHECK_DECAY, until decay_times decays have passed on
    two checks in a row (never at the first check), capped at step_limit."""
    steps, interval, checks, converged_checks = 0, first_interval, 0, 0
    while steps < step_limit:
        steps += min(interval, step_limit - steps)
        checks += 1
        converged_checks = converged_checks + 1 if checks >= 2 and steps >= decay_steps * decay_times else 0
        if converged_checks >= 2:
            break
        if checks >= 2 and not converged_checks and interval < MIN_CHECK_DECAY * decay_steps:
            interval = min(2 * interval, largest_interval)
    return steps


@dataclass
class FaceCondition:
    """Condition of one outer window face: kind (lattice FACE_* code) and its SI profile along the face (host numpy
    float64 1D: tangential wall velocity [m/s], inward normal velocity [m/s] or pressure [Pa]; None for slip /
    periodic)."""
    kind: int
    profile: object = None
    parabolic: bool = False


class FluidFields2D:
    """Full-grid float32 SI fields: velocity u_x, u_y and its magnitude u_abs [m/s], pressure [Pa], coupled
    body_force_x / _y and applied body_force_x_ext / _y_ext [N/m^3], on array module xp."""

    def __init__(self, nx: int, ny: int, xp):
        self.nx, self.ny = nx, ny
        for name in ('u_x', 'u_y', 'u_abs', 'pressure', 'body_force_x', 'body_force_y', 'body_force_x_ext',
                     'body_force_y_ext'):
            setattr(self, name, xp.zeros((nx, ny), dtype=DTYPE))


class FluidPhysics(PhysicsSolver):
    """2D incompressible laminar flow on the physical window (uniform square cells): cells with viscosity > 0 are fluid,
    the rest are no-slip solids; outer faces default to no-slip walls."""

    name = 'fluid'

    def __init__(self, domain, time_mode: str = 'stationary', relaxation_time: float = 1.0, tolerance: float = 1e-6,
                 max_steps: int = 0, precision: str = 'single'):
        if time_mode not in TIME_MODES:
            raise ValueError(f"time_mode must be one of {TIME_MODES}, got {time_mode!r}")
        if precision not in POPULATION_DTYPES:
            raise ValueError(f"precision must be one of {', '.join(POPULATION_DTYPES)}, got {precision!r}")
        if precision == 'half_storage' and time_mode != 'transient':
            raise ValueError('half_storage (fp16 populations) runs transient flow only: the steady march already sits '
                             'at the float32 rounding floor')
        self.precision = precision
        if not 0.5 < relaxation_time:
            raise ValueError(f"relaxation_time must be > 0.5, got {relaxation_time}")
        if not tolerance > 0.0 or int(max_steps) < 0:
            raise ValueError(f"need tolerance > 0 and max_steps >= 0, got {tolerance}, {max_steps}")
        self.time_mode = time_mode
        self.relaxation_time = float(relaxation_time)
        self.tolerance = float(tolerance)
        self.max_steps = int(max_steps)
        self.material_schema = dict(FLUID_SCHEMA)
        self.faces = {face: FaceCondition(FACE_WALL) for face in FACES}
        self.probes = []
        self.runtime_warnings = []
        self.n_solves = 0              # steady marches / transient step calls (coupling state revisions)
        self.n_unconverged = 0
        self.last_warning = None
        self.total_steps = 0           # lattice steps over the run (work_counters)
        self.last_march_steps = 0
        self.last_remaining = math.nan   # estimated remaining relative velocity change of the last steady march
        self.peak_mach = 0.0
        self.dt_lbm = None             # lattice time step of relaxation_time on the most viscous fluid [s]
        self.fields = None
        self._lattice_dt = None        # dt the populations are scaled for
        self._lattice_dirty = True     # fluid mask / relaxation rates / step budget need a refresh (materials, faces)
        self._force_dirty = True
        self._solution_dirty = True
        self._faces_dirty = True
        self._mach_warned = False
        self._vorticity = None
        self._relaxation_dt = None     # dt the relaxation rates were built for
        self._check_interval = 100     # first / transient lattice steps between host checks (set by _refresh_lattice)
        self._largest_check_interval = 100   # a steady march doubles its interval up to this
        self._max_steps = 0
        self._last_check_step = 0
        # Interpolated walls (solver_defaults.conformal_boundaries): the geometry locator, the host fluid mask the links
        # were built for (and their periodic axis spans), device wall rows / fractions (bouzidi.wall_link_fractions);
        # None = halfway bounce-back.
        self._locator = None
        self._wall_mask = None
        self._wall_periodic_spans = None
        self._wall_row = None
        self._wall_fraction = None
        self.n_wall_links = 0
        self._initial_flow = None      # window (u_x, u_y [m/s], p [Pa]) of set_initial_flow until finalize applies them
        self._force_present = False    # a body force was ever added / injected (else the lattice force is all 0)
        # (key) -> [CUDA graph of GRAPH_STEPS launches or None after its eager run, kept inputs] (_fixed_block; gpu)
        self._step_graphs = {}
        self.n_step_graphs = 0         # graph captures made
        self.prepare(domain)

    # --- PhysicsSolver contract ---------------------------------------------

    def prepare(self, domain) -> None:
        """Register the material arrays, check the window is uniform with square cells, allocate fields and lattice."""
        self.domain = domain
        self.xp = xp = domain.xp
        domain.materials.add_properties(self.material_schema)
        nx, ny = domain.grid.nx, domain.grid.ny
        window_x, window_y = domain.physical_window()
        x_start, x_stop, _ = window_x.indices(nx)
        y_start, y_stop, _ = window_y.indices(ny)
        self.window = (slice(x_start, x_stop), slice(y_start, y_stop))
        self.window_shape = (x_stop - x_start, y_stop - y_start)
        dx_cells, dy_cells = domain.cell_widths()
        widths = np.concatenate((np.asarray(dx_cells)[x_start:x_stop], np.asarray(dy_cells)[y_start:y_stop]))
        if not np.allclose(widths, widths[0], rtol=1e-6, atol=0.0):
            raise ValueError(f"laminar flow '{self.name}' needs a uniform mesh with square cells on the physical window "
                             f'(dx from {widths.min():.4g} to {widths.max():.4g} m): set the mesh mode to uniform')
        self.dx = float(widths[0])
        wx, wy = self.window_shape
        self.face_length = max(wx, wy)
        if (self.fields is None or (self.fields.nx, self.fields.ny) != (nx, ny)
                or array_module_of(self.fields.u_x) is not xp):
            self.fields = FluidFields2D(nx, ny, xp)
            self._force_present = False
        lattice_shape = (N_DIRECTIONS, wx, wy)
        population_dtype = POPULATION_DTYPES[self.precision]
        self._populations = [xp.zeros(lattice_shape, dtype=population_dtype),
                             xp.zeros(lattice_shape, dtype=population_dtype)]
        self._lattice_x = xp.zeros(self.window_shape, dtype=DTYPE)
        self._lattice_y = xp.zeros(self.window_shape, dtype=DTYPE)
        self._deviation = xp.zeros(self.window_shape, dtype=DTYPE)
        self._saved_x = xp.zeros(self.window_shape, dtype=DTYPE)
        self._saved_y = xp.zeros(self.window_shape, dtype=DTYPE)
        self._fluid = xp.zeros(self.window_shape, dtype=np.bool_)
        self._mass_density = xp.zeros(self.window_shape, dtype=DTYPE)
        self._omega_plus = xp.ones(self.window_shape, dtype=DTYPE)
        self._omega_minus = xp.ones(self.window_shape, dtype=DTYPE)
        # one fluid viscosity on the gpu: the kernel reads these 1-element rates (relaxation_stride 0, _update_relaxation)
        self._uniform_omega_plus = xp.ones(1, dtype=DTYPE)
        self._uniform_omega_minus = xp.ones(1, dtype=DTYPE)
        self._relaxation_stride = np.int32(1)
        self._force_lattice_x = xp.zeros(self.window_shape, dtype=DTYPE)
        self._force_lattice_y = xp.zeros(self.window_shape, dtype=DTYPE)
        self._face_kind = xp.zeros(len(FACES), dtype=np.int32)
        self._uploaded_face_kinds = None   # host tuple of _face_kind (skip the upload when a value of t keeps the kinds)
        self._face_values = xp.zeros((len(FACES), 3, self.face_length), dtype=DTYPE)
        self._face_values_host = np.zeros((len(FACES), 3, self.face_length), dtype=DTYPE)   # built here, one upload
        self._edge_density = None   # {face: host float32 mass density along the face row} (0 = solid), per refresh
        self._no_wall_row = xp.full(1, -1, dtype=np.int32)        # kernel stand-ins while halfway (no wall links)
        self._no_wall_fraction = xp.zeros(N_DIRECTIONS - 1, dtype=DTYPE)
        self._wall_mask = self._wall_row = self._wall_fraction = None
        self._kinematic_viscosity = None   # float64 window [m^2/s], 0 in solid cells
        self._grid_args = (np.int32(wx), np.int32(wy), np.int32(self.face_length))
        self._publish_grid = (np.int32(ny), np.int32(x_start), np.int32(y_start))   # full-grid row, window origin
        self._launch = ((wx * wy + BLOCK_SIZE - 1) // BLOCK_SIZE,), (BLOCK_SIZE,)
        self._step_graphs = {}
        self.use_step_graphs = STEP_GRAPHS and xp is not np
        fields = self.fields
        window = self.window
        self._field_windows = tuple(getattr(fields, name)[window] for name in ('u_x', 'u_y', 'u_abs', 'pressure'))
        x_range, y_range = (x_start, x_stop), (y_start, y_stop)
        self._pml_copies = [pair for name in ('u_x', 'u_y', 'u_abs', 'pressure')
                            for pair in edge_extension_views(getattr(fields, name), x_range, y_range)]
        self._lattice_dt = None
        self._lattice_dirty = self._force_dirty = self._solution_dirty = self._faces_dirty = True

    def finalize(self) -> None:
        """Read the materials (fluid cells, relaxation rates, lattice time step) and the face conditions once; start
        from the initial flow of set_initial_flow (equilibrium populations at dt_lbm)."""
        self._refresh_lattice()
        if self._initial_flow is not None:
            self._start_from_initial_flow()

    def materials_changed(self, names) -> None:
        if set(FLUID_SCHEMA) & set(names):
            self._lattice_dirty = self._force_dirty = self._solution_dirty = self._faces_dirty = True

    def step(self, t: float = 0.0, dt: float = math.inf) -> None:
        """stationary: march to a steady state when the materials, forces or faces changed (always at the first call);
        transient: advance dt in ceil(dt / dt_lbm) lattice steps. Probes sample after each solve / step and at every
        dt = inf call."""
        t = 0.0 if t is None else float(t)
        finite_step = dt is not None and math.isfinite(dt)
        if self._lattice_dirty:
            self._refresh_lattice()
        if self.time_mode == 'transient':
            if not finite_step:
                raise ValueError(f"laminar flow '{self.name}' is transient: step needs a finite dt (use time_mode "
                                 "'stationary' for a steady solve)")
            n_steps = max(1, math.ceil(dt / self.dt_lbm * (1.0 - 1e-9)))
            self._prepare_lattice(dt / n_steps)
            self._run_steps(n_steps, STORE_PUBLISHED)   # the last step publishes the SI fields (lattice_to_physical)
            if self.total_steps - self._last_check_step >= self._check_interval:
                self._velocity_change()   # health check (NaN, Mach) once per check interval
            self._extend_into_ring()
            self.n_solves += 1
            self._record_probes(t + dt)
            return
        re_solve = self._solution_dirty or self.n_solves == 0
        if re_solve:
            self._prepare_lattice(self.dt_lbm)
            self._march_to_steady()
            self._publish()
            self.n_solves += 1
            self._solution_dirty = False
        if re_solve or not finite_step:
            self._record_probes(t)

    def dt_limit(self) -> float:
        if self.time_mode == 'stationary':
            return math.inf
        if self._lattice_dirty:
            self._refresh_lattice()
        return self.dt_lbm

    def state(self) -> dict:
        fields = self.fields
        return {'u_x': fields.u_x, 'u_y': fields.u_y, 'u_abs': fields.u_abs, 'pressure': fields.pressure,
                'body_force_x': fields.body_force_x, 'body_force_y': fields.body_force_y,
                'viscosity': self.domain.materials.viscosity}

    def inject_source(self, name: str, array) -> None:
        if name not in FORCE_SLOTS:
            raise ValueError(f"laminar flow '{self.name}' accepts sources {list(FORCE_SLOTS)}; got '{name}'")
        self.xp.copyto(getattr(self.fields, name), self.xp.asarray(array, dtype=DTYPE))
        self._force_dirty = self._solution_dirty = self._force_present = True

    def renderer(self):
        return FluidRenderer(self)

    # --- fluid API (features, scripts) -------------------------------------------

    def set_wall(self, face: str, tangential_velocity=0.0) -> None:
        """No-slip wall on an outer face, moving along it at tangential_velocity [m/s] (+x on y faces, +y on x faces;
        scalar or full-grid array)."""
        self._set_face(face, FaceCondition(FACE_WALL, self._face_profile(face, tangential_velocity)))

    def set_slip(self, face: str) -> None:
        """Free-slip (symmetry) face: specular reflection, no normal flow, no shear."""
        self._set_face(face, FaceCondition(FACE_SLIP))

    def set_velocity_inlet(self, face: str, normal_velocity, profile: str = 'uniform') -> None:
        """Prescribed inward normal velocity [m/s] (scalar or full-grid array); 'parabolic' multiplies it by the
        discrete Poiseuille shape (mean 1) across each run of fluid cells along the face."""
        if profile not in ('uniform', 'parabolic'):
            raise ValueError(f"profile must be 'uniform' or 'parabolic', got {profile!r}")
        self._set_face(face, FaceCondition(FACE_VELOCITY, self._face_profile(face, normal_velocity),
                                           profile == 'parabolic'))

    def set_pressure(self, face: str, pressure) -> None:
        """Prescribed pressure [Pa] (scalar or full-grid array) on an outer face: inlet or outlet."""
        self._set_face(face, FaceCondition(FACE_PRESSURE, self._face_profile(face, pressure)))

    def set_periodic(self, axis: str) -> None:
        """Periodic pair of faces along axis 'x' or 'y' (flow leaving one face re-enters through the other)."""
        if axis not in _AXIS_FACES:
            raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
        for face in _AXIS_FACES[axis]:
            self._set_face(face, FaceCondition(FACE_PERIODIC))

    def set_face_value(self, face: str, value) -> None:
        """New SI value (scalar or full-grid array) of a wall / velocity / pressure face that keeps its kind (a
        time-dependent feature value): the lattice face values refresh before the next step, no lattice rebuild."""
        if face not in FACES:
            raise ValueError(f"unknown face {face!r}; expected one of {FACES}")
        condition = self.faces[face]
        if condition.profile is None:
            raise ValueError(f"laminar flow '{self.name}': face {face} ({FACE_KIND_NAMES[condition.kind]}) has no "
                             'value to set')
        condition.profile = self._face_profile(face, value)
        self._faces_dirty = self._solution_dirty = True

    def set_initial_flow(self, mask, u_x=0.0, u_y=0.0, pressure=0.0) -> None:
        """Initial velocity [m/s] and gauge pressure [Pa] (scalars or full-grid arrays) on the window cells under
        mask (later calls win their cells): the fluid cells start at the equilibrium populations of that flow at
        finalize, where dt_lbm is known (transient: the state at t = 0; stationary: the start of the first march)."""
        xp = self.xp
        mask = xp.asarray(mask)
        if mask.shape != (self.fields.nx, self.fields.ny):
            raise ValueError(f'initial flow mask must be a full-grid {(self.fields.nx, self.fields.ny)} array, got '
                             f'{mask.shape}')
        if self._initial_flow is None:
            self._initial_flow = tuple(xp.zeros(self.window_shape, dtype=DTYPE) for _ in range(3))
        mask_window = mask[self.window]
        for target, value in zip(self._initial_flow, (u_x, u_y, pressure)):
            values = xp.asarray(value, dtype=DTYPE)
            xp.copyto(target, values[self.window] if values.ndim else values, where=mask_window)

    def add_body_force(self, mask, force_x=0.0, force_y=0.0) -> None:
        """Add a body force density [N/m^3] (scalars or full-grid arrays) to body_force_*_ext on window cells under
        mask."""
        xp = self.xp
        mask = xp.asarray(mask)
        if mask.shape != (self.fields.nx, self.fields.ny):
            raise ValueError(f'body force mask must be a full-grid {(self.fields.nx, self.fields.ny)} array, got '
                             f'{mask.shape}')
        for name, value in (('body_force_x_ext', force_x), ('body_force_y_ext', force_y)):
            if np.ndim(value) == 0 and array_module_of(value) is np:   # host scalar (a value of t): no device transfer
                if value == 0.0:
                    continue
                value = DTYPE(value)
            else:
                value = xp.asarray(value, dtype=DTYPE)
            target = getattr(self.fields, name)
            target += xp.where(mask, value, DTYPE(0.0))
            self._force_present = True
        self._force_dirty = self._solution_dirty = True

    def add_probe(self, ix: int, iy: int, component: str = 'u_abs') -> FieldProbe:
        """Record `component` (PROBE_QUANTITIES) at full-grid window cell (ix, iy) whenever step records probes."""
        if component not in PROBE_QUANTITIES:
            raise ValueError(f"probe component must be one of {list(PROBE_QUANTITIES)}, got {component!r}")
        window_x, window_y = self.window
        if not (window_x.start <= ix < window_x.stop and window_y.start <= iy < window_y.stop):
            raise ValueError(f'flow probe cell ({ix}, {iy}) lies outside the physical window')
        probe = FieldProbe(ix, iy, component)
        self.probes.append(probe)
        return probe

    def add_probe_at(self, x: float, y: float, component: str = 'u_abs') -> FieldProbe:
        """add_probe in the cell containing physical (x, y) [m]."""
        self.domain.validate_physical_pos(x, y, 'Flow probe')
        return self.add_probe(*self.domain.pos_to_cell(x, y), component)

    def set_conformal_boundaries(self, locator):
        """Interpolated (Bouzidi) bounce-back at the drawn solid walls: locator.material_index(x, y) (model.
        boundary_locator.GeometryLocator, duck-typed) locates each wall along its links; None = halfway."""
        self._locator = locator
        self._wall_mask = self._wall_row = self._wall_fraction = None
        self._lattice_dirty = True

    @property
    def fluid_mask(self):
        """bool window mask of the fluid cells on the Domain's array module (valid after finalize)."""
        return self._fluid

    def work_counters(self) -> dict:
        return {'lattice_steps': self.total_steps, 'solves': self.n_solves}

    def solve_summary(self) -> str:
        walls = f', {self.n_wall_links} interpolated wall links' if self._wall_row is not None else ''
        dt_text = 'not set (before finalize)' if self.dt_lbm is None else f'{self.dt_lbm:.4g} s'
        return (f'lattice Boltzmann D2Q9 TRT, {int(self._fluid.sum())} fluid cells{walls}, '
                f'dt_lbm = {dt_text}, {self.total_steps} lattice steps (last steady march {self.last_march_steps}), peak lattice Mach '
                f'{self.peak_mach:.3g}')

    # --- internals ------------------------------------------------------------

    def _set_face(self, face, condition):
        if face not in FACES:
            raise ValueError(f"unknown face {face!r}; expected one of {FACES}")
        self.faces[face] = condition
        self._faces_dirty = self._solution_dirty = True
        self._lattice_dirty = True   # the check interval / step limit depend on the face kinds

    def _face_profile(self, face, value):
        """Host float64 1D profile along the face from a scalar (no device work: values of t) or the window boundary row
        of a full-grid array (one transfer when it lives on the device)."""
        if face not in FACES:
            raise ValueError(f"unknown face {face!r}; expected one of {FACES}")
        window_x, window_y = self.window
        wx, wy = self.window_shape
        length = wy if face.startswith('x') else wx
        if np.ndim(value) == 0:
            return np.full(length, float(value), dtype=np.float64)
        array = self.xp.asarray(value, dtype=np.float64)
        if array.shape != (self.fields.nx, self.fields.ny):
            raise ValueError(f'face values must be a scalar or a full-grid array, got shape {array.shape}')
        row = {'x_lo': array[window_x.start, window_y], 'x_hi': array[window_x.stop - 1, window_y],
               'y_lo': array[window_x, window_y.start], 'y_hi': array[window_x, window_y.stop - 1]}[face]
        return np.array(to_host(row), dtype=np.float64)   # a copy: on the cpu to_host returns the live view

    def _refresh_lattice(self):
        """Fluid mask, SI mass density and kinematic viscosity from the materials; dt_lbm; reset cells that became
        fluid to rest. Raises when no cell is fluid or a fluid cell has no density."""
        xp = self.xp
        materials = self.domain.materials
        viscosity = materials.viscosity[self.window].astype(np.float64)
        density = materials.rho[self.window].astype(np.float64)
        fluid = viscosity > 0.0
        n_fluid, n_without_density = (int(value) for value in
                                      to_host(xp.stack([fluid.sum(), (fluid & ~(density > 0.0)).sum()])))
        if n_fluid == 0:
            raise ValueError(f"laminar flow '{self.name}': no window cell has a viscosity > 0; assign a fluid material "
                             '(viscosity and rho) to the flow region')
        if n_without_density:
            raise ValueError(f"laminar flow '{self.name}': {n_without_density} fluid cells (viscosity > 0) have no "
                             'density rho > 0; set rho on every fluid material')
        kinematic = xp.where(fluid, viscosity / xp.where(fluid, density, 1.0), 0.0)
        keep = fluid & self._fluid   # cells fluid before and after keep their state; new fluid cells start at rest
        for array in (*self._populations, self._lattice_x, self._lattice_y, self._deviation):
            stored = half_bits(array)   # fp16 bits x keep: 0 bits = +0.0, no cupy float16 kernel
            stored *= keep
        xp.copyto(self._fluid, fluid)
        if self._locator is not None:
            self._refresh_wall_links()
        xp.copyto(self._mass_density, xp.where(fluid, density, 0.0).astype(DTYPE))
        # host face rows of the mass density (> 0 exactly on fluid cells) for the face values: one transfer per refresh
        density_rows = {'x_lo': self._mass_density[0, :], 'x_hi': self._mass_density[-1, :],
                        'y_lo': self._mass_density[:, 0], 'y_hi': self._mass_density[:, -1]}
        edge_rows = np.array(to_host(xp.concatenate([density_rows[face] for face in FACES])))
        offsets = np.cumsum([0] + [density_rows[face].size for face in FACES])
        self._edge_density = {face: edge_rows[offsets[index]:offsets[index + 1]] for index, face in enumerate(FACES)}
        self._kinematic_viscosity = kinematic
        nu_max, nu_min = (float(value) for value in to_host(xp.stack([
            kinematic.max(), xp.where(fluid, kinematic, np.inf).min()])))
        self.dt_lbm = (self.relaxation_time - 0.5) * self.dx ** 2 / (3.0 * nu_max)
        # check interval and auto step limit from the slowest diffusive decay (lowest lattice viscosity)
        periodic_axes = tuple(self.faces[lo].kind == FACE_PERIODIC for lo, _ in _AXIS_FACES.values())
        self._check_interval, self._largest_check_interval, self._max_steps = steady_march_steps(
            self.window_shape, self._shear_free_counts(), periodic_axes, self.relaxation_time, nu_min / nu_max,
            self.max_steps)
        self._last_check_step = self.total_steps
        self._lattice_dirty = False
        self._relaxation_dt = None   # relaxation rates are rebuilt by _prepare_lattice
        self._force_dirty = self._faces_dirty = True

    def _refresh_wall_links(self):
        """Wall rows / fractions of the current fluid mask (host bisection on the drawn geometry; skipped while the
        mask and the periodic axes are unchanged). A material is solid where most of its window cells have viscosity 0."""
        fluid = to_host(self._fluid).copy()   # a copy: on the cpu to_host returns the live mask itself
        # links wrap across a periodic axis like the kernel's streaming: the window edges as the period
        periodic_spans = tuple(
            (float(edges[window.start]), float(edges[window.stop]))
            if self.faces[axis_faces[0]].kind == FACE_PERIODIC else None
            for edges, window, axis_faces in zip(self.domain.cell_edges(), self.window, _AXIS_FACES.values()))
        if (self._wall_mask is not None and np.array_equal(fluid, self._wall_mask)
                and periodic_spans == self._wall_periodic_spans):
            return
        x_centres, y_centres = (np.asarray(centres, dtype=np.float64)[window]
                                for centres, window in zip(self.domain.cell_centers(), self.window))
        index = self._locator.material_index(x_centres[:, None], y_centres[None, :]).ravel()
        cells = np.bincount(index)
        fluid_material = np.bincount(index, weights=fluid.ravel(), minlength=cells.size) > 0.5 * cells
        locator = self._locator

        def solid(x, y):
            material = locator.material_index(x, y)
            return ~fluid_material[np.minimum(material, fluid_material.size - 1)] | (material >= fluid_material.size)
        rows, fractions = wall_link_fractions(fluid, x_centres, y_centres, solid, periodic_spans)
        self._wall_mask = fluid
        self._wall_periodic_spans = periodic_spans
        self.n_wall_links = int(np.count_nonzero(fractions != 0.5))
        xp = self.xp
        self._wall_row = xp.asarray(rows)
        self._wall_fraction = xp.asarray(fractions) if fractions.size else xp.zeros(N_DIRECTIONS - 1, dtype=DTYPE)

    def _shear_free_counts(self):
        """Shear-free faces (slip / pressure) per axis; exactly one makes the slowest mode a quarter wave (free-surface
        film)."""
        return tuple(sum(self.faces[face].kind in (FACE_SLIP, FACE_PRESSURE) for face in axis_faces)
                     for axis_faces in _AXIS_FACES.values())

    def _prepare_lattice(self, dt):
        """Scale the lattice for time step dt [s] (re-equilibrate the populations when dt changes), then refresh the
        relaxation rates, forces and face values that changed."""
        if self._lattice_dt is not None and abs(dt - self._lattice_dt) > 1e-12 * dt:
            self._rescale_populations(self._lattice_dt, dt)
        self._lattice_dt = dt
        if self._relaxation_dt != dt:
            self._update_relaxation(dt)
        if self._force_dirty:
            scale = DTYPE(dt * dt / self.dx)
            window = self.window
            fields = self.fields
            for coupled, applied, out in ((fields.body_force_x, fields.body_force_x_ext, self._force_lattice_x),
                                          (fields.body_force_y, fields.body_force_y_ext, self._force_lattice_y)):
                force_to_lattice(coupled[window], applied[window], self._fluid, self._mass_density, scale, out)
            self._force_dirty = False
        if self._faces_dirty:
            self._update_face_values(dt)
            self._faces_dirty = False

    def _update_relaxation(self, dt):
        """TRT rates per cell: tau+ = 1/2 + 3 nu dt / dx^2, (tau+ - 1/2)(tau- - 1/2) = 3/16; one fluid viscosity on the
        gpu: the 1-element rates too (relaxation_stride 0, 8 bytes per cell and step fewer)."""
        xp = self.xp
        tau_offset = 3.0 * self._kinematic_viscosity * dt / self.dx ** 2
        smallest, largest = (float(value) for value in to_host(xp.stack([
            xp.where(self._fluid, tau_offset, np.inf).min(), xp.where(self._fluid, tau_offset, -np.inf).max()])))
        if smallest < MIN_TAU_OFFSET:
            raise ValueError(f"laminar flow '{self.name}': time step {dt:.4g} s gives a lattice relaxation time "
                             f'tau = 0.5 + {smallest:.3g} in the least viscous fluid (needs > 0.5 + {MIN_TAU_OFFSET:g}); '
                             'a shorter step than the fluid can take (do not run transient flow on a wave time step), '
                             'or viscosities that differ too much between the fluids')
        if smallest < TAU_OFFSET_WARNING:
            self.runtime_warnings.append(f'lattice relaxation time 0.5 + {smallest:.3g} in the least viscous fluid: '
                                         'close to the stability limit (raise relaxation_time)')
        tau_offset = xp.where(self._fluid, tau_offset, 0.5)
        xp.copyto(self._omega_plus, (1.0 / (0.5 + tau_offset)).astype(DTYPE))
        xp.copyto(self._omega_minus, (1.0 / (0.5 + MAGIC_LAMBDA / tau_offset)).astype(DTYPE))
        uniform = smallest == largest and xp is not np   # host twins keep the per-cell arrays (one numba signature)
        if uniform:   # the per-cell float64 expressions on the same double: the same float32 rates
            self._uniform_omega_plus.fill(DTYPE(1.0 / (0.5 + smallest)))
            self._uniform_omega_minus.fill(DTYPE(1.0 / (0.5 + MAGIC_LAMBDA / smallest)))
        self._relaxation_stride = np.int32(0 if uniform else 1)
        self._relaxation_dt = dt

    def _update_face_values(self, dt):
        """Face kinds and lattice face values (velocity x / y, density deviation) for time step dt: built on the host
        from the host profiles and edge rows (no device work per face), then one upload."""
        velocity_scale = dt / self.dx
        kinds = []
        values_host = self._face_values_host
        values_host.fill(0.0)
        for index, face in enumerate(FACES):
            condition = self.faces[face]
            kinds.append(condition.kind)
            profile = condition.profile
            if profile is None:
                continue
            length = profile.size
            values = values_host[index]
            if condition.kind == FACE_WALL:
                tangential_x, tangential_y = (0.0, 1.0) if face.startswith('x') else (1.0, 0.0)
                values[0, :length] = profile * tangential_x * velocity_scale
                values[1, :length] = profile * tangential_y * velocity_scale
            elif condition.kind == FACE_VELOCITY:
                if condition.parabolic:
                    profile = profile * _parabolic_shape(self._edge_density[face] > 0.0)
                normal_x, normal_y = _INWARD_NORMAL[face]
                values[0, :length] = profile * normal_x * velocity_scale
                values[1, :length] = profile * normal_y * velocity_scale
            elif condition.kind == FACE_PRESSURE:
                density = self._edge_density[face].astype(np.float64)
                values[2, :length] = np.where(density > 0.0, 3.0 * profile * velocity_scale ** 2
                                              / np.where(density > 0.0, density, 1.0), 0.0)
        for axis, (lo, hi) in _AXIS_FACES.items():
            if (self.faces[lo].kind == FACE_PERIODIC) != (self.faces[hi].kind == FACE_PERIODIC):
                raise ValueError(f"laminar flow '{self.name}': periodic {axis} needs both {lo} and {hi} periodic, got "
                                 f'{FACE_KIND_NAMES[self.faces[lo].kind]} / {FACE_KIND_NAMES[self.faces[hi].kind]}')
        if self.xp is np:
            np.copyto(self._face_values, values_host)
        else:
            # async on the current stream (ordered after the previous step's kernel): a pageable source is staged at
            # once, so no device sync (the synchronous set() waits for the stream) and the host buffer is free again
            self._face_values.set(values_host, stream=self.xp.cuda.get_current_stream())
        if tuple(kinds) != self._uploaded_face_kinds:
            self._face_kind[...] = self.xp.asarray(kinds, dtype=np.int32)
            self._uploaded_face_kinds = tuple(kinds)

    def _rescale_populations(self, old_dt, new_dt):
        """Re-equilibrate the populations at the same SI velocity and pressure for a new lattice time step."""
        ratio = new_dt / old_dt
        deviation = self._deviation * DTYPE(ratio * ratio)
        lattice_x, lattice_y = self._lattice_x * DTYPE(ratio), self._lattice_y * DTYPE(ratio)
        populations = equilibrium_populations(deviation, lattice_x, lattice_y)
        populations *= self._fluid[None, :, :]
        self._store_populations(populations)
        xp = self.xp
        xp.copyto(self._lattice_x, lattice_x)
        xp.copyto(self._lattice_y, lattice_y)
        xp.copyto(self._deviation, deviation)
        self._force_dirty = self._faces_dirty = True

    def _store_populations(self, populations):
        """Float32 populations g into the current buffer (half_storage: stored = g x POPULATION_SCALE as fp16 bits)."""
        scale = POPULATION_SCALE if self.precision == 'half_storage' else 1.0   # lattice.step_kernel's unit
        store_populations(populations, DTYPE(scale), half_bits(self._populations[0]))

    def _start_from_initial_flow(self):
        """Equilibrium populations of the set_initial_flow velocity / pressure on the fluid cells, scaled for dt_lbm
        (a later step length rescales them like any dt change); publishes the t = 0 fields and frees the inputs."""
        xp = self.xp
        dt = self.dt_lbm
        velocity_scale = DTYPE(dt / self.dx)   # lattice velocity per m/s
        velocity_x, velocity_y, pressure = self._initial_flow
        fluid = self._fluid
        density = self._mass_density
        lattice_x = xp.where(fluid, velocity_x * velocity_scale, DTYPE(0.0)).astype(DTYPE)
        lattice_y = xp.where(fluid, velocity_y * velocity_scale, DTYPE(0.0)).astype(DTYPE)
        # p = (rho_lat - 1) / 3 rho U^2 with U = dx / dt
        deviation = xp.where(fluid, DTYPE(3.0) * pressure * velocity_scale * velocity_scale
                             / xp.where(fluid, density, DTYPE(1.0)), DTYPE(0.0)).astype(DTYPE)
        populations = equilibrium_populations(deviation, lattice_x, lattice_y)
        populations *= fluid[None, :, :]
        self._store_populations(populations)
        xp.copyto(self._lattice_x, lattice_x)
        xp.copyto(self._lattice_y, lattice_y)
        xp.copyto(self._deviation, deviation)
        self._lattice_dt = dt
        self._initial_flow = None
        self._publish()

    def _step_arguments(self, source, target, store_mode):
        """lbm_trt_step arguments from populations source to target at a lattice.STORE_* mode (the SI scales only for
        STORE_PUBLISHED, else 0: graph keys then do not follow dt)."""
        bouzidi = self._wall_row is not None
        uniform = not self._relaxation_stride
        if store_mode == STORE_PUBLISHED:
            velocity_scale = self.dx / self._lattice_dt
            scales = (DTYPE(velocity_scale), DTYPE(velocity_scale ** 2 / 3.0))
        else:
            scales = (DTYPE(0.0), DTYPE(0.0))
        fields = self.fields
        return (source, target, self._lattice_x, self._lattice_y, self._deviation, self._fluid,
                self._uniform_omega_plus if uniform else self._omega_plus,
                self._uniform_omega_minus if uniform else self._omega_minus,
                self._force_lattice_x, self._force_lattice_y, self._face_kind, self._face_values, *self._grid_args,
                self._wall_row if bouzidi else self._no_wall_row,
                self._wall_fraction if bouzidi else self._no_wall_fraction,
                self._relaxation_stride, np.int32(1 if self._force_present or self.xp is np else 0),
                np.int32(store_mode), fields.u_x, fields.u_y, fields.u_abs, fields.pressure, self._mass_density,
                *self._publish_grid, *scales)

    def _run_steps(self, n_steps, final_mode):
        """n_steps lattice steps: all but the last at STORE_BOUNDARY (nothing reads the interior macros in between;
        on the gpu in CUDA graphs of GRAPH_STEPS launches), the last at final_mode (STORE_LATTICE / STORE_PUBLISHED)."""
        kernel = step_kernel(self.precision, self._wall_row is not None)
        launch = self._launch
        silent_steps = n_steps - 1
        if silent_steps > 0:
            first, second = self._populations
            forward = self._step_arguments(first, second, STORE_BOUNDARY)
            backward = self._step_arguments(second, first, STORE_BOUNDARY)
            if self.use_step_graphs and silent_steps >= GRAPH_STEPS:
                def block():   # fixed arguments, no sync, no allocation, no timed profiler part
                    for index in range(GRAPH_STEPS):
                        kernel(*launch, backward if index % 2 else forward)
                # ids of the arrays (kept alive with the entry) and the scalar values: a replaced wall table, a new
                # relaxation / force stride or reallocated populations make a new graph
                key = (id(kernel), *(value.item() if isinstance(value, np.generic) else id(value)
                                     for value in forward))
                while silent_steps >= GRAPH_STEPS:
                    self._fixed_block(key, (kernel, forward, backward), block)
                    silent_steps -= GRAPH_STEPS
                    self.total_steps += GRAPH_STEPS
            for index in range(silent_steps):
                kernel(*launch, backward if index % 2 else forward)
                self._populations.reverse()
                self.total_steps += 1
        source, target = self._populations
        kernel(*launch, self._step_arguments(source, target, final_mode))
        self._populations.reverse()
        self.total_steps += 1

    def _fixed_block(self, key, kept, block):
        """Run block() eagerly at the first use of a key (loads the kernel modules outside a capture) and when capture
        failed, else replay its CUDA graph; populations, macros, rates, forces and face tables are rewritten in place,
        never reassigned while a graph of them lives (as em3d engine3d._fixed_block)."""
        entry = self._step_graphs.get(key)
        if entry is None:
            if len(self._step_graphs) >= _MAX_STEP_GRAPHS:
                self._step_graphs.clear()
            self._step_graphs[key] = [None, kept]
            block()
            return
        if entry[0] is None:
            entry[0] = self._capture(block)
            if entry[0] is None:
                block()
                return
        entry[0].launch(self.xp.cuda.get_current_stream())

    def _capture(self, block):
        """CUDA graph of block() (None, with a warning and single launches from now on, when capture fails)."""
        cp = self.xp
        stream = cp.cuda.Stream(non_blocking=True)
        try:
            with stream:
                stream.begin_capture()
                try:
                    block()
                finally:
                    graph = stream.end_capture()
        except Exception as error:   # capture unsupported here: the same kernels launched one by one
            self.use_step_graphs = False
            self._step_graphs.clear()
            warnings.warn(f"laminar flow '{self.name}': CUDA graph capture of the lattice steps failed "
                          f'({type(error).__name__}: {error}); launching them one by one from now on', RuntimeWarning,
                          stacklevel=4)
            return None
        self.n_step_graphs += 1
        return graph

    def _velocity_change(self):
        """(max |u - u_saved| over both components, peak |u|) in lattice units, one host sync; saves u, checks NaN and
        the lattice Mach number."""
        xp = self.xp
        change = xp.maximum(xp.abs(self._lattice_x - self._saved_x).max(), xp.abs(self._lattice_y - self._saved_y).max())
        peak = xp.sqrt(self._lattice_x * self._lattice_x + self._lattice_y * self._lattice_y).max()
        change, peak = (float(value) for value in to_host(xp.stack([change, peak])))
        xp.copyto(self._saved_x, self._lattice_x)
        xp.copyto(self._saved_y, self._lattice_y)
        self._last_check_step = self.total_steps
        if not (math.isfinite(change) and math.isfinite(peak)):
            raise RuntimeError(f"laminar flow '{self.name}' diverged (NaN / inf velocity after {self.total_steps} "
                               'lattice steps): lower the inlet velocities, pressures or forces, or relaxation_time')
        mach = math.sqrt(3.0) * peak
        self.peak_mach = max(self.peak_mach, mach)
        if mach > MACH_LIMIT:
            raise RuntimeError(f"laminar flow '{self.name}': lattice Mach number {mach:.3g} > {MACH_LIMIT:g} (the "
                               'lattice-Boltzmann equilibrium fails); lower relaxation_time, refine the mesh or lower '
                               'the driving velocity / pressure / force')
        if mach > MACH_WARNING and not self._mach_warned:
            self._mach_warned = True
            self.runtime_warnings.append(f'lattice Mach number {mach:.3g} > {MACH_WARNING:g}: compressibility errors '
                                         'grow as Ma^2; lower relaxation_time or refine the mesh')
        return change, peak

    def _march_to_steady(self):
        """Lattice steps until the extrapolated remaining velocity change (geometric decay between checks) is <=
        tolerance * peak, or the change stalled at the float32 noise floor, on two checks in a row; else max steps. The
        check interval doubles (up to the largest) while a check decays less than MIN_CHECK_DECAY."""
        self.xp.copyto(self._saved_x, self._lattice_x)
        self.xp.copyto(self._saved_y, self._lattice_y)
        steps, previous_change, previous_batch, converged_checks = 0, None, None, 0
        interval = self._check_interval
        remaining, peak = math.inf, 0.0
        while steps < self._max_steps:
            batch = min(interval, self._max_steps - steps)
            self._run_steps(batch, STORE_LATTICE)   # the check below reads the macros of the last step only
            steps += batch
            change, peak = self._velocity_change()
            decay = (None if previous_change is None or previous_change == 0.0
                     else decay_per_check(change / previous_change, batch / previous_batch))
            # a real decay shrinks the change every check; a small change that did not shrink over an interval of the
            # same length is rounding noise
            at_floor = change == 0.0 or (batch == previous_batch
                                         and previous_change <= change <= FLOAT32_STALL_CHANGE * peak)
            if at_floor:
                remaining = change
            elif decay is not None:
                remaining = change * decay / (1.0 - decay)
            else:
                remaining = math.inf
            previous_change, previous_batch = change, batch
            converged_checks = converged_checks + 1 if at_floor or remaining <= self.tolerance * peak else 0
            if converged_checks >= 2 or peak == 0.0:
                break
            if (not converged_checks and decay is not None and decay > math.exp(-MIN_CHECK_DECAY)
                    and interval < self._largest_check_interval):
                interval = min(2 * interval, self._largest_check_interval)
        self.last_march_steps = steps
        self.last_remaining = remaining / peak if peak > 0.0 else 0.0
        if converged_checks < 2 and peak > 0.0:
            self.n_unconverged += 1
            self.last_warning = (f"laminar flow '{self.name}': the steady march did not converge in {steps} lattice "
                                 f'steps (estimated remaining relative change {self.last_remaining:.3g} > tolerance '
                                 f'{self.tolerance:g}); raise max_steps or relaxation_time')

    def _publish(self):
        """SI velocity, its magnitude and pressure on the window from the lattice state, edge-extended into the PML ring."""
        velocity_scale = self.dx / self._lattice_dt
        to_physical(self._lattice_x, self._lattice_y, self._deviation, self._fluid, self._mass_density,
                    DTYPE(velocity_scale), DTYPE(velocity_scale ** 2 / 3.0), *self._field_windows)
        self._extend_into_ring()

    def _extend_into_ring(self):
        """Window edge rows of the SI fields copied into the PML ring (after _publish or a STORE_PUBLISHED step)."""
        for destination, source in self._pml_copies:
            self.xp.copyto(destination, source)

    def _record_probes(self, t):
        for probe in self.probes:
            probe.record(self.fields, t)

    def vorticity_window(self):
        """Vorticity dv/dx - du/dy [1/s] on the window (central differences, one-sided at the edges; a solid neighbour
        enters as the halfway-wall image -u of the fluid cell, so the wall shear is not 25 % low; 0 in solids)."""
        xp = self.xp
        if self._vorticity is None or array_module_of(self._vorticity) is not xp:
            self._vorticity = xp.zeros(self.window_shape, dtype=DTYPE)
        velocity_x, velocity_y = self._field_windows[0], self._field_windows[1]
        self._vorticity.fill(0.0)
        if self.window_shape[0] > 1:
            self._vorticity += self._wall_difference(velocity_y, 0)
        if self.window_shape[1] > 1:
            self._vorticity -= self._wall_difference(velocity_x, 1)
        self._vorticity *= self._fluid
        return self._vorticity

    def _wall_difference(self, values, axis):
        """xp.gradient of the window `values` along axis, interior cells with a solid neighbour corrected from the
        stored 0 to the image -u of the halfway bounce-back wall: + u / (2 dx) per solid lo neighbour, - per hi one."""
        xp = self.xp
        gradient = xp.gradient(values, self.dx, axis=axis)
        solid = ~self._fluid
        interior = [slice(None)] * values.ndim
        lower, upper = list(interior), list(interior)
        interior[axis], lower[axis], upper[axis] = slice(1, -1), slice(None, -2), slice(2, None)
        interior, lower, upper = tuple(interior), tuple(lower), tuple(upper)
        image_weight = (solid[lower].astype(DTYPE) - solid[upper].astype(DTYPE)) * DTYPE(0.5 / self.dx)
        gradient[interior] += image_weight * values[interior]
        return gradient


def _parabolic_shape(fluid_row):
    """Discrete Poiseuille shape s (n - s) per run of fluid cells (s = cell centre from the run's wall), each run
    normalised to mean 1; 0 on solid cells (numpy float64)."""
    shape = np.zeros(fluid_row.size)
    start = None
    for index in range(fluid_row.size + 1):
        inside = index < fluid_row.size and bool(fluid_row[index])
        if inside and start is None:
            start = index
        elif not inside and start is not None:
            count = index - start
            centres = np.arange(count) + 0.5
            run = centres * (count - centres)
            shape[start:index] = run / run.mean()
            start = None
    return shape
