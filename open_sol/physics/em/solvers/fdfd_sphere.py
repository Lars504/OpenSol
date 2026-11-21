"""Spherically symmetric FDFD on a spherical Domain (x = r - radius_min; y = theta over the whole 0..pi, one cell
across on a 1D line): SpherePhysics solves the radial equation of every listed multipole term (spherical_multipoles)
on the host and keeps float32 sphere-RMS radial profiles (the same along theta; the state() ones on the run backend,
the rendered-only ones host numpy) for the renderer; its sources, probes and result functions."""
import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from ....core.backend import check_backend_array, host_copy, to_host
from ....core.constants import C0, DTYPE
from ....core.solver_base import PhysicsSolver
from ....core.wavelengths import wavelength_label
from ....solvers.arpack_shift_invert import ArpackShiftInvertSolver
from ....solvers.sparse_direct import SparseDirectSolver, scipy_available
from ..eigenfrequency import q_factor_of, resonance_table_result
from ..fdfd_monitors import _ascending_wavelength
from ..harmonic import cell_permeability, cell_permittivity, write_float32
from ..materials import (DrudePole, LorentzPole, material_solution_state, register_em_arrays,
                         restore_material_solution)
from ..renderer_sphere import SphereRenderer
from ..spherical_multipoles import (COMPONENT_UNITS, DIPOLE_ORIENTATIONS, FAMILY_TEXTS, INNER_FACES, OUTER_FACES,
                                    SPHERE_FAMILIES, SPHERE_FIELD_COMPONENTS, SphereDipole, SphereMaterials,
                                    absorbed_power_density, assemble_sphere, centre_fields, homogeneous_dipole_power,
                                    mie_cross_sections, outgoing_coefficient, plane_wave_amplitude, plane_wave_rhs,
                                    solve_term_resonances, sphere_grid)
from .fdtd_2d_base import add_pec_cells, record_dispersive_region

SPHERE_POLARIZATION = 'sphere'        # no in-plane polarisation (sheet placement helpers refuse it)
SPHERE_MATERIAL_SCHEMA_KEY = 'ez'     # EM_MATERIAL_SCHEMAS key: eps_r (isotropic), mu_r, sigma_e, sigma_m
SPHERE_PERMITTIVITY_ARRAYS = ('eps_r',)
SPHERE_PERMITTIVITY_SLOTS = {'eps_r_update': 'eps_r'}
SYSTEM_MATERIALS = frozenset({'eps_r', 'sigma_e', 'mu_r', 'sigma_m'})
THETA_SPAN_TOLERANCE = 1e-5    # relative: the float32 angle widths of a 0..pi window sum to pi within this
THETA_UNIFORM_TOLERANCE = 1e-6   # relative spread of a material along theta that still counts as spherically symmetric
# float32 window-shaped fields SphereRenderer reads (SpherePhysics.fields); only DEVICE_SPHERE_FIELDS (the state() /
# loss-protocol keys couplings read) are run-backend arrays, the rendered-only rest host numpy (CentreFieldRenderer
# uploads the read window)
SPHERE_STATE_NAMES = SPHERE_FIELD_COMPONENTS + ('e_intensity', 'h_abs', 'q_abs')
DEVICE_SPHERE_FIELDS = frozenset({'e_intensity', 'q_abs'})


class SpherePlaneWave:
    """Incident x-polarised plane wave along +z of amplitude E0 [V/m]; per solve the Mie coefficient -c / alpha of every
    solved term and its cross sections."""

    def __init__(self, amplitude):
        amplitude = float(amplitude)
        if not (math.isfinite(amplitude) and amplitude > 0.0):
            raise ValueError(f'plane wave amplitude must be finite and > 0 [V/m], got {amplitude!r}')
        self.amplitude = amplitude
        self._pending = {}
        self._records = {}   # omega -> {(family, order): (coefficient, C_sca, C_ext)}

    def begin(self):
        self._pending = {}

    def record_term(self, system, coefficient):
        scattering, extinction = mie_cross_sections(system, coefficient)
        self._pending[(system.family, system.order)] = (complex(coefficient), scattering, extinction)

    def finish(self, omega):
        self._records[float(omega)] = self._pending
        self._pending = {}

    def recorded(self):
        """(omega float64 in ascending wavelength order, [{(family, order): (coefficient, C_sca, C_ext)}])."""
        return _ascending_wavelength(self._records, 'the plane wave')


class SphereDipoleSource:
    """A SphereDipole and its emitted power per solve and term."""

    def __init__(self, dipole):
        self.dipole = dipole
        self._pending = {}
        self._homogeneous = math.nan
        self._records = {}   # omega -> ({(family, order): W}, homogeneous W)

    def begin(self):
        self._pending = {}
        self._homogeneous = math.nan

    def record_term(self, system, power):
        self._pending[(system.family, system.order)] = float(power)
        if math.isnan(self._homogeneous):
            cell = int(np.clip(np.searchsorted(system.grid.edges, self.dipole.radius, side='right') - 1, 0,
                               system.grid.n_cells - 1))
            self._homogeneous = homogeneous_dipole_power(self.dipole.moment, system.omega,
                                                         system.materials.eps[cell], system.materials.mu[cell])

    def finish(self, omega):
        self._records[float(omega)] = (self._pending, self._homogeneous)
        self._pending = {}

    def recorded(self):
        return _ascending_wavelength(self._records, f'the dipole source at r = {self.dipole.radius:.6g} m')


class SphereFieldProbe:
    """Sphere-RMS amplitude of one component in the radial cell ix per driven solve, and its phase when the solve had
    exactly one term and source class (NaN otherwise)."""

    def __init__(self, ix, field_component):
        if field_component not in SPHERE_FIELD_COMPONENTS:
            raise ValueError(f'field probe component must be one of {SPHERE_FIELD_COMPONENTS}, got '
                             f'{field_component!r}')
        self.ix = int(ix)
        self.field_component = field_component
        self.unit = COMPONENT_UNITS[field_component]
        self._values = {}   # omega -> (rms, phase)

    def record(self, physics):
        component = self.field_component
        rms = float(physics.centre_rms[component][self.ix])
        single = physics.single_phasors
        phase = float(np.angle(single[component][self.ix])) if single is not None and rms > 0.0 else math.nan
        self._values[float(physics.solved_omega)] = (rms, phase)

    def recorded(self):
        omegas, values = _ascending_wavelength(self._values, f'the field probe {self.field_component} at radial cell '
                                                             f'{self.ix}')
        return omegas, np.array(values, dtype=np.float64).reshape(-1, 2)


@dataclass(frozen=True, eq=False)
class SphereResonance:
    """One resonance of a multipole term: complex omega [rad/s] (Im > 0 decays), vacuum wavelength 2 pi C0 / Re omega,
    Q, pml_fraction 0 (no PML), the node values of y, the term (family, order) and its SphereSystem."""
    omega: complex
    wavelength: float
    q_factor: float
    pml_fraction: float
    field: np.ndarray
    family: str
    order: int
    system: object
    residual: float = 0.0
    region_fraction: float = math.nan

    def text(self):
        """'l = 3 er, lambda = 812.4 nm, Q = 154.3'."""
        return f'l = {self.order} {self.family}, lambda = {self.wavelength * 1e9:.6g} nm, Q = {self.q_factor:.4g}'


@dataclass(frozen=True, eq=False)
class SphereEigenSolve:
    """Resonances of one solve_eigenfrequencies, term by term (orders in list order, er before hr), nearest the target
    first within each, and the solve info."""
    resonances: list
    info: dict

    def summary(self):
        info = self.info
        return (f"{len(self.resonances)} of {info['n_requested']} resonances near lambda = "
                f"{info['target_wavelength'] * 1e9:.6g} nm over {info['n_terms']} multipole terms "
                f"({info['n_unknowns']:,} radial unknowns each; shift-invert {info['eigen_seconds']:.3g} s, "
                f"{info['operator_applications']} operator applications; exact radiation condition: "
                f"{info['refine_iterations']} refinement iterations in {info['refine_seconds']:.3g} s, "
                f"{info['candidates_dropped']} candidates dropped)")


class SpherePhysics(PhysicsSolver):
    """Frequency-domain EM of a spherically symmetric structure (eps(r), mu(r)): per multipole term (family 'er' / 'hr',
    order l) a radial solve on the host, driven (plane wave, dipoles) or for resonances; shown fields = the sphere-RMS
    sum over every solved term and source class."""
    # Source classes of a driven solve, solved apart (their m sets are orthogonal or, plane wave vs dipoles, refused
    # together): the plane wave (m = +-1), radial dipoles (m = 0, 'er' only), tangential dipoles (m = +-1). One factor
    # per term serves every class. Materials are re-read from the MaterialGrid at every solve (1D: cheap).

    name = 'em'
    type_label = 'em_fdfd_sphere'
    coordinate_grids = ('polar',)
    permittivity_arrays = SPHERE_PERMITTIVITY_ARRAYS
    permittivity_slots = SPHERE_PERMITTIVITY_SLOTS
    state_names = SPHERE_STATE_NAMES
    field_components = SPHERE_FIELD_COMPONENTS
    component_units = COMPONENT_UNITS

    def __init__(self, domain, wavelength=1.55e-6, multipole_orders=(1,), families=SPHERE_FAMILIES,
                 inner_face='centre', outer_face='radiation', linear_solver=None, eigen_solver=None):
        if getattr(domain, 'dimensions', 2) != 2:
            raise ValueError("SpherePhysics needs a 2D spherical (r, theta) domain or a 1D spherical line (coordinates "
                             "'spherical')")
        coordinates = domain.coordinates
        if not coordinates.polar:
            raise ValueError("SpherePhysics needs coordinates 'spherical' (a polar grid solved as (r, theta))")
        window = domain.physical_window()
        if window != (slice(0, domain.grid.nx), slice(0, domain.grid.ny)):
            raise ValueError('SpherePhysics: the window must be the whole grid (no padding)')
        angle_span = float(np.sum(domain.cell_widths()[1]))
        if abs(coordinates.angle_min) > 1e-9 or abs(angle_span - math.pi) > THETA_SPAN_TOLERANCE * math.pi:
            raise ValueError(f'SpherePhysics needs the whole polar angle range theta = 0 .. pi (every radial cell a '
                             f'whole shell), got angle_min {coordinates.angle_min:.6g}, span {angle_span:.6g} rad')
        self.domain = domain
        self.xp = domain.xp
        self.polarization = SPHERE_POLARIZATION
        self.axisymmetric = True   # polar + axisymmetric = spherical (Session weights by whole shells)
        self.orders = _orders(multipole_orders)
        families = tuple(families)
        if not families or any(family not in SPHERE_FAMILIES for family in families):
            raise ValueError(f'families must be a non-empty subset of {SPHERE_FAMILIES}, got {families!r}')
        self.families = tuple(family for family in SPHERE_FAMILIES if family in families)
        if inner_face not in INNER_FACES or outer_face not in OUTER_FACES:
            raise ValueError(f'inner_face must be one of {INNER_FACES}, outer_face one of {OUTER_FACES}')
        self.inner_face, self.outer_face = inner_face, outer_face
        self.grid = sphere_grid(coordinates.radius_min, domain.cell_widths()[0])
        if (inner_face == 'centre') != (self.grid.edges[0] == 0.0):
            raise ValueError(f"inner face '{inner_face}': the sphere centre (radius_min 0) takes 'centre' and only it")
        wavelength = float(wavelength)
        if not (math.isfinite(wavelength) and wavelength > 0.0):
            raise ValueError(f'SpherePhysics wavelength must be finite and > 0 [m], got {wavelength!r}')
        self._omega = 2.0 * math.pi * C0 / wavelength
        self._private_solvers = []
        if linear_solver is None:
            linear_solver = SparseDirectSolver(back_end='scipy_splu', backend='cpu')
            self._private_solvers.append(linear_solver)
        if linear_solver.back_end_choice == 'cusolver_qr':
            raise ValueError("SpherePhysics factorises a host matrix: a linear solver with back_end 'cusolver_qr' "
                             'cannot serve it')
        if eigen_solver is None:
            eigen_solver = ArpackShiftInvertSolver()
            self._private_solvers.append(eigen_solver)
        self.linear_solver, self.eigen_solver = linear_solver, eigen_solver
        register_em_arrays(domain.materials, SPHERE_MATERIAL_SCHEMA_KEY)
        self.sources = []
        self.monitors = []
        self.dispersive_regions = []
        self.pec_cells = None
        self.finalize_warnings = []
        self.runtime_warnings = []
        self.enable_q_accum = False   # loss-protocol flag set by the runner; q_abs is computed by every solve
        self.solved_omega = None
        self.eigen_solve = None
        self.selected_resonance = None
        self.centre_rms = None        # host {component: float64 (n,)} of the shown solution
        self.single_phasors = None    # host {component: complex128 (n,)} when it was one term and source class
        self.n_solves = 0
        self.n_term_solves = 0
        self._solution_dirty = True
        shape = (domain.grid.nx, domain.grid.ny)
        self.fields = {name: (self.xp if name in DEVICE_SPHERE_FIELDS else np).zeros(shape, dtype=DTYPE)
                       for name in self.state_names}

    # --- settings ---

    @property
    def omega(self) -> float:
        return self._omega

    @omega.setter
    def omega(self, omega):
        omega = float(omega)
        if not (math.isfinite(omega) and omega > 0.0):
            raise ValueError(f'SpherePhysics omega must be finite and > 0 [rad/s], got {omega!r}')
        if omega != self._omega:
            self._omega = omega
            self._solution_dirty = True

    @property
    def terms(self) -> tuple:
        """(family, order) of every solved multipole term, order by order (er before hr)."""
        return tuple((family, order) for order in self.orders for family in self.families)

    # --- PhysicsSolver contract ---

    def prepare(self, domain):
        return

    def step(self, t: float = None, dt: float = None):
        """Solve at self.omega when the materials, sources or omega changed since the last solve."""
        if self._solution_dirty:
            self._solve()

    def dt_limit(self) -> float:
        return math.inf

    def state(self) -> dict:
        """Float32 run-backend e_intensity [V^2/m^2] and q_abs [W/m^3] (sphere means) of the shown solution, eps_r,
        sigma_e and 'q_accum': None (loss-protocol key)."""
        materials = self.domain.materials
        return {'e_intensity': self.fields['e_intensity'], 'q_abs': self.fields['q_abs'], 'eps_r': materials.eps_r,
                'sigma_e': materials.sigma_e, 'q_accum': None}

    def inject_source(self, name: str, array):
        if name not in self.permittivity_slots:
            raise ValueError(f"SpherePhysics accepts sources {', '.join(self.permittivity_slots)}; got '{name}'")
        xp = self.xp
        xp.copyto(getattr(self.domain.materials, self.permittivity_slots[name]), xp.asarray(array, dtype=DTYPE))
        self._solution_dirty = True

    def finalize(self) -> None:
        self.finalize_warnings = []
        if not scipy_available():
            raise ValueError('em_fdfd_sphere needs scipy (host sparse direct and eigen solvers), which is not '
                             'installed: pip install scipy')

    def materials_changed(self, names) -> None:
        if not SYSTEM_MATERIALS.isdisjoint(names):
            self._solution_dirty = True

    def take_absorbed_power(self, merge_previous: bool = False):
        """EM loss protocol: q_abs [W/m^3] (sphere mean) of the last solve, None before the first."""
        return None if self.solved_omega is None else self.fields['q_abs']

    def renderer(self):
        return SphereRenderer(self)

    def solve_harmonic(self, omega):
        """frequency_domain study hook: solve now at omega [rad/s] and record the monitors."""
        self.omega = omega
        self._solve()

    def selection_label(self):
        if self.selected_resonance is not None:
            resonance = self.eigen_solve.resonances[self.selected_resonance - 1]
            return f'resonance {self.selected_resonance}: {resonance.text()}'
        if self.solved_omega is None:
            return None
        return wavelength_label(2.0 * math.pi * C0 / self.solved_omega)

    def work_counters(self):
        counters = {'solves': self.n_solves, 'term_solves': self.n_term_solves}
        for solver in self._private_solvers:
            counters.update(solver.work_counters())
        return counters

    # --- materials, sources and monitors ---

    def host_materials(self, omega) -> SphereMaterials:
        """SphereMaterials at omega [rad/s]: eps (sigma_e and poles included), mu (sigma_m) per radial cell, PEC; on an
        (r, theta) window the theta mean of each radial row, ValueError when a row varies along theta (the structure
        must be spherically symmetric: shells drawn as circles about the origin). Computed on the host from float32
        copies of the four material arrays and the pole masks (no complex128 device temporaries)."""
        materials = self.domain.materials
        cells = SimpleNamespace(**{name: to_host(getattr(materials, name)) for name in SYSTEM_MATERIALS})
        regions = [(pole, box, None if mask is None else to_host(mask))
                   for pole, box, mask in self.dispersive_regions]
        eps = _radial_profile(cell_permittivity(cells, 'eps_r', regions, omega, xp=np),
                              'the permittivity (eps_r, sigma_e, poles)')
        mu = _radial_profile(cell_permeability(cells, omega, xp=np), 'the permeability (mu_r, sigma_m)')
        if self.pec_cells is None:
            pec = np.zeros(self.grid.n_cells, dtype=bool)
        else:
            cells = np.asarray(to_host(self.pec_cells), dtype=bool)
            if np.any(cells != cells[:, :1]):
                raise ValueError('em_fdfd_sphere: a pec_domain covers part of a shell (it varies along theta); select '
                                 'whole shells (circles about the origin)')
            pec = cells[:, 0].copy()
        return SphereMaterials(eps=eps, mu=mu, pec=pec)

    def radius_of(self, x) -> float:
        """Radius [m] of a logical window position x = r - radius_min."""
        return float(self.domain.coordinates.radius_min + float(x))

    def add_plane_wave(self, amplitude=1.0):
        """Incident plane wave of amplitude E0 [V/m] (x-polarised, along +z); needs the outer face 'radiation'."""
        if self.outer_face != 'radiation':
            raise ValueError("a plane wave needs bc_x_hi 'radiation' (an open exterior)")
        return self._add_source(SpherePlaneWave(amplitude))

    def add_dipole_source(self, x, orientation='radial', moment=1e-12, phase=0.0):
        """Current element of moment I d [A m] and phase [deg] at the logical position x (radius radius_min + x),
        'radial' or 'tangential'."""
        if orientation not in DIPOLE_ORIENTATIONS:
            raise ValueError(f'dipole orientation must be one of {DIPOLE_ORIENTATIONS}, got {orientation!r}')
        radius = self.radius_of(x)
        if not self.grid.edges[0] < radius <= self.grid.edges[-1]:
            raise ValueError(f'dipole radius {radius:.6g} m lies outside the radial window ({self.grid.edges[0]:.6g}, '
                             f'{self.grid.edges[-1]:.6g}] m')
        moment = float(moment) * complex(np.exp(1j * math.radians(float(phase))))
        return self._add_source(SphereDipoleSource(SphereDipole(radius=radius, orientation=orientation,
                                                                moment=moment)))

    def add_field_probe(self, x, y, field_component='et'):
        """Sphere-RMS amplitude of one component (SPHERE_FIELD_COMPONENTS) in the radial cell containing (x, y)."""
        self.domain.validate_physical_pos(x, y, 'Field probe')
        probe = SphereFieldProbe(self.domain.pos_to_cell_x(x), field_component)
        self.monitors.append(probe)
        return probe

    def _add_source(self, source):
        self.sources.append(source)
        self._solution_dirty = True
        return source

    def add_dispersive(self, pole, mask):
        """Drude / Lorentz pole on a run-backend bool (nx, 1) cell mask (None = everywhere)."""
        if not isinstance(pole, (DrudePole, LorentzPole)):
            raise TypeError(f'add_dispersive needs a DrudePole or LorentzPole, got {type(pole).__name__}')
        if mask is not None:
            check_backend_array(mask, self.fields['e_intensity'].shape, self.xp.bool_, 'Pole mask', xp=self.xp)
        record_dispersive_region(self.dispersive_regions, pole, mask)
        self._solution_dirty = True

    def add_pec_region(self, mask):
        """Perfect electric conductor (spherical shells) on a run-backend bool (nx, 1) cell mask."""
        check_backend_array(mask, self.fields['e_intensity'].shape, self.xp.bool_, 'PEC mask', xp=self.xp)
        self.pec_cells = add_pec_cells(self.pec_cells, mask)
        self._solution_dirty = True

    # --- driven solve ---

    def _source_classes(self):
        """[(class name, sources)] of the driven solve: the plane wave, radial dipoles, tangential dipoles; ValueError
        for a plane wave beside dipoles or several plane waves."""
        plane_waves = [source for source in self.sources if isinstance(source, SpherePlaneWave)]
        dipoles = [source for source in self.sources if isinstance(source, SphereDipoleSource)]
        if plane_waves and dipoles:
            raise ValueError('em_fdfd_sphere: a plane wave and dipole sources cannot share one physics (their '
                             'interference is not modelled); use two physics or two runs')
        if len(plane_waves) > 1:
            raise ValueError(f'em_fdfd_sphere: at most one plane wave per physics, got {len(plane_waves)}')
        classes = [('plane_wave', plane_waves)] if plane_waves else []
        for orientation in DIPOLE_ORIENTATIONS:
            members = [source for source in dipoles if source.dipole.orientation == orientation]
            if members:
                classes.append((orientation, members))
        return classes

    def _solve(self):
        """Every term at omega: one factor of A, one solve per source class; sphere-RMS sums of the centre fields and
        q_abs into the display fields; the sources' and monitors' records."""
        omega = self._omega
        materials = self.host_materials(omega)
        classes = self._source_classes()
        n_cells = self.grid.n_cells
        squares = {name: np.zeros(n_cells) for name in SPHERE_FIELD_COMPONENTS}
        absorbed = np.zeros(n_cells)
        single, responses = None, 0
        for source in self.sources:
            source.begin()
        for family, order in self.terms:
            system = assemble_sphere(self.grid, materials, family, order, omega, self.inner_face, self.outer_face)
            work = []
            for class_name, members in classes:
                if class_name == 'plane_wave':
                    alpha = plane_wave_amplitude(system, members[0].amplitude)
                    rhs = plane_wave_rhs(system, alpha)
                else:
                    alpha = 0.0
                    rhs = sum(member.dipole.node_rhs(system) for member in members)
                rhs = system.restricted(rhs)
                if np.any(rhs):
                    work.append((class_name, members, rhs, alpha))
            if not work:
                continue
            factor = self.linear_solver.factorize_host(system.operator)
            try:
                for class_name, members, rhs, alpha in work:
                    values = system.full(factor.solve_host(rhs))
                    fields = centre_fields(system, values)
                    for name in SPHERE_FIELD_COMPONENTS:
                        squares[name] += np.abs(fields[name]) ** 2
                    absorbed += absorbed_power_density(system, fields)
                    single, responses = fields, responses + 1
                    if class_name == 'plane_wave':
                        members[0].record_term(system, -outgoing_coefficient(system, values, alpha) / alpha)
                    else:
                        for member in members:
                            member.record_term(system, member.dipole.emitted_power(system, values))
            finally:
                release = getattr(factor, 'release', None)
                if release is not None:
                    release()
            self.n_term_solves += 1
        for source in self.sources:
            source.finish(omega)
        self.solved_omega = omega
        self.selected_resonance = None
        self._show(squares, absorbed, single if responses == 1 else None)
        for monitor in self.monitors:
            monitor.record(self)
        self.n_solves += 1
        self._solution_dirty = False

    def _show(self, squares, absorbed, single):
        """Write the float32 display fields from the summed |radial functions|^2 and q_abs; keep the host RMS profiles
        and the single-term phasors (probes)."""
        rms = {name: np.sqrt(values) for name, values in squares.items()}
        host = dict(rms, e_intensity=squares['er'] + squares['et'], h_abs=np.sqrt(squares['hr'] + squares['ht']),
                    q_abs=absorbed)
        for name, values in self.fields.items():   # the radial profile along every theta of the window
            write_float32(values, np.broadcast_to(np.asarray(host[name])[:, None], values.shape))
        self.centre_rms = rms
        self.single_phasors = single

    # --- resonances (eigenfrequency study) ---

    def solve_eigenfrequencies(self, n_resonances):
        """Resonances nearest omega (the wavelength setting) per term (solve_term_resonances: shift-invert, then the
        exact radiation condition at each complex omega; materials at the target); resonance 1 shown."""
        n_resonances = int(n_resonances)
        omega = self._omega
        materials = self.host_materials(omega)
        resonances, warnings = [], []
        totals = {'n_unknowns': 0, 'eigen_seconds': 0.0, 'refine_seconds': 0.0, 'operator_applications': 0,
                  'refine_iterations': 0, 'candidates_dropped': 0}
        for family, order in self.terms:
            system = assemble_sphere(self.grid, materials, family, order, omega, self.inner_face, self.outer_face)
            candidates, info = solve_term_resonances(system, n_resonances, self.eigen_solver)
            for key in totals:
                totals[key] = max(totals[key], info[key]) if key == 'n_unknowns' else totals[key] + info[key]
            if not info['arpack_converged']:
                warnings.append(f'{family} l = {order}: ARPACK did not converge every eigenvalue')
            if len(candidates) < n_resonances:
                warnings.append(f'{family} ({FAMILY_TEXTS[family]}) l = {order}: {len(candidates)} of {n_resonances} '
                                'resonances found near the target')
            for candidate in candidates:
                resonances.append(SphereResonance(
                    omega=complex(candidate.omega), wavelength=2.0 * math.pi * C0 / candidate.omega.real,
                    q_factor=q_factor_of(candidate.omega), pml_fraction=0.0, field=candidate.node_values,
                    family=family, order=order, system=system, residual=candidate.residual))
        self.eigen_solve = SphereEigenSolve(resonances, dict(
            totals, warnings=warnings, n_requested=n_resonances * len(self.terms), n_terms=len(self.terms),
            target_wavelength=2.0 * math.pi * C0 / omega, multipole_orders=list(self.orders),
            families=list(self.families)))
        self.selected_resonance = None
        if resonances:
            self.select_resonance(1)
        return self.eigen_solve

    def select_resonance(self, number):
        """Show resonance `number` (1-based): its centre fields scaled to a peak sphere-RMS |E| of 1 V/m."""
        if self.eigen_solve is None:
            raise ValueError('no eigen solve yet: run solve_eigenfrequencies first')
        count = len(self.eigen_solve.resonances)
        if not 1 <= int(number) <= count:
            raise ValueError(f'resonance number must be in 1..{count}, got {number}')
        resonance = self.eigen_solve.resonances[int(number) - 1]
        fields = centre_fields(resonance.system, resonance.field, resonance.omega)
        magnitude = np.sqrt(np.abs(fields['er']) ** 2 + np.abs(fields['et']) ** 2)
        peak = int(np.argmax(magnitude))
        scale = 1.0 / (fields['et'][peak] if abs(fields['et'][peak]) >= abs(fields['er'][peak])
                       else fields['er'][peak])
        fields = {name: values * scale for name, values in fields.items()}
        squares = {name: np.abs(values) ** 2 for name, values in fields.items()}
        self._show(squares, absorbed_power_density(resonance.system, fields, resonance.omega.real), fields)
        self.selected_resonance = int(number)

    # --- snapshot solutions (solver_hooks.SolutionStateSolver) ---

    def solution_state(self, materials=False):
        """Host copies of the shown fields as (nx, 1) radial profiles (_show writes them the same along theta), the
        solved omega and the selected resonance (materials: also the permittivity arrays)."""
        state = {'fields': {name: host_copy(values[:, :1]) for name, values in self.fields.items()},
                 'solved_omega': self.solved_omega, 'selected_resonance': self.selected_resonance}
        if materials:
            state['materials'] = material_solution_state(self, (*self.permittivity_arrays, 'sigma_e'))
        return state

    def restore_solution(self, state):
        for name, values in self.fields.items():
            write_float32(values, np.broadcast_to(state['fields'][name], values.shape))
        self.solved_omega = state['solved_omega']
        self.selected_resonance = state['selected_resonance']
        if 'materials' in state:
            restore_material_solution(self, state['materials'])


def _radial_profile(values, what) -> np.ndarray:
    """complex128 (nr,) theta mean of an (nr, ntheta) host array; ValueError when a row spreads by more than
    THETA_UNIFORM_TOLERANCE of the largest |value|."""
    values = np.asarray(values, dtype=np.complex128)
    deviation = np.abs(values - values[:, :1])
    limit = THETA_UNIFORM_TOLERANCE * max(float(np.max(np.abs(values))), 1.0)
    if float(np.max(deviation)) > limit:
        rows = np.flatnonzero(np.any(deviation > limit, axis=1))
        raise ValueError(f'em_fdfd_sphere: {what} varies along theta in radial cells {rows[:5].tolist()} (first of '
                         f'{rows.size}); the structure must be spherically symmetric: draw shells as circles about the '
                         'origin and keep material expressions independent of the angle')
    return values.mean(axis=1)


def _orders(orders) -> tuple:
    """Distinct integer multipole orders >= 1 in list order; ValueError otherwise."""
    values = tuple(orders)
    if not values or any(float(order) != int(order) or int(order) < 1 for order in values):
        raise ValueError(f'multipole orders must be integers >= 1, got {values!r}')
    values = tuple(int(order) for order in values)
    if len(set(values)) != len(values):
        raise ValueError(f'multipole orders must be distinct, got {values!r}')
    return values


# ---- results ----
def sphere_probe_result(probe) -> dict:
    """'xy' of a SphereFieldProbe vs ascending wavelength: series abs (sphere-RMS, unit of the component) and phase
    [rad] (NaN unless one term and source class was solved); extras frequency, wavelength, component, unit."""
    omegas, values = probe.recorded()
    frequency = omegas / (2.0 * math.pi)
    wavelength = C0 / frequency
    return {'kind': 'xy', 'x': wavelength, 'x_label': 'Wavelength', 'x_unit': 'm',
            'y_label': SphereRenderer.field_infos[probe.field_component].label,
            'series': {'abs': values[:, 0], 'phase': values[:, 1]}, 'series_units': {'abs': probe.unit, 'phase': 'rad'},
            'frequency': frequency, 'wavelength': wavelength.copy(), 'component': probe.field_component,
            'unit': probe.unit}


def _term_table(records, families, value_of):
    """(orders, {family: float64 (n_wavelengths, n_orders)}) of per-term records (NaN where a term is missing)."""
    orders = sorted({order for record in records for (_, order) in record})
    tables = {family: np.full((len(records), len(orders)), np.nan) for family in families}
    for row, record in enumerate(records):
        for (family, order), entry in record.items():
            if family in tables:
                tables[family][row, orders.index(order)] = value_of(entry)
    return orders, tables


def sphere_plane_wave_result(source) -> dict:
    """'xy' of a SpherePlaneWave vs ascending wavelength: series sigma_ext, sigma_sca, sigma_abs [m^2] summed over the
    solved terms; extras frequency, wavelength, multipole_orders, a_real / a_imag ('er': electric Mie coefficients) and
    b_real / b_imag ('hr': magnetic) float64 (n_wavelengths, n_orders), NaN where a family was not solved."""
    omegas, records = source.recorded()
    frequency = omegas / (2.0 * math.pi)
    wavelength = C0 / frequency
    scattering = np.array([sum(entry[1] for entry in record.values()) for record in records], dtype=np.float64)
    extinction = np.array([sum(entry[2] for entry in record.values()) for record in records], dtype=np.float64)
    orders = sorted({order for record in records for (_, order) in record})
    result = {'kind': 'xy', 'x': wavelength, 'x_label': 'Wavelength', 'x_unit': 'm', 'y_label': 'Cross section',
              'series': {'sigma_ext': extinction, 'sigma_sca': scattering, 'sigma_abs': extinction - scattering},
              'series_units': {'sigma_ext': 'm^2', 'sigma_sca': 'm^2', 'sigma_abs': 'm^2'},
              'frequency': frequency, 'wavelength': wavelength.copy(), 'multipole_orders': orders}
    for family, letter in (('er', 'a'), ('hr', 'b')):
        for part, take in (('real', np.real), ('imag', np.imag)):
            _, table = _term_table(records, (family,), lambda entry, take=take: float(take(entry[0])))
            result[f'{letter}_{part}'] = table[family]
    return result


def sphere_dipole_result(source) -> dict:
    """'xy' of a SphereDipoleSource vs ascending wavelength: series power [W] (emitted, summed over the solved terms),
    power_homogeneous [W] (eta k^2 |I d|^2 / (12 pi) of the source cell's medium unbounded) and purcell (their ratio);
    extras frequency, wavelength, radius, orientation, multipole_orders, power_orders float64 (n_wavelengths,
    n_orders) (both families summed)."""
    omegas, records = source.recorded()
    frequency = omegas / (2.0 * math.pi)
    wavelength = C0 / frequency
    power = np.array([sum(per_term.values()) for per_term, _ in records], dtype=np.float64)
    homogeneous = np.array([value for _, value in records], dtype=np.float64)
    orders, tables = _term_table([per_term for per_term, _ in records], SPHERE_FAMILIES, float)
    power_orders = np.nansum(np.stack([tables[family] for family in SPHERE_FAMILIES]), axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        purcell = power / homogeneous
    return {'kind': 'xy', 'x': wavelength, 'x_label': 'Wavelength', 'x_unit': 'm', 'y_label': 'Emitted power',
            'series': {'power': power, 'power_homogeneous': homogeneous, 'purcell': purcell},
            'series_units': {'power': 'W', 'power_homogeneous': 'W', 'purcell': '1'},
            'frequency': frequency, 'wavelength': wavelength.copy(), 'radius': source.dipole.radius,
            'orientation': source.dipole.orientation, 'multipole_orders': orders, 'power_orders': power_orders}


def sphere_resonance_table_result(table, physics) -> dict:
    """eigenfrequency.resonance_table_result (pml_fraction 0: no PML) plus the last columns multipole_order and family;
    extras multipole_orders and families (the solved lists)."""
    result = resonance_table_result(table, physics)
    resonances = physics.eigen_solve.resonances
    result['columns'] = list(result['columns']) + ['multipole_order', 'family']
    result['units'] = list(result['units']) + ['1', '']
    result['rows'] = [list(row) + [resonance.order, resonance.family]
                      for row, resonance in zip(result['rows'], resonances)]
    result['multipole_orders'] = list(physics.eigen_solve.info['multipole_orders'])
    result['families'] = list(physics.eigen_solve.info['families'])
    return result
