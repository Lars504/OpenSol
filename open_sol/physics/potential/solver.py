"""Quasi-static potential solvers on the DiffusionSystem: electric currents (-div(sigma_dc grad V) = s) and
electrostatics (-div(eps0 eps_dc grad V) = rho_free + rho_ion, rho_ion of linearised or Boltzmann electrolytes, the
nonlinear one by damped Newton), 2D per unit z length or axisymmetric (r, z) whole rings."""
import functools
import math
from dataclasses import dataclass

import numpy as np

from ...core.backend import cp, to_host
from ...core.constants import DTYPE, EPS0
from ...core.cut_cells import locate_held_edges
from ...core.damped_iteration import DampedIteration, DampingSettings
from ...core.mirror import touch_factor
from ...core.fv_diffusion import (BCSide, DiffusionPhysicsBase, DiffusionSystem, ScalarBoundaryConfig,
                                  edge_extension_views)
from ...core.sources import ContinuousWave, GaussianPulse, RickerWavelet
from .electrolyte import (ELECTROLYTE_MODELS, boltzmann_terms, number_densities, screening_coefficient,
                          species_concentration, thermal_voltage)
from .fields import PotentialFields2D
from .renderer import CurrentsRenderer, ElectrostaticsRenderer


@dataclass(frozen=True)
class KindSpec:
    """Vocabulary of one potential kind: coefficient arrays and their scale, source slot / field, flux names, units."""
    label: str
    coefficient_scale: float    # multiplies the coefficient arrays at assembly (1 for sigma_dc, eps0 for eps_dc)
    coefficient_arrays: tuple   # MaterialGrid arrays (xx, yy, xy) the operator reads
    schema: dict                # MaterialGrid defaults of those arrays (electrostatics: + POCKELS_SCHEMA)
    slot: str                   # inject_source name (the coupled source)
    flux_names: tuple           # state / renderer names of flux_x, flux_y
    source_field: str           # state / renderer name of the coupled source (applied: '<name>_ext')
    with_joule: bool            # q_joule allocated and computed
    with_ions: bool             # electrolytes allowed, rho_ion allocated
    flux_unit: str             # unit of a held-region flux per unit z length
    ring_flux_unit: str         # unit of a held-region flux of the whole ring (axisymmetric)
    anchor_text: str            # anchor list of the singular-system message


# Pockels coefficients [m/V] per optical component (diagonal of r for one applied direction), read only by the
# electrostatics_em_pockels coupling; registered with the electrostatics arrays so scripts see the same schema.
POCKELS_SCHEMA = {'r_xx': 0.0, 'r_yy': 0.0, 'r_zz': 0.0}
# In-plane Pockels tensor of a crystal in model axes (pockels_r<I><k>: index-ellipsoid row I = 1 xx, 2 yy, 3 zz, 6 xy
# vs the field component k = 1 x, 2 y) [m/V]: optional electrostatics MaterialGrid arrays (any one set allocates all
# eight, default 0), read only by electrostatics_em_pockels (both E components, beside r_ii E_k; row 6 = eps_xy).
POCKELS_TENSOR_ARRAYS = ('pockels_r11', 'pockels_r12', 'pockels_r21', 'pockels_r22', 'pockels_r31', 'pockels_r32',
                         'pockels_r61', 'pockels_r62')
OPTIONAL_POCKELS_SCHEMA = {name: (0.0, POCKELS_TENSOR_ARRAYS) for name in POCKELS_TENSOR_ARRAYS}

# The schemas also hold the zz component (sigma_dc_zz / eps_dc_zz, read only by the tetrahedron solver
# FEMPotentialPhysics3D; 2D ignores it, as heat's k_zz).
KINDS = {
    'currents': KindSpec('electric currents', 1.0, ('sigma_dc_xx', 'sigma_dc_yy', 'sigma_dc_xy'),
                         {'sigma_dc_xx': 0.0, 'sigma_dc_yy': 0.0, 'sigma_dc_xy': 0.0, 'sigma_dc_zz': 0.0},
                         'current_source', ('J_x', 'J_y'), 'current_source', True, False, 'A/m', 'A',
                         'voltage boundary or terminal cell'),
    'electrostatics': KindSpec('electrostatics', EPS0, ('eps_dc_xx', 'eps_dc_yy', 'eps_dc_xy'),
                               {'eps_dc_xx': 1.0, 'eps_dc_yy': 1.0, 'eps_dc_xy': 0.0, 'eps_dc_zz': 1.0,
                                **POCKELS_SCHEMA},
                               'charge_density', ('D_x', 'D_y'), 'rho_free', False, True, 'C/m', 'C',
                               'voltage boundary, terminal cell or electrolyte cell'),
}

# Material property -> MaterialGrid arrays it writes (isotropic entries expand to xx, yy and zz; r to all three).
POTENTIAL_PROPERTY_ARRAYS = {
    'sigma_dc': ('sigma_dc_xx', 'sigma_dc_yy', 'sigma_dc_zz'), 'sigma_dc_xx': ('sigma_dc_xx',),
    'sigma_dc_yy': ('sigma_dc_yy',), 'sigma_dc_zz': ('sigma_dc_zz',), 'sigma_dc_xy': ('sigma_dc_xy',),
    'eps_dc': ('eps_dc_xx', 'eps_dc_yy', 'eps_dc_zz'), 'eps_dc_xx': ('eps_dc_xx',), 'eps_dc_yy': ('eps_dc_yy',),
    'eps_dc_zz': ('eps_dc_zz',), 'eps_dc_xy': ('eps_dc_xy',),
    'r': tuple(POCKELS_SCHEMA), 'r_xx': ('r_xx',), 'r_yy': ('r_yy',), 'r_zz': ('r_zz',),
    **{name: (name,) for name in POCKELS_TENSOR_ARRAYS},
}

# Scalar output (E5) prefix per kind: <prefix><feature> = held_flux of a voltage / terminal region [A/m | C/m].
HELD_SCALAR_PREFIXES = {'currents': 'current_', 'electrostatics': 'charge_'}
# Scalar outputs of an electrolyte feature: <prefix><feature> = the ElectrolyteRegion attribute.
ELECTROLYTE_SCALAR_PREFIXES = {'ionic_strength_': 'ionic_strength', 'debye_length_': 'debye_length'}

# Newton step cap of the nonlinear electrolyte: max |dV| per iteration in units of min V_T / max |z|.
NEWTON_MAX_STEP = 4.0

REMEDY_TEXT = 'Add a voltage or terminal feature touching every conducting region.'
RENDERERS = {'currents': CurrentsRenderer, 'electrostatics': ElectrostaticsRenderer}
TERMINAL_WAVEFORMS = ('none', 'ricker', 'gaussian', 'cw')   # 'none' = held at the voltage (the DC part)
_DRIVEN_QUANTITIES = 5   # V, E_x, E_y, flux_x, flux_y per basis solution

# V, E, D = basis_dc + sum_w s_w(t) basis_w (elementwise, all quantities stacked): work += value * basis
_add_scaled = cp.ElementwiseKernel('float32 basis, float32 value', 'float32 work', 'work += value * basis',
                                   'potential_add_scaled_basis')


def _add_scaled_host(basis, value, work):
    """numpy twin of _add_scaled."""
    work += value * basis


@dataclass(frozen=True)
class TerminalWaveform:
    """Time factor s(t) of a driven voltage / terminal (the elastic_waves source vocabulary): 'ricker', 'gaussian' (sin
    carrier, relative bandwidth) or 'cw' (ramped sin) at frequency [Hz]; equal waveforms share one basis solve."""
    kind: str
    frequency: float
    bandwidth: float = 0.5

    def __post_init__(self):
        if self.kind not in TERMINAL_WAVEFORMS[1:]:
            raise ValueError(f"terminal waveform must be one of {', '.join(TERMINAL_WAVEFORMS[1:])}, got {self.kind!r}")
        if not (self.frequency > 0.0 and self.bandwidth > 0.0):
            raise ValueError(f'terminal waveform needs frequency > 0 and bandwidth > 0, got {self.frequency!r}, '
                             f'{self.bandwidth!r}')

    def __call__(self, t) -> float:
        return float(_temporal(self)(t))


@functools.lru_cache(maxsize=None)
def _temporal(waveform):
    """core.sources waveform object of a TerminalWaveform."""
    if waveform.kind == 'ricker':
        return RickerWavelet(waveform.frequency)
    if waveform.kind == 'gaussian':
        return GaussianPulse(waveform.frequency, waveform.bandwidth)
    return ContinuousWave(waveform.frequency)


@dataclass(frozen=True, eq=False)
class HeldRegion:
    """Runtime object of a voltage / terminal feature: held outer faces and / or held window cells at one voltage
    (times its waveform when driven), the faces behind a Stern layer when stern_capacitance > 0."""
    name: str
    voltage: float
    faces: tuple            # outer faces held by a Dirichlet (Stern layer: Robin) condition
    mask_window: object     # cupy bool window mask of the held cells, or None
    waveform: object = None   # TerminalWaveform, None = held at the voltage
    stern_capacitance: float = 0.0   # C_S [F/m^2] of the faces' Stern layer, 0 = none


def held_face_bc(voltage, stern_capacitance=0.0) -> BCSide:
    """BC of a voltage face: Dirichlet at the face, or behind a Stern layer (C_S > 0) Robin h = C_S, external = voltage
    (C_S in series with the half-cell conductance)."""
    if stern_capacitance > 0.0:
        return BCSide('robin', h=stern_capacitance, external=voltage)
    return BCSide('dirichlet', value=voltage)


def _face_held_by(bc, region) -> bool:
    """True while `bc` is still the voltage face BC of `region` (no later feature replaced it)."""
    if region.stern_capacitance > 0.0:
        return bc.kind == 'robin' and bc.external == region.voltage and bc.h == region.stern_capacitance
    return bc.kind == 'dirichlet' and bc.value == region.voltage


@dataclass(frozen=True)
class ElectrolyteRegion:
    """Runtime object of an electrolyte feature: ionic strength [mol/L] and Debye length [m] (mean eps_dc at build)."""
    name: str
    ionic_strength: float
    debye_length: float


@dataclass(frozen=True, eq=False)
class IonicMedium:
    """Ion species of one add_electrolyte call on its window cells (model 'linear': rho_ion = -coefficient (V -
    bulk_potential); 'nonlinear': the Boltzmann sum of electrolyte.boltzmann_terms)."""
    mask_window: object        # cupy bool window mask
    model: str
    concentrations: tuple      # bulk c_i [mol/L]
    densities: tuple           # n_i [1/m^3]
    valences: tuple
    temperature: float         # K
    bulk_potential: float      # V
    coefficient: float         # linearised screening k [F/m^3]


class PotentialPhysics(DiffusionPhysicsBase):
    """2D quasi-static potential V on the physical window (electric currents or electrostatics, `kind`; planar or
    axisymmetric (r, z); azimuthal_order m != 0: V is the amplitude of V cos(m phi)): one DiffusionSystem solve per
    change of operator or sources (a damped Newton loop of them with a nonlinear electrolyte); outputs edge-extended
    into PML."""

    name = "currents"
    _probe_field = 'V'
    _probe_label = 'potential probe'
    conformal_locator = None   # model.boundary_locator.GeometryLocator of set_conformal_boundaries (cut-cell held edges)
    mirror_faces = frozenset()   # model mirror faces on the window edge (set_mirror_faces): held fluxes join images

    def set_conformal_boundaries(self, locator):
        """solver_defaults.conformal_boundaries: held features pass locator.inside(entities) (model.boundary_locator,
        duck-typed) to add_fixed_potential, perimeter current_density / surface_charge take the drawn outline,
        non-conducting walls become cut cells at the first assembly (re-located when the conducting cells change);
        None = staircased."""
        self.conformal_locator = locator
        self._set_insulated_wall_locator(locator)

    def set_mirror_faces(self, faces):
        """Mirror planes of the model (core.mirror.mirror_faces: an EM mirror face on the window edge, where this
        physics' face is the even mirror): held_flux of a region touching one counts its image too (x 2 per face)."""
        self.mirror_faces = frozenset(faces)

    def __init__(self, domain, kind: str = 'currents', tol: float = 1e-8, max_iter: int = 5000,
                 compute_residual: bool = False, precision: str = 'float64', axisymmetric: bool = False,
                 nonlinear_tolerance: float = 1e-6, nonlinear_max_iterations: int = 100,
                 nonlinear_relaxation: float = 1.0, linear_solver=None, azimuthal_order: int = 0):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {list(KINDS)}, got {kind!r}")
        self.kind = kind
        self.spec = KINDS[kind]
        self.material_schema = dict(self.spec.schema)
        system = DiffusionSystem(   # linear_solver (the build's pcg) replaces precision / tol / max_iter
            domain, precision, tol, max_iter, label=lambda: f"{self.spec.label} '{self.name}'",
            unknown_name='potential', anchor_text=self.spec.anchor_text, remedy_text=REMEDY_TEXT,
            compute_residual=compute_residual, axisymmetric=axisymmetric, linear_solver=linear_solver,
            azimuthal_order=azimuthal_order)
        self._init_diffusion(domain, ScalarBoundaryConfig(), system)
        self._source_dirty = True
        self.held_regions = []   # HeldRegions of add_held_region, reported by scalars()
        self.electrolytes = []   # ElectrolyteRegions of add_electrolyte_region, reported by scalars()
        self.ionic_media = []    # IonicMedium of every add_electrolyte: the assembly writes the ionic diagonal from them
        self._ion_mask = None    # union of their window masks
        self.runtime_warnings = []   # append-only (unconverged nonlinear electrolyte solves); the studies log new ones
        # driven terminals (decision 4 of the piezoelectric plan): every held potential in order (waveform or None,
        # window mask, values), the driven outer faces {face: (waveform, its BCSide)}, the basis solutions
        # {waveform or None: (5, window) float32 V, E_x, E_y, flux_x, flux_y} and their held-region charges
        self._cell_holds = []
        self._driven_faces = {}
        self._bases = None
        self._basis_fluxes = None
        self._superposition = None
        self._zero_sources = None
        self._drive_time = 0.0
        self._n_superpositions = 0
        self.newton = DampedIteration(
            DampingSettings(nonlinear_tolerance, nonlinear_max_iterations, nonlinear_relaxation, NEWTON_MAX_STEP),
            label=lambda: f"{self.spec.label} '{self.name}' nonlinear electrolyte")

        self.prepare(domain)

    # --- PhysicsSolver contract ---------------------------------------------

    def prepare(self, domain) -> None:
        """Register the coefficient arrays, prepare the window system and allocate the fields; assembly is deferred."""
        self.domain = domain
        self.xp = domain.xp   # numpy (cpu backend) or cupy
        domain.materials.add_properties(self.material_schema)
        self.system.prepare(domain)

        nx, ny = domain.grid.nx, domain.grid.ny
        if self.fields is None or (self.fields.nx, self.fields.ny) != (nx, ny):
            self.fields = PotentialFields2D(nx, ny, self.spec.with_joule, self.spec.with_ions, xp=self.xp)
        window = self.system.window
        self._V_window = self.fields.V[window]
        self._source_window = self.fields.source[window]
        self._applied_source_window = self.fields.source_ext[window]
        x_range, y_range = self._bind_window(self.fields.V)
        derived_arrays = [self.fields.E_x, self.fields.E_y, self.fields.flux_x, self.fields.flux_y]
        derived_arrays.extend(array for array in (self.fields.q_joule, self.fields.rho_ion) if array is not None)
        self._derived_pml_copies = [pair for array in derived_arrays
                                    for pair in edge_extension_views(array, x_range, y_range)]
        self._source_dirty = True

    def finalize(self) -> None:
        """Assemble once (fails early on a conducting region without a held potential)."""
        self.rebuild_operator()

    def materials_changed(self, names) -> None:
        if set(self.spec.coefficient_arrays) & set(names):
            self.system.operator_dirty = True
        if self.spec.coefficient_arrays[2] in names:   # the xy array: kxy_active re-decided at the next assembly
            self.system.cross_term_dirty = True

    def step(self, t: float = 0.0, dt: float = math.inf) -> None:
        """Re-solve only when the operator or a source changed (always at the first call); probes sample after a
        re-solve and at every dt = inf call (stationary iteration, end-of-run flush), never on unchanged time steps.
        With driven terminals: the superposition at t + dt (t when dt is inf) every call (_step_driven)."""
        t = 0.0 if t is None else float(t)
        if self.waveforms:
            self._step_driven(t, dt)
            return
        system = self.system
        re_solve = system.operator_dirty or self._source_dirty or system.n_solves == 0
        if re_solve:
            if self.nonlinear:
                self._solve_nonlinear()
            else:
                if system.operator_dirty:
                    self._assemble()
                system.build_rhs(self._source_window, self._applied_source_window)
                system.solve()
            self.xp.copyto(self._V_window, system.unknown2d)
            self._extend_into_pml()
            self._update_derived_fields()
            self._source_dirty = False
        if re_solve or dt is None or not math.isfinite(dt):
            self._record_probes(t)

    def solve_coupled_source(self, time: float) -> None:
        """Re-solve within a tick after a coupled source change (two-way piezoelectric coupling): the fields at `time`
        (driven terminals: their superposition there), probes sampled at that time."""
        self.step(float(time), math.inf)

    def reset_records(self) -> None:
        """RerunSolver (reused sweep build): clear the probe series; the next run re-solves when an input changed."""
        for probe in self.probes:
            probe.clear()

    def dt_limit(self) -> float:
        return math.inf

    def state(self) -> dict:
        fields = self.fields
        spec = self.spec
        materials = self.domain.materials
        state = {
            'V': fields.V, 'E_x': fields.E_x, 'E_y': fields.E_y,
            spec.flux_names[0]: fields.flux_x, spec.flux_names[1]: fields.flux_y,
            spec.source_field: fields.source, f'{spec.source_field}_ext': fields.source_ext,
        }
        for array_name in spec.coefficient_arrays:
            state[array_name] = getattr(materials, array_name)
        if fields.q_joule is not None:
            state['q_joule'] = fields.q_joule
        if fields.rho_ion is not None:
            state['rho_ion'] = fields.rho_ion
        return state

    def inject_source(self, name: str, array) -> None:
        if name != self.spec.slot:
            raise ValueError(f"{self.spec.label} '{self.name}' accepts source '{self.spec.slot}'; got '{name}'")
        self.xp.copyto(self.fields.source, self.xp.asarray(array, dtype=DTYPE))
        self._source_dirty = True

    def renderer(self):
        return RENDERERS[self.kind](self)

    # --- potential-specific API (shared settings, set_face_bc, add_boundary_flux, probes: DiffusionPhysicsBase) ---

    def add_fixed_potential(self, mask, V, waveform=None, inside=None) -> None:
        """Hold V [V] (scalar or full-grid array) on window cells under mask, times a TerminalWaveform s(t) when given
        (driven); also writes V. inside(x, y) -> bool (host, the drawn held region; 2D): held edge at its drawn
        position (core.cut_cells cut cells)."""
        system = self.system
        mask_window = system.window_mask(mask, 'fixed potential').copy()
        potential = system.window_values(V, 'fixed potential', system.dtype)
        system.set_fixed(mask_window, potential)
        if inside is not None:
            locate_held_edges(system, mask_window, inside)
        self._cell_holds.append((waveform, mask_window, potential))
        self.xp.copyto(self._V_window, potential, where=mask_window)
        self._extend_into_pml()

    def set_driven_face_voltage(self, face: str, voltage: float, waveform, stern_capacitance: float = 0.0) -> None:
        """V = voltage s(t) on an outer face (a TerminalWaveform s; behind a Stern layer of C_S [F/m^2] when > 0; a
        later BC on the face replaces it)."""
        bc = held_face_bc(voltage, stern_capacitance)
        self.set_face_bc(face, bc)
        self._driven_faces[face] = (waveform, bc)

    @property
    def waveforms(self) -> tuple:
        """Distinct TerminalWaveforms of the driven held cells and of the driven faces still holding their BC."""
        found = [waveform for waveform, _, _ in self._cell_holds if waveform is not None]
        found += [waveform for face, (waveform, bc) in self._driven_faces.items() if getattr(self.bcs, face) is bc]
        return tuple(dict.fromkeys(found))

    @property
    def state_revision(self) -> int:
        """Counter of produced states (solves + driven superpositions): coupling transfers skip while unchanged."""
        return self.system.n_solves + self._n_superpositions

    def add_source(self, mask, density) -> None:
        """Add a volumetric source density (scalar or full-grid array) to source_ext on window cells under mask."""
        system = self.system
        system.add_masked(self._applied_source_window, system.window_mask(mask, f'{self.spec.slot} source'), density,
                          f'{self.spec.slot} source density')
        self._source_dirty = True

    def add_electrolyte(self, mask, concentrations, valences, temperature: float = 298.15,
                        bulk_potential: float = 0.0, model: str = 'linear') -> IonicMedium:
        """Ion species (concentrations [mol/L], valences) on window cells under mask (electrostatics only; held cells
        carry none; overlapping electrolytes add): model 'linear' (Debye-Hueckel) or 'nonlinear' (Boltzmann)."""
        if not self.spec.with_ions:
            raise ValueError(f"{self.spec.label} '{self.name}' has no electrolytes (electrostatics only)")
        if model not in ELECTROLYTE_MODELS:
            raise ValueError(f"electrolyte model must be one of {ELECTROLYTE_MODELS}, got {model!r}")
        if model == 'nonlinear' and self.azimuthal_order:
            raise ValueError(f"{self.spec.label} '{self.name}': a nonlinear electrolyte needs azimuthal_order 0 (the "
                             f"Boltzmann charge of an order-{self.azimuthal_order} potential is no single order)")
        concentrations = tuple(float(value) for value in concentrations)
        valences = tuple(float(value) for value in valences)
        if (not concentrations or len(concentrations) != len(valences) or 0.0 in valences
                or not all(math.isfinite(value) and value >= 0.0 for value in concentrations)):
            raise ValueError(f"electrolyte needs one non-zero valence per finite concentration >= 0, got "
                             f"{concentrations} and {valences}")
        if not float(temperature) > 0.0:
            raise ValueError(f"electrolyte temperature must be > 0 K, got {temperature}")
        mask_window = self.system.window_mask(mask, 'electrolyte').copy()
        medium = IonicMedium(mask_window, model, concentrations, number_densities(concentrations), valences,
                             float(temperature), float(bulk_potential),
                             screening_coefficient(concentrations, valences, temperature))
        self.ionic_media.append(medium)
        self._ion_mask = mask_window.copy() if self._ion_mask is None else self._ion_mask | mask_window
        self.system.operator_dirty = True
        return medium

    @property
    def nonlinear(self) -> bool:
        """True when an electrolyte uses the nonlinear (Boltzmann) model: step runs the damped Newton loop."""
        return any(medium.model == 'nonlinear' for medium in self.ionic_media)

    @property
    def ion_species_count(self) -> int:
        """Longest species list of the electrolytes (0 without one): the renderer lists c_ion_1 .. up to it."""
        return max((len(medium.valences) for medium in self.ionic_media), default=0)

    def ion_concentration(self, species_index: int):
        """Window float32 concentration [mol/L] of species `species_index` (0-based list entry) of every electrolyte at
        the current V (electrolyte.species_concentration; overlapping media add), 0 on held cells and without ions."""
        xp = self.xp
        potential = self._V_window.astype(np.float64)
        total = xp.zeros(potential.shape, dtype=np.float64)
        for medium in self.ionic_media:
            if species_index < len(medium.valences):
                concentration = species_concentration(
                    potential, medium.bulk_potential, medium.concentrations[species_index],
                    medium.valences[species_index], medium.temperature, medium.model, xp)
                total += xp.where(medium.mask_window, concentration, 0.0)
        return xp.where(self.system.fixed_user, 0.0, total).astype(DTYPE)

    def work_counters(self) -> dict:
        """The Newton loop's counters with a nonlinear electrolyte, else {} (the CG counters are the linear solver's,
        summary 'solvers' roles)."""
        if not self.nonlinear:
            return {}
        return {**self.newton.work_counters(), 'diagonal_refreshes': self.system.n_diagonal_refreshes}

    def add_electrolyte_region(self, region: ElectrolyteRegion) -> ElectrolyteRegion:
        """Keep an electrolyte feature's region for scalars(); returns it."""
        self.electrolytes.append(region)
        return region

    def mean_relative_permittivity(self, mask) -> float:
        """Mean of (eps_dc_xx + eps_dc_yy) / 2 over the window cells under mask (build-time syncs; NaN for an empty
        mask)."""
        materials = self.domain.materials
        mask_window = self.system.window_mask(mask, 'permittivity mean')
        window = self.system.window
        permittivity = 0.5 * (materials.eps_dc_xx[window].astype(np.float64) + materials.eps_dc_yy[window])
        n_cells = int(self.xp.count_nonzero(mask_window))
        return float(self.xp.where(mask_window, permittivity, 0.0).sum()) / n_cells if n_cells else math.nan

    def rebuild_operator(self) -> None:
        """Re-assemble the operator, held-cell elimination, anchor check and Jacobi preconditioner."""
        self._assemble()

    def warm_start_fields(self) -> dict:
        """Host copies of the window V and coupled source, to seed a build on the same mesh (sweep warm start)."""
        return {'V': np.array(to_host(self.system.unknown2d)), 'source': np.array(to_host(self._source_window))}

    def seed_warm_start(self, fields) -> tuple:
        """Write warm_start_fields() of another build (V on free cells, source); returns the state slots now seeded
        (the derived E, flux and q_joule wait for the next solve)."""
        self.system.seed_unknown(fields['V'])
        self.xp.copyto(self._V_window, self.system.unknown2d)
        self._extend_into_pml()
        self.xp.copyto(self._source_window, self.xp.asarray(fields['source'], dtype=DTYPE))
        self._source_dirty = True
        return ('V', self.spec.source_field)

    def add_held_region(self, region: HeldRegion) -> HeldRegion:
        """Keep a voltage / terminal region for scalars(); returns it."""
        self.held_regions.append(region)
        return region

    @property
    def flux_unit(self) -> str:
        """Unit of held_flux: per unit z length (A/m, C/m), whole rings when axisymmetric (A, C), per unit area on a 1D
        slab (A/m^2, C/m^2: its one cell across is 1 m wide)."""
        if self.axisymmetric:
            return self.spec.ring_flux_unit
        coordinates = self.system.domain.coordinates
        return f'{self.spec.flux_unit}^2' if coordinates.line and not coordinates.polar else self.spec.flux_unit

    def scalars(self) -> dict:
        """current_<feature> or charge_<feature> [flux_unit] = held_flux of every kept region, ionic_strength_<feature>
        [mol/L] / debye_length_<feature> [m] of every electrolyte; {} before a solve."""
        if self.system.n_solves == 0:
            return {}
        prefix = HELD_SCALAR_PREFIXES[self.kind]
        scalars = {f'{prefix}{region.name}': self.held_flux(region) for region in self.held_regions}
        for region in self.electrolytes:
            scalars.update({f'{name_prefix}{region.name}': getattr(region, attribute)
                            for name_prefix, attribute in ELECTROLYTE_SCALAR_PREFIXES.items()})
        return scalars

    def held_flux(self, region: HeldRegion) -> float:
        """Flux [flux_unit: A/m, C/m; axisymmetric A, C] leaving the held faces / cells of `region` (> 0 out);
        faces a later feature re-set (no longer dirichlet at region.voltage) are skipped; held cells: sum of A V - b;
        x mirror_factor(region) (its images under the model's mirror planes)."""
        if self.system.n_solves == 0:
            raise RuntimeError(f"{self.spec.label} '{self.name}': the flux of '{region.name}' needs a solve first")
        if self._bases is not None:   # driven: the basis fluxes superposed at the time of the current fields
            index = next((position for position, kept in enumerate(self.held_regions) if kept is region), None)
            if index is None:
                raise ValueError(f"held region '{region.name}' was not added with add_held_region")
            flux = (self._basis_fluxes[None][index]
                    + sum(waveform(self._drive_time) * fluxes[index]
                          for waveform, fluxes in self._basis_fluxes.items() if waveform is not None))
        else:
            flux = self._region_flux(region, {face: getattr(self.bcs, face) for face in region.faces})
        return float(flux) * self.mirror_factor(region)

    def mirror_factor(self, region: HeldRegion) -> float:
        """core.mirror.touch_factor of the mirror faces `region` touches: a held outer face on the plane or meeting it
        (any face but the opposite one), held cells reaching the window edge layer there."""
        touched = set()
        for mirror in self.mirror_faces:
            opposite = mirror[0] + ('_hi' if mirror.endswith('lo') else '_lo')
            if any(face != opposite for face in region.faces):
                touched.add(mirror)
            elif region.mask_window is not None:
                axis = 'xy'.index(mirror[0])
                layer = [slice(None)] * 2
                layer[axis] = 0 if mirror.endswith('lo') else -1
                if bool(region.mask_window[tuple(layer)].any()):
                    touched.add(mirror)
        return touch_factor(self.mirror_faces, touched)

    def _region_flux(self, region, face_bcs) -> float:
        """held_flux of the last assembly; a face counts while face_bcs[face] is still the region's voltage BC
        (_face_held_by: dirichlet at region.voltage, Stern layer robin)."""
        still_held_faces = [face for face in region.faces if _face_held_by(face_bcs[face], region)]
        flux = sum(self.system.face_inward_flux(face) for face in still_held_faces)
        if region.mask_window is not None:
            flux += self.system.fixed_region_outflow(region.mask_window)
        return float(flux)

    # --- internals ------------------------------------------------------------

    def _coefficients(self):
        """(c_xx, c_yy, c_xy) window copies in the solve dtype, scaled by the kind's coefficient scale."""
        materials = self.domain.materials
        window = self.system.window
        dtype = self.system.dtype
        scale = self.spec.coefficient_scale
        coefficients = []
        for array_name in self.spec.coefficient_arrays:
            values = getattr(materials, array_name)[window].astype(dtype)
            if scale != 1.0:
                values *= scale
            coefficients.append(values)
        return tuple(coefficients)

    def _assemble(self, potential=None, diagonal_only=False):
        """Assemble with the ionic diagonal linearised at `potential` (float64 window; None = the current V; exact for
        linear electrolytes); diagonal_only: DiffusionSystem.refresh_diagonal when only that diagonal changed."""
        system = self.system
        if self.ionic_media:
            self._write_ionic_diagonal(system.unknown2d if potential is None else potential)
            # held cells are ideal conductors without ions (cleared here: cells held later too)
            system.set_diagonal_loss(system.fixed_user, 0.0, 0.0, 0.0)
        if diagonal_only and system.refresh_diagonal():
            return
        c_xx, c_yy, c_xy = self._coefficients()
        system.assemble(c_xx, c_yy, c_xy, self.bcs, 0.0)

    def _ionic_terms(self, potential):
        """(rho_ion [C/m^3], screening -d rho_ion / dV [F/m^3]) of every electrolyte at a window potential (float64)."""
        xp = self.xp
        potential = potential.astype(np.float64, copy=False)
        charge = xp.zeros(potential.shape, dtype=np.float64)
        screening = xp.zeros(potential.shape, dtype=np.float64)
        for medium in self.ionic_media:
            if medium.model == 'nonlinear':
                medium_charge, medium_screening = boltzmann_terms(
                    potential, medium.bulk_potential, medium.densities, medium.valences, medium.temperature, xp)
            else:
                medium_charge = -medium.coefficient * (potential - medium.bulk_potential)
                medium_screening = medium.coefficient
            charge += xp.where(medium.mask_window, medium_charge, 0.0)
            screening += xp.where(medium.mask_window, medium_screening, 0.0)
        return charge, screening

    def _write_ionic_diagonal(self, potential):
        """Replace the diagonal on the electrolyte cells by the tangent of rho_ion at `potential`: rho_ion(V) ~
        -k0 (V - V_ext), k0 the screening, V_ext = V0 + rho_ion(V0) / k0 (linear: the k-weighted bulk potential)."""
        xp = self.xp
        potential = potential.astype(np.float64, copy=False)
        charge, screening = self._ionic_terms(potential)
        screens = screening > 0.0
        external = xp.where(screens, potential + charge / xp.where(screens, screening, 1.0), 0.0)
        self.system.set_diagonal_loss(self._ion_mask, screening, external, 0.0)

    def _solve_nonlinear(self):
        """Damped Newton on V (core.damped_iteration): each iteration refreshes the ionic tangent at the current V (a
        full assembly only when more changed), warm-starts the CG there and solves; converged on max |dV| / (min V_T /
        max |z|) of the nonlinear electrolytes."""
        system = self.system
        scale = min(thermal_voltage(medium.temperature) / max(abs(valence) for valence in medium.valences)
                    for medium in self.ionic_media if medium.model == 'nonlinear')
        potential = system.unknown2d.astype(np.float64)

        def propose(current):
            self._assemble(current, diagonal_only=True)
            system.seed_unknown(current)
            system.build_rhs(self._source_window, self._applied_source_window)
            system.solve()
            return system.unknown2d.astype(np.float64)

        if not self.newton.run(potential, propose, scale):
            self.runtime_warnings.append(self.newton.last_warning)
        system.seed_unknown(potential)   # unconverged: the damped state (converged: the last solution already)

    def _step_driven(self, t, dt):
        """Driven terminals: (re)solve the basis solutions when stale (all after an operator change, the DC one after a
        source change), then write V, E, D = DC + sum_w s_w(time) basis_w at time = t + dt (t for dt inf) and sample
        the probes there."""
        system = self.system
        if self._bases is None or system.operator_dirty or system.n_solves == 0:
            self._solve_bases()
        elif self._source_dirty:   # the live assembly is the DC one (solved last)
            self._solve_basis(None, {face: getattr(self.bcs, face) for face in system.faces})
        self._source_dirty = False
        time = t + dt if dt is not None and math.isfinite(dt) else t
        self._superpose(time)
        self._record_probes(time)

    def _record_probes(self, time):
        """Sample every probe at `time`; at time > 0 a sample already at that time (a re-solve within the same tick,
        the end-of-run flush) is replaced, not repeated (stationary iterations at t = 0 keep one sample each)."""
        for probe in self.probes:
            probe.record(self.fields, time, replace_same_time=time > 0.0)

    def _solve_bases(self):
        """One solve per distinct waveform (its holds at their voltage, every other hold, BC value and source 0), then
        the DC solve (undriven holds, BCs and sources; driven holds 0), which stays the live assembly."""
        if self.ionic_media:
            raise ValueError(f"{self.spec.label} '{self.name}': driven terminals (waveform) cannot be combined with an "
                             'electrolyte (the superposition needs a source-free linear operator)')
        system = self.system
        waveforms = self.waveforms
        saved_bcs = {face: getattr(self.bcs, face) for face in system.faces}
        xp = self.xp
        if self._superposition is None:
            shape = (_DRIVEN_QUANTITIES, *system.window_shape)
            self._superposition = xp.empty(shape, dtype=DTYPE)
            self._zero_sources = xp.zeros(system.window_shape, dtype=DTYPE)
        self._bases, self._basis_fluxes = {}, {}
        try:
            for waveform in (*waveforms, None):
                # the bases differ only in held / face values: an assembled operator is re-eliminated, not rebuilt
                refreshed = True
                for face, bc in saved_bcs.items():
                    old_bc, new_bc = getattr(self.bcs, face), self._basis_bc(face, bc, waveform)
                    setattr(self.bcs, face, new_bc)
                    if new_bc is not old_bc and refreshed:
                        refreshed = system.update_face_value(face, old_bc, new_bc)
                zero = xp.zeros((), dtype=system.dtype)
                for hold_waveform, mask_window, potential in self._cell_holds:
                    system.set_fixed(mask_window, potential if hold_waveform == waveform else zero)
                if not (refreshed and (system.refresh_held_values() or not system.operator_dirty)):
                    self._assemble()
                self._solve_basis(waveform, saved_bcs)
        finally:
            for face, bc in saved_bcs.items():
                setattr(self.bcs, face, bc)

    def _basis_bc(self, face, bc, waveform):
        """The BC of `face` in the basis of `waveform` (None = DC): unchanged where the face belongs to that basis
        (driven by it, or undriven in the DC basis), else the same kind with value 0."""
        driven = self._driven_faces.get(face)
        owner = driven[0] if driven is not None and driven[1] is bc else None
        if owner == waveform:
            return bc
        if bc.kind == 'dirichlet':
            return BCSide('dirichlet', value=0.0, conductance_factor=bc.conductance_factor)
        if bc.kind == 'robin':
            return BCSide('robin', h=bc.h, external=0.0, conductance_factor=bc.conductance_factor)
        return BCSide('neumann')

    def _solve_basis(self, waveform, saved_bcs):
        """Solve the assembled basis (sources only in the DC one), keep V, E, D and the held-region fluxes."""
        system = self.system
        if waveform is None:
            system.build_rhs(self._source_window, self._applied_source_window)
        else:
            system.build_rhs(self._zero_sources, self._zero_sources)
        system.solve()
        xp = self.xp
        xp.copyto(self._V_window, system.unknown2d)
        self._update_derived_fields()
        fields, window = self.fields, system.window
        basis = xp.empty((_DRIVEN_QUANTITIES, *system.window_shape), dtype=DTYPE)
        for plane, array in zip(basis, (fields.V, fields.E_x, fields.E_y, fields.flux_x, fields.flux_y)):
            xp.copyto(plane, array[window])
        self._bases[waveform] = basis
        self._basis_fluxes[waveform] = [self._region_flux(region, {face: saved_bcs[face] for face in region.faces})
                                        for region in self.held_regions]

    def _superpose(self, time):
        """V, E, D of the window = DC + sum_w s_w(time) basis_w, edge-extended into the PML ring."""
        xp = self.xp
        add_scaled = _add_scaled_host if self.system.backend == 'cpu' else _add_scaled
        work = self._superposition
        xp.copyto(work, self._bases[None])
        for waveform, basis in self._bases.items():
            if waveform is not None:
                value = waveform(time)
                if value != 0.0:
                    add_scaled(basis, DTYPE(value), work)
        fields, window = self.fields, self.system.window
        for plane, array in zip(work, (fields.V, fields.E_x, fields.E_y, fields.flux_x, fields.flux_y)):
            xp.copyto(array[window], plane)
        self._extend_into_pml()
        for destination, source in self._derived_pml_copies:
            xp.copyto(destination, source)
        self._drive_time = float(time)
        self._n_superpositions += 1

    def _update_derived_fields(self):
        """E (-grad V from the axis face fluxes), flux (J or D, + the xy cross part) and q_joule, rho_ion on the window
        (E and flux 0 in held cells, E 0 in non-conducting cells), edge-extended into the PML ring like V."""
        system = self.system
        window = system.window
        fields = self.fields
        c_xx, c_yy = system.assembled_c_xx, system.assembled_c_yy
        flux_x, flux_y = system.axis_flux_at_centres()
        xp = self.xp
        fields.E_x[window] = xp.where(c_xx > 0, flux_x / xp.where(c_xx > 0, c_xx, 1.0), 0.0)
        fields.E_y[window] = xp.where(c_yy > 0, flux_y / xp.where(c_yy > 0, c_yy, 1.0), 0.0)
        cross_flux = system.cross_flux_at_centres()
        if cross_flux is not None:   # sigma_dc_xy / eps_dc_xy: -c_xy (dV/dy, dV/dx)
            flux_x, flux_y = flux_x + cross_flux[0], flux_y + cross_flux[1]
        fields.flux_x[window] = flux_x
        fields.flux_y[window] = flux_y
        if fields.q_joule is not None:
            fields.q_joule[window] = system.dissipation_density()
        if self.ionic_media:   # exact at the solved V (the solve used the tangent at the previous Newton iterate)
            charge, _ = self._ionic_terms(system.unknown2d)
            fields.rho_ion[window] = xp.where(system.fixed_user, 0.0, charge)
        for destination, source in self._derived_pml_copies:
            xp.copyto(destination, source)

    def _applied_source_changed(self):
        self._source_dirty = True

    def _indicator_flux(self):
        """error_indicator's flux: the superposed J / D fields with driven terminals (the live assembly holds the DC
        basis), else None (the system's solve-precision flux)."""
        if not self.waveforms:
            return None
        window = self.system.window
        return self.fields.flux_x[window], self.fields.flux_y[window]
