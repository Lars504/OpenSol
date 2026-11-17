"""Shared part of the membrane solvers (MembraneBase): per-element coefficients of D (1 + j eta) del^4 w - div(T (1 +
j eta) grad w) - rho_s omega^2 w = p (material + coupled tension, plate bending, holes), the fem.harmonic system rebuilt
per PML reference, the eigenfrequency / frequency_response study hooks, harmonic probes and the per-element phasor
state. Host numpy (scipy in fem)."""
# exp(+j omega t); w = out-of-plane displacement [m], T = tension tensor [N/m] (sigma h of a film), D = bending stiffness
# [N m] (E h^3 / 12 (1 - nu^2)), rho_s = areal density [kg/m^2], eta = loss factor, p = pressure [Pa]. A plate (bending)
# solve runs when any material sets bending_stiffness (the optional array exists at prepare): Hermite line elements in
# 1D, the C0 interior penalty form on P2 triangles. A hole is an element with rho_s <= 0, every tension component 0
# (material + coupled) and no bending stiffness: it carries no dofs and its outline is a free edge. Subclasses
# (solver_fem: triangles, solver_line: 1D) provide the geometry: _prepare_geometry, _element_values, _field_array,
# _new_system, _apply_coefficients, _pml, _probe_value, _element_phasors, renderer.
import math
import time
from dataclasses import dataclass

import numpy as np

from ...core.backend import to_host
from ...core.constants import DTYPE, FACES
from ...core.harmonic_probe import HarmonicProbeRecord
from ...core.solver_base import PhysicsSolver
from ...fem.harmonic_resonances import solve_harmonic_resonances, solve_system_resonances
from ...fem.resonance_search import solve_resonances_at_targets, solve_resonances_in_band
from ...fem.reduced_outputs import field_policy, wants_fields
from ...solvers.arpack_shift_invert import ArpackShiftInvertSolver
from .materials import (BENDING_ARRAY, COEFFICIENT_ARRAYS, COUPLED_TENSION_SLOTS, DENSITY_ARRAY, LOSS_ARRAY,
                        MEMBRANE_SCHEMA, POISSON_ARRAY, TENSION_ARRAYS)
from .renderer import STATE_FIELD_NAMES

DISCRETIZATIONS = {'quadratic': 2, 'linear': 1}   # setting discretization -> Lagrange order
RESONANCE_LABEL = 'membrane'                      # names the resonances in the eigen-solve summary


@dataclass(frozen=True, eq=False)
class MembraneProbe:
    """Harmonic probe of w at one location (triangles: a mesh node; 1D: x [m]) with its HarmonicProbeRecord."""
    name: str
    location: object
    record: HarmonicProbeRecord


@dataclass(frozen=True, eq=False)
class ResonanceTableMonitor:
    """Runtime object of a resonance_table feature (its result reads the solver's eigen_solve)."""
    name: str


@dataclass(frozen=True, eq=False)
class MembraneCoefficients:
    """Host float64 per element: tension components [N/m] (material + coupled), loss factor, areal density, bending
    stiffness [N m] and plate Poisson ratio (0 without a plate), the hole mask, and whether any loss is set (complex
    operator)."""
    tension_xx: np.ndarray
    tension_yy: np.ndarray
    tension_xy: np.ndarray
    loss: np.ndarray
    density: np.ndarray
    hole: np.ndarray
    lossy: bool
    bending: np.ndarray
    poisson: np.ndarray

    def complex_bending(self) -> np.ndarray:
        """D (1 + j eta) per element (complex128 with loss, else float64)."""
        return self.bending * (1.0 + 1j * self.loss) if self.lossy else self.bending.copy()

    def wave_speed(self, elements, omega=0.0) -> float:
        """Largest phase speed [m/s] over the elements (bool mask) with mass and a restoring force: of the propagating
        plate wave at omega [rad/s], c^2 = (T + sqrt(T^2 + 4 D rho_s omega^2)) / (2 rho_s) (T = the largest principal
        tension, above max(T_xx, T_yy) with shear T_xy; D the bending stiffness; = T / rho_s without D or at omega 0); 0
        if none."""
        mean = 0.5 * (self.tension_xx + self.tension_yy)
        radius = np.hypot(0.5 * (self.tension_xx - self.tension_yy), self.tension_xy)
        tension = np.maximum(mean + radius, 0.0)   # the fastest film wave runs along the largest principal tension
        bending_term = 4.0 * self.bending * self.density * float(omega) ** 2
        squared = np.zeros_like(tension)
        usable = elements & (self.density > 0.0)
        squared[usable] = ((tension[usable] + np.sqrt(tension[usable] ** 2 + bending_term[usable]))
                           / (2.0 * self.density[usable]))
        usable &= squared > 0.0
        if not usable.any():
            return 0.0
        return float(np.sqrt(np.max(squared[usable])))


class MembraneBase(PhysicsSolver):
    """Membrane solver shared part (module header); the studies drive it through solve_eigenfrequencies /
    select_resonance and reset_frequency_response / solve_frequency_response (step does nothing)."""

    name = 'membrane'
    type_label = 'membrane'
    material_schema = dict(MEMBRANE_SCHEMA)
    coordinate_grids = ('cartesian',)
    conformal_locator = None     # moot: no tensor walls
    source_slots = COUPLED_TENSION_SLOTS

    def __init__(self, domain, *, linear_solver=None, eigen_solver=None, target_frequency=0.0,
                 discretization='quadratic', pml_frequency=None):
        if discretization not in DISCRETIZATIONS:
            raise ValueError(f"discretization must be one of {', '.join(DISCRETIZATIONS)}, got {discretization!r}")
        target_frequency = float(target_frequency)
        if not (math.isfinite(target_frequency) and target_frequency >= 0.0):
            raise ValueError(f'membrane target_frequency must be finite and >= 0 Hz, got {target_frequency!r}')
        if pml_frequency is not None and not (math.isfinite(float(pml_frequency)) and float(pml_frequency) > 0.0):
            raise ValueError(f'membrane pml_frequency must be None or a finite value > 0 Hz, got {pml_frequency!r}')
        self.order = DISCRETIZATIONS[discretization]
        self.target_frequency = target_frequency
        self.pml_frequency = None if pml_frequency is None else float(pml_frequency)
        self.linear_solver = linear_solver   # 'linear' sparse_direct runtime (None: the system's private SuperLU)
        self._owns_eigen_solver = eigen_solver is None
        self.eigen_solver = ArpackShiftInvertSolver() if eigen_solver is None else eigen_solver
        self.prepare(domain)

    def _label(self):
        return f"membrane '{self.name}'"

    # --- PhysicsSolver contract ---------------------------------------------
    def prepare(self, domain) -> None:
        """Register the material arrays, allocate the per-element state and slots (clears conditions, loads and
        probes); the harmonic system is built at the first solve (its PML strength needs the reference frequency)."""
        self.domain = domain
        self.xp = domain.xp
        domain.materials.add_properties(self.material_schema)
        self.bending = BENDING_ARRAY in set(domain.materials.property_names())   # plate: a material sets D
        self.fields = {name: self.xp.zeros(self._field_shape(), dtype=DTYPE)
                       for name in STATE_FIELD_NAMES + COUPLED_TENSION_SLOTS}
        self.eigen_solve = None          # fem.harmonic_resonances.HarmonicResonanceSolve of the last eigen solve
        self.selected_resonance = None   # 1-based resonance shown, None = a driven solve (or nothing)
        self.response_frequency = None   # [Hz] of the shown driven solve
        self.probes = []
        self.system = None
        self._system_key = None
        self._pml_of = None   # (omega_reference, coefficients, mesh, PML stretch) of the last _ready_system
        self._coefficients = None
        self._coefficients_dirty = True
        self._coefficients_applied = False   # the system holds the current coefficients
        self._response_reference = None  # [Hz] PML reference of the running frequency response
        self.reduced_sweep = None        # info of the reduced sweep model (plan_frequency_response) or None
        self._field_policy = None        # fields of a sweep (fem.reduced_outputs.field_policy) or None = all
        self._unpublished_frequency = None   # [Hz] solved without publishing its fields (field policy)
        self.n_frequency_solves = 0
        self.assembly_seconds = 0.0      # host wall time of _ready_system (dof maps, PML, assembly, pencil)
        self._prepare_geometry()
        self.values = np.zeros(self._n_full(), dtype=np.complex128)   # full-numbering w of the shown solution
        self._device_values = None   # cupy copy of values when they are the system's last direct cuDSS solve (R6)

    def finalize(self) -> None:
        """Read the coefficients once: a negative or non-finite areal density, or a model of holes only, fails here."""
        self._read_coefficients()

    def materials_changed(self, names) -> None:
        if set(names) & COEFFICIENT_ARRAYS:
            self._coefficients_dirty = True

    def step(self, t: float = 0.0, dt: float = math.inf) -> None:
        """Nothing: the eigenfrequency and frequency_response studies call their hooks."""

    def dt_limit(self) -> float:
        return math.inf

    def state(self) -> dict:
        """Per element (1D: per cell) w_re / w_im / w_abs of the shown solution, the coupled slots and the materials."""
        state = dict(self.fields)
        materials = self.domain.materials
        for name in materials.property_names():
            if name in COEFFICIENT_ARRAYS:
                state[name] = getattr(materials, name)
        return state

    def inject_source(self, name: str, array) -> None:
        """Write a coupled tension slot [N/m]: one value per element (1D: per cell), added to the material tension."""
        if name not in self.source_slots:
            raise ValueError(f"{self._label()} accepts sources {', '.join(self.source_slots)}; got '{name}'")
        target = self.fields[name]
        values = self.xp.asarray(array, dtype=DTYPE)
        if values.size != target.size:
            raise ValueError(f"{self._label()}: source '{name}' needs {target.size} values (one per element / cell), "
                             f'got {values.size}')
        self.xp.copyto(target, values.reshape(target.shape))
        self._coefficients_dirty = True

    def set_conformal_boundaries(self, locator) -> None:
        """solver_defaults.conformal_boundaries: nothing to do."""

    # --- properties, counters ---------------------------------------------------
    @property
    def n_solves(self) -> int:
        return self.n_frequency_solves

    def work_counters(self) -> dict:
        """Driven solves, assemblies and assembly_seconds (system build wall time); the private eigen / linear
        runtimes' counters when no build runtime counts them (scripts)."""
        counters = {'frequency_solves': self.n_frequency_solves,
                    'assemblies': 0 if self.system is None else self.system.n_assemblies,
                    'assembly_seconds': self.assembly_seconds}
        if self._owns_eigen_solver and self.eigen_solver.n_eigen_solves:
            counters.update(self.eigen_solver.work_counters())
        if self.linear_solver is None and self.system is not None and self.n_frequency_solves:
            counters.update({f'private_{name}': value
                             for name, value in self.system.linear_solver.work_counters().items()
                             if name != 'back_end'})
        return counters

    def solve_summary(self):
        """One log line: unknowns, assemblies and driven solves (None before a system exists)."""
        system = self.system
        if system is None or system.pencil is None:
            return None
        pencil = system.pencil
        kind = 'plate (bending + tension)' if self.bending else 'membrane'
        return (f'host {self._element_text()} {kind}: {pencil.free.size:,} free + {pencil.fixed.size:,} clamped '
                f'unknowns, {system.n_assemblies} assemblies, {self.n_frequency_solves} driven solves')

    def _element_text(self) -> str:
        return f'P{self.order}'

    # --- coefficients and the system ------------------------------------------------
    def _host_array(self, array) -> np.ndarray:
        return self._element_values(np.asarray(to_host(array), dtype=np.float64))

    def _read_coefficients(self) -> MembraneCoefficients:
        """MembraneCoefficients of the materials + coupled slots (cached until a material or slot changes)."""
        if self._coefficients is not None and not self._coefficients_dirty:
            return self._coefficients
        materials = self.domain.materials
        present = set(materials.property_names())

        def material(name):
            if name not in present:
                return np.zeros(self._n_elements())
            return self._host_array(getattr(materials, name))

        tension = {}
        for name, slot in zip(TENSION_ARRAYS, COUPLED_TENSION_SLOTS):
            tension[name] = material(name) + self._host_array(self.fields[slot])
        loss, density = material(LOSS_ARRAY), material(DENSITY_ARRAY)
        bending, poisson = material(BENDING_ARRAY), material(POISSON_ARRAY)
        for name, values in (*tension.items(), (LOSS_ARRAY, loss), (DENSITY_ARRAY, density), (BENDING_ARRAY, bending),
                             (POISSON_ARRAY, poisson)):
            if not np.all(np.isfinite(values)):
                raise ValueError(f'{self._label()}: {name} is not finite in every element')
        if (density < 0.0).any():
            raise ValueError(f'{self._label()}: areal_density must be >= 0 in every element (min {density.min():.4g} '
                             'kg/m^2)')
        if (loss < 0.0).any():
            raise ValueError(f'{self._label()}: loss_factor must be >= 0 in every element (min {loss.min():.4g})')
        if (bending < 0.0).any():
            raise ValueError(f'{self._label()}: bending_stiffness must be >= 0 in every element (min '
                             f'{bending.min():.4g} N m)')
        if ((poisson <= -1.0) | (poisson > 0.5)).any():
            raise ValueError(f'{self._label()}: plate_poisson_ratio must lie in (-1, 0.5] in every element')
        hole = ((density <= 0.0) & (tension['tension_xx'] == 0.0) & (tension['tension_yy'] == 0.0)
                & (tension['tension_xy'] == 0.0) & (bending == 0.0))
        if hole.all():
            raise ValueError(f'{self._label()}: no element has tension or areal_density, so there is no membrane: '
                             'assign a material with tension and areal_density')
        self._coefficients = MembraneCoefficients(tension['tension_xx'], tension['tension_yy'],
                                                  tension['tension_xy'], loss, density, hole, bool(np.any(loss != 0.0)),
                                                  bending, poisson)
        self._coefficients_dirty = False
        self._coefficients_applied = False
        return self._coefficients

    def _ready_system(self, omega_reference):
        """The harmonic system for a PML reference angular frequency [rad/s] (rebuilt when the PML stretch changes),
        its coefficients brought up to date and assembled (timed: assembly_seconds)."""
        started = time.perf_counter()
        coefficients = self._read_coefficients()
        cached, mesh = self._pml_of, getattr(self.domain, 'mesh', None)   # a sweep keeps one reference: build once
        if cached is None or not (cached[0] == omega_reference and cached[1] is coefficients and cached[2] is mesh):
            cached = self._pml_of = (omega_reference, coefficients, mesh, self._pml(omega_reference, coefficients))
        pml = cached[3]
        key = None if pml is None else tuple((face, pml.widths[face], pml.strength[face]) for face in pml.faces)
        if self.system is None or key != self._system_key:
            self._release_factor()
            self.system = self._new_system(pml)
            self._system_key = key
            self._coefficients_applied = False
        if not self._coefficients_applied:
            self._apply_coefficients(self.system, coefficients)
            self._coefficients_applied = True
        self.system.ensure_assembled()   # here, not lazily inside the first solve: the build timer holds it
        self.assembly_seconds += time.perf_counter() - started
        return self.system

    def _discard_system(self) -> None:
        """A condition or load changed: the next solve rebuilds the system (features apply before any solve)."""
        self._release_factor()
        self.system = None
        self._system_key = None

    def _release_factor(self) -> None:
        """Free the driven factor of the system being replaced (a cuDSS factor lives outside the cupy pool; within one
        system the next frequency refactorises it in place: factorize_host(previous=), plan kept)."""
        factor = None if self.system is None else self.system.factor
        release = getattr(factor, 'release', None)
        if release is not None:
            release()

    # --- outer face API (the definition callbacks; _valid_faces: FACES, a 1D line LINE_FACES) ----------------------
    _valid_faces = FACES

    def _check_face(self, face) -> None:
        if face not in self._valid_faces:
            raise ValueError(f'{self._label()}: unknown face {face!r}; expected one of {self._valid_faces}')

    def clamp_face(self, face) -> None:
        """Hold w = 0 (with bending also the normal slope) on an outer face / line end (the window edge; with a PML on
        that face its outer side, held anyway)."""
        self._check_face(face)
        self._face_clamps.add(face)
        self._discard_system()

    def support_face(self, face) -> None:
        """Hold w = 0 on an outer face / line end, its slope free (simply supported; = clamp_face without bending)."""
        self._check_face(face)
        self._face_supports.add(face)
        self._discard_system()

    def symmetry_face(self, face) -> None:
        """Even mirror plane on an outer face / line end: with bending the normal slope held (2D: Nitsche) and w free;
        a film / string keeps its free edge."""
        self._check_face(face)
        self._face_symmetries.add(face)
        self._discard_system()

    def add_face_force(self, face, force) -> None:
        """Add a transverse force per length [N/m] (complex amplitude) along an outer face / at a line end (1D: per
        length along y); a held face drops it."""
        self._check_face(face)
        self._face_forces[face] = self._face_forces.get(face, 0.0) + complex(force)
        self._discard_system()

    # --- eigenfrequency study hooks -------------------------------------------
    def solve_eigenfrequencies(self, n_resonances):
        """Up to n_resonances resonances nearest target_frequency (fem.harmonic_resonances on K u = omega^2 M u,
        clamped dofs held at 0, PML strength set at pml_frequency or the target) -> HarmonicResonanceSolve."""
        self.eigen_solve = None
        reference = self.pml_frequency if self.pml_frequency is not None else self.target_frequency
        system = self._ready_system(2.0 * math.pi * reference)
        self.eigen_solve = solve_system_resonances(system, self.target_frequency, n_resonances,
                                                   eigen_solver=self.eigen_solver, label=RESONANCE_LABEL)
        self.selected_resonance = None
        return self.eigen_solve

    def solve_eigenfrequencies_targets(self, n_resonances, target_frequencies):
        """Union of the n_resonances resonances nearest each target [Hz] (fem.resonance_search; PML strength at
        pml_frequency, else the mean target), frequency order -> HarmonicResonanceSolve."""
        targets = [float(target) for target in target_frequencies]
        reference = sum(targets) / len(targets) if targets else self.target_frequency
        return self._multi_shift_solve(reference, lambda solve_at, count_below: solve_resonances_at_targets(
            solve_at, targets, n_resonances))

    def solve_eigenfrequencies_band(self, f_lo, f_hi, n_per_slice, max_resonances):
        """Every resonance with f_lo <= Re f < f_hi [Hz] (at most max_resonances; inertia-counted for a real pencil;
        PML strength at pml_frequency, else the band centre), frequency order -> HarmonicResonanceSolve."""
        return self._multi_shift_solve(0.5 * (float(f_lo) + float(f_hi)), lambda solve_at, count_below:
                                       solve_resonances_in_band(solve_at, f_lo, f_hi, n_per_slice, max_resonances,
                                                                count_below))

    def _multi_shift_solve(self, reference, search):
        """search(solve_at, count_below) on the system built for the PML reference [Hz] (pml_frequency wins) -> its
        HarmonicResonanceSolve, kept as eigen_solve."""
        self.eigen_solve = None
        reference = self.pml_frequency if self.pml_frequency is not None else reference
        system = self._ready_system(2.0 * math.pi * reference)
        solve_at, count_below = self._shift_solver(system)
        try:
            self.eigen_solve = search(solve_at, count_below)
        finally:
            if count_below is not None:   # the kept inertia factor (InertiaCounter)
                count_below.release()
        self.selected_resonance = None
        return self.eigen_solve

    def _shift_solver(self, system):
        """(solve_at, count_below) of the system's eigen pencil (pencil_shift_solver)."""
        eigen = system.eigen_pencil()
        return pencil_shift_solver(eigen.stiffness, eigen.mass, self.eigen_solver, in_window=eigen.in_window,
                                   energy_weights=eigen.energy_weights(), in_region=eigen.in_region,
                                   field_of=eigen.spread)

    def select_resonance(self, number):
        """Show resonance `number` (1-based): w scaled to 1 m real at its peak displacement dof (Hermite slope dofs
        do not count)."""
        resonance = self.eigen_solve.resonances[number - 1]
        values = np.array(resonance.field, dtype=np.complex128)
        displacement = values[self._displacement_dofs()]
        peak = displacement[int(np.argmax(np.abs(displacement)))] if displacement.size else 0.0
        self.values = values / peak if peak != 0.0 else values
        self._device_values = None
        self.selected_resonance = number
        self.response_frequency = None
        self._publish()

    # --- frequency_response study hooks -------------------------------------------
    def reset_frequency_response(self):
        """Forget the probe / port records, the PML reference and any reduced model of an earlier frequency_response run."""
        self._response_reference = None
        self.reduced_sweep = None
        self._field_policy, self._unpublished_frequency = None, None
        if self.system is not None:
            self.system.clear_reduced()
        for probe in self.probes:
            probe.record.clear()
        self._reset_records()

    def plan_frequency_response(self, frequencies, options):
        """ReducedSweepSolver hook (after reset_frequency_response): options {method, tolerance, moments, max_points,
        modal_count, fields}; 'direct' -> None (only the field policy, R6), else the system's reduced model over
        frequencies [Hz] -> its info dict."""
        from ...solvers.batched_direct import BatchedDirectSolver
        method = options.get('method', 'direct')
        batched = method == 'direct' and isinstance(self.linear_solver, BatchedDirectSolver)
        if method == 'direct' and not batched:
            self._field_policy = field_policy(options, frequencies)
            return None
        frequencies = np.asarray(frequencies, dtype=np.float64).ravel()
        if not (frequencies.size and np.all(np.isfinite(frequencies)) and np.all(frequencies > 0.0)):
            raise ValueError(f'{self._label()}: a reduced or batched frequency sweep needs frequencies > 0 Hz')
        if self._response_reference is None:   # as solve_frequency_response: the first frequency in solve order
            self._response_reference = self.pml_frequency if self.pml_frequency is not None else float(frequencies[0])
        system = self._ready_system(2.0 * math.pi * self._response_reference)
        if batched:   # solvers.batched_direct: the frequencies in batches (port terms per frequency at batch time)
            system.plan_batched(2.0 * math.pi * frequencies)
            self._field_policy = field_policy(options, frequencies)
            return None
        self.reduced_sweep = system.plan_reduced(
            2.0 * math.pi * frequencies, method, tolerance=options.get('tolerance', 1e-6),
            moments=options.get('moments', 4), max_points=options.get('max_points', 40),
            modal_count=options.get('modal_count', 20), eigen_solver=self.eigen_solver)
        self._field_policy = field_policy(options, frequencies)
        return self.reduced_sweep

    def solve_frequency_response(self, frequency):
        """Driven response at frequency [Hz] > 0 of every pressure / line / point force (complex amplitudes) with the
        loss factor: one complex factor of K + j omega C - omega^2 M on the 'linear' role (or the planned reduced model,
        probes / projections from L V); records the probes; publishes fields only where the field policy asks."""
        frequency = float(frequency)
        if not (math.isfinite(frequency) and frequency > 0.0):
            raise ValueError(f'{self._label()}: a frequency response needs a frequency > 0 Hz, got {frequency!r}')
        if self._response_reference is None:
            self._response_reference = self.pml_frequency if self.pml_frequency is not None else frequency
        system = self._ready_system(2.0 * math.pi * self._response_reference)
        fields = wants_fields(self._field_policy, frequency)
        system.solve(2.0 * math.pi * frequency, fields=fields)   # a direct solve writes system.values either way
        self.response_frequency = frequency
        self.selected_resonance = None
        self.n_frequency_solves += 1
        for probe in self.probes:
            probe.record.record(frequency, complex(system.linear_output(
                ('probe', probe.name), lambda values, probe=probe: self._probe_value(probe, values))))
        self._record_frequency(frequency)
        if fields:
            self.values = np.array(system.values, dtype=np.complex128)
            self._device_values = system.device_values()
            self._publish()
        self._unpublished_frequency = None if fields else frequency

    def publish_frequency_response(self):
        """ReducedSweepSolver hook: show the last solved frequency's fields when the field policy skipped them (direct
        values, or V y of a fields-free reduced solve)."""
        frequency, self._unpublished_frequency = self._unpublished_frequency, None
        system = self.system
        if frequency is None or system is None:
            return
        self.values = np.array(system.publish_fields(), dtype=np.complex128)
        self._device_values = system.device_values()   # None after a reduced publish (V y on the host)
        self._publish()

    # --- probes and state ---------------------------------------------------------
    def _new_probe(self, name, location) -> MembraneProbe:
        probe = MembraneProbe(str(name), location, HarmonicProbeRecord('w', unit='m', label=f'w at {name}'))
        self.probes.append(probe)
        return probe

    def _publish(self) -> None:
        """Per-element phasor state (w at the element centroid / cell centre) of the shown solution."""
        phasors = self._element_phasors()
        stacked = np.stack([self._field_array(values) for values in (phasors.real, phasors.imag, np.abs(phasors))])
        stacked = self.xp.asarray(stacked, dtype=DTYPE)   # one transfer for the three fields
        for name, values in zip(('w_re', 'w_im', 'w_abs'), stacked):
            self.xp.copyto(self.fields[name], values)

    def _display_values(self, component) -> np.ndarray:
        """Host per-element values of a non-phasor display field: total tension_xx / _yy / _xy, areal_density."""
        coefficients = self._read_coefficients()
        return {'tension_xx': coefficients.tension_xx, 'tension_yy': coefficients.tension_yy,
                'tension_xy': coefficients.tension_xy, 'areal_density': coefficients.density}[component]

    # --- subclass hooks -------------------------------------------------------------
    def _prepare_geometry(self) -> None:
        raise NotImplementedError

    def _field_shape(self) -> tuple:
        raise NotImplementedError

    def _n_elements(self) -> int:
        raise NotImplementedError

    def _n_full(self) -> int:
        raise NotImplementedError

    def _element_values(self, array) -> np.ndarray:
        """Host float64 (n_elements,) of a field-shaped host array."""
        raise NotImplementedError

    def _field_array(self, values) -> np.ndarray:
        """Field-shaped host array of per-element values (inverse of _element_values; 0 outside the window)."""
        raise NotImplementedError

    def _new_system(self, pml):
        raise NotImplementedError

    def _apply_coefficients(self, system, coefficients) -> None:
        raise NotImplementedError

    def _pml(self, omega_reference, coefficients):
        return None

    def _probe_value(self, probe, values) -> complex:
        """w of the probe for full-numbering values (linear in values)."""
        raise NotImplementedError

    def _element_phasors(self) -> np.ndarray:
        raise NotImplementedError

    def _displacement_dofs(self):
        """Index of the full-numbering dofs that are displacements (every one; Hermite lines: the w dofs)."""
        return slice(None)

    def _record_frequency(self, frequency) -> None:
        """After a driven solve at frequency [Hz]: record feature outputs beyond the probes (ports)."""

    def _reset_records(self) -> None:
        """Forget the records _record_frequency wrote."""


def complex_amplitude(amplitude, phase_degrees):
    """amplitude (number or host array) x exp(j phase) [deg] as complex128."""
    return np.asarray(amplitude, dtype=np.complex128) * complex(np.exp(1j * math.radians(float(phase_degrees))))


def pencil_shift_solver(stiffness, mass, eigen_solver, pencil='auto', **filters):
    """(solve_at(target [Hz], n) -> HarmonicResonanceSolve, count_below(omega^2) -> eigenvalues below (an
    InertiaCounter: the caller releases it), or None for a complex pencil) of a host K, M on the free unknowns
    (fem.resonance_search); filters as solve_harmonic_resonances."""
    def solve_at(target_frequency, n_resonances):
        return solve_harmonic_resonances(stiffness, mass, target_frequency, n_resonances, eigen_solver=eigen_solver,
                                         pencil=pencil, label=RESONANCE_LABEL, **filters)

    real = not (np.iscomplexobj(stiffness.data) or np.iscomplexobj(mass.data))
    return solve_at, (InertiaCounter(stiffness, mass, eigen_solver) if real else None)


class InertiaCounter:
    """count_below(omega^2) -> eigenvalues below of a real symmetric host K, M: one ShiftPencil of the eigen solver and
    its factor kept across counts (refactorised in place, a cuDSS plan reused, as mechanics.modal.ModalPencil);
    release() frees the factor (a cuDSS factor lives outside the cupy pool)."""

    def __init__(self, stiffness, mass, eigen_solver):
        self.stiffness, self.mass = stiffness.tocsr(), mass.tocsr()
        self.eigen_solver = eigen_solver
        self.pencil = None
        self.factor = None

    def __call__(self, eigenvalue):
        if self.pencil is None:
            self.pencil = self.eigen_solver.shift_pencil((self.stiffness,), self.mass, np.float64, symmetric=True)
        try:
            shift_inertia, self.factor = self.eigen_solver.count_below(self.stiffness, self.mass, eigenvalue,
                                                                       previous=self.factor, pencil=self.pencil)
        except Exception:
            self.release()
            raise
        return None if shift_inertia is None else shift_inertia.n_below

    def release(self) -> None:
        factor, self.factor = self.factor, None
        self.pencil = None
        release = getattr(factor, 'release', None)
        if release is not None:
            release()
