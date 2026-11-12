"""Static 3D solid mechanics on a tetrahedron mesh (the solid_mechanics_3d create branch on 'tetrahedra'): vector
Lagrange P2 (default) or P1 on the solid window elements (affine), isotropic E / nu or the 21 c_IJ (stiffness3d),
thermal strain, supports and
tractions on window faces or object interface faces, body forces; one factorisation per assembly
(system_fem3d.ElasticFEMSystem3D); vibration modes nearest a target for the eigenfrequency study (modal.solve_modes on
the 'eigen' role: SuperLU or cuDSS LDL^T of K - sigma M). Setting piezoelectric: a potential on the window elements,
electrodes, the static coupled solve (piezo_fem3d + piezo.PiezoSystem on the 'linear' runtime) and the short / open
circuit modes of the coupled pencil (the 'eigen' role). Frequency response (elastic or piezoelectric, with losses): one
complex symmetric factor per frequency by the 'linear' runtime (frequency_response.HarmonicSystem). Periodic axes (Bloch
/ Floquet unit cells, eigenfrequency only): the complex Hermitian pencil P^H K P of bloch3d, complex modes. PML faces
(eigenfrequency only, elastic): the mesh's shell stretched (pml3d), complex symmetric pencil, radiation Q. A free body
(target_frequency > 0 or a frequency response): rigid motions counted, their modes dropped, static solves refused.
Prestress (new_modules_plan K2): initial_stress materials load every static solve; stress_stiffening adds K_sigma of the
static state to every pencil. A mechanical_loss_factor damps the modes (complex, K3). Complex probes / reactions per
frequency (K4). Geometric nonlinearity (K6): total-Lagrangian Newton with load steps, reactions from the internal
force, pencils on its tangent. Strain / kinetic energy densities (K5). Squeeze-film gas damping per resonance after
each eigen solve (squeeze_film.SqueezeFilmDamping, S6w). Forces in N."""
import math
import time

import numpy as np

from ...core.backend import to_host
from ...core.constants import DTYPE, EPS0, FACES_3D
from ...core.damped_iteration import DampedIteration, DampingSettings
from ...core.harmonic_probe import HarmonicProbeRecord
from ...core.monitors import FieldProbe
from ...core.solver_base import PhysicsSolver
from ...fem.assembly import pattern_sum
from ...fem.dofs3d import lagrange_dofs_3d
from ...fem.elasticity import StressField
from ...fem.elasticity3d import (CENTROID_3D, COMPONENTS, TangentAssembler3D, assemble_body_load_3d,
                                 assemble_elastic_stiffness_3d, assemble_face_traction_3d,
                                 assemble_geometric_stiffness_3d, assemble_stress_load_3d, element_gradients_3d,
                                 element_strains_3d, initial_stress_force_3d,
                                 local_dof_points_3d, nodal_average_3d)
from ...fem.elements3d import lagrange_basis_3d
from ...fem.mesh3d import POOR_RADIUS_RATIO
from ...fem.reduced_outputs import field_policy, wants_fields
from ...fem.renderer3d import LagrangeSampleRenderer
from ...fem.sampling3d import centroid_values_3d, evaluate_lagrange_3d
from ...solvers.arpack_shift_invert import ArpackShiftInvertSolver
from .bloch3d import bloch_cell_3d, bloch_prolongation
from .fields3d import (ELECTRIC_OUTPUT_ARRAYS_3D, ENERGY_OUTPUT_ARRAYS_3D, HARMONIC_IMAG_FIELDS_3D, INPUT_ARRAYS_3D,
                       OUTPUT_ARRAYS_3D, SOURCE_SLOTS_3D)
from .frequency_response import FrequencyPoint, HarmonicSystem
from .materials import (DENSITY_SCHEMA, ELECTROMECHANICAL_ARRAYS_3D, EXPANSION_ARRAYS, LOSS_ARRAYS, MECHANICAL_SCHEMA,
                        PERMITTIVITY_TENSOR_ARRAYS_3D, PIEZO_DEFAULTS_3D, PIEZO_TENSOR_ARRAYS_3D, PRESTRESS_ARRAYS_3D,
                        PRESTRESS_NAMES, STIFFNESS_ARRAYS, STIFFNESS_TENSOR_ARRAYS_3D, lame_coefficients)
from .modal import ModalPencil, MultiShiftModes
from .nonlinear_start import NonlinearStart, solve_from_start
from .piezo import ELECTRODE_CONNECTIONS, RESISTIVE_LOAD, Electrode, PiezoSystem
from .piezo_fem3d import (block_layout, check_permittivity_3d, coupled_mass_3d, coupled_operator_3d,
                          permittivity_tensor_3d, piezo_tensor_3d, potential_active)
from .pml3d import DEFAULT_PML_REFLECTION, elastic_pml_shell_3d, ring_wave_speed
from .renderer3d import SolidRenderer3D
from .solver_common import MATERIAL_STATE_ARRAYS, FEMSolidCommon, host_array as _host
from .stiffness3d import check_positive_definite_3d, element_stiffness_3d, stress_3d, thermal_stress_3d
from .system_fem3d import ElasticFEMSystem3D
from .tangent_update import TangentPolicy, newton_proposer

_EXPANSION_NAMES = ('alpha_xx', 'alpha_yy', 'alpha_zz')
# MaterialGrid arrays the kept static state (prestress) depends on; another changed array (optics) keeps it
_PRESTRESS_INPUTS = STIFFNESS_ARRAYS | EXPANSION_ARRAYS | PRESTRESS_NAMES
# output groups of the shown solution, each computed on its first read (_ensure_outputs): displacement and potential
# (dof values, cheap), strain / stress (element points of the solid), E / D (element points of the window). An
# eigenfrequency study selects every mode for its snapshots, which mostly store u_abs / electric_potential only.
_PRIMARY_OUTPUTS = ('u_x', 'u_y', 'u_z', 'u_abs', 'electric_potential')
_ELECTRIC_POINT_OUTPUTS = tuple(name for name in ELECTRIC_OUTPUT_ARRAYS_3D if name not in _PRIMARY_OUTPUTS)
_OUTPUT_GROUPS = {'primary': _PRIMARY_OUTPUTS,
                  'mechanical': tuple(name for name in OUTPUT_ARRAYS_3D if name not in _PRIMARY_OUTPUTS),
                  'electric': _ELECTRIC_POINT_OUTPUTS,
                  'energy': ENERGY_OUTPUT_ARRAYS_3D}
# initial-stress force share on the free dofs above which sigma0 counts as not self-equilibrated with the supports
_UNBALANCED_INITIAL_STRESS = 1e-6
# Newton scale floor: the largest displacement counts at least this fraction of the mesh extent (a zero solution)
_NEWTON_SCALE_FLOOR = 1e-12


class _ElementFields:
    """Per-element float32 outputs (element-centroid values, 0 off the solid; electric ones 0 off the window) and the
    coupled / applied inputs on the domain's array module: the state() arrays (shape (n_elements,), D12)."""

    def __init__(self, n_elements, xp):
        self.shape = (int(n_elements),)
        for name in OUTPUT_ARRAYS_3D + ELECTRIC_OUTPUT_ARRAYS_3D + ENERGY_OUTPUT_ARRAYS_3D + INPUT_ARRAYS_3D:
            setattr(self, name, xp.zeros(self.shape, dtype=DTYPE))


class SolidMechanicsFEMPhysics3D(FEMSolidCommon, MultiShiftModes, PhysicsSolver):
    """3D static elasticity on a fem.domain3d.TetrahedronDomain (window elements; a PML shell is void unless pml_faces
    make it an elastic PML, eigenfrequency only): host float64
    assembly, state() arrays per element (centroid values), display fields by lagrange_field (P1 / P2 nodal:
    displacement and potential exact, strain / stress averaged at each dof over the solid elements around it, E / D over
    the window elements); both computed per _OUTPUT_GROUPS group on first read after a solve or mode selection."""

    _output_groups = _OUTPUT_GROUPS

    name = 'solid'
    material_schema = {**MECHANICAL_SCHEMA, **DENSITY_SCHEMA}
    output_arrays = OUTPUT_ARRAYS_3D + ELECTRIC_OUTPUT_ARRAYS_3D + ENERGY_OUTPUT_ARRAYS_3D
    input_arrays = INPUT_ARRAYS_3D
    source_slots = SOURCE_SLOTS_3D
    warm_start_extrapolate = True   # a sweep extrapolates warm_start_fields over the last two values (model.study)

    def __init__(self, domain, order: int = 2, linear_solver=None, eigen_solver=None, target_frequency: float = 0.0,
                 piezoelectric: bool = False, periodic=(False, False, False), bloch_k=(0.0, 0.0, 0.0), pml_faces=(),
                 pml_reflection: float = DEFAULT_PML_REFLECTION, stress_stiffening: bool = False, free_body=None,
                 geometric_nonlinearity: bool = False, load_steps: int = 1, nonlinear_tolerance: float = 1e-8,
                 nonlinear_max_iterations: int = 25, tangent_update: str = 'when_slow'):
        if order not in (1, 2):
            raise ValueError(f'order must be 1 (linear) or 2 (quadratic), got {order!r}')
        if stress_stiffening and piezoelectric:
            raise ValueError('stress_stiffening is not available with piezoelectric on')
        if geometric_nonlinearity and piezoelectric:
            raise ValueError('geometric_nonlinearity is not available with piezoelectric on')
        if isinstance(load_steps, bool) or int(load_steps) != load_steps or load_steps < 1:
            raise ValueError(f'load_steps must be an integer >= 1, got {load_steps!r}')
        # total-Lagrangian Newton of every static solve (new_modules_plan K6): load steps and the damped loop
        self.geometric_nonlinearity = bool(geometric_nonlinearity)
        self.load_steps = int(load_steps)
        self._newton = (DampedIteration(DampingSettings(tolerance=float(nonlinear_tolerance),
                                                        max_iterations=int(nonlinear_max_iterations)),
                                        lambda: f'{self._label()} geometrically nonlinear solve')
                        if self.geometric_nonlinearity else None)
        # tangent refactorised per Newton iteration or kept while the residual falls fast (plan G4, modified Newton)
        self._tangent_policy = TangentPolicy(tangent_update)
        # (converged every load step, warning or None, Newton iterations, start origin text) of the last nonlinear solve
        self._nonlinear_record = None
        # K_sigma of the static state in every pencil (new_modules_plan K2); free rigid motions allowed (None: target > 0)
        self.stress_stiffening = bool(stress_stiffening)
        self._free_body = None if free_body is None else bool(free_body)
        pml_faces = tuple(pml_faces)
        unknown = [face for face in pml_faces if face not in FACES_3D]
        if unknown:
            raise ValueError(f'pml_faces must be outer faces {FACES_3D}, got {unknown}')
        if pml_faces and (piezoelectric or any(periodic)):
            raise ValueError('an elastic PML shell runs with piezoelectric off and no periodic axis')
        # elastic PML shell (pml3d): its faces and target reflection; the eigenfrequency study only
        self.pml_faces = pml_faces
        self.pml_reflection = float(pml_reflection)
        target_frequency = float(target_frequency)
        if not (math.isfinite(target_frequency) and target_frequency >= 0.0):
            raise ValueError(f'target_frequency must be finite and >= 0 [Hz], got {target_frequency!r}')
        bloch_k = tuple(float(value) for value in bloch_k)
        if len(bloch_k) != 3 or not all(math.isfinite(value) for value in bloch_k):
            raise ValueError(f'bloch_k must be three finite wavenumbers [rad/m], got {bloch_k!r}')
        self.order = int(order)
        self.linear_solver = linear_solver
        self.target_frequency = target_frequency
        # Bloch / Floquet axes (x, y, z) and their wavenumbers: the eigenfrequency study on P^H K P (bloch3d)
        self.periodic = tuple(bool(value) for value in periodic)
        self.bloch_k = bloch_k
        # potential + electrodes, the static coupled solve on the 'linear' runtime (piezo_fem3d, piezo.PiezoSystem)
        self.piezoelectric = bool(piezoelectric)
        self._piezo_counters = {'piezo_solves': 0}
        self._owns_eigen_solver = eigen_solver is None
        self.eigen_solver = ArpackShiftInvertSolver() if eigen_solver is None else eigen_solver
        self.eigen_solve = None          # modal.ModalSolve of the last solve_eigenfrequencies
        self.modal_mass = None           # host CSR consistent mass of that solve
        self.optomechanical_couplings = []   # optomechanics.OptomechanicalCoupling of the features (optomechanics study)
        self.selected_resonance = None   # 1-based mode shown, None = the static solve
        self.frequency_points = []       # frequency_response.FrequencyPoint per solved frequency, in solve order
        self.finalize_warnings = []
        self.finalize_notes = []         # info lines the builder logs: the assembly time by part
        self._assembly_seconds = {}      # part -> wall s of the last _assemble
        self.fields = None
        self.system = None
        self.prepare(domain)

    def _label(self):
        return f"solid mechanics 3D '{self.name}'"

    # --- PhysicsSolver contract ---------------------------------------------

    def prepare(self, domain) -> None:
        """Register the material arrays, allocate the system and the per-element fields (clears supports, loads and
        probes); assembly is deferred."""
        if getattr(domain, 'grid_kind', 'tensor') != 'tetrahedra':
            raise ValueError(f'{self._label()}: SolidMechanicsFEMPhysics3D needs a tetrahedron mesh (fem '
                             'TetrahedronDomain); a tensor grid uses SolidMechanics3DPhysics')
        self.domain = domain
        domain.materials.add_properties(self.material_schema)
        if self.piezoelectric:
            domain.materials.add_properties(PIEZO_DEFAULTS_3D)   # no-op for the arrays a material allocated
        self.system = ElasticFEMSystem3D(domain.mesh, self.order, self._label, self.linear_solver,
                                         backend=domain.backend.name)
        if any(self.periodic):
            try:
                self.system.bloch = bloch_cell_3d(self.system.full, self.periodic, self.bloch_k)
            except ValueError as exc:
                raise ValueError(f'{self._label()}: {exc}') from None
        # modes about a target > 0 or a frequency response (free body, free Bloch cell): free rigid motions counted,
        # static solves refused (the create passes free_body by the study; scripts: target > 0)
        self.system.rigid_motions_allowed = (self.target_frequency > 0.0 if self._free_body is None
                                             else self._free_body)
        n_elements = domain.n_elements
        self.fields = _ElementFields(n_elements, domain.xp)
        self._window_elements = np.asarray(domain.window_elements(), dtype=bool)
        # elements that may carry stiffness: the window, plus the PML shell (materials extruded into it)
        self._elastic_elements = np.ones(n_elements, dtype=bool) if self.pml_faces else self._window_elements
        for face in self.pml_faces:   # the shell's outer side is clamped
            self.system.hold(self.system.full.face_dofs(np.asarray(domain.mesh.face_tags[face], dtype=np.int64)),
                             0.0, 0.0, 0.0)
        self._element_map = lagrange_dofs_3d(domain.mesh, 0)   # P0: inputs, materials, field_dofs
        self._body_force = np.zeros((n_elements, COMPONENTS))   # applied [N/m^3] per element
        self._tractions = []                                     # [(mesh faces, (t_x, t_y, t_z))]
        self._stiffness = None                                   # host (n_elements, 6, 6) of the last assembly
        self._drop_host_inputs()
        self._solid = None
        self._nodal = {}                                         # output name -> host float64 full-mesh nodal values
        self._pending_outputs = set()                            # _OUTPUT_GROUPS of the shown solution not computed
        self._point_cache = {}                                   # (imaginary, points) -> _point_values of the solid
        self._output_thermal = True                              # their stress includes -beta dT
        self._complex_displacement = None                        # host (n_dofs, 3): u_abs = |u| of a complex solution
        self._shows_mode = False
        self._shows_harmonic = False                             # a frequency-response solution is shown
        self._shows_complex = False                              # the shown solution is complex (harmonic, Bloch)
        self._harmonic_operators = None                          # host (mass, loss or None) of the frequency response
        self._harmonic = None                                    # elastic frequency_response.HarmonicSystem
        self.reduced_sweep = None                                # info of plan_frequency_response, or None
        self._field_policy = None                                # fem.reduced_outputs field policy of that plan
        self._unpublished_frequency = None                       # [Hz] solved without showing its fields (field policy)
        self._unpublished_solution = None   # its piezoelectric direct solution (elastic: HarmonicSystem.last_solution)
        self._loads_dirty = True
        self.probes = []                                         # (FieldProbe, element, reference)
        self._electrodes = []                                    # piezo.Electrode of the electrode features
        self._piezo_system = None                                # piezo.PiezoSystem of the last assembly
        self._piezo_tensor = None                                # host (n_elements, 3, 6) e of that assembly
        self._permittivity = None                                # host (n_elements, 3, 3) relative eps of it
        self._potential = None                                   # host scalar-map phi [V] of the last static solve
        self._piezo_reaction = None                              # host (n_dofs, 3) u rows of A x - f [N]
        self._electrode_values = {}                              # {electrode name: (V, Q [C])} of that solve
        self._output_prestress = True                            # the shown stress includes the initial stress
        self._output_green_lagrange = False                      # shown strains are Green-Lagrange (nonlinear static)
        self._output_omega = None                                # rad/s of a shown mode / response (None: static)
        # (with_loads, K_sigma host CSR or None, text, the nonlinear tangent K_T host CSR or None)
        self._prestress = None
        self._pencil_factor = None   # the static K_ff factor solve_static_state kept for the modal pencil (owned here)
        self.frequency_probes = []                               # (HarmonicProbeRecord, element, reference)
        self._supports = []                                      # SupportRegion of the support features
        self._harmonic_reactions = {}                            # support name -> HarmonicProbeRecord (F_x, F_y, F_z)
        self._record_rows = None                                 # (key, probe weights, support rows): _record_layout
        self.squeeze_films = []                                  # squeeze_film.SqueezeFilmDamping, evaluated per mode
        self._nonlinear_start = NonlinearStart()                 # Newton start: sweep seed, last converged state
        self._tangent_assembler = None                           # TangentAssembler3D of the last assembly (lazy)

    def finalize(self) -> None:
        """Assemble once (fails early on a solid region that can move as a rigid body); a warning on sliver elements
        and on an initial stress that is not self-equilibrated where no static solve relaxes it (a note elsewhere)."""
        self._assemble()
        seconds = self._assembly_seconds
        self.finalize_notes = [f"assembly {sum(seconds.values()):.3g} s: "
                               + ', '.join(f'{part} {value:.2f} s' for part, value in seconds.items())]
        if self.system.n_rigid_motions:
            self.finalize_notes.append(f'{self.system.n_rigid_motions} free rigid motion(s) (free body): the '
                                       'eigenfrequency study drops their 0 Hz modes; static solves are refused')
        quality = self.domain.mesh.quality
        self.finalize_warnings = [] if not quality.n_poor else [
            f'{quality.n_poor:,} of {self.domain.n_elements:,} tetrahedra have a radius ratio below '
            f'{POOR_RADIUS_RATIO:g} and an obtuse dihedral angle (smallest ratio {quality.smallest:.3g}): such '
            'slivers stiffen the elements and worsen the conditioning; check thin gaps in the geometry or set h_min']
        share = self.initial_stress_imbalance()
        if share is not None and share > _UNBALANCED_INITIAL_STRESS:
            refused = self._static_refused()
            if refused and self.prestressed:
                self.finalize_warnings.append(
                    f'the initial stress is not self-equilibrated with the supports (largest unbalanced nodal force '
                    f'{share:.3g} of the largest initial-stress nodal force: free surfaces, material jumps or a '
                    f'varying sigma0 load the solid), and {refused} has no static solve to relax it: K_sigma of the '
                    'unrelaxed sigma0 is not that of an equilibrium state. Balance it (supports or boundary loads on '
                    'the stressed faces, a uniform sigma0 across the cell) or solve the relaxed state in a model that '
                    'has a static solve')
            else:
                self.finalize_notes.append(f'initial stress: not self-equilibrated with the supports (largest '
                                           f'unbalanced nodal force {share:.3g} of the largest initial-stress nodal '
                                           'force); a static solve relaxes it (total stress sigma0 + C eps)')

    def initial_stress_imbalance(self):
        """Largest |initial-stress force| on the free dofs (folded onto the Bloch masters at phase 1) over its largest
        |value| anywhere: ~0 when sigma0 is self-equilibrated with the supports; None without an initial stress (call
        after an assembly)."""
        initial = self._initial_stress()
        if initial is None:
            return None
        system = self.system
        force = initial_stress_force_3d(system.full, initial, elements=self._solid)
        largest = float(np.abs(force).max())
        if not largest > 0.0:
            return None
        if system.bloch is not None:   # the static folding: every hi-side dof's force joins its lo-side master
            master, _ = system.bloch.vector_masters()
            _, prolongation = bloch_prolongation(master, np.ones(master.size), system.free)
            unbalanced = prolongation.T @ force
        else:
            unbalanced = force[system.free]
        return float(np.abs(unbalanced).max()) / largest if unbalanced.size else 0.0

    def materials_changed(self, names) -> None:
        names = set(names)
        self._ensure_outputs()   # pending outputs belong to the inputs of their solve (alpha, rho, sigma0)
        self._drop_host_inputs()
        if STIFFNESS_ARRAYS & names:
            self.system.operator_dirty = True
            self._loads_dirty = True   # the thermal load scales with the stiffness
        if EXPANSION_ARRAYS & names:
            self._loads_dirty = True
        if self.piezoelectric and ELECTROMECHANICAL_ARRAYS_3D & names:
            self.system.operator_dirty = True
        if PRESTRESS_NAMES & names:
            self._loads_dirty = True
        if {*LOSS_ARRAYS, *DENSITY_SCHEMA} & names:
            self._harmonic_operators = None
            self._drop_harmonic()
        if names & _PRESTRESS_INPUTS or (self.piezoelectric and ELECTROMECHANICAL_ARRAYS_3D & names):
            self._drop_prestress()   # the static state and K_sigma (optics or other arrays leave them)

    def step(self, t: float = 0.0, dt: float = math.inf) -> None:
        """Re-solve only when the stiffness, a support, a load or an input changed (always at the first call); probes
        sample after a re-solve and at every dt = inf call."""
        t = 0.0 if t is None else float(t)
        system = self.system
        re_solve = system.operator_dirty or self._loads_dirty or system.n_solves == 0
        if re_solve:
            self._static_solve()
        if re_solve or dt is None or not math.isfinite(dt):
            self._record_probes(t)

    def _static_refused(self) -> str:
        """Why the solid has no static solve ('' when it has one): periodic axes, an elastic PML or free rigid
        motions (call after an assembly)."""
        if self.system.bloch is not None:
            return 'a Bloch cell (periodic axes)'
        if self.pml_faces:
            return 'an elastic PML'
        if self.system.n_rigid_motions:
            return f'a free body ({self.system.n_rigid_motions} free rigid motions)'
        return ''

    def _static_solve(self):
        """The static solve of every load, prescribed displacement, initial stress and input (assembling first when
        dirty); ValueError for a Bloch cell, an elastic PML or a free body."""
        system = self.system
        if system.bloch is not None or self.pml_faces:
            what = 'periodic axes (Bloch cells)' if system.bloch is not None else 'an elastic PML'
            raise ValueError(f'{self._label()}: {what} is solved by the eigenfrequency study only, not by static solves')
        if system.operator_dirty:
            self._assemble()
        self._refuse_free_body('a static solve')
        if self.piezoelectric:
            self._solve_piezo(self._total_load())
        elif self.geometric_nonlinearity:
            displacement, residual = self._nonlinear_state(with_loads=True)
            system.accept_solution(displacement, self._external_load(), residual=residual)
        else:
            system.solve(self._total_load())
        self._shows_mode = self._shows_harmonic = self._shows_complex = False
        self.selected_resonance = None
        self._write_outputs(green_lagrange=self.geometric_nonlinearity)
        self._loads_dirty = False

    # --- geometric nonlinearity (total-Lagrangian Newton, new_modules_plan K6) ----

    def _fixed_stress(self, thermal=True):
        """fem.elasticity.StressField of the stress that is no strain response: sigma0 (initial stress) - beta dT
        (thermal); None when both are 0."""
        initial = self._initial_stress()
        thermal_stress = None
        if thermal:
            values = np.where(self._solid[:, None], self._thermal_stress() * self._temperature_change()[:, None], 0.0)
            thermal_stress = values if values.any() else None
        if initial is None and thermal_stress is None:
            return None
        return StressField(prescribed=initial, thermal=thermal_stress)

    def _nonlinear_state(self, with_loads=True):
        """(displacement (n_dofs, 3), residual f_int - f_ext (3 n_dofs,)) of the geometrically nonlinear static state:
        Newton on f_int(u) = int (B + B_NL)^T S dV = f_ext (S = C E(u) + sigma0 - beta dT) over load_steps increments of
        the dead loads and held values (with_loads False: no loads, holds at 0, sigma0 - beta dT alone); each step the
        damped loop of core.damped_iteration on the fixed pattern of TangentAssembler3D (tangent_update: a factorisation
        per iteration, or kept while the residual falls fast; tangent_update.newton_proposer). Start (nonlinear_start):
        a sweep seed or the last converged state of this assembly, ramped from its load and held values (the cold ramp
        from u = 0 when it fails), else u = 0."""
        system = self.system
        n_vector = COMPONENTS * system.full.n_dofs
        fixed_stress = self._fixed_stress(thermal=True)   # sigma0 and the thermal strain are static in both kinds
        external = self._external_load() if with_loads else np.zeros(n_vector)
        target = system.fixed_values() if with_loads else np.zeros(n_vector)
        held = (system.held & system.active[:, None]).reshape(-1)
        assembler = self._tangent_assembler_3d()
        window = np.asarray(self.domain.mesh.window, dtype=np.float64)
        floor = _NEWTON_SCALE_FLOOR * float(np.max(window[3:] - window[:3]))
        scale = np.array(floor)   # max |proposal| of the latest Newton step: updated in place by propose
        latest = {}               # 'state', 'force': the displacement of the latest internal force and that force
        policy = self._tangent_policy
        policy.reset()            # the runtime may have factorised K or a pencil since the last Newton solve
        propose_for = newton_proposer(policy, assembler, system, fixed_stress, held, system.free, scale, floor, latest)
        state, self._nonlinear_record = solve_from_start(self._nonlinear_start, with_loads, self._newton,
                                                         self.load_steps, propose_for, scale, external, target,
                                                         on_cold_ramp=policy.reset,
                                                         active=np.repeat(system.active, COMPONENTS))
        policy.reset()            # release the held tangent values
        if latest and np.array_equal(latest['state'], state):
            force = latest['force']
        else:
            force = assembler.force(state, initial_stress=fixed_stress)
        return state.reshape(-1, COMPONENTS), force - external

    def _tangent_assembler_3d(self):
        """fem.elasticity3d.TangentAssembler3D of the last assembly (solid elements, free set), built on first use; on
        the GPU with a cuDSS 'linear' back end a device assembler (plan G5: tangents assembled and refactorised there)."""
        if self._tangent_assembler is None:
            self._tangent_assembler = TangentAssembler3D(self.system.full, self._stiffness, elements=self._solid,
                                                         free=self.system.free, device=self.system.device_tangent)
        return self._tangent_assembler

    def _refuse_free_body(self, what):
        """ValueError for `what` (a static solve) while the last assembly left rigid motions free (allowed for the
        vibration modes about a target > 0 and the frequency response at f > 0 only)."""
        count = self.system.n_rigid_motions
        if count:
            raise ValueError(f'{self._label()}: {count} rigid motion(s) of the solid are free (supports missing; '
                             'allowed only for the vibration modes of an eigenfrequency study with target_frequency > '
                             f'0, which drops their 0 Hz modes, and for a frequency response), so {what} is not '
                             'available: add Fixed, Roller or Displacement supports or a Fixed (domain) region that '
                             'stop every rigid motion')

    # --- prestress (stress stiffening, new_modules_plan K2) -------------------------

    def solve_static_state(self, with_loads=True) -> str:
        """PrestressSolver hook: the static state whose K_sigma (int grad N . sigma grad N dV x I3 of its total stress)
        joins every later pencil until the next assembly / material / input change; with_loads: every load (the step's
        static solve, shown afterwards), else the initial stress and thermal strain alone with every hold at 0 (a
        frequency response, whose loads are its drive; dT is static, never a drive). A Bloch cell, an elastic PML or a
        free body: the initial stress alone, unrelaxed. Geometric nonlinearity: the pencil stiffness is the tangent K_T
        of the nonlinear static state instead."""
        system = self.system
        if system.operator_dirty:
            self._assemble()
        initial = self._initial_stress()
        refused = self._static_refused()
        solid = self._solid
        tangent = None
        if refused:
            stress = None if initial is None else StressField(prescribed=initial)
            text = (f'stress stiffening: {refused} has no static solve, K_sigma of the initial stress alone'
                    if initial is not None else f'stress stiffening: {refused} and no initial stress, K_sigma = 0')
        elif with_loads and self.geometric_nonlinearity:
            self._static_solve()
            stress = None
            tangent = self._tangent_assembler_3d().tangent(system.unknown,
                                                           initial_stress=self._fixed_stress(thermal=True))
            text = ('geometric nonlinearity: the pencil stiffness is the tangent K_T of the nonlinear static state '
                    f'(every load, initial stress and thermal strain; {self._nonlinear_text()})')
        elif with_loads:
            # a K_ff factor the modal pencil refactorises takes the pencil's kind (LDL^T, not the SPD request)
            system.linear_solver.positive_definite = not self._pencil_handover()
            try:
                self._static_solve()
            finally:
                system.linear_solver.positive_definite = True
            stress = StressField(prescribed=initial, material=self._stiffness, displacement=system.unknown.copy(),
                                 thermal=np.where(solid[:, None], self._thermal_stress()
                                                  * self._temperature_change()[:, None], 0.0))
            text = 'stress stiffening: K_sigma of the static state (every load, initial stress and thermal strain)'
        elif (background := self._fixed_stress(thermal=True)) is None:
            stress = None
            text = ('stress stiffening: no initial stress or thermal strain (the loads are the harmonic drive), '
                    'K_sigma = 0')
        elif self.geometric_nonlinearity:
            displacement, _ = self._nonlinear_state(with_loads=False)
            stress = None
            tangent = self._tangent_assembler_3d().tangent(displacement, initial_stress=background)
            text = ('geometric nonlinearity: the pencil stiffness is the tangent K_T of the nonlinear state of the '
                    f'initial stress and thermal strain (holds at 0, loads are the drive; {self._nonlinear_text()})')
        else:
            relaxed = system.solve_homogeneous(self._total_load(dead_loads=False))
            stress = StressField(prescribed=background.prescribed, material=self._stiffness, displacement=relaxed,
                                 thermal=background.thermal)
            text = ('stress stiffening: K_sigma of the relaxed initial stress and thermal strain (holds at 0, loads '
                    'are the drive)')
        geometric = None if stress is None else self._geometric_stiffness(stress)
        self._prestress = (bool(with_loads), geometric, text, tangent)
        self._drop_harmonic()
        if not refused:   # the pencil / harmonic factorisations follow: free the static factor (cuDSS: device memory)
            if with_loads and tangent is None and self._pencil_handover():
                # the modal pencil's first shift factor refactorises it on the kept analysis / plan (same free pattern:
                # K + K_sigma on K's kept pattern, M inside it); released with the pencil, or by _drop_harmonic
                self._pencil_factor = system.hand_over_factor()
            else:
                system.release_factor()
        return text

    def _pencil_handover(self) -> bool:
        """True when the static K_ff factor can be the real symmetric modal pencil's first shift factor: the 'linear'
        back end (cuDSS or MUMPS / PARDISO) is the eigen factorization, no piezoelectric pencil and no loss (complex)."""
        from ...solvers.sparse_direct import CUDSS_BACK_ENDS, HOST_DIRECT_BACK_ENDS
        back_end = self.system.runtime.back_end
        return (back_end in CUDSS_BACK_ENDS + HOST_DIRECT_BACK_ENDS
                and getattr(self.eigen_solver, 'factorization', None) == back_end and not self.piezoelectric
                and not self._loss_values(LOSS_ARRAYS[0], self._solid).any())

    def _take_pencil_factor(self):
        factor, self._pencil_factor = self._pencil_factor, None
        return factor

    def warm_start_fields(self) -> dict:
        """WarmStartSolver: host copies of the last converged nonlinear state (u (n_dofs, 3), its external load and held
        values, flat) and its kind 'with_loads'; {} without one (a linear solid's direct solve needs no start)."""
        last = self._nonlinear_start.last
        if last is None:
            return {}
        with_loads, state, external, held = last
        return {'u': state.reshape(-1, COMPONENTS).copy(), 'external': external.copy(), 'held': held.copy(),
                'with_loads': with_loads}

    def seed_warm_start(self, fields) -> tuple:
        """WarmStartSolver: warm_start_fields() of another build start the next nonlinear solve of their kind (sizes
        checked there, another size ignored with a note); no state slot is seeded."""
        if fields and self.geometric_nonlinearity:
            self._nonlinear_start.set_seed(fields.get('with_loads', True), fields['u'], fields['external'],
                                           fields['held'])
        return ()

    def _geometric_stiffness(self, stress):
        """Host CSR K_sigma of a fem.elasticity.StressField over the solid elements (an elastic PML: the shell's part
        stretched as its K)."""
        system = self.system
        if system.pml is None:
            return assemble_geometric_stiffness_3d(system.full, stress, elements=self._solid).tocsr()
        window = self.domain.mesh.window_elements()
        return pattern_sum([
            assemble_geometric_stiffness_3d(system.full, stress, elements=self._solid & window),
            assemble_geometric_stiffness_3d(system.full, stress, elements=self._solid & ~window,
                                            stretch=system.pml.stretches)])

    def _drop_prestress(self):
        self._prestress = None
        self._drop_harmonic()

    def _drop_harmonic(self):
        """Drop the elastic HarmonicSystem and a static factor kept for the modal pencil, releasing their factors first
        (a cuDSS / MUMPS / PARDISO factor lives outside the cupy pool / numpy)."""
        getattr(self._take_pencil_factor(), 'release', lambda: None)()
        if self._harmonic is not None and self._harmonic.factor is not None:
            getattr(self._harmonic.factor, 'release', lambda: None)()
        self._harmonic = None

    def _pencil_stiffness(self, with_loads):
        """Host K of the pencils: the system operator, + K_sigma of the static state (stress_stiffening; solved here
        when none is kept for this study kind); geometric nonlinearity: the tangent K_T of the nonlinear state."""
        if not self.prestressed:
            return self.system.operator
        if self._prestress is None or self._prestress[0] != with_loads:
            self.solve_static_state(with_loads)
        _, geometric, _, tangent = self._prestress
        if tangent is not None:
            return tangent
        return self.system.operator if geometric is None else pattern_sum([self.system.operator, geometric])

    def _initial_stress(self):
        """Host (n_elements, 6) initial stress [Pa] in Voigt rows on the solid elements (0 elsewhere; cached,
        read-only); None when no material sets one."""
        return self._host_input('initial_stress', self._read_initial_stress)

    def _read_initial_stress(self):
        materials = self.domain.materials
        names = set(materials.property_names())
        if not names.intersection(PRESTRESS_ARRAYS_3D):
            return None
        stress = np.zeros((self.domain.n_elements, len(PRESTRESS_ARRAYS_3D)))
        for row, name in enumerate(PRESTRESS_ARRAYS_3D):
            if name in names:
                stress[:, row] = np.where(self._solid, _host(getattr(materials, name)), 0.0)
        return stress if stress.any() else None

    def inject_source(self, name: str, array) -> None:
        """Write a coupled slot: one value per element (this mesh's own: same-mesh couplings)."""
        if name not in self.source_slots:
            raise ValueError(f"{self._label()} accepts sources {', '.join(self.source_slots)}; got '{name}'")
        xp = self.domain.xp
        values = xp.asarray(array, dtype=DTYPE).reshape(-1)
        target = getattr(self.fields, name)
        if values.size != target.size:
            raise ValueError(f"{self._label()}: source '{name}' needs one value per element ({target.size}), got "
                             f'{values.size}')
        self._ensure_outputs()   # pending outputs belong to the inputs of their solve (the thermal stress)
        self._drop_host_inputs()
        xp.copyto(target, values)
        self._loads_dirty = True
        self._drop_prestress()

    # --- records ---------------------------------------------------------------

    def work_counters(self) -> dict:
        """The private eigen solver's counters once it ran; the private linear runtime's when no build runtime counts
        them (scripts); piezoelectric: the coupled back substitutions (their factorisations count on the runtime)."""
        counters = {}
        if self._owns_eigen_solver and self.eigen_solver.n_eigen_solves:
            counters.update(self.eigen_solver.work_counters())
        if self.linear_solver is None and self.system.runtime.n_factorizations:
            counters.update({f'fem_{name}': value for name, value in self.system.linear_solver.work_counters().items()
                             if name != 'back_end'})
        if self.piezoelectric:
            counters.update(self._piezo_counters)
        if self._newton is not None and self._newton.n_runs:   # load steps run, Newton iterations, tangent factors
            counters.update(self._newton.work_counters())
            counters.update(self._tangent_policy.counters())
        return counters

    def solve_summary(self) -> str:
        summary = self.system.solve_summary()
        if self.geometric_nonlinearity and summary:
            summary += f'; geometrically nonlinear: {self._nonlinear_text()}'
        if self.piezoelectric and self._piezo_system is not None and summary:
            summary += (f'; piezoelectric: {int(self._piezo_system.free.sum()):,} free coupled dofs (the factor above is '
                        'the coupled one)')
        if self.system.bloch is not None and summary:
            summary += f'; Bloch cell ({self.system.bloch.text()})'
        if self.system.pml is not None and summary:
            summary += f'; elastic PML ({self.system.pml.text()})'
        return summary

    def renderer(self):
        return LagrangeSampleRenderer(self, 'solid', SolidRenderer3D.field_infos, self._time_label)

    def _time_label(self):
        return f'solve {self.n_solves}'

    # --- supports, loads, probes ----------------------------------------------

    def hold_face(self, face: str, u_x=None, u_y=None, u_z=None):
        """Hold components [m] (None = free) on every dof of an outer face of the window; returns the held dof mask."""
        if face not in FACES_3D:
            raise ValueError(f'unknown face {face!r}; expected one of {FACES_3D}')
        return self.hold_faces(self.system.face_faces(face), u_x, u_y, u_z)

    def hold_faces(self, faces, u_x=None, u_y=None, u_z=None):
        """Hold components [m] (None = free) on every dof of mesh faces (an object's interface faces); returns the held
        dof mask."""
        system = self.system
        return system.hold(system.full.face_dofs(np.asarray(faces, dtype=np.int64)), u_x, u_y, u_z)

    def hold_cells(self, mask, u_x=0.0, u_y=0.0, u_z=0.0):
        """Hold components [m] (None = free) on every dof of the window elements under the element mask."""
        system = self.system
        return system.hold(system.full.element_dofs_of(self._element_mask(mask, 'fixed domain')), u_x, u_y, u_z)

    def add_face_traction(self, face: str, t_x: float, t_y: float, t_z: float) -> None:
        """Add a traction (t_x, t_y, t_z) [Pa] on an outer face of the window (consistent face loads)."""
        if face not in FACES_3D:
            raise ValueError(f'unknown face {face!r}; expected one of {FACES_3D}')
        self.add_surface_traction(self.system.face_faces(face), t_x, t_y, t_z)

    def add_surface_traction(self, faces, t_x: float, t_y: float, t_z: float) -> None:
        """Add a traction [Pa] on mesh faces (window faces or an object's interface faces): int t N dA per dof, so the
        total is t times the area of the faces beside a solid element (others add nothing). An entry on the same faces
        takes the sum (a reused sweep build re-adds changes: the list never grows per value)."""
        faces = np.asarray(faces, dtype=np.int64)
        traction = (float(t_x), float(t_y), float(t_z))
        for index, (entry_faces, entry_traction) in enumerate(self._tractions):
            if entry_faces is faces or np.array_equal(entry_faces, faces):
                self._tractions[index] = (entry_faces, tuple(a + b for a, b in zip(entry_traction, traction)))
                break
        else:
            self._tractions.append((faces, traction))
        self._loads_dirty = True

    def set_held_values(self, dof_mask, u_x=None, u_y=None, u_z=None) -> None:
        """Set the held values [m] (None = unchanged) on a hold's dof mask (n_dofs, 3) in place: the held set and the
        assembly stay (a reused sweep build's prescribed displacement)."""
        mask = np.asarray(dof_mask, dtype=bool)
        for component, value in enumerate((u_x, u_y, u_z)):
            if value is not None:
                self.system.held_values[mask[:, component], component] = float(value)
        self._loads_dirty = True
        if self.prestressed:   # K_sigma / K_T of the static state follow the held values
            self._drop_prestress()

    def reset_records(self) -> None:
        """RerunSolver (reused sweep build): clear the probe and frequency records, drop a kept static state (prestress)
        and mark the loads dirty, so the next study run solves afresh; the last converged nonlinear state stays the
        Newton start."""
        for probe, _, _ in self.probes:
            probe.clear()
        self.reset_frequency_response()
        if self.prestressed:
            self._drop_prestress()
        self._loads_dirty = True

    def add_body_force(self, mask, f_x, f_y, f_z) -> None:
        """Add a body force density [N/m^3] (numbers or per-element arrays) on the window elements under mask; void
        elements drop it."""
        elements = self._element_mask(mask, 'body force')
        for component, value in enumerate((f_x, f_y, f_z)):
            self._body_force[:, component] += np.where(elements, self._element_values(value, 'body force'), 0.0)
        self._loads_dirty = True

    def add_probe_at(self, point, component: str = 'u_abs') -> FieldProbe:
        """Record an output field at (x, y, z) [m] (its nodal display field evaluated in the containing element)
        whenever step records probes."""
        if component not in self.output_arrays:
            raise ValueError(f"probe component {component!r} is not one of {', '.join(self.output_arrays)}")
        elements, reference = self.domain.locate(np.asarray(point, dtype=np.float64).reshape(1, 3))
        if elements[0] < 0:
            raise ValueError(f'solid probe at {tuple(float(value) for value in point)} m lies outside the mesh')
        probe = FieldProbe(int(elements[0]), 0, component)
        self.probes.append((probe, int(elements[0]), reference[0]))
        return probe

    def reaction(self, region) -> tuple:
        """Force (F_x, F_y, F_z) [N] the support of region exerts on the solid in the last static solve (piezoelectric:
        the u rows of A x - f, the coupling term included)."""
        if self._shows_mode or self._shows_harmonic:
            shown = 'a vibration mode' if self._shows_mode else 'a frequency response (complex reactions)'
            raise RuntimeError(f'{self._label()}: support reactions need a static solve ({shown} is shown)')
        if not self.piezoelectric:
            return self.system.reaction(region.dof_mask)
        if self._piezo_reaction is None:
            raise RuntimeError(f'{self._label()}: reactions need a static solve first')
        system = self.system
        selected = np.asarray(region.dof_mask, dtype=bool) & system.held & system.active[:, None]
        totals = np.where(selected, self._piezo_reaction, 0.0).sum(axis=0)
        return float(totals[0]), float(totals[1]), float(totals[2])

    # --- piezoelectric electrodes ---------------------------------------------

    def add_face_electrode(self, name, faces, connection='voltage', voltage=0.0, charge=0.0,
                           load_resistance=50.0) -> Electrode:
        """Electrode on every node of outer faces of the window (edges and corners included): connection 'voltage'
        holds phi at voltage [V], 'floating' shares one phi carrying the net charge [C], 'resistive_load' shares one phi
        tied to ground through load_resistance [Ohm] (eigenfrequency and frequency response only)."""
        for face in faces:
            if face not in FACES_3D:
                raise ValueError(f'unknown face {face!r}; expected one of {FACES_3D}')
        mesh_faces = [self.system.face_faces(face) for face in faces]
        return self.add_surface_electrode(name, np.concatenate(mesh_faces) if mesh_faces else [], connection, voltage,
                                          charge, load_resistance)

    def add_surface_electrode(self, name, faces, connection='voltage', voltage=0.0, charge=0.0,
                              load_resistance=50.0) -> Electrode:
        """Electrode on every node of mesh faces (window faces or an object's interface faces); connection, voltage,
        charge and load_resistance as add_face_electrode."""
        return self._add_electrode(name, self.system.full.face_dofs(np.asarray(faces, dtype=np.int64)), connection,
                                   voltage, charge, load_resistance)

    def add_cell_electrode(self, name, mask, connection='voltage', voltage=0.0, charge=0.0,
                           load_resistance=50.0) -> Electrode:
        """Electrode on every node of the window elements under the element mask (a meshed metal electrode)."""
        elements = self._element_mask(mask, f"electrode '{name}'")
        return self._add_electrode(name, self.system.full.element_dofs_of(elements), connection, voltage, charge,
                                   load_resistance)

    def _add_electrode(self, name, scalar_dofs, connection, voltage, charge, load_resistance):
        if not self.piezoelectric:
            raise ValueError(f"{self._label()}: electrode '{name}' needs piezoelectric=True")
        connections = ELECTRODE_CONNECTIONS + (RESISTIVE_LOAD,)
        if connection not in connections:
            raise ValueError(f"electrode connection must be one of {', '.join(connections)}, got {connection!r}")
        load_resistance = float(load_resistance)
        if connection == RESISTIVE_LOAD and not (math.isfinite(load_resistance) and load_resistance > 0.0):
            raise ValueError(f"electrode '{name}': load_resistance must be finite and > 0 Ohm, got {load_resistance!r}")
        dofs = COMPONENTS * self.system.n_dofs + np.unique(np.asarray(scalar_dofs, dtype=np.int64))
        electrode = Electrode(str(name), dofs, connection, float(voltage), float(charge), load_resistance)
        self._electrodes.append(electrode)
        self.system.operator_dirty = True   # the held / tied potentials change the reduced operator
        self._loads_dirty = True
        return electrode

    @property
    def electrodes(self) -> tuple:
        """piezo.Electrode of the electrode features, in apply order."""
        return tuple(self._electrodes)

    def electrode_values(self, electrode) -> tuple:
        """(V [V], Q [C]) of an electrode in the last static solve or the shown vibration mode (per its 1 m peak
        displacement): its potential and the net free charge on it (a voltage electrode: the charge its source
        delivered)."""
        if self._shows_complex:
            shown = ('a frequency response: read them through an Admittance feature' if self._shows_harmonic
                     else 'a complex mode (a Bloch cell or a resistive_load electrode: the Resonance table gives its '
                          'damping)')
            raise RuntimeError(f"{self._label()}: electrode '{electrode.name}' values are complex for {shown}")
        if electrode.name not in self._electrode_values:
            raise RuntimeError(f"{self._label()}: electrode '{electrode.name}' values need a static solve or a shown "
                               'vibration mode first')
        return self._electrode_values[electrode.name]

    # --- eigenfrequency study hooks -------------------------------------------

    def solve_eigenfrequencies(self, n_resonances):
        """Up to n_resonances vibration modes nearest target_frequency (0 = the lowest) on the _modal_pencil, then the
        gas damping of every squeeze_film_damping feature per resonance -> ModalSolve, kept as eigen_solve.
        solve_eigenfrequencies_targets / _band (modal.MultiShiftModes) share the pencil and the films."""
        self.eigen_solve = None
        pencil, finish = self._modal_pencil(self.target_frequency)
        try:
            solve = pencil.solve_at(2.0 * math.pi * self.target_frequency, n_resonances)
        finally:
            pencil.release()
        self.eigen_solve = finish(solve)
        return self.eigen_solve

    def set_axial_wavenumber(self, wavenumber):
        """Guided (2.5D) k_z of the optomechanics study: not available on a tetrahedron mesh (ValueError)."""
        raise ValueError(f'{self._label()}: no guided (2.5D) modes on a tetrahedron mesh')

    def modal_overlap_data(self):
        """Host data the g0 overlap (optomechanics_fem) reads after an eigenfrequency solve: kind 'tetrahedra', the full
        Lagrange map of the modes, solid elements, p11 / p12 per element and the consistent mass."""
        if self.eigen_solve is None or self.modal_mass is None:
            raise ValueError(f'{self._label()}: no eigenfrequency solve yet')
        if self.piezoelectric:
            raise ValueError(f'{self._label()}: the optomechanical overlap of a piezoelectric solid is not modelled '
                             '(its modes carry the potential as a fourth dof per node)')
        if self.system.bloch is not None:
            raise ValueError(f'{self._label()}: the optomechanical overlap of a Bloch cell (periodic axes) is not '
                             'modelled')
        materials = self.domain.materials
        return {'kind': 'tetrahedra', 'dofs': self.system.full, 'components': COMPONENTS,
                'solid': np.array(to_host(self._solid), dtype=bool),
                'p11': _host(materials.photoelastic_p11), 'p12': _host(materials.photoelastic_p12), 'p44': None,
                'mass': self.modal_mass, 'axial_wavenumber': None}

    def add_squeeze_film(self, film):
        """Register a squeeze_film.SqueezeFilmDamping (evaluated after every eigenfrequency solve); returns it."""
        self.squeeze_films.append(film)
        return film

    def _modal_pencil(self, lowest_target):
        """(modal.ModalPencil, finish) of K (+ K_sigma) and the consistent mass of the solid elements (supports held at
        0; piezoelectric: the coupled pencil of PiezoSystem, voltage electrodes shorted, floating ones open,
        resistive_load ones damped through their resistor (complex modes); periodic axes: on the Bloch prolongation,
        complex modes; loss / PML: the damped complex pencil); finish(ModalSolve) expands piezo modes and evaluates the
        squeeze films. lowest_target [Hz] (None: a band) <= 0 with free rigid motions raises."""
        system = self.system
        if system.operator_dirty:
            self._assemble()
        if system.n_rigid_motions and lowest_target is not None and lowest_target <= 0.0:
            raise ValueError(f'{self._label()}: {system.n_rigid_motions} rigid motion(s) are free and the target '
                             'frequency is 0, where K - 0 M is singular: set the target > 0 (the 0 Hz rigid-body modes '
                             'are then dropped) or add supports')
        self.modal_mass = system.mass(self._solid_density('an eigenfrequency study'))
        bloch = system.bloch
        bloch_text = '' if bloch is None else bloch.text()
        if self.piezoelectric:   # coupled pencil: voltage electrodes short-circuited, floating ones open, loads damp
            piezo = self._piezo_system
            self.modal_mass = coupled_mass_3d(self.modal_mass)
            prolongation = None if bloch is None else bloch_prolongation(*bloch.coupled_masters(), piezo.free)[1]
            pencil = piezo.modal_pencil(self.modal_mass, self.eigen_solver, prolongation=prolongation,
                                        bloch_text=bloch_text, rigid_motions=system.n_rigid_motions)
            return pencil, lambda solve: self._evaluated_films(piezo.expanded(solve))
        stiffness = self._pencil_stiffness(with_loads=True)
        loss = self._stiffness_loss()   # eta K of the elastic stiffness only (K_sigma carries no loss)
        prolongation = None if bloch is None else bloch_prolongation(*bloch.vector_masters(), system.free)[1]
        if system.pml is not None or loss is not None:   # complex pencil: radiation and / or material Q
            if loss is not None:
                stiffness = pattern_sum([stiffness.astype(np.complex128), 1j * loss])
            pencil = ModalPencil(stiffness, self.modal_mass, system.free, COMPONENTS, self.eigen_solver,
                                 prolongation=prolongation, rigid_motions=system.n_rigid_motions, damped=True,
                                 shell_dofs=None if system.pml is None else system.shell_only_dofs(),
                                 bloch_text=bloch_text, pml_text='' if system.pml is None else system.pml.text(),
                                 loss_text='' if loss is None else 'mechanical_loss_factor')
        else:
            pencil = ModalPencil(stiffness, self.modal_mass, system.free, COMPONENTS, self.eigen_solver,
                                 prolongation=prolongation, rigid_motions=system.n_rigid_motions, bloch_text=bloch_text)
            if prolongation is None:   # the static K_ff factor: previous= of the first shift (released with the pencil)
                pencil.factor = self._take_pencil_factor()
        getattr(self._take_pencil_factor(), 'release', lambda: None)()   # a complex pencil cannot refactorise it
        return pencil, self._evaluated_films

    def _evaluated_films(self, solve):
        """solve after the gas damping of every squeeze_film_damping feature per resonance."""
        if self.squeeze_films:
            n_vector = COMPONENTS * self.system.n_dofs
            for film in self.squeeze_films:
                film.evaluate(self.domain.mesh, self._solid, solve.resonances, self.modal_mass, n_vector)
        return solve

    def _stiffness_loss(self, stretched=True):
        """Host CSR eta K of the solid elements (mechanical_loss_factor; an elastic PML shell stretched as K unless
        stretched is False); None when every eta is 0."""
        eta = self._loss_values(LOSS_ARRAYS[0], self._solid)
        if not eta.any():
            return None
        material = eta[:, None, None] * self._stiffness
        system = self.system
        if system.pml is None or not stretched:
            return assemble_elastic_stiffness_3d(system.full, material, elements=self._solid).tocsr()
        window = self.domain.mesh.window_elements()
        return pattern_sum([
            assemble_elastic_stiffness_3d(system.full, material, elements=self._solid & window),
            assemble_elastic_stiffness_3d(system.full, material, elements=self._solid & ~window,
                                          stretch=system.pml.stretches)])

    def select_resonance(self, number):
        """Show mode `number` (1-based): displacement, strain and the modal stress (no thermal part) of the
        peak-normalised mode shape; piezoelectric: its potential, E / D and the electrode values of the mode; a complex
        Bloch mode: its real part (phase 0) and u_abs = |u|, no electrode values."""
        resonance = self.eigen_solve.resonances[number - 1]
        mode = np.asarray(resonance.displacement)
        omega = float(resonance.omega)   # the time-averaged energy densities of the mode
        if np.iscomplexobj(mode):
            self._write_complex_outputs(mode, omega=omega)
        else:
            mode = mode.astype(np.float64, copy=False)
            n_vector = COMPONENTS * self.system.n_dofs
            self.system.unknown = mode[:n_vector].reshape(-1, COMPONENTS)
            if self.piezoelectric:
                self._potential = mode[n_vector:]
                self._electrode_values = self._piezo_system.electrode_values(mode, self._piezo_system.operator @ mode)
            self._write_outputs(thermal=False, omega=omega)
            self._shows_complex = False
        self._shows_mode, self._shows_harmonic = True, False
        self.selected_resonance = number

    # --- frequency_response study hooks -----------------------------------------

    def reset_frequency_response(self):
        """Forget the FrequencyPoints, frequency probe and support reaction records of an earlier frequency_response run
        (the study calls it first)."""
        self.frequency_points = []
        for record, _, _ in self.frequency_probes:
            record.clear()
        for record in self._harmonic_reactions.values():
            record.clear()
        self.reduced_sweep, self._field_policy, self._unpublished_frequency = None, None, None
        self._unpublished_solution = None
        self._record_rows = None
        if self._harmonic is not None:
            self._harmonic.clear_reduced()
        if self._piezo_system is not None:
            self._piezo_system.clear_harmonic_plan()

    def _harmonic_inputs(self):
        """(load, mass, loss) of a driven solve: the checks, assembly, harmonic operators, relaxed prestress and the
        elastic HarmonicSystem (kept as _harmonic)."""
        system = self.system
        if system.bloch is not None or self.pml_faces:
            what = 'periodic axes (Bloch cells)' if system.bloch is not None else 'an elastic PML'
            raise ValueError(f'{self._label()}: {what} is solved by the eigenfrequency study only, not by a frequency '
                             'response')
        if system.operator_dirty:
            self._assemble()
        # frequency independent: kept until a load input changes (_loads_dirty) or an assembly replaces the solid set;
        # the dead loads only: the initial stress and thermal strain are static (the prestress), never a drive
        key = (system.full, self._solid)
        if self._harmonic_load is None or any(old is not new for old, new in zip(self._harmonic_load[0], key)):
            load = self._external_load()
            load.flags.writeable = False
            self._harmonic_load = (key, load)
        load = self._harmonic_load[1]
        if self._harmonic_operators is None:
            self._harmonic_operators = self._build_harmonic_operators()
        mass, loss = self._harmonic_operators
        if not self.piezoelectric:
            if self.prestressed and (self._prestress is None or self._prestress[0]):
                self.solve_static_state(with_loads=False)   # drops _harmonic
            if self._harmonic is None:
                self._harmonic = HarmonicSystem(self._pencil_stiffness(with_loads=False), loss, mass, system.free,
                                                runtime=system.runtime)
        return load, mass, loss

    def _harmonic_solve(self, omega, load, mass, loss, fields=True):
        """(x, A x) over every dof at omega [rad/s] (elastic: A x exact only at the output rows); fields False: x too
        (the reduced model, or a direct solve whose solution the HarmonicSystem keeps for publish, R6)."""
        system = self.system
        if self.piezoelectric:
            return self._piezo_system.solve_harmonic(omega, mass, loss, load, system.held_values, self._piezo_counters,
                                                     fields=fields)
        if not self._harmonic.has_output_rows:   # also with fields: A x is read only at the record rows
            self._harmonic.set_output_rows(self._harmonic_output_rows())
        return self._harmonic.solve(omega, system.fixed_values(), load, fields=fields, product_at_rows=True)

    def _harmonic_output_rows(self) -> np.ndarray:
        """Int64 rows (node x 3 + component) the frequency probes and support reactions read."""
        system = self.system
        rows = [(COMPONENTS * np.asarray(system.full.element_dofs[element], dtype=np.int64)[:, None]
                 + np.arange(COMPONENTS)).ravel() for _, element, _ in self.frequency_probes]
        for region in self._supports:
            selected = np.asarray(to_host(region.dof_mask), dtype=bool) & system.held & system.active[:, None]
            rows.append(np.flatnonzero(selected.ravel()))
        return np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)

    def plan_frequency_response(self, frequencies, options):
        """ReducedSweepSolver hook (after reset_frequency_response): options {method, tolerance, moments, max_points,
        modal_count, fields}; 'direct' -> None (only the field policy, R6), else the HarmonicSystem's reduced model over
        frequencies [Hz] (piezo: the PiezoSystem's) with the record rows precomputed -> its info dict."""
        method = options.get('method', 'direct')
        if method == 'direct':
            self._field_policy = field_policy(options, frequencies)
            return None
        frequencies = np.asarray(frequencies, dtype=np.float64).ravel()
        if not (frequencies.size and np.all(np.isfinite(frequencies)) and np.all(frequencies > 0.0)):
            raise ValueError(f'{self._label()}: a reduced frequency sweep needs frequencies > 0 Hz')
        load, mass, loss = self._harmonic_inputs()
        omegas = 2.0 * math.pi * frequencies
        reduced = dict(tolerance=options.get('tolerance', 1e-6), moments=options.get('moments', 4),
                       max_points=options.get('max_points', 40), modal_count=options.get('modal_count', 20),
                       eigen_solver=self.eigen_solver, label=self._label())
        rows = self._harmonic_output_rows()
        if self.piezoelectric:
            info = self._piezo_system.plan_harmonic(omegas, method, mass, loss, load, self.system.held_values,
                                                    self._piezo_counters, rows=rows, **reduced)
        else:
            info = self._harmonic.plan_reduced(omegas, method, self.system.fixed_values(), load, **reduced)
            self._harmonic.set_output_rows(rows)
        self._field_policy = field_policy(options, frequencies)
        self.reduced_sweep = info
        return info

    def solve_frequency_response(self, frequency):
        """frequency_response study hook: the driven harmonic response at frequency [Hz] > 0 of every load, prescribed
        displacement and electrode (amplitudes at phase 0; the initial stress is no drive) with the mechanical /
        dielectric losses (and K_sigma of the relaxed initial stress with stress_stiffening): one complex symmetric
        factor of the reduced pencil per frequency by the 'linear' runtime (host SuperLU or cuDSS LDL^T, plan reused
        across frequencies), or the plan_frequency_response model; a free body is regular at f > 0. Appends a
        FrequencyPoint, records the frequency probes and support reactions and shows the response (real parts, u_abs =
        the amplitude |u|) where the field policy asks."""
        frequency = float(frequency)
        if not (math.isfinite(frequency) and frequency > 0.0):
            raise ValueError(f'{self._label()}: a frequency response needs a frequency > 0 Hz, got {frequency!r}')
        system = self.system
        load, mass, loss = self._harmonic_inputs()
        omega = 2.0 * math.pi * frequency
        fields = wants_fields(self._field_policy, frequency)
        planned = self.reduced_sweep is not None
        # a skipped piezoelectric direct frequency still solves in full (its electrode charges read every phi row)
        solution, product = self._harmonic_solve(omega, load, mass, loss,
                                                 fields or (not planned and self.piezoelectric))
        electrodes = self._piezo_system.electrode_values(solution, product) if self.piezoelectric else {}
        self.frequency_points.append(FrequencyPoint(frequency, electrodes))
        n_vector = COMPONENTS * system.n_dofs
        self._record_harmonic(frequency, solution[:n_vector].reshape(-1, COMPONENTS),
                              (product[:n_vector] - load).reshape(-1, COMPONENTS))
        if fields:
            self._show_frequency_response(solution, omega)
        self._unpublished_frequency = None if fields else frequency
        # elastic direct: the HarmonicSystem keeps the solution (last_solution); piezoelectric direct: kept here
        self._unpublished_solution = None if fields or planned or not self.piezoelectric else solution

    def publish_frequency_response(self):
        """ReducedSweepSolver hook: show the last solved frequency when the field policy skipped it (the kept direct
        solution: the piezoelectric one here, the elastic one in the HarmonicSystem; or a re-solve of the reduced
        model)."""
        frequency, self._unpublished_frequency = self._unpublished_frequency, None
        solution, self._unpublished_solution = self._unpublished_solution, None
        if frequency is None:
            return
        omega = 2.0 * math.pi * frequency
        if solution is None and self.reduced_sweep is None and self._harmonic is not None:
            solution = self._harmonic.last_solution()
        if solution is None:
            load, mass, loss = self._harmonic_inputs()
            solution, _ = self._harmonic_solve(omega, load, mass, loss)
        self._show_frequency_response(solution, omega)

    def _show_frequency_response(self, solution, omega):
        self._write_complex_outputs(solution, omega=omega)
        self._shows_mode, self._shows_harmonic = False, True
        self.selected_resonance = None

    # --- complex records of the frequency response (K4) ------------------------

    def add_frequency_probe_at(self, point, label='') -> HarmonicProbeRecord:
        """Record the complex displacement (u_x, u_y, u_z) [m] at (x, y, z) [m] (the nodal field evaluated in the
        containing element) at every solved frequency."""
        elements, reference = self.domain.locate(np.asarray(point, dtype=np.float64).reshape(1, 3))
        if elements[0] < 0:
            raise ValueError(f'frequency probe at {tuple(float(value) for value in point)} m lies outside the mesh')
        record = HarmonicProbeRecord(('u_x', 'u_y', 'u_z'), unit='m', label=label or 'Displacement')
        self.frequency_probes.append((record, int(elements[0]), reference[0]))
        return record

    def register_support(self, region):
        """Keep a support feature's region: its complex reaction is recorded per solved frequency."""
        self._supports.append(region)
        self._harmonic_reactions[region.name] = HarmonicProbeRecord(('F_x', 'F_y', 'F_z'), unit='N',
                                                                    label=f'Reaction of {region.name}')
        return region

    def _record_harmonic(self, frequency, displacement, residual):
        """Record the frequency probes (complex nodal displacement (n_dofs, 3) at their points) and the support
        reactions (sum of the complex A x - f (n_dofs, 3) over each support's held active dofs)."""
        probes, supports = self._record_layout()
        for (record, _, _), (dofs, weights) in zip(self.frequency_probes, probes):
            values = weights @ displacement[dofs]   # (3,) complex: the basis at the point, 0 on inactive dofs
            record.record(frequency, {name: complex(values[component])
                                      for component, name in enumerate(record.components)})
        for region, rows in zip(self._supports, supports):
            totals = [complex(residual[component_rows, component].sum())
                      for component, component_rows in enumerate(rows)]
            self._harmonic_reactions[region.name].record(frequency, {'F_x': totals[0], 'F_y': totals[1],
                                                                     'F_z': totals[2]})

    def _record_layout(self):
        """([(element dofs, basis weights at the point, 0 on inactive dofs)] per frequency probe, [(held active dof
        rows per component)] per support) of the frequency-response records: kept per sweep (reset_frequency_response)
        and assembly (the record pass reads only these rows per frequency)."""
        system = self.system
        key = (system.full, self._solid)
        kept = self._record_rows
        if kept is None or any(old is not new for old, new in zip(kept[0], key)):
            probes = []
            for _, element, reference in self.frequency_probes:
                dofs = np.asarray(system.full.element_dofs[element], dtype=np.int64)
                basis = lagrange_basis_3d(system.full.order, reference[None, :])[0][0]
                probes.append((dofs, np.where(system.active[dofs], basis, 0.0)))
            supports = []
            for region in self._supports:
                selected = np.asarray(to_host(region.dof_mask), dtype=bool) & system.held & system.active[:, None]
                supports.append(tuple(np.flatnonzero(selected[:, component]) for component in range(COMPONENTS)))
            kept = self._record_rows = (key, probes, supports)
        return kept[1], kept[2]

    def _build_harmonic_operators(self):
        """Host (mass, loss or None) of the last assembly: the consistent mass of the solid elements (piezoelectric: with
        the massless potential block) and the loss operator eta K (mechanical_loss_factor of the solid elements;
        piezoelectric: [[eta K, 0], [0, +tan delta K_phiphi]] with the dielectric_loss_tangent of the window elements)."""
        import scipy.sparse
        system = self.system
        mass = system.mass(self._solid_density('a frequency response'))
        stiffness_loss = self._stiffness_loss(stretched=False)
        if not self.piezoelectric:
            return mass, stiffness_loss
        tan_delta = self._loss_values(LOSS_ARRAYS[1], self._window_elements)
        if stiffness_loss is None and not tan_delta.any():
            return coupled_mass_3d(mass), None
        if stiffness_loss is None:
            stiffness_loss = scipy.sparse.csr_matrix(system.operator.shape)
        loss = coupled_operator_3d(system.full, stiffness_loss, np.zeros_like(self._piezo_tensor),
                                   -tan_delta[:, None, None] * self._permittivity, self._solid, self._window_elements)
        return coupled_mass_3d(mass), loss

    def _loss_values(self, name, elements):
        """Host float64 loss array `name` (LOSS_ARRAYS) on the bool element mask, 0 elsewhere (0 when no material set
        it)."""
        materials = self.domain.materials
        if name not in set(materials.property_names()):
            return np.zeros(self.domain.n_elements)
        return np.where(elements, _host(getattr(materials, name)), 0.0)

    def _write_complex_outputs(self, solution, omega=None):
        """Write the real parts of a complex solution (a response, a Bloch mode) over every (coupled) dof into the
        outputs (no thermal part, no initial stress: both are static) and u_abs = |u| (nodal and at the centroids), the
        energy densities as time averages at omega [rad/s]; electrode tables become unavailable."""
        system = self.system
        n_vector = COMPONENTS * system.n_dofs
        displacement = solution[:n_vector].reshape(-1, COMPONENTS)
        system.unknown = np.ascontiguousarray(displacement.real)
        if self.piezoelectric:
            self._potential = np.ascontiguousarray(solution[n_vector:].real)
        self._write_outputs(False, complex_displacement=np.array(displacement), prestress=False, omega=omega)
        self._electrode_values = {}
        self._shows_complex = True

    # --- display fields ---------------------------------------------------------

    def field_dofs(self, name):
        """fem DofMap3D of state field or source slot `name` (the transfer contract): every one is per element (P0)."""
        if name not in (*self.output_arrays, *self.input_arrays, *MATERIAL_STATE_ARRAYS, *self.source_slots):
            raise ValueError(f"{self._label()} has no state field or source slot '{name}'")
        return self._element_map

    def lagrange_field(self, component) -> tuple:
        """(DofMap3D, host coefficients) of a renderer component: outputs on the full-mesh P1 / P2 map (0 off the
        solid), inputs and E / nu per element (P0)."""
        if component in self.output_arrays:
            self._ensure_outputs((component,))
            values = self._nodal.get(component)
            return self.system.full, (np.zeros(self.system.n_dofs) if values is None else values)
        if component in HARMONIC_IMAG_FIELDS_3D:   # Im of a shown complex solution (0 for a real one)
            complex_displacement = self._complex_displacement
            if complex_displacement is None:
                return self.system.full, np.zeros(self.system.n_dofs)
            column = HARMONIC_IMAG_FIELDS_3D.index(component)
            return self.system.full, np.where(self.system.active, complex_displacement[:, column].imag, 0.0)
        if component in INPUT_ARRAYS_3D or component in MATERIAL_STATE_ARRAYS:
            source = self.fields if component in INPUT_ARRAYS_3D else self.domain.materials
            return self._element_map, _host(getattr(source, component))
        raise KeyError(f"{self._label()} has no field '{component}'")

    # --- internals ----------------------------------------------------------------

    def _assemble(self):
        """Element Voigt stiffness of the window elements (E / nu or the c_IJ entries; others void; ValueError for a
        poisson_ratio outside (-1, 0.5) where youngs_modulus > 0 and a c_IJ set that is not positive definite), then the
        system assembly (the factor follows lazily)."""
        self._ensure_outputs()   # pending outputs read the stiffness of their solve
        started = time.perf_counter()
        materials = self.domain.materials
        window = self._elastic_elements
        youngs_modulus = np.where(window, _host(materials.youngs_modulus), 0.0)
        poisson_ratio = np.where(window, _host(materials.poisson_ratio), 0.0)
        elastic = youngs_modulus > 0.0
        out_of_range = elastic & ((poisson_ratio <= -1.0) | (poisson_ratio >= 0.5))
        if out_of_range.any():
            raise ValueError(f"{self._label()}: poisson_ratio must lie in (-1, 0.5) in every element with youngs_modulus "
                             f"> 0; found values from {poisson_ratio[elastic].min():.6g} to "
                             f'{poisson_ratio[elastic].max():.6g}')
        lame_lambda, mu = lame_coefficients(youngs_modulus * elastic, poisson_ratio * elastic)
        names = set(materials.property_names())
        tensor = {name: np.where(window, _host(getattr(materials, name)), np.nan)
                  for name in STIFFNESS_TENSOR_ARRAYS_3D if name in names}
        stiffness = element_stiffness_3d(lame_lambda, mu, tensor)
        check_positive_definite_3d(stiffness, self._label)
        self._drop_host_inputs()   # beta and the solid-masked inputs follow the new stiffness
        self._stiffness = stiffness.matrix
        self._solid = stiffness.solid
        self._point_cache = {}                    # its point values are per solid element
        self._harmonic_operators = self._prestress = None
        self._drop_harmonic()
        self._tangent_assembler = None            # the solid / held sets or the stiffness may have changed
        self._nonlinear_start.forget_last()
        if self.pml_faces:
            self.system.pml = self._pml_shell()
        self._assembly_seconds = {'element stiffness': time.perf_counter() - started}
        # geometric nonlinearity: K at first use (a static Newton solve never reads it; pencils / piezo do)
        self.system.assemble(stiffness.matrix, stiffness.solid,
                             defer_operator=self.geometric_nonlinearity and not self.piezoelectric)
        self._assembly_seconds.update(self.system.assembly_seconds)
        if self.piezoelectric:
            self._assemble_piezo()

    def _pml_shell(self):
        """pml3d.ElasticPMLShell3D at the target frequency for the fastest wave sqrt(max c_II / rho) of the shell's solid
        elements (the window's when the shell is void)."""
        wave_speed = ring_wave_speed(np.einsum('mii->mi', self._stiffness).max(axis=1),
                                     _host(self.domain.materials.rho), self._solid, self._window_elements, self._label)
        try:
            return elastic_pml_shell_3d(self.domain.mesh, self.pml_faces, 2.0 * math.pi * self.target_frequency,
                                        wave_speed, self.pml_reflection)
        except ValueError as exc:
            raise ValueError(f'{self._label()}: {exc}') from None

    def _assemble_piezo(self):
        """piezo.PiezoSystem of the current materials (e of the solid elements, eps_dc of every window element) on the
        assembled K_uu and the electrodes; its factor (the 'linear' runtime) follows with the first static solve, an
        earlier factor handed over for cuDSS plan reuse. ValueError for a permittivity that is not positive definite."""
        started = time.perf_counter()
        system = self.system
        materials = self.domain.materials
        window = self._window_elements
        n_elements = self.domain.n_elements
        names = set(materials.property_names())
        piezo = piezo_tensor_3d({name: np.where(self._solid, _host(getattr(materials, name)), 0.0)
                                 for name in PIEZO_TENSOR_ARRAYS_3D if name in names}, n_elements)
        permittivity = permittivity_tensor_3d({name: _host(getattr(materials, name))
                                               for name in PERMITTIVITY_TENSOR_ARRAYS_3D if name in names}, n_elements)
        check_permittivity_3d(permittivity, window, self._label)
        bloch = system.bloch
        if bloch is not None:   # a tied / held phi there would contradict its periodic image
            on_periodic = bloch.on_periodic_faces
            for electrode in self._electrodes:
                if on_periodic[np.asarray(electrode.dofs) - COMPONENTS * system.n_dofs].any():
                    raise ValueError(f"{self._label()}: electrode '{electrode.name}' touches a periodic face: electrodes "
                                     'of a Bloch cell must stay off its periodic faces')
        operator = coupled_operator_3d(system.full, system.operator, piezo, permittivity, self._solid, window)
        coupled = time.perf_counter()
        displacement_dofs, potential_dofs = block_layout(system.n_dofs)
        previous = None if self._piezo_system is None else self._piezo_system.factor
        self._piezo_system = PiezoSystem(operator, self._electrodes, displacement_dofs, potential_dofs,
                                         system.held & system.active[:, None], self._label,
                                         potential_active=potential_active(system.full, window), runtime=system.runtime,
                                         previous=previous, require_ground=bloch is None or bloch.trivial)
        self._piezo_tensor, self._permittivity = piezo, permittivity
        self._assembly_seconds.update({'piezo coupling, dielectric and block operator': coupled - started,
                                       'electrode reduction G^T A G': time.perf_counter() - coupled})

    def _solve_piezo(self, load):
        """The static coupled solve (piezo.PiezoSystem): u into the system unknown, phi, reactions and electrode values
        kept."""
        system = self.system
        solution, product = self._piezo_system.solve_static(load, system.held_values, self._piezo_counters)
        n_vector = COMPONENTS * system.n_dofs
        system.accept_solution(solution[:n_vector], load)
        self._potential = solution[n_vector:]
        self._piezo_reaction = (product[:n_vector] - load).reshape(-1, COMPONENTS)
        self._electrode_values = self._piezo_system.electrode_values(solution, product)

    def _expansion(self):
        """Host (n_elements, 3) alpha_xx / yy / zz [1/K] (cached, read-only)."""
        materials = self.domain.materials
        return self._host_input('expansion', lambda: np.stack([_host(getattr(materials, name))
                                                               for name in _EXPANSION_NAMES], axis=1))

    def _thermal_stress(self):
        """Host (n_elements, 6) Voigt beta_I = c_IJ alpha_J [Pa/K] (shear rows non-zero for an anisotropic c; cached,
        read-only)."""
        return self._host_input('beta', lambda: thermal_stress_3d(self._stiffness, self._expansion()))

    def _total_load(self, dead_loads=True):
        """Load vector (3 n_dofs): (dead_loads) tractions on faces beside a solid element and the applied + coupled body
        forces of the solid elements, the thermal stress load int B^T beta dT and the initial-stress force -int B^T
        sigma0 dV."""
        full = self.system.full
        load = self._external_load() if dead_loads else np.zeros(COMPONENTS * full.n_dofs)
        initial = self._initial_stress()
        if initial is not None:
            load += initial_stress_force_3d(full, initial, elements=self._solid)
        stress = self._thermal_stress() * self._temperature_change()[:, None]
        load += assemble_stress_load_3d(full, np.where(self._solid[:, None], stress, 0.0), elements=self._solid)
        return load

    def _external_load(self):
        """The dead-load part of _total_load (3 n_dofs): tractions on faces beside a solid element and the applied +
        coupled body forces of the solid elements (no initial-stress or thermal term)."""
        full = self.system.full
        load = np.zeros(COMPONENTS * full.n_dofs)
        pairs = self.domain.mesh.face_elements
        for faces, traction in self._tractions:
            first, second = pairs[faces, 0], pairs[faces, 1]
            beside_solid = self._solid[first] | ((second >= 0) & self._solid[np.maximum(second, 0)])
            load += assemble_face_traction_3d(full, faces[beside_solid], traction)
        coupled = np.stack([_host(getattr(self.fields, name)) for name in ('body_force_x', 'body_force_y',
                                                                            'body_force_z')], axis=1)
        body = np.where(self._solid[:, None], self._body_force + coupled, 0.0)
        load += assemble_body_load_3d(full, tuple(body[:, component] for component in range(COMPONENTS)),
                                      elements=self._solid)
        return load

    def _point_outputs(self, elements, strains, thermal, field=None, prestress=False):
        """{output: (m, q)} strain / stress values at points of mesh elements from Voigt strains (m, q, 6): T = c S -
        beta dT (thermal False: dT = 0, a mode shape) - e^T E with the electric field (m, q, 3) at the points (None: no
        potential) + the initial stress sigma0 (prestress: a static solve)."""
        alpha = self._expansion()[elements]
        beta = self._thermal_stress()[elements]
        temperature_change = self._temperature_change()[elements][:, None] if thermal else 0.0
        normal = [strains[..., axis] for axis in range(3)]
        voigt = stress_3d(self._stiffness[elements], strains)
        if field is not None:
            voigt = voigt - np.einsum('mkJ,mqk->mqJ', self._piezo_tensor[elements], field)
        initial = self._initial_stress() if prestress else None
        if initial is not None:
            voigt = voigt + initial[elements][:, None, :]
        stresses = [voigt[..., row] - beta[:, row:row + 1] * temperature_change for row in range(6)]
        shear_yz, shear_xz, shear_xy = stresses[3:]
        von_mises = np.sqrt(0.5 * ((stresses[0] - stresses[1]) ** 2 + (stresses[1] - stresses[2]) ** 2
                                   + (stresses[2] - stresses[0]) ** 2)
                            + 3.0 * (shear_xy ** 2 + shear_xz ** 2 + shear_yz ** 2))
        outputs = {}
        for axis, letter in enumerate('xyz'):
            outputs[f'strain_{letter}{letter}'] = normal[axis]
            outputs[f'elastic_strain_{letter}{letter}'] = normal[axis] - alpha[:, axis:axis + 1] * temperature_change
            outputs[f'stress_{letter}{letter}'] = stresses[axis]
        outputs.update({'strain_xy': 0.5 * strains[..., 5], 'strain_xz': 0.5 * strains[..., 4],
                        'strain_yz': 0.5 * strains[..., 3], 'stress_xy': shear_xy, 'stress_xz': shear_xz,
                        'stress_yz': shear_yz, 'von_mises': von_mises})
        return outputs

    def _write_outputs(self, thermal=True, complex_displacement=None, prestress=None, green_lagrange=False,
                       omega=None):
        """Mark the outputs of system.unknown / _potential pending (thermal False: no -beta dT, a mode shape;
        complex_displacement: u_abs = |u| of a complex solution; prestress None = thermal: the stress includes sigma0;
        green_lagrange: Green-Lagrange strains, a nonlinear static solve; omega [rad/s]: energy densities as time
        averages of a mode / response, None static); each _OUTPUT_GROUPS group is computed on its first read
        (_ensure_outputs)."""
        self._nodal = {}
        self._output_thermal = thermal
        self._output_prestress = thermal if prestress is None else prestress
        self._output_green_lagrange = bool(green_lagrange)
        self._output_omega = None if omega is None else float(omega)
        self._complex_displacement = complex_displacement
        self._pending_outputs = set(_OUTPUT_GROUPS)
        self._point_cache = {}

    def _write_element_field(self, name, elements, values):
        """The per-element state array `name`: values on `elements`, 0 elsewhere."""
        full_values = np.zeros(self.domain.n_elements)
        full_values[elements] = values
        xp = self.domain.xp
        xp.copyto(getattr(self.fields, name), xp.asarray(full_values, dtype=DTYPE))

    def _write_primary_outputs(self):
        """Displacement and u_abs (nodal exact, 0 off the solid; centroids of the solid elements) and the potential
        (nodal exact; centroids of the window elements)."""
        system = self.system
        displacement = system.unknown
        solid = np.flatnonzero(self._solid)
        nodal = self._nodal
        for component, name in enumerate(('u_x', 'u_y', 'u_z')):
            nodal[name] = np.where(system.active, displacement[:, component], 0.0)
        def centroid_displacement(field):   # (m, 3) at the solid centroids: basis weights only, no element geometry
            field = np.asarray(field, dtype=np.float64)
            return np.stack([centroid_values_3d(system.full, field[:, component], solid)
                             for component in range(COMPONENTS)], axis=1)
        centre = centroid_displacement(displacement)
        centre_values = {name: centre[:, component] for component, name in enumerate(('u_x', 'u_y', 'u_z'))}
        complex_displacement = self._complex_displacement
        if complex_displacement is None:
            nodal['u_abs'] = np.sqrt(nodal['u_x'] ** 2 + nodal['u_y'] ** 2 + nodal['u_z'] ** 2)
            centre_values['u_abs'] = np.sqrt(np.sum(centre ** 2, axis=-1))
        else:
            nodal['u_abs'] = np.where(system.active, np.sqrt(np.sum(np.abs(complex_displacement) ** 2, axis=1)), 0.0)
            centre_imag = centroid_displacement(complex_displacement.imag)
            centre_values['u_abs'] = np.sqrt(np.sum(centre ** 2 + centre_imag ** 2, axis=-1))
        for name, values in centre_values.items():
            self._write_element_field(name, solid, values)
        if self._potential is not None:
            nodal['electric_potential'] = self._potential.copy()
            window = np.flatnonzero(self._window_elements)
            self._write_element_field('electric_potential', window,
                                      centroid_values_3d(system.full, self._potential, window))

    def _write_mechanical_outputs(self):
        """Strain / stress / von Mises: nodal values averaged from the element values at the dof points over the solid
        elements (0 off the solid) and the centroid values of the solid elements."""
        full = self.system.full
        solid = np.flatnonzero(self._solid)
        thermal = self._output_thermal
        local_points = local_dof_points_3d(self.order)
        local_strains = self._point_values(local_points)[1]
        prestress = self._output_prestress
        local = self._point_outputs(solid, local_strains, thermal, self._electric_field(solid, local_points), prestress)
        names = list(local)
        averaged = nodal_average_3d(full, np.stack([local[name] for name in names], axis=-1), elements=solid)
        self._nodal.update({name: averaged[:, column] for column, name in enumerate(names)})
        centre_strains = self._point_values(CENTROID_3D)[1]
        centre = self._point_outputs(solid, centre_strains, thermal, self._electric_field(solid, CENTROID_3D),
                                     prestress)
        for name in names:
            self._write_element_field(name, solid, centre[name][:, 0])
        self._release_point_values('mechanical')

    def _point_values(self, reference_points, imaginary=False):
        """Read-only (values (m, q, 3), Voigt strains (m, q, 6)) of the shown solution (imaginary: the imaginary part of
        a complex one) at reference points of the solid elements (Green-Lagrange after a nonlinear static solve, else
        linear); one evaluation per solution, part and point set, shared by the output groups."""
        key = (bool(imaginary), np.asarray(reference_points, dtype=np.float64).tobytes())
        found = self._point_cache.get(key)
        if found is None:
            displacement = (np.ascontiguousarray(self._complex_displacement.imag) if imaginary
                            else self.system.unknown)
            found = element_strains_3d(self.system.full, displacement, reference_points,
                                       elements=np.flatnonzero(self._solid),
                                       green_lagrange=self._output_green_lagrange and not imaginary)
            for array in found:
                array.flags.writeable = False
            self._point_cache[key] = found
        return found

    def _release_point_values(self, group):
        """Drop the shared _point_values once no other pending output group (besides `group`, being written) reads
        them (electric only with a potential)."""
        readers = self._pending_outputs - {'primary', group}
        if self._potential is None:
            readers.discard('electric')
        if not readers:
            self._point_cache = {}

    def _energy_values(self, elements, reference_points):
        """(strain, kinetic) energy densities (m, q) [J/m^3] at reference points of the solid elements (elements =
        np.flatnonzero(_solid)): static 1/2 eps_el : C eps_el and 0 (eps_el = eps - alpha dT, Green-Lagrange after a
        nonlinear solve); a mode / response at omega the time averages 1/4 (eps_r : C eps_r + eps_i : C eps_i) and
        1/4 rho omega^2 |u|^2 (u = u_r + j u_i)."""
        stiffness = self._stiffness[elements]
        values, strains = self._point_values(reference_points)
        if self._output_thermal:
            change = self._temperature_change()[elements][:, None]
            strains = strains.copy()
            strains[..., :3] -= self._expansion()[elements][:, None, :] * change[..., None]
        omega = self._output_omega
        factor = 0.5 if omega is None else 0.25
        strain_energy = factor * np.einsum('mqi,mij,mqj->mq', strains, stiffness, strains)
        if omega is None:
            return strain_energy, np.zeros_like(strain_energy)
        squared = np.sum(values ** 2, axis=-1)
        if self._complex_displacement is not None:
            imaginary, imaginary_strains = self._point_values(reference_points, imaginary=True)
            strain_energy = strain_energy + 0.25 * np.einsum('mqi,mij,mqj->mq', imaginary_strains, stiffness,
                                                             imaginary_strains)
            squared = squared + np.sum(imaginary ** 2, axis=-1)
        density = self._host_input('density', lambda: _host(self.domain.materials.rho))[elements]
        return strain_energy, 0.25 * omega * omega * density[:, None] * squared

    def _write_energy_outputs(self):
        """Strain / kinetic energy densities: nodal values averaged from the element values at the dof points over the
        solid elements (0 off the solid) and the centroid values of the solid elements."""
        solid = np.flatnonzero(self._solid)
        local = self._energy_values(solid, local_dof_points_3d(self.order))
        averaged = nodal_average_3d(self.system.full, np.stack(local, axis=-1), elements=solid)
        centre = self._energy_values(solid, CENTROID_3D)
        for column, name in enumerate(ENERGY_OUTPUT_ARRAYS_3D):
            self._nodal[name] = averaged[:, column]
            self._write_element_field(name, solid, centre[column][:, 0])
        self._release_point_values('energy')

    def _electric_field(self, elements, reference_points):
        """E = -grad phi (m, q, 3) [V/m] of the last static solve at reference points of mesh elements; None without
        one."""
        if self._potential is None:
            return None
        return -element_gradients_3d(self.system.full, self._potential, reference_points, elements=elements)[1]

    def _electric_values(self, reference_points):
        """(window elements, {electric output: (m, q)}) at reference points of the window elements: phi, E = -grad phi
        and D = eps0 eps E + e S (S of the solid elements)."""
        system = self.system
        full = system.full
        window = np.flatnonzero(self._window_elements)
        potential, gradients = element_gradients_3d(full, self._potential, reference_points, elements=window)
        field = -gradients
        flux = EPS0 * np.einsum('mij,mqj->mqi', self._permittivity[window], field)
        solid_rows = self._solid[window]
        if solid_rows.any():   # the solid window elements' rows of the shared _point_values (over every solid element)
            strains = self._point_values(reference_points)[1][np.searchsorted(np.flatnonzero(self._solid),
                                                                              window[solid_rows])]
            flux[solid_rows] += np.einsum('miJ,mqJ->mqi', self._piezo_tensor[window[solid_rows]], strains)
        values = {'electric_potential': potential}
        for axis, letter in enumerate('xyz'):
            values[f'electric_field_{letter}'] = field[..., axis]
            values[f'electric_displacement_{letter}'] = flux[..., axis]
        return window, values

    def _write_electric_outputs(self):
        """E / D (piezoelectric only): nodal values averaged over the window elements at each dof and the centroid
        values of the window elements (0 off the window)."""
        if self._potential is None:
            return
        window, local = self._electric_values(local_dof_points_3d(self.order))
        averaged = nodal_average_3d(self.system.full, np.stack([local[name] for name in _ELECTRIC_POINT_OUTPUTS],
                                                               axis=-1), elements=window)
        self._nodal.update({name: averaged[:, column] for column, name in enumerate(_ELECTRIC_POINT_OUTPUTS)})
        _, centre = self._electric_values(CENTROID_3D)
        for name in _ELECTRIC_POINT_OUTPUTS:
            self._write_element_field(name, window, centre[name][:, 0])
        self._release_point_values('electric')

    def _record_probes(self, t):
        self._ensure_outputs(probe.field_component for probe, _, _ in self.probes)
        for probe, element, reference in self.probes:
            values = self._nodal.get(probe.field_component)
            value = 0.0 if values is None else float(evaluate_lagrange_3d(self.system.full, values, np.array([element]),
                                                                          reference[None, :], gradients=False)[0][0])
            probe.record_value(np.float32(value), t)
