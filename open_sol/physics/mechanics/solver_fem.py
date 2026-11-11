"""Static 2D solid mechanics on a triangle mesh (mesh plan 2.7, stage T5): vector Lagrange P2 (default, COMSOL's
quadratic) or P1 elements on the solid window elements, isoparametric on curved boundaries; plane strain / plane stress
/ generalized plane strain or axisymmetric (r, z) whole rings; isotropic or anisotropic c_ij (stiffness.ElementStiffness,
per element: D12), thermal strain, initial stress, supports, tractions on outer faces / lines, body forces; one factor
of K per assembly (system_fem.ElasticFEMSystem); vibration modes (modal.solve_modes; a loss factor: damped modes; stress
stiffening: K + K_sigma of the static state; a free body about a target > 0; an elastic PML ring: radiation Q; an
azimuthal order m >= 1 of the axisymmetric solid) and the elastic frequency response with complex probes / reactions per
frequency; strain / kinetic energy densities. Geometric nonlinearity (new_modules_plan K6, planar plane strain / plane
stress and axisymmetric): total-Lagrangian Newton with load steps, reactions from the internal force, pencils on its
tangent."""
import math
from types import SimpleNamespace

import numpy as np

from ...core.backend import to_host
from ...core.constants import DTYPE, FACES
from ...core.damped_iteration import DampedIteration, DampingSettings
from ...core.harmonic_probe import HarmonicProbeRecord
from ...core.metric import ring_angle
from ...core.monitors import FieldProbe
from ...core.solver_base import PhysicsSolver
from ...fem.assembly import assemble_load, nodal_average, pattern_sum
from ...fem.dofs import lagrange_dofs
from ...fem.elasticity import (AXIS_TOLERANCE, CENTROID, COMPONENTS, StressField, TangentAssembler,
                               assemble_body_load, assemble_edge_traction, assemble_stress_load,
                               azimuthal_element_strains, element_strains, green_lagrange_strains,
                               initial_stress_force, local_dof_points)
from ...fem.elements import element_geometry, lagrange_basis
from ...fem.fixed_pattern import scatter_rows
from ...fem.indicators import CHUNK_ITEMS, add_edge_terms, edge_sides, lagrange_at
from ...fem.reduced_outputs import field_policy, wants_fields
from ...solvers.arpack_shift_invert import ArpackShiftInvertSolver
from .fields import ENERGY_DENSITY_FIELDS, HARMONIC_IMAG_FIELDS, INPUT_ARRAYS, OUTPUT_ARRAYS, SOURCE_SLOTS
from .frequency_response import FrequencyPoint, HarmonicSystem
from .generalized_plane_strain import (BENDING_TERMS, condition_matrix, condition_targets, strain_term_values,
                                       terms_about_origin)
from .materials import (DENSITY_SCHEMA, EXPANSION_ARRAYS, LOSS_ARRAYS, MECHANICAL_SCHEMA, PRESTRESS_ARRAYS,
                        PRESTRESS_NAMES, SHEAR_PHOTOELASTIC_SCHEMA, STIFFNESS_ARRAYS, STIFFNESS_TENSOR_ARRAYS,
                        lame_coefficients)
from .modal import ModalPencil, MultiShiftModes
from .nonlinear_start import NonlinearStart, solve_from_start
from .pml2d import DEFAULT_PML_REFLECTION, check_ring, elastic_pml_ring, ring_wave_speed
from .renderer import SolidRenderer
from .solver import GENERALIZED_PLANE_STRAIN, GEOMETRIES, OUT_OF_PLANE_CONDITIONS, PLANE_STRAIN, PLANE_STRESS
from .solver_common import MATERIAL_STATE_ARRAYS, FEMSolidCommon, host_array as _host
from .stiffness import (ElementStiffness, check_positive_definite, element_stiffness, elastic_stress,
                        operator_coefficients, plane_stress_strain, thermal_coefficients)
from .system_fem import ElasticFEMSystem
from .tangent_update import TangentPolicy, newton_proposer

DISCRETIZATIONS ={'quadratic': 2, 'linear': 1}   # setting discretization -> Lagrange order (mesh plan D5)
_OPERATOR_NAMES = ('c11', 'c12', 'c22', 'c66', 'c16', 'c26')
_HOOP_NAMES = ('c13', 'c23', 'c33')
# Newton scale floor: the largest displacement counts at least this fraction of the mesh window extent (a zero solution)
_NEWTON_SCALE_FLOOR = 1e-12


class _ElementFields:
    """Per-element float32 outputs (element-centroid values, 0 off the solid) and the coupled / applied inputs, on the
    domain's array module: the state() arrays (shape (n_elements,), like a P0 field)."""

    def __init__(self, n_elements, xp):
        self.shape = (int(n_elements),)
        for name in OUTPUT_ARRAYS + INPUT_ARRAYS:
            setattr(self, name, xp.zeros(self.shape, dtype=DTYPE))


# output groups of the shown solution, each computed on its first read (_ensure_outputs, as in 3D): displacement
# (dof values and centroid interpolation, cheap) and strain / stress / energy (element points of the solid). An
# eigenfrequency study selects every mode for its snapshots, which mostly store u_abs only.
_PRIMARY_OUTPUTS = ('u_x', 'u_y', 'u_z', 'u_abs')
_OUTPUT_GROUPS = {'primary': _PRIMARY_OUTPUTS + HARMONIC_IMAG_FIELDS,
                  'derived': tuple(name for name in OUTPUT_ARRAYS if name not in _PRIMARY_OUTPUTS)
                  + ENERGY_DENSITY_FIELDS}


def _restricted(stiffness, elements) -> ElementStiffness:
    """ElementStiffness of mesh elements as (m, 1) columns (broadcast against (m, q) point values)."""
    return ElementStiffness({name: values[elements][:, None] for name, values in stiffness.values.items()},
                            np.asarray(stiffness.solid)[elements][:, None], stiffness.anisotropic, stiffness.monoclinic)


class SolidMechanicsFEMPhysics(FEMSolidCommon, MultiShiftModes, PhysicsSolver):
    """2D static elasticity on a fem.domain.TriangleDomain (window elements; the PML layer ring is void unless pml_faces
    make it an elastic PML, eigenfrequency only): host float64 solve, state() arrays per element (centroid values),
    display fields by fem_field (P1 / P2 nodal: displacement exact, strain / stress / energy densities smoothed by nodal
    averaging over the solid elements). azimuthal_order m >= 1 (axisymmetric, eigenfrequency only): the amplitudes
    (U_r, U_z, U_phi) of u_r, u_z ~ cos(m phi), u_phi ~ sin(m phi) (fields u_x, u_y, u_z). geometric_nonlinearity:
    every static solve total-Lagrangian (Green-Lagrange strain, second Piola-Kirchhoff stress outputs)."""

    _output_groups = _OUTPUT_GROUPS

    name = 'solid'
    material_schema = {**MECHANICAL_SCHEMA, **DENSITY_SCHEMA, **SHEAR_PHOTOELASTIC_SCHEMA}
    output_arrays = OUTPUT_ARRAYS
    input_arrays = INPUT_ARRAYS
    source_slots = SOURCE_SLOTS
    piezoelectric = False   # electrode features are tensor only (validation)
    guided = False          # no 2.5D modes on triangles (validation)
    warm_start_extrapolate = True   # a sweep extrapolates warm_start_fields over the last two values (model.study)

    def __init__(self, domain, linear_solver=None, eigen_solver=None, target_frequency: float = 0.0,
                 geometry: str = 'planar', out_of_plane: str = PLANE_STRAIN, thickness=None,
                 discretization: str = 'quadratic', axial_force: float = 0.0, bending: str = 'xy',
                 bending_moment_x: float = 0.0, bending_moment_y: float = 0.0, stress_stiffening: bool = False,
                 free_body=None, azimuthal_order: int = 0, pml_faces=(), pml_reflection: float = DEFAULT_PML_REFLECTION,
                 geometric_nonlinearity: bool = False, load_steps: int = 1, nonlinear_tolerance: float = 1e-8,
                 nonlinear_max_iterations: int = 25, tangent_update: str = 'when_slow'):
        # K_sigma of the static state in every pencil (new_modules_plan K2); free rigid motions allowed (None: target > 0)
        self.stress_stiffening = bool(stress_stiffening)
        self._free_body = None if free_body is None else bool(free_body)
        if geometry not in GEOMETRIES:
            raise ValueError(f"geometry must be one of {', '.join(GEOMETRIES)}, got {geometry!r}")
        if discretization not in DISCRETIZATIONS:
            raise ValueError(f"discretization must be one of {', '.join(DISCRETIZATIONS)}, got {discretization!r}")
        if bending not in BENDING_TERMS:
            raise ValueError(f"bending must be one of {', '.join(BENDING_TERMS)}, got {bending!r}")
        self._axisymmetric = geometry == 'axisymmetric'
        self.out_of_plane = PLANE_STRAIN if self._axisymmetric else out_of_plane
        if self.out_of_plane not in OUT_OF_PLANE_CONDITIONS:
            raise ValueError(f"out_of_plane must be one of {', '.join(OUT_OF_PLANE_CONDITIONS)}, got {out_of_plane!r}")
        # azimuthal order m of the axisymmetric solid (new_modules_plan K7): 0 = torsion-free rings, >= 1 three components
        if isinstance(azimuthal_order, bool) or int(azimuthal_order) != azimuthal_order or int(azimuthal_order) < 0:
            raise ValueError(f'azimuthal_order must be an integer >= 0, got {azimuthal_order!r}')
        self.azimuthal_order = int(azimuthal_order)
        if self.azimuthal_order and not self._axisymmetric:
            raise ValueError(f"azimuthal_order {self.azimuthal_order} needs geometry 'axisymmetric'")
        # elastic PML ring (pml2d, new_modules_plan K5): its faces and target reflection; the eigenfrequency study only
        self.pml_faces = tuple(pml_faces)
        unknown = [face for face in self.pml_faces if face not in FACES]
        if unknown:
            raise ValueError(f'pml_faces must be outer faces {FACES}, got {unknown}')
        if self.pml_faces and self._axisymmetric and 'x_lo' in self.pml_faces:
            raise ValueError('an axisymmetric solid has no PML on x_lo (the axis r = 0 lies there)')
        if self.pml_faces and self.out_of_plane == GENERALIZED_PLANE_STRAIN:
            raise ValueError('an elastic PML runs in the eigenfrequency study, which refuses generalized plane strain')
        self.pml_reflection = float(pml_reflection)
        # total-Lagrangian Newton of every static solve (new_modules_plan K6): load steps and the damped loop
        if isinstance(load_steps, bool) or int(load_steps) != load_steps or load_steps < 1:
            raise ValueError(f'load_steps must be an integer >= 1, got {load_steps!r}')
        self.geometric_nonlinearity = bool(geometric_nonlinearity)
        self.load_steps = int(load_steps)
        if self.geometric_nonlinearity:
            refused = (f'azimuthal order {self.azimuthal_order}' if self.azimuthal_order else
                       'an elastic PML' if self.pml_faces else
                       'generalized plane strain' if self.out_of_plane == GENERALIZED_PLANE_STRAIN else '')
            if refused:
                raise ValueError(f'geometric_nonlinearity is not available with {refused} (planar plane strain / '
                                 'plane stress or axisymmetric at azimuthal order 0 only)')
        self._newton = (DampedIteration(DampingSettings(tolerance=float(nonlinear_tolerance),
                                                        max_iterations=int(nonlinear_max_iterations)),
                                        lambda: f'{self._label()} geometrically nonlinear solve')
                        if self.geometric_nonlinearity else None)
        # tangent refactorised per Newton iteration or kept while the residual falls fast (plan G4, modified Newton)
        self._tangent_policy = TangentPolicy(tangent_update)
        # (converged every load step, warning or None, Newton iterations) of the last nonlinear solve
        self._nonlinear_record = None
        if thickness is not None and not (math.isfinite(float(thickness)) and float(thickness) > 0.0):
            raise ValueError(f'thickness must be None or a finite value > 0 [m], got {thickness!r}')
        self.thickness = None if thickness is None or self._axisymmetric else float(thickness)
        self.axial_force = float(axial_force)   # [N] net F_z on the drawn cross-section (generalized plane strain)
        self.bending_terms = BENDING_TERMS[bending]
        self.bending_moments = (float(bending_moment_x), float(bending_moment_y))   # [N m] about the drawing origin
        self.order = DISCRETIZATIONS[discretization]
        self.linear_solver = linear_solver   # 'linear' SparseDirectSolver runtime (None: system_fem makes a private one)
        self.target_frequency = float(target_frequency)   # [Hz] eigenfrequency study target (0 = the lowest modes)
        self._owns_eigen_solver = eigen_solver is None
        self.eigen_solver = ArpackShiftInvertSolver() if eigen_solver is None else eigen_solver
        self.eigen_solve = None          # modal.ModalSolve of the last solve_eigenfrequencies
        self.modal_mass = None           # host CSR consistent mass of that solve
        self.optomechanical_couplings = []   # optomechanics.OptomechanicalCoupling of the features (optomechanics study)
        self.selected_resonance = None   # 1-based mode shown, None = the static solve
        self.frequency_points = []       # frequency_response.FrequencyPoint per solved frequency (no electrodes)
        # (key, frequency_response.HarmonicSystem on the 'linear' runtime) of the driven solves, see _harmonic_system
        self._harmonic = None
        self.reduced_sweep = None        # its info dict
        self._field_policy = None        # fem.reduced_outputs field policy of that plan
        self._unpublished_frequency = None   # [Hz] solved without showing its fields (field policy)
        self.fields = None
        self.system = None
        self.prepare(domain)

    def _label(self):
        return f"solid mechanics '{self.name}'"

    # --- PhysicsSolver contract ---------------------------------------------

    def prepare(self, domain) -> None:
        """Register the material arrays, allocate the system and the per-element fields (clears supports, loads and
        probes; axisymmetric: the axis rule on x_lo, u_r = 0 at m = 0, u_z = 0 and u_phi = -u_r at m = 1, all held at
        m >= 2; an elastic PML: the ring's outer side clamped); assembly is deferred."""
        if getattr(domain, 'grid_kind', 'tensor') != 'triangles':
            raise ValueError(f'{self._label()}: SolidMechanicsFEMPhysics needs a triangle mesh (fem TriangleDomain); a '
                             'tensor grid uses SolidMechanicsPhysics')
        self.domain = domain
        domain.materials.add_properties(self.material_schema)
        self.system = ElasticFEMSystem(domain.mesh, self.order, self._axisymmetric, self._label, self.linear_solver,
                                       backend=domain.backend.name, azimuthal_order=self.azimuthal_order)
        # modes about a target > 0 or a frequency response: free rigid motions counted, static solves refused
        self.system.rigid_motions_allowed = (self.target_frequency > 0.0 if self._free_body is None
                                             else self._free_body)
        n_elements = domain.n_elements
        self.fields = _ElementFields(n_elements, domain.xp)
        self._window_elements = np.asarray(domain.window_elements(), dtype=bool)
        self._window_device = domain.xp.asarray(self._window_elements)
        if self.pml_faces:
            check_ring(domain.mesh, self.pml_faces, self._label)
        # elements that may carry stiffness: the window, plus an elastic PML ring (materials extruded into it)
        self._elastic_elements = np.ones(n_elements, dtype=bool) if self.pml_faces else self._window_elements
        self._pml_stiffness = None                   # host stretched K of the last assembly (PML pencils)
        self._energy_centroids = None                # {energy density: host per-element centroid values}
        self._element_dofs = None                    # P0 map of the input / material fields (fem_field)
        self._body_force = np.zeros((n_elements, COMPONENTS))   # applied [N/m^3] per element
        self._tractions = []                         # [(mesh edges, t_x, t_y)] of the features
        self._stiffness = None                       # host ElementStiffness of the last assembly (every element)
        self._drop_host_inputs()
        self._coefficients = None                    # host operator coefficients of that assembly (fem.elasticity)
        self._solid = None                           # host bool per element
        self._nodal = {}                             # output name -> host float64 full-mesh nodal values
        self._shows_mode = False                     # a vibration mode is shown: no reactions
        self._shows_harmonic = False                 # a frequency-response solution is shown: no reactions
        self._harmonic_operators = None              # host (loss or None, mass) of the frequency response
        self._unit_axial = None                      # generalized plane strain (terms, loads, solutions, matrix)
        self._axial_terms = np.zeros(3)              # its (e0, e1, e2) about the window centre
        x0, y0, x1, y1 = domain.mesh.window
        self._term_geometry = SimpleNamespace(reference=(0.5 * (x0 + x1), 0.5 * (y0 + y1)))
        self._loads_dirty = True
        self.probes = []
        self._output_prestress = True                # the shown stress includes the initial stress
        self._output_green_lagrange = False          # shown strains are Green-Lagrange (nonlinear static)
        self._pending_outputs = set()                # _OUTPUT_GROUPS of the shown solution not computed yet
        # (with_loads, K_sigma host CSR or None, text, the nonlinear tangent K_T host CSR or None)
        self._prestress = None
        self._complex_displacement = None            # host (n_dofs, 2) of a shown frequency response (solid map)
        self._imag_centroids = None                  # {u_x_im, u_y_im: host per-element centroid Im u} of it
        self.frequency_probes = []                   # (HarmonicProbeRecord, mesh node)
        self._supports = []                          # SupportRegion of the support features
        self._harmonic_reactions = {}                # support name -> HarmonicProbeRecord (F_x, F_y)
        self._record_rows = None                     # (dof map, probe rows, support rows): _record_layout
        self._nonlinear_start = NonlinearStart()     # Newton start: sweep seed, last converged state (solid map)
        self._full_seed = None                       # a sweep seed on full-mesh dof rows, mapped by the next solve
        self._tangent_assembler = None               # fem.elasticity.TangentAssembler of the last assembly (lazy)
        if self._axisymmetric:   # the axis rule on the x_lo window edge (r = 0)
            self._hold_axis()
        mesh = domain.mesh
        for face in self.pml_faces:   # the ring's outer side is clamped
            edges = np.asarray(mesh.edge_tags.get(face, ()), dtype=np.int64)
            self.system.hold(self.system.global_ids_of_edges(edges), 0.0, 0.0, 0.0)

    def _hold_axis(self):
        """Regularity on the axis (x_lo): m = 0 u_r = 0; m = 1 u_z = 0 and u_phi = -u_r (tied); m >= 2 every
        amplitude 0."""
        if self.azimuthal_order == 0:
            self.hold_face('x_lo', u_x=0.0)
        elif self.azimuthal_order == 1:
            self.hold_face('x_lo', u_y=0.0)
            system = self.system
            system.tie_axis(system.global_ids_of_edges(system.face_edges('x_lo')))
        else:
            self.hold_face('x_lo', 0.0, 0.0, u_z=0.0)

    def finalize(self) -> None:
        """Assemble once (fails early on a solid region that can move as a rigid body)."""
        self._assemble()

    def materials_changed(self, names) -> None:
        names = set(names)
        self._ensure_outputs()   # pending outputs belong to the inputs of their solve
        self._drop_host_inputs()
        if STIFFNESS_ARRAYS & names:
            self.system.operator_dirty = True
            self._loads_dirty = True   # the thermal load scales with the stiffness
        if (EXPANSION_ARRAYS | PRESTRESS_NAMES) & names:
            self._loads_dirty = True
        if {*LOSS_ARRAYS, *DENSITY_SCHEMA} & names:
            self._harmonic_operators = None
        if self.pml_faces and set(DENSITY_SCHEMA) & names:   # the PML strength follows the ring's wave speed
            self.system.operator_dirty = True
        if names:
            self._prestress = None

    def _eigen_only(self) -> str:
        """Why the solid runs in the eigenfrequency study only ('' when not): an elastic PML or an azimuthal order."""
        if self.pml_faces:
            return 'an elastic PML'
        if self.azimuthal_order:
            return f'azimuthal order {self.azimuthal_order}'
        return ''

    def step(self, t: float = 0.0, dt: float = math.inf) -> None:
        """Re-solve only when the stiffness, a support, a load or an input changed (always at the first call); probes
        sample after a re-solve and at every dt = inf call. ValueError with an elastic PML or an azimuthal order (the
        eigenfrequency study only)."""
        only = self._eigen_only()
        if only:
            raise ValueError(f'{self._label()}: {only} is solved by the eigenfrequency study only (no stationary or '
                             'time-dependent solve)')
        t = 0.0 if t is None else float(t)
        system = self.system
        re_solve = system.operator_dirty or self._loads_dirty or system.n_solves == 0
        if re_solve:
            self._static_solve()
        if re_solve or dt is None or not math.isfinite(dt):
            self._record_probes(t)

    def _static_solve(self):
        """The static solve of every load, prescribed displacement, initial stress and input (assembling first when
        dirty; an elastic PML ring is an unstretched clamped elastic continuation here); ValueError for a free body and
        an azimuthal order."""
        system = self.system
        if self.azimuthal_order:
            raise ValueError(f'{self._label()}: azimuthal order {self.azimuthal_order} has no static solve (its loads '
                             'would be cos(m phi) parts; the eigenfrequency study only)')
        if system.operator_dirty:
            self._assemble()
        if system.n_rigid_motions:
            raise ValueError(f'{self._label()}: {system.n_rigid_motions} rigid motion(s) of the solid are free '
                             '(supports missing; allowed only for the vibration modes about a target_frequency > 0 and '
                             'for a frequency response), so a static solve is not available: add supports that stop '
                             'every rigid motion')
        if self.geometric_nonlinearity:
            displacement, residual = self._nonlinear_state(with_loads=True)
            system.accept_solution(displacement, self._external_load(), residual=residual)
        else:
            system.solve(self._total_load())
            if self.out_of_plane == GENERALIZED_PLANE_STRAIN:
                self._solve_generalized()
        self._shows_mode = self._shows_harmonic = False
        self.selected_resonance = None
        self._write_outputs(green_lagrange=self.geometric_nonlinearity)
        self._loads_dirty = False

    # --- geometric nonlinearity (total-Lagrangian Newton, new_modules_plan K6) ----

    def _fixed_stress(self, thermal=True):
        """fem.elasticity.StressField of the stress that is no strain response, in the fem Voigt rows: sigma0 (initial
        stress) - beta dT (thermal; plane stress: reduced beta); None when both are 0."""
        initial = self._initial_stress()
        prescribed = None if initial is None else initial[:, self._stress_rows()]
        thermal_stress = None
        if thermal:
            values = self._thermal_rows()
            thermal_stress = values if values.any() else None
        if prescribed is None and thermal_stress is None:
            return None
        return StressField(prescribed=prescribed, thermal=thermal_stress)

    def _nonlinear_state(self, with_loads=True):
        """(displacement (n_dofs, 2), residual f_int - f_ext (2 n_dofs,)) of the geometrically nonlinear static state on
        the solid map: Newton on f_int(u) = int (B + B_NL)^T S dV = f_ext (S = C E(u) + sigma0 - beta dT) over load_steps
        increments of the dead loads and held values (with_loads False: no loads, holds at 0, sigma0 - beta dT alone);
        each step the damped loop of core.damped_iteration on the fixed pattern of fem.elasticity.TangentAssembler
        (tangent_update: a factorisation per iteration, or kept while the residual falls fast; newton_proposer). Start
        (nonlinear_start): a sweep seed or the last converged state of this assembly, ramped from its load and held
        values (the cold ramp from u = 0 when it fails), else u = 0."""
        system = self.system
        dofs = system.dofs
        n_vector = COMPONENTS * dofs.n_dofs
        fixed_stress = self._fixed_stress(thermal=True)   # sigma0 and the thermal strain are static in both kinds
        external = self._external_load() if with_loads else np.zeros(n_vector)
        target = system.fixed_values() if with_loads else np.zeros(n_vector)
        held = system.held[dofs.global_ids].reshape(-1)
        assembler = self._tangent_assembler_2d()
        x0, y0, x1, y1 = self.domain.mesh.window
        floor = _NEWTON_SCALE_FLOOR * max(x1 - x0, y1 - y0)
        scale = np.array(floor)   # max |proposal| of the latest Newton step: updated in place by propose
        latest = {}               # 'state', 'force': the displacement of the latest internal force and that force
        policy = self._tangent_policy
        policy.reset()            # the runtime may have factorised K or a pencil since the last Newton solve
        propose_for = newton_proposer(policy, assembler, system, fixed_stress, held, system.free, scale, floor, latest)
        self._take_full_seed()
        state, self._nonlinear_record = solve_from_start(self._nonlinear_start, with_loads, self._newton,
                                                         self.load_steps, propose_for, scale, external, target,
                                                         on_cold_ramp=policy.reset)
        policy.reset()            # release the held tangent values
        if latest and np.array_equal(latest['state'], state):
            force = latest['force']
        else:
            force = assembler.force(state, initial_stress=fixed_stress)
        return state.reshape(-1, COMPONENTS), force - external

    def _tangent_assembler_2d(self):
        """fem.elasticity.TangentAssembler of the last assembly (solid map, coefficients, free set), built on first
        use."""
        if self._tangent_assembler is None:
            self._tangent_assembler = TangentAssembler(self.system.dofs, self._coefficients,
                                                       axisymmetric=self._axisymmetric, free=self.system.free)
        return self._tangent_assembler

    def warm_start_fields(self) -> dict:
        """WarmStartSolver: host copies of the last converged nonlinear state on the full-mesh dof rows (n_global, 2):
        u, its external load and held values (0 off the solid map), and its kind 'with_loads'; {} without one (a linear
        solid's direct solve needs no start)."""
        last = self._nonlinear_start.last
        if last is None:
            return {}
        with_loads, state, external, held = last
        system = self.system
        rows = system.dofs.global_ids

        def full_rows(vector):
            values = np.zeros((system.n_global, system.components))
            values[rows] = vector.reshape(-1, system.components)
            return values

        return {'u': full_rows(state), 'external': full_rows(external), 'held': full_rows(held),
                'with_loads': with_loads}

    def seed_warm_start(self, fields) -> tuple:
        """WarmStartSolver: warm_start_fields() of another build start the next nonlinear solve of their kind (mapped
        onto the solid map of that solve; another mesh is ignored with a note); no state slot is seeded."""
        if fields and self.geometric_nonlinearity:
            self._full_seed = dict(fields)
        return ()

    def _take_full_seed(self):
        """Hand a pending full-mesh sweep seed to the NonlinearStart on the current solid map."""
        seed, self._full_seed = self._full_seed, None
        if seed is None:
            return
        system = self.system
        shape = (system.n_global, system.components)
        if any(np.shape(seed[name]) != shape for name in ('u', 'external', 'held')):
            self._nonlinear_start.note = f'the sweep seed has {np.shape(seed["u"])} dof rows, this mesh {shape}: ignored'
            return
        rows = system.dofs.global_ids
        self._nonlinear_start.set_seed(seed.get('with_loads', True),
                                       *(np.asarray(seed[name], dtype=np.float64)[rows].reshape(-1)
                                         for name in ('u', 'external', 'held')))

    # --- prestress (stress stiffening, new_modules_plan K2) -------------------------

    def solve_static_state(self, with_loads=True) -> str:
        """PrestressSolver hook: the static state whose K_sigma (int grad N . sigma grad N dA per component, axisymmetric
        + the hoop term) joins every later pencil until the next assembly / material / input change; with_loads: every
        load (the step's static solve, shown afterwards), else the initial stress and thermal strain alone with every
        hold at 0 (a frequency response, whose loads are its drive; dT is static, never a drive). A free body or an
        azimuthal order m >= 1: the initial stress alone, unrelaxed. An elastic PML ring: the static state with the ring
        as a clamped elastic continuation, K_sigma of the ring stretched. Geometric nonlinearity: the pencil stiffness
        is the tangent K_T of the nonlinear static state instead."""
        system = self.system
        if system.operator_dirty:
            self._assemble()
        initial = self._initial_stress()
        rows = self._stress_rows()
        refused = self._static_refused()
        tangent = None
        if refused:
            stress = None if initial is None else StressField(prescribed=initial[:, rows])
            text = (f'stress stiffening: {refused} has no static solve, K_sigma of the initial stress alone'
                    if initial is not None else f'stress stiffening: {refused} without initial stress, K_sigma = 0')
        elif with_loads and self.geometric_nonlinearity:
            self._static_solve()
            stress = None
            tangent = self._tangent_assembler_2d().tangent(system.unknown,
                                                           initial_stress=self._fixed_stress(thermal=True))
            text = ('geometric nonlinearity: the pencil stiffness is the tangent K_T of the nonlinear static state '
                    f'(every load, initial stress and thermal strain; {self._nonlinear_text()})')
        elif with_loads:
            self._static_solve()
            stress = StressField(prescribed=None if initial is None else initial[:, rows], material=self._coefficients,
                                 displacement=system.unknown.copy(), thermal=self._thermal_rows())
            text = 'stress stiffening: K_sigma of the static state (every load, initial stress and thermal strain)'
            if system.pml is not None:
                text += '; the PML ring solved as a clamped elastic continuation'
        elif (background := self._fixed_stress(thermal=True)) is None:
            stress = None
            text = ('stress stiffening: no initial stress or thermal strain (the loads are the harmonic drive), '
                    'K_sigma = 0')
        elif self.geometric_nonlinearity:
            displacement, _ = self._nonlinear_state(with_loads=False)
            stress = None
            tangent = self._tangent_assembler_2d().tangent(displacement, initial_stress=background)
            text = ('geometric nonlinearity: the pencil stiffness is the tangent K_T of the nonlinear state of the '
                    f'initial stress and thermal strain (holds at 0, loads are the drive; {self._nonlinear_text()})')
        else:
            relaxed = system.solve_homogeneous(self._total_load(dead_loads=False))
            stress = StressField(prescribed=background.prescribed, material=self._coefficients,
                                 displacement=relaxed.reshape(-1, COMPONENTS), thermal=background.thermal)
            text = ('stress stiffening: K_sigma of the relaxed initial stress and thermal strain (holds at 0, loads '
                    'are the drive)')
        geometric = None if stress is None else system.geometric_stiffness(stress)
        self._prestress = (bool(with_loads), geometric, text, tangent)
        if not refused:   # the pencil / harmonic factorisations follow: free the static factor (cuDSS: device memory)
            system.release_factor()
        return text

    def _static_refused(self) -> str:
        """Why the stress-stiffened pencils take the initial stress alone ('' when a static state exists): an azimuthal
        order m >= 1 or free rigid motions (call after an assembly)."""
        if self.azimuthal_order:
            return f'azimuthal order {self.azimuthal_order}'
        if self.system.n_rigid_motions:
            return f'a free body ({self.system.n_rigid_motions} free rigid motions)'
        return ''

    def _pencil_stiffness(self, with_loads):
        """Host K of the pencils: the system operator (an elastic PML: the stretched one), + K_sigma of the static state
        (stress_stiffening; solved here when none is kept for this study kind); geometric nonlinearity: the tangent K_T
        of the nonlinear state."""
        system = self.system
        if system.pml is None:
            base = system.operator
        else:
            if self._pml_stiffness is None:
                self._pml_stiffness = system.stiffness(self._coefficients, stretched=True)
            base = self._pml_stiffness
        if not self.prestressed:
            return base
        if self._prestress is None or self._prestress[0] != with_loads:
            self.solve_static_state(with_loads)
        _, geometric, _, tangent = self._prestress
        if tangent is not None:
            return tangent
        return base if geometric is None else pattern_sum([base, geometric])

    def _stress_rows(self):
        """Columns of _initial_stress() in the fem.elasticity Voigt rows: planar (xx, yy, xy), axisymmetric (rr = xx,
        zz = yy, hoop = zz, rz = xy)."""
        return [0, 1, 3, 2] if self._axisymmetric else [0, 1, 2]

    def _initial_stress(self):
        """Host (n_elements, 4) initial stress [Pa] (PRESTRESS_ARRAYS order xx, yy, xy, zz) on the solid elements (0
        elsewhere; cached, read-only); None when no material sets one. ValueError for generalized plane strain, and for
        a nonzero zz under plane stress."""
        return self._host_input('initial_stress', self._read_initial_stress)

    def _read_initial_stress(self):
        materials = self.domain.materials
        names = set(materials.property_names())
        if not names.intersection(PRESTRESS_ARRAYS):
            return None
        stress = np.zeros((self.domain.n_elements, len(PRESTRESS_ARRAYS)))
        for column, name in enumerate(PRESTRESS_ARRAYS):
            if name in names:
                stress[:, column] = np.where(self._solid, _host(getattr(materials, name)), 0.0)
        if not stress.any():
            return None
        if self.out_of_plane == GENERALIZED_PLANE_STRAIN:
            raise ValueError(f'{self._label()}: initial stress is not available with generalized plane strain (use '
                             'plane_strain, plane_stress or axisymmetric)')
        if self.out_of_plane == PLANE_STRESS and stress[:, 3].any():
            raise ValueError(f'{self._label()}: plane stress holds sigma_zz = 0, so initial_stress_zz must be 0')
        return stress

    def _thermal_rows(self):
        """(n_elements, k) beta dT of the thermal load in the fem.elasticity Voigt rows (plane stress: reduced beta)."""
        beta_x, beta_y, beta_z, beta_xy = self._thermal_coefficients(plane_stress=self.out_of_plane == PLANE_STRESS)
        temperature_change = self._temperature_change()
        rows = (beta_x, beta_y, beta_z, beta_xy) if self._axisymmetric else (beta_x, beta_y, beta_xy)
        return np.where(self._solid[:, None], np.stack([_host(row) * temperature_change for row in rows], axis=1), 0.0)

    def state(self) -> dict:
        """Per-element arrays (n_elements,): the outputs at the element centroids, the inputs and E / nu."""
        self._ensure_outputs()
        state = {name: getattr(self.fields, name) for name in self.output_arrays + self.input_arrays}
        state.update({name: getattr(self.domain.materials, name) for name in MATERIAL_STATE_ARRAYS})
        return state

    def inject_source(self, name: str, array) -> None:
        """Write a coupled slot: one value per element (a triangle target of a mesh transfer, or this mesh's own)."""
        if name not in self.source_slots:
            raise ValueError(f"{self._label()} accepts sources {', '.join(self.source_slots)}; got '{name}'")
        self._ensure_outputs()   # pending outputs belong to the inputs of their solve (the temperature change)
        self._drop_host_inputs()
        xp = self.domain.xp
        values = xp.asarray(array, dtype=DTYPE).reshape(-1)
        target = getattr(self.fields, name)
        if values.size != target.size:
            raise ValueError(f"{self._label()}: source '{name}' needs one value per element ({target.size}), got "
                             f'{values.size}')
        xp.copyto(target, values)
        self._loads_dirty = True
        self._prestress = None

    # --- properties --------------------------------------------------------------

    @property
    def axisymmetric(self) -> bool:
        return self._axisymmetric

    @property
    def ring_angle(self) -> float:
        """Angle weight of the (r, z) rings: 2 pi, pi for an azimuthal order m >= 1 (core.metric.ring_angle)."""
        return ring_angle(self.azimuthal_order)

    @property
    def force_unit(self) -> str:
        return 'N' if self._axisymmetric or self.thickness is not None else 'N/m'

    @property
    def axial_strain_terms(self) -> tuple:
        """(e0, e1, e2) of eps_zz = e0 + e1 x + e2 y [1, 1/m, 1/m] (x, y of the drawing) of the last generalized
        plane-strain solve (zeros otherwise)."""
        return terms_about_origin(self._term_geometry, self._axial_terms)

    def work_counters(self) -> dict:
        """The private eigen solver's counters once it ran; the private linear runtime's when no build runtime counts
        them (scripts; also the frequency-response factorisations, which run on it); the Newton loop of a geometrically
        nonlinear solid (its tangent factorisations count on the 'linear' runtime too; tangent_factorizations / _reuses
        split the Newton iterations)."""
        counters = {}
        if self._owns_eigen_solver and self.eigen_solver.n_eigen_solves:
            counters.update(self.eigen_solver.work_counters())
        if self.linear_solver is None and self.system.runtime.n_factorizations:
            counters.update({f'fem_{name}': value for name, value in self.system.linear_solver.work_counters().items()
                             if name != 'back_end'})
        if self._newton is not None and self._newton.n_runs:   # load steps run, Newton iterations, tangent factors
            counters.update(self._newton.work_counters())
            counters.update(self._tangent_policy.counters())
        return counters

    def solve_summary(self) -> str:
        summary = self.system.solve_summary()
        if self.geometric_nonlinearity and summary:
            summary += f'; geometrically nonlinear: {self._nonlinear_text()}'
        return summary

    def renderer(self):
        return SolidFEMRenderer(self)

    # --- supports, loads, probes ----------------------------------------------

    def hold_face(self, face: str, u_x=None, u_y=None, u_z=None):
        """Hold u_x and / or u_y [m] (None = free) on every dof of an outer face of the window (u_z: only None or 0;
        it holds U_phi of an azimuthal order m >= 1 and is ignored otherwise); returns the held dof mask."""
        if face not in FACES:
            raise ValueError(f"unknown face {face!r}; expected one of {FACES}")
        if u_z not in (None, 0.0):
            raise ValueError(f'hold_face: u_z can only be held at 0, got {u_z!r}')
        system = self.system
        return system.hold(system.global_ids_of_edges(system.face_edges(face)), u_x, u_y, u_z)

    def hold_cells(self, mask, u_x=0.0, u_y=0.0, u_z=0.0):
        """Hold components [m] (None = free; u_z = u_phi of an azimuthal order, only None or 0) on every dof of the
        window elements under the element mask (n_elements,)."""
        if u_z not in (None, 0.0):
            raise ValueError(f'hold_cells: u_z can only be held at 0, got {u_z!r}')
        system = self.system
        return system.hold(system.global_ids_of_elements(self._element_mask(mask, 'fixed domain')), u_x, u_y, u_z)

    def add_face_traction(self, face: str, t_x: float, t_y: float) -> None:
        """Add a traction (t_x, t_y) [Pa] on an outer face of the window (consistent edge loads)."""
        if face not in FACES:
            raise ValueError(f"unknown face {face!r}; expected one of {FACES}")
        self.add_edge_traction(self.system.face_edges(face), t_x, t_y)

    def add_edge_traction(self, edges, t_x: float, t_y: float) -> None:
        """Add a traction (t_x, t_y) [Pa] on mesh edges (a line entity or outer face): int t N ds per dof, so the total
        is t times the length (axisymmetric: the ring area); edges off the solid add nothing. An entry on the same edges
        takes the sum (a reused sweep build re-adds changes: the list never grows per value)."""
        edges = np.asarray(edges, dtype=np.int64)
        for index, (entry_edges, entry_x, entry_y) in enumerate(self._tractions):
            if entry_edges is edges or np.array_equal(entry_edges, edges):
                self._tractions[index] = (entry_edges, entry_x + float(t_x), entry_y + float(t_y))
                break
        else:
            self._tractions.append((edges, float(t_x), float(t_y)))
        self._loads_dirty = True

    def set_held_values(self, dof_mask, u_x=None, u_y=None, u_z=None) -> None:
        """Set the held values [m] (None = unchanged) on a hold's dof mask (n_global, components) in place: the held set
        and the assembly stay (a reused sweep build's prescribed displacement)."""
        mask = np.asarray(dof_mask, dtype=bool)
        held_values = self.system.held_values
        for component, value in tuple(enumerate((u_x, u_y, u_z)))[:mask.shape[1]]:
            if value is not None:
                held_values[mask[:, component], component] = float(value)
        self._loads_dirty = True
        if self.prestressed:   # K_sigma / K_T of the static state follow the held values
            self._prestress = None

    def reset_records(self) -> None:
        """RerunSolver (reused sweep build): clear the probe and frequency records, drop a kept static state (prestress)
        and mark the loads dirty, so the next study run solves afresh; the last converged nonlinear state stays the
        Newton start."""
        for probe in self.probes:
            probe.clear()
        self.reset_frequency_response()
        if self.prestressed:
            self._prestress = None
        self._loads_dirty = True

    def add_body_force(self, mask, f_x, f_y) -> None:
        """Add a body force density (f_x, f_y) [N/m^3] (numbers or per-element arrays) on the elements under mask;
        void elements drop it."""
        elements = self._element_mask(mask, 'body force')
        for component, value in enumerate((f_x, f_y)):
            self._body_force[:, component] += np.where(elements, self._element_values(value, 'body force'), 0.0)
        self._loads_dirty = True

    def add_node_probe(self, node: int, component: str = 'u_abs') -> FieldProbe:
        """Record an output field or energy density at mesh node `node` (its nodal value) whenever step records
        probes."""
        if component not in OUTPUT_ARRAYS + ENERGY_DENSITY_FIELDS:
            raise ValueError(f"probe component {component!r} is not one of "
                             f"{', '.join(OUTPUT_ARRAYS + ENERGY_DENSITY_FIELDS)}")
        if not 0 <= int(node) < self.domain.mesh.n_points:
            raise ValueError(f'solid probe node {node} is not a mesh node')
        probe = FieldProbe(int(node), 0, component)
        self.probes.append(probe)
        return probe

    def reaction(self, region) -> tuple:
        """Force (F_x, F_y) the support of region exerts on the solid in the last static solve (N/m, N with the
        thickness setting or axisymmetric)."""
        if self._shows_mode or self._shows_harmonic:
            shown = 'a vibration mode' if self._shows_mode else 'a frequency response (complex reactions)'
            raise RuntimeError(f'{self._label()}: reactions need a static solve ({shown} is shown)')
        force_x, force_y = self.system.reaction(region.dof_mask)
        depth = 1.0 if self.thickness is None else self.thickness
        return force_x * depth, force_y * depth

    def error_indicator(self):
        """eta^2 per element (float64 (n_elements,)) of the last static solve for mesh refinement (S7.7, mesh plan
        A3): Kelly traction jumps h_E / c int_E |sigma_0 n - sigma_1 n - t|^2 ds (1/2 per side of an edge between solid
        elements; free and traction edges of the solid one-sided, components held along the whole edge skipped), c =
        (c11 + c22) / 2 of the element; 2 pi r weights axisymmetric; nominal tractions after a nonlinear solve.
        ValueError without a static solve."""
        system, mesh = self.system, self.domain.mesh
        if self._shows_mode or self._shows_harmonic or system.n_solves == 0:
            raise ValueError(f'{self._label()}: the error indicator needs a static solve (not a vibration mode or a '
                             'frequency response)')
        solid = self._solid
        diagonal = 0.5 * (_host(self._stiffness['c11']) + _host(self._stiffness['c22']))
        compliance = np.where(solid & (diagonal > 0.0), 1.0 / np.where(diagonal > 0.0, diagonal, 1.0), 0.0)
        applied = np.zeros((mesh.n_edges, COMPONENTS))
        for edges, t_x, t_y in self._tractions:   # an edge listed twice in one traction counts twice
            applied += scatter_rows(mesh.n_edges, edges, np.broadcast_to([t_x, t_y], (np.size(edges), COMPONENTS)))
        eta_squared = np.zeros(mesh.n_elements)
        candidates = np.unique(mesh.element_edges[solid])
        for start in range(0, candidates.size, CHUNK_ITEMS):
            sides = edge_sides(mesh, candidates[start:start + CHUNK_ITEMS], solid, system.dofs.geometry_order)
            jump = self._edge_tractions(sides, 0, np.arange(len(sides.edges)))
            two_sided = np.flatnonzero(sides.two_sided)
            if two_sided.size:
                jump[two_sided] -= self._edge_tractions(sides, 1, two_sided)
            jump -= applied[sides.edges][:, None, :]
            ends = mesh.edges[sides.edges].astype(np.int64)
            ids = ends if self.order == 1 else np.concatenate([ends, mesh.n_points + sides.edges[:, None]], axis=1)
            held = system.held[np.searchsorted(system.full_dofs.global_ids, ids)].all(axis=1)   # (k, 2)
            jump *= ~(held & ~sides.two_sided[:, None])[:, None, :]
            weights = sides.weights * (2.0 * math.pi * sides.points[..., 0] if self._axisymmetric else 1.0)
            add_edge_terms(eta_squared, sides, np.sum(np.sum(jump ** 2, axis=-1) * weights, axis=1), compliance)
        return eta_squared

    def _edge_tractions(self, sides, side, rows):
        """(k, q, 2) traction sigma n (n out of side 0, thermal stress included) at the edge points of `rows` from the
        element on `side`; after a nonlinear static solve the nominal traction F S n of the reference edge (Green-Lagrange
        strain, second Piola-Kirchhoff stress S, F = I + H)."""
        dofs, displacement = self.system.dofs, self.system.unknown
        elements = sides.elements[rows, side]
        reference = sides.reference_points(side, rows)
        values_x, gradients_x = lagrange_at(dofs, displacement[:, 0], elements, reference)
        _, gradients_y = lagrange_at(dofs, displacement[:, 1], elements, reference)
        h_xx, h_xy, h_yx, h_yy = gradients_x[..., 0], gradients_x[..., 1], gradients_y[..., 0], gradients_y[..., 1]
        nonlinear = self._output_green_lagrange
        columns = ([h_xx + 0.5 * (h_xx ** 2 + h_yx ** 2), h_yy + 0.5 * (h_xy ** 2 + h_yy ** 2)] if nonlinear
                   else [h_xx, h_yy])
        if self._axisymmetric:   # hoop u_r / r, on the axis its limit d u_r / d r
            radius = sides.points[rows][..., 0]
            x0, y0, x1, y1 = self.domain.mesh.window
            on_axis = radius <= AXIS_TOLERANCE * max(x1 - x0, y1 - y0)
            hoop = np.where(on_axis, h_xx, values_x / np.where(on_axis, 1.0, radius))
            columns.append(hoop + 0.5 * hoop ** 2 if nonlinear else hoop)
        columns.append(h_xy + h_yx + (h_xx * h_xy + h_yx * h_yy if nonlinear else 0.0))
        outputs = self._point_outputs(elements, np.stack(columns, axis=-1), thermal=True, points=sides.points[rows],
                                      prestress=True)
        normal = sides.normals[rows]
        traction_x = outputs['stress_xx'] * normal[..., 0] + outputs['stress_xy'] * normal[..., 1]
        traction_y = outputs['stress_xy'] * normal[..., 0] + outputs['stress_yy'] * normal[..., 1]
        if nonlinear:   # P n = (I + H) S n
            traction_x, traction_y = (traction_x + h_xx * traction_x + h_xy * traction_y,
                                      traction_y + h_yx * traction_x + h_yy * traction_y)
        return np.stack([traction_x, traction_y], axis=-1)

    # --- eigenfrequency study hooks -------------------------------------------

    def solve_eigenfrequencies(self, n_resonances):
        """Up to n_resonances vibration modes nearest target_frequency: modal.solve_modes on K (+ K_sigma with
        stress_stiffening) and the consistent mass of the solid elements (supports held at 0; a mechanical_loss_factor:
        the damped modes of K_el (1 + j eta), solve_damped_modes; an elastic PML: the stretched complex pencil, PML modes
        filtered, radiation Q; a free body: its rigid motions asked on top and dropped; azimuthal order 1: on the axis
        tie prolongation) -> ModalSolve, kept as eigen_solve. solve_eigenfrequencies_targets / _band
        (modal.MultiShiftModes) share the _modal_pencil."""
        self.eigen_solve = None
        pencil, _ = self._modal_pencil(self.target_frequency)
        try:
            self.eigen_solve = pencil.solve_at(2.0 * math.pi * self.target_frequency, n_resonances)
        finally:
            pencil.release()
        return self.eigen_solve

    def _modal_pencil(self, lowest_target):
        """(modal.ModalPencil of solve_eigenfrequencies' pencil, identity finish); lowest_target [Hz] (None: a band) <=
        0 with free rigid motions raises."""
        system = self.system
        if system.operator_dirty:
            self._assemble()
        if system.n_rigid_motions and lowest_target is not None and lowest_target <= 0.0:
            raise ValueError(f'{self._label()}: {system.n_rigid_motions} rigid motion(s) are free and the target '
                             'frequency is 0, where K - 0 M is singular: set the target > 0 or add supports')
        self.modal_mass = system.mass(self._solid_density('an eigenfrequency study'))
        stiffness = self._pencil_stiffness(with_loads=True)
        loss = self._stiffness_loss()
        components = system.components
        if loss is not None or system.pml is not None:   # complex pencil: K_el (1 + j eta) (K_sigma lossless), PML
            if loss is not None:
                stiffness = pattern_sum([stiffness.astype(np.complex128), 1j * loss])
            pencil = ModalPencil(stiffness, self.modal_mass, system.free, components, self.eigen_solver,
                                 prolongation=system.prolongation, rigid_motions=system.n_rigid_motions, damped=True,
                                 shell_dofs=None if system.pml is None else system.ring_only_dofs(),
                                 pml_text='' if system.pml is None else system.pml.text(),
                                 loss_text='' if loss is None else 'mechanical_loss_factor')
        else:
            pencil = ModalPencil(stiffness, self.modal_mass, system.free, components, self.eigen_solver,
                                 prolongation=system.prolongation, rigid_motions=system.n_rigid_motions)
        return pencil, lambda solve: solve

    def set_axial_wavenumber(self, wavenumber):
        """Guided (2.5D) k_z of the optomechanics study: not available on a triangle mesh (ValueError)."""
        raise ValueError(f'{self._label()}: no guided (2.5D) modes on a triangle mesh')

    def modal_overlap_data(self):
        """Host data the g0 overlap (optomechanics_fem) reads after an eigenfrequency solve: kind 'triangles', the solid
        Lagrange map and components of the modes, solid elements, p11 / p12 / p44 per element and the consistent mass."""
        if self.eigen_solve is None or self.modal_mass is None:
            raise ValueError(f'{self._label()}: no eigenfrequency solve yet')
        materials = self.domain.materials
        return {'kind': 'triangles', 'axisymmetric': self._axisymmetric, 'azimuthal_order': self.azimuthal_order,
                'dofs': self.system.dofs, 'components': self.system.components,
                'solid': np.array(self._solid, dtype=bool),
                'p11': _host(materials.photoelastic_p11), 'p12': _host(materials.photoelastic_p12),
                'p44': _host(materials.photoelastic_p44), 'mass': self.modal_mass, 'axial_wavenumber': None}

    def select_resonance(self, number):
        """Show mode `number` (1-based): displacement, strain and the modal stress (no thermal part, no initial stress)
        of the peak-normalised mode shape and its time-averaged energy densities (a damped complex mode: its real part,
        u_abs = |u|)."""
        resonance = self.eigen_solve.resonances[number - 1]
        mode = np.asarray(resonance.displacement).reshape(-1, self.system.components)
        if np.iscomplexobj(mode):
            self._show_harmonic(mode, thermal=False, omega=resonance.omega)
        else:
            self.system.unknown = np.array(mode, dtype=np.float64)
            self._complex_displacement = None
            self._write_outputs(thermal=False, omega=resonance.omega)
        self._shows_mode, self._shows_harmonic = True, False
        self.selected_resonance = number

    def _stiffness_loss(self, stretched=True):
        """Host CSR eta K of the solid elements (mechanical_loss_factor of the kept operator coefficients; an elastic
        PML ring stretched as K unless stretched is False); None when every eta is 0."""
        materials = self.domain.materials
        if LOSS_ARRAYS[0] not in set(materials.property_names()):
            return None
        eta = np.where(self._solid, _host(getattr(materials, LOSS_ARRAYS[0])), 0.0)
        if not eta.any():
            return None
        scaled = {name: np.asarray(values, dtype=np.float64) * eta for name, values in self._coefficients.items()}
        return self.system.stiffness(scaled, stretched=stretched and self.system.pml is not None)

    # --- frequency_response study hooks -----------------------------------------

    def reset_frequency_response(self):
        """Forget the FrequencyPoints, frequency probe and support reaction records of an earlier frequency_response run
        (the study calls it first)."""
        self.frequency_points = []
        for record, _ in self.frequency_probes:
            record.clear()
        for record in self._harmonic_reactions.values():
            record.clear()
        self.reduced_sweep = None
        self._field_policy, self._unpublished_frequency = None, None
        self._record_rows = None
        if self._harmonic is not None:
            self._harmonic[1].clear_reduced()

    def _harmonic_inputs(self):
        """(load, loss, mass) of a driven solve: the checks, assembly and harmonic operators."""
        if self.out_of_plane == GENERALIZED_PLANE_STRAIN:
            raise ValueError(f'{self._label()}: generalized plane strain has no frequency response (the uniform eps_zz '
                             'carries no mass)')
        only = self._eigen_only()
        if only:
            raise ValueError(f'{self._label()}: {only} is solved by the eigenfrequency study only (no frequency '
                             'response yet)')
        system = self.system
        if system.operator_dirty:
            self._assemble()
        # the load is frequency independent: kept until a load input changes (_loads_dirty) or the dof map is rebuilt;
        # the dead loads only: the initial stress and thermal strain are static (the prestress), never a drive
        if self._harmonic_load is None or self._harmonic_load[0] is not system.dofs:
            load = self._external_load()
            load.flags.writeable = False
            self._harmonic_load = (system.dofs, load)
        load = self._harmonic_load[1]
        if self._harmonic_operators is None:
            self._harmonic_operators = self._build_harmonic_operators()
        loss, mass = self._harmonic_operators
        return load, loss, mass

    def _harmonic_system(self, loss, mass):
        """The frequency_response.HarmonicSystem of the driven solves (direct and reduced) on the 'linear' runtime (host
        SuperLU, or cuDSS from a DevicePencil: one plan across frequencies; counted there), kept while the assembly,
        the harmonic operators and the relaxed prestress stay (rebuilt otherwise, the old factor handed on as previous)."""
        if self.prestressed and (self._prestress is None or self._prestress[0]):
            self.solve_static_state(with_loads=False)
        key = (self.system.operator, self._harmonic_operators, self._prestress)
        kept = self._harmonic
        if kept is None or any(old is not new for old, new in zip(kept[0], key)):
            self._harmonic = (key, HarmonicSystem(self._pencil_stiffness(with_loads=False), loss, mass,
                                                  self.system.free, runtime=self.system.runtime,
                                                  previous=None if kept is None else kept[1].factor))
        return self._harmonic[1]

    def _harmonic_solve(self, omega, fields=True):
        """(x, A x, load) at omega [rad/s] by the kept HarmonicSystem: one complex factor, or its plan_frequency_response
        model; A x exact only at the record rows; fields False: x too (a direct solve keeps its solution for
        publish_frequency_response, R6)."""
        load, loss, mass = self._harmonic_inputs()
        harmonic = self._harmonic_system(loss, mass)
        if not harmonic.has_output_rows:   # also with fields: A x is read only at the record rows
            harmonic.set_output_rows(self._harmonic_output_rows())
        solution, product = harmonic.solve(omega, self.system.fixed_values(), load, fields=fields,
                                           product_at_rows=True)
        return solution, product, load

    def _harmonic_output_rows(self) -> np.ndarray:
        """Int64 rows (system dof x 2 + component) the frequency probes and support reactions read."""
        system = self.system
        ids = system.dofs.global_ids
        rows = []
        for _, node in self.frequency_probes:
            matches = np.flatnonzero(ids == node)
            if matches.size:
                rows.append(COMPONENTS * int(matches[0]) + np.arange(COMPONENTS))
        for region in self._supports:
            selected = (np.asarray(to_host(region.dof_mask), dtype=bool) & system.held)[ids]
            rows.append(np.flatnonzero(selected.ravel()))
        return np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)

    def plan_frequency_response(self, frequencies, options):
        """ReducedSweepSolver hook (after reset_frequency_response): options {method, tolerance, moments, max_points,
        modal_count, fields}; 'direct' -> None (only the field policy, R6), else a reduced model of the kept
        HarmonicSystem (_harmonic_system) over frequencies [Hz] with the record rows precomputed -> its info dict."""
        method = options.get('method', 'direct')
        if method == 'direct':
            self._field_policy = field_policy(options, frequencies)
            return None
        frequencies = np.asarray(frequencies, dtype=np.float64).ravel()
        if not (frequencies.size and np.all(np.isfinite(frequencies)) and np.all(frequencies > 0.0)):
            raise ValueError(f'{self._label()}: a reduced frequency sweep needs frequencies > 0 Hz')
        load, loss, mass = self._harmonic_inputs()
        harmonic = self._harmonic_system(loss, mass)
        info = harmonic.plan_reduced(
            2.0 * math.pi * frequencies, method, self.system.fixed_values(), load,
            tolerance=options.get('tolerance', 1e-6), moments=options.get('moments', 4),
            max_points=options.get('max_points', 40), modal_count=options.get('modal_count', 20),
            eigen_solver=self.eigen_solver, label=self._label())
        harmonic.set_output_rows(self._harmonic_output_rows())
        self.reduced_sweep = info
        self._field_policy = field_policy(options, frequencies)
        return info

    def solve_frequency_response(self, frequency):
        """frequency_response study hook: the driven harmonic response at frequency [Hz] > 0 of every load and
        prescribed displacement (amplitudes at phase 0; the initial stress is no drive) with the mechanical loss factor,
        (K (1 + j eta) + K_sigma - omega^2 M) u = f by one complex symmetric factor of the kept HarmonicSystem on the
        'linear' runtime (host SuperLU or cuDSS LDL^T, plan reused across frequencies; a free body is regular at f > 0)
        or the plan_frequency_response model; records the frequency probes and support reactions, shows the real parts,
        u_abs = |u| where the field policy asks (a skipped frequency is solved at the record rows only)."""
        frequency = float(frequency)
        if not (math.isfinite(frequency) and frequency > 0.0):
            raise ValueError(f'{self._label()}: a frequency response needs a frequency > 0 Hz, got {frequency!r}')
        fields = wants_fields(self._field_policy, frequency)
        omega = 2.0 * math.pi * frequency
        solution, product, load = self._harmonic_solve(omega, fields)
        self.frequency_points.append(FrequencyPoint(frequency, {}))
        displacement = solution.reshape(-1, COMPONENTS)
        self._record_harmonic(frequency, displacement, (product - load).reshape(-1, COMPONENTS))
        if fields:
            self._show_harmonic(displacement, omega=omega)
        self._unpublished_frequency = None if fields else frequency

    def publish_frequency_response(self):
        """ReducedSweepSolver hook: show the last solved frequency when the field policy skipped it (a direct solution
        kept by the HarmonicSystem, or a re-solve of the reduced model)."""
        frequency, self._unpublished_frequency = self._unpublished_frequency, None
        if frequency is None:
            return
        omega = 2.0 * math.pi * frequency
        solution = None
        if self.reduced_sweep is None and self._harmonic is not None:
            solution = self._harmonic[1].last_solution()
        if solution is None:
            solution, _, _ = self._harmonic_solve(omega)
        self._show_harmonic(solution.reshape(-1, COMPONENTS), omega=omega)

    # --- complex records of the frequency response (K4) ------------------------

    def add_frequency_probe(self, node, label='') -> HarmonicProbeRecord:
        """Record the complex displacement (u_x, u_y) [m] at mesh node `node` at every solved frequency."""
        if not 0 <= int(node) < self.domain.mesh.n_points:
            raise ValueError(f'frequency probe node {node} is not a mesh node')
        record = HarmonicProbeRecord(('u_x', 'u_y'), unit='m', label=label or 'Displacement')
        self.frequency_probes.append((record, int(node)))
        return record

    def register_support(self, region):
        """Keep a support feature's region: its complex reaction is recorded per solved frequency."""
        self._supports.append(region)
        self._harmonic_reactions[region.name] = HarmonicProbeRecord(('F_x', 'F_y'), unit=self.force_unit,
                                                                    label=f'Reaction of {region.name}')
        return region

    def _record_harmonic(self, frequency, displacement, residual):
        """Record the frequency probes (complex displacement (n_dofs, 2) of the solid map at their nodes, 0 off the
        solid) and the support reactions (sum of the complex A x - f over each support's held dofs, x the thickness)."""
        probes, supports = self._record_layout()
        for (record, _), row in zip(self.frequency_probes, probes):
            values = displacement[row] if row >= 0 else np.zeros(COMPONENTS, dtype=np.complex128)
            record.record(frequency, {'u_x': complex(values[0]), 'u_y': complex(values[1])})
        depth = 1.0 if self.thickness is None else self.thickness
        for region, rows in zip(self._supports, supports):
            totals = [depth * complex(residual[component_rows, component].sum())
                      for component, component_rows in enumerate(rows)]
            self._harmonic_reactions[region.name].record(frequency, {'F_x': totals[0], 'F_y': totals[1]})

    def _record_layout(self):
        """([solid-map row of each frequency probe node, -1 off the solid], [(held dof rows per component)] per
        support) of the frequency-response records: kept per sweep (reset_frequency_response) and dof map."""
        system = self.system
        kept = self._record_rows
        if kept is None or kept[0] is not system.dofs:
            ids = system.dofs.global_ids
            probes = []
            for _, node in self.frequency_probes:
                matches = np.flatnonzero(ids == node)
                probes.append(int(matches[0]) if matches.size else -1)
            supports = []
            for region in self._supports:
                selected = (np.asarray(to_host(region.dof_mask), dtype=bool) & system.held)[ids]
                supports.append(tuple(np.flatnonzero(selected[:, component]) for component in range(COMPONENTS)))
            kept = self._record_rows = (system.dofs, probes, supports)
        return kept[1], kept[2]

    def _build_harmonic_operators(self):
        """Host (loss operator eta K or None, consistent mass) of the last assembly (eta = mechanical_loss_factor of the
        solid elements)."""
        return self._stiffness_loss(stretched=False), self.system.mass(self._solid_density('a frequency response'))

    def _show_harmonic(self, displacement, thermal=True, omega=None):
        """Write the real parts of a complex response or damped mode (n_dofs, c; thermal False: a mode) into the outputs
        (no initial stress), u_abs = |u| (nodal and at the centroids), the imaginary parts (u_x_im / u_y_im) and the
        time-averaged energy densities at omega [rad/s]; reactions become unavailable."""
        self.system.unknown = np.ascontiguousarray(displacement.real)
        self._write_outputs(thermal=thermal, prestress=False, omega=0.0 if omega is None else float(omega),
                            imaginary=np.ascontiguousarray(displacement.imag))
        self._complex_displacement = np.array(displacement, dtype=np.complex128)   # read by the primary group
        self._shows_mode, self._shows_harmonic = False, True
        self.selected_resonance = None

    # --- display fields ---------------------------------------------------------

    def field_dofs(self, name):
        """fem DofMap of state field or source slot `name` (the transfer contract): every one is per element (P0)."""
        if name not in self.state() and name not in self.source_slots:
            raise ValueError(f"{self._label()} has no state field or source slot '{name}'")
        if self._element_dofs is None:
            self._element_dofs = lagrange_dofs(self.domain.mesh, 0)
        return self._element_dofs

    def fem_field(self, component):
        """(Lagrange DofMap, host coefficients) of a field: outputs on the full-mesh P1 / P2 map (0 off the solid),
        inputs and E / nu per element (P0)."""
        if component in OUTPUT_ARRAYS or component in HARMONIC_IMAG_FIELDS or component in ENERGY_DENSITY_FIELDS:
            self._ensure_outputs((component,))
            values = self._nodal.get(component)   # Im u: 0 unless a complex one is shown
            return self.system.full_dofs, (np.zeros(self.system.n_global) if values is None else values)
        if component in INPUT_ARRAYS or component in MATERIAL_STATE_ARRAYS:
            source = self.fields if component in INPUT_ARRAYS else self.domain.materials
            return self.field_dofs(component), _host(getattr(source, component))
        raise ValueError(f"{self._label()}: field '{component}' is not available on a triangle mesh")

    def window_field(self, component):
        """Backend float32 values of a per-element field on the window elements (renderer.device_field)."""
        xp = self.domain.xp
        self._ensure_outputs((component,))
        if component in HARMONIC_IMAG_FIELDS or component in ENERGY_DENSITY_FIELDS:
            kept = self._imag_centroids if component in HARMONIC_IMAG_FIELDS else self._energy_centroids
            values = np.zeros(self.domain.n_elements) if kept is None else kept[component]
            return xp.asarray(values, dtype=DTYPE)[self._window_device]
        source = self.domain.materials if component in MATERIAL_STATE_ARRAYS else self.fields
        return getattr(source, component)[self._window_device]

    # --- internals ----------------------------------------------------------------

    def _element_stiffness(self):
        """Host ElementStiffness of the window elements (E / nu or the c_ij entries; ring elements void unless an elastic
        PML makes them stiff); ValueError for a poisson_ratio outside (-1, 0.5), a c_ij set that is not positive
        definite, shear couplings in an axisymmetric solid and any c_ij entry with an azimuthal order m >= 1."""
        materials = self.domain.materials
        window = self._elastic_elements
        youngs_modulus = np.where(window, _host(materials.youngs_modulus), 0.0)
        poisson_ratio = np.where(window, _host(materials.poisson_ratio), 0.0)
        solid = youngs_modulus > 0.0
        out_of_range = solid & ((poisson_ratio <= -1.0) | (poisson_ratio >= 0.5))
        if out_of_range.any():
            raise ValueError(f"{self._label()}: poisson_ratio must lie in (-1, 0.5) in every element with "
                             f"youngs_modulus > 0; found values from {poisson_ratio[solid].min():.6g} to "
                             f'{poisson_ratio[solid].max():.6g}')
        lame_lambda, mu = lame_coefficients(youngs_modulus * solid, poisson_ratio * solid)
        names = set(materials.property_names())
        tensor = {name: np.where(window, _host(getattr(materials, name)), np.nan) if name in names else None
                  for name in STIFFNESS_TENSOR_ARRAYS}
        stiffness = element_stiffness(lame_lambda, mu, tensor, np.float64)
        label = self._label
        if self._axisymmetric:
            if stiffness.monoclinic:
                raise ValueError(f'{label()}: an axisymmetric solid takes orthotropic c_ij only (c16 = c26 = c36 = 0)')
            if self.azimuthal_order and stiffness.anisotropic:
                raise ValueError(f'{label()}: azimuthal order {self.azimuthal_order} takes isotropic materials only '
                                 '(youngs_modulus / poisson_ratio; the r-phi and z-phi shear moduli of a c_ij set are '
                                 'not defined in 2D)')
            check_positive_definite(stiffness, '1236', label, 'axisymmetric: rr, zz, hoop and rz shear')
        elif self.out_of_plane == PLANE_STRAIN:
            check_positive_definite(stiffness, '126', label, 'plane strain: in-plane block')
        else:
            check_positive_definite(stiffness, '1236', label, f'{self.out_of_plane}: with the zz row')
        return stiffness

    def _assemble(self):
        """Element stiffness, the operator coefficients of the condition (plane stress condensed, generalized plane
        strain = plane strain; axisymmetric with the hoop terms) and the system assembly (the factor follows lazily)."""
        self._ensure_outputs()   # pending outputs read the stiffness, dof map and unknown of their solve
        stiffness = self._element_stiffness()
        self._drop_host_inputs()   # beta and the solid-masked inputs follow the new stiffness
        self._stiffness = stiffness
        self._solid = np.asarray(stiffness.solid, dtype=bool)
        coefficients = dict(zip(_OPERATOR_NAMES, operator_coefficients(stiffness, self.out_of_plane == PLANE_STRESS)))
        if self._axisymmetric:
            coefficients.update({name: stiffness[name] for name in _HOOP_NAMES})
        self._coefficients = coefficients
        self._harmonic_operators = None
        self._prestress = None
        self._unit_axial = None
        self._pml_stiffness = None
        self._tangent_assembler = None            # the solid map, held set or coefficients may have changed
        self._nonlinear_start.forget_last()
        if self.pml_faces:
            self.system.pml = self._pml_ring()
        self.system.assemble(coefficients, self._solid)

    def _pml_ring(self):
        """fem.pml.PMLStretch of the elastic PML ring at the target frequency for the fastest wave sqrt(max c_II /
        rho) of the ring's solid elements (the window's when the ring is void)."""
        coefficients = self._coefficients
        names = ('c11', 'c22', 'c66') + (('c33',) if self._axisymmetric else ())
        diagonal = np.max(np.stack([np.asarray(_host(coefficients[name])) for name in names]), axis=0)
        speed = ring_wave_speed(diagonal, _host(self.domain.materials.rho), self._solid, self._window_elements,
                                self._label)
        return elastic_pml_ring(self.domain.mesh, self.pml_faces, 2.0 * math.pi * self.target_frequency, speed,
                                self.pml_reflection, self._axisymmetric, self._label)

    # --- generalized plane strain (superposition, as generalized_plane_strain.py on the tensor mesh) ---------------

    def _term_values(self, term, points):
        """h_term at points (..., 2): 1, x - x_ref, y - y_ref (reference = the window centre)."""
        if term == 0:
            return np.ones(points.shape[:-1])
        return points[..., term - 1] - self._term_geometry.reference[term - 1]

    def _term_integral(self, values, *terms) -> float:
        """Integral over the solid elements of per-element values times the product of the terms' h (quadrature exact
        on straight elements: sum of the load vector of the solid map, the shape functions sum to 1)."""
        values = np.asarray(values, dtype=np.float64)

        def integrand(points, elements):
            product = np.broadcast_to(values[elements][:, None], points.shape[:-1])
            for term in terms:
                product = product * self._term_values(term, points)
            return product
        return float(np.sum(assemble_load(self.system.dofs, integrand, degree=2 * self.order + 2,
                                          axisymmetric=False)))

    def _unit_axial_solves(self):
        """(terms, loads f_k = -int B^T (c13, c23, c36) h_k dA, homogeneous solutions u_k, condition matrix S_mk =
        int c33 h_k h_m dA - u_k . f_m) of the free terms; ValueError when S is not positive definite."""
        system, stiffness = self.system, self._stiffness
        coupling = np.stack([_host(stiffness.entry('1', '3')), _host(stiffness.entry('2', '3')),
                             _host(stiffness.entry('3', '6'))], axis=1)
        terms = self.bending_terms
        loads, solutions = [], []
        for term in terms:
            load = -assemble_stress_load(system.dofs, coupling,
                                         weight=lambda points, term=term: self._term_values(term, points))
            loads.append(load)
            solutions.append(system.solve_homogeneous(load))
        c33 = _host(stiffness['c33'])
        matrix = condition_matrix(terms, loads, solutions,
                                  lambda term_k, term_m: self._term_integral(c33, term_k, term_m),
                                  lambda solution, load: float(solution @ load), self._label(), 'elements')
        return terms, loads, solutions, matrix

    def _solve_generalized(self):
        """eps_zz = e0 + e1 (x - x_ref) + e2 (y - y_ref) on the eps_zz = 0 static solve by superposition of the unit
        terms (kept until the next assembly): int sigma_zz h_m dA = (F, moments) hold exactly (int c3J S_J h_m = -u .
        f_m by the same quadrature); the kept load gains sum e_k f_k, so reactions stay K u - f."""
        system = self.system
        if self._unit_axial is None:
            self._unit_axial = self._unit_axial_solves()
        terms, loads, solutions, matrix = self._unit_axial
        _, _, beta_z, _ = self._thermal_coefficients()
        thermal_stress_z = _host(beta_z) * self._temperature_change()
        targets = condition_targets(self._term_geometry, self.axial_force, *self.bending_moments)
        displacement = system.unknown.reshape(-1)
        values = strain_term_values(matrix, terms, loads, targets,
                                    lambda term: self._term_integral(thermal_stress_z, term),
                                    lambda load: float(displacement @ load))
        coefficients = np.zeros(3)
        for term, value, load, solution in zip(terms, values, loads, solutions):
            coefficients[term] = value
            displacement = displacement + value * solution
            system.load = system.load + value * load
        system.unknown = displacement.reshape(-1, COMPONENTS)
        self._axial_terms = coefficients

    def _axial_strain_at(self, points):
        """eps_zz of the last generalized plane-strain solve at points (..., 2)."""
        return sum(self._axial_terms[term] * self._term_values(term, points) for term in range(3))

    def _expansion(self):
        """Host (alpha_xx, alpha_yy, alpha_zz) [1/K] per element (cached, read-only)."""
        materials = self.domain.materials
        return self._host_input('expansion', lambda: tuple(_host(getattr(materials, name))
                                                           for name in ('alpha_xx', 'alpha_yy', 'alpha_zz')))

    def _thermal_coefficients(self, plane_stress=False):
        """(beta_x, beta_y, beta_z, beta_xy) of the last assembly's stiffness (plane_stress: reduced; cached)."""
        return self._host_input(('beta', plane_stress), lambda: thermal_coefficients(
            self._stiffness, *self._expansion(), plane_stress=plane_stress))

    def _total_load(self, dead_loads=True):
        """Load vector of the solid dofs: (dead_loads) tractions and applied + coupled body forces on the solid
        elements, the thermal stress load int B^T beta dT (plane stress: reduced beta) and the initial-stress force -int B^T
        sigma0 dV."""
        system = self.system
        dofs = system.dofs
        axisymmetric = self._axisymmetric
        load = self._external_load() if dead_loads else np.zeros(COMPONENTS * dofs.n_dofs)
        initial = self._initial_stress()
        if initial is not None:
            load += initial_stress_force(dofs, initial[:, self._stress_rows()], axisymmetric=axisymmetric)
        beta_x, beta_y, beta_z, beta_xy = self._thermal_coefficients(plane_stress=self.out_of_plane == PLANE_STRESS)
        temperature_change = self._temperature_change()
        rows = (beta_x, beta_y, beta_z, beta_xy) if axisymmetric else (beta_x, beta_y, beta_xy)
        load += assemble_stress_load(dofs, np.stack([row * temperature_change for row in rows], axis=1),
                                     axisymmetric=axisymmetric)
        return load

    def _external_load(self):
        """The dead-load part of _total_load (2 n_dofs): tractions and the applied + coupled body forces of the solid
        elements (no initial-stress or thermal term)."""
        dofs = self.system.dofs
        axisymmetric = self._axisymmetric
        load = np.zeros(COMPONENTS * dofs.n_dofs)
        for edges, t_x, t_y in self._tractions:
            load += assemble_edge_traction(dofs, edges, t_x, t_y, axisymmetric=axisymmetric)
        coupled = np.stack([_host(self.fields.body_force_x), _host(self.fields.body_force_y)], axis=1)
        body = np.where(self._solid[:, None], self._body_force + coupled, 0.0)
        load += assemble_body_load(dofs, body[:, 0], body[:, 1], axisymmetric=axisymmetric)
        return load

    def _point_outputs(self, elements, strains, thermal, points=None, prestress=False, energy_scale=0.5):
        """{output: (m, q)} strain / stress values at points (m, q, 2) of mesh elements from Voigt strains (m, q, k; k
        = 6 for an azimuthal order: the outputs are the cos parts at phi = 0): the out-of-plane strain of the condition
        (plane stress sigma_zz = 0, generalized plane strain e0 + e1 x + e2 y, axisymmetric the hoop row), sigma = C
        eps - beta dT (thermal False: dT = 0 and eps_zz = 0, a mode shape) + the initial stress sigma0 (prestress: a
        static solve); strain_energy_density = energy_scale x (C eps_el) : eps_el (no sigma0; azimuthal: + the sin
        shears mu (g_zt^2 + g_rt^2))."""
        stiffness = _restricted(self._stiffness, elements)
        rows = strains.shape[-1]
        strain_xx, strain_yy, shear = strains[..., 0], strains[..., 1], strains[..., 2 if rows == 3 else 3]
        alpha_xx, alpha_yy, alpha_zz = (values[elements][:, None] for values in self._expansion())
        beta_x, beta_y, beta_z, beta_xy = (values[elements][:, None] for values in self._thermal_coefficients())
        temperature_change = self._temperature_change()[elements][:, None] if thermal else 0.0
        if self._axisymmetric:
            strain_zz = strains[..., 2]
        elif self.out_of_plane == PLANE_STRESS:
            strain_zz = plane_stress_strain(stiffness, strain_xx, strain_yy, shear, beta_z * temperature_change)
        elif self.out_of_plane == GENERALIZED_PLANE_STRAIN and thermal:
            strain_zz = self._axial_strain_at(points)
        else:
            strain_zz = np.zeros_like(strain_xx)
        stress_xx, stress_yy, stress_zz, stress_xy = elastic_stress(stiffness, strain_xx, strain_yy, strain_zz, shear)
        stress_xx = stress_xx - beta_x * temperature_change
        stress_yy = stress_yy - beta_y * temperature_change
        stress_xy = stress_xy - beta_xy * temperature_change
        stress_zz = (np.zeros_like(strain_xx) if self.out_of_plane == PLANE_STRESS
                     else stress_zz - beta_z * temperature_change)
        elastic_xx = strain_xx - alpha_xx * temperature_change
        elastic_yy = strain_yy - alpha_yy * temperature_change
        elastic_zz = strain_zz - alpha_zz * temperature_change   # axisymmetric: the hoop part
        contraction = stress_xx * elastic_xx + stress_yy * elastic_yy + stress_zz * elastic_zz + stress_xy * shear
        if rows == 6:   # the sin-part shears of an azimuthal order (isotropic: c44 = c55 = c66)
            contraction = contraction + stiffness['c66'] * (strains[..., 4] ** 2 + strains[..., 5] ** 2)
        initial = self._initial_stress() if prestress else None
        if initial is not None:   # columns xx, yy, xy, zz (axisymmetric: rr, zz, rz, hoop)
            stress_xx = stress_xx + initial[elements, 0][:, None]
            stress_yy = stress_yy + initial[elements, 1][:, None]
            stress_xy = stress_xy + initial[elements, 2][:, None]
            stress_zz = stress_zz + initial[elements, 3][:, None]
        von_mises = np.sqrt(0.5 * ((stress_xx - stress_yy) ** 2 + (stress_yy - stress_zz) ** 2
                                   + (stress_zz - stress_xx) ** 2) + 3.0 * stress_xy ** 2)
        return {'strain_xx': strain_xx, 'strain_yy': strain_yy, 'strain_xy': 0.5 * shear,
                'elastic_strain_xx': elastic_xx, 'elastic_strain_yy': elastic_yy, 'elastic_strain_zz': elastic_zz,
                'stress_xx': stress_xx, 'stress_yy': stress_yy, 'stress_xy': stress_xy, 'stress_zz': stress_zz,
                'von_mises': von_mises, 'strain_energy_density': energy_scale * contraction}

    def _strains(self, displacement, reference_points):
        """(displacement (m, q, c), Voigt strains (m, q, k)) of the solid map at reference points: fem.elasticity
        element_strains (Green-Lagrange strains after a nonlinear static solve), or azimuthal_element_strains for an
        order m >= 1."""
        dofs = self.system.dofs
        if self.azimuthal_order:
            return azimuthal_element_strains(dofs, displacement, reference_points, self.azimuthal_order)
        values, strains = element_strains(dofs, displacement, reference_points, axisymmetric=self._axisymmetric)
        if self._output_green_lagrange:
            strains = green_lagrange_strains(dofs, displacement, reference_points, axisymmetric=self._axisymmetric)
        return values, strains

    def _density_values(self):
        """Host rho per element on the solid (0 elsewhere; cached, read-only)."""
        return self._host_input('density', lambda: np.where(self._solid, _host(self.domain.materials.rho), 0.0))

    def _output_values(self, displacement, imaginary, reference_points, thermal, points, prestress, omega):
        """(displacement values (m, q, c), {output: (m, q)}) at reference points of the solid elements: _point_outputs
        and the energy densities, static (omega None) 1/2 sigma : eps_el and 0, else the time averages 1/4 (sigma_r :
        eps_r + sigma_i : eps_i) and 1/4 rho omega^2 |u|^2 (imaginary: Im u (n_dofs, c) or None)."""
        elements = self.system.dofs.elements
        values, strains = self._strains(displacement, reference_points)
        scale = 0.5 if omega is None else 0.25
        outputs = self._point_outputs(elements, strains, thermal, points, prestress, energy_scale=scale)
        squared = np.sum(values ** 2, axis=-1)
        if imaginary is not None:
            imaginary_values, imaginary_strains = self._strains(imaginary, reference_points)
            imaginary_outputs = self._point_outputs(elements, imaginary_strains, False, points, False,
                                                    energy_scale=scale)
            outputs['strain_energy_density'] = (outputs['strain_energy_density']
                                                + imaginary_outputs['strain_energy_density'])
            squared = squared + np.sum(imaginary_values ** 2, axis=-1)
        if omega is None:
            outputs['kinetic_energy_density'] = np.zeros_like(squared)
        else:
            outputs['kinetic_energy_density'] = (0.25 * omega ** 2 * self._density_values()[elements][:, None]
                                                 * squared)
        return values, outputs

    def _write_outputs(self, thermal=True, prestress=None, omega=None, imaginary=None, green_lagrange=False):
        """Mark the outputs of system.unknown pending; each _OUTPUT_GROUPS group is computed on its first read
        (_ensure_outputs): nodal display values (displacement exact, strain / stress / energy densities averaged from
        the element values at the dof points over the solid elements) and the per-element centroid fields (0 off the
        solid); probes read the nodal values. prestress None = thermal: the stress includes the initial stress (a static
        solve). omega None: a static solve (strain energy 1/2 sigma : eps_el, kinetic 0); omega [rad/s]: a mode or a
        response (time averages, imaginary = Im u of a complex one, _output_values; _show_harmonic sets the complex
        displacement after). green_lagrange: Green-Lagrange strains and second Piola-Kirchhoff stresses of a nonlinear
        static solve. An azimuthal order: u_z = U_phi, the strain / stress outputs the cos parts (phi = 0)."""
        self._output_thermal = thermal
        self._output_prestress = thermal if prestress is None else prestress
        self._output_omega = omega
        self._output_imaginary = imaginary
        self._output_green_lagrange = bool(green_lagrange)
        self._complex_displacement = self._imag_centroids = self._energy_centroids = None
        self._nodal = {}
        self._pending_outputs = set(_OUTPUT_GROUPS)

    def _write_element_field(self, name, values):
        """The per-element state array `name`: values on the solid map's elements, 0 elsewhere."""
        full_values = np.zeros(self.domain.n_elements)
        full_values[self.system.dofs.elements] = values
        xp = self.domain.xp
        xp.copyto(getattr(self.fields, name), xp.asarray(full_values, dtype=DTYPE))

    def _write_primary_outputs(self):
        """Displacement (nodal exact; centroids by the basis weights, no element geometry), u_abs (|u| of a complex
        solution) and the imaginary parts u_x_im / u_y_im of a complex one."""
        system = self.system
        dofs = system.dofs
        displacement = system.unknown
        full_index = np.searchsorted(system.full_dofs.global_ids, dofs.global_ids)
        nodal = self._nodal
        names = ('u_x', 'u_y', 'u_z')[:system.components]
        for component, name in enumerate(names):
            nodal[name] = np.zeros(system.n_global)
            nodal[name][full_index] = displacement[:, component]
        basis = lagrange_basis(self.order, CENTROID)[0][0]   # (n_local,) at the reference centroid
        centre = np.einsum('n,mnc->mc', basis, displacement[dofs.element_dofs])
        for component, name in enumerate(names):
            self._write_element_field(name, centre[:, component])
        complex_displacement = self._complex_displacement
        if complex_displacement is None:
            nodal['u_abs'] = np.sqrt(sum(nodal[name] ** 2 for name in names))
            self._write_element_field('u_abs', np.sqrt(np.sum(centre ** 2, axis=-1)))
            return
        nodal['u_abs'] = np.zeros(system.n_global)
        nodal['u_abs'][full_index] = np.sqrt(np.sum(np.abs(complex_displacement) ** 2, axis=1))
        centre_imag = np.einsum('n,mnc->mc', basis, complex_displacement.imag[dofs.element_dofs])
        self._write_element_field('u_abs', np.sqrt(np.sum(centre ** 2 + centre_imag ** 2, axis=-1)))
        self._imag_centroids = {}
        for component, name in enumerate(HARMONIC_IMAG_FIELDS):
            nodal[name] = np.zeros(system.n_global)
            nodal[name][full_index] = complex_displacement[:, component].imag
            self._imag_centroids[name] = np.zeros(self.domain.n_elements)
            self._imag_centroids[name][dofs.elements] = centre_imag[:, component]

    def _write_derived_outputs(self):
        """Strain / stress / von Mises and the energy densities (_output_values): nodal averages over the solid elements
        and their centroid values; the electric outputs stay 0 (no piezoelectric)."""
        system = self.system
        dofs = system.dofs
        displacement = system.unknown
        thermal, prestress, omega = self._output_thermal, self._output_prestress, self._output_omega
        imaginary = self._output_imaginary
        full_index = np.searchsorted(system.full_dofs.global_ids, dofs.global_ids)
        generalized = self.out_of_plane == GENERALIZED_PLANE_STRAIN
        local_points = dofs.dof_points()[dofs.element_dofs] if generalized else None   # local dof k = column k
        _, local = self._output_values(displacement, imaginary, local_dof_points(self.order), thermal, local_points,
                                       prestress, omega)
        averaged = nodal_average(dofs, np.stack(list(local.values()), axis=-1))
        for column, name in enumerate(local):
            self._nodal[name] = np.zeros(system.n_global)
            self._nodal[name][full_index] = averaged[:, column]
        centre_points = (element_geometry(dofs.mesh, dofs.elements, CENTROID, dofs.geometry_order).points
                         if generalized else None)
        _, centre = self._output_values(displacement, imaginary, CENTROID, thermal, centre_points, prestress, omega)
        for name in _OUTPUT_GROUPS['derived']:
            if name in OUTPUT_ARRAYS:   # electric outputs: not in centre, 0
                self._write_element_field(name, centre[name][:, 0] if name in centre else 0.0)
        self._energy_centroids = {}
        for name in ENERGY_DENSITY_FIELDS:
            values = np.zeros(self.domain.n_elements)
            values[dofs.elements] = centre[name][:, 0]
            self._energy_centroids[name] = values

    def _record_probes(self, t):
        self._ensure_outputs(probe.field_component for probe in self.probes)
        full_ids = self.system.full_dofs.global_ids
        for probe in self.probes:
            values = self._nodal.get(probe.field_component)
            value = 0.0 if values is None else values[np.searchsorted(full_ids, probe.ix)]
            probe.record_value(np.float32(value), t)


class SolidFEMRenderer(SolidRenderer):
    """SolidRenderer of a SolidMechanicsFEMPhysics: fem_field (Session on triangle meshes) and per-window-element
    device_field values. No om_integrand (the BOR coupling table has none), so default snapshots do not store it."""

    def __init__(self, physics):
        super().__init__(physics)
        self.field_infos = {name: info for name, info in self.field_infos.items() if name != 'om_integrand'}

    def device_field(self, component: str):
        self.field_info(component)
        return self.physics.window_field(component)

    def fem_field(self, component: str):
        self.field_info(component)
        return self.physics.fem_field(component)
