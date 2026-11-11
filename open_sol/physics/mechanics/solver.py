"""Static 2D solid mechanics: planar per unit z length (plane strain, plane stress or generalized plane strain) or
axisymmetric (r, z) whole rings; small strain, isotropic or anisotropic (c_ij, stiffness.py) linear elastic cells,
thermal strain from the temperature-rise slot, supports, tractions and body forces; one ElasticSystem solve per change;
vibration modes for the eigenfrequency study (modal.solve_modes on the host, or 2.5D guided modes at k_z); the driven
harmonic response of the frequency_response study (host complex SuperLU); piezoelectric (setting): nodal potential and
electrodes, host coupled solves (piezo.PiezoSystem)."""
import math

import numpy as np
from ...core.backend import to_host

from ...core.constants import DTYPE, FACES
from ...core.fv_diffusion import edge_extension_views
from ...core.monitors import FieldProbe
from ...fem.reduced_outputs import field_policy, wants_fields
from ...solvers.arpack_shift_invert import ArpackShiftInvertSolver
from .fields import ELECTRIC_OUTPUT_ARRAYS, INPUT_ARRAYS, OUTPUT_ARRAYS, SOURCE_SLOTS, MechanicsFields2D
from .guided_modes import GUIDED_COMPONENTS, check_antiplane_supports, guided_free_dofs, guided_operators
from .frequency_response import FrequencyPoint, HarmonicSystem, solve_harmonic
from .materials import (DENSITY_SCHEMA, ELECTROMECHANICAL_ARRAYS, LOSS_ARRAYS, MECHANICAL_SCHEMA, PERMITTIVITY_ARRAYS,
                        PIEZO_ARRAYS, PIEZO_DEFAULTS, PIEZO_OUT_OF_PLANE_ARRAYS, SHEAR_PHOTOELASTIC_SCHEMA,
                        STIFFNESS_TENSOR_ARRAYS)
from .modal import ModalPencil, MultiShiftModes
from .piezo import (DISPLACEMENT_COMPONENTS, ELECTRODE_CONNECTIONS, PIEZO_COMPONENTS, Electrode, PiezoSystem,
                    coupled_mass, coupled_operator, interleaved_layout, node_potential_dofs)
from .renderer import SolidRenderer
from .solver_common import SolidMechanicsBase
from .stiffness import (check_positive_definite, element_stiffness, elastic_stress, operator_coefficients,
                        plane_stress_strain, thermal_coefficients)
from .system import (FACE_NODES, ElasticPattern, ElasticSystem, axisymmetric_mass_stencil, gauss_stiffness_stencil,
                     mass_stencil, stiffness_stencil)
from .generalized_plane_strain import BENDING_TERMS, solve_strain_terms, terms_about_origin, unit_strain_solves

# out_of_plane conditions of a planar solid: eps_zz = 0, sigma_zz = 0, or eps_zz = e0 + e1 x + e2 y set by the net
# axial force and the two bending moments (generalized plane strain)
PLANE_STRAIN, PLANE_STRESS, GENERALIZED_PLANE_STRAIN = 'plane_strain', 'plane_stress', 'generalized_plane_strain'
OUT_OF_PLANE_CONDITIONS = (PLANE_STRAIN, PLANE_STRESS, GENERALIZED_PLANE_STRAIN)
GEOMETRIES = ('planar', 'axisymmetric')


class SolidMechanicsPhysics(MultiShiftModes, SolidMechanicsBase):
    """2D static elasticity on the physical window (Q1 elements on the cells, displacement on the cell corners,
    block-Jacobi PCG; piezoelectric: + nodal phi, host SuperLU): planar plane strain / stress / generalized plane strain
    or axisymmetric (r, z); outputs at the cell centres, 0 in void cells (phi / E on every window cell), edge-extended
    into the PML. Arrays on the domain's backend (gpu or cpu)."""

    material_schema = {**MECHANICAL_SCHEMA, **DENSITY_SCHEMA, **SHEAR_PHOTOELASTIC_SCHEMA}
    output_arrays = OUTPUT_ARRAYS
    input_arrays = INPUT_ARRAYS
    source_slots = SOURCE_SLOTS

    def __init__(self, domain, tol: float = 1e-8, max_iter: int = 10000, precision: str = 'float64',
                 compute_residual: bool = False, linear_solver=None, target_frequency: float = 0.0,
                 eigen_solver=None, axial_wavenumber=None, guided_by_study=False, piezoelectric: bool = False,
                 geometry: str = 'planar', out_of_plane: str = PLANE_STRAIN, axial_force: float = 0.0,
                 thickness=None, bending: str = 'xy', bending_moment_x: float = 0.0, bending_moment_y: float = 0.0):
        if geometry not in GEOMETRIES:
            raise ValueError(f"geometry must be one of {', '.join(GEOMETRIES)}, got {geometry!r}")
        if out_of_plane not in OUT_OF_PLANE_CONDITIONS:
            raise ValueError(f"out_of_plane must be one of {', '.join(OUT_OF_PLANE_CONDITIONS)}, got {out_of_plane!r}")
        if bending not in BENDING_TERMS:
            raise ValueError(f"bending must be one of {', '.join(BENDING_TERMS)}, got {bending!r}")
        # (r, z) body of revolution about the x_lo window edge; the out-of-plane condition is the hoop strain u_r / r
        self._axisymmetric = geometry == 'axisymmetric'
        self.out_of_plane = PLANE_STRAIN if self._axisymmetric else out_of_plane
        self.axial_force = float(axial_force)   # [N] net F_z on the drawn cross-section (generalized plane strain)
        # generalized plane strain: free eps_zz terms (0 uniform, 1 x, 2 y) and the moments [N m] about the drawing
        # origin, M_x = integral y sigma_zz dA, M_y = -integral x sigma_zz dA
        self.bending_terms = BENDING_TERMS[bending]
        self.bending_moments = (float(bending_moment_x), float(bending_moment_y))
        # [m] out-of-plane thickness of a planar solid: reactions in N; None = per m of depth (N/m)
        if thickness is not None and not (math.isfinite(float(thickness)) and float(thickness) > 0.0):
            raise ValueError(f'thickness must be None or a finite value > 0 [m], got {thickness!r}')
        self.thickness = None if thickness is None or self._axisymmetric else float(thickness)
        if (self._axisymmetric or self.out_of_plane != PLANE_STRAIN) and (
                piezoelectric or axial_wavenumber is not None or guided_by_study):
            raise ValueError('piezoelectric and guided (2.5D) modes need planar plane strain (geometry planar, '
                             'out_of_plane plane_strain)')
        system = ElasticSystem(precision, float(tol), int(max_iter), label=lambda: f"solid mechanics '{self.name}'",
                               compute_residual=compute_residual, linear_solver=linear_solver,
                               backend=domain.backend.name)
        system.axisymmetric = self._axisymmetric
        self.target_frequency = float(target_frequency)   # [Hz] eigenfrequency study target (0 = the lowest modes)
        # nodal potential + electrodes, static and modal solves of the coupled operator on the host (piezo.py)
        self.piezoelectric = bool(piezoelectric)
        if self.piezoelectric and (axial_wavenumber is not None or guided_by_study):
            raise ValueError('a piezoelectric solid_mechanics has no guided (2.5D) modes: leave axial_wavenumber unset')
        self._piezo_counters = {'piezo_factorizations': 0, 'piezo_solves': 0, 'piezo_factor_host_bytes': 0,
                                'piezo_harmonic_factorizations': 0}
        # frequency_response study: complex host factorisations outside any solver role (piezoelectric: piezo_*)
        self._harmonic_counters = {'harmonic_factorizations': 0, 'harmonic_factor_host_bytes': 0}
        self.frequency_points = []   # frequency_response.FrequencyPoint per solved frequency, in solve order
        self._reduced_harmonic = None   # elastic frequency_response.HarmonicSystem of plan_frequency_response, or None
        self.reduced_sweep = None       # info of plan_frequency_response
        self._field_policy = None       # fem.reduced_outputs field policy of that plan
        self._unpublished_frequency = None   # [Hz] solved without showing its fields (field policy)
        self._unpublished_solution = None    # direct solution at _unpublished_frequency (None: re-solve the reduced model)
        # [rad/m] k_z of the guided (2.5D) vibration modes; None = plane-strain modes (and every static solve), or not
        # set yet when guided_by_study (backward Brillouin: the optomechanics study sets 2 Re beta before the solve)
        if axial_wavenumber is not None and not (math.isfinite(float(axial_wavenumber)) and axial_wavenumber >= 0.0):
            raise ValueError(f'axial_wavenumber must be None or a finite k_z >= 0 [rad/m], got {axial_wavenumber!r}')
        self.axial_wavenumber = None if axial_wavenumber is None else float(axial_wavenumber)
        self._guided = axial_wavenumber is not None or bool(guided_by_study)
        self._owns_eigen_solver = eigen_solver is None   # scripts: a private runtime, reported in work_counters
        self.eigen_solver = ArpackShiftInvertSolver() if eigen_solver is None else eigen_solver
        self.eigen_solve = None          # modal.ModalSolve of the last solve_eigenfrequencies
        self.modal_mass = None           # host scipy CSR consistent mass [kg/m] of that solve (all dofs)
        self.selected_resonance = None   # 1-based mode shown in the outputs, None = the static solve
        self.optomechanical_couplings = []   # optomechanics.OptomechanicalCoupling of the features (optomechanics study)
        self._init_solid(system, domain)

    def prepare(self, domain) -> None:
        """Base prepare plus the per-element applied body force and the held-u_z node mask of the guided modes; assembly
        is deferred."""
        super().prepare(domain)
        system = self.system
        xp = system.xp
        self._applied_body_force = xp.zeros((2, *system.element_shape), dtype=system.dtype)   # [N/m^3] per element
        self._held_axial = xp.zeros(system.node_shape, dtype=xp.bool_)   # u_z held at 0 in a guided modal solve
        self._axial_nodes = None   # nodal u_z amplitude [m] of the shown guided mode
        self._lame = None              # (lambda, mu) of the guided-mode coefficients (isotropic only)
        self._stiffness = None         # stiffness.ElementStiffness of the last assembly
        self._solid = None
        self._electrodes = []          # piezo.Electrode of the electrode features
        self._piezo_system = None      # piezo.PiezoSystem of the last assembly
        # window (e11, e12, e16, e21, e22, e26, e13, e23) [C/m^2] of the solid cells, solve dtype
        self._piezo_coefficients = None
        self._potential_nodes = None   # nodal phi [V] of the shown solve / mode
        self._piezo_reaction = None    # host (nx_nodes, ny_nodes, 2) (A x)_u - f [N/m] of the last static solve
        self._electrode_values = {}    # {electrode name: (V, Q [C/m])} of the last static solve
        # generalized plane strain: (e0, e1, e2) of eps_zz = e0 + e1 x + e2 y [1, 1/m, 1/m] (x, y of the drawing) and
        # its element-centre field of the last solve; the unit-term solves of the last assembly (None = stale)
        self._axial_strain = (0.0, 0.0, 0.0)
        self._axial_strain_field = 0.0
        self._unit_axial_solve = None  # generalized_plane_strain.UnitStrainSolves
        self._harmonic_operators = None   # host (K or A, loss, M) of the frequency response; None = stale
        if self.piezoelectric:
            domain.materials.add_properties(PIEZO_DEFAULTS)   # no-op for the arrays a material allocated
        if self._axisymmetric:   # the axis rule: u_r = 0 on the x_lo nodes (r = 0)
            self.hold_face('x_lo', u_x=0.0)

    @property
    def axisymmetric(self) -> bool:
        """True for an (r, z) body of revolution (Session ring-weights field derived values)."""
        return self._axisymmetric

    @property
    def ring_angle(self) -> float:
        """Angle [rad] of the ring weights: 2 pi (no azimuthal order)."""
        return 2.0 * math.pi

    @property
    def force_unit(self) -> str:
        """Unit of tractions times areas, reactions and nodal loads: N (whole ring) or N/m (per m of depth)."""
        return 'N' if self._axisymmetric or self.thickness is not None else 'N/m'

    @property
    def axial_strain_terms(self) -> tuple:
        """(e0, e1, e2) of eps_zz = e0 + e1 x + e2 y [1, 1/m, 1/m] (x, y from the window corner) of the last
        generalized plane-strain solve (zeros otherwise)."""
        return self._axial_strain

    def work_counters(self) -> dict:
        """The private eigen solver's counters once it ran (a build-injected one reports in its role record), plus the
        piezoelectric host factorisations / solves (outside any solver role)."""
        counters = {}
        if self._owns_eigen_solver and self.eigen_solver.n_eigen_solves:
            counters.update(self.eigen_solver.work_counters())
        if self.piezoelectric:
            counters.update(self._piezo_counters)
        elif self._harmonic_counters['harmonic_factorizations']:
            counters.update(self._harmonic_counters)
        return counters

    def materials_changed(self, names) -> None:
        """Base rules; with piezoelectric a piezo_e* / eps_dc_* change re-assembles the coupled operator; a loss or rho
        change rebuilds the frequency-response operators."""
        super().materials_changed(names)
        if self.piezoelectric and ELECTROMECHANICAL_ARRAYS & set(names):
            self.system.operator_dirty = True
            self._loads_dirty = True
        if {*LOSS_ARRAYS, *DENSITY_SCHEMA} & set(names):
            self._harmonic_operators = None

    @property
    def guided(self) -> bool:
        """True when the eigenfrequency solve finds 2.5D guided modes (axial_wavenumber set, or set by the study)."""
        return self._guided

    def set_axial_wavenumber(self, axial_wavenumber):
        """k_z [rad/m, finite > 0] of the next guided modal solve (the optomechanics study: 2 Re beta of the backward
        Brillouin pump); ValueError for a plane-strain solid."""
        if not self._guided:
            raise ValueError(f'{self.system.label()} solves plane-strain modes: create it with axial_wavenumber or '
                             'guided_by_study to set a k_z')
        wavenumber = float(axial_wavenumber)
        if not (math.isfinite(wavenumber) and wavenumber > 0.0):
            raise ValueError(f'{self.system.label()}: k_z must be finite and > 0 [rad/m], got {axial_wavenumber!r}')
        self.axial_wavenumber = wavenumber
        self.eigen_solve = None

    def finalize(self) -> None:
        """Plane strain: the base assembly (rigid-mode check). Guided: element coefficients only (k_z > 0 has no rigid
        motion), at k_z = 0 the plane-strain and antiplane support checks."""
        if not self.guided:
            super().finalize()
            return
        self._guided_coefficients()

    def renderer(self):
        return SolidRenderer(self)

    # --- supports, loads, probes ----------------------------------------------

    def hold_face(self, face: str, u_x=None, u_y=None, u_z=None):
        """Hold u_x and / or u_y [m] (None = free) on every node of an outer face, and u_z (only 0: read by a guided
        modal solve, plane strain has no u_z); returns the held in-plane dof mask."""
        if face not in FACES:
            raise ValueError(f"unknown face {face!r}; expected one of {FACES}")
        if u_z not in (None, 0.0):
            raise ValueError(f'hold_face: u_z can only be held at 0 (a guided mode is homogeneous), got {u_z!r}')
        system = self.system
        dof_mask = system.xp.zeros((*system.node_shape, 2), dtype=system.xp.bool_)
        values = system.xp.zeros((*system.node_shape, 2), dtype=system.dtype)
        nodes = FACE_NODES[face]
        for component, value in ((0, u_x), (1, u_y)):
            if value is not None:
                dof_mask[nodes + (component,)] = True
                values[nodes + (component,)] = float(value)
        system.hold(dof_mask, values)
        if u_z is not None:
            self._held_axial[nodes] = True
        return dof_mask

    def hold_cells(self, mask, u_x: float = 0.0, u_y: float = 0.0):
        """Hold both components [m] (and u_z at 0 for a guided modal solve) on every corner node of the window cells
        under the full-grid mask; returns the held in-plane dof mask."""
        system = self.system
        nodes = system.cell_nodes(system.window_mask(mask, 'fixed domain'))
        dof_mask = system.xp.stack((nodes, nodes), axis=-1)
        values = system.xp.zeros((*system.node_shape, 2), dtype=system.dtype)
        values[..., 0] = float(u_x)
        values[..., 1] = float(u_y)
        system.hold(dof_mask, values)
        self._held_axial |= nodes
        return dof_mask

    def add_face_traction(self, face: str, t_x: float, t_y: float) -> None:
        """Add a traction (t_x, t_y) [Pa = N/m^2] on an outer face: each node gets t times its tributary length."""
        if face not in FACES:
            raise ValueError(f"unknown face {face!r}; expected one of {FACES}")
        system = self.system
        lengths = system.face_node_lengths(face)
        nodes = FACE_NODES[face]
        system.applied_load[nodes + (0,)] += float(t_x) * lengths
        system.applied_load[nodes + (1,)] += float(t_y) * lengths
        self._loads_dirty = True

    def add_segment_traction(self, start, end, t_x: float, t_y: float) -> None:
        """Add a traction (t_x, t_y) [Pa] on the axis-parallel segment start -> end ((x, y) [m], physical) acting on
        the node line nearest to it (an outer face or an interior line): each node gets t times the integral of its
        shape function over the segment's part inside the window (axisymmetric: whole-ring weights, N)."""
        system = self.system
        (x0, y0), (x1, y1) = ((float(x), float(y)) for x, y in (start, end))
        node_x, node_y = system.node_positions('x'), system.node_positions('y')
        tolerance = 1e-9 * max(float(node_x[-1]), float(node_y[-1]))
        if abs(y1 - y0) <= tolerance:   # along x on the nearest node row
            row = int(np.argmin(np.abs(node_y - 0.5 * (y0 + y1))))
            nodes = (slice(None), row)
            lengths = system.segment_node_lengths('x', min(x0, x1), max(x0, x1))
        elif abs(x1 - x0) <= tolerance:   # along y on the nearest node column
            column = int(np.argmin(np.abs(node_x - 0.5 * (x0 + x1))))
            nodes = (column, slice(None))
            lengths = system.segment_node_lengths('y', min(y0, y1), max(y0, y1), radius=node_x[column])
        else:
            raise ValueError(f'a segment load needs a horizontal or vertical segment, got ({x0:g}, {y0:g}) -> '
                             f'({x1:g}, {y1:g})')
        system.applied_load[nodes + (0,)] += float(t_x) * lengths
        system.applied_load[nodes + (1,)] += float(t_y) * lengths
        self._loads_dirty = True

    def add_body_force(self, mask, f_x, f_y) -> None:
        """Add a body force density (f_x, f_y) [N/m^3] (scalars or full-grid arrays) on the window cells under mask;
        void cells (effective c11 <= 0) drop it."""
        system = self.system
        mask_window = system.window_mask(mask, 'body force')
        for component, value in ((0, f_x), (1, f_y)):
            density = system.window_values(value, 'body force density', system.dtype)
            self._applied_body_force[component] += system.xp.where(mask_window, density, 0.0)
        self._loads_dirty = True

    def add_temperature_change(self, mask, delta_T) -> None:
        """Add a temperature change [K] above the stress-free state (scalar or full-grid array) to temperature_rise_ext
        on the window cells under mask."""
        system = self.system
        mask_window = system.window_mask(mask, 'temperature change')
        change = system.window_values(delta_T, 'temperature change', DTYPE)
        self.fields.temperature_rise_ext[system.window] += system.xp.where(mask_window, change, DTYPE(0.0))
        self._loads_dirty = True

    def add_probe(self, ix: int, iy: int, component: str = 'u_abs') -> FieldProbe:
        """Record an output field at full-grid cell (ix, iy) whenever step records probes."""
        if component not in OUTPUT_ARRAYS:
            raise ValueError(f"probe component {component!r} is not one of {', '.join(OUTPUT_ARRAYS)}")
        window_x, window_y = self.system.window
        if not (window_x.start <= ix < window_x.stop and window_y.start <= iy < window_y.stop):
            raise ValueError(f"solid probe cell ({ix}, {iy}) lies outside the solve window x [{window_x.start}, "
                             f"{window_x.stop}), y [{window_y.start}, {window_y.stop})")
        probe = FieldProbe(ix, iy, component)
        self.probes.append(probe)
        return probe

    def add_probe_at(self, x: float, y: float, component: str = 'u_abs') -> FieldProbe:
        """add_probe in the cell containing physical (x, y) [m]."""
        self.domain.validate_physical_pos(x, y, 'Solid probe')
        return self.add_probe(*self.domain.pos_to_cell(x, y), component)

    # --- piezoelectric electrodes ---------------------------------------------

    def add_face_electrode(self, name, faces, connection='voltage', voltage=0.0, charge=0.0) -> Electrode:
        """Electrode on every node of the outer faces (corners included): connection 'voltage' holds phi at voltage [V]
        (0 in a modal solve), 'floating' shares one phi carrying charge [C/m] (0 in a modal solve)."""
        nodes = np.zeros(self.system.node_shape, dtype=bool)
        for face in faces:
            if face not in FACES:
                raise ValueError(f"unknown face {face!r}; expected one of {FACES}")
            nodes[FACE_NODES[face]] = True
        return self._add_electrode(name, nodes, connection, voltage, charge)

    def add_cell_electrode(self, name, mask, connection='voltage', voltage=0.0, charge=0.0) -> Electrode:
        """Electrode on every corner node of the window cells under the full-grid mask (a drawn metal electrode);
        connection, voltage and charge as add_face_electrode."""
        system = self.system
        nodes = np.array(to_host(system.cell_nodes(system.window_mask(mask, f"electrode '{name}'"))))
        return self._add_electrode(name, nodes, connection, voltage, charge)

    def _add_electrode(self, name, nodes, connection, voltage, charge):
        if not self.piezoelectric:
            raise ValueError(f"{self.system.label()}: electrode '{name}' needs piezoelectric=True")
        if connection not in ELECTRODE_CONNECTIONS:
            raise ValueError(f"electrode connection must be one of {', '.join(ELECTRODE_CONNECTIONS)}, got "
                             f'{connection!r}')
        electrode = Electrode(str(name), node_potential_dofs(nodes), connection, float(voltage), float(charge))
        self._electrodes.append(electrode)
        self.system.operator_dirty = True   # the held / tied potentials change the reduced operator
        self._loads_dirty = True
        return electrode

    @property
    def electrodes(self) -> tuple:
        """piezo.Electrode of the electrode features, in apply order."""
        return tuple(self._electrodes)

    def electrode_values(self, electrode) -> tuple:
        """(V [V], Q [C/m]) of an electrode in the last static solve: its potential and the net free charge on it (a
        voltage electrode: the charge its source delivered)."""
        if electrode.name not in self._electrode_values:
            raise RuntimeError(f"{self.system.label()}: electrode '{electrode.name}' values need a static solve first")
        return self._electrode_values[electrode.name]

    def reaction(self, region) -> tuple:
        """Base reaction; piezoelectric: the u rows of A x - f of the last static solve (the coupling term included);
        times thickness when set (N instead of N/m)."""
        if not self.piezoelectric:
            force_x, force_y = super().reaction(region)
        else:
            if self._piezo_reaction is None:
                raise RuntimeError(f'{self.system.label()}: reactions need a solve first')
            totals = np.where(to_host(region.dof_mask), self._piezo_reaction, 0.0).sum(axis=(0, 1))
            force_x, force_y = float(totals[0]), float(totals[1])
        depth = 1.0 if self.thickness is None else self.thickness
        return force_x * depth, force_y * depth

    # --- eigenfrequency study hooks -------------------------------------------

    def solve_eigenfrequencies(self, n_resonances):
        """eigenfrequency study hook: up to n_resonances vibration modes nearest target_frequency (modal.solve_modes with
        the 'eigen' solver on the host float64 stiffness and consistent mass of the solid cells; guided: the 2.5D pencil
        at axial_wavenumber) -> ModalSolve, kept as eigen_solve. solve_eigenfrequencies_targets / _band
        (modal.MultiShiftModes) share the _modal_pencil."""
        self.eigen_solve = None
        pencil, finish = self._modal_pencil(self.target_frequency)
        try:
            solve = pencil.solve_at(2.0 * math.pi * self.target_frequency, n_resonances)
        finally:
            pencil.release()
        self.eigen_solve = finish(solve)
        return self.eigen_solve

    def _modal_pencil(self, lowest_target):
        """(modal.ModalPencil, finish) of the host float64 stiffness and consistent mass of the solid cells (supports
        held; piezoelectric: the coupled pencil, voltage electrodes short-circuited, floating ones open, finish expands
        the modes; guided: the 2.5D pencil at axial_wavenumber); lowest_target unused (no free rigid motions here)."""
        if self.guided:
            return self._guided_pencil(), lambda solve: solve
        if self.out_of_plane == GENERALIZED_PLANE_STRAIN:
            raise ValueError(f'{self.system.label()}: generalized plane strain has no vibration modes (the uniform '
                             'eps_zz carries no mass): use plane_strain or plane_stress')
        system = self.system
        if system.operator_dirty:
            self._assemble()
        if self.piezoelectric:   # coupled pencil: voltage electrodes short-circuited, floating ones open
            solid_density = system.xp.where(self._solid, self._solid_density(), 0.0)
            self.modal_mass = coupled_mass(to_host(solid_density).astype(np.float64), system.dx_window,
                                           system.dy_window)
            piezo = self._piezo_system
            return piezo.modal_pencil(self.modal_mass, self.eigen_solver), piezo.expanded
        if self._harmonic_operators is None:   # host float64 K / M, never the solve-precision device CSR
            self._harmonic_operators = self._build_harmonic_operators()
        stiffness, _, self.modal_mass = self._harmonic_operators
        return (ModalPencil(stiffness, self.modal_mass, to_host(system.free) > 0.5, 2, self.eigen_solver),
                lambda solve: solve)

    def _solid_density(self):
        """Window rho in the solve dtype; ValueError when a solid cell has none (rho <= 0)."""
        system = self.system
        density = self.domain.materials.rho[system.window].astype(system.dtype)
        massless = self._solid & ~(density > 0)
        if bool(massless.any()):
            raise ValueError(f'{system.label()}: {int(massless.sum())} solid cell(s) (youngs_modulus or c11 > 0) have '
                             'no density rho > 0: an eigenfrequency or frequency_response study needs the mass of '
                             'every solid cell')
        return density

    def _guided_coefficients(self):
        """Element (lambda, mu) and solid cells from the current materials, without the plane-strain assembly; at k_z =
        0 the plane-strain rigid-mode check and the antiplane support check (both raise)."""
        lame_lambda, mu, solid = self._element_lame()
        self._stiffness = element_stiffness(lame_lambda, mu, self._tensor_arrays(), self.system.dtype)
        if self._stiffness.anisotropic:
            raise ValueError(f'{self.system.label()}: guided (2.5D) modes need isotropic cells (youngs_modulus / '
                             'poisson_ratio); a material sets c_ij entries')
        self._lame = (lame_lambda, mu)
        self._solid = solid
        if self.axial_wavenumber == 0.0:
            self.system._check_rigid_modes(solid)
            check_antiplane_supports(to_host(solid), to_host(self._held_axial), self.system.label)

    def _guided_pencil(self):
        """ModalPencil of the guided (2.5D) modes at k_z = axial_wavenumber: guided_modes host pencil of the solid
        cells, free = touching a solid cell and not held (u_z by Fixed / Fixed (domain)), 3 components per node."""
        system = self.system
        if self.axial_wavenumber is None:
            raise ValueError(f'{system.label()}: the guided vibration modes have no k_z yet: the optomechanics study '
                             'sets 2 Re beta of the optical mode (backward scattering) before this solve')
        self._guided_coefficients()
        density = system.xp.where(self._solid, self._solid_density(), 0.0)
        lame_lambda, mu = (to_host(values).astype(np.float64) for values in self._lame)
        stiffness, self.modal_mass = guided_operators(lame_lambda, mu, to_host(density).astype(np.float64),
                                                      system.dx_window, system.dy_window, self.axial_wavenumber)
        free = guided_free_dofs(stiffness, to_host(system.held), to_host(self._held_axial))
        return ModalPencil(stiffness, self.modal_mass, free, GUIDED_COMPONENTS, self.eigen_solver,
                           axial_wavenumber=self.axial_wavenumber)

    def modal_overlap_data(self):
        """Host data the g0 overlap (optomechanics) reads after an eigenfrequency solve: window slices, node shape,
        window widths / node edges [m], solid cells, photoelastic p11 / p12 / p44 (NaN = unset), the consistent mass of
        that solve and its axial_wavenumber (guided modes: 3 dofs per node; None = plane strain)."""
        if self.eigen_solve is None or self.modal_mass is None:
            raise ValueError(f'{self.system.label()}: no eigenfrequency solve yet')
        if self.piezoelectric:
            raise ValueError(f'{self.system.label()}: the optomechanical overlap of a piezoelectric solid is not '
                             'modelled (its modes carry the potential as a third dof per node)')
        if self._axisymmetric or self.out_of_plane != PLANE_STRAIN:
            raise ValueError(f'{self.system.label()}: the optomechanical overlap needs planar plane strain')
        system = self.system
        window_x, window_y = system.window
        x_edges, y_edges = self.domain.cell_edges()
        materials = self.domain.materials
        return {'window': system.window, 'node_shape': system.node_shape, 'dx': system.dx_window,
                'dy': system.dy_window, 'x_nodes': np.asarray(x_edges[window_x.start:window_x.stop + 1]),
                'y_nodes': np.asarray(y_edges[window_y.start:window_y.stop + 1]),
                'solid': np.array(to_host(self._solid)),
                'p11': to_host(materials.photoelastic_p11[system.window]).astype(np.float64),
                'p12': to_host(materials.photoelastic_p12[system.window]).astype(np.float64),
                'p44': to_host(materials.photoelastic_p44[system.window]).astype(np.float64),
                'mass': self.modal_mass, 'axial_wavenumber': self.axial_wavenumber if self.guided else None}

    def select_resonance(self, number):
        """Show mode `number` (1-based) of the last eigenfrequency solve in the outputs: displacement, strain and the
        modal stress (no thermal part) of the peak-normalised mode shape."""
        resonance = self.eigen_solve.resonances[number - 1]
        system = self.system
        xp = system.xp
        if self.guided:   # (u_x, u_y) into the plane-strain unknown, the u_z amplitude w beside it
            nodes = resonance.displacement.reshape(*system.node_shape, GUIDED_COMPONENTS)
            xp.copyto(system.unknown, xp.asarray(nodes[..., :2].reshape(-1), dtype=system.dtype))
            self._axial_nodes = xp.array(nodes[..., 2], dtype=system.dtype)
        elif self.piezoelectric:   # (u_x, u_y) into the unknown, the mode's phi beside it
            nodes = resonance.displacement.reshape(*system.node_shape, PIEZO_COMPONENTS)
            xp.copyto(system.unknown, xp.asarray(nodes[..., :DISPLACEMENT_COMPONENTS].reshape(-1), dtype=system.dtype))
            self._potential_nodes = xp.array(nodes[..., DISPLACEMENT_COMPONENTS], dtype=system.dtype)
        else:
            xp.copyto(system.unknown, xp.asarray(resonance.displacement, dtype=system.dtype))
        self._write_outputs(thermal=False)
        for destination, source in self._pml_copies:
            xp.copyto(destination, source)
        self.selected_resonance = number

    # --- frequency_response study hooks -----------------------------------------

    def reset_frequency_response(self):
        """Forget the FrequencyPoints and any reduced model of an earlier frequency_response run (the study calls it
        first)."""
        self.frequency_points = []
        self._reduced_harmonic, self.reduced_sweep = None, None
        self._field_policy, self._unpublished_frequency, self._unpublished_solution = None, None, None
        if self._piezo_system is not None:
            self._piezo_system.clear_harmonic_plan()

    def _harmonic_inputs(self):
        """(operator, loss, mass, load, held / fixed values) of a driven solve: the checks, assembly and operators."""
        system = self.system
        if self.guided or self.out_of_plane == GENERALIZED_PLANE_STRAIN:
            raise ValueError(f'{system.label()}: the frequency response is not modelled for guided (2.5D) modes or '
                             'generalized plane strain (the uniform eps_zz carries no mass)')
        if system.operator_dirty:
            self._assemble()
        # the drive: tractions and body forces; the thermal strain is static, never a harmonic drive
        system.build_rhs(self._total_load(thermal=False))
        self._loads_dirty = True   # load_full now holds the drive: the next step rebuilds the static load
        if self._harmonic_operators is None:
            self._harmonic_operators = self._build_harmonic_operators()
        operator, loss, mass = self._harmonic_operators
        load = to_host(system.load_full).astype(np.float64)
        held = to_host(system.held_values if self.piezoelectric else system.fixed_values).astype(np.float64)
        return operator, loss, mass, load, held

    def _harmonic_solve(self, omega, fields=True):
        """(x, A x) at omega [rad/s]: a private complex SuperLU (frequency_response.solve_harmonic), the PiezoSystem's
        HarmonicSystem, or the plan_frequency_response model (fields False: exact only at the electrode rows)."""
        operator, loss, mass, load, held = self._harmonic_inputs()
        if self.piezoelectric:
            return self._piezo_system.solve_harmonic(omega, mass, loss, load, held, self._piezo_counters, fields=fields)
        if self._reduced_harmonic is not None:
            return self._reduced_harmonic.solve(omega, held, load, fields=fields)
        solution, product, host_bytes = solve_harmonic(operator, loss, mass, omega, to_host(self.system.free) > 0.5,
                                                       held, load)
        counters = self._harmonic_counters
        counters['harmonic_factorizations'] += 1
        counters['harmonic_factor_host_bytes'] = max(counters['harmonic_factor_host_bytes'], host_bytes)
        return solution, product

    def plan_frequency_response(self, frequencies, options):
        """ReducedSweepSolver hook (after reset_frequency_response): options {method, tolerance, moments, max_points,
        modal_count, fields}; 'direct' -> None (only the field policy, R6), else a reduced model of the pencil
        (piezoelectric: of the PiezoSystem's HarmonicSystem, electrode rows precomputed) over frequencies [Hz] -> its
        info dict."""
        method = options.get('method', 'direct')
        if method == 'direct':
            self._field_policy = field_policy(options, frequencies)
            return None
        frequencies = np.asarray(frequencies, dtype=np.float64).ravel()
        if not (frequencies.size and np.all(np.isfinite(frequencies)) and np.all(frequencies > 0.0)):
            raise ValueError(f'{self.system.label()}: a reduced frequency sweep needs frequencies > 0 Hz')
        operator, loss, mass, load, held = self._harmonic_inputs()
        omegas = 2.0 * math.pi * frequencies
        reduced = dict(tolerance=options.get('tolerance', 1e-6), moments=options.get('moments', 4),
                       max_points=options.get('max_points', 40), modal_count=options.get('modal_count', 20),
                       eigen_solver=self.eigen_solver, label=self.system.label())
        if self.piezoelectric:
            info = self._piezo_system.plan_harmonic(omegas, method, mass, loss, load, held, self._piezo_counters,
                                                    **reduced)
        else:
            harmonic = HarmonicSystem(operator, loss, mass, to_host(self.system.free) > 0.5)
            info = harmonic.plan_reduced(omegas, method, held, load, **reduced)
            harmonic.set_output_rows(np.zeros(0, dtype=np.int64))
            counters = self._harmonic_counters
            counters['harmonic_factorizations'] += harmonic.factorizations
            counters['harmonic_factor_host_bytes'] = max(counters['harmonic_factor_host_bytes'],
                                                         int(getattr(harmonic.factor, 'host_bytes', 0) or 0))
            self._reduced_harmonic = harmonic
        self.reduced_sweep = info
        self._field_policy = field_policy(options, frequencies)
        return info

    def solve_frequency_response(self, frequency):
        """frequency_response study hook: the driven harmonic response at frequency [Hz] > 0 of every load, prescribed
        displacement and electrode (amplitudes at phase 0) with the mechanical / dielectric losses (one complex host
        SuperLU, frequency_response.solve_harmonic, or the plan_frequency_response model); appends a FrequencyPoint and
        shows the response (real parts, u_abs = the amplitude |u|) where the field policy asks."""
        frequency = float(frequency)
        if not (math.isfinite(frequency) and frequency > 0.0):
            raise ValueError(f'{self.system.label()}: a frequency response needs a frequency > 0 Hz, got {frequency!r}')
        omega = 2.0 * math.pi * frequency
        fields = wants_fields(self._field_policy, frequency)
        planned = self.reduced_sweep is not None
        solution, product = self._harmonic_solve(omega, fields or not planned)
        electrodes = self._piezo_system.electrode_values(solution, product) if self.piezoelectric else {}
        self.frequency_points.append(FrequencyPoint(frequency, electrodes))
        if fields:
            self._show_frequency_response(solution)
        self._unpublished_frequency = None if fields else frequency
        self._unpublished_solution = None if fields or planned else solution

    def publish_frequency_response(self):
        """ReducedSweepSolver hook: show the last solved frequency when the field policy skipped it (the kept direct
        solution, or a re-solve of the reduced model)."""
        frequency, self._unpublished_frequency = self._unpublished_frequency, None
        solution, self._unpublished_solution = self._unpublished_solution, None
        if frequency is None:
            return
        if solution is None:
            solution = self._harmonic_solve(2.0 * math.pi * frequency)[0]
        self._show_frequency_response(solution)

    def _show_frequency_response(self, solution):
        """Show a complex response over every dof (piezoelectric: with its potential)."""
        system = self.system
        if self.piezoelectric:
            nodes = solution.reshape(*system.node_shape, PIEZO_COMPONENTS)
            self._show_harmonic(nodes[..., :DISPLACEMENT_COMPONENTS], nodes[..., DISPLACEMENT_COMPONENTS])
        else:
            self._show_harmonic(solution.reshape(*system.node_shape, DISPLACEMENT_COMPONENTS), None)

    def _build_harmonic_operators(self):
        """Host float64 (stiffness or the coupled operator, loss operator or None, consistent mass) of the last
        assembly for the frequency response: loss = eta K (+ tan delta K_phiphi with piezoelectric)."""
        import scipy.sparse
        system = self.system
        solid = self._solid
        density = to_host(system.xp.where(solid, self._solid_density(), 0.0)).astype(np.float64)
        materials = self.domain.materials
        names = set(materials.property_names())
        eta_name, tan_name = LOSS_ARRAYS
        eta = (to_host(system.xp.where(solid, materials.mechanical_loss_factor[system.window], 0.0)).astype(np.float64)
               if eta_name in names else None)
        tan_delta = (to_host(materials.dielectric_loss_tangent[system.window]).astype(np.float64)
                     if tan_name in names and self.piezoelectric else None)
        if self.piezoelectric:
            mass = coupled_mass(density, system.dx_window, system.dy_window)
            loss = None
            if eta is not None or tan_delta is not None:
                elastic = tuple(to_host(values).astype(np.float64) * (0.0 if eta is None else eta)
                                for values in operator_coefficients(self._stiffness, False))
                zeros = np.zeros(system.element_shape)
                permittivity = tuple(-to_host(getattr(materials, name)[system.window]).astype(np.float64)
                                     * (0.0 if tan_delta is None else tan_delta) for name in PERMITTIVITY_ARRAYS)
                loss = coupled_operator(elastic, (zeros,) * len(PIEZO_ARRAYS), permittivity, system.dx_window,
                                        system.dy_window)
            return self._piezo_system.operator, loss, mass
        pattern = ElasticPattern(*system.node_shape, xp=np)
        if self._axisymmetric:
            mass_values = axisymmetric_mass_stencil(density, system.dx_window, system.dy_window, system.r_nodes[:-1],
                                                    np.float64, xp=np)
        else:
            mass_values = mass_stencil(density, system.dx_window, system.dy_window, np.float64, xp=np)
        mass = pattern.csr(mass_values, scipy.sparse)
        loss = None if eta is None else pattern.csr(self._host_stiffness_stencil(eta), scipy.sparse)
        return pattern.csr(self._host_stiffness_stencil(None), scipy.sparse), loss, mass

    def _host_stiffness_stencil(self, scale):
        """Host float64 stencil of the assembled formulation's stiffness, every coefficient times the window array scale
        (None = 1): the frequency response's K and its loss part eta K."""
        system = self.system
        stiffness = self._stiffness
        factor = 1.0 if scale is None else scale
        coefficients = operator_coefficients(stiffness, self.out_of_plane == PLANE_STRESS)
        c11, c12, c22, c66, c16, c26 = (to_host(values).astype(np.float64) * factor for values in coefficients)
        if self._axisymmetric:
            hoop = tuple(to_host(stiffness[name]).astype(np.float64) * factor for name in ('c13', 'c23', 'c33'))
            return gauss_stiffness_stencil(c11, c12, c22, c66, system.dx_window, system.dy_window, np.float64,
                                           element_r_lo=system.r_nodes[:-1], hoop=hoop, xp=np)
        monoclinic = stiffness.monoclinic
        return stiffness_stencil(c11, c12, c22, c66, system.dx_window, system.dy_window, np.float64, xp=np,
                                 c16=c16 if monoclinic else None, c26=c26 if monoclinic else None)

    def _show_harmonic(self, displacement, potential):
        """Write the real parts of a complex nodal response ((nx_nodes, ny_nodes, 2) u, (nx_nodes, ny_nodes) phi or
        None) into the outputs, u_abs = the amplitude of the complex centre displacement; reactions and static electrode
        values become unavailable."""
        system = self.system
        xp = system.xp
        xp.copyto(system.unknown, xp.asarray(displacement.real.reshape(-1), dtype=system.dtype))
        if potential is not None:
            self._potential_nodes = xp.array(potential.real, dtype=system.dtype)
        self._piezo_reaction = None
        self._electrode_values = {}
        self._write_outputs()
        centre = 0.25 * (displacement[:-1, :-1] + displacement[1:, :-1] + displacement[:-1, 1:] + displacement[1:, 1:])
        amplitude = xp.asarray(np.sqrt(np.sum(np.abs(centre) ** 2, axis=-1)), dtype=DTYPE)
        self.fields.u_abs[system.window] = xp.where(self._solid, amplitude, 0.0)
        for destination, source in self._pml_copies:
            xp.copyto(destination, source)
        self.selected_resonance = None

    # --- base-class hooks -----------------------------------------------------

    def _new_fields(self, grid_shape):
        if len(grid_shape) != 2:
            raise ValueError(f'{self.system.label()} needs a 2D grid, got shape {grid_shape} (3D: solid_mechanics_3d)')
        return MechanicsFields2D(*grid_shape, self.system.xp)

    def _pml_copy_pairs(self):
        window_x, window_y = self.system.window
        x_range, y_range = (window_x.start, window_x.stop), (window_y.start, window_y.stop)
        return [pair for name in OUTPUT_ARRAYS
                for pair in edge_extension_views(getattr(self.fields, name), x_range, y_range)]

    def _assemble(self):
        """Element stiffness (stiffness.ElementStiffness: E / nu or the c_ij entries; c11 <= 0 is void) and the CSR
        assembly (rigid-mode check): plane strain / generalized plane strain C, plane stress the reduced C (sigma_zz =
        0), axisymmetric C with the hoop terms (orthotropic only); piezoelectric: also the host coupled operator with its
        electrode reduction."""
        stiffness = self._element_stiffness()
        self._stiffness = stiffness
        self._solid = stiffness.solid
        self._unit_axial_solve = None
        self._harmonic_operators = None
        c11, c12, c22, c66, c16, c26 = operator_coefficients(stiffness, self.out_of_plane == PLANE_STRESS)
        if self._axisymmetric:
            self.system.assemble(c11, c12, c22, c66, stiffness.solid,
                                 hoop=(stiffness['c13'], stiffness['c23'], stiffness['c33']))
        else:
            self.system.assemble(c11, c12, c22, c66, stiffness.solid, c16=c16 if stiffness.monoclinic else None,
                                 c26=c26 if stiffness.monoclinic else None)
        if self.piezoelectric:
            self._assemble_piezo()

    def _tensor_arrays(self):
        """{c_ij: window view of its material array, None where the array is not allocated (= the isotropic value)}."""
        materials = self.domain.materials
        names = set(materials.property_names())
        return {name: getattr(materials, name)[self.system.window] if name in names else None
                for name in STIFFNESS_TENSOR_ARRAYS}

    def _element_stiffness(self):
        """ElementStiffness of the current materials; ValueError for a c_ij set that is not positive definite (the
        block the formulation reads) and for c16 / c26 / c36 in an axisymmetric solid."""
        lame_lambda, mu, _ = self._element_lame()
        stiffness = element_stiffness(lame_lambda, mu, self._tensor_arrays(), self.system.dtype)
        label = self.system.label
        if self._axisymmetric:
            if stiffness.monoclinic:
                raise ValueError(f'{label()}: an axisymmetric solid takes orthotropic c_ij only (c16 = c26 = c36 = '
                                 '0): the hoop terms have no shear coupling')
            check_positive_definite(stiffness, '1236', label, 'axisymmetric: rr, zz, hoop and rz shear')
        elif self.out_of_plane == PLANE_STRAIN:
            check_positive_definite(stiffness, '126', label, 'plane strain: in-plane block')
        else:
            check_positive_definite(stiffness, '1236', label, f'{self.out_of_plane}: with the zz row')
        return stiffness

    def _assemble_piezo(self):
        """piezo.PiezoSystem of the current materials (e of the solid cells, eps_dc of every window cell) and
        electrodes; ValueError for a permittivity that is not positive definite."""
        system = self.system
        materials = self.domain.materials
        window = system.window
        solid = self._solid
        names = set(materials.property_names())
        self._piezo_coefficients = tuple(
            system.xp.where(solid, getattr(materials, name)[window], 0.0).astype(system.dtype) if name in names
            else system.xp.zeros(system.element_shape, dtype=system.dtype)
            for name in PIEZO_ARRAYS + PIEZO_OUT_OF_PLANE_ARRAYS)
        eps_xx, eps_yy, eps_xy = (getattr(materials, name)[window].astype(np.float64) for name in PERMITTIVITY_ARRAYS)
        indefinite = ~((eps_xx > 0.0) & (eps_yy > 0.0) & (eps_xx * eps_yy > eps_xy * eps_xy))
        if bool(indefinite.any()):
            raise ValueError(f'{system.label()}: {int(indefinite.sum())} window cell(s) have a static permittivity that '
                             'is not positive definite (need eps_dc_xx > 0, eps_dc_yy > 0 and eps_dc_xy^2 < eps_dc_xx '
                             'eps_dc_yy)')
        elastic = tuple(to_host(values).astype(np.float64) for values in operator_coefficients(self._stiffness, False))
        operator = coupled_operator(elastic, tuple(to_host(values).astype(np.float64)
                                                   for values in self._piezo_coefficients[:len(PIEZO_ARRAYS)]),
                                    tuple(to_host(values) for values in (eps_xx, eps_yy, eps_xy)),
                                    system.dx_window, system.dy_window)
        displacement_dofs, potential_dofs = interleaved_layout(operator.shape[0] // PIEZO_COMPONENTS)
        self._piezo_system = PiezoSystem(operator, self._electrodes, displacement_dofs, potential_dofs,
                                         np.array(to_host(system.held)), system.label)

    def _solve_system(self):
        """PCG of the plane-strain system; piezoelectric: the host SuperLU solve of the coupled system (u into the
        unknown, phi, reactions and electrode values kept); generalized plane strain: + the eps_zz terms."""
        if self.out_of_plane == GENERALIZED_PLANE_STRAIN:
            self._solve_generalized()
            return
        if not self.piezoelectric:
            self.system.solve()
            return
        system = self.system
        xp = system.xp
        piezo_system = self._piezo_system
        solution, product = piezo_system.solve_static(to_host(system.load_full).astype(np.float64),
                                                      to_host(system.held_values).astype(np.float64),
                                                      self._piezo_counters)
        nodes = solution.reshape(*system.node_shape, PIEZO_COMPONENTS)
        xp.copyto(system.unknown, xp.asarray(nodes[..., :DISPLACEMENT_COMPONENTS].reshape(-1), dtype=system.dtype))
        self._potential_nodes = xp.array(nodes[..., DISPLACEMENT_COMPONENTS], dtype=system.dtype)
        load = to_host(system.load_full).astype(np.float64).reshape(*system.node_shape, DISPLACEMENT_COMPONENTS)
        self._piezo_reaction = (product.reshape(*system.node_shape, PIEZO_COMPONENTS)[..., :DISPLACEMENT_COMPONENTS]
                                - load)
        self._electrode_values = piezo_system.electrode_values(solution, product)
        system.direct_solves += 1

    def _solve_generalized(self):
        """Generalized plane strain eps_zz = e0 + e1 x + e2 y (the free terms of the bending setting) by superposition
        (generalized_plane_strain.py): the eps_zz = 0 solve plus unit-term solves kept until the next assembly; the
        axial force and both bending moments hold exactly for bilinear u. The kept load gains sum e_k f_k, so reactions
        stay K u - f."""
        system = self.system
        stiffness = self._stiffness
        system.solve()
        if self._unit_axial_solve is None:   # in-plane stress per unit eps_zz: c13, c23, c36
            self._unit_axial_solve = unit_strain_solves(
                system, (stiffness.entry('1', '3'), stiffness.entry('2', '3'), stiffness.entry('3', '6')),
                stiffness['c33'], self.bending_terms)
        _, _, beta_z, _ = self._thermal_coefficients()
        geometry = self._unit_axial_solve.geometry
        coefficients = solve_strain_terms(system, self._unit_axial_solve, beta_z * self._temperature_change(),
                                          self.axial_force, *self.bending_moments)
        self._axial_strain = terms_about_origin(geometry, coefficients)
        self._axial_strain_field = geometry.strain_field(coefficients)

    def _build_rhs(self):
        self.system.build_rhs(self._total_load())

    def _expansion_coefficients(self):
        """(alpha_xx, alpha_yy, alpha_zz) [1/K] per element in the solve dtype."""
        materials = self.domain.materials
        window = self.system.window
        dtype = self.system.dtype
        return tuple(getattr(materials, name)[window].astype(dtype) for name in ('alpha_xx', 'alpha_yy', 'alpha_zz'))

    def _thermal_coefficients(self, plane_stress=False):
        """(beta_x, beta_y, beta_z, beta_xy) [Pa/K] per element: the 3D thermal stress C : alpha per kelvin (z = the
        out-of-plane axis; axisymmetric: x = r, y = z, z = the hoop theta); plane_stress: beta_x / _y / _xy reduced by
        sigma_zz = 0."""
        return thermal_coefficients(self._stiffness, *self._expansion_coefficients(), plane_stress=plane_stress)

    def _out_of_plane_strain(self, strain_xx, strain_yy, shear, centre_x, temperature_change):
        """eps_zz per element: plane strain 0, plane stress the sigma_zz = 0 value (beta_z dT - c31 eps_xx - c32 eps_yy
        - c36 g) / c33, generalized plane strain e0 + e1 x + e2 y at the centres, axisymmetric the hoop u_r / r_c."""
        if self._axisymmetric:
            return centre_x / self.system.r_centres
        if self.out_of_plane == PLANE_STRESS:
            _, _, beta_z, _ = self._thermal_coefficients()
            return plane_stress_strain(self._stiffness, strain_xx, strain_yy, shear, beta_z * temperature_change)
        if self.out_of_plane == GENERALIZED_PLANE_STRAIN:
            return self._axial_strain_field
        return 0.0

    def _temperature_change(self):
        window = self.system.window
        return (self.fields.temperature_rise[window] + self.fields.temperature_rise_ext[window]).astype(
            self.system.dtype)

    def _total_load(self, thermal=True):
        """Nodal load [N/m; axisymmetric N]: features' tractions + body forces (applied + coupled, solid cells only) +
        (thermal) the thermal load integral of B^T C eps_th per element (plane stress: the reduced thermal stress beta_I
        - c_I3 beta_z / c33; axisymmetric: with the hoop part; monoclinic: with the shear part beta_xy)."""
        system = self.system
        xp = system.xp
        window = system.window
        dtype = system.dtype
        load = system.applied_load.copy()
        if thermal:
            beta_x, beta_y, beta_z, beta_xy = self._thermal_coefficients(plane_stress=self.out_of_plane == PLANE_STRESS)
            temperature_change = self._temperature_change()
            system.add_stress_load(load, beta_x * temperature_change, beta_y * temperature_change,
                                   beta_z * temperature_change if self._axisymmetric else None,
                                   stress_xy=beta_xy * temperature_change if self._stiffness.monoclinic else None)

        fields = self.fields
        for component, coupled in ((0, fields.body_force_x), (1, fields.body_force_y)):
            density = coupled[window].astype(dtype) + self._applied_body_force[component]
            system.add_volume_load(load[..., component], xp.where(self._solid, density, 0.0))
        return load

    def _write_outputs(self, thermal=True):
        """Displacement, strain, stress and von Mises at the cell centres of the window (0 in void cells; thermal False:
        dT = 0, a mode shape; a shown guided mode adds u_z = its quadrature amplitude and strain_zz = k_z u_z, the phi
        = 0 values; the _zz outputs are the out-of-plane, axisymmetric the hoop components); the base step then
        edge-extends them into the PML."""
        system = self.system
        window = system.window
        solid = self._solid
        stiffness = self._stiffness
        centre, d_dx, d_dy = system.element_kinematics()
        strain_xx = d_dx[..., 0]
        strain_yy = d_dy[..., 1]
        shear = d_dy[..., 0] + d_dx[..., 1]   # engineering shear g = 2 strain_xy
        strain_xy = 0.5 * shear               # tensor shear strain
        axial = self._axial_nodes if not thermal else None
        beta_x, beta_y, beta_z, beta_xy = self._thermal_coefficients()
        temperature_change = self._temperature_change() if thermal else 0.0
        if axial is None:   # the out-of-plane (axisymmetric: hoop) strain of the condition
            u_z = 0.0
            strain_zz = self._out_of_plane_strain(strain_xx, strain_yy, shear, centre[..., 0], temperature_change)
        else:
            u_z = 0.25 * (axial[:-1, :-1] + axial[1:, :-1] + axial[:-1, 1:] + axial[1:, 1:])
            strain_zz = self.axial_wavenumber * u_z
        stress_xx, stress_yy, stress_zz, stress_xy = elastic_stress(stiffness, strain_xx, strain_yy, strain_zz, shear)
        stress_xx = stress_xx - beta_x * temperature_change
        stress_yy = stress_yy - beta_y * temperature_change
        stress_xy = stress_xy - beta_xy * temperature_change
        if self.out_of_plane == PLANE_STRESS:
            stress_zz = 0.0   # = the expression up to rounding
        else:
            stress_zz = stress_zz - beta_z * temperature_change
        electric = self._electric_outputs()
        if electric is not None:   # T_J -= e_kJ E_k (J = xx, yy, xy, zz: the e_k3 column only enters stress_zz)
            _, field_x, field_y = electric
            e11, e12, e16, e21, e22, e26, e13, e23 = self._piezo_coefficients
            stress_xx = stress_xx - (e11 * field_x + e21 * field_y)
            stress_yy = stress_yy - (e12 * field_x + e22 * field_y)
            stress_xy = stress_xy - (e16 * field_x + e26 * field_y)
            stress_zz = stress_zz - (e13 * field_x + e23 * field_y)
            for name, values in zip(ELECTRIC_OUTPUT_ARRAYS, electric):
                getattr(self.fields, name)[window] = values
        xp = system.xp
        von_mises = xp.sqrt(0.5 * ((stress_xx - stress_yy) ** 2 + (stress_yy - stress_zz) ** 2
                                   + (stress_zz - stress_xx) ** 2) + 3.0 * stress_xy ** 2)
        alpha_xx, alpha_yy, alpha_zz = self._expansion_coefficients()
        outputs = {'u_x': centre[..., 0], 'u_y': centre[..., 1], 'u_z': u_z,
                   'u_abs': xp.sqrt(centre[..., 0] ** 2 + centre[..., 1] ** 2 + u_z ** 2),
                   'strain_xx': strain_xx, 'strain_yy': strain_yy, 'strain_xy': strain_xy,
                   'elastic_strain_xx': strain_xx - alpha_xx * temperature_change,
                   'elastic_strain_yy': strain_yy - alpha_yy * temperature_change,
                   'elastic_strain_zz': strain_zz - alpha_zz * temperature_change,   # axisymmetric: the hoop part
                   'stress_xx': stress_xx, 'stress_yy': stress_yy, 'stress_xy': stress_xy, 'stress_zz': stress_zz,
                   'von_mises': von_mises}
        for name, values in outputs.items():
            getattr(self.fields, name)[window] = xp.where(solid, values, 0.0)

    def _electric_outputs(self):
        """(phi [V], E_x, E_y [V/m]) at the window cell centres (mean of the 4 corners, E = -grad phi of the bilinear
        element) of the shown piezoelectric solve / mode; None without one."""
        potential = self._potential_nodes if self.piezoelectric else None
        if potential is None:
            return None
        system = self.system
        south_west, south_east = potential[:-1, :-1], potential[1:, :-1]
        north_east, north_west = potential[1:, 1:], potential[:-1, 1:]
        centre = 0.25 * (south_west + south_east + north_east + north_west)
        field_x = -((south_east - south_west) + (north_east - north_west)) / (2.0 * system.dx_device[:, None])
        field_y = -((north_west - south_west) + (north_east - south_east)) / (2.0 * system.dy_device[None, :])
        return centre, field_x, field_y
