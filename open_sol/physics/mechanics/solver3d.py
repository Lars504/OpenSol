"""Static 3D solid mechanics on a Domain3D: small strain, isotropic linear elastic cells, thermal strain from the
temperature-rise slot, supports, tractions and body forces; one ElasticSystem3D solve per change; the lowest vibration
modes for the eigenfrequency study (matrix-free shift-invert). Forces in N."""
import math
from collections.abc import Mapping

import numpy as np

from ...core.backend import host_copy, write_host_into
from ...core.box_values import BoxValues
from ...core.constants import DTYPE
from ...core.fv_diffusion_3d import edge_extension_views_3d
from ...core.monitors import FieldProbe
from ...solvers.arpack_shift_invert import ArpackShiftInvertSolver
from .fields3d import INPUT_ARRAYS_3D, OUTPUT_ARRAYS_3D, SOURCE_SLOTS_3D, MechanicsFields3D
from .materials import DENSITY_SCHEMA, MECHANICAL_SCHEMA
from .renderer3d import SolidRenderer3D
from .solver_common import MATERIAL_STATE_ARRAYS, SolidMechanicsBase, _require_published
from .system3d import ElasticSystem3D

_EXPANSION_NAMES = ('alpha_xx', 'alpha_yy', 'alpha_zz')


class _ShownSolution:
    """What the outputs show: the nodal displacement, the (lambda, mu) of its assembly and its free thermal strains
    (ElasticSystem3D.thermal_strains; None = dT = 0). unknown is the live system array until a change of it without a
    new solve copies it (_freeze_unknown); an assembly rebinds lambda / mu, so these stay the solve's."""

    __slots__ = ('unknown', 'lame_lambda', 'mu', 'thermal_strain')

    def __init__(self, unknown, lame_lambda, mu, thermal_strain):
        self.unknown, self.lame_lambda, self.mu, self.thermal_strain = unknown, lame_lambda, mu, thermal_strain


class _SolidState3D(Mapping):
    """state() of SolidMechanics3DPhysics: an output computed on access (SolidMechanics3DPhysics.output: a fresh
    full-grid float32 array per new solution), the inputs and E / nu as stored; unknown keys raise KeyError."""

    def __init__(self, physics):
        self._physics = physics

    def __getitem__(self, name):
        physics = self._physics
        if name in OUTPUT_ARRAYS_3D:
            return physics.output(name)
        if name in INPUT_ARRAYS_3D:
            return getattr(physics.fields, name)
        if name in MATERIAL_STATE_ARRAYS:
            return getattr(physics.domain.materials, name)
        raise KeyError(f"{physics.system.label()}: no state field {name!r}")

    def __contains__(self, name):   # without computing an output
        return name in OUTPUT_ARRAYS_3D or name in INPUT_ARRAYS_3D or name in MATERIAL_STATE_ARRAYS

    def __iter__(self):
        return iter(OUTPUT_ARRAYS_3D + INPUT_ARRAYS_3D + MATERIAL_STATE_ARRAYS)

    def __len__(self):
        return len(OUTPUT_ARRAYS_3D) + len(INPUT_ARRAYS_3D) + len(MATERIAL_STATE_ARRAYS)


class _OutputFields:
    """Attribute view of the computed outputs for FieldProbe.record (getattr(fields, component))."""

    def __init__(self, physics):
        self._physics = physics

    def __getattr__(self, name):
        return self._physics.output(name)


class SolidMechanics3DPhysics(SolidMechanicsBase):
    """3D static elasticity on the physical window (trilinear bricks on the cells, displacement on the cell corners,
    matrix-free 3 x 3 block-Jacobi PCG); outputs at the cell centres, 0 in void cells, edge-extended into any PML shell,
    computed on read (`output`, state()) from the shown solution (_ShownSolution), never kept as 20 arrays. Arrays on
    the domain's backend (gpu: kernels3d RawKernels, cpu: their numpy twins)."""

    material_schema = {**MECHANICAL_SCHEMA, **DENSITY_SCHEMA}
    output_arrays = OUTPUT_ARRAYS_3D
    input_arrays = INPUT_ARRAYS_3D
    source_slots = SOURCE_SLOTS_3D

    def __init__(self, domain, tol: float = 1e-8, max_iter: int = 10000, precision: str = 'float64',
                 compute_residual: bool = False, linear_solver=None, eigen_solver=None):
        system = ElasticSystem3D(precision, float(tol), int(max_iter),
                                 label=lambda: f"solid mechanics 3D '{self.name}'", compute_residual=compute_residual,
                                 linear_solver=linear_solver, backend=domain.backend.name)
        self._owns_eigen_solver = eigen_solver is None   # scripts: a private runtime, reported in work_counters
        self.eigen_solver = ArpackShiftInvertSolver() if eigen_solver is None else eigen_solver
        self.eigen_solve = None          # modal.ModalSolve of the last solve_eigenfrequencies
        self.selected_resonance = None   # 1-based mode shown in the outputs, None = the static solve
        self._shows_mode = False         # an eigen solve ran or a mode is shown since the last static solve
        self._solid = None
        self._shown = None               # _ShownSolution of the outputs, None = nothing solved (zero outputs)
        self._restored = None            # {output: host float32 full grid} of restore_solution until the next solve
        self._output_cache = None        # (name, array) of the latest computed output
        # a temperature input was written (inject_source temperature_rise, add_temperature_change, restore): only then
        # is the shown solution's thermal strain kept (dT = 0 else: the inputs start at 0)
        self._thermal_inputs = False
        self._init_solid(system, domain)

    def renderer(self):
        return SolidRenderer3D(self)

    def work_counters(self) -> dict:
        """The private eigen solver's counters once it ran (a build-injected one reports in its role record)."""
        if self._owns_eigen_solver and self.eigen_solver.n_eigen_solves:
            return self.eigen_solver.work_counters()
        return {}

    # --- eigenfrequency study hooks -------------------------------------------

    def solve_eigenfrequencies(self, n_resonances):
        """eigenfrequency study hook: the n_resonances lowest vibration modes (ElasticSystem3D.lowest_modes:
        matrix-free shift-invert at 0 with the consistent mass of rho on the solid cells) -> ModalSolve."""
        self.eigen_solve = None
        system = self.system
        if system.operator_dirty:
            self._assemble()
        density = self.domain.materials.rho[system.window].astype(system.dtype)
        massless = self._solid & ~(density > 0)
        if bool(massless.any()):
            raise ValueError(f'{system.label()}: {int(massless.sum())} solid cell(s) (youngs_modulus > 0) have no '
                             'density rho > 0: an eigenfrequency study needs the mass of every solid cell')
        del massless
        system.xp.multiply(density, self._solid, out=density)
        self._shows_mode = True   # the inner solves used rhs as scratch: reactions wait for the next static solve
        self.eigen_solve = system.lowest_modes(density, n_resonances, self.eigen_solver)
        return self.eigen_solve

    def select_resonance(self, number):
        """Show mode `number` (1-based) of the last eigenfrequency solve in the outputs: displacement, strain and the
        modal stress (no thermal part) of the peak-normalised mode shape."""
        resonance = self.eigen_solve.resonances[number - 1]
        system = self.system
        xp = system.xp
        self._drop_shown()   # the unknown changes
        xp.copyto(system.unknown, xp.asarray(resonance.displacement, dtype=system.dtype))
        self._show_outputs(thermal=False)
        self._shows_mode = True
        self.selected_resonance = number

    def step(self, t: float = 0.0, dt: float = math.inf) -> None:
        """Base step; a static re-solve ends the eigenfrequency state (reactions allowed again). A dirty operator means
        re-assembly, rhs and solve in this call: the shown solution goes before the assembly, so _assemble makes no
        _freeze_unknown copy of the unknown (a dof vector beside the assembly transients; nothing reads the outputs
        before the new show)."""
        solves = self.system.n_solves
        if self.system.operator_dirty:
            self._drop_shown()
        super().step(t, dt)
        if self.system.n_solves != solves:
            self._shows_mode = False
            self.selected_resonance = None

    def reaction(self, region) -> tuple:
        """Base reaction; RuntimeError after an eigenfrequency solve until the next static solve (its inner solves are
        no static solve)."""
        if self._shows_mode:
            raise RuntimeError(f'{self.system.label()}: support reactions need a static solve; an eigenfrequency '
                               'study has none')
        return super().reaction(region)

    # --- outputs computed on read ---------------------------------------------

    def output(self, name):
        """Full-grid float32 output `name` (OUTPUT_ARRAYS_3D) of the shown solution, computed on read: window cells
        (0 in void cells) edge-extended into any PML shell, zeros before the first solve; the latest one is kept until
        another output is read or the solution changes. Read-only (shared with later readers): copy before writing."""
        if name not in OUTPUT_ARRAYS_3D:
            raise KeyError(f"{self.system.label()}: no output {name!r}; outputs are {', '.join(OUTPUT_ARRAYS_3D)}")
        cached = self._output_cache
        if cached is not None and cached[0] == name:
            return cached[1]
        self._output_cache = None   # one computed output at a time
        system = self.system
        xp = system.xp
        if self._restored is not None:
            array = xp.array(self._restored[name], dtype=DTYPE)   # own copy: the snapshot's arrays stay unwritten
        else:
            array = xp.zeros(self.fields.shape, dtype=DTYPE)
            shown = self._shown
            if shown is not None:
                system.write_output(array, name, shown.unknown, shown.lame_lambda, shown.mu, shown.thermal_strain)
                for destination, source in edge_extension_views_3d(array, system.window):   # in order (corners)
                    xp.copyto(destination, source)
        self._output_cache = (name, array)
        return array

    def state(self) -> Mapping:
        """Outputs (computed on access), inputs and E / nu by name (_SolidState3D)."""
        return _SolidState3D(self)

    def solution_state(self, materials=False) -> dict:
        """Host copies of every output (computed one at a time) and input field; materials: E / nu too."""
        _require_published(self._unpublished_frequency, self.system.label())
        fields = {name: host_copy(self.output(name)) for name in self.output_arrays}
        fields.update({name: host_copy(getattr(self.fields, name)) for name in self.input_arrays})
        state = {'fields': fields}
        if materials:
            state['materials'] = {name: host_copy(getattr(self.domain.materials, name))
                                  for name in MATERIAL_STATE_ARRAYS}
        return state

    def restore_solution(self, state) -> None:
        """Write a solution_state back: inputs (and E / nu) in place, the outputs served from its host copies until the
        next solve (no re-solve follows a read)."""
        fields = state['fields']
        for name in self.input_arrays:
            write_host_into(getattr(self.fields, name), fields[name])
        for name, values in state.get('materials', {}).items():
            write_host_into(getattr(self.domain.materials, name), values)
        self._drop_shown()
        self._restored = {name: fields[name] for name in self.output_arrays}
        self._thermal_inputs = True   # the restored inputs may hold a temperature change

    def seed_warm_start(self, fields) -> tuple:
        """Base warm start; the shown outputs keep their own displacement (a copy, when one is shown)."""
        self._freeze_unknown()
        return super().seed_warm_start(fields)

    def inject_source(self, name: str, array) -> None:
        super().inject_source(name, array)
        if name == 'temperature_rise':
            self._thermal_inputs = True

    def _drop_shown(self) -> None:
        """Forget the shown solution, restored outputs and the computed output (a new solution follows)."""
        self._shown, self._restored, self._output_cache = None, None, None

    def _freeze_unknown(self) -> None:
        """Copy the shown displacement before the live unknown changes without a new solve (an assembly rewrites its
        fixed values, a warm start its free ones)."""
        shown = self._shown
        if shown is not None and shown.unknown is self.system.unknown:
            shown.unknown = shown.unknown.copy()

    def _record_probes(self, t) -> None:
        """Probes by component (each output computed once per record)."""
        fields = _OutputFields(self)
        for probe in sorted(self.probes, key=lambda probe: probe.field_component):
            probe.record(fields, t)

    # --- supports, loads, probes ----------------------------------------------

    def hold_face(self, face: str, u_x=None, u_y=None, u_z=None):
        """Hold components [m] (None = free) on every node of an outer face (FACES_3D); returns the held dof mask."""
        return self.system.hold_face(face, u_x, u_y, u_z)

    def hold_cells(self, mask, u_x=0.0, u_y=0.0, u_z=0.0):
        """Hold components [m] (None = free) on every corner node of the window cells under the full-grid mask;
        returns the held dof mask."""
        system = self.system
        return system.hold_cells(system.window_mask(mask, 'fixed domain'), u_x, u_y, u_z)

    def add_face_traction(self, face: str, t_x: float, t_y: float, t_z: float) -> None:
        """Add a traction (t_x, t_y, t_z) [Pa] on an outer face: each node gets t times its tributary area [N]."""
        self.system.add_face_traction(face, t_x, t_y, t_z)
        self._loads_dirty = True

    def add_body_force(self, mask, f_x, f_y, f_z) -> None:
        """Add a body force density [N/m^3] (scalars, full-grid or (nx, ny, 1) arrays, BoxValues) on the window cells
        under mask; void cells drop it."""
        system = self.system
        mask_window = system.window_mask(mask, 'body force')
        densities = []
        for component, value in enumerate((f_x, f_y, f_z)):
            if isinstance(value, BoxValues):   # added on its box only
                value.add_masked_to(system.body_force_target()[component], mask_window, system.window)
                densities.append(0.0)
            else:
                densities.append(self._window_values(value, 'body force density'))
        system.add_body_force(mask_window, *densities)
        self._loads_dirty = True

    def add_temperature_change(self, mask, delta_T) -> None:
        """Add a temperature change [K] above the stress-free state (scalar, full-grid or (nx, ny, 1) array, BoxValues)
        to temperature_rise_ext on the window cells under mask."""
        system = self.system
        mask_window = system.window_mask(mask, 'temperature change')
        target = self.fields.temperature_rise_ext[system.window]
        if isinstance(delta_T, BoxValues):   # added on its box only
            delta_T.add_masked_to(target, mask_window, system.window)
        else:
            change = system.xp.asarray(self._window_values(delta_T, 'temperature change'), dtype=DTYPE)
            target += system.xp.where(mask_window, change, DTYPE(0.0))
        self._thermal_inputs = True
        self._loads_dirty = True

    def add_probe(self, ix: int, iy: int, iz: int, component: str = 'u_abs') -> FieldProbe:
        """Record an output field at full-grid cell (ix, iy, iz) whenever step records probes."""
        if component not in OUTPUT_ARRAYS_3D:
            raise ValueError(f"probe component {component!r} is not one of {', '.join(OUTPUT_ARRAYS_3D)}")
        cell = (ix, iy, iz)
        window = self.system.window
        if not all(axis_window.start <= index < axis_window.stop for axis_window, index in zip(window, cell)):
            ranges = ', '.join(f'{axis} [{axis_window.start}, {axis_window.stop})'
                               for axis, axis_window in zip('xyz', window))
            raise ValueError(f'solid probe cell {cell} lies outside the solve window {ranges}')
        probe = FieldProbe(ix, iy, component, iz=iz)
        self.probes.append(probe)
        return probe

    def add_probe_at(self, x: float, y: float, z: float, component: str = 'u_abs') -> FieldProbe:
        """add_probe in the cell containing physical (x, y, z) [m]."""
        self.domain.validate_physical_pos(x, y, z, 'Solid probe')
        return self.add_probe(*self.domain.pos_to_cell(x, y, z), component)

    # --- base-class hooks -----------------------------------------------------

    def prepare(self, domain) -> None:
        """Base prepare (new system buffers): nothing is shown until the next solve."""
        self._drop_shown()
        super().prepare(domain)

    def _new_fields(self, grid_shape):
        if len(grid_shape) != 3:
            raise ValueError(f'{self.system.label()} needs a 3D grid, got shape {grid_shape}')
        return MechanicsFields3D(*grid_shape, self.system.xp)

    def _pml_copy_pairs(self):
        return []   # no output arrays: output() edge-extends each computed one

    def _assemble(self):
        """Element Lame coefficients (cells with E <= 0 are void) and the matrix-free assembly (the shown outputs keep
        the previous lambda / mu and a copy of their displacement)."""
        lame_lambda, mu, solid = self._element_lame()
        self._solid = solid
        self._freeze_unknown()   # assemble rewrites the fixed values of unknown
        self.system.assemble(lame_lambda, mu, solid)

    def _solve_system(self):
        """The base static solve; the shown outputs (and the previous lambda / mu they hold) are dropped first."""
        self._drop_shown()
        super()._solve_system()

    def _expansion_arrays(self):
        """Full-grid float32 alpha_xx / alpha_yy / alpha_zz [1/K] (read by the fused load and output kernels)."""
        materials = self.domain.materials
        return tuple(getattr(materials, name) for name in _EXPANSION_NAMES)

    def _build_rhs(self):
        fields = self.fields
        self.system.build_rhs(self._expansion_arrays(), fields.temperature_rise, fields.temperature_rise_ext,
                              (fields.body_force_x, fields.body_force_y, fields.body_force_z))

    def _write_outputs(self, thermal=True):
        """Show the current solution: the outputs are computed on read from it (output); its free thermal strains are
        kept once a temperature input was written (thermal False: dT = 0, a mode shape)."""
        system = self.system
        if system.lame_lambda is None:
            raise RuntimeError(f'{system.label()}: assemble before the outputs')
        self._drop_shown()   # the previous thermal strains go first
        strains = None
        if thermal and self._thermal_inputs:
            fields = self.fields
            strains = system.thermal_strains(self._expansion_arrays(), fields.temperature_rise,
                                             fields.temperature_rise_ext)
        self._shown = _ShownSolution(system.unknown, system.lame_lambda, system.mu, strains)

    def _window_values(self, value, label):
        """A scalar as float, or the window part of a full-grid (nx, ny, nz) or xy-layer (nx, ny, 1) array (callers add
        BoxValues on their box themselves)."""
        if np.ndim(value) == 0:
            return float(value)
        array = self.system.xp.asarray(value)
        grid_shape = self.system.grid_shape
        if array.ndim != 3 or tuple(array.shape[:2]) != grid_shape[:2] or array.shape[2] not in (1, grid_shape[2]):
            raise ValueError(f'{self.system.label()}: {label} must be a scalar or a {grid_shape} / '
                             f'{grid_shape[:2] + (1,)} array, got shape {tuple(array.shape)}')
        window_x, window_y, window_z = self.system.window
        return array[window_x, window_y, window_z if array.shape[2] == grid_shape[2] else slice(None)]
