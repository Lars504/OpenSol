"""Static 3D solid mechanics on a Domain3D: small strain, isotropic linear elastic cells, thermal strain from the
temperature-rise slot, supports, tractions and body forces; one ElasticSystem3D solve per change; the lowest vibration
modes for the eigenfrequency study (matrix-free shift-invert). Forces in N."""
import math

import numpy as np

from ...core.box_values import BoxValues
from ...core.constants import DTYPE
from ...core.fv_diffusion_3d import edge_extension_views_3d
from ...core.monitors import FieldProbe
from ...solvers.arpack_shift_invert import ArpackShiftInvertSolver
from .fields3d import INPUT_ARRAYS_3D, OUTPUT_ARRAYS_3D, SOURCE_SLOTS_3D, MechanicsFields3D
from .materials import DENSITY_SCHEMA, MECHANICAL_SCHEMA
from .renderer3d import SolidRenderer3D
from .solver_common import SolidMechanicsBase
from .system3d import ElasticSystem3D

_EXPANSION_NAMES = ('alpha_xx', 'alpha_yy', 'alpha_zz')


class SolidMechanics3DPhysics(SolidMechanicsBase):
    """3D static elasticity on the physical window (trilinear bricks on the cells, displacement on the cell corners,
    matrix-free 3 x 3 block-Jacobi PCG); outputs at the cell centres, 0 in void cells, edge-extended into any PML shell.
    Arrays on the domain's backend (gpu: kernels3d RawKernels, cpu: their numpy twins)."""

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
        xp.copyto(system.unknown, xp.asarray(resonance.displacement, dtype=system.dtype))
        self._write_outputs(thermal=False)
        for destination, source in self._pml_copies:
            xp.copyto(destination, source)
        self._shows_mode = True
        self.selected_resonance = number

    def step(self, t: float = 0.0, dt: float = math.inf) -> None:
        """Base step; a static re-solve ends the eigenfrequency state (reactions allowed again)."""
        solves = self.system.n_solves
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
                value.add_masked_to(system.applied_body_force[component], mask_window, system.window)
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

    def _new_fields(self, grid_shape):
        if len(grid_shape) != 3:
            raise ValueError(f'{self.system.label()} needs a 3D grid, got shape {grid_shape}')
        return MechanicsFields3D(*grid_shape, self.system.xp)

    def _pml_copy_pairs(self):
        window = self.system.window
        return [pair for name in OUTPUT_ARRAYS_3D
                for pair in edge_extension_views_3d(getattr(self.fields, name), window)]

    def _assemble(self):
        """Element Lame coefficients (cells with E <= 0 are void) and the matrix-free assembly."""
        lame_lambda, mu, solid = self._element_lame()
        self._solid = solid
        self.system.assemble(lame_lambda, mu, solid)

    def _expansion_arrays(self):
        """Full-grid float32 alpha_xx / alpha_yy / alpha_zz [1/K] (read by the fused load and output kernels)."""
        materials = self.domain.materials
        return tuple(getattr(materials, name) for name in _EXPANSION_NAMES)

    def _build_rhs(self):
        fields = self.fields
        self.system.build_rhs(self._expansion_arrays(), fields.temperature_rise, fields.temperature_rise_ext,
                              (fields.body_force_x, fields.body_force_y, fields.body_force_z))

    def _write_outputs(self, thermal=True):
        fields = self.fields
        self.system.write_outputs([getattr(fields, name) for name in OUTPUT_ARRAYS_3D], self._expansion_arrays(),
                                  fields.temperature_rise, fields.temperature_rise_ext, thermal=thermal)

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
