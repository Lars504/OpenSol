"""Dimension-free part of the static solid mechanics physics (2D plane strain and 3D): the PhysicsSolver contract with
the re-solve logic, coupled source slots, convergence record, reactions, warm start and the element Lame coefficients;
ElasticSystemBase = the PCGSolver and settings shared by ElasticSystem / ElasticSystem3D."""
import itertools
import math
from dataclasses import dataclass

import numpy as np

from ...core.backend import array_module, ndimage_module, sparse_module, to_host
from ...core.constants import DTYPE
from ...core.error_indicators import compliance, face_jump_indicator
from ...core.solver_base import PhysicsSolver
from ...solvers.pcg import SOLVER_DTYPE_NAMES, PCGSolver
from .materials import (EXPANSION_ARRAYS, MECHANICAL_SCHEMA, PHOTOELASTIC_ARRAYS, SOLVER_PRECISIONS, STIFFNESS_ARRAYS,
                        lame_coefficients)

MATERIAL_STATE_ARRAYS = ('youngs_modulus', 'poisson_ratio')


class ElasticSystemBase:
    """Precision / tolerance settings, the run backend (xp / sparse / ndimage modules) and the solvers.pcg.PCGSolver of
    an elastic window system; the convergence record is the PCGSolver's."""

    def __init__(self, precision, tol, max_iter, label, compute_residual=False, linear_solver=None, backend='gpu'):
        if linear_solver is not None:   # the build's 'linear' solver (ctx.solvers): its precision, tol and limit win
            precision = SOLVER_DTYPE_NAMES[linear_solver.precision]
        if precision not in SOLVER_PRECISIONS:
            raise ValueError(f"precision must be one of {', '.join(SOLVER_PRECISIONS)}, got {precision!r}")
        self.precision = precision
        self.dtype = np.float64 if precision == 'float64' else np.float32
        self.label = label                  # () -> e.g. "solid mechanics 'solid'"
        self.backend = backend              # 'gpu' (cupy / cupyx) or 'cpu' (numpy / scipy)
        self.xp = array_module(backend)
        self.sparse = sparse_module(backend)
        self.ndimage = ndimage_module(backend)
        if linear_solver is None:
            linear_solver = PCGSolver(SOLVER_PRECISIONS[precision], float(tol), int(max_iter), label=label,
                                      backend=backend)
        if getattr(linear_solver, 'backend', backend) != backend:
            raise ValueError(f"{label()}: the linear solver runs on the {linear_solver.backend} backend, the domain on "
                             f"the {backend} one")
        linear_solver.label = label         # messages name the physics
        linear_solver.compute_residual = bool(compute_residual)
        self.linear_solver = linear_solver
        self.direct_solves = 0              # static solves outside the PCG (2D piezoelectric host SuperLU)

    # shared window-system layer (2D ElasticSystem, 3D ElasticSystem3D); hooks _assembled() / _solvable()
    def cell_nodes(self, mask_window):
        """Bool node array (node_shape) of the corner nodes of the masked window cells (2D or 3D)."""
        nodes = self.xp.zeros(self.node_shape, dtype=self.xp.bool_)
        cells = mask_window.shape
        for shift in itertools.product((0, 1), repeat=len(cells)):
            nodes[tuple(slice(start, start + count) for start, count in zip(shift, cells))] |= mask_window
        return nodes

    def window_mask(self, mask, label):
        """Window view (bool) of a full-grid mask; ValueError naming `label` for another shape."""
        mask = self.xp.asarray(mask)
        if mask.shape != self.grid_shape:
            raise ValueError(f"{label}: mask must be a full-grid {self.grid_shape} boolean array, got shape "
                             f"{mask.shape}")
        return mask[self.window].astype(self.xp.bool_, copy=False)

    def seed_unknown(self, values) -> None:
        """Warm start: write a flat nodal displacement on the free dofs of the last assembly."""
        if not self._assembled():
            raise RuntimeError('assemble before seed_unknown')
        values = self.xp.asarray(values, dtype=self.dtype)
        if values.shape != (self.n_dofs,):
            raise ValueError(f'{self.label()}: seed must have {self.n_dofs} dofs, got shape {values.shape}')
        self.xp.copyto(self.unknown, values, where=self.free.astype(self.xp.bool_, copy=False))

    def solve(self) -> None:
        """Block-Jacobi PCG (PCGSolver, stop reference ||b||) on the eliminated system from the current unknown; the
        solver records the convergence."""
        if not self._solvable():
            raise RuntimeError('assemble and build_rhs before solve')
        solution = self._solution
        self.xp.subtract(self.unknown, self.fixed_values, out=solution)
        self.linear_solver.solve(self.rhs, solution)
        self.xp.add(solution, self.fixed_values, out=self.unknown)

    def require_backend(self, domain) -> None:
        """ValueError when the domain runs on another backend than the system was built for."""
        if domain.backend.name != self.backend:
            raise ValueError(f"{self.label()}: the system was built for the {self.backend} backend, the domain runs on "
                             f"{domain.backend.name}")

    @property
    def n_solves(self) -> int:
        return self.linear_solver.n_solves + self.direct_solves

    @property
    def last_iters(self) -> int:
        return self.linear_solver.last_iters

    @property
    def last_converged(self) -> bool:
        return self.linear_solver.last_converged

    @property
    def last_residual(self) -> float:
        return self.linear_solver.last_residual

    @property
    def last_warning(self):
        return self.linear_solver.last_warning

    @property
    def n_unconverged(self) -> int:
        return self.linear_solver.n_unconverged


class HostInputCache:
    """Mixin of the FEM solids: host copies of material / input arrays (alpha, dT, sigma0, beta, rho) read by every
    load, prestress and output pass, kept read-only until _drop_host_inputs (materials_changed, inject_source, an
    assembly, a temperature feature)."""

    _host_inputs = None

    def _host_input(self, key, compute):
        """compute() once per key until the next _drop_host_inputs (arrays and tuples of arrays made read-only)."""
        if self._host_inputs is None:
            self._host_inputs = {}
        if key not in self._host_inputs:
            value = compute()
            for array in (value if isinstance(value, tuple) else (value,)):
                if isinstance(array, np.ndarray):
                    array.flags.writeable = False
            self._host_inputs[key] = value
        return self._host_inputs[key]

    def _drop_host_inputs(self) -> None:
        self._host_inputs = None


def host_array(array) -> np.ndarray:
    """Host float64 numpy copy of a numpy / cupy array (core.backend.to_host)."""
    return np.asarray(to_host(array), dtype=np.float64)


class FEMSolidCommon(HostInputCache):
    """The FEM solids' shared layer (SolidMechanicsFEMPhysics on triangles, SolidMechanicsFEMPhysics3D on tetrahedra):
    records and Newton reporting, element masks / values, the temperature change and density reads, the loads flag;
    the subclass gives system, fields, _solid, _window_elements, _assemble and _label."""

    _output_groups = {}   # output group -> its field names (computed on first read)

    def _ensure_outputs(self, names=None):
        """Compute the pending output groups (_pending_outputs) holding any of `names` (None = every group)."""
        names = None if names is None else set(names)
        for group, members in self._output_groups.items():
            if group in self._pending_outputs and (names is None or names.intersection(members)):
                getattr(self, f'_write_{group}_outputs')()
                self._pending_outputs.discard(group)

    def _nonlinear_text(self) -> str:
        """'N Newton iterations over M load step(s), <start>' of the last nonlinear solve (+ unconverged)."""
        if self._nonlinear_record is None:
            return 'no nonlinear solve yet'
        converged, _, iterations, origin = self._nonlinear_record
        return (f'{iterations} Newton iteration(s) over {self.load_steps} load step(s), {origin}'
                + ('' if converged else ', NOT converged'))

    @property
    def prestressed(self) -> bool:
        """PrestressSolver: the studies call solve_static_state first (setting stress_stiffening or
        geometric_nonlinearity: the pencils then linearise about the nonlinear static state)."""
        return self.stress_stiffening or self.geometric_nonlinearity

    def dt_limit(self) -> float:
        return math.inf

    def state(self) -> dict:
        """Per-element arrays (n_elements,): the outputs at the element centroids (every pending group computed), the
        inputs and E / nu."""
        self._ensure_outputs()
        state = {name: getattr(self.fields, name) for name in self.output_arrays + self.input_arrays}
        state.update({name: getattr(self.domain.materials, name) for name in MATERIAL_STATE_ARRAYS})
        return state

    @property
    def _loads_dirty(self) -> bool:
        """A load input changed since the last static solve (setting True also drops the kept frequency-response load)."""
        return self._load_inputs_changed

    @_loads_dirty.setter
    def _loads_dirty(self, value) -> None:
        self._load_inputs_changed = bool(value)
        if value:
            self._harmonic_load = None   # (dof map key, read-only load) of _harmonic_inputs

    @property
    def n_solves(self) -> int:
        return self.system.n_solves

    @property
    def precision(self) -> str:
        return 'float64'

    @property
    def last_converged(self) -> bool:
        nonlinear = self._nonlinear_record is None or self._nonlinear_record[0]
        return self.system.linear_solver.last_converged and nonlinear

    @property
    def last_warning(self):
        if self._nonlinear_record is not None and not self._nonlinear_record[0]:
            return self._nonlinear_record[1]
        return self.system.linear_solver.last_warning

    @property
    def n_unconverged(self) -> int:
        return self.system.linear_solver.n_unconverged + (0 if self._newton is None else self._newton.n_unconverged)

    def add_temperature_change(self, mask, delta_T) -> None:
        """Add a temperature change [K] (number or per-element array) to temperature_rise_ext under mask."""
        elements = self._element_mask(mask, 'temperature change')
        change = np.where(elements, self._element_values(delta_T, 'temperature change'), 0.0)
        self._ensure_outputs()   # pending outputs belong to the inputs of their solve (as inject_source)
        xp = self.domain.xp
        self.fields.temperature_rise_ext += xp.asarray(change, dtype=DTYPE)
        self._drop_host_inputs()
        self._loads_dirty = True

    def rebuild_operator(self) -> None:
        self._assemble()

    def harmonic_reaction_table(self, region) -> dict:
        """'table' of the support's complex reaction per solved frequency (frequency_response.reaction_table)."""
        from .frequency_response import reaction_table   # lazy: frequency_response is the FEM solids' module
        return reaction_table(self._harmonic_reactions[region.name], region.name)

    def _element_mask(self, mask, what):
        """Host bool (n_elements,) of a build mask, window elements only."""
        mask = np.asarray(to_host(mask), dtype=bool).reshape(-1)
        if mask.size != self.domain.n_elements:
            raise ValueError(f'{self._label()} {what}: mask has {mask.size} values, expected one per element '
                             f'({self.domain.n_elements})')
        return mask & self._window_elements

    def _element_values(self, value, what):
        """Host float64 (n_elements,) of a number or a per-element array."""
        if isinstance(value, (int, float, np.integer, np.floating)):
            return np.full(self.domain.n_elements, float(value))
        values = host_array(value).reshape(-1)
        if values.size == 1:
            return np.full(self.domain.n_elements, float(values[0]))
        if values.size != self.domain.n_elements:
            raise ValueError(f'{self._label()} {what}: {values.size} values, expected one per element')
        return values

    def _solid_density(self, study):
        """Host rho per element (0 off the solid); ValueError naming `study` when a solid element has none."""
        density = host_array(self.domain.materials.rho)
        massless = self._solid & ~(density > 0.0)
        if massless.any():
            raise ValueError(f'{self._label()}: {int(massless.sum())} solid element(s) (youngs_modulus or c11 > 0) have '
                             f'no density rho > 0: {study} needs the mass of every solid element')
        return np.where(self._solid, density, 0.0)

    def _temperature_change(self):
        """Host dT [K] per element: coupled temperature_rise + applied temperature_rise_ext (cached, read-only)."""
        return self._host_input('temperature_change', lambda: host_array(self.fields.temperature_rise)
                                + host_array(self.fields.temperature_rise_ext))


@dataclass(frozen=True, eq=False)
class SupportRegion:
    """Runtime object of a support feature: its name and held dofs (node-shaped backend bool, last axis the component)."""
    name: str
    dof_mask: object


class SolidMechanicsBase(PhysicsSolver):
    """Static elasticity on the physical window, one system solve per change. Subclasses set output_arrays /
    input_arrays / source_slots and supply _new_fields, _pml_copy_pairs, _assemble, _build_rhs, _write_outputs, renderer."""

    name = 'solid'
    material_schema = MECHANICAL_SCHEMA
    output_arrays = ()
    input_arrays = ()
    source_slots = ()

    def _init_solid(self, system, domain) -> None:
        """Bind the window system (ElasticSystem / ElasticSystem3D) and prepare on the domain."""
        self.system = system
        self.domain = domain
        self.fields = None
        self.probes = []
        self.prepare(domain)

    # --- PhysicsSolver contract ---------------------------------------------

    def prepare(self, domain) -> None:
        """Register the mechanical arrays, allocate the window system and fields (clears supports and loads); assembly
        is deferred."""
        self.domain = domain
        domain.materials.add_properties(self.material_schema)
        self.system.prepare(domain)
        grid_shape = tuple(domain.materials.shape)
        if self.fields is None or getattr(self.fields, 'shape', None) != grid_shape:
            self.fields = self._new_fields(grid_shape)
        self._pml_copies = self._pml_copy_pairs()
        self._loads_dirty = True

    def finalize(self) -> None:
        """Assemble once (fails early on a solid region that can move as a rigid body)."""
        self._assemble()

    def materials_changed(self, names) -> None:
        names = set(names)
        if STIFFNESS_ARRAYS & names:
            self.system.operator_dirty = True
            self._loads_dirty = True   # the thermal load scales with lambda and mu
        if (EXPANSION_ARRAYS | PHOTOELASTIC_ARRAYS) & names:
            self._loads_dirty = True

    def step(self, t: float = 0.0, dt: float = math.inf) -> None:
        """Re-solve only when the stiffness, a support, a load or an input changed (always at the first call); probes
        sample after a re-solve and at every dt = inf call, never on unchanged time steps."""
        t = 0.0 if t is None else float(t)
        system = self.system
        re_solve = system.operator_dirty or self._loads_dirty or system.n_solves == 0
        if re_solve:
            if system.operator_dirty:
                self._assemble()
            self._build_rhs()
            self._solve_system()
            self._write_outputs()
            for destination, source in self._pml_copies:   # in order: later 3D axes span the extended earlier ones
                self.system.xp.copyto(destination, source)
            self._loads_dirty = False
        if re_solve or dt is None or not math.isfinite(dt):
            for probe in self.probes:
                probe.record(self.fields, t)

    def dt_limit(self) -> float:
        return math.inf

    def state(self) -> dict:
        fields = self.fields
        materials = self.domain.materials
        state = {name: getattr(fields, name) for name in self.output_arrays + self.input_arrays}
        state.update({name: getattr(materials, name) for name in MATERIAL_STATE_ARRAYS})
        return state

    def inject_source(self, name: str, array) -> None:
        if name not in self.source_slots:
            raise ValueError(f"{self.system.label()} accepts sources {', '.join(self.source_slots)}; got '{name}'")
        xp = self.system.xp
        xp.copyto(getattr(self.fields, name), xp.asarray(array, dtype=DTYPE))
        self._loads_dirty = True

    # --- solver settings and convergence record (the system holds them) -------

    @property
    def window(self):
        """Slices (x, y[, z]) of the physical window the displacement is solved on."""
        return self.system.window

    @property
    def precision(self) -> str:
        return self.system.precision

    @property
    def n_solves(self) -> int:
        return self.system.n_solves

    @property
    def last_iters(self) -> int:
        return self.system.last_iters

    @property
    def last_converged(self) -> bool:
        return self.system.last_converged

    @property
    def last_warning(self):
        """Message of the latest non-converged solve (never cleared), None before one."""
        return self.system.last_warning

    @property
    def n_unconverged(self) -> int:
        return self.system.n_unconverged

    # --- reactions, re-assembly, warm start -----------------------------------

    def reaction(self, region: SupportRegion) -> tuple:
        """Force components the support of `region` exerts on the solid in the last solve (sum of K u - f over its
        held dofs; a dof held by two supports counts in both): N/m in 2D, N in 3D."""
        return self.system.reaction(region.dof_mask)

    def rebuild_operator(self) -> None:
        """Re-assemble the stiffness, held-dof elimination, rigid-mode check and preconditioner."""
        self._assemble()

    def warm_start_fields(self) -> dict:
        """Host copy of the nodal displacement, to seed a build on the same mesh (sweep warm start)."""
        return {'u': np.array(to_host(self.system.unknown))}

    def seed_warm_start(self, fields) -> tuple:
        """Write warm_start_fields() of another build on the free dofs; no state slot is seeded (the outputs wait for
        the next solve)."""
        self.system.seed_unknown(fields['u'])
        return ()

    def error_indicator(self):
        """S7.7 hook (mesh plan A2): Kelly traction-jump eta^2 parts (2, *window), host float64, of the shown stresses
        across faces between solid elements, weighted 1 / c11 per element (2D; (r, z) faces ring-weighted)."""
        if self.domain.dimensions != 2 or getattr(self, '_stiffness', None) is None:
            raise NotImplementedError(f'{self.system.label()}: the traction-jump error indicator needs a solved 2D '
                                      'solid')
        system = self.system
        window = system.window
        stress_xx, stress_yy, stress_xy = (np.asarray(to_host(getattr(self.fields, name)[window]), dtype=np.float64)
                                           for name in ('stress_xx', 'stress_yy', 'stress_xy'))
        solid = np.asarray(to_host(self._solid), dtype=bool)
        dx, dy = system.dx_window[:, None], system.dy_window[None, :]
        if system.axisymmetric:   # ring faces: r_face dz (x faces), r_c dr (y faces); 2 pi drops out
            r_nodes = system.r_nodes
            x_area, y_area = r_nodes[1:-1, None] * dy, 0.5 * (r_nodes[:-1] + r_nodes[1:])[:, None] * dx
        else:
            x_area, y_area = dy, dx
        stiffness_scale = compliance(to_host(self._stiffness.values['c11']))
        return face_jump_indicator((stress_xx, stress_xy), (stress_xy, stress_yy), solid[:-1, :] & solid[1:, :],
                                   solid[:, :-1] & solid[:, 1:], dx, dy, x_area, y_area, stiffness_scale,
                                   stiffness_scale)

    # --- shared internals -----------------------------------------------------

    def _solve_system(self):
        """The static solve of step (the system's PCG; the 2D piezoelectric solid overrides it)."""
        self.system.solve()

    def _element_lame(self):
        """(lambda, mu, solid) per window element in the solve dtype: void cells (youngs_modulus <= 0) get lambda = mu
        = 0; ValueError for a poisson_ratio outside (-1, 0.5) in a solid cell."""
        materials, system = self.domain.materials, self.system
        youngs_modulus = materials.youngs_modulus[system.window].astype(system.dtype)   # fresh copies
        poisson_ratio = materials.poisson_ratio[system.window].astype(system.dtype)
        solid = youngs_modulus > 0
        out_of_range = solid & ((poisson_ratio <= -1.0) | (poisson_ratio >= 0.5))
        if bool(out_of_range.any()):
            raise ValueError(f"{system.label()}: poisson_ratio must lie in (-1, 0.5) in every cell with youngs_modulus "
                             f"> 0; found values from {float(poisson_ratio[solid].min()):.6g} to "
                             f"{float(poisson_ratio[solid].max()):.6g}")
        del out_of_range
        system.xp.multiply(youngs_modulus, solid, out=youngs_modulus)
        system.xp.multiply(poisson_ratio, solid, out=poisson_ratio)
        lame_lambda, mu = lame_coefficients(youngs_modulus, poisson_ratio)
        return lame_lambda, mu, solid
