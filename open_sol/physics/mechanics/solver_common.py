"""Dimension-free part of the static solid mechanics physics (2D plane strain and 3D): the PhysicsSolver contract with
the re-solve logic, coupled source slots, convergence record, reactions, warm start and the element Lame coefficients;
ElasticSystemBase = the PCGSolver and settings shared by ElasticSystem / ElasticSystem3D."""
import itertools
import math
from dataclasses import dataclass

import numpy as np

from ...core.backend import array_module, host_copy, ndimage_module, sparse_module, to_host, write_host_into
from ...core.constants import DTYPE
from ...core.error_indicators import compliance, face_jump_indicator
from ...core.solver_base import PhysicsSolver
from ...solvers.pcg import SOLVER_DTYPE_NAMES, PCGSolver
from .materials import (EXPANSION_ARRAYS, MECHANICAL_SCHEMA, PHOTOELASTIC_ARRAYS, SOLVER_PRECISIONS, STIFFNESS_ARRAYS,
                        lame_coefficients)

MATERIAL_STATE_ARRAYS = ('youngs_modulus', 'poisson_ratio')


class ElasticSystemBase:
    """Precision / tolerance settings, the run backend (xp / sparse / ndimage modules) and the solvers.pcg.PCGSolver of
    an elastic window system (2D: or a sparse_direct runtime through DiffusionDirectSolver); the convergence record is
    the linear solver's."""

    accepts_direct = False   # 2D ElasticSystem: a sparse_direct runtime factorises A_solve

    def __init__(self, precision, tol, max_iter, label, compute_residual=False, linear_solver=None, backend='gpu'):
        from ...solvers.sparse_direct import DiffusionDirectSolver, SparseDirectSolver   # numpy only at module level
        if isinstance(linear_solver, SparseDirectSolver):
            linear_solver = DiffusionDirectSolver(linear_solver)   # PCGSolver protocol: factorize / solve
        self.direct = isinstance(linear_solver, DiffusionDirectSolver)   # exact solves of a factor of A_solve
        if self.direct and not self.accepts_direct:
            raise ValueError(f'{label()}: the sparse_direct linear solver is not available on a 3D rectangular mesh; '
                             'assign Automatic or a pcg node')
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


def _require_published(unpublished_frequency, label) -> None:
    """RuntimeError when the shown fields are not the last solved frequency's (a fields-free solve of the field policy):
    a solution_state then would hold another frequency's fields."""
    if unpublished_frequency is not None:
        raise RuntimeError(f'{label}: the shown fields are not those of the last solved frequency '
                           f'({unpublished_frequency:.6g} Hz, skipped by the field policy): no solution state of it')


def _copy_or_none(array):
    return None if array is None else np.array(array)


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

    # --- snapshot solutions (solver_hooks.SolutionStateSolver) ----------------
    def solution_state(self, materials=False) -> dict:
        """Host copies of the input fields and the shown solution: without materials its displacement and output
        settings (_shown_solution; the outputs recompute on read, on this assembly), with materials (couplings may
        re-assemble) the computed outputs, nodal values and E / nu. RuntimeError while a sweep frequency is unpublished."""
        _require_published(self._unpublished_frequency, self._label())
        state = {'inputs': {name: host_copy(getattr(self.fields, name)) for name in self.input_arrays}}
        if not materials:
            state['assembly'] = self._stiffness   # identity only: restore refuses a state of another assembly
            state['shown'] = None if not (self._pending_outputs or self._nodal) else self._shown_solution()
            return state
        self._ensure_outputs()
        state['outputs'] = {name: host_copy(getattr(self.fields, name)) for name in self.output_arrays}
        state['nodal'] = {name: np.array(values) for name, values in self._nodal.items()}
        state['computed'] = self._computed_outputs()
        state['materials'] = {name: host_copy(getattr(self.domain.materials, name)) for name in MATERIAL_STATE_ARRAYS}
        return state

    def restore_solution(self, state) -> None:
        """Write a solution_state back: the input fields in place, then the shown solution marked pending (outputs
        recompute on read) or the computed outputs / E / nu in place; shown None = nothing solved yet (zero outputs)."""
        for name, values in state['inputs'].items():
            write_host_into(getattr(self.fields, name), values)
        self._drop_host_inputs()
        if 'outputs' in state:
            for name, values in state['outputs'].items():
                write_host_into(getattr(self.fields, name), values)
            for name, values in state['materials'].items():
                write_host_into(getattr(self.domain.materials, name), values)
            self._nodal = {name: values.copy() for name, values in state['nodal'].items()}
            self._restore_computed_outputs(state['computed'])
            self._pending_outputs = set()
            return
        if state['assembly'] is not self._stiffness:
            raise RuntimeError(f'{self._label()}: the solid was re-assembled since this solution state was taken (its '
                               'outputs would recompute on another stiffness)')
        if state['shown'] is None:
            for name in self.output_arrays:
                getattr(self.fields, name).fill(0.0)
            self._nodal = {}
            self._restore_computed_outputs(None)
            self._pending_outputs = set()
            return
        self._show_solution_state(state['shown'])

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
            raise ValueError(f'{self._label()}: {int(massless.sum())} solid element(s) (youngs_modulus, c11 or a shear '
                             f'/ bulk modulus > 0; liquids in the mixed formulation) have no density rho > 0: {study} '
                             'needs the mass of every solid element')
        return np.where(self._solid, density, 0.0)

    def _temperature_change(self):
        """Host dT [K] per element: coupled temperature_rise + applied temperature_rise_ext (cached, read-only)."""
        return self._host_input('temperature_change', lambda: host_array(self.fields.temperature_rise)
                                + host_array(self.fields.temperature_rise_ext))


@dataclass(frozen=True, eq=False)
class HarmonicOperators:
    """Host pieces of the frequency-response pencil of one assembly (the FEM solids' _build_harmonic_operators):
    `operator` the constant real part beyond the system K (None = the solver's pencil stiffness _pencil_stiffness;
    mixed: the saddle coupling + the constant groups' G0 D_g and pressure blocks, mixed_harmonic_operators), `loss` (or
    None), `mass`, `terms` (frequency_response.FrequencyTerm), `scale_terms` (SchurScaleTerm), `n_unknowns` (rows of the
    full vector: c n_u (+ n_p mixed, x = [u; p])), `pressure` (the mixed fem PressureSpace / PressureSpace3D, or None),
    `free` (bool over the unknowns, or over the columns of column_map) and `column_map` (None, the m = 1 axis ties,
    mixed: ties (+) the selection of the free pressure dofs), `groups` the viscoelastic.ViscoelasticGroup of the terms
    (mixed: every member, the pressure zones), `pressure_gauges` the pressure dofs held at 0 as the gauge of an enclosed
    incompressible region (touching incompressible groups are one region; int64, mixed) and `volume_checks` one
    (description, held u dofs int64, weights float64) per gauge: the region's summed coupling row 1^T B on the held
    displacements (check_held_volumes); mixed also `coupling` B (n_p, n_u), `pressure_masses` M_p,g and
    `pressure_free` (bool (n_p,), axis and gauge dofs False) as assembled, for the projected drives of
    drive_projection."""
    operator: object
    loss: object
    mass: object
    free: object
    n_unknowns: int
    column_map: object = None
    terms: tuple = ()
    scale_terms: tuple = ()
    pressure: object = None
    groups: tuple = ()
    pressure_gauges: object = ()
    volume_checks: tuple = ()
    coupling: object = None
    pressure_masses: tuple = ()
    pressure_free: object = None

    def drive_projection(self):
        """fem.elastic_drives.PressureProjection of the mixed pieces (a physics drive's mean normal stress projected
        onto the pressure space), None for the displacement formulation."""
        if self.pressure is None:
            return None
        from ...fem.elastic_drives import PressureProjection
        return PressureProjection(space=self.pressure, coupling=self.coupling, masses=tuple(self.pressure_masses),
                                  free=self.pressure_free)

    def check_held_volumes(self, fixed, label) -> None:
        """ValueError when the held displacements `fixed` (over every unknown, [u; p]) change the volume of an enclosed
        incompressible region: its pinned gauge drops the region's volume equation 1^T B u = 0, which the free
        displacements cannot meet (B^T 1 vanishes on them), so the system is solvable only when 1^T B u_held = 0;
        refused when |1^T B u_held| > HELD_VOLUME_TOLERANCE x sum |1^T B| |u_held| (per solve: the held entries
        only)."""
        for description, dofs, weights in self.volume_checks:
            held = np.asarray(fixed)[dofs]
            change = complex(weights @ held)
            scale = float(np.abs(weights) @ np.abs(held))
            if abs(change) > HELD_VOLUME_TOLERANCE * scale:
                raise ValueError(f'{label()}: the prescribed displacements change the volume of {description} by '
                                 f'{abs(change):.4g} (int div u dV, {abs(change) / scale:.3g} of the held motion '
                                 f'across its boundary, tolerance {HELD_VOLUME_TOLERANCE:g}): a liquid without bulk '
                                 'modulus enclosed by held displacements keeps its volume; prescribe a '
                                 'volume-preserving motion, free a displacement on its boundary or give it a '
                                 'bulk_modulus')

    def summary(self) -> str:
        """'' or '; mixed: N pressure dofs in K groups[ (G enclosed incompressible region(s): ...)]' and / or ';
        viscoelastic: N group(s), M frequency term(s)'."""
        text = ''
        if self.pressure is not None:
            text += f'; mixed: {self.pressure.n_dofs:,} pressure dofs in {self.pressure.n_groups} groups'
            if len(self.pressure_gauges):
                text += (f' ({len(self.pressure_gauges)} enclosed incompressible region(s), touching incompressible '
                         'groups counted as one: the pressure level of a liquid enclosed by held displacements is '
                         'free, p = 0 at one node of each)')
        if self.terms:
            text += f'; viscoelastic: {len(self.groups)} group(s), {len(self.terms)} frequency term(s)'
        return text


# |B^T 1| on the free displacements below this share of its largest value: the constant pressure of an incompressible
# region is a null vector of the saddle point (a liquid enclosed by held displacements; round-off ~1e-15 there)
_ENCLOSED_TOLERANCE = 1e-9
# |1^T B u_held| of an enclosed incompressible region relative to sum |1^T B| |u_held| above which its held
# displacements change its volume (HarmonicOperators.check_held_volumes); first estimate: the exact-quadrature row sums
# leave round-off ~1e-15 for a volume-preserving motion (a rigid bead in a closed box)
HELD_VOLUME_TOLERANCE = 1e-9


def mixed_harmonic_operators(groups, deviatoric, coupling, pressure_masses, space, mass, displacement_free,
                             pressure_free, prolongation, label) -> HarmonicOperators:
    """HarmonicOperators of the mixed u-p frequency response (both FEM solids) in x = [u; p] (n = n_u + n_p) from the
    viscoelastic.ViscoelasticGroup of every member (the pressure zones, in index order), their unit-shear stiffness D_g
    (n_u, n_u), the coupling B (n_p, n_u) and pressure masses M_p,g (fem.mixed_elasticity), the space, the u mass,
    the free u dofs (bool (n_u,)), the free pressure dofs (bool (n_p,): the axis ones False at m >= 1) and the m = 1
    axis-tie prolongation (n_u, n_r) or None. A = [[sum_g G*_g D_g - w^2 M, -B^T], [-B, -sum_g M_p,g / K*_g]]
    (fem.mixed_elasticity header): constant groups folded into operator (G0 D_g, -Re(1 / K*) M_p,g) and loss (eta G0
    D_g, -Im(1 / K*) M_p,g), omega-dependent ones FrequencyTerms (G*_g D_g; -1 / K*_g M_p,g); an incompressible group's
    pressure block is 0 but kept in the pattern (stored zeros without a column map, else a FrequencyTerm of coefficient
    0: a congruence drops exact zeros); one SchurScaleTerm of B (the per-frequency Schur proxy sum_j B_pj^2 / d_j of
    the pressure rows, frequency_response). Column map: None (free = u free + pressure free) or block_diag(P,
    selection of the free pressure dofs) (free = every column). The gauges of enclosed incompressible regions
    (_pressure_gauges: over the union of the incompressible groups) are held at 0. ValueError for an incompressible
    region whose every coupled displacement is held."""
    import scipy.sparse
    from ...fem.assembly import pattern_sum
    from ...fem.mixed_elasticity import embedded, saddle_coupling
    from .frequency_response import FrequencyTerm, SchurScaleTerm
    groups = tuple(groups)
    n_p, n_u = coupling.shape
    if space.n_dofs != n_p or space.n_groups != len(groups) or len(deviatoric) != len(groups) \
            or len(pressure_masses) != len(groups) or any(group.index != index for index, group in enumerate(groups)):
        raise ValueError(f'{label()}: the mixed pieces disagree ({len(groups)} groups, {space.n_groups} pressure '
                         f'zones, {n_p} pressure dofs, B {coupling.shape})')
    n_total = n_u + n_p
    pressure_free = np.array(pressure_free, dtype=bool)
    gauges, volume_checks = _pressure_gauges(coupling, space,
                                             [group.index for group in groups if group.bulk.incompressible],
                                             pressure_free, np.asarray(displacement_free, dtype=bool), prolongation,
                                             label)
    pressure_free[gauges] = False
    if prolongation is None:
        column_map = None
        free = np.concatenate([np.asarray(displacement_free, dtype=bool), pressure_free])
    else:
        kept = np.flatnonzero(pressure_free)
        selection = scipy.sparse.csr_matrix((np.ones(kept.size), (kept, np.arange(kept.size))), shape=(n_p, kept.size))
        # the m = 1 column map (booked: definition_viscoelastic.viscoelastic_memory_items)
        column_map = scipy.sparse.block_diag((prolongation, selection), format='csr')
        free = np.ones(column_map.shape[1], dtype=bool)
    # the embedded blocks, scaled constant copies and pattern sums (booked:
    # definition_viscoelastic.viscoelastic_memory_items, 'mixed saddle-point blocks ...')
    constant = [saddle_coupling(coupling, n_u)]
    loss_parts, terms = [], []
    for group, shear_matrix, pressure_mass in zip(groups, deviatoric, pressure_masses):
        index, shear, bulk = group.index, group.shear, group.bulk
        shear_block = embedded(shear_matrix, n_total)
        if shear.frequency_dependent:
            terms.append(FrequencyTerm(shear_block, shear.at, f'G* of group {index}'))
        else:
            constant.append(shear.relaxed * shear_block)
            if shear.loss_factor:
                loss_parts.append((shear.relaxed * shear.loss_factor) * shear_block)
        pressure_block = embedded(pressure_mass, n_total, row_offset=n_u, column_offset=n_u)
        if bulk.frequency_dependent:
            terms.append(FrequencyTerm(pressure_block, _negative_compliance(bulk), f'-1 / K* of group {index}'))
        elif bulk.incompressible:   # 1 / K* = 0: the pressure diagonal stays stored (PARDISO symmetric types need it)
            if column_map is None:
                constant.append(pressure_block * 0.0)
            else:
                terms.append(FrequencyTerm(pressure_block, _zero_coefficient, f'incompressible group {index}'))
        else:
            compliance = bulk.compliance(0.0)   # constant: (1 - j eta) / (K0 (1 + eta^2))
            constant.append(-compliance.real * pressure_block)
            if compliance.imag:
                loss_parts.append(-compliance.imag * pressure_block)
    # the pressure rows' row scale: B embedded at the pressure rows / u columns, its Schur proxy taken per frequency
    # (frequency_response.SchurScaleTerm: a fixed modulus weight is (h / delta)^2 off in inertia-dominated cells;
    # booked: definition_viscoelastic 'mixed Schur row scale')
    scale_terms = (SchurScaleTerm(embedded(coupling, n_total, row_offset=n_u)),)
    return HarmonicOperators(pattern_sum(constant), pattern_sum(loss_parts) if loss_parts else None,
                             embedded(mass, n_total), free, n_total, column_map=column_map, terms=tuple(terms),
                             scale_terms=scale_terms, pressure=space, groups=groups,
                             pressure_gauges=np.asarray(gauges, dtype=np.int64), volume_checks=volume_checks,
                             coupling=coupling.tocsr(), pressure_masses=tuple(mass.tocsr() for mass in pressure_masses),
                             pressure_free=_read_only_mask(pressure_free))


def _read_only_mask(mask) -> np.ndarray:
    mask = np.array(mask, dtype=bool)
    mask.flags.writeable = False
    return mask


def _negative_compliance(law):
    """omega -> -1 / K*(omega) of a bulk ModulusLaw (the pressure block coefficient)."""
    return lambda omega: -law.compliance(omega)


def _zero_coefficient(omega) -> complex:
    """The coefficient of an incompressible group's pressure block (kept for its pattern)."""
    return 0j


def _pressure_gauges(coupling, space, incompressible, pressure_free, displacement_free, prolongation,
                     label) -> tuple:
    """(int64 pressure dofs held at 0, volume checks): one gauge per connected region of the UNION of the
    incompressible groups (fem.mixed_elasticity pressure_regions: groups touching at a mesh node are one region; its
    free dofs) whose constant pressure q = 1 is a null vector of the saddle point: B^T q vanishes on the free
    displacements (liquids enclosed by held displacements: int q div v dV is a boundary integral of v . n there, so the
    pressure level is undetermined; on an interface between two incompressible groups the terms of both cancel, since v
    is continuous there, so neither group alone is enclosed although the pair is; at an azimuthal order m >= 1 q is no
    cos(m phi) null vector and no region is pinned). The gauge is the region's dof farthest from its centroid (a wall
    far from the drive). Per gauge a volume check (description, held u dofs, B^T q there;
    HarmonicOperators.check_held_volumes): the pin drops the equation q^T B u = 0, which only the held displacements
    enter. ValueError for a region of one group whose every coupled displacement is held (no free displacement in an
    incompressible zone: its whole pressure block is singular)."""
    from ...fem.mixed_elasticity import pressure_regions
    gauges, checks = [], []
    if not incompressible:
        return np.zeros(0, dtype=np.int64), ()
    coupling = coupling.tocsr()
    if prolongation is None:
        free_rows = displacement_free

        def restricted(vector):
            return vector[displacement_free]
    else:
        prolongation = prolongation.tocsr()
        free_rows = np.diff(prolongation.indptr) > 0   # u dofs that some reduced column moves (ties included)

        def restricted(vector):
            return prolongation.T @ vector
    points = space.dof_points()

    def describe(region):
        groups = np.unique(np.searchsorted(space.offsets, region, side='right') - 1)
        low, high = points[region].min(axis=0), points[region].max(axis=0)
        where = ', '.join(f'{low[axis]:.4g} .. {high[axis]:.4g} m' for axis in range(low.size))
        return (f"the incompressible region of viscoelastic parameter group{'s' if groups.size > 1 else ''} "
                f"{', '.join(str(int(group)) for group in groups)} (no bulk modulus; {region.size} pressure dofs in "
                f'{where})')

    for group in incompressible:   # a pressure dof coupled to held displacements only is singular in any union
        for region in pressure_regions(space, [group]):
            region = region[pressure_free[region]]
            if not region.size:
                continue
            rows = coupling[region]
            if not free_rows[np.unique(rows.indices[np.abs(rows.data) > 0.0])].any():
                raise ValueError(f'{label()}: {describe(region)} has every displacement held, so its pressure is '
                                 'undetermined: give the held region a bulk_modulus or youngs_modulus, or leave a free '
                                 'displacement in it')
    for region in pressure_regions(space, incompressible):
        region = region[pressure_free[region]]
        if not region.size:
            continue
        force = np.asarray(coupling[region].sum(axis=0)).ravel()   # B^T q of q = 1 on the region
        scale = float(np.abs(force).max())
        residual = restricted(force)
        if scale > 0.0 and (not residual.size or float(np.abs(residual).max()) <= _ENCLOSED_TOLERANCE * scale):
            offsets = points[region] - points[region].mean(axis=0)
            gauges.append(int(region[np.argmax(np.sum(offsets ** 2, axis=1))]))
            held = np.flatnonzero(~free_rows & (force != 0.0))   # the region's boundary dofs (a few per node)
            checks.append((describe(region), held, force[held]))
    return np.asarray(gauges, dtype=np.int64), tuple(checks)


@dataclass(frozen=True, eq=False)
class SupportRegion:
    """Runtime object of a support feature: its name and held dofs (node-shaped backend bool, last axis the component)."""
    name: str
    dof_mask: object


class SolidMechanicsBase(PhysicsSolver):
    """Static elasticity on the physical window, one system solve per change. Subclasses set output_arrays /
    input_arrays / source_slots and supply _new_fields, _pml_copy_pairs, _assemble, _build_rhs, _write_outputs (window
    only: callers go through _show_outputs / _extend_into_pml), renderer."""

    name = 'solid'
    material_schema = MECHANICAL_SCHEMA
    output_arrays = ()
    input_arrays = ()
    source_slots = ()
    _unpublished_frequency = None   # [Hz] a frequency response solved without showing its fields (2D field policy)

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
            self._show_outputs()
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

    # --- snapshot solutions (solver_hooks.SolutionStateSolver) ----------------
    def solution_state(self, materials=False) -> dict:
        """Host copies of every output and input field (full grid: the outputs are written eagerly per solve);
        materials: E / nu too. RuntimeError while a frequency response's last solve is unpublished."""
        _require_published(self._unpublished_frequency, self.system.label())
        state = {'fields': {name: host_copy(getattr(self.fields, name))
                            for name in self.output_arrays + self.input_arrays}}
        if materials:
            state['materials'] = {name: host_copy(getattr(self.domain.materials, name))
                                  for name in MATERIAL_STATE_ARRAYS}
        return state

    def restore_solution(self, state) -> None:
        """Write a solution_state back in place (the loads stay as they are: no re-solve follows a read)."""
        for name, values in state['fields'].items():
            write_host_into(getattr(self.fields, name), values)
        for name, values in state.get('materials', {}).items():
            write_host_into(getattr(self.domain.materials, name), values)

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
        """S7.7 hook: Kelly traction-jump eta^2 parts (2, *window), host float64, of the shown stresses
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

    def _show_outputs(self, thermal=True) -> None:
        """_write_outputs (window only), then _extend_into_pml."""
        self._write_outputs(thermal=thermal)
        self._extend_into_pml()

    def _extend_into_pml(self) -> None:
        """Edge-extend the window outputs into the PML ring (after every output write)."""
        for destination, source in self._pml_copies:   # in order: later 3D axes span the extended earlier ones
            self.system.xp.copyto(destination, source)

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
