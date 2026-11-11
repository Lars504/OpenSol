"""3D small-strain linear elasticity on the physical window of a Domain3D: trilinear bricks on the window cells, nodal
(u_x, u_y, u_z) on the cell corners, matrix-free stiffness product, constrained dofs eliminated, 3 x 3 block-Jacobi PCG.
Forces in N. Arrays on the run backend (ElasticSystemBase xp / ndimage; kernels3d twins dispatch on them)."""
import itertools
import warnings

import numpy as np
from ...core.backend import to_host

from ...core.constants import FACES_3D
from ...solvers.pcg import CG_ITERATIONS_PER_AXIS_CELL, PCGSolver
from .kernels3d import BrickKernels, FusedBrickIteration
from .multigrid3d import (CYCLE_COST_IN_JACOBI_ITERATIONS, FALLBACK_ITERATIONS, FALLBACK_MIN_ITERATIONS,
                          BrickMultigrid)
from .modal import LANCZOS_TOLERANCE_PER_CG, solve_lowest_modes_matrix_free
from .solver_common import ElasticSystemBase

# Node index of each outer face on a (nx_nodes, ny_nodes, nz_nodes[, 3]) node array, and the face's normal component.
FACE_NODES_3D = {'x_lo': (0, slice(None), slice(None)), 'x_hi': (-1, slice(None), slice(None)),
                 'y_lo': (slice(None), 0, slice(None)), 'y_hi': (slice(None), -1, slice(None)),
                 'z_lo': (slice(None), slice(None), 0), 'z_hi': (slice(None), slice(None), -1)}
NORMAL_COMPONENT_3D = {'x_lo': 0, 'x_hi': 0, 'y_lo': 1, 'y_hi': 1, 'z_lo': 2, 'z_hi': 2}

RIGID_RANK_TOLERANCE = 1e-9   # rigid-mode Gram eigenvalue below this fraction of the largest = a free rigid motion
_MOTION_TOLERANCE = 1e-6  # relative size below which a motion component (rotation, slide, direction) reads as 0
_CORNER_SHIFTS = tuple(itertools.product((0, 1), repeat=3))


class ElasticSystem3D(ElasticSystemBase):
    """Window system of the 3D displacement: element Lame coefficients (the operator), held dofs (eliminated),
    rigid-mode rank check, block-Jacobi PCG (matrix-free PCGSolver; 'mixed': float32 inner copies of the operator) or
    the geometric multigrid preconditioner (multigrid3d), fused load / output kernels and reactions."""

    # --- allocation -----------------------------------------------------------

    def prepare(self, domain) -> None:
        """Allocate every window / node buffer on the domain's physical window (clears supports and loads; assembly
        is deferred)."""
        dimensions = getattr(domain, 'dimensions', 2)
        if dimensions != 3:
            raise ValueError(f'{self.label()}: ElasticSystem3D needs a Domain3D, got a {dimensions}D domain')
        self.require_backend(domain)
        xp = self.xp
        self.grid_shape = tuple(domain.grid.shape)
        self.window = tuple(slice(window.start, window.stop) for window in domain.physical_window())
        self.element_shape = tuple(window.stop - window.start for window in self.window)
        self.node_shape = tuple(n + 1 for n in self.element_shape)
        self.n_nodes = int(np.prod(self.node_shape))
        self.n_dofs = 3 * self.n_nodes
        if self.n_dofs >= 2 ** 31:
            raise ValueError(f'{self.label()}: {self.n_nodes} nodes exceed the int32 dof indexing of the kernels '
                             f'(3 x nodes must stay below 2^31); coarsen the mesh')

        dtype = self.dtype
        self.widths_host = tuple(np.asarray(widths[window], dtype=np.float64)
                                 for widths, window in zip(domain.cell_widths(), self.window))
        self.widths_device = tuple(xp.asarray(widths, dtype=dtype) for widths in self.widths_host)
        self.node_coordinates_host = tuple(np.concatenate(([0.0], np.cumsum(widths))) for widths in self.widths_host)
        self.kernels = BrickKernels(dtype, self.element_shape, tuple(window.start for window in self.window),
                                    self.grid_shape)

        node_shape, n_dofs = self.node_shape, self.n_dofs
        self.unknown = xp.zeros(n_dofs, dtype=dtype)                      # nodal (u_x, u_y, u_z) [m], interleaved
        self.held = xp.zeros((*node_shape, 3), dtype=xp.bool_)           # dofs held by supports
        self.held_values = xp.zeros((*node_shape, 3), dtype=dtype)       # their prescribed displacement [m]
        self.applied_load = xp.zeros((*node_shape, 3), dtype=dtype)      # nodal forces of the load features [N]
        self.applied_body_force = xp.zeros((3, *self.element_shape), dtype=dtype)   # feature body forces [N/m^3]
        self.load_full = xp.zeros(n_dofs, dtype=dtype)                   # total nodal load of the last build_rhs
        self.rhs = xp.zeros(n_dofs, dtype=dtype)
        self.fixed_values = xp.zeros(n_dofs, dtype=dtype)    # prescribed values on held / pinned dofs, 0 on free ones
        self.fixed_coupling = xp.zeros(n_dofs, dtype=dtype)  # K . fixed_values
        self.free = xp.zeros(n_dofs, dtype=xp.bool_)
        self.inverse_blocks = xp.zeros((self.n_nodes, 6), dtype=dtype)   # reduced inverse 3 x 3 blocks per node
        # CG unknown w = u - u_fix; between solves the scratch of reaction (solve rebuilds it from unknown)
        self._solution = xp.zeros(n_dofs, dtype=dtype)
        # reaction: K u of the last solve kept in _solution while (solves, assemblies) are unchanged (every PCG solve,
        # the Lanczos inner ones too, counts in n_solves and overwrites _solution)
        self._assemblies = 0
        self._reaction_key = None
        # 'mixed' solver: float32 widths here, float32 (lame_lambda, mu, inverse blocks) per assembly for the inner CG
        mixed = self.linear_solver.precision == 'mixed'
        # pcg preconditioner 'multigrid': the geometric V-cycle of multigrid3d, float32 at every precision (double: on
        # the float32 copies, cast at both ends) in place of block Jacobi, until a solve trips the guard
        # (_guarded_solve); no fused iteration then
        uses_multigrid = (isinstance(self.linear_solver, PCGSolver)
                          and self.linear_solver.preconditioner == 'multigrid')
        double_cycle_cast = uses_multigrid and not mixed and np.dtype(dtype) != np.float32
        self.widths_single = (tuple(xp.asarray(widths, dtype=np.float32) for widths in self.widths_host)
                              if mixed or double_cycle_cast else None)
        self._single_operator = None
        self._cycle_out = xp.zeros(n_dofs, dtype=np.float32) if double_cycle_cast else None
        single_callbacks = ({'single_operator': self._apply_single_operator,
                             'single_preconditioner': self._precondition_single} if mixed else {})
        self.multigrid = BrickMultigrid(self.element_shape, np.float32, xp) if uses_multigrid else None
        self._multigrid_active = uses_multigrid
        if uses_multigrid:
            self.linear_solver.preconditioner_counters = self.multigrid.work_counters
            single_callbacks['capture_blocks'] = True
        elif isinstance(self.linear_solver, PCGSolver):   # two fused launches per CG step, check blocks as CUDA graphs
            if mixed:
                single_callbacks['single_fused_iteration'] = FusedBrickIteration(
                    self.kernels, self._fused_operator_single, np.float32, xp)
            else:
                single_callbacks['fused_iteration'] = FusedBrickIteration(self.kernels, self._fused_operator, dtype, xp)
            single_callbacks['capture_blocks'] = True
        # matrix-free binding (the callbacks read the operator of the latest assembly); allocates the PCG workspace
        self.linear_solver.factorize(self._apply_solve_operator, self._precondition, n_unknowns=n_dofs,
                                     **single_callbacks)

        self.lame_lambda = None     # element Lame coefficients [Pa] of the last assembly (the operator)
        self.mu = None
        self._rhs_built = False
        self._rigid_signature = None
        self.operator_dirty = True

    # --- node geometry ----------------------------------------------------------

    def tributary_lengths(self, axis: int):
        """(n + 1,) tributary length [m] of every node line along an axis (half of each adjacent cell)."""
        widths = self.widths_device[axis]
        lengths = self.xp.zeros(widths.size + 1, dtype=self.dtype)
        lengths[:-1] += 0.5 * widths
        lengths[1:] += 0.5 * widths
        return lengths

    def face_node_areas(self, face):
        """Tributary area [m^2] of every node of an outer face (a quarter of each adjacent face cell), shaped like
        the face's node slice."""
        normal_axis = NORMAL_COMPONENT_3D[_face(face)]
        first, second = (self.tributary_lengths(axis) for axis in range(3) if axis != normal_axis)
        return first[:, None] * second[None, :]

    # --- supports and loads -----------------------------------------------------

    def hold_face(self, face, u_x=None, u_y=None, u_z=None):
        """Hold components [m] (None = free; a scalar or a face-node-shaped array) on every node of an outer face,
        written straight into held_values (later holds win); returns the held dof mask."""
        xp = self.xp
        nodes = FACE_NODES_3D[_face(face)]
        dof_mask = xp.zeros((*self.node_shape, 3), dtype=xp.bool_)
        for component, value in enumerate((u_x, u_y, u_z)):
            if value is not None:
                dof_mask[nodes + (component,)] = True
                self.held_values[nodes + (component,)] = xp.asarray(value, dtype=self.dtype)
        self.held |= dof_mask
        self.operator_dirty = True
        return dof_mask

    def hold_cells(self, mask_window, u_x=0.0, u_y=0.0, u_z=0.0):
        """Hold components [m] (None = free, else a scalar) on every corner node of the masked window cells;
        returns the held dof mask."""
        xp = self.xp
        nodes = self.cell_nodes(mask_window)
        dof_mask = xp.zeros((*self.node_shape, 3), dtype=xp.bool_)
        for component, value in enumerate((u_x, u_y, u_z)):
            if value is not None:
                dof_mask[..., component] = nodes
                xp.copyto(self.held_values[..., component], xp.asarray(float(value), dtype=self.dtype), where=nodes)
        self.held |= dof_mask
        self.operator_dirty = True
        return dof_mask

    def add_face_traction(self, face, t_x, t_y, t_z) -> None:
        """Add a traction (t_x, t_y, t_z) [Pa] on an outer face: each node gets t times its tributary area [N]."""
        nodes = FACE_NODES_3D[_face(face)]
        areas = self.face_node_areas(face)
        for component, traction in enumerate((t_x, t_y, t_z)):
            if traction:
                self.applied_load[nodes + (component,)] += float(traction) * areas

    def add_body_force(self, mask_window, f_x, f_y, f_z) -> None:
        """Add a body force density [N/m^3] (scalars or arrays broadcastable to the window) on the masked window
        cells; void elements drop it in the load kernel."""
        xp = self.xp
        for component, density in enumerate((f_x, f_y, f_z)):
            if isinstance(density, (int, float)) and density == 0:
                continue
            self.applied_body_force[component] += xp.where(mask_window, xp.asarray(density, dtype=self.dtype), 0.0)

    # --- assembly -------------------------------------------------------------

    def assemble(self, lame_lambda, mu, solid_elements) -> None:
        """Keep the element Lame coefficients [Pa] (window-shaped; mu = 0 marks void) as the operator, pin unconnected
        dofs, check rigid modes per solid region, then the fixed coupling and the reduced inverse blocks."""
        xp = self.xp
        dtype = self.dtype
        for values, name in ((lame_lambda, 'lame_lambda'), (mu, 'mu'), (solid_elements, 'solid_elements')):
            if tuple(values.shape) != self.element_shape:
                raise ValueError(f'{self.label()}: {name} must have the window element shape {self.element_shape}, '
                                 f'got {tuple(values.shape)}')
        lame_lambda = xp.ascontiguousarray(lame_lambda, dtype=dtype)
        mu = xp.ascontiguousarray(mu, dtype=dtype)
        kernels, widths = self.kernels, self.widths_device

        kernels.diagonal_blocks(self.inverse_blocks, lame_lambda, mu, widths)
        held = self.held.reshape(-1)
        unconnected = (self.inverse_blocks[:, :3] <= 0).reshape(-1) & ~held   # dofs touching no solid element
        xp.logical_not(held | unconnected, out=self.free)
        del unconnected
        self._check_rigid_modes(solid_elements)

        self.fixed_values.fill(0)
        xp.copyto(self.fixed_values, self.held_values.reshape(-1), where=held)
        kernels.product(self.fixed_values, self.fixed_coupling, lame_lambda, mu, widths, self.free, masked=False)
        kernels.reduce_invert_blocks(self.inverse_blocks, self.free)
        if self.widths_single is not None:   # mixed / double multigrid: float32 operator (blocks inverted in float64)
            self._single_operator = None   # free the previous copies first
            self._single_operator = tuple(values.astype(np.float32)
                                          for values in (lame_lambda, mu, self.inverse_blocks))
        xp.copyto(self.unknown, self.fixed_values, where=~self.free)
        self.lame_lambda, self.mu = lame_lambda, mu
        if self.multigrid is not None:   # the hierarchy of this assembly, level 0 = the CG's own (inner) operator
            if self.widths_single is not None:
                level_lambda, level_mu, level_blocks = self._single_operator
                level_widths = self.widths_single
            else:
                level_lambda, level_mu, level_blocks, level_widths = lame_lambda, mu, self.inverse_blocks, widths
            self.multigrid.build(level_lambda, level_mu, self.free, level_blocks, level_widths, self.widths_host,
                                 kernels)
        if isinstance(self.linear_solver, PCGSolver):
            self.linear_solver.release_block_graphs()   # captured CG blocks hold the replaced operator arrays
        self._rhs_built = False
        self.operator_dirty = False
        self._assemblies += 1

    def _check_rigid_modes(self, solid_elements):
        """Raise ValueError when a solid region (26-connected solid elements) admits a rigid motion: rank of the 6 x 6
        Gram matrix of the rigid-mode rows at its held dofs below 6. Skipped while the solid and held sets are
        unchanged."""
        xp = self.xp
        signature = (solid_elements.copy(), self.held.copy())
        cached = self._rigid_signature
        if (cached is not None and bool(xp.array_equal(cached[0], signature[0]))
                and bool(xp.array_equal(cached[1], signature[1]))):
            return
        labels, n_regions = self.ndimage.label(solid_elements, structure=np.ones((3, 3, 3), dtype=bool))
        n_regions = int(n_regions)
        if n_regions:
            gram, centroid, scale, counts = self._rigid_mode_gram(labels, n_regions)
            eigenvalues, eigenvectors = np.linalg.eigh(gram)   # ascending per region
            threshold = RIGID_RANK_TOLERANCE * eigenvalues[:, -1]
            unstable = eigenvalues[:, 0] <= threshold         # no held dof at all gives gram = 0: unstable too
            if unstable.any():
                first = int(np.flatnonzero(unstable)[0])
                if counts[first] == 0:
                    reason = 'it has no support at all'
                else:
                    null_basis = eigenvectors[first][:, eigenvalues[first] <= threshold[first]]
                    reason = (f'{null_basis.shape[1]} rigid motion(s) stay free, e.g. '
                              + self._rigid_motion_text(null_basis, centroid[first], scale[first]))
                cell = [int(index) for index in to_host(xp.argwhere(labels == first + 1)[0])]
                x_cell, y_cell, z_cell = (float(np.sum(widths[:index]) + 0.5 * widths[index])
                                          for widths, index in zip(self.widths_host, cell))
                raise ValueError(
                    f"{self.label()}: {int(np.count_nonzero(unstable))} of {n_regions} solid region(s) (cells with "
                    'youngs_modulus > 0, joined through shared faces, edges or corners) can move as a rigid body, so '
                    f'the solve is singular; e.g. the region containing the cell at x = {x_cell:.6g} m, y = '
                    f'{y_cell:.6g} m, z = {z_cell:.6g} m (from the window corner): {reason}. Hold every solid region '
                    'with Fixed, Roller or Displacement supports on outer faces or a Fixed (domain) region, e.g. '
                    'rollers on x_lo, y_lo and z_lo, or give the material around it a youngs_modulus so that it '
                    'joins a held region.')
        self._rigid_signature = signature

    def _rigid_mode_gram(self, labels, n_regions):
        """Host (gram (n, 6, 6), centroid (n, 3) [m], scale (n,) [m], held-dof count (n,)) per region: Gram matrix of
        the rows g = (e_a, x~ x e_a) of its held dofs, x~ = position relative to their centroid over their rms spread."""
        xp = self.xp
        nx_nodes, ny_nodes, nz_nodes = self.node_shape
        padded = xp.zeros(tuple(n + 2 for n in labels.shape), dtype=labels.dtype)
        padded[1:-1, 1:-1, 1:-1] = labels
        node_labels = padded[:nx_nodes, :ny_nodes, :nz_nodes].copy()   # max over the 8 adjacent elements
        for a, b, c in _CORNER_SHIFTS[1:]:
            xp.maximum(node_labels, padded[a:a + nx_nodes, b:b + ny_nodes, c:c + nz_nodes], out=node_labels)
        del padded
        held_dofs = xp.flatnonzero(self.held & (node_labels > 0)[..., None])
        n_bins = n_regions + 1
        if held_dofs.size == 0:
            return (np.zeros((n_regions, 6, 6)), np.zeros((n_regions, 3)), np.ones(n_regions),
                    np.zeros(n_regions, dtype=np.int64))
        node, component = held_dofs // 3, held_dofs % 3
        region = node_labels.reshape(-1)[node].astype(xp.int64)
        del node_labels
        node_i, node_j, node_k = node // (ny_nodes * nz_nodes), (node // nz_nodes) % ny_nodes, node % nz_nodes
        position = xp.stack([xp.asarray(coordinates)[index] for coordinates, index in
                             zip(self.node_coordinates_host, (node_i, node_j, node_k))], axis=1)
        counts = xp.bincount(region, minlength=n_bins)
        safe_counts = xp.maximum(counts, 1).astype(xp.float64)
        centroid = xp.stack([xp.bincount(region, weights=position[:, axis], minlength=n_bins) for axis in range(3)],
                            axis=1) / safe_counts[:, None]
        relative = position - centroid[region]
        scale = xp.sqrt(xp.bincount(region, weights=(relative ** 2).sum(axis=1), minlength=n_bins) / safe_counts)
        scale = xp.where(scale > 0, scale, 1.0)
        relative /= scale[region][:, None]
        rows = xp.zeros((held_dofs.size, 6), dtype=xp.float64)
        rows[xp.arange(held_dofs.size), component] = 1.0
        rows[:, 3:] = xp.cross(relative, xp.eye(3, dtype=xp.float64)[component])   # u_a = t_a + omega . (x x e_a)
        gram = xp.zeros((n_bins, 6, 6), dtype=xp.float64)
        for p in range(6):
            for q in range(p, 6):
                gram[:, p, q] = gram[:, q, p] = xp.bincount(region, weights=rows[:, p] * rows[:, q], minlength=n_bins)
        return to_host(gram[1:]), to_host(centroid[1:]), to_host(scale[1:]), to_host(counts[1:])

    def _rigid_motion_text(self, null_basis, centroid, scale) -> str:
        return rigid_motion_text(null_basis, centroid, scale,
                                 max(coordinates[-1] for coordinates in self.node_coordinates_host))

    # --- solve ----------------------------------------------------------------

    def build_rhs(self, alphas, temperature_rise, temperature_rise_ext, body_forces) -> None:
        """Total nodal load [N] into load_full by the fused load kernel (full-grid float32 alpha_xx / yy / zz, the two
        temperature rises [K], coupled body_force_x / y / z [N/m^3]), then rhs = D_free (load - K u_fix)."""
        if self.lame_lambda is None:
            raise RuntimeError('assemble before build_rhs')
        self.kernels.load(self.load_full, self.applied_load, self.lame_lambda, self.mu, alphas, temperature_rise,
                          temperature_rise_ext, body_forces, self.applied_body_force, self.widths_device)
        self.xp.subtract(self.load_full, self.fixed_coupling, out=self.rhs)
        self.xp.multiply(self.rhs, self.free, out=self.rhs)
        self._rhs_built = True

    def _assembled(self) -> bool:
        return self.lame_lambda is not None

    def _solvable(self) -> bool:
        return self.lame_lambda is not None and self._rhs_built

    def _apply_solve_operator(self, vector, out):
        """out = A_solve vector for vector = 0 on fixed dofs (D_free K D_free + D_fixed acts as the masked product)."""
        self.kernels.product(vector, out, self.lame_lambda, self.mu, self.widths_device, self.free, masked=True)

    def _precondition(self, residual, out):
        if self._multigrid_active and residual.dtype == self.multigrid.dtype:
            self.multigrid.precondition(residual, out)
            return
        if self._multigrid_active and self._cycle_out is not None:   # double CG: float32 cycle, cast back
            self.multigrid.precondition(residual, self._cycle_out)
            self.xp.copyto(out, self._cycle_out)
            return
        self.kernels.precondition(self.inverse_blocks, residual, out)

    def solve(self) -> None:
        """The base PCG solve under the multigrid guard (_guarded_solve)."""
        self._guarded_solve(super().solve)

    def multigrid_budget(self) -> int:
        """CG iterations a multigrid solve may take (_guarded_solve): the block-Jacobi estimate (pcg
        CG_ITERATIONS_PER_AXIS_CELL x the longest window axis) in V-cycles, within the multigrid3d fallback bounds."""
        jacobi = CG_ITERATIONS_PER_AXIS_CELL * max(self.element_shape)
        return int(min(FALLBACK_ITERATIONS, max(FALLBACK_MIN_ITERATIONS, jacobi // CYCLE_COST_IN_JACOBI_ITERATIONS)))

    def _guarded_solve(self, solve_once) -> None:
        """solve_once(); with multigrid capped at multigrid_budget() iterations: a solve that did not converge within
        them switches this system to block Jacobi for good (warning) and re-solves from where it stopped (that attempt
        then does not count as unconverged)."""
        solver = self.linear_solver
        unconverged_before = solver.n_unconverged
        budget = self.multigrid_budget() if self._multigrid_active else None
        limit = solver.max_iterations
        if budget is not None:
            solver.max_iterations = min(limit, budget)
        try:
            solve_once()
        finally:
            solver.max_iterations = limit
        if not self._multigrid_active or (solver.last_converged and solver.last_iters <= budget):
            return
        self._multigrid_active = False
        self.multigrid.fallbacks += 1
        solver.release_block_graphs()   # captured blocks hold the V-cycle launches
        converged = solver.last_converged
        warnings.warn(f'{self.label()}: the multigrid preconditioner '
                      + (f'needed {solver.last_iters} CG iterations' if converged
                         else f'did not converge within {solver.last_iters} CG iterations (the block-Jacobi cost)')
                      + ' on this geometry (coarse levels of thin solids in void, strong stiffness contrast or strong '
                        'grading are weak); this solid continues with block Jacobi', RuntimeWarning, stacklevel=3)
        if not converged:
            solver.n_unconverged = unconverged_before
            solve_once()

    def _apply_single_operator(self, vector, out):
        """Float32 A_solve vector of the mixed inner CG (the float32 RawKernel on the copies of the last assembly)."""
        lame_lambda, mu, _ = self._single_operator
        self.kernels.product(vector, out, lame_lambda, mu, self.widths_single, self.free, masked=True)

    def _precondition_single(self, residual, out):
        if self._multigrid_active:
            self.multigrid.precondition(residual, out)
            return
        self.kernels.precondition(self._single_operator[2], residual, out)

    def _fused_operator(self):
        """(lame_lambda, mu, inverse blocks, widths, free dofs) of the latest assembly (the fused CG iteration)."""
        return self.lame_lambda, self.mu, self.inverse_blocks, self.widths_device, self.free

    def _fused_operator_single(self):
        """The float32 copies of _fused_operator (mixed inner CG)."""
        lame_lambda, mu, inverse_blocks = self._single_operator
        return lame_lambda, mu, inverse_blocks, self.widths_single, self.free

    # --- post-processing --------------------------------------------------------

    def reaction(self, dof_mask) -> tuple:
        """(F_x, F_y, F_z) [N] the supports exert on the solid at the masked dofs: sum of K u - f of the last solve."""
        if self.n_solves == 0:
            raise RuntimeError(f'{self.label()}: reactions need a solve first')
        xp = self.xp
        product = self._solution   # free between solves: K u kept there for the next reaction of the same solve
        key = (self.n_solves, self._assemblies)
        if self._reaction_key != key:
            self.kernels.product(self.unknown, product, self.lame_lambda, self.mu, self.widths_device, self.free,
                                 masked=False)
            self._reaction_key = key
        dofs = xp.flatnonzero(xp.asarray(dof_mask))
        nodal = product[dofs].astype(np.float64) - self.load_full[dofs].astype(np.float64)
        totals = to_host(xp.bincount(dofs % 3, weights=nodal, minlength=3)) if dofs.size else np.zeros(3)
        return float(totals[0]), float(totals[1]), float(totals[2])

    def write_outputs(self, output_arrays, alphas, temperature_rise, temperature_rise_ext, thermal=True) -> None:
        """Cell-centre outputs of the unknown into full-grid float32 arrays (kernels3d.OUTPUT_ARRAYS_3D order; window
        cells only, 0 in void cells); thermal False: dT = 0 (a mode shape)."""
        if self.lame_lambda is None:
            raise RuntimeError('assemble before write_outputs')
        self.kernels.outputs(output_arrays, self.unknown, self.lame_lambda, self.mu, alphas, temperature_rise,
                             temperature_rise_ext, self.widths_device, thermal=thermal)

    # --- vibration modes (eigenfrequency study) ----------------------------------

    def lowest_modes(self, density, n_resonances, eigen_solver):
        """modal.ModalSolve of the n_resonances lowest modes of K u = omega^2 M u (consistent mass of the window element
        densities [kg/m^3], 0 = void): matrix-free shift-invert at sigma = 0, one PCG solve of A_solve per Lanczos step.
        Uses rhs / the CG unknown as scratch (the next static solve rebuilds both)."""
        if self.lame_lambda is None:
            raise RuntimeError('assemble before lowest_modes')
        if self.dtype != np.float64:
            raise ValueError(f"{self.label()}: the eigenfrequency study needs a double-precision 'linear' pcg solver "
                             '(its inner shift-invert solves), got single')
        xp = self.xp
        density = xp.ascontiguousarray(density, dtype=self.dtype)
        free_index = xp.flatnonzero(self.free)
        lanczos_vector = xp.zeros(self.n_dofs, dtype=self.dtype)   # 0 on fixed dofs throughout
        rhs, solution = self.rhs, self._solution
        kernels, widths, solver = self.kernels, self.widths_device, self.linear_solver
        iterations_before, unconverged_before = solver.total_iters, solver.n_unconverged
        self._rhs_built = False

        def mass(vector):
            lanczos_vector[free_index] = xp.asarray(vector, dtype=self.dtype)
            kernels.mass_product(lanczos_vector, rhs, density, widths, self.free)
            return np.array(to_host(rhs[free_index]))

        def inverse_stiffness(vector):
            rhs.fill(0)
            rhs[free_index] = xp.asarray(vector, dtype=self.dtype)
            solution.fill(0)
            self._guarded_solve(lambda: solver.solve(rhs, solution))
            return np.array(to_host(solution[free_index]))

        modal_solve = solve_lowest_modes_matrix_free(
            inverse_stiffness, mass, to_host(free_index), self.n_dofs, n_resonances, 3, eigen_solver,
            LANCZOS_TOLERANCE_PER_CG * solver.tolerance)
        modal_solve.info['cg_iterations'] = solver.total_iters - iterations_before
        n_unconverged = solver.n_unconverged - unconverged_before
        if n_unconverged:
            modal_solve.info['warnings'].append(
                f'{n_unconverged} inner PCG solve(s) did not converge (last: {solver.last_warning}): the modes are '
                'inaccurate; raise the linear solver max_iterations')
        return modal_solve


def _face(face):
    if face not in FACES_3D:
        raise ValueError(f"unknown face {face!r}; expected one of {FACES_3D}")
    return face


def rigid_motion_text(null_basis, centroid, scale, extent) -> str:
    """Text of one free motion u = t + omega~ x (x - c) / s of a rigid-mode null space (6, k) (rows (e_a, x~ x e_a));
    a pure slide when one exists; extent [m] sets the round-off zero of the axis point (also ElasticFEMSystem3D)."""
    rotation_part = null_basis[3:, :]
    _, singular, right = np.linalg.svd(rotation_part)
    rotation_rank = int(np.count_nonzero(singular > _MOTION_TOLERANCE))
    vector = null_basis @ right[rotation_rank] if rotation_rank < null_basis.shape[1] else null_basis[:, 0]
    translation, omega = vector[:3], vector[3:]
    if np.linalg.norm(omega) <= _MOTION_TOLERANCE * np.linalg.norm(vector):
        return f'a slide along {_direction_text(translation)}'
    point = centroid + scale * np.cross(omega, translation) / np.dot(omega, omega)
    point[np.abs(point) < 1e-9 * extent] = 0.0
    text = (f'a rotation about the axis along {_direction_text(omega)} through ({point[0]:.6g}, {point[1]:.6g}, '
            f'{point[2]:.6g}) m')
    if abs(np.dot(translation, omega)) > _MOTION_TOLERANCE * np.linalg.norm(translation) * np.linalg.norm(omega):
        text += ' combined with a slide along that axis'
    return text


def _direction_text(vector) -> str:
    """'x' / 'y' / 'z' for an axis-aligned direction, else the unit vector."""
    unit = np.asarray(vector, dtype=np.float64) / np.linalg.norm(vector)
    unit[np.abs(unit) < _MOTION_TOLERANCE] = 0.0
    axes = np.flatnonzero(unit)
    if axes.size == 1:
        return 'xyz'[int(axes[0])]
    return '(' + ', '.join(f'{value:.3g}' for value in unit) + ')'
