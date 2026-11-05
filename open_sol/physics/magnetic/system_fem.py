"""MagneticFEMSystem: 2D magnetostatics of A (planar A_z [T m], axisymmetric A_phi [T m]) on a triangle mesh (magnetic
plan section 10, stage 5): Lagrange P1 / P2 on the window elements in B-space (fem.magnetostatics), Dirichlet faces on
the window edge, the axis A = 0, held elements, the external field as nodal Dirichlet values; linear per-axis nu with a
remanence load, or isotropic saturating curves nu(|B|); the minimum of the convex energy W(a) = int w(B) dV - loads . a
by exact-tangent Newton with an Armijo line search on W (one linear solve without saturating elements), each step
solved by the 'linear' role runtime on the run backend. Host float64 numpy, scipy inside functions."""
import numpy as np

from ...core.backend import array_module, sparse_module, to_host
from ...core.damped_iteration import DampingSettings
from ...fem.assembly import assemble_load, nodal_average
from ...fem.diffusion import floating_region_count, ring_dof_locations, window_face_edges
from ...fem.dofs import lagrange_dofs
from ...fem.elasticity import local_dof_points
from ...fem.elements import lagrange_basis
from ...fem.fixed_pattern import FixedPatternCsr
from ...fem.magnetostatics import (assemble_curl_tangent, curl_quadrature, curl_tangent_blocks, curl_vector,
                                   element_integrals, local_flux_density, quadrature_flux_density, quadrature_integral,
                                   quadrature_values)
from ...fem.mesher import MERGE_TOLERANCE
from ...fem.sampling import evaluate

FACES = ('x_lo', 'x_hi', 'y_lo', 'y_hi')
ARMIJO_FRACTION = 1e-4                      # sufficient decrease of W per unit of the predicted R . delta
MIN_STEP = DampingSettings.min_relaxation   # smallest line-search step before the loop gives up
ENERGY_ROUNDING = 1e-12                     # relative slack of the energy test (round-off of W near the minimum)
CENTROID = np.array([[1.0 / 3.0, 1.0 / 3.0]])


class MagneticFEMSystem:
    """Magnetostatics of one TriangleMesh: `values` = A in the full-mesh numbering (nodes, then n_points + edge for
    P2; the PML layer ring holds the window field at the clamped position), `dofs` the window-element map, `blocks` its
    cached curl quadrature. The owner sets faces / held regions / the external field and the materials (set_materials),
    then solve(current density per element). Integrals are per radian when axisymmetric."""

    def __init__(self, mesh, order, axisymmetric, label, linear_solver=None, backend='cpu',
                 anchor_text='magnetic insulation face, held potential or external field', remedy_text=''):
        from ...solvers.sparse_direct import DiffusionDirectSolver, SparseDirectSolver
        if order not in (1, 2):
            raise ValueError(f'{label()}: element order must be 1 or 2, got {order!r}')
        if linear_solver is None:   # scripts: the triangle Automatic (mesh plan D11)
            linear_solver = SparseDirectSolver(backend=backend)
        if isinstance(linear_solver, SparseDirectSolver):
            linear_solver = DiffusionDirectSolver(linear_solver)
        if getattr(linear_solver, 'backend', backend) != backend:
            raise ValueError(f'{label()}: the linear solver runs on the {linear_solver.backend} backend, the mesh on '
                             f'the {backend} one')
        linear_solver.label = label
        self.linear_solver = linear_solver
        self.mesh = mesh
        self.order = int(order)
        self.geometry_order = 1 if self.order == 1 else 2   # D6: isoparametric P2 on curved elements
        self.axisymmetric = bool(axisymmetric)
        self.backend = backend
        self.label = label
        self.anchor_text = anchor_text
        self.remedy_text = remedy_text
        self.full = lagrange_dofs(mesh, self.order, geometry_order=self.geometry_order)
        self.element_map = lagrange_dofs(mesh, 0, geometry_order=self.geometry_order)
        self.window_elements = mesh.pml_faces == 0
        self.dofs = lagrange_dofs(mesh, self.order, self.window_elements, geometry_order=self.geometry_order)
        self._window_index = np.searchsorted(self.full.global_ids, self.dofs.global_ids)   # dofs -> full numbering
        self.blocks = curl_quadrature(self.dofs, self.axisymmetric)
        count = mesh.n_elements   # mesh-constant integrals of the blocks' rule (read-only)
        self.element_measure = element_integrals(self.blocks, [np.ones(block.weights.shape) for block in self.blocks],
                                                 count)   # int dV per element (per radian when axisymmetric)
        self.element_area = element_integrals(self.blocks, [1.0 / block.points[..., 0] if self.axisymmetric
                                                            else np.ones(block.weights.shape) for block in self.blocks],
                                              count)      # plain x-y / r-z area
        for array in (self.element_measure, self.element_area):
            array.flags.writeable = False
        self.values = np.zeros(self.full.n_dofs)
        self.started = False          # values hold a solution (or a seeded one): Newton starts from them
        x0, y0, x1, y1 = mesh.window
        self._tolerance = MERGE_TOLERANCE * max(x1 - x0, y1 - y0)
        # conditions (later ones win shared dofs: faces < external field < held elements < the axis)
        self.face_kinds = dict.fromkeys(FACES, 'dirichlet')   # 'dirichlet' (A = 0) or 'natural' (symmetry)
        self.external_field = None    # (B_x, B_y) [T]
        self.held = []                # (host bool element mask, held value: planar A, axisymmetric psi = r A)
        # materials (set_materials): per-axis linear nu, remanence, saturating curves per element
        self.reluctivity = None       # float64 (n_elements, 2) [m/H]
        self.remanence = None         # float64 (n_elements, 2) [T] or None
        self.curves = ()              # callables (|B| (k, q), mesh elements (k,)) -> (nu, nu_d, w)
        self.curve_index = np.full(mesh.n_elements, -1, dtype=np.int64)
        self.operator_dirty = True    # constraints changed
        self._free = self._fixed = self._fixed_values = None
        self._remanence_load = np.zeros(self.dofs.n_dofs)
        self._linear_matrix = None    # assembled K of a model without saturating elements (factor kept)
        self._tangent_pattern = None  # (blocks, free, FixedPatternCsr) of the Newton tangent
        self._ring = None
        self._state = None            # (B, D or None, H_curve, w) point lists of the last solve
        self.n_solves = 0
        self.n_runs = 0               # Newton loops
        self.total_iterations = 0
        self.total_backtracks = 0
        self.n_unconverged = 0
        self.last_iterations = 0
        self.last_residuals = []      # max |R_free| / scale per iterate of the last Newton loop (initial first)
        self.last_energies = []       # W per iterate of the last Newton loop
        self.last_warning = None

    # ---- conditions ----
    def set_face(self, face, kind) -> None:
        """'dirichlet' (A = 0: magnetic insulation) or 'natural' (dA/dn = 0: symmetry) on an outer face's window edge."""
        if kind not in ('dirichlet', 'natural'):
            raise ValueError(f"{self.label()}: face kind must be 'dirichlet' or 'natural', got {kind!r}")
        self.face_kinds[face] = kind
        self.operator_dirty = True

    def set_external_field(self, B_x, B_y) -> None:
        """Uniform applied flux density [T]: A = B_x y - B_y x (axisymmetric A_phi = B_y r / 2) on every window-edge
        dof, coordinates from the window's x_lo / y_lo corner."""
        if self.axisymmetric and B_x != 0.0:
            raise ValueError(f'{self.label()}: an axisymmetric external field is axial only (B_x = B_r must be 0, got '
                             f'{B_x})')
        self.external_field = (float(B_x), float(B_y))
        self.operator_dirty = True

    def add_held(self, element_mask, value) -> None:
        """Hold every dof of the masked elements at the planar A_z [T m] or the axisymmetric flux function psi = r
        A_phi [T m^2] (A_phi = psi / r there; psi != 0 on the axis is refused at assembly)."""
        self.held.append((np.asarray(element_mask, dtype=bool).copy(), float(value)))
        self.operator_dirty = True

    def face_edges(self, face) -> np.ndarray:
        return window_face_edges(self.mesh, face)

    def held_elements(self) -> np.ndarray:
        """Bool (n_elements,) window elements of every held region."""
        held = np.zeros(self.mesh.n_elements, dtype=bool)
        for mask, _ in self.held:
            held |= mask
        return held & self.window_elements

    def constraints(self) -> tuple:
        """(free, fixed, fixed values) of the window dof map, re-derived after a condition change (magnetic_fields_ac
        assembles its own complex operator on them)."""
        if self.operator_dirty:
            self._constrain()
        return self._free, self._fixed, self._fixed_values

    # ---- materials ----
    def set_materials(self, reluctivity, remanence=None, curves=(), curve_index=None) -> None:
        """Per-axis linear nu (n_elements, 2) [m/H], remanence (n_elements, 2) [T] or None, saturating curves (callables
        (|B|, elements) -> (nu, nu_d, w)) with curve_index per element (-1 linear)."""
        self.reluctivity = np.asarray(reluctivity, dtype=np.float64)
        self.remanence = None if remanence is None else np.asarray(remanence, dtype=np.float64)
        self.curves = tuple(curves)
        self.curve_index = (np.full(self.mesh.n_elements, -1, dtype=np.int64) if curve_index is None
                            else np.asarray(curve_index, dtype=np.int64))
        self._remanence_load = np.zeros(self.dofs.n_dofs)
        if self.remanence is not None and self.remanence.any():
            drives = [np.broadcast_to((self.reluctivity[block.elements] * self.remanence[block.elements])[:, None, :],
                                      block.weights.shape + (2,)) for block in self.blocks]
            self._remanence_load = curl_vector(self.blocks, drives, self.dofs.n_dofs)
        self._linear_matrix = None

    @property
    def nonlinear(self) -> bool:
        """True when an element follows a saturating curve (Newton runs)."""
        return bool(self.curves) and bool((self.curve_index >= 0).any())

    def constitutive(self, elements, flux, tangent=True):
        """(H_curve = dw/dB (m, q, 2), D = dH/dB (m, q, 2, 2) or None, w (m, q)) at flux densities (m, q, 2) of mesh
        elements: linear diag(nu) B and B . diag(nu) B / 2, saturating nu(|B|) B with D = nu I + (nu_d - nu) e e^T
        (e = B / |B|; nu I at B = 0). Remanence is a load, not part of this law."""
        nu = self.reluctivity[elements][:, None, :]
        field = nu * flux
        energy = 0.5 * np.sum(field * flux, axis=-1)
        stiffness = None
        if tangent:
            stiffness = np.zeros(flux.shape + (2,))
            stiffness[..., 0, 0] = nu[..., 0]
            stiffness[..., 1, 1] = nu[..., 1]
        curve_index = self.curve_index[elements]
        for index, curve in enumerate(self.curves):
            rows = np.flatnonzero(curve_index == index)
            if not rows.size:
                continue
            local = flux[rows]
            magnitude = np.sqrt(np.sum(local * local, axis=-1))
            secant, differential, curve_energy = curve(magnitude, elements[rows])
            field[rows] = secant[..., None] * local
            energy[rows] = curve_energy
            if tangent:   # nu I + (nu_d - nu) e e^T written per component (no (k, q, 2, 2) temporaries)
                safe = np.where(magnitude > 0.0, magnitude, 1.0)
                along_x, along_y = local[..., 0] / safe, local[..., 1] / safe
                excess = differential - secant
                cross = excess * along_x * along_y
                block = np.empty(local.shape + (2,))
                block[..., 0, 0] = secant + excess * along_x * along_x
                block[..., 1, 1] = secant + excess * along_y * along_y
                block[..., 0, 1] = cross
                block[..., 1, 0] = cross
                stiffness[rows] = block
        return field, stiffness, energy

    # ---- assembly and solve ----
    def _constrain(self) -> None:
        """Fixed dofs and values: Dirichlet faces, the external field, held elements, the axis (later wins)."""
        dofs, mesh = self.dofs, self.mesh
        fixed_value = np.full(dofs.n_dofs, np.nan)
        points = dofs.dof_points()
        x0, y0 = mesh.window[0], mesh.window[1]
        for face, kind in self.face_kinds.items():
            if kind == 'dirichlet':
                fixed_value[dofs.edge_dofs(self.face_edges(face))] = 0.0
        if self.external_field is not None:
            B_x, B_y = self.external_field
            ids = np.unique(np.concatenate([dofs.edge_dofs(self.face_edges(face)) for face in FACES]))
            x, y = points[ids, 0] - x0, points[ids, 1] - y0
            fixed_value[ids] = 0.5 * B_y * x if self.axisymmetric else B_x * y - B_y * x
        for mask, value in self.held:
            ids = dofs.edge_dofs(mesh.element_edges[mask & self.window_elements].ravel())
            if self.axisymmetric:
                radius = points[ids, 0]
                on_axis = radius <= self._tolerance
                if value != 0.0 and on_axis.any():
                    raise ValueError(f'{self.label()}: a held region at psi = {value:g} T m^2 touches the axis, where '
                                     'psi = r A_phi is 0: hold 0 there or keep the region off the axis')
                fixed_value[ids] = np.where(on_axis, 0.0, value / np.where(on_axis, 1.0, radius))
            else:
                fixed_value[ids] = value
        if self.axisymmetric:
            fixed_value[dofs.edge_dofs(self.face_edges('x_lo'))] = 0.0
        self._fixed = np.flatnonzero(~np.isnan(fixed_value))
        self._free = np.flatnonzero(np.isnan(fixed_value))
        if not self._free.size:
            raise ValueError(f'{self.label()}: every vector-potential dof is held, nothing is solved')
        self._fixed_values = fixed_value[self._fixed]
        self._linear_matrix = None
        self.operator_dirty = False

    def solve(self, current_density, tolerance=1e-6, max_iterations=100) -> bool:
        """Minimise W for the current density J [A/m^2] per element (P0 load int J phi dV) plus the remanence load,
        from the last solution (fixed dofs at their values): one linear solve without saturating elements, else
        Newton; writes `values` (PML ring: clamped window values) and the point state; False when Newton stopped
        unconverged (last_warning)."""
        if self.reluctivity is None:
            raise RuntimeError(f'{self.label()}: set_materials before solve')
        if self.operator_dirty:
            self._constrain()
        loads = self._loads(current_density)
        potential = self.values[self._window_index].copy()
        potential[self._fixed] = self._fixed_values
        converged = True
        if self.nonlinear:
            potential, converged = self._newton(potential, loads, tolerance, max_iterations)
        else:
            potential = self._linear_solve(potential, loads)
        self.values[self._window_index] = potential
        self._extend_into_ring(self.values)
        self.n_solves += 1
        self.started = True
        return converged

    def _loads(self, current_density) -> np.ndarray:
        """(n_dofs,) remanence load + int J phi dV of a per-element current density (None: none) on the window."""
        loads = self._remanence_load.copy()
        if current_density is not None:
            density = np.asarray(current_density, dtype=np.float64)
            if density.any():
                loads += assemble_load(self.dofs, np.where(self.window_elements, density, 0.0),
                                       axisymmetric=self.axisymmetric)
        return loads

    def weak_form(self, potential, current_density=None, tangent=True) -> tuple:
        """(W, R = internal force - loads, K_t CSR or None) at dof values of the window map, unconstrained: the energy
        functional, its gradient and Hessian on one quadrature (the self-test's finite-difference check)."""
        if self.reluctivity is None:
            raise RuntimeError(f'{self.label()}: set_materials before weak_form')
        state, residual, energy, _ = self._evaluate(np.asarray(potential, dtype=np.float64),
                                                    self._loads(current_density))
        matrix = assemble_curl_tangent(self.blocks, state[1], self.dofs.n_dofs).tocsr() if tangent else None
        return energy, residual, matrix

    def prepare_solve(self) -> None:
        """Constraints and, without saturating elements, the assembled, anchor-checked and factorised K (build-time
        failures surface before the first solve)."""
        if self.reluctivity is None:
            raise RuntimeError(f'{self.label()}: set_materials before prepare_solve')
        if self.operator_dirty:
            self._constrain()
        if not self.nonlinear and self._linear_matrix is None:
            self._linear_matrix = self._assemble_linear()

    def _assemble_linear(self):
        """K = int Q^T D(B = 0) Q dV (the linear operator; the B = 0 limit of saturating curves), anchor check and
        factor of its free block; returns K."""
        tangents = [self.constitutive(block.elements, np.zeros(block.weights.shape + (2,)))[1]
                    for block in self.blocks]
        matrix = assemble_curl_tangent(self.blocks, tangents, self.dofs.n_dofs).tocsr()
        reduced = matrix[self._free][:, self._free].tocsr()
        self._check_anchors(reduced)
        self._factorize(reduced)
        return matrix

    def _lifted_solve(self, matrix, potential, loads):
        """potential with its free dofs = the factorised K_ff^-1 (loads_f - K_fc a_c)."""
        rhs = loads[self._free] - matrix[self._free][:, self._fixed] @ potential[self._fixed]
        potential[self._free] = self._solve_factorized(rhs, potential[self._free])
        return potential

    def _linear_solve(self, potential, loads):
        """K a = loads on the free dofs with the fixed values lifted; K assembled and factorised once per change."""
        if self._linear_matrix is None:
            self._linear_matrix = self._assemble_linear()
        potential = self._lifted_solve(self._linear_matrix, potential, loads)
        self._state = self._point_state(potential)
        return potential

    def _newton(self, potential, loads, tolerance, max_iterations):
        """Exact-tangent Newton on W with the Armijo test W(a + s d) <= W(a) + ARMIJO_FRACTION s R . d (+ round-off
        slack), s halved down to MIN_STEP; stop when max |s d| <= tol max |a| and max |R_free| <= tol x max(|loads|,
        |internal force|). Without a previous solution (started False) it starts from the B = 0 limit solve."""
        free = self._free
        self.n_runs += 1
        self.last_warning = None
        anchors_checked = not self.started   # _assemble_linear checks them on the same pattern
        if anchors_checked:
            potential = self._lifted_solve(self._assemble_linear(), potential, loads)
        state, residual, energy, scale = self._evaluate(potential, loads)
        self.last_residuals = [float(np.abs(residual[free]).max(initial=0.0)) / scale]
        self.last_energies = [energy]
        iterations, converged = 0, False
        while iterations < max_iterations:
            reduced = self._reduced_tangent(state[1])
            if iterations == 0 and not anchors_checked:
                self._check_anchors(reduced)
            self._factorize(reduced)
            step_free = self._solve_factorized(-residual[free], np.zeros(free.size))
            direction = np.zeros_like(potential)
            direction[free] = step_free
            slope = float(residual[free] @ step_free)   # < 0 for the SPD tangent
            if not slope < 0.0:
                if self.last_residuals[-1] <= tolerance:   # already at the minimum (an unchanged re-solve)
                    converged = True
                    break
                self.last_warning =(f'{self.label()}: Newton step {iterations + 1} is no descent direction (R . d = '
                                     f'{slope:.3g}); stopped')
                break
            slack = ENERGY_ROUNDING * (abs(energy) + abs(float(loads @ potential)) + abs(slope))
            step = 1.0
            while True:
                trial = potential + step * direction
                trial_energy = self._energy(trial, loads)
                if trial_energy <= energy + ARMIJO_FRACTION * step * slope + slack:
                    break
                if step <= MIN_STEP:
                    break
                step *= 0.5
                self.total_backtracks += 1
            if trial_energy > energy + slack:
                self.last_warning = (f'{self.label()}: the line search of Newton step {iterations + 1} found no '
                                     f'energy decrease down to step {MIN_STEP:g}; stopped')
                break
            potential = trial
            iterations += 1
            state, residual, energy, scale = self._evaluate(potential, loads)
            relative_residual = float(np.abs(residual[free]).max(initial=0.0)) / scale
            self.last_residuals.append(relative_residual)
            self.last_energies.append(energy)
            change = step * float(np.abs(step_free).max(initial=0.0))
            if (change <= tolerance * float(np.abs(potential).max(initial=0.0))
                    and relative_residual <= tolerance):
                converged = True
                break
        else:
            self.last_warning = (f'{self.label()}: Newton did not converge in {max_iterations} steps (relative '
                                 f'residual {self.last_residuals[-1]:.3g}); the fields are those of the last step')
        self.last_iterations = iterations
        self.total_iterations += iterations
        if not converged:
            self.n_unconverged += 1
            if self.last_warning is None:
                self.last_warning = f'{self.label()}: Newton stopped unconverged'
        self._state = state
        return potential, converged

    def _reduced_tangent(self, tangents):
        """Canonical free-free CSR of int Q^T D Q dV: element blocks summed into a FixedPatternCsr kept while the
        blocks and the free set stay (no COO sort and row / column slicing per Newton step)."""
        kept = self._tangent_pattern
        if kept is None or kept[0] is not self.blocks or kept[1] is not self._free:
            free_mask = np.zeros(self.dofs.n_dofs, dtype=bool)
            free_mask[self._free] = True
            pattern = FixedPatternCsr(self.dofs.n_dofs, [block.element_dofs for block in self.blocks], free=free_mask)
            kept = self._tangent_pattern = (self.blocks, self._free, pattern)
        pattern = kept[2]
        data = pattern.zeros()
        for group, local in enumerate(curl_tangent_blocks(self.blocks, tangents)):
            pattern.add(data, group, local)
        return pattern.reduced(data)

    def _evaluate(self, potential, loads):
        """((B, D, H_curve, w) point lists, residual R = internal - loads, W, residual scale) at dof values."""
        flux = quadrature_flux_density(self.blocks, potential)
        laws = [self.constitutive(block.elements, block_flux) for block, block_flux in zip(self.blocks, flux)]
        fields = [law[0] for law in laws]
        internal = curl_vector(self.blocks, fields, self.dofs.n_dofs)
        energy = quadrature_integral(self.blocks, [law[2] for law in laws]) - float(loads @ potential)
        scale = max(float(np.abs(loads).max(initial=0.0)), float(np.abs(internal).max(initial=0.0)), 1e-300)
        return (flux, [law[1] for law in laws], fields, [law[2] for law in laws]), internal - loads, energy, scale

    def _energy(self, potential, loads) -> float:
        """W(a) = int w(B) dV - loads . a (one quadrature, no tangent)."""
        flux = quadrature_flux_density(self.blocks, potential)
        energies = [self.constitutive(block.elements, block_flux, tangent=False)[2]
                    for block, block_flux in zip(self.blocks, flux)]
        return quadrature_integral(self.blocks, energies) - float(loads @ potential)

    def _point_state(self, potential):
        flux = quadrature_flux_density(self.blocks, potential)
        laws = [self.constitutive(block.elements, block_flux, tangent=False) for block, block_flux
                in zip(self.blocks, flux)]
        return flux, None, [law[0] for law in laws], [law[2] for law in laws]

    def _check_anchors(self, reduced) -> None:
        """Planar only (the axisymmetric A / r term anchors every region): a free region of row sums ~0 is singular."""
        if self.axisymmetric:
            return
        floating = floating_region_count(reduced)
        if floating:
            raise ValueError(f'{self.label()}: {floating} connected region(s) of the vector potential touch no '
                             f'{self.anchor_text}, so the system is singular. {self.remedy_text}')

    def _solve_dtype(self):
        return np.float32 if getattr(self.linear_solver, 'precision', 'double') == 'single' else np.float64

    def _factorize(self, reduced) -> None:
        self.linear_solver.factorize(sparse_module(self.backend).csr_matrix(reduced.astype(self._solve_dtype())))

    def _solve_factorized(self, rhs, initial) -> np.ndarray:
        """Host float64 solution of the factorised reduced system (iterative solvers start from `initial`)."""
        xp = array_module(self.backend)
        dtype = self._solve_dtype()
        solution = xp.asarray(initial, dtype=dtype)
        self.linear_solver.solve(xp.asarray(rhs, dtype=dtype), solution)
        return np.asarray(to_host(solution), dtype=np.float64)

    def full_values(self, potential) -> np.ndarray:
        """Full-mesh array (real or complex) of window dof values; the PML ring dofs take the clamped window values."""
        potential = np.asarray(potential)
        values = np.zeros(self.full.n_dofs, dtype=np.result_type(potential.dtype, np.float64))
        values[self._window_index] = potential
        self._extend_into_ring(values)
        return values

    def nodal_flux_density(self, potential) -> np.ndarray:
        """Full-mesh (n_dofs, 2) nodal average of B = curl A of window dof values (real or complex) from the element
        values at their local dof points (axis points: the A / r limit), ring-extended."""
        flux = local_flux_density(self.dofs, np.asarray(potential), local_dof_points(self.order), self.axisymmetric,
                                  self._tolerance)
        complex_flux = np.iscomplexobj(flux)
        averaged = nodal_average(self.dofs, np.concatenate([flux.real, flux.imag], axis=-1) if complex_flux else flux)
        if complex_flux:
            averaged = averaged[:, :2] + 1j * averaged[:, 2:]
        return np.stack([self.full_values(averaged[:, column]) for column in range(2)], axis=1)

    def _extend_into_ring(self, values) -> None:
        """PML ring dofs of a full-mesh nodal field = its value at the position clamped into the window."""
        if self._ring is None:
            self._ring = ring_dof_locations(self.full)
        ids, location = self._ring
        if ids.size:
            values[ids] = evaluate(self.full, values, location=location)

    # ---- post-processing ----
    def physical_fields(self):
        """(B, H) point lists of the last solve: H = H_curve - nu B_r (remanence of the linear elements)."""
        flux, _, fields, _ = self._state
        if self.remanence is None:
            return flux, fields
        physical = [field - self.reluctivity[block.elements][:, None, :] * self.remanence[block.elements][:, None, :]
                    for block, field in zip(self.blocks, fields)]
        return flux, physical

    def active_elements(self) -> np.ndarray:
        """Float (n_elements,) 1 on window elements outside held regions, else 0 (held axisymmetric psi / r
        interpolates to a spurious B; tensor held cells read 0 too)."""
        return np.where(self.window_elements & ~self.held_elements(), 1.0, 0.0)

    def element_means(self) -> dict:
        """Per mesh element (0 off the window): 'A' at the centroid, means of 'B' (n, 2), 'H' (n, 2) and 'w' of the
        last solve (0 in held elements)."""
        count = self.mesh.n_elements
        flux, field = self.physical_fields()
        measure = self.element_measure
        scale = self.active_elements() / np.where(measure > 0.0, measure, 1.0)
        means = {'B': element_integrals(self.blocks, flux, count) * scale[:, None],
                 'H': element_integrals(self.blocks, field, count) * scale[:, None],
                 'w': element_integrals(self.blocks, self._state[3], count) * scale}
        means['A'] = np.where(self.window_elements,   # ring dofs hold clamped window values: no ring state
                              self.values[self.full.element_dofs] @ lagrange_basis(self.order, CENTROID)[0][0], 0.0)
        return means

    def stored_energy(self) -> float:
        """int w(B) dV over the active elements of the last solve (per radian when axisymmetric): the energy of B,
        remanence load and held elements excluded."""
        active = self.active_elements()
        return quadrature_integral(self.blocks, [energy * active[block.elements][:, None]
                                                 for block, energy in zip(self.blocks, self._state[3])])

    def region_integrals(self, element_mask) -> tuple:
        """(int A dV, int dA) over the masked window elements of the last solve: A per radian when axisymmetric (r
        weights), the plain x-y / r-z area."""
        mask = np.asarray(element_mask, dtype=bool)
        count = self.mesh.n_elements
        potential = self.values[self._window_index]
        integral = element_integrals(self.blocks, quadrature_values(self.blocks, potential), count)
        return float(integral[mask].sum()), float(self.element_area[mask].sum())

    def nodal_fields(self) -> dict:
        """Full-mesh nodal display values of the last solve: 'B_x', 'B_y', 'H_x', 'H_y' averaged from the element
        values at their local dof points over the window elements (axis points: the A / r limit), ring-extended."""
        dofs = self.dofs
        reference = local_dof_points(self.order)
        potential = self.values[self._window_index]
        flux = local_flux_density(dofs, potential, reference, self.axisymmetric, self._tolerance)
        field, _, _ = self.constitutive(dofs.elements, flux, tangent=False)
        if self.remanence is not None:
            field = field - (self.reluctivity[dofs.elements] * self.remanence[dofs.elements])[:, None, :]
        averaged = nodal_average(dofs, np.concatenate([flux, field], axis=-1))
        result = {}
        for column, name in enumerate(('B_x', 'B_y', 'H_x', 'H_y')):
            values = np.zeros(self.full.n_dofs)
            values[self._window_index] = averaged[:, column]
            self._extend_into_ring(values)
            result[name] = values
        return result
