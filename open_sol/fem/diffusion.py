"""FEM scalar diffusion on triangle meshes (mesh plan 2.7, stage T4): -div(C grad u) + a u (+ c du/dt) = f by Lagrange
P1 / P2 on the conducting window elements. FEMDiffusionSystem: host assembly (fem.assembly), Dirichlet / Neumann /
Robin edge conditions (outer faces on the window edge, object outlines), held element regions, P0 and point sources,
implicit-Euler capacity, (r, z) ring weights per radian; the linear solve by a solver of the PCGSolver protocol on the
run backend. FEMDiffusionPhysicsBase: the shared physics wrapper of heat and electric_currents / electrostatics on
triangles. fem_diffusion_estimate / fem_system_estimate: their memory estimate; cudss_diffusion_estimate: a
DiffusionDirectSolver's cuDSS factor."""
import math
from dataclasses import dataclass

import numpy as np

from ..core.backend import array_module, sparse_module, to_host
from ..core.constants import DTYPE
from ..core.fv_diffusion import BCSide
from ..core.memory_estimate import (FLOAT32_BYTES, FLOAT64_BYTES, FLOAT64_CSR_ENTRY_BYTES, INT32_BYTES, MemoryEstimate,
                                    combine, host_real_lu_bytes)
from ..core.monitors import FieldProbe
from ..core.solver_base import PhysicsSolver
from .assembly import (_element_groups, _lagrange_at, _weights, assemble_edge_load, assemble_edge_mass, assemble_load,
                       assemble_mass, assemble_stiffness, element_basis_integrals, element_gradient_integrals,
                       pattern_sum)
from .dofs import lagrange_dofs
from .elements import element_geometry, lagrange_basis
from .mesh import one_sided, side_members
from .mesher import MERGE_TOLERANCE
from .sampling import evaluate, locator

FACES = ('x_lo', 'x_hi', 'y_lo', 'y_hi')
RING_ANGLE = 2.0 * math.pi        # whole rings of an axisymmetric (r, z) solve (integrals here are per radian)
ANCHOR_RTOL = 1e-10               # reduced row sum / diagonal above which a row anchors its connected region
FEM_FILL_FACTOR = 2.0             # SuperLU fill of an FEM operator over the 5-point heuristic (uncalibrated)


@dataclass(frozen=True)
class EdgeCondition:
    """Condition on mesh edges in the unknown's units: 'dirichlet' (u = value), 'neumann' (inward flux value [c u / m])
    or 'robin' (inward flux h (external - u); h and external numbers or arrays per mesh edge / 3D face, e.g. a
    linearised radiation loss)."""
    kind: str = 'neumann'
    value: float = 0.0
    h: float = 0.0
    external: float = 0.0


class FEMDiffusionSystemCommon:
    """Dimension-free part of FEMDiffusionSystem (triangles) and diffusion3d.FEMDiffusionSystem3D (tetrahedra): the
    linear solver, condition sequences and factor reuse, held regions, volumetric losses, the anchor check and the
    factorisation / solve of the reduced operator. Subclasses provide `values`, `face_conditions` and _full_ids."""

    def __init__(self, mesh, order, linear_solver, backend, label, unknown_name, anchor_text, remedy_text,
                 require_anchors):
        from ..solvers.sparse_direct import DiffusionDirectSolver, SparseDirectSolver
        if order not in (1, 2):
            raise ValueError(f'{label()}: element order must be 1 or 2, got {order!r}')
        if linear_solver is None:   # scripts: the triangle / tetrahedra Automatic (mesh plan D11: host SuperLU)
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
        self.backend = backend
        self.label = label
        self.unknown_name = unknown_name
        self.anchor_text = anchor_text
        self.remedy_text = remedy_text
        self.require_anchors = bool(require_anchors)
        self.held = []              # (element mask, value, sequence): every (active) dof of the elements fixed
        self.loss_coefficient = np.zeros(mesh.n_elements)   # a per element
        self.loss_drive = np.zeros(mesh.n_elements)         # a u_ext per element
        self.n_solves = 0
        self.operator_dirty = True
        self._constants_stale = False   # a Dirichlet value / Robin external changed: new lift and loads, same factor
        self._sequence = 0          # later conditions win shared dofs
        self._matrix = self._reduced = self._capacity_matrix = None
        self._free = self._fixed = self._dof_owner = None
        self._coupling = None       # reduced rows x fixed columns of the last assembly (the lift)
        self._lift = self._static_load = self._load = self._solution = None

    def _full_ids(self, ids) -> np.ndarray:
        """Full-mesh ids (into `values`) of system dofs (the numbering of _free / _fixed)."""
        raise NotImplementedError

    # ---- conditions and sources ----
    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def _note_change(self, previous, condition) -> None:
        """Operator dirty when a condition change alters the matrix, else (a Dirichlet value, a Robin external at the
        same h) only the lift and loads are refreshed at the next solve; Neumann values are read at every solve."""
        if not _same_operator(previous, condition):
            self.operator_dirty = True
        elif condition.kind != 'neumann':
            self._constants_stale = True

    def set_face_condition(self, face, condition, keep_sequence=False) -> None:
        """Condition of an outer face (the window edge / faces on that side); a value-only change keeps the factor;
        keep_sequence (a value of time t): a set face keeps its place in the order, so shared corner / edge dofs stay
        with their build owner (else it becomes the latest condition)."""
        previous, sequence = self.face_conditions[face]
        if not (keep_sequence and sequence):
            sequence = self._next_sequence()   # dof owners are re-read with the constants
        self.face_conditions[face] = (condition, sequence)
        self._note_change(previous, condition)

    def add_held(self, element_mask, value) -> None:
        """Fix u = value on every dof of the masked elements (held cells: a terminal / reservoir)."""
        mask = np.asarray(element_mask, dtype=bool).copy()
        self.held.append((mask, float(value), self._next_sequence()))
        self.set_element_values(mask, value)
        self.operator_dirty = True

    def add_loss(self, element_mask, coefficient, external) -> None:
        """Add a volumetric loss a (u - external) on the masked elements (a, external numbers or per element)."""
        mask = np.asarray(element_mask, dtype=bool)
        count = self.mesh.n_elements
        coefficient = np.broadcast_to(np.asarray(coefficient, dtype=np.float64), (count,))
        external = np.broadcast_to(np.asarray(external, dtype=np.float64), (count,))
        if (coefficient[mask] < 0.0).any():
            raise ValueError(f'{self.label()}: a volumetric loss coefficient must be >= 0')
        self.loss_coefficient += np.where(mask, coefficient, 0.0)
        self.loss_drive += np.where(mask, coefficient * external, 0.0)
        self.operator_dirty = True

    # ---- factor and solve ----
    def _check_anchors(self, reduced, anchored_rows) -> None:
        """ValueError when a connected region of free dofs has no anchor (no anchored_rows row and all its reduced row
        sums ~0: u + const solves it too)."""
        floating = floating_region_count(reduced, anchored_rows)
        if floating:
            raise ValueError(f'{self.label()}: {floating} connected region(s) of the {self.unknown_name} touch no '
                             f'{self.anchor_text}, so the system is singular. {self.remedy_text}')

    def _solve_dtype(self):
        return np.float32 if getattr(self.linear_solver, 'precision', 'double') == 'single' else np.float64

    def _factorize(self) -> None:
        """Factorise (bind) the reduced operator on the run backend; the free values seed the iterative solution."""
        self._solution = None
        if not self._free.size:
            return
        dtype = self._solve_dtype()
        operator = sparse_module(self.backend).csr_matrix(self._reduced.astype(dtype))   # gpu: one device copy
        self.linear_solver.factorize(operator)
        self._solution = array_module(self.backend).asarray(self.values[self._full_ids(self._free)], dtype=dtype)

    def _solve_reduced(self, load) -> None:
        """Keep the load (reactions) and solve the reduced operator for the free values of it (minus the lift)."""
        self._load = load
        if self._free.size:
            rhs = array_module(self.backend).asarray(load[self._free] - self._lift, dtype=self._solve_dtype())
            # pcg stop reference (PHY-1): a transient step's rhs is dominated by C/dt u_old, so stop against the
            # step's initial residual (x0 = the previous step) instead of ||b||
            reference = 'initial_residual' if self._capacity_matrix is not None else 'rhs'
            self.linear_solver.solve(rhs, self._solution, reference)
            self.values[self._full_ids(self._free)] = np.asarray(to_host(self._solution), dtype=np.float64)


class FEMDiffusionSystem(FEMDiffusionSystemCommon):
    """Lagrange system of one scalar unknown on a TriangleMesh. The owner sets conditions and sources, calls assemble(
    tensor, capacity_over_dt) while operator_dirty, then solve(source density); `values` = u in the full-mesh numbering
    (nodes, then n_points + edge for P2): dofs of no active element keep their last value, the PML layer ring gets the
    window field at the clamped position. Integrals, reactions and fluxes of an axisymmetric solve are per radian."""

    def __init__(self, mesh, order=2, linear_solver=None, *, backend='cpu', axisymmetric=False,
                 label=lambda: 'fem diffusion', unknown_name='unknown',
                 anchor_text='Dirichlet or Robin edge, held region or volumetric loss', remedy_text='',
                 require_anchors=True):
        super().__init__(mesh, order, linear_solver, backend, label, unknown_name, anchor_text, remedy_text,
                         require_anchors)
        self.geometry_order = 1 if self.order == 1 else 2   # D6: isoparametric P2 on curved elements
        self.axisymmetric = bool(axisymmetric)
        self.full = lagrange_dofs(mesh, self.order, geometry_order=self.geometry_order)   # state / display numbering
        self.element_map = lagrange_dofs(mesh, 0, geometry_order=self.geometry_order)     # P0 fields
        self.values = np.zeros(self.full.n_dofs)
        self.window_elements = mesh.pml_faces == 0
        self.face_conditions = {face: (EdgeCondition(), 0) for face in FACES}   # face -> (condition, sequence)
        self.edge_conditions = []   # [edges, EdgeCondition, sequence] of object outlines
        self.point_sources = {}     # node -> power (whole ring when axisymmetric)
        self.dofs = None            # DofMap of the active elements (last assembly)
        self._face_edges = {}
        self._ring = None

    def _full_ids(self, ids) -> np.ndarray:
        return self.dofs.global_ids[ids]

    # ---- conditions and sources ----
    def add_edge_condition(self, edges, condition) -> int:
        """Condition on mesh edges (object outlines); returns its handle for set_edge_condition."""
        self.edge_conditions.append([np.unique(np.asarray(edges, dtype=np.int64)), condition, self._next_sequence()])
        if condition.kind != 'neumann':
            self.operator_dirty = True
        return len(self.edge_conditions) - 1

    def set_edge_condition(self, handle, condition) -> None:
        """Replace the condition of an add_edge_condition handle (a value-only change keeps the factor)."""
        entry = self.edge_conditions[handle]
        self._note_change(entry[1], condition)
        entry[1] = condition

    def set_element_values(self, element_mask, value) -> None:
        """Write u = value on every dof of the masked elements (initial / held values)."""
        ids = np.unique(self.full.element_dofs[np.asarray(element_mask, dtype=bool)])
        self.values[ids] = value

    def set_loss(self, element_mask, coefficient, external) -> None:
        """Replace every volumetric loss by a (u - external) on the masked elements (for a physics owning the whole
        loss term: potential's electrolytes)."""
        self.loss_coefficient = np.zeros(self.mesh.n_elements)
        self.loss_drive = np.zeros(self.mesh.n_elements)
        self.add_loss(element_mask, coefficient, external)

    def add_point_source(self, node, power) -> None:
        """Add a point (2D: line; axisymmetric: ring of total) source of `power` at a mesh node."""
        node = int(node)
        if not 0 <= node < self.mesh.n_points:
            raise ValueError(f'{self.label()}: node {node} is not a mesh node')
        self.point_sources[node] = self.point_sources.get(node, 0.0) + float(power)

    def face_edges(self, face) -> np.ndarray:
        """Int64 mesh edges on the window edge of an outer face (the inner side of a PML ring when ringed)."""
        cached = self._face_edges.get(face)
        if cached is None:
            cached = self._face_edges[face] = window_face_edges(self.mesh, face)
        return cached

    def _all_edge_conditions(self):
        """(edges, condition, sequence) of the outer faces, then the outlines."""
        for face, (condition, sequence) in self.face_conditions.items():
            yield self.face_edges(face), condition, sequence
        for edges, condition, sequence in self.edge_conditions:
            yield edges, condition, sequence

    # ---- assembly ----
    def assemble(self, tensor, capacity_over_dt=None) -> None:
        """Assemble on the window elements that conduct (tensor (n_elements, 2, 2) of C), lose or (capacity_over_dt
        per element, transient) store; eliminate the fixed dofs, check every free region's anchor, factorise."""
        mesh = self.mesh
        tensor = np.asarray(tensor, dtype=np.float64)
        if tensor.shape != (mesh.n_elements, 2, 2):
            raise ValueError(f'{self.label()}: coefficient tensor has shape {tensor.shape}, expected '
                             f'({mesh.n_elements}, 2, 2)')
        active = (tensor[:, 0, 0] > 0.0) | (tensor[:, 1, 1] > 0.0) | (self.loss_coefficient > 0.0)
        if capacity_over_dt is not None:
            capacity_over_dt = np.broadcast_to(np.asarray(capacity_over_dt, dtype=np.float64), (mesh.n_elements,))
            active |= capacity_over_dt > 0.0
        active &= self.window_elements
        if not active.any():
            raise ValueError(f'{self.label()}: no window element conducts, loses or stores (all coefficients 0)')
        dofs = lagrange_dofs(mesh, self.order, active, geometry_order=self.geometry_order)
        axisymmetric = self.axisymmetric
        terms = [assemble_stiffness(dofs, tensor, axisymmetric=axisymmetric)]
        storing = active & (self.loss_coefficient > 0.0)
        if self.loss_coefficient.any():
            terms.append(assemble_mass(dofs, self.loss_coefficient, axisymmetric=axisymmetric))
        self._capacity_matrix = None
        if capacity_over_dt is not None:
            self._capacity_matrix = assemble_mass(dofs, capacity_over_dt, axisymmetric=axisymmetric)
            terms.append(self._capacity_matrix)
            storing |= active & (capacity_over_dt > 0.0)
        anchored_dofs = np.zeros(dofs.n_dofs, dtype=bool)   # anchors by topology (floating_region_count)
        anchored_dofs[dofs.element_dofs[dofs.element_position[storing]].ravel()] = True
        inside = dofs.element_position >= 0
        for edges, condition, _ in self._all_edge_conditions():
            if len(edges) and robin_assembled(condition):
                terms.append(assemble_edge_mass(dofs, edges, condition.h, axisymmetric=axisymmetric))
                anchoring = robin_anchoring(edges, condition)
                first, second = side_members(mesh.edge_elements[anchoring], inside)
                anchored_dofs[dofs.edge_dofs(anchoring[first | second])] = True
        matrix = pattern_sum(terms)   # one pattern whatever the coefficients (cuDSS plan reuse)
        fixed_value, _ = self._fixed_values(dofs)
        fixed = np.flatnonzero(~np.isnan(fixed_value))
        free = np.flatnonzero(np.isnan(fixed_value))
        free_rows = matrix[free]   # sliced once for the reduced operator and the lift coupling
        reduced = free_rows[:, free].tocsr()
        if self.require_anchors:
            self._check_anchors(reduced, anchored_dofs[free])
        self.dofs, self._matrix, self._reduced = dofs, matrix, reduced
        self._free, self._fixed = free, fixed
        self._coupling = free_rows[:, fixed].tocsr()
        self._refresh_constant_terms()
        self._factorize()
        self.operator_dirty = False

    def _fixed_values(self, dofs) -> tuple:
        """(value per active dof, NaN when free; sequence of the condition holding it, 0 when free) of the Dirichlet
        edges and held regions; later sequences win shared dofs."""
        constraints = []   # (sequence, dof indices, value)
        for edges, condition, sequence in self._all_edge_conditions():
            if len(edges) and condition.kind == 'dirichlet':
                constraints.append((sequence, dofs.edge_dofs(edges), condition.value))
        for mask, value, sequence in self.held:
            constraints.append((sequence, dofs.edge_dofs(self.mesh.element_edges[mask].ravel()), value))
        fixed_value = np.full(dofs.n_dofs, np.nan)
        owner = np.zeros(dofs.n_dofs, dtype=np.int64)
        for sequence, indices, value in sorted(constraints, key=lambda entry: entry[0]):
            fixed_value[indices] = value
            owner[indices] = sequence
        return fixed_value, owner

    def _refresh_constant_terms(self) -> None:
        """Loss / Robin drives, fixed values, their owners and the lift of the current conditions on the last
        assembly (the fixed dof set and the matrix unchanged)."""
        dofs, axisymmetric = self.dofs, self.axisymmetric
        static_load = np.zeros(dofs.n_dofs)
        if self.loss_coefficient.any():
            static_load += assemble_load(dofs, self.loss_drive, axisymmetric=axisymmetric)
        for edges, condition, _ in self._all_edge_conditions():
            if len(edges) and robin_assembled(condition):
                static_load += assemble_edge_load(dofs, edges, condition.h * condition.external,
                                                  axisymmetric=axisymmetric)
        fixed_value, self._dof_owner = self._fixed_values(dofs)
        fixed = self._fixed
        self._lift = (self._coupling @ fixed_value[fixed]) if fixed.size else np.zeros(self._free.size)
        self.values[dofs.global_ids[fixed]] = fixed_value[fixed]
        self._static_load = static_load
        self._constants_stale = False

    # ---- solve ----
    def solve(self, source_density=None) -> None:
        """Solve with the P0 source density f per element (None: none) on top of the edge, loss and point terms
        (transient: + C/dt u of the current values); writes `values` and extends them into the PML ring."""
        if self.operator_dirty or self.dofs is None:
            raise RuntimeError(f'{self.label()}: assemble before solve')
        if self._constants_stale:
            self._refresh_constant_terms()
        dofs = self.dofs
        axisymmetric = self.axisymmetric
        load = self._static_load.copy()
        if source_density is not None:
            density = np.asarray(source_density, dtype=np.float64)
            if density.any():
                load += assemble_load(dofs, density, axisymmetric=axisymmetric)
        for edges, condition, _ in self._all_edge_conditions():
            if condition.kind == 'neumann' and condition.value != 0.0 and len(edges):
                load += assemble_edge_load(dofs, edges, condition.value, axisymmetric=axisymmetric)
        for node, power in self.point_sources.items():
            load[dofs.node_dofs([node])] += power / (RING_ANGLE if axisymmetric else 1.0)
        if self._capacity_matrix is not None:
            load += self._capacity_matrix @ self.values[dofs.global_ids]
        self._solve_reduced(load)
        self._extend_into_ring()
        self.n_solves += 1

    def _extend_into_ring(self) -> None:
        """PML ring dofs = the field at their position clamped into the window (the tensor edge extension)."""
        if self._ring is None:
            self._ring = ring_dof_locations(self.full)
        ids, location = self._ring
        if ids.size:
            self.values[ids] = evaluate(self.full, self.values, location=location)

    # ---- reactions, fluxes, post-processing ----
    def reaction(self, dof_indices) -> float:
        """Sum of (A u - b) of the last solve over active dofs: the flux leaving those fixed dofs into the domain [c u
        per m; axisymmetric per radian]; sources on them are subtracted."""
        u = self.values[self.dofs.global_ids]
        residual = self._matrix @ u - self._load
        return float(residual[np.asarray(dof_indices, dtype=np.int64)].sum())

    def face_flux(self, face) -> float:
        """inward_flux of an outer face; a Dirichlet face counts only the dofs its condition holds (a corner dof belongs
        to the later condition), so the reactions of all faces add up without double counting."""
        condition, sequence = self.face_conditions[face]
        edges = self.face_edges(face)
        if condition.kind != 'dirichlet' or not len(edges):
            return self.inward_flux(edges, condition)
        face_dofs = self.dofs.edge_dofs(edges)
        return self.reaction(face_dofs[self._dof_owner[face_dofs] == sequence])

    def region_dofs(self, element_mask) -> np.ndarray:
        """Active dofs of the masked elements."""
        return self.dofs.edge_dofs(self.mesh.element_edges[np.asarray(element_mask, dtype=bool)].ravel())

    def inward_flux(self, edges, condition) -> float:
        """Flux entering the domain through mesh edges under `condition` (dirichlet: the reaction of their dofs)."""
        if not len(edges):
            return 0.0
        dofs, axisymmetric = self.dofs, self.axisymmetric
        if condition.kind == 'dirichlet':
            return self.reaction(dofs.edge_dofs(edges))
        if condition.kind == 'neumann':
            return condition.value * float(assemble_edge_load(dofs, edges, 1.0, axisymmetric=axisymmetric).sum())
        u = self.values[dofs.global_ids]
        drive = assemble_edge_load(dofs, edges, condition.h * condition.external, axisymmetric=axisymmetric).sum()
        return float(drive - (assemble_edge_mass(dofs, edges, condition.h, axisymmetric=axisymmetric) @ u).sum())

    def centroid_values(self) -> np.ndarray:
        """Float64 u at every element's reference centroid (D12: the per-element state)."""
        basis = lagrange_basis(self.order, np.array([[1.0 / 3.0, 1.0 / 3.0]]))[0][0]
        return self.values[self.full.element_dofs] @ basis

    def element_integrals(self, tensor=None) -> dict:
        """Per mesh element (0 off the active ones) of the last solution: 'measure' (area; x r axisymmetric), 'value'
        (int u), 'gradient' (n, 2) (int grad u) and, with tensor, 'dissipation' (int grad u . C grad u)."""
        mesh, dofs = self.mesh, self.dofs
        count = mesh.n_elements
        result = {'measure': np.zeros(count), 'value': np.zeros(count), 'gradient': np.zeros((count, 2))}
        if tensor is not None:
            result['dissipation'] = np.zeros(count)
        if dofs is None:
            return result
        u = self.values[dofs.global_ids]
        base_degree = 2 * self.order + int(self.axisymmetric)
        if tensor is None:   # linear in u: cached per-element integrals of the full map (no geometry per call)
            elements = dofs.elements
            local = u[dofs.element_dofs]
            positions = self.full.element_position[elements]
            measure, value_integrals = element_basis_integrals(self.full, base_degree, self.axisymmetric)
            gradient_integrals = element_gradient_integrals(self.full, base_degree, self.axisymmetric)
            result['measure'][elements] = measure[positions]
            result['value'][elements] = np.einsum('mn,mn->m', value_integrals[positions], local)
            result['gradient'][elements] = np.einsum('mnc,mn->mc', gradient_integrals[positions], local)
            return result
        for positions, rule in _element_groups(dofs, base_degree):   # straight / curved chunks (CURVED_DEGREE)
            elements = dofs.elements[positions]
            geometry = element_geometry(mesh, elements, rule.points, self.geometry_order)
            basis, gradients = _lagrange_at(dofs, geometry, rule)
            local = u[dofs.element_dofs[positions]]
            weights = _weights(geometry, rule, self.axisymmetric)
            point_gradients = np.einsum('mqnc,mn->mqc', gradients, local)
            result['measure'][elements] = weights.sum(axis=1)
            result['value'][elements] = np.einsum('qn,mn,mq->m', basis, local, weights)
            result['gradient'][elements] = np.einsum('mqc,mq->mc', point_gradients, weights)
            result['dissipation'][elements] = np.einsum('mqc,mcd,mqd,mq->m', point_gradients, tensor[elements],
                                                        point_gradients, weights)
        return result


def _same_operator(previous, condition) -> bool:
    """True when replacing EdgeCondition `previous` by `condition` leaves the matrix unchanged (same kind; Robin: same
    h): only right-hand-side terms move."""
    return previous.kind == condition.kind and (condition.kind != 'robin' or previous.h is condition.h
                                                 or np.array_equal(previous.h, condition.h))


def robin_assembled(condition) -> bool:
    """A Robin condition with some h > 0 (h a number or per mesh entity): its mass and drive are assembled (an array
    on all its entities, zeros included, so the pattern stays while h changes)."""
    return condition.kind == 'robin' and bool(np.any(np.asarray(condition.h) > 0.0))


def robin_anchoring(entities, condition) -> np.ndarray:
    """The entities of an assembled Robin condition that anchor (h > 0 there)."""
    h = np.asarray(condition.h)
    return entities if h.ndim == 0 else entities[h[entities] > 0.0]


def window_face_edges(mesh, face) -> np.ndarray:
    """Int64 mesh edges of the window edge on an outer face: window-subset boundary edges with both ends on its line
    (the inner side of a PML ring when ringed)."""
    boundary = np.flatnonzero(one_sided(mesh.edge_elements, mesh.pml_faces == 0))
    x0, y0, x1, y1 = mesh.window
    axis = 0 if face[0] == 'x' else 1
    line = {'x_lo': x0, 'x_hi': x1, 'y_lo': y0, 'y_hi': y1}[face]
    tolerance = MERGE_TOLERANCE * max(x1 - x0, y1 - y0)
    ends = mesh.points[mesh.edges[boundary]][:, :, axis]
    return boundary[np.all(np.abs(ends - line) <= tolerance, axis=1)].astype(np.int64)


def ring_dof_locations(full) -> tuple:
    """(int64 ids, sampling Location) of the dofs of a full-mesh Lagrange map outside the window and their positions
    clamped just inside it ((empty, None) without a PML ring): the edge extension of window fields into the ring."""
    mesh = full.mesh
    if (mesh.pml_faces == 0).all():
        return np.zeros(0, dtype=np.int64), None
    points = full.dof_points()
    x0, y0, x1, y1 = mesh.window
    inset = MERGE_TOLERANCE * max(x1 - x0, y1 - y0)
    clamped = np.column_stack([np.clip(points[:, 0], x0 + inset, x1 - inset),
                               np.clip(points[:, 1], y0 + inset, y1 - inset)])
    ids = np.flatnonzero(np.any(np.abs(clamped - points) > 2.0 * inset, axis=1))
    return ids, locator(mesh, full.geometry_order).locate(clamped[ids])


def floating_region_count(reduced, anchored_rows=None) -> int:
    """Connected regions of a reduced (free-dof) CSR whose rows all sum to ~0 (ANCHOR_RTOL of the diagonal): u + const
    solves them too, so the system is singular there. anchored_rows: bool per row the caller knows anchored (loss,
    capacity or Robin terms, which can sit below ANCHOR_RTOL of the stiffness on micrometre elements)."""
    from scipy.sparse.csgraph import connected_components
    if reduced.shape[0] == 0:
        return 0
    diagonal = reduced.diagonal()
    row_sums = np.asarray(reduced.sum(axis=1)).ravel()
    known = np.zeros(reduced.shape[0], dtype=bool) if anchored_rows is None else np.asarray(anchored_rows, dtype=bool)
    anchored_rows = known | (row_sums > ANCHOR_RTOL * np.abs(diagonal))
    pattern = reduced.copy()
    pattern.eliminate_zeros()
    count, labels = connected_components(pattern, directed=False)
    anchored = np.zeros(count, dtype=bool)
    anchored[labels[anchored_rows]] = True
    return int(count - anchored.sum())


def coefficient_tensor(c_xx, c_yy, c_xy=None) -> np.ndarray:
    """Float64 (n, 2, 2) symmetric tensors of per-element components (host or backend arrays)."""
    c_xx = np.asarray(to_host(c_xx), dtype=np.float64)
    tensor = np.zeros(c_xx.shape + (2, 2))
    tensor[:, 0, 0] = c_xx
    tensor[:, 1, 1] = np.asarray(to_host(c_yy), dtype=np.float64)
    if c_xy is not None:
        tensor[:, 0, 1] = tensor[:, 1, 0] = np.asarray(to_host(c_xy), dtype=np.float64)
    return tensor


def element_means(integrals) -> tuple:
    """(mean value (n,), mean gradient (n, 2)) per element from element_integrals; 0 where the measure is 0."""
    measure = integrals['measure']
    safe = np.where(measure > 0.0, measure, 1.0)
    inside = measure > 0.0
    return (np.where(inside, integrals['value'] / safe, 0.0),
            np.where(inside[:, None], integrals['gradient'] / safe[:, None], 0.0))


class FEMDiffusionPhysicsCommon(PhysicsSolver):
    """Dimension-free part of the FEM diffusion physics wrappers (FEMDiffusionPhysicsBase on triangles,
    diffusion3d.FEMDiffusionPhysicsBase3D on tetrahedra): settings / records, outer-face BCSides on `_faces`, held /
    initial / loss terms on element masks and the per-element field helpers; field = reference_value + the unknown."""

    coordinate_grids = ('cartesian',)
    conformal_locator = None     # moot on element meshes: the elements follow the drawn outlines / interfaces
    reference_value = 0.0
    slot_names = ()              # inject_source names (per element, like every state field)
    _probe_field = 'u'
    _faces = FACES               # outer faces of set_face_bc (3D: FACES_3D)

    # ---- settings and records ----
    @property
    def azimuthal_order(self) -> int:
        return 0

    @property
    def order(self) -> int:
        """Lagrange order of the unknown (1 linear, 2 quadratic)."""
        return self.system.order

    @property
    def n_solves(self) -> int:
        return self.system.n_solves

    @property
    def last_converged(self) -> bool:
        return self.system.linear_solver.last_converged

    @property
    def last_warning(self):
        return self.system.linear_solver.last_warning

    @property
    def n_unconverged(self) -> int:
        return self.system.linear_solver.n_unconverged

    def set_conformal_boundaries(self, locator) -> None:
        """solver_defaults.conformal_boundaries: nothing to do (the mesh conforms to the drawn outlines)."""

    # ---- boundary conditions ----
    def _condition(self, bc) -> EdgeCondition:
        """EdgeCondition (3D: the same FaceCondition) of a BCSide in the unknown's units (field - reference_value;
        conductance_factor unused)."""
        if not isinstance(bc, BCSide):
            raise TypeError(f'bc must be a BCSide, got {type(bc).__name__}')
        if bc.kind == 'dirichlet':
            return EdgeCondition('dirichlet', value=bc.value - self.reference_value)
        if bc.kind == 'robin':
            return EdgeCondition('robin', h=bc.h, external=bc.external - self.reference_value)
        return EdgeCondition('neumann', value=bc.value)

    def set_face_bc(self, face: str, bc: BCSide, keep_sequence=False) -> None:
        """Condition of one outer face (the window edge / faces on that side)."""
        if face not in self._faces:
            raise ValueError(f'unknown face {face!r}; expected one of {self._faces}')
        self.system.set_face_condition(face, self._condition(bc), keep_sequence=keep_sequence)
        self.bcs[face] = bc
        self._rhs_changed()

    def set_face_value(self, face: str, bc: BCSide) -> None:
        """set_face_bc for a value of time t (a value-only change keeps the factor and the face's place in the
        order)."""
        self.set_face_bc(face, bc, keep_sequence=True)

    # ---- held regions, losses, sources ----
    def element_mask(self, mask) -> np.ndarray:
        """Host bool per element of a run-backend element mask."""
        mask = np.asarray(to_host(mask), dtype=bool)
        if mask.shape != (self.system.mesh.n_elements,):
            raise ValueError(f'element mask has shape {mask.shape}, expected ({self.system.mesh.n_elements},)')
        return mask

    def add_fixed_value(self, mask, value) -> None:
        """Hold the field at `value` (a number) on every dof of the masked elements."""
        self.system.add_held(self.element_mask(mask), float(value) - self.reference_value)
        self._values_changed()

    def set_initial_value(self, mask, value) -> None:
        """Write the field `value` (a number) on the dofs of the masked elements."""
        self.system.set_element_values(self.element_mask(mask), float(value) - self.reference_value)
        self._values_changed()

    def add_volumetric_loss(self, mask, coefficient, external) -> None:
        """Add a loss coefficient (field - external) on the masked elements (numbers or per-element arrays)."""
        coefficient = np.asarray(to_host(coefficient), dtype=np.float64)
        external = np.asarray(to_host(external), dtype=np.float64) - self.reference_value
        self.system.add_loss(self.element_mask(mask), coefficient, external)
        self._rhs_changed()

    def _add_masked(self, target, mask, value) -> None:
        """target += value on the masked elements (value a number or per-element run-backend array)."""
        xp = self.xp
        target += xp.where(mask, xp.asarray(value, dtype=DTYPE), DTYPE(0.0)).astype(DTYPE, copy=False)

    # ---- fields ----
    def field_dofs(self, name):
        """fem DofMap of state field or source slot `name` (the transfer contract): every one is per element (P0)."""
        if name not in self.state() and name not in self.slot_names:
            raise ValueError(f"{type(self).__name__} has no state field or source slot '{name}'")
        return self.system.element_map

    def nodal_field(self, offset=0.0) -> tuple:
        """(full-mesh Lagrange DofMap, host float64 coefficients + offset) of the unknown."""
        return self.system.full, self.system.values + offset

    def element_field(self, array) -> tuple:
        """(P0 DofMap, host float64 per-element values) of a per-element array."""
        return self.system.element_map, np.asarray(to_host(array), dtype=np.float64)

    # ---- hooks ----
    def _rhs_changed(self) -> None:
        """Sources or boundary values changed (potential re-solves when it sees it)."""

    def _values_changed(self) -> None:
        """Field values were written directly (held / initial values)."""


class FEMDiffusionPhysicsBase(FEMDiffusionPhysicsCommon):
    """Shared wrapper of a FEMDiffusionSystem physics (heat, electric_currents / electrostatics on a 'triangles' mesh):
    FEMDiffusionPhysicsCommon plus outline edge conditions, point sources at nodes, node probes and the (r, z) ring."""

    def _init_fem(self, domain, order, linear_solver, axisymmetric, label, unknown_name, anchor_text, remedy_text,
                  require_anchors=True) -> None:
        if getattr(domain, 'grid_kind', 'tensor') != 'triangles':
            raise ValueError(f'{label()}: the finite-element solver needs a triangle mesh (fem.domain.TriangleDomain)')
        self.domain = domain
        self.xp = domain.xp
        self.system = FEMDiffusionSystem(domain.mesh, order, linear_solver, backend=domain.backend.name,
                                         axisymmetric=axisymmetric, label=label, unknown_name=unknown_name,
                                         anchor_text=anchor_text, remedy_text=remedy_text,
                                         require_anchors=require_anchors)
        self.bcs = {face: BCSide('neumann') for face in FACES}
        self.probes = []   # (FieldProbe, node)

    @property
    def axisymmetric(self) -> bool:
        return self.system.axisymmetric

    @property
    def ring_angle(self) -> float:
        return RING_ANGLE

    def add_edge_condition(self, edges, bc: BCSide) -> int:
        """Condition on mesh edges (object outlines: BuildContext.boundary_edges); returns its handle."""
        handle = self.system.add_edge_condition(edges, self._condition(bc))
        self._rhs_changed()
        return handle

    def set_edge_condition(self, handle, bc: BCSide) -> None:
        self.system.set_edge_condition(handle, self._condition(bc))
        self._rhs_changed()

    def add_point_source_node(self, node: int, power: float) -> None:
        """Point source at a mesh node (2D: line source per m; axisymmetric: whole-ring total)."""
        self.system.add_point_source(node, power)
        self._rhs_changed()

    def add_probe_node(self, node: int) -> FieldProbe:
        """Record the field at a mesh node (a point object's node) whenever the physics records probes."""
        node = int(node)
        if not 0 <= node < self.system.mesh.n_points:
            raise ValueError(f'probe node {node} is not a mesh node')
        probe = FieldProbe(node, 0, self._probe_field)
        self.probes.append((probe, node))
        return probe

    def _record_probes(self, time, replace_same_time=False) -> None:
        for probe, node in self.probes:
            probe.record_value(np.float32(self.reference_value + self.system.values[node]), time, replace_same_time)


def lagrange_entries_per_dof(mesh, order) -> float:
    """Mean stored entries per row of a scalar Lagrange operator from a Triangle- / TetrahedronMeshEstimate's counts:
    each dof + 2 x the dof pairs sharing an element (P1: V + 2E; P2: V + 7E + 12T on triangles, V + 7E + 12F + 6T on
    tetrahedra, over V + E dofs); fixed rows are not taken out (a slight upper bound)."""
    nodes, edges, elements = int(mesh.n_nodes), int(mesh.n_edges), int(mesh.n_elements)
    if int(order) == 1:
        return (nodes + 2 * edges) / max(nodes, 1)
    if getattr(mesh, 'kind', '') == 'tetrahedra':   # Euler V - E + F - T = 1 when the estimate has no faces
        faces = int(getattr(mesh, 'n_faces', 0)) or edges + elements - nodes + 1
        return (nodes + 7 * edges + 12 * faces + 6 * elements) / max(nodes + edges, 1)
    return (nodes + 7 * edges + 12 * elements) / max(nodes + edges, 1)


def fem_diffusion_estimate(mesh, order, element_fields, nodal_fields=1, direct=False, multigrid=False, cudss=None,
                           refactorizes=False, active_fraction=1.0) -> MemoryEstimate:
    """FEMDiffusionSystem + its physics on a TriangleMeshEstimate (fem_system_estimate; SuperLU fill FEM_FILL_FACTOR on
    the 2D heuristic; direct: direct_factor_item); cudss = the 'linear' cuDSS back end (cudss_back_end) or None."""
    n_dofs = int(mesh.n_nodes) + (int(mesh.n_edges) if order == 2 else 0)
    n_active = active_dofs(n_dofs, active_fraction)
    direct_factor = direct_factor_item(direct, FEM_FILL_FACTOR * host_real_lu_bytes(n_active),
                                       'host SuperLU factor L + U (float64, FEM)', 'FEM')
    return fem_system_estimate(n_dofs, lagrange_entries_per_dof(mesh, order), 6 if order == 2 else 3,
                               int(mesh.n_elements),
                               element_fields, nodal_fields, 2, direct_factor=direct_factor, multigrid=multigrid,
                               cudss=cudss, refactorizes=refactorizes, active_fraction=active_fraction,
                               host_direct=host_direct_name(direct))


def host_direct_name(direct):
    """`direct` of an estimator (definition_helpers.linear_solver_direct: the resolved non-cuDSS sparse_direct back
    end, truthy; True = SuperLU) when it names a MUMPS / PARDISO back end (host_direct.HOST_DIRECT_BACK_ENDS), else
    None."""
    from ..solvers.host_direct import HOST_DIRECT_BACK_ENDS   # numpy only at module level
    return direct if isinstance(direct, str) and direct in HOST_DIRECT_BACK_ENDS else None


def direct_factor_item(direct, superlu_bytes, superlu_label, detail):
    """(label, bytes) host item of a DiffusionDirectSolver's factor, None when `direct` is falsy: the SuperLU heuristic
    superlu_bytes under superlu_label, or for a MUMPS / PARDISO back end (host_direct_name) that heuristic x
    host_direct.superlu_bytes_scale (real Cholesky / LDL^T, float32 values when mixed; uncalibrated)."""
    if not direct:
        return None
    back_end = host_direct_name(direct)
    if back_end is None:
        return superlu_label, int(superlu_bytes)
    from ..solvers.host_direct import factor_label, host_direct_precision, superlu_bytes_scale
    dtype = 'float32' if host_direct_precision(back_end) == 'mixed' else 'float64'
    return (f'host {factor_label(back_end)} factor ({dtype}, {detail}, SuperLU heuristic scaled, uncalibrated)',
            int(superlu_bytes * superlu_bytes_scale(back_end, symmetric=True, complex_values=False)))


def active_dofs(n_dofs, active_fraction) -> int:
    """Unknowns of a FEM diffusion system: the dofs of the active elements' share (at least 1)."""
    return max(int(round(int(n_dofs) * min(max(float(active_fraction), 0.0), 1.0))), 1)


def painted_fraction(mesh, material_names) -> float:
    """Share of an element-mesh estimate's elements the named spec materials paint (its material_elements; 1 without
    them): the active elements of a FEM diffusion system, an upper bound."""
    painted = getattr(mesh, 'material_elements', None)
    if not painted or not int(mesh.n_elements):
        return 1.0
    return min(sum(painted.get(name, 0.0) for name in material_names) / int(mesh.n_elements), 1.0)


def fem_system_estimate(n_dofs, entries_per_dof, local_dofs, n_elements, element_fields, nodal_fields, dimensions,
                        direct_factor=None, multigrid=False, cudss=None, refactorizes=False,
                        active_fraction=1.0, host_direct=None) -> MemoryEstimate:
    """FEMDiffusionSystem[3D] and physics: float32 element state, host nodal arrays, the operator and linear solver on
    the active dofs (pcg keeps its run-backend copy, SuperLU / cuDSS / MUMPS / PARDISO take it as a temporary);
    refactorizes: per change; host_direct: the MUMPS / PARDISO back end of direct_factor (its own arrays beside the
    library factor, host_direct.solver_host_bytes) or None."""
    n_active = active_dofs(n_dofs, active_fraction)
    entries = int(round(float(entries_per_dof) * n_active))
    csr_bytes = FLOAT64_CSR_ENTRY_BYTES * entries + INT32_BYTES * (n_active + 1)
    items = [(f'FEM element fields ({element_fields} x float32)', element_fields * FLOAT32_BYTES * int(n_elements))]
    host_items = [('FEM operator (host CSR, full + reduced)', 2 * csr_bytes),
                  (f'FEM nodal values and loads ({nodal_fields + 4} x float64)', (nodal_fields + 4) * FLOAT64_BYTES
                   * int(n_dofs)),
                  ('FEM assembly COO (transient)', 3 * FLOAT64_BYTES * local_dofs * local_dofs * int(n_elements))]
    if cudss:   # FEMDiffusionSystem._factorize: the device CSR is factorize_device_csr's input only
        items.append(('FEM solution vector (run backend)', FLOAT64_BYTES * n_active))
        own = MemoryEstimate.from_items(items, host_items=host_items)
        return combine((own, cudss_diffusion_estimate(n_active, entries, dimensions, cudss, input_copy=True,
                                                      refactorizes=refactorizes)))
    build_extra, transient_items = 0, []
    if direct_factor is not None:   # DiffusionDirectSolver: the run-backend copy goes to the host and is dropped
        items.append(('FEM solution vector (run backend)', FLOAT64_BYTES * n_active))
        host_items.append(direct_factor)
        if host_direct is not None:
            from ..solvers.host_direct import host_direct_precision, solver_host_bytes
            host_items.append(('host direct solver copies (equilibrated operator, row scale; mixed: GMRES vectors)',
                               solver_host_bytes(n_active, entries, complex_values=False, symmetric=True,
                                                 mixed=host_direct_precision(host_direct) == 'mixed')))
        build_extra = refactor_extra = csr_bytes
    else:   # pcg keeps the run-backend operator; a re-assembly makes the new copy before the old one goes
        items.append(('FEM operator and CG vectors (run backend)', csr_bytes + 4 * FLOAT64_BYTES * n_active))
        refactor_extra = csr_bytes
        if multigrid:
            from ..solvers import multigrid as amg
            items.append(('multigrid hierarchy (operators, prolongations, vectors)',
                          amg.hierarchy_bytes(n_active, FLOAT64_BYTES, entries_per_dof)))
            build_extra = amg.setup_extra_bytes(n_active, FLOAT64_BYTES, entries_per_dof)
            refactor_extra += build_extra
    if refactorizes:
        transient_items.append(('refactorisation', 'FEM operator copy and solver setup (run backend)', refactor_extra))
    return MemoryEstimate.from_items(items, build_extra=build_extra, host_items=host_items,
                                     transient_items=transient_items)


def cudss_back_end(solvers):
    """'cudss' / 'cudss_mixed' when the resolved 'linear' solver ({role: ResolvedSolver} of MemoryContext) is
    sparse_direct on a cuDSS back end, else None."""
    from ..solvers.sparse_direct import CUDSS_BACK_ENDS   # numpy only at module level
    resolved = (solvers or {}).get('linear')
    if resolved is None or resolved.solver_type.name != 'sparse_direct':
        return None
    back_end = resolved.values.get('back_end')
    return back_end if back_end in CUDSS_BACK_ENDS else None


def cudss_diffusion_estimate(n_unknowns, entries, dimensions, back_end='cudss', input_copy=True,
                             refactorizes=False) -> MemoryEstimate:
    """DiffusionDirectSolver on cuDSS (real Cholesky of a device CSR, factorize_device_csr): pool copies, index maps and
    vectors (kept), the plan's peak (build extra; input_copy: the device CSR is a caller temporary), solve /
    refactorisation peaks (transient), the factor outside the pool (dimensions 2 / 3)."""
    from ..solvers.cudss_factor import (factor_device_bytes, index_map_bytes, operator_copy_bytes, refresh_peak_bytes,
                                        solve_peak_bytes)
    n_unknowns, entries = int(n_unknowns), int(entries)
    mixed = back_end == 'cudss_mixed'
    lower = (entries + n_unknowns) // 2
    maps = index_map_bytes(n_unknowns, entries)
    # the device plan's CSR copies (double, + the float32 one when mixed) share the maps' lower indices / indptr
    values = operator_copy_bytes(n_unknowns, entries, FLOAT64_BYTES, symmetric=True, mixed=mixed) - (
        (2 if mixed else 1) * INT32_BYTES * (lower + n_unknowns + 1))
    rhs_buffer = (FLOAT32_BYTES if mixed else FLOAT64_BYTES) * n_unknowns   # the factor dtype
    items = (('cuDSS index maps (rows, columns, lower take, diagonal slots, transpose map)', maps),
             ('cuDSS lower-triangle values, diagonal, rhs buffer and row scale',
              values + rhs_buffer + FLOAT64_BYTES * n_unknowns))
    # refresh_peak_bytes counts the uploaded values: a temporary input CSR adds its index arrays, a kept one (the
    # DiffusionSystem's A_solve) is no new allocation
    input_bytes = INT32_BYTES * (entries + n_unknowns + 1) if input_copy else -FLOAT64_BYTES * entries
    refresh = refresh_peak_bytes(entries, FLOAT64_BYTES, symmetry_check=True) + input_bytes
    # first plan: the peak is the transpose map's int64 keys; the kept transpose map, lower values (+ their float32
    # copy), diagonal and rhs buffer come after it
    plan_peak = (refresh - 4 * entries - FLOAT64_BYTES * (lower + n_unknowns) - rhs_buffer
                 - (FLOAT32_BYTES * lower if mixed else 0))
    transient_items = [('linear solve', 'cuDSS solve vectors (mixed: the GMRES basis)',
                        solve_peak_bytes(n_unknowns, FLOAT64_BYTES, mixed))]
    if refactorizes:   # a refresh, or its probe solve on the uploaded / equilibrated values, whichever is larger
        transient_items.append(('refactorisation', 'cuDSS refactorisation (input CSR, refresh or probe peak)',
                                refresh_peak_bytes(entries, FLOAT64_BYTES, symmetry_check=True, n_unknowns=n_unknowns,
                                                   mixed=mixed) + input_bytes))
    permanent, extra = factor_device_bytes(n_unknowns, entries, FLOAT64_BYTES, dimensions, mixed)
    return MemoryEstimate.from_items(items, build_extra=plan_peak, transient_items=transient_items,
                                     external_items=(('cuDSS factor (outside the cupy pool)', permanent),),
                                     external_extra=extra)


def fem_time_system(mesh, order) -> tuple:
    """(unknowns, CSR entries per row) of the Lagrange system on a TriangleMeshEstimate or TetrahedronMeshEstimate,
    for core.time_estimate.diffusion_time_estimate(entries_per_row=) (the FEM assembly priced as the finite-volume one)."""
    n_dofs = int(mesh.n_nodes) + (int(mesh.n_edges) if order == 2 else 0)
    return n_dofs, lagrange_entries_per_dof(mesh, order)


_LOCAL_DOFS = {('triangles', 1): 3, ('triangles', 2): 6, ('tetrahedra', 1): 4, ('tetrahedra', 2): 10}
_SOLVE_ELEMENT_BYTES = 32         # per element: source transfers to float64, their sum, T upload and T0 add
_SOLVE_LOCAL_BYTES = 64           # per element dof: P0 load (cached integrals x source, bincount) and centroid gather
_SOLVE_UNKNOWN_BYTES = 48         # per unknown: load copy, rhs gather - lift, solution scatter
_SOLVE_NUMPY_CALLS = 60


def fem_solve_host_ops(mesh, order, transient) -> tuple:
    """Ops of the host work around one FEMDiffusionSystem[3D] solve (heat recipe on triangles / tetrahedra): source
    transfers and the P0 load on cached element integrals, rhs / solution transfers and scatters, the capacity product
    (transient) and T at the centroids. The heat flux is computed on read, so no per-solve item."""
    from ..core.time_estimate import HostStream, PythonCalls, Transfer   # lazy: estimates are CPU work of the model
    unknowns, entries = fem_time_system(mesh, order)
    elements = float(mesh.n_elements)
    local = _LOCAL_DOFS[('tetrahedra' if getattr(mesh, 'kind', '') == 'tetrahedra' else 'triangles', int(order))]
    stream = elements * (_SOLVE_ELEMENT_BYTES + _SOLVE_LOCAL_BYTES * local) + _SOLVE_UNKNOWN_BYTES * unknowns
    if transient:   # C/dt u_old: one host CSR product (8 B value + 4 B index per entry, vector read and written)
        stream += unknowns * (12.0 * entries + 16.0)
    return (HostStream(stream), Transfer(12.0 * elements + 16.0 * unknowns), PythonCalls(_SOLVE_NUMPY_CALLS))
