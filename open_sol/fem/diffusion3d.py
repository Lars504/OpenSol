"""FEM scalar diffusion on tetrahedron meshes (the 2D fem.diffusion analogue, igor3d_plan T4b): -div(C grad u) + a u
(+ c du/dt) = f by Lagrange P1 / P2 on the active window elements (full 3 x 3 tensor C). FEMDiffusionSystem3D: host
assembly (fem.assembly3d), Dirichlet / Neumann / Robin face conditions (outer faces on the window, object interface
faces), held element regions and fixed dofs, P0, point and gradient (extra) loads, implicit-Euler capacity; the linear
solve by a solver of the PCGSolver protocol on the run backend. FEMDiffusionPhysicsBase3D: the shared physics wrapper
(heat 3D, magnetostatics_3d). fem_diffusion_estimate_3d: their memory estimate."""
# Numbering: the full-mesh map throughout (nodes, then n_points + edge for P2); the assembled matrices keep it, the
# solve uses the free dofs of the active elements (window elements with conduction, loss or capacity). Dofs of no
# active element keep their value; PML shell dofs get the field at their position clamped into the window.
import numpy as np

from ..core.backend import array_module, sparse_module, to_host
from ..core.constants import FACES_3D
from ..core.fv_diffusion import BCSide
from ..core.memory_estimate import MemoryEstimate, host_real_lu_bytes
from ..core.monitors import FieldProbe
from .assembly3d import (assemble_face_load_3d, assemble_face_mass_3d, assemble_lagrange_load_3d,
                         assemble_lagrange_mass_3d, assemble_lagrange_stiffness_3d, assemble_point_load_3d,
                         element_integrals_3d)
from .assembly import pattern_sum
from .diffusion import (EdgeCondition as FaceCondition,   # FaceCondition: the same kinds on faces
                        FEMDiffusionPhysicsCommon, _same_operator, active_dofs, fem_system_estimate,
                        floating_region_count, lagrange_entries_per_dof, robin_anchoring, robin_assembled)
from .dofs3d import lagrange_dofs_3d
from .mesh import side_members
from .mesh3d import PLANE_TOLERANCE, window_faces
from .sampling3d import PointLocator3D, centroid_values_3d, evaluate_lagrange_3d

FEM_FILL_FACTOR_3D = {1: 2.0, 2: 3.0}      # SuperLU fill over the 7-point 3D heuristic at equal unknowns (uncalibrated)


class FEMDiffusionSystem3D:
    """Lagrange system of one scalar unknown on a TetrahedronMesh. The owner sets conditions and sources, calls
    assemble(tensor, capacity_over_dt) while operator_dirty, then solve(source density); `values` = u in the full-mesh
    numbering (module header)."""

    def __init__(self, mesh, order=2, linear_solver=None, *, backend='cpu', label=lambda: 'fem diffusion',
                 unknown_name='unknown', anchor_text='Dirichlet or Robin face, held region or volumetric loss',
                 remedy_text='', require_anchors=True):
        from ..solvers.sparse_direct import DiffusionDirectSolver, SparseDirectSolver
        if order not in (1, 2):
            raise ValueError(f'{label()}: element order must be 1 or 2, got {order!r}')
        if linear_solver is None:   # scripts: the tetrahedra Automatic (host SuperLU)
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
        self.full = lagrange_dofs_3d(mesh, self.order)        # state / display numbering
        self.element_map = lagrange_dofs_3d(mesh, 0)          # P0 fields
        self.values = np.zeros(self.full.n_dofs)
        self.window_elements = mesh.window_elements()
        self.face_conditions = {face: (FaceCondition(), 0) for face in FACES_3D}   # face -> (condition, sequence)
        self.surface_conditions = []   # [faces, FaceCondition, sequence] of object interfaces
        self.held = []                 # (element mask, value, sequence): every active dof of the elements fixed
        self.fixed_dofs = []           # [dof ids, values (per dof), sequence] of add_fixed_dofs (anchors, potentials)
        self.loss_coefficient = np.zeros(mesh.n_elements)   # a per element
        self.loss_drive = np.zeros(mesh.n_elements)         # a u_ext per element
        self.point_sources = []        # (element, reference (3,), power)
        self.active = None             # bool per element of the last assembly
        self._active_dofs = None       # bool per full dof of the last assembly
        self.n_solves = 0
        self.operator_dirty = True
        self._constants_stale = False  # a Dirichlet value / Robin external changed: new lift and loads, same factor
        self._sequence = 0             # later conditions win shared dofs
        self._window_faces = {}
        self._shell = None
        self._matrix = self._reduced = self._capacity_matrix = None
        self._free = self._fixed = self._dof_owner = None
        self._coupling = None          # reduced rows x fixed columns of the last assembly (the lift)
        self._lift = self._static_load = self._load = self._solution = None

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
        """Condition of an outer face (the window faces on that side); a value-only change keeps the factor;
        keep_sequence (a value of time t): a set face keeps its place in the order, so shared edge dofs stay with their
        build owner (else it becomes the latest condition)."""
        previous, sequence = self.face_conditions[face]
        if not (keep_sequence and sequence):
            sequence = self._next_sequence()   # dof owners are re-read with the constants
        self.face_conditions[face] = (condition, sequence)
        self._note_change(previous, condition)

    def add_surface_condition(self, faces, condition) -> int:
        """Condition on mesh faces (object interfaces); returns its handle for set_surface_condition."""
        self.surface_conditions.append([np.unique(np.asarray(faces, dtype=np.int64)), condition,
                                        self._next_sequence()])
        if condition.kind != 'neumann':
            self.operator_dirty = True
        return len(self.surface_conditions) - 1

    def set_surface_condition(self, handle, condition) -> None:
        """Replace the condition of an add_surface_condition handle (a value-only change keeps the factor)."""
        entry = self.surface_conditions[handle]
        self._note_change(entry[1], condition)
        entry[1] = condition

    def add_held(self, element_mask, value) -> None:
        """Fix u = value on every dof of the masked elements (held cells: a terminal / reservoir)."""
        mask = np.asarray(element_mask, dtype=bool).copy()
        self.held.append((mask, float(value), self._next_sequence()))
        self.set_element_values(mask, value)
        self.operator_dirty = True

    def add_fixed_dofs(self, dof_ids, values) -> int:
        """Fix u = values (a number or one per dof) on explicit full-mesh dofs (an anchor, a boundary potential);
        returns its handle for set_fixed_dofs. Dofs of no active element are skipped at assembly."""
        self.fixed_dofs.append([np.zeros(0, dtype=np.int64), np.zeros(0), 0])
        handle = len(self.fixed_dofs) - 1
        self.set_fixed_dofs(handle, dof_ids, values)
        return handle

    def set_fixed_dofs(self, handle, dof_ids, values) -> None:
        """Replace the dofs / values of an add_fixed_dofs handle (empty ids: none); it takes a new sequence, so it wins
        dofs shared with every earlier condition."""
        ids = np.asarray(dof_ids, dtype=np.int64).reshape(-1)
        values = np.broadcast_to(np.asarray(values, dtype=np.float64), ids.shape).copy()
        self.fixed_dofs[handle] = [ids, values, self._next_sequence()]
        self.values[ids] = values
        self.operator_dirty = True

    def set_element_values(self, element_mask, value) -> None:
        """Write u = value on every dof of the masked elements (initial / held values)."""
        self.values[self.full.element_dofs_of(np.asarray(element_mask, dtype=bool))] = value

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

    def add_point_source(self, element, reference, power) -> None:
        """Add a point source of total `power` at a located point (element >= 0, reference coordinates (3,)); dropped
        when that element is not active."""
        if int(element) < 0:
            raise ValueError(f'{self.label()}: the point source lies outside the mesh')
        self.point_sources.append((int(element), np.asarray(reference, dtype=np.float64).reshape(3), float(power)))

    def face_faces(self, face) -> np.ndarray:
        """Int64 mesh faces of the window on an outer face (fem.mesh3d.window_faces, cached)."""
        cached = self._window_faces.get(face)
        if cached is None:
            cached = self._window_faces[face] = window_faces(self.mesh, face)
        return cached

    def _all_face_conditions(self):
        """(faces, condition, sequence) of the outer faces, then the interfaces."""
        for face, (condition, sequence) in self.face_conditions.items():
            yield self.face_faces(face), condition, sequence
        for faces, condition, sequence in self.surface_conditions:
            yield faces, condition, sequence

    def _active_faces(self, faces) -> np.ndarray:
        """The faces with an active element on either side (their nodes are active dofs)."""
        first, second = side_members(self.mesh.face_elements[faces], self.active)
        return faces[first | second]

    # ---- assembly ----
    def assemble(self, tensor, capacity_over_dt=None) -> None:
        """Assemble on the window elements that conduct (tensor (n_elements, 3, 3) of C), lose or (capacity_over_dt
        per element, transient) store; eliminate the fixed dofs, check every free region's anchor, factorise."""
        mesh, full = self.mesh, self.full
        tensor = np.asarray(tensor, dtype=np.float64)
        if tensor.shape != (mesh.n_elements, 3, 3):
            raise ValueError(f'{self.label()}: coefficient tensor has shape {tensor.shape}, expected '
                             f'({mesh.n_elements}, 3, 3)')
        active = (np.einsum('mii->mi', tensor) > 0.0).any(axis=1) | (self.loss_coefficient > 0.0)
        if capacity_over_dt is not None:
            capacity_over_dt = np.broadcast_to(np.asarray(capacity_over_dt, dtype=np.float64), (mesh.n_elements,))
            active |= capacity_over_dt > 0.0
        active &= self.window_elements
        if not active.any():
            raise ValueError(f'{self.label()}: no window element conducts, loses or stores (all coefficients 0)')
        self.active = active
        terms = [assemble_lagrange_stiffness_3d(full, tensor, elements=active)]
        if self.loss_coefficient.any():
            terms.append(assemble_lagrange_mass_3d(full, self.loss_coefficient, elements=active))
        self._capacity_matrix = None
        if capacity_over_dt is not None:
            self._capacity_matrix = assemble_lagrange_mass_3d(full, capacity_over_dt, elements=active)
            terms.append(self._capacity_matrix)
        active_dofs = np.zeros(full.n_dofs, dtype=bool)
        active_dofs[full.element_dofs[active].ravel()] = True
        self._active_dofs = active_dofs
        storing = active & (self.loss_coefficient > 0.0)
        if capacity_over_dt is not None:
            storing |= active & (capacity_over_dt > 0.0)
        anchored_dofs = np.zeros(full.n_dofs, dtype=bool)   # anchors by topology (floating_region_count)
        anchored_dofs[full.element_dofs[storing].ravel()] = True
        for faces, condition, _ in self._all_face_conditions():
            faces = self._active_faces(faces) if len(faces) else faces
            if len(faces) and robin_assembled(condition):
                terms.append(assemble_face_mass_3d(full, faces, condition.h))
                anchored_dofs[full.face_dofs(robin_anchoring(faces, condition))] = True
        fixed_value, _ = self._fixed_values()
        held = ~np.isnan(fixed_value)
        fixed = np.flatnonzero(active_dofs & held)
        free = np.flatnonzero(active_dofs & ~held)
        matrix = pattern_sum(terms)   # one pattern whatever the coefficients (cuDSS plan reuse)
        free_rows = matrix[free]   # sliced once for the reduced operator and the lift coupling
        reduced = free_rows[:, free].tocsr()
        if self.require_anchors:
            floating = floating_region_count(reduced, anchored_dofs[free])
            if floating:
                raise ValueError(f'{self.label()}: {floating} connected region(s) of the {self.unknown_name} touch no '
                                 f'{self.anchor_text}, so the system is singular. {self.remedy_text}')
        self._matrix, self._reduced = matrix, reduced
        self._free, self._fixed = free, fixed
        self._coupling = free_rows[:, fixed].tocsr()
        self._refresh_constant_terms()
        self._factorize()
        self.operator_dirty = False

    def _fixed_values(self) -> tuple:
        """(value per full dof, NaN when free; sequence of the condition holding it, 0 when free) of the Dirichlet faces,
        held regions and fixed dofs on the active dofs of the last assembly; later sequences win shared dofs."""
        full, active_dofs = self.full, self._active_dofs
        constraints = []   # (sequence, dof indices, value)
        for faces, condition, sequence in self._all_face_conditions():
            faces = self._active_faces(faces) if len(faces) else faces
            if len(faces) and condition.kind == 'dirichlet':
                constraints.append((sequence, full.face_dofs(faces), condition.value))
        for mask, value, sequence in self.held:
            ids = full.element_dofs_of(mask)
            constraints.append((sequence, ids[active_dofs[ids]], value))
        for ids, values, sequence in self.fixed_dofs:
            keep = active_dofs[ids]
            constraints.append((sequence, ids[keep], values[keep]))
        fixed_value = np.full(full.n_dofs, np.nan)
        owner = np.zeros(full.n_dofs, dtype=np.int64)   # sequence of the condition holding each dof (0 free)
        for sequence, indices, value in sorted(constraints, key=lambda entry: entry[0]):
            fixed_value[indices] = value
            owner[indices] = sequence
        return fixed_value, owner

    def _refresh_constant_terms(self) -> None:
        """Loss / Robin drives, fixed values, their owners and the lift of the current conditions on the last assembly
        (the fixed dof set and the matrix unchanged)."""
        full, active = self.full, self.active
        static_load = np.zeros(full.n_dofs)
        if self.loss_coefficient.any():
            static_load += assemble_lagrange_load_3d(full, self.loss_drive, elements=active)
        for faces, condition, _ in self._all_face_conditions():
            faces = self._active_faces(faces) if len(faces) else faces
            if len(faces) and robin_assembled(condition):
                static_load += assemble_face_load_3d(full, faces, condition.h * condition.external)
        fixed_value, self._dof_owner = self._fixed_values()
        fixed = self._fixed
        self._lift = (self._coupling @ fixed_value[fixed]) if fixed.size else np.zeros(self._free.size)
        self.values[fixed] = fixed_value[fixed]
        self._static_load = static_load
        self._constants_stale = False

    def _solve_dtype(self):
        return np.float32 if getattr(self.linear_solver, 'precision', 'double') == 'single' else np.float64

    def _factorize(self) -> None:
        """Factorise (bind) the reduced operator on the run backend; the free values seed an iterative solution."""
        self._solution = None
        if not self._free.size:
            return
        dtype = self._solve_dtype()
        operator = sparse_module(self.backend).csr_matrix(self._reduced.astype(dtype))   # gpu: one device copy
        self.linear_solver.factorize(operator)
        self._solution = array_module(self.backend).asarray(self.values[self._free], dtype=dtype)

    # ---- solve ----
    def solve(self, source_density=None, extra_load=None) -> None:
        """Solve with the P0 source density f per element (None: none) and an assembled (n_dofs,) extra_load (gradient
        loads of drives; None: none) on top of the face, loss and point terms (transient: + C/dt u of the current
        values); writes `values` and extends them into the PML shell."""
        if self.operator_dirty or self.active is None:
            raise RuntimeError(f'{self.label()}: assemble before solve')
        if self._constants_stale:
            self._refresh_constant_terms()
        full = self.full
        load = self._static_load.copy()
        if extra_load is not None:
            load += np.asarray(extra_load, dtype=np.float64)
        if source_density is not None:
            density = np.asarray(source_density, dtype=np.float64)
            if density.any():
                load += assemble_lagrange_load_3d(full, density, elements=self.active)
        for faces, condition, _ in self._all_face_conditions():
            if condition.kind == 'neumann' and condition.value != 0.0 and len(faces):
                load += assemble_face_load_3d(full, self._active_faces(faces), condition.value)
        sources = [entry for entry in self.point_sources if self.active[entry[0]]]
        if sources:
            elements, references, powers = zip(*sources)
            load += assemble_point_load_3d(full, np.array(elements), np.array(references), np.array(powers))
        if self._capacity_matrix is not None:
            load += self._capacity_matrix @ self.values
        self._load = load
        if self._free.size:
            rhs = array_module(self.backend).asarray(load[self._free] - self._lift, dtype=self._solve_dtype())
            reference = 'initial_residual' if self._capacity_matrix is not None else 'rhs'   # PHY-1 stop rule
            self.linear_solver.solve(rhs, self._solution, reference)
            self.values[self._free] = np.asarray(to_host(self._solution), dtype=np.float64)
        self._extend_into_shell()
        self.n_solves += 1

    def _extend_into_shell(self) -> None:
        """PML shell dofs = the field at their position clamped just inside the window (the tensor edge extension)."""
        if self._shell is None:
            self._shell = shell_dof_locations(self.full)
        ids, elements, reference = self._shell
        if ids.size:
            values, _ = evaluate_lagrange_3d(self.full, self.values, elements, reference, gradients=False)
            found = elements >= 0
            self.values[ids[found]] = values[found]

    # ---- reactions, fluxes, post-processing ----
    def reaction(self, dof_indices) -> float:
        """Sum of (A u - b) of the last solve over dofs: the flux leaving those fixed dofs into the domain [c u m];
        sources on them are subtracted."""
        residual = self._matrix @ self.values - self._load
        return float(residual[np.asarray(dof_indices, dtype=np.int64)].sum())

    def face_flux(self, face) -> float:
        """inward_flux of an outer face; a Dirichlet face counts only the dofs its condition holds (a shared dof belongs
        to the later condition), so the reactions of all faces add up without double counting."""
        condition, sequence = self.face_conditions[face]
        faces = self.face_faces(face)
        if condition.kind != 'dirichlet' or not len(faces):
            return self.inward_flux(faces, condition)
        face_dofs = self.full.face_dofs(self._active_faces(faces))
        return self.reaction(face_dofs[self._dof_owner[face_dofs] == sequence])

    def inward_flux(self, faces, condition) -> float:
        """Flux entering the domain through mesh faces under `condition` (dirichlet: the reaction of their dofs)."""
        faces = self._active_faces(np.asarray(faces, dtype=np.int64)) if len(faces) else faces
        if not len(faces):
            return 0.0
        full = self.full
        if condition.kind == 'dirichlet':
            return self.reaction(full.face_dofs(faces))
        if condition.kind == 'neumann':
            return condition.value * float(assemble_face_load_3d(full, faces, 1.0).sum())
        drive = assemble_face_load_3d(full, faces, condition.h * condition.external).sum()
        return float(drive - (assemble_face_mass_3d(full, faces, condition.h) @ self.values).sum())

    def value_at(self, elements, reference) -> np.ndarray:
        """u at located points (NaN where the element is -1)."""
        values, _ = evaluate_lagrange_3d(self.full, self.values, elements, reference, gradients=False)
        return values

    def centroid_values(self) -> np.ndarray:
        """Float64 u at every element's reference centroid (D12: the per-element state)."""
        return centroid_values_3d(self.full, self.values)

    def element_integrals(self, tensor=None) -> dict:
        """fem.assembly3d.element_integrals_3d of the last solution over the active elements (0 elsewhere)."""
        elements = np.zeros(self.mesh.n_elements, dtype=bool) if self.active is None else self.active
        return element_integrals_3d(self.full, self.values, elements=elements, tensor=tensor)


def shell_dof_locations(full) -> tuple:
    """(int64 ids, elements, reference coordinates) of the dofs of a full-mesh Lagrange map outside the window and of
    their positions clamped just inside it (empty without a PML shell)."""
    mesh = full.mesh
    if mesh.window_elements().all():
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros((0, 3))
    points = full.dof_points()
    low, high = np.array(mesh.window[:3]), np.array(mesh.window[3:])
    inset = PLANE_TOLERANCE * float((high - low).max())
    clamped = np.clip(points, low + inset, high - inset)
    ids = np.flatnonzero(np.any(np.abs(clamped - points) > 2.0 * inset, axis=1))
    elements, reference = PointLocator3D(mesh).locate(clamped[ids])
    return ids, elements, reference


def coefficient_tensor_3d(c_xx, c_yy, c_zz) -> np.ndarray:
    """Float64 (n, 3, 3) diagonal tensors of per-element components (host or backend arrays)."""
    diagonal = np.stack([np.asarray(to_host(component), dtype=np.float64) for component in (c_xx, c_yy, c_zz)],
                        axis=1)
    return diagonal[:, :, None] * np.eye(3)


class FEMDiffusionPhysicsBase3D(FEMDiffusionPhysicsCommon):
    """Shared wrapper of a FEMDiffusionSystem3D physics on a 'tetrahedra' mesh (heat 3D, potential 3D,
    magnetostatics_3d): FEMDiffusionPhysicsCommon on FACES_3D plus interface conditions, point sources / probes at
    located points; lagrange_field for fem.renderer3d.LagrangeSampleRenderer."""

    _faces = FACES_3D

    def _init_fem(self, domain, order, linear_solver, label, unknown_name, anchor_text, remedy_text,
                  require_anchors=True) -> None:
        if getattr(domain, 'grid_kind', 'tensor') != 'tetrahedra':
            raise ValueError(f'{label()}: this finite-element solver needs a tetrahedron mesh '
                             '(fem.domain3d.TetrahedronDomain)')
        self.domain = domain
        self.xp = domain.xp
        self.system = FEMDiffusionSystem3D(domain.mesh, order, linear_solver, backend=domain.backend.name, label=label,
                                           unknown_name=unknown_name, anchor_text=anchor_text,
                                           remedy_text=remedy_text, require_anchors=require_anchors)
        self.bcs = {face: BCSide('neumann') for face in FACES_3D}
        self.probes = []   # (FieldProbe, element, reference)

    @property
    def axisymmetric(self) -> bool:
        return False

    def add_surface_condition(self, faces, bc: BCSide) -> int:
        """Condition on mesh faces (object interfaces: BuildContext.boundary_faces); returns its handle."""
        handle = self.system.add_surface_condition(faces, self._condition(bc))
        self._rhs_changed()
        return handle

    def set_surface_condition(self, handle, bc: BCSide) -> None:
        self.system.set_surface_condition(handle, self._condition(bc))
        self._rhs_changed()

    def _locate_point(self, point, what) -> tuple:
        """(element, reference (3,)) of one point (x, y, z) [m]; ValueError outside the mesh."""
        elements, reference = self.domain.locate(np.asarray(point, dtype=np.float64).reshape(1, 3))
        if elements[0] < 0:
            raise ValueError(f'{what} at {tuple(float(value) for value in point)} m lies outside the mesh')
        return int(elements[0]), reference[0]

    def add_point_source_at(self, point, power: float) -> None:
        """Point source of total `power` at (x, y, z) [m] (the exact delta load sum P phi_i(x0))."""
        self.system.add_point_source(*self._locate_point(point, 'point source'), power)
        self._rhs_changed()

    def add_probe_at(self, point) -> FieldProbe:
        """Record the field at (x, y, z) [m] (evaluated in its element) whenever the physics records probes."""
        element, reference = self._locate_point(point, 'probe')
        probe = FieldProbe(element, 0, self._probe_field)
        self.probes.append((probe, element, reference))
        return probe

    def _record_probes(self, time, replace_same_time=False) -> None:
        for probe, element, reference in self.probes:
            value = self.system.value_at(np.array([element]), reference[None, :])[0]
            probe.record_value(np.float32(self.reference_value + value), time, replace_same_time)


def fem_diffusion_estimate_3d(mesh, order, element_fields, nodal_fields=1, direct=False, multigrid=False, cudss=None,
                              refactorizes=False, active_fraction=1.0) -> MemoryEstimate:
    """FEMDiffusionSystem3D + its physics on a TetrahedronMeshEstimate (fem.diffusion.fem_system_estimate,
    uncalibrated; SuperLU fill FEM_FILL_FACTOR_3D on the 3D heuristic); cudss = the 'linear' cuDSS back end or None."""
    n_dofs = int(mesh.n_nodes) + (int(mesh.n_edges) if order == 2 else 0)
    n_active = active_dofs(n_dofs, active_fraction)
    direct_factor = (('host SuperLU factor L + U (float64, FEM 3D, uncalibrated)',
                      int(FEM_FILL_FACTOR_3D[order] * host_real_lu_bytes(n_active, dimensions=3))) if direct else None)
    return fem_system_estimate(n_dofs, lagrange_entries_per_dof(mesh, order), 10 if order == 2 else 4,
                               int(mesh.n_elements),
                               element_fields, nodal_fields, 3, direct_factor=direct_factor, multigrid=multigrid,
                               cudss=cudss, refactorizes=refactorizes, active_fraction=active_fraction)
