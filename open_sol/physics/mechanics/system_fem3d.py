"""ElasticFEMSystem3D: the host float64 vector P1 / P2 elasticity of a solid on a tetrahedron mesh (fem.elasticity3d):
holds on the full-mesh dof ids, the active dofs of the solid elements, the 6 x 6 Gram rigid-mode check per solid region,
one factorisation of K_ff per assembly by the 'linear' sparse_direct runtime (through sparse_direct.DiffusionDirectSolver,
the operator handed over on the run backend), solves, Newton tangent solves (geometric nonlinearity), reactions and the
consistent mass. numpy at module level."""
import math
import time

import numpy as np

from ...fem.dofs3d import lagrange_dofs_3d
from ...fem.assembly import pattern_sum
from ...fem.elasticity3d import COMPONENTS, assemble_elastic_mass_3d, assemble_elastic_stiffness_3d
from ...fem.mesh3d import window_faces
from .pml3d import ring_only_dofs
from .system3d import RIGID_RANK_TOLERANCE, rigid_motion_text
from .system_fem import DirectElasticSystem, factor_text


class ElasticFEMSystem3D(DirectElasticSystem):
    """Vector Lagrange elasticity of one TetrahedronMesh on the full-mesh scalar map `full` (P1 nodes; P2 nodes, then
    n_points + edge): `held` / `held_values` (n_dofs, 3), `operator` K (host CSR, dof 3 d + component), `active` (scalar
    dofs of the solid elements), `free` (bool over the vector dofs), `unknown` (host (n_dofs, 3)). K_ff is factorised
    lazily by the first solve after an assembly. `bloch` (bloch3d.BlochCell3D, set by the caller) makes the rigid-mode
    check periodic; with `rigid_motions_allowed` the free rigid motions are counted (n_rigid_motions), not refused."""

    components = COMPONENTS
    _release_solution = True       # the free-dof solution vector lives on the run backend (device memory)
    _reaction_components = COMPONENTS

    def __init__(self, mesh, order, label, linear_solver=None, backend='cpu'):
        from ...solvers.sparse_direct import DiffusionDirectSolver, SparseDirectSolver
        if order not in (1, 2):
            raise ValueError(f'{label()}: element order must be 1 or 2, got {order!r}')
        if linear_solver is None:   # scripts: the tetrahedra Automatic (host SuperLU)
            linear_solver = SparseDirectSolver(backend=backend)
        if not isinstance(linear_solver, SparseDirectSolver):
            raise ValueError(f'{label()}: a solid on a tetrahedron mesh solves with the sparse_direct linear solver, got '
                             f'{type(linear_solver).__name__}: assign Automatic or a sparse_direct node')
        if linear_solver.backend != backend:
            raise ValueError(f'{label()}: the linear solver runs on the {linear_solver.backend} backend, the mesh on the '
                             f'{backend} one')
        self.runtime = linear_solver
        self.linear_solver = DiffusionDirectSolver(linear_solver)   # PCGSolver protocol: factorize / solve
        self.linear_solver.label = label
        self.mesh = mesh
        self.order = int(order)
        self.label = label
        self.backend = backend
        self.full = lagrange_dofs_3d(mesh, self.order)
        self.n_dofs = self.full.n_dofs
        self.held = np.zeros((self.n_dofs, COMPONENTS), dtype=bool)
        self.held_values = np.zeros((self.n_dofs, COMPONENTS))
        self.unknown = np.zeros((self.n_dofs, COMPONENTS))
        self.operator = None
        self.solid = None
        self.active = None
        self.free = None
        self.load = None              # full load vector (3 n_dofs) of the last static solve
        self.residual = None          # f_int - f_ext (3 n_dofs) of a nonlinear static solve: its reactions
        self.operator_dirty = True
        self.assembly_seconds = {}    # part -> wall s of the last assemble (the physics' finalize note)
        self.n_solves = 0             # static solves (coupling revision counter)
        self._solution = None         # run-backend free-dof solution of the linear solver
        self._factor_holds_operator = False   # the linear solver's factor is K_ff (not a Newton tangent)
        self._tangent_held = False            # the linear solver holds the factor of the last tangent solve
        self._window_faces = {}
        self.bloch = None                    # bloch3d.BlochCell3D of the periodic axes (None: no periodic face)
        self.rigid_motions_allowed = False   # modes about a target > 0 (free body, Bloch cell): rigid motions counted
        self.n_rigid_motions = 0             # free rigid motions of the last assembly (rigid_motions_allowed only)
        self.pml = None                      # pml3d.ElasticPMLShell3D (set by the caller): shell elements stretched

    # ---- geometry of holds and loads ----
    def face_faces(self, face) -> np.ndarray:
        """Int64 mesh faces of the window on an outer face (fem.mesh3d.window_faces, cached)."""
        cached = self._window_faces.get(face)
        if cached is None:
            cached = self._window_faces[face] = window_faces(self.mesh, face)
        return cached

    def hold(self, dof_ids, u_x=None, u_y=None, u_z=None) -> np.ndarray:
        """Hold components [m] (None = free) on full-mesh scalar dof ids (later holds win the value); returns the held
        dof mask (n_dofs, 3)."""
        dof_ids = np.asarray(dof_ids, dtype=np.int64).reshape(-1)
        dof_mask = np.zeros((self.n_dofs, COMPONENTS), dtype=bool)
        for component, value in enumerate((u_x, u_y, u_z)):
            if value is not None:
                dof_mask[dof_ids, component] = True
                self.held_values[dof_ids, component] = float(value)
        self.held |= dof_mask
        self.operator_dirty = True
        return dof_mask

    # ---- assembly and solve ----
    @property
    def operator(self):
        """K (host CSR) of the last assembly; a deferred one (assemble(defer_operator=True)) is assembled here at first
        use (a geometric-nonlinear static solve never reads it: Newton uses the tangent assembler, reactions the
        residual)."""
        if self._operator is None and self._operator_build is not None:
            build, self._operator_build = self._operator_build, None
            started = time.perf_counter()
            self._operator = build()
            self.assembly_seconds['stiffness K'] = time.perf_counter() - started
        return self._operator

    @operator.setter
    def operator(self, value):
        self._operator, self._operator_build = value, None

    def _operator_missing(self) -> bool:
        return self._operator is None and self._operator_build is None

    def assemble(self, material, solid, defer_operator=False) -> None:
        """K of the solid elements (host bool (n_elements,)) from the per-element Voigt stiffness (n_elements, 6, 6),
        the free set and the rigid-mode check (K_ff is factorised by the next solve); with `pml` the shell's solid
        elements stretched (complex symmetric K). defer_operator: K is assembled at the first read of `operator`
        (the caller keeps material unchanged until then)."""
        solid = np.asarray(solid, dtype=bool)
        if not solid.any():
            raise ValueError(f'{self.label()}: no element of the physical window has stiffness (youngs_modulus or '
                             'c11 > 0)')
        full = self.full
        self.solid = solid.copy()
        started = time.perf_counter()
        pml, solid_elements = self.pml, self.solid

        def build():
            if pml is None:
                return assemble_elastic_stiffness_3d(full, material, elements=solid_elements).tocsr()
            window = self.mesh.window_elements()
            # pattern_sum: a scipy `+` would drop entries summing to 0, so the pencils' patterns would follow the values
            return pattern_sum([assemble_elastic_stiffness_3d(full, material, elements=solid_elements & window),
                                assemble_elastic_stiffness_3d(full, material, elements=solid_elements & ~window,
                                                              stretch=pml.stretches)])
        self.operator = None if defer_operator else build()
        if defer_operator:
            self._operator_build = build
        assembled = time.perf_counter()
        active = np.zeros(self.n_dofs, dtype=bool)
        active[full.element_dofs[solid].ravel()] = True
        self.active = active
        self._check_rigid_modes(self.held & active[:, None])
        self.assembly_seconds = ({} if defer_operator else {'stiffness K': assembled - started})
        self.assembly_seconds['rigid-mode check'] = time.perf_counter() - assembled
        free = (active[:, None] & ~self.held).reshape(-1)
        if not free.any():
            raise ValueError(f'{self.label()}: every displacement is held, nothing is solved')
        self.free = free
        self._solution = None   # factorised by the first solve (an eigenfrequency study never factorises K alone)
        self.unknown = self.fixed_values().reshape(-1, COMPONENTS)
        self.operator_dirty = False

    @property
    def device_tangent(self) -> bool:
        """Newton tangents assembled and factorised on the GPU (plan G5): the gpu backend with a cuDSS 'linear' back
        end (SuperLU factors on the host, so its tangents stay host)."""
        from ...solvers.sparse_direct import CUDSS_BACK_ENDS
        return self.backend == 'gpu' and self.runtime.back_end in CUDSS_BACK_ENDS

    def solve_tangent_device(self, values, indptr, indices, rhs) -> np.ndarray:
        """Host free-dof solution x of T_ff x = rhs for the free-free tangent as device values on the host canonical
        pattern (indptr, indices) of a device TangentAssembler3D: cuDSS refreshes its factor on the device."""
        self._require_assembled()
        n_free = int(np.count_nonzero(self.free))
        if len(indptr) != n_free + 1:
            raise ValueError(f'{self.label()}: reduced tangent of {len(indptr) - 1} rows, expected {n_free}')
        # the solve right after is the factor's check (a reused plan skips its probe solves)
        self.linear_solver.factorize_device_values(values, indptr, indices, defer_check=True)
        return self._tangent_solution(rhs)

    def fixed_values(self) -> np.ndarray:
        """Flat (3 n_dofs,) held values on the held active dofs, 0 elsewhere."""
        return np.where(self.held & self.active[:, None], self.held_values, 0.0).reshape(-1)

    def _reaction_mask(self, dof_mask) -> np.ndarray:
        return np.asarray(dof_mask, dtype=bool) & self.held & self.active[:, None]

    def solve_homogeneous(self, load) -> np.ndarray:
        """Host (n_dofs, 3) K^-1 load on the free dofs with every held dof at 0, on the K factor of the last assembly
        (the relaxed initial stress of a frequency response's prestress); not a static solve (n_solves unchanged)."""
        return self._homogeneous(load).reshape(-1, COMPONENTS)

    def mass(self, density):
        """Host CSR consistent mass of the solid elements (density per mesh element [kg/m^3]); with `pml` the shell's
        part weighted by the stretch Jacobian s_x s_y s_z (complex)."""
        if self.pml is None:
            return assemble_elastic_mass_3d(self.full, density, elements=self.solid)
        window = self.mesh.window_elements()
        density = np.asarray(density, dtype=np.float64)

        def stretched_density(points, elements):
            return density[elements][:, None] * self.pml.jacobian(points)

        # each part on the scalar pattern of its own kept K block pattern (window, stretched shell), then x I3
        shell = assemble_elastic_mass_3d(self.full, stretched_density, elements=self.solid & ~window,
                                         degree=2 * self.order + 2)
        return pattern_sum([assemble_elastic_mass_3d(self.full, density, elements=self.solid & window), shell])

    def shell_only_dofs(self) -> np.ndarray:
        """Host bool over the vector dofs: dofs of shell elements that no window element touches (the PML share of a
        mode)."""
        return ring_only_dofs(self.full.element_dofs, self.mesh.window_elements(), self.n_dofs, COMPONENTS)

    def solve_summary(self) -> str:
        """One line: elements, free dofs and the factor of the last assembly ('' before one)."""
        if self.free is None:
            return ''
        return (f'P{self.order} elasticity on {int(self.solid.sum()):,} tetrahedra: {int(self.free.sum()):,} free dofs, '
                f'{factor_text(self.runtime)}')

    # ---- rigid-mode check ----
    def _check_rigid_modes(self, held):
        """ValueError for a solid region (solid elements joined through shared dofs and periodic partners) whose held
        dofs leave a rigid motion free: rank of the 6 x 6 Gram matrix of the rows (e_a, x~ x e_a) at its held dofs, plus
        the rows the periodic axes exclude (bloch3d), below 6; a Bloch phase != 1 admits none. With
        rigid_motions_allowed the free motions are counted in n_rigid_motions instead."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        self.n_rigid_motions = 0
        bloch = self.bloch
        if bloch is not None and not bloch.trivial:
            return
        full = self.full
        local = full.element_dofs[self.solid]
        first = np.broadcast_to(local[:, :1], local.shape).ravel()   # every element dof joined to its first dof
        second = local.ravel()
        if bloch is not None:   # a region continues across its periodic faces
            first = np.concatenate([first, np.arange(self.n_dofs)])
            second = np.concatenate([second, bloch.master])
        graph = coo_matrix((np.ones(first.size, dtype=np.int8), (first, second)), shape=(self.n_dofs, self.n_dofs))
        _, labels = connected_components(graph, directed=False)
        points = full.dof_points()
        extent = float(np.max(np.array(self.mesh.window[3:]) - np.array(self.mesh.window[:3])))
        periodic_rows = np.zeros((0, 6)) if bloch is None else bloch.rigid_rows()
        active = np.flatnonzero(self.active)   # node indices per region (one sort, no full pass per region)
        region_of = labels[active]
        order = np.argsort(region_of, kind='stable')
        boundaries = np.flatnonzero(np.diff(region_of[order])) + 1
        for members in (np.split(active[order], boundaries) if active.size else []):
            node, component = np.nonzero(held[members])
            node = members[node]
            if node.size == 0 and not periodic_rows.size:
                if self.rigid_motions_allowed:
                    self.n_rigid_motions += 6
                    continue
                reason = 'it has no support at all'
            else:
                position = points[node] if node.size else points[members]
                centroid = position.mean(axis=0)
                relative = points[node] - centroid
                scale = float(np.sqrt(np.mean(np.sum((position - centroid) ** 2, axis=1)))) or 1.0
                relative /= scale
                rows = np.zeros((node.size, 6))
                rows[np.arange(node.size), component] = 1.0
                rows[:, 3:] = np.cross(relative, np.eye(3)[component])   # u_a = t_a + omega . (x x e_a)
                rows = np.concatenate([rows, math.sqrt(max(node.size, 1)) * periodic_rows])   # comparable weight
                eigenvalues, eigenvectors = np.linalg.eigh(rows.T @ rows)
                threshold = RIGID_RANK_TOLERANCE * eigenvalues[-1]
                if eigenvalues[0] > threshold:
                    continue
                null_basis = eigenvectors[:, eigenvalues <= threshold]
                if self.rigid_motions_allowed:
                    self.n_rigid_motions += null_basis.shape[1]
                    continue
                reason = (f'{null_basis.shape[1]} rigid motion(s) stay free, e.g. '
                          + rigid_motion_text(null_basis, centroid, scale, extent))
            low, high = points[members].min(axis=0), points[members].max(axis=0)
            where = ', '.join(f'{axis} {low[index]:.4g} .. {high[index]:.4g} m' for index, axis in enumerate('xyz'))
            remedy = ('; a free Bloch cell (every periodic phase 1, e.g. k = 0) solves only about an eigenfrequency '
                      'target_frequency > 0, where its rigid modes are dropped' if bloch is not None else
                      '; a free body (free-free) solves only for its vibration modes in an eigenfrequency study with '
                      'target_frequency > 0, where its rigid-body modes are dropped, or in a frequency response')
            raise ValueError(f'{self.label()}: the solid region at {where} (elements with youngs_modulus or c11 > 0 '
                             f'joined through shared nodes) can move as a rigid body, so the solve is singular: '
                             f'{reason}. Hold '
                             'every solid region with Fixed, Roller or Displacement supports or a Fixed (domain) '
                             f'region, e.g. rollers on x_lo, y_lo and z_lo{remedy}.')
