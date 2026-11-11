"""ElasticFEMSystem: the host float64 vector P1 / P2 elasticity of a solid on a triangle mesh (mesh plan 2.7, stage T5):
dofs on the solid window elements, holds kept on the full-mesh dof ids (nodes, P2 edge nodes), assembly by
fem.elasticity, rigid-mode check per solid region, one factorisation of K_ff per assembly by the 'linear' sparse_direct
runtime (through sparse_direct.DiffusionDirectSolver, lazily at the first solve), solves, Newton tangent solves
(geometric nonlinearity, new_modules_plan K6), reactions and the consistent mass; an elastic PML ring (stretched pencil operators, new_modules_plan K5) and the azimuthal order m >= 1 form (three
components, K7). numpy at module level, scipy inside functions."""
import numpy as np

from ...core.backend import array_module, sparse_module, to_host
from ...fem.dofs import lagrange_dofs
from ...fem.mesh import side_members
from ...fem.elasticity import (AZIMUTHAL_COMPONENTS, COMPONENTS, assemble_azimuthal_geometric_stiffness,
                               assemble_azimuthal_mass, assemble_azimuthal_stiffness, assemble_elastic_mass,
                               assemble_elastic_stiffness, assemble_geometric_stiffness)
from .pml2d import ring_only_dofs

_RANK_TOLERANCE = 1e-8   # smallest / largest singular value of the held rigid-mode rows of a region
_FACE_AXES = {'x_lo': (0, 0), 'x_hi': (0, 2), 'y_lo': (1, 1), 'y_hi': (1, 3)}   # face -> (coordinate, window index)


def factor_text(runtime) -> str:
    """'factor ... MiB of host memory (SuperLU) / device memory (cuDSS) in ... s' of a sparse_direct runtime's last
    factorisation (a cuDSS factor leaves host_bytes 0; also after an 'auto' SuperLU fallback the bytes tell)."""
    if not runtime.host_bytes and runtime.cudss_device_bytes:
        where = f'{runtime.cudss_device_bytes / 2 ** 20:.4g} MiB of device memory (cuDSS)'
    else:
        where = f'{runtime.host_bytes / 2 ** 20:.4g} MiB of host memory'
    return f'factor {where} in {runtime.last_factor_seconds or 0.0:.3g} s'


class DirectElasticSystem:
    """Factor / solve layer of ElasticFEMSystem and ElasticFEMSystem3D: K_ff factorised on demand by the 'linear'
    runtime, Newton tangent factors, static solves and reactions over the flat free mask `free` (c n) and
    fixed_values(); hooks _check_static() and _reaction_mask(dof_mask)."""

    _release_solution = False   # release_factor also drops the free-dof solution vector (3D: device memory)
    _reaction_components = 2    # components a reaction reports (F_x, F_y; 3D F_z too)

    def _operator_missing(self) -> bool:
        """No K of an assembly (3D: neither assembled nor deferred; reading `operator` there would assemble it)."""
        return self.operator is None

    def _require_assembled(self) -> None:
        if self.operator_dirty or self._operator_missing():
            raise RuntimeError(f'{self.label()}: assemble before solve')

    def _check_static(self) -> None:
        """Hook: raise when the system has no static solve (2D: an azimuthal order)."""

    def _reaction_mask(self, dof_mask) -> np.ndarray:
        """Hook: bool (solid dofs, c) of the held dofs of dof_mask a reaction sums over."""
        raise NotImplementedError

    def release_factor(self) -> None:
        """Free the linear solver's factor (K_ff or a tangent; cuDSS device memory) before another large factorisation;
        the next solve factorises again (2D keeps the solution vector, so solve_summary still names the factor)."""
        self.linear_solver.release_factor()
        if self._release_solution:
            self._solution = None
        self._factor_holds_operator = False
        self._tangent_held = False

    def _solve_free(self, rhs) -> np.ndarray:
        """Host float64 K_ff^-1 rhs (free-dof vector), factorising K_ff on the run backend first when needed (the
        'linear' runtime counts it; also after a Newton tangent took the factor)."""
        xp = array_module(self.backend)
        if self._solution is None or not self._factor_holds_operator:
            free = np.flatnonzero(self.free)
            reduced = self.operator[free][:, free].tocsr()
            self.linear_solver.factorize(sparse_module(self.backend).csr_matrix(reduced))
            self._factor_holds_operator = True
            self._tangent_held = False
            if self._solution is None:
                self._solution = xp.zeros(free.size, dtype=np.float64)
        self.linear_solver.solve(xp.asarray(rhs), self._solution, 'rhs')
        return np.array(to_host(self._solution), dtype=np.float64)

    def solve(self, load) -> None:
        """unknown = K^-1 load on the free dofs with the held values lifted (one back substitution)."""
        self._require_assembled()
        self._check_static()
        load = np.asarray(load, dtype=np.float64)
        fixed = self.fixed_values()
        free = np.flatnonzero(self.free)
        solution = fixed.copy()
        solution[free] = self._solve_free((load - self.operator @ fixed)[free])
        self.unknown = solution.reshape(-1, self.components)
        self.load = load
        self.residual = None
        self.n_solves += 1

    def _homogeneous(self, load) -> np.ndarray:
        """Flat (c n,) K^-1 load with every held dof at 0 on the K factor of the last assembly (not a static solve)."""
        self._require_assembled()
        free = np.flatnonzero(self.free)
        solution = np.zeros(self.free.size)
        solution[free] = self._solve_free(np.asarray(load, dtype=np.float64)[free])
        return solution

    def _tangent_solution(self, rhs) -> np.ndarray:
        """Host free-dof solution of rhs on the tangent the runtime just factorised (marks the tangent held)."""
        self._factor_holds_operator = False
        self._tangent_held = True
        xp = array_module(self.backend)
        if self._solution is None:
            self._solution = xp.zeros(int(np.count_nonzero(self.free)), dtype=np.float64)
        self.linear_solver.solve(xp.asarray(np.asarray(rhs, dtype=np.float64)), self._solution, 'rhs')
        return np.array(to_host(self._solution), dtype=np.float64)

    def solve_tangent_reduced(self, reduced, rhs) -> np.ndarray:
        """Host free-dof solution x of T_ff x = rhs for the host canonical free-free tangent CSR of the fem tangent
        assembler (handed to the factor as it is: no slicing, no device round trip)."""
        self._require_assembled()
        self._check_static()
        n_free = int(np.count_nonzero(self.free))
        if reduced.shape != (n_free, n_free):
            raise ValueError(f'{self.label()}: reduced tangent of shape {reduced.shape}, expected ({n_free}, {n_free})')
        # host direct: a value refresh of the kept analysis, checked by the solve right after (no probe solves)
        self.linear_solver.factorize_host_operator(reduced, symmetric_known=True, defer_check=True)
        return self._tangent_solution(rhs)

    def solve_last_tangent(self, rhs) -> np.ndarray:
        """Host free-dof solution on the tangent factor of the latest tangent solve (modified Newton: one back
        substitution); the caller guarantees no other factorisation on the runtime since."""
        if self.operator_dirty or self._operator_missing() or self._solution is None or not self._tangent_held:
            raise RuntimeError(f'{self.label()}: no tangent factor to reuse (solve_tangent_reduced first)')
        xp = array_module(self.backend)
        self.linear_solver.solve(xp.asarray(np.asarray(rhs, dtype=np.float64)), self._solution, 'rhs')
        return np.array(to_host(self._solution), dtype=np.float64)

    def accept_solution(self, displacement, load, residual=None) -> None:
        """Take a static solution solved outside the K factor (a nonlinear Newton solve, the 3D piezoelectric coupled
        solve): unknown = displacement (n, c), the kept load (c n,), residual f_int - f_ext (c n,) for the reactions
        (None = K u - f); one more static solve."""
        self._require_assembled()
        self.unknown = np.asarray(displacement, dtype=np.float64).reshape(-1, self.components)
        self.load = np.asarray(load, dtype=np.float64)
        self.residual = None if residual is None else np.asarray(residual, dtype=np.float64).reshape(-1)
        self.n_solves += 1

    def reaction(self, dof_mask) -> tuple:
        """(F_x, F_y[, F_z]): sum of K u - f (a nonlinear solve: of its residual f_int - f_ext, the nominal force) over
        the held dofs of dof_mask in the last static solve (the force the support exerts on the solid)."""
        if self.load is None:
            raise RuntimeError(f'{self.label()}: reactions need a static solve first')
        residual = (self.operator @ self.unknown.reshape(-1) - self.load if self.residual is None
                    else self.residual).reshape(-1, self.components)
        totals = np.where(self._reaction_mask(dof_mask), residual, 0.0).sum(axis=0)
        return tuple(float(value) for value in totals[:self._reaction_components])


class ElasticFEMSystem(DirectElasticSystem):
    """Vector Lagrange elasticity of one TriangleMesh: `held` / `held_values` (n_global, c) on the full-mesh dof ids of
    the order (P1: nodes; P2: nodes, then n_points + edge; c = `components`: 2, or 3 for an azimuthal order m >= 1:
    U_r, U_z, U_phi), the solid dof map `dofs` (DofMap of the last assembly), `operator` K (host real CSR, c dof +
    component; with a PML ring the ring as an unstretched elastic continuation: the static prestress state), `free`
    (bool over the vector dofs), `prolongation` (m = 1 axis ties u_phi = -u_r: CSR x = P x_r over the free dofs, else
    None), `unknown` (host (n_dofs, c) of the last solve or shown mode). K_ff is factorised by the first solve after an
    assembly (an eigenfrequency or frequency-response study never factorises K alone). `pml` (fem.pml.PMLStretch, set
    by the caller) stretches the pencil operators (stiffness(stretched=True), mass, geometric_stiffness)."""

    def __init__(self, mesh, order, axisymmetric, label, linear_solver=None, backend='cpu', azimuthal_order=0):
        from ...solvers.sparse_direct import DiffusionDirectSolver, SparseDirectSolver
        if order not in (1, 2):
            raise ValueError(f'ElasticFEMSystem: order must be 1 or 2, got {order!r}')
        azimuthal_order = int(azimuthal_order)
        if azimuthal_order < 0 or (azimuthal_order and not axisymmetric):
            raise ValueError(f'ElasticFEMSystem: an azimuthal order (got {azimuthal_order}) needs >= 0 and, above 0, '
                             'an axisymmetric solid')
        if linear_solver is None:   # scripts: the triangles Automatic (host SuperLU)
            linear_solver = SparseDirectSolver(backend=backend)
        if not isinstance(linear_solver, SparseDirectSolver):
            raise ValueError(f'{label()}: a solid on a triangle mesh solves with the sparse_direct linear solver, got '
                             f'{type(linear_solver).__name__}: assign Automatic or a sparse_direct node')
        if linear_solver.backend != backend:
            raise ValueError(f'{label()}: the linear solver runs on the {linear_solver.backend} backend, the mesh on the '
                             f'{backend} one')
        self.runtime = linear_solver
        self.linear_solver = DiffusionDirectSolver(linear_solver)   # PCGSolver protocol: factorize / solve
        self.linear_solver.label = label
        self.backend = backend
        self.mesh = mesh
        self.order = int(order)
        self.axisymmetric = bool(axisymmetric)
        self.azimuthal_order = azimuthal_order
        self.components = AZIMUTHAL_COMPONENTS if azimuthal_order else COMPONENTS
        self.label = label                       # () -> "solid mechanics 'solid'"
        self.full_dofs = lagrange_dofs(mesh, self.order)   # every element: display / state numbering
        self.n_global = self.full_dofs.n_dofs
        self.held = np.zeros((self.n_global, self.components), dtype=bool)
        self.held_values = np.zeros((self.n_global, self.components))
        self.dofs = None
        self.operator = None
        self.free = None
        self.prolongation = None
        self.unknown = None
        self.load = None                         # full load vector (c n_dofs) of the last static solve
        self.residual = None                     # f_int - f_ext (c n_dofs) of a nonlinear static solve: its reactions
        self.pml = None                         # fem.pml.PMLStretch of an elastic PML ring (set by the caller)
        self._axis_ties = None                   # full-mesh ids of the m = 1 axis nodes (u_phi = -u_r there)
        self._solution = None                    # run-backend free-dof solution; None = nothing factorised
        self._factor_holds_operator = False      # the linear solver's factor is K_ff (not a Newton tangent)
        self._tangent_held = False               # the linear solver holds the factor of the last solve_tangent_reduced
        self.operator_dirty = True
        self.n_solves = 0                        # static solves (coupling revision counter)
        self.rigid_motions_allowed = False       # free body (modes about a target > 0, frequency response): counted
        self.n_rigid_motions = 0                 # free rigid motions of the last assembly (rigid_motions_allowed only)
        x0, y0, x1, y1 = mesh.window
        self._window = (x0, y0, x1, y1)
        self._tolerance = 1e-9 * max(x1 - x0, y1 - y0)

    # ---- geometry of holds and loads ----
    def global_ids_of_edges(self, edges) -> np.ndarray:
        """Sorted full-mesh dof ids on mesh edges (their nodes, P2 also the edge node)."""
        edges = np.asarray(edges, dtype=np.int64)
        ids = [self.mesh.edges[edges].ravel().astype(np.int64)]
        if self.order == 2:
            ids.append(self.mesh.n_points + edges)
        return np.unique(np.concatenate(ids))

    def global_ids_of_elements(self, element_mask) -> np.ndarray:
        """Sorted full-mesh dof ids of the elements under a host bool mask (n_elements,)."""
        elements = np.flatnonzero(element_mask)
        ids = [self.mesh.triangles[elements].ravel().astype(np.int64)]
        if self.order == 2:
            ids.append(self.mesh.n_points + self.mesh.element_edges[elements].ravel().astype(np.int64))
        return np.unique(np.concatenate(ids))

    def face_edges(self, face) -> np.ndarray:
        """Mesh edges on an outer face of the physical window (both nodes on its line, an adjacent window element):
        the window edge also when a PML layer ring surrounds it."""
        if face not in _FACE_AXES:
            raise ValueError(f"unknown face {face!r}; expected one of {', '.join(_FACE_AXES)}")
        axis, index = _FACE_AXES[face]
        line = self._window[index]
        other_low, other_high = self._window[1 - axis], self._window[3 - axis]
        tolerance = self._tolerance
        points = self.mesh.points
        on_line = ((np.abs(points[:, axis] - line) <= tolerance) & (points[:, 1 - axis] >= other_low - tolerance)
                   & (points[:, 1 - axis] <= other_high + tolerance))
        edges = self.mesh.edges
        candidates = on_line[edges[:, 0]] & on_line[edges[:, 1]]
        first, second = side_members(self.mesh.edge_elements, self.mesh.pml_faces == 0)
        return np.flatnonzero(candidates & (first | second))

    def hold(self, global_ids, u_x=None, u_y=None, u_z=None) -> np.ndarray:
        """Hold u_x and / or u_y [m] (and u_z = U_phi of an azimuthal order m >= 1; ignored with two components; None =
        free) on full-mesh dof ids (later holds win the value); returns the held dof mask (n_global, c)."""
        dof_mask = np.zeros((self.n_global, self.components), dtype=bool)
        for component, value in tuple(enumerate((u_x, u_y, u_z)))[:self.components]:
            if value is not None:
                dof_mask[global_ids, component] = True
                self.held_values[global_ids, component] = float(value)
        self.held |= dof_mask
        self.operator_dirty = True   # the free set changes
        return dof_mask

    def tie_axis(self, global_ids) -> None:
        """Azimuthal order 1: tie u_phi = -u_r on these full-mesh dof ids (the axis nodes; regularity of the m = 1
        field), applied by `prolongation` at every assembly."""
        if self.azimuthal_order != 1:
            raise ValueError('tie_axis applies to azimuthal order 1 only')
        self._axis_ties = np.unique(np.asarray(global_ids, dtype=np.int64))
        self.operator_dirty = True

    # ---- assembly and solve ----
    def stiffness(self, coefficients, stretched=False):
        """Host CSR K of the solid map from per-element coefficients (fem.elasticity); stretched: the PML ring's
        complex form (pml set), else the real unstretched one."""
        stretch = self._stretch_arguments() if stretched else {}
        if self.azimuthal_order:
            return assemble_azimuthal_stiffness(self.dofs, coefficients, self.azimuthal_order, **stretch)
        return assemble_elastic_stiffness(self.dofs, coefficients, axisymmetric=self.axisymmetric, **stretch)

    def assemble(self, coefficients, solid) -> None:
        """K of the solid elements (host bool (n_elements,)) from per-element coefficients (fem.elasticity), the
        free set (m = 1: the axis ties) and the rigid-mode check (K_ff is factorised by the next solve)."""
        if not solid.any():
            raise ValueError(f'{self.label()}: no element of the physical window has stiffness (youngs_modulus or c11 '
                             '> 0)')
        self.dofs = lagrange_dofs(self.mesh, self.order, elements=solid)
        self.operator = self.stiffness(coefficients)
        held = self.held[self.dofs.global_ids]
        self._check_rigid_modes(held)
        self.free = ~held.reshape(-1)
        self.prolongation = self._tie_prolongation()
        if not self.free.any():
            raise ValueError(f'{self.label()}: every displacement is held, nothing is solved')
        self._solution = None
        self._factor_holds_operator = False
        self.unknown = self.fixed_values().reshape(-1, self.components)
        self.operator_dirty = False

    def fixed_values(self) -> np.ndarray:
        """Flat (c n_dofs,) held values of the solid dofs (0 on free ones)."""
        ids = self.dofs.global_ids
        return np.where(self.held[ids], self.held_values[ids], 0.0).reshape(-1)

    def _check_static(self) -> None:
        if self.prolongation is not None:
            raise RuntimeError(f'{self.label()}: an azimuthal order has no static solve')

    def _reaction_mask(self, dof_mask) -> np.ndarray:
        return (np.asarray(dof_mask, dtype=bool) & self.held)[self.dofs.global_ids]

    def solve_homogeneous(self, load) -> np.ndarray:
        """Flat (c n_dofs,) K^-1 load with every held dof at 0 on the factor of the last assembly (generalized plane
        strain unit terms; not a static solve)."""
        return self._homogeneous(load)

    def mass(self, density):
        """Host CSR consistent mass of the solid dofs (density per mesh element [kg/m^3]); with `pml` the ring's part
        weighted by the stretch Jacobian (complex)."""
        stretch = self._stretch_arguments()
        if self.azimuthal_order:
            return assemble_azimuthal_mass(self.dofs, density, **stretch)
        return assemble_elastic_mass(self.dofs, density, axisymmetric=self.axisymmetric, **stretch)

    def geometric_stiffness(self, stress):
        """Host CSR K_sigma of a fem.elasticity.StressField on the solid map (with `pml` the ring stretched as K; an
        azimuthal order: the order-m form of a prescribed axisymmetric stress)."""
        stretch = self._stretch_arguments()
        if self.azimuthal_order:
            return assemble_azimuthal_geometric_stiffness(self.dofs, stress, self.azimuthal_order, **stretch).tocsr()
        return assemble_geometric_stiffness(self.dofs, stress, axisymmetric=self.axisymmetric, **stretch).tocsr()

    def ring_only_dofs(self) -> np.ndarray:
        """Host bool over the vector dofs: dofs of PML ring elements no window element touches (the PML share of a
        mode)."""
        dofs = self.dofs
        return ring_only_dofs(dofs.element_dofs, dofs.mesh.pml_faces[dofs.elements] == 0, dofs.n_dofs,
                              self.components)

    def solve_summary(self) -> str:
        """One line: free dofs and the factor of the last assembly ('' before one)."""
        if self.free is None:
            return ''
        factor = ('nothing factorised (no static solve)' if self._solution is None else factor_text(self.runtime))
        order = f', azimuthal order {self.azimuthal_order}' if self.azimuthal_order else ''
        ring = f', elastic PML ring ({self.pml.text()})' if self.pml is not None else ''
        return (f"P{self.order} elasticity on {len(self.dofs.elements):,} triangles{order}{ring}: "
                f'{int(self.free.sum()):,} free dofs, {factor}')

    # ---- internals ----
    def _stretch_arguments(self) -> dict:
        """stretch= / radial_ratio= of the fem.elasticity assemblers ({} without a PML ring)."""
        if self.pml is None:
            return {}
        return {'stretch': self.pml.stretches, 'radial_ratio': self.pml.radial_ratio if self.axisymmetric else None}

    def _tie_prolongation(self):
        """Azimuthal order 1: CSR P (c n_dofs, n_reduced) over the free dofs with u_phi = -u_r on the tied axis nodes
        (their u_phi taken out of `free`; a held u_r holds u_phi at 0 too); None without ties on the solid map."""
        from scipy.sparse import csr_matrix

        if self._axis_ties is None:
            return None
        ids = self.dofs.global_ids
        positions = np.searchsorted(ids, self._axis_ties)
        inside = positions < ids.size
        positions = positions[inside]
        nodes = positions[ids[positions] == self._axis_ties[inside]]
        if not nodes.size:
            return None
        components = self.components
        radial, azimuthal = components * nodes, components * nodes + 2
        free = self.free.copy()
        tied = free[radial] & free[azimuthal]   # a held u_r or u_phi keeps both at their held 0
        free[azimuthal] = False
        self.free = free
        columns = np.cumsum(free) - 1
        free_rows = np.flatnonzero(free)
        rows = np.concatenate([free_rows, azimuthal[tied]])
        cols = np.concatenate([columns[free_rows], columns[radial[tied]]])
        data = np.concatenate([np.ones(free_rows.size), -np.ones(int(tied.sum()))])
        return csr_matrix((data, (rows, cols)), shape=(free.size, int(free.sum())))

    def _check_rigid_modes(self, held):
        """ValueError for a solid region (elements joined through nodes) whose held dofs leave a rigid motion free:
        planar the rank of the (1, 0), (0, 1), (-y, x) rows at its held dofs must be 3, axisymmetric it needs a held
        u_z (u_r strains the hoop); azimuthal order 1: the (U_r, U_z, U_phi) rows of the lateral translation (1, 0, -1)
        and the tilt (z, -r, -z) need rank 2, order >= 2 has no rigid motion. With rigid_motions_allowed the free
        motions are counted in n_rigid_motions."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        self.n_rigid_motions = 0
        if self.azimuthal_order >= 2:
            return
        dofs = self.dofs
        local = dofs.element_dofs
        first = np.broadcast_to(local[:, :1], local.shape).ravel()   # every element dof joined to its first dof
        graph = coo_matrix((np.ones(first.size, dtype=np.int8), (first, local.ravel())),
                           shape=(dofs.n_dofs, dofs.n_dofs))
        n_regions, labels = connected_components(graph, directed=False)
        points = dofs.dof_points()
        scale = max(self._window[2] - self._window[0], self._window[3] - self._window[1])
        remedy = ('; a free body (free-free) solves only for its vibration modes in an eigenfrequency study with '
                  'target_frequency > 0 (triangle mesh), where its rigid-body modes are dropped, or in a frequency '
                  'response')
        # dof indices per region (one sort, not a full-length comparison per region: arrays of many pillars)
        order = np.argsort(labels, kind='stable')
        groups = np.split(order, np.cumsum(np.bincount(labels, minlength=n_regions))[:-1])
        for members in groups:
            where = self._region_text(points[members])
            if self.azimuthal_order == 1:
                inside = np.zeros(dofs.n_dofs, dtype=bool)
                inside[members] = True
                rank = self._azimuthal_rank(held, inside, points, scale)
                if rank == 2:
                    continue
                if self.rigid_motions_allowed:
                    self.n_rigid_motions += 2 - rank
                    continue
                raise ValueError(f'{self.label()}: the solid region {where} can translate sideways or tilt (azimuthal '
                                 'order 1: hold u_r or u_z at two heights, e.g. a Fixed face or a Fixed (domain) '
                                 f'region){remedy}')
            held_x, held_y = members[held[members, 0]], members[held[members, 1]]   # dof indices
            if self.axisymmetric:
                if not held_y.size:
                    if self.rigid_motions_allowed:
                        self.n_rigid_motions += 1
                        continue
                    raise ValueError(f'{self.label()}: the solid region {where} holds no u_z (u_y), so it can slide '
                                     f'along the axis: add a support holding u_y on it{remedy}')
                continue
            rank = 0
            if held_x.size or held_y.size:
                held_points = np.concatenate([points[held_x], points[held_y]])
                centre = held_points.mean(axis=0)
                rows = np.zeros((len(held_points), 3))
                n_x = int(held_x.size)
                rows[:n_x, 0] = 1.0
                rows[:n_x, 2] = -(points[held_x, 1] - centre[1]) / scale
                rows[n_x:, 1] = 1.0
                rows[n_x:, 2] = (points[held_y, 0] - centre[0]) / scale
                singular = np.linalg.svd(rows, compute_uv=False)
                rank = int(np.count_nonzero(singular > _RANK_TOLERANCE * singular[0]))
            if rank == 3:
                continue
            if self.rigid_motions_allowed:
                self.n_rigid_motions += 3 - rank
                continue
            if not held_x.size or not held_y.size:
                missing = 'x' if not held_x.size else 'y'
                raise ValueError(f'{self.label()}: the solid region {where} holds no u_{missing}, so it can slide '
                                 f'along {missing}: add a support holding u_{missing} on it{remedy}')
            raise ValueError(f'{self.label()}: the supports of the solid region {where} let it rotate (every held '
                             'u_x on one line along x and every held u_y on one line along y, or a single held '
                             f'point): hold u_x at two heights or u_y at two positions{remedy}')

    @staticmethod
    def _azimuthal_rank(held, inside, points, scale) -> int:
        """Rank of the m = 1 rigid rows (translation, tilt) at the held (U_r, U_z, U_phi) dofs of a region."""
        held_r, held_z, held_phi = (held[:, component] & inside for component in range(3))
        if not (held_r.any() or held_z.any() or held_phi.any()):
            return 0
        heights = points[:, 1] / scale
        rows = np.concatenate([np.column_stack([np.ones(int(held_r.sum())), heights[held_r]]),
                               np.column_stack([np.zeros(int(held_z.sum())), -points[held_z, 0] / scale]),
                               np.column_stack([-np.ones(int(held_phi.sum())), -heights[held_phi]])])
        singular = np.linalg.svd(rows, compute_uv=False)
        if not singular[0] > 0.0:
            return 0
        return int(np.count_nonzero(singular > _RANK_TOLERANCE * singular[0]))

    @staticmethod
    def _region_text(points):
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        return f'x {x_min:.4g} .. {x_max:.4g} m, y {y_min:.4g} .. {y_max:.4g} m'
