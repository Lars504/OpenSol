"""Complex scalar harmonic FEM on tetrahedron meshes (new_modules_plan S2): ScalarHarmonicSystem3D, the fem.harmonic
system -div(a grad u) + (b + j omega c - omega^2 m) u = f by Lagrange P1 / P2 on an element selection, with a fem.pml
shell, Dirichlet / Neumann / Robin face conditions and point loads, over assembly3d / dofs3d. Host numpy, scipy inside
functions."""
# Conventions as fem.harmonic (module header there): conditions, PML outer sides, sequences, active selection, the
# pencil and the per-frequency factor are HarmonicSystemBase's. 3D specifics: the full-mesh DofMap3D numbering throughout
# (the system = the dofs of the active elements, sorted full ids; matrices restricted from the full numbering), a: number,
# (n_elements,), (n_elements, 3) diagonal, (n_elements, 3, 3) or callable (assembly3d forms); outer faces on the mesh's
# face_tags (outer side of the meshed box) or, on a face whose shell is not stretched, on window_faces; face terms only
# on faces with an active element on either side; affine elements.
import numpy as np

from .assembly3d import (SplitCoefficient, assemble_face_load_3d, assemble_face_mass_3d, assemble_lagrange_load_3d,
                         assemble_lagrange_mass_3d, assemble_lagrange_stiffness_3d, assemble_point_load_3d,
                         scalar_coefficient_at, tensor_coefficient_at)
from .dofs3d import lagrange_dofs_3d
from .elements3d import lagrange_basis_3d
from .harmonic import HarmonicSystemBase, dirichlet_values, is_zero, system_positions
from .mesh import side_members
from .mesh3d import window_faces
from .pml import shell_widths
from .quadrature3d import TETRAHEDRON_MAX_DEGREE

FACES_3D = ('x_lo', 'x_hi', 'y_lo', 'y_hi', 'z_lo', 'z_hi')
PML_EXTRA_DEGREE_3D = 2   # rule degree added for the stretched coefficients


class ScalarHarmonicSystem3D(HarmonicSystemBase):
    """Complex scalar harmonic system on a TetrahedronMesh (module header): Lagrange order 1 / 2, pml = a 3D
    fem.pml.PMLStretch of the mesh's shell (pml_shell) or None; linear_solver = the 'linear' role runtime
    (SparseDirectSolver; None = host SuperLU)."""
    faces = FACES_3D

    def __init__(self, mesh, order=2, linear_solver=None, *, pml=None, pml_outer='dirichlet', symmetric=True,
                 label=lambda: 'harmonic fem 3d'):
        if order not in (1, 2):
            raise ValueError(f'{label()}: element order must be 1 or 2, got {order!r}')
        self.mesh = mesh
        self.order = int(order)
        self.full = lagrange_dofs_3d(mesh, self.order)
        self.axisymmetric = False
        self._shell_widths = shell_widths(mesh)
        self._locator = None
        self._active_mask = None   # active elements of the last assembly
        super().__init__(self.full.n_dofs, linear_solver, pml, pml_outer, symmetric, label)
        self._check_pml()

    def _check_pml(self) -> None:
        pml = self.pml
        if pml is None:
            return
        if pml.dimensions != 3:
            raise ValueError(f'{self.label()}: a tetrahedron mesh needs a 3D PMLStretch, got {pml.dimensions}D')
        missing = [face for face in pml.faces if self._shell_widths[face] <= 0.0]
        if missing:
            raise ValueError(f"{self.label()}: PML on {', '.join(missing)} but the mesh has no shell there")

    def active_elements(self) -> np.ndarray:
        if self.active is not None:
            if self.active.shape != (self.mesh.n_elements,):
                raise ValueError(f'{self.label()}: active mask has shape {self.active.shape}, expected '
                                 f'({self.mesh.n_elements},)')
            return self.active
        if self.pml is not None:
            return np.ones(self.mesh.n_elements, dtype=bool)
        return self.mesh.window_elements()

    def face_entities(self, face) -> np.ndarray:
        """Int64 mesh faces a face condition acts on (fem.harmonic module header)."""
        mesh = self.mesh
        if self._shell_widths[face] > 0.0 and (self.pml is None or face not in self.pml.faces):
            return window_faces(mesh, face)
        return np.asarray(mesh.face_tags.get(face, np.zeros(0)), dtype=np.int64)

    def _rule_degree(self, default):
        """None (the assembler's default) without a PML, else default + PML_EXTRA_DEGREE_3D (capped)."""
        if self.pml is None:
            return None
        return min(max(int(default) + PML_EXTRA_DEGREE_3D, 1), TETRAHEDRON_MAX_DEGREE)

    def _stretched(self, coefficient, tensor, split=True):
        """The coefficient unchanged without a PML, else its values x the stretch factors at points: with split a
        SplitCoefficient (the unstretched elements, no node beyond a layered window face, keep the plain coefficient:
        affine reference forms; the stiffness / mass assemblers only) unless the coefficient is itself a callable, else
        the callable."""
        pml = self.pml
        if pml is None:
            return coefficient
        n_elements = self.mesh.n_elements

        def at_points(points, elements):
            n_points = points.shape[1]
            if tensor:
                values = tensor_coefficient_at(coefficient, points, elements, n_elements, n_points)
                return values * pml.tensor_factor(points)
            values = scalar_coefficient_at(coefficient, points, elements, n_elements, n_points)
            return values * pml.volume_factor(points)
        if callable(coefficient) or not split:
            return at_points
        # convex elements with every node on the window side of each layered face: every rule point has s = 1
        varying = pml.in_layer(self.mesh.points[self.mesh.tets]).any(axis=1)
        return SplitCoefficient(coefficient, at_points, varying)

    def _assemble_terms(self) -> dict:
        mesh, full, order = self.mesh, self.full, self.order
        active = self.active_elements()
        if not active.any():
            raise ValueError(f'{self.label()}: no active element')
        global_ids = np.unique(full.element_dofs[active].ravel())
        self._active_mask = active.copy()

        whole = global_ids.size == full.n_dofs   # every dof active (the usual PML case): no fancy-index copies

        def restrict(matrix):
            return matrix.tocsr() if whole else matrix.tocsr()[global_ids][:, global_ids].tocsr()

        coefficients = self.coefficients
        stiffness = [restrict(assemble_lagrange_stiffness_3d(full, self._stretched(coefficients['a'], True), active,
                                                             self._rule_degree(2 * order - 2)))]
        damping, mass = [], []
        mass_degree = self._rule_degree(2 * order)
        for key, terms in (('b', stiffness), ('damping', damping), ('mass', mass)):
            if not is_zero(coefficients[key]):
                terms.append(restrict(assemble_lagrange_mass_3d(full, self._stretched(coefficients[key], False),
                                                                active, mass_degree)))
        boundary_load = np.zeros(len(global_ids), dtype=np.complex128)
        constraints = []
        all_points = full.dof_points()
        pairs = mesh.face_elements
        for faces, condition, sequence in self._condition_items():
            if not len(faces):
                continue
            if condition.kind == 'dirichlet':
                full_ids = full.face_dofs(faces)
                positions, keep = system_positions(global_ids, full_ids)
                constraints.append((sequence, positions,
                                    dirichlet_values(condition.value, all_points[full_ids[keep]])))
                continue
            first, second = side_members(pairs[faces], active)
            faces = faces[first | second]
            if not len(faces):
                continue
            if condition.kind == 'robin':
                if not is_zero(condition.admittance):
                    stiffness.append(restrict(assemble_face_mass_3d(full, faces, condition.admittance)))
                if not is_zero(condition.damping):
                    damping.append(restrict(assemble_face_mass_3d(full, faces, condition.damping)))
            if not is_zero(condition.value):
                boundary_load = boundary_load + assemble_face_load_3d(full, faces, condition.value)[global_ids]
        in_window = mesh.in_window(all_points[global_ids])
        return {'global_ids': global_ids, 'stiffness': stiffness, 'damping': damping, 'mass': mass,
                'boundary_load': boundary_load, 'constraints': constraints, 'in_window': in_window}

    def _source_load(self) -> np.ndarray:
        full, global_ids = self.full, self.global_ids
        load = np.zeros(len(global_ids), dtype=np.complex128)
        if self.source is not None and not is_zero(self.source):
            full_load = assemble_lagrange_load_3d(full, self._stretched(self.source, False, split=False),
                                                  self._active_mask,
                                                  self._rule_degree(self.order + 2))
            load = load + full_load[global_ids]
        if self.point_loads:
            if self._locator is None:
                from .sampling3d import PointLocator3D
                self._locator = PointLocator3D(self.mesh)
            points = np.array([point for point, _ in self.point_loads], dtype=np.float64).reshape(-1, 3)
            elements, reference = self._locator.locate(points)
            for index, (point, _) in enumerate(self.point_loads):
                if elements[index] < 0 or not self._active_mask[elements[index]]:
                    raise ValueError(f'{self.label()}: point load at {point.tolist()} m lies outside the active '
                                     'elements')
            values = np.array([value for _, value in self.point_loads], dtype=np.complex128)
            load = load + assemble_point_load_3d(full, elements, reference, values)[global_ids]
        return load

    def _element_rows(self, element_mask) -> np.ndarray:
        """Bool per system dof touched by the masked active elements."""
        rows = np.zeros(len(self.global_ids), dtype=bool)
        mask = np.asarray(element_mask, dtype=bool) & self._active_mask
        positions, _ = system_positions(self.global_ids, np.unique(self.full.element_dofs[mask].ravel()))
        rows[positions] = True
        return rows

    # ---- fields ----
    def centroid_values(self) -> np.ndarray:
        """Complex128 u at every element's centroid (D12 per-element state; 0 on inactive elements)."""
        basis = lagrange_basis_3d(self.order, np.array([[0.25, 0.25, 0.25]]))[0][0]
        return self.values[self.full.element_dofs] @ basis

    def lagrange_field(self) -> tuple:
        """(full-mesh DofMap3D, complex128 coefficients) of u (renderers split re / im / abs / phase)."""
        return self.full, self.values
