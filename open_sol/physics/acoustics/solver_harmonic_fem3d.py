"""pressure_acoustics on tetrahedron meshes (PressureAcousticsFEMPhysics3D): solver_harmonic_fem's AcousticHarmonicBase
on fem.harmonic3d (Lagrange P1 / P2, affine), the PML on the mesh's shell (fem.pml.pml_shell), outer faces on their
window faces, object interface faces as boundaries and LagrangeSampleRenderer display. Host numpy, scipy inside
functions."""
# Conventions as solver_harmonic_fem (module header there); 3D specifics: point sources and probes by
# TetrahedronDomain.locate, entity loads over mesh faces (assembly3d face forms), the background source operators on the
# full-mesh DofMap3D restricted to the dofs of their elements.
import numpy as np

from ...core.constants import FACES_3D
from ...fem.assembly3d import (assemble_face_load_3d, assemble_lagrange_mass_3d, assemble_lagrange_stiffness_3d,
                               assemble_point_load_3d, face_areas_3d)
from ...fem.dofs3d import lagrange_dofs_3d
from ...fem.elements3d import lagrange_basis_3d
from ...fem.harmonic3d import ScalarHarmonicSystem3D
from ...fem.mesh3d import window_faces
from ...fem.pml import pml_shell
from ...fem.quadrature3d import triangle_face_rule
from ...fem.renderer3d import LagrangeSampleRenderer
from ...fem.sampling3d import centroid_values_3d, evaluate_lagrange_3d
from .solver_harmonic_fem import AcousticHarmonicBase, oriented_normals


class PressureAcousticsFEMPhysics3D(AcousticHarmonicBase):
    """pressure_acoustics on a fem.domain3d.TetrahedronDomain (module header): window elements (plus the PML shell
    when pml_faces), outer faces on their window faces (fem.mesh3d.window_faces)."""

    grid_kind = 'tetrahedra'
    dimensions = 3
    faces = FACES_3D

    def __init__(self, domain, order=2, linear_solver=None, **options):
        if options.get('axisymmetric'):
            raise ValueError('pressure_acoustics: a 3D model has no axisymmetric geometry (it is solved in full)')
        super().__init__(domain, order, linear_solver, **options)

    def renderer(self):
        return LagrangeSampleRenderer(self, self.name, self._field_infos(), self._time_label)

    def lagrange_field(self, component) -> tuple:
        """(DofMap3D, host coefficients) of a display component (the Session's tetrahedron reads)."""
        return self._component_field(component)

    def _dof_maps(self) -> tuple:
        return lagrange_dofs_3d(self.mesh, self.order), lagrange_dofs_3d(self.mesh, 0)

    def _new_system(self, pml):
        return ScalarHarmonicSystem3D(self.mesh, self.order, self.linear_solver, pml=pml, pml_outer='dirichlet',
                                      symmetric=True, label=self._label)

    def _window_elements(self) -> np.ndarray:
        return np.asarray(self.mesh.window_elements(), dtype=bool)

    def _make_pml(self, wave_speed, omega_reference):
        try:
            return pml_shell(self.mesh, wave_speed, omega_reference, self.pml_reflection, faces=self.pml_faces)
        except ValueError as exc:
            raise ValueError(f'{self._label()}: {exc} (another physics on this mesh may pad other faces)') from None

    def _window_face_entities(self, face) -> np.ndarray:
        return np.asarray(window_faces(self.mesh, face), dtype=np.int64)

    def _entity_load(self, entities, value) -> np.ndarray:
        return assemble_face_load_3d(self.full, np.asarray(entities, dtype=np.int64), value)

    def _entity_quadrature(self, entities) -> tuple:
        """assemble_face_load_3d's default rule (degree order + 2 on affine faces; the face basis sums to 1): (points
        (k, q, 3), rule weight x face area)."""
        faces = np.asarray(entities, dtype=np.int64)
        bary, rule_weights = triangle_face_rule(self.order + 2)
        points = np.einsum('qi,kic->kqc', bary, self.mesh.points[self.mesh.faces[faces]])
        return points, face_areas_3d(self.mesh, faces)[:, None] * rule_weights[None, :]

    def _entity_pairs(self, entities) -> np.ndarray:
        return self.mesh.face_elements[np.asarray(entities, dtype=np.int64)]

    def _n_entities(self) -> int:
        return self.mesh.n_faces

    def _entity_normals(self, entities, member) -> np.ndarray:
        mesh = self.mesh
        corners = mesh.points[mesh.faces[entities]]       # (k, 3, 3)
        normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        return oriented_normals(normals, corners.mean(axis=1), mesh.face_elements[entities], member,
                                mesh.centroids())

    def _source_operators(self, stiffness_coefficient, mass_coefficient, elements) -> tuple:
        full = self.full
        ids = np.unique(full.element_dofs[elements].ravel()).astype(np.int64)
        stiffness = assemble_lagrange_stiffness_3d(full, stiffness_coefficient, elements).tocsr()
        mass = assemble_lagrange_mass_3d(full, mass_coefficient, elements).tocsr()
        return stiffness[ids][:, ids].tocsr(), mass[ids][:, ids].tocsr(), ids

    def _in_window(self, points) -> np.ndarray:
        return np.asarray(self.mesh.in_window(np.asarray(points, dtype=np.float64).reshape(-1, 3)), dtype=bool)

    def _locate(self, point) -> tuple:
        elements, reference = self.domain.locate(np.asarray(point, dtype=np.float64).reshape(1, 3))
        element = int(elements[0])
        if element < 0:
            raise ValueError(f'{self._label()}: point {tuple(float(value) for value in point)} m lies outside the mesh')
        return element, np.asarray(reference[0], dtype=np.float64)

    def _point_load(self, point, value) -> np.ndarray:
        element, reference = self._locate(point)
        if not self._active[element]:
            raise ValueError(f'{self._label()}: point source at {tuple(float(item) for item in point)} m lies outside '
                             'the fluid (in a rigid obstacle or a non-fluid element)')
        vector = assemble_point_load_3d(self.full, np.array([element]), reference[None, :],
                                        np.array([complex(value)]))
        return np.asarray(vector, dtype=np.complex128)

    def _point_weights(self, element, reference) -> tuple:
        """The full map's element dofs and its Lagrange basis at the reference point (evaluate_lagrange_3d's basis)."""
        basis, _ = lagrange_basis_3d(self.order, np.asarray(reference, dtype=np.float64)[None, :], False)
        return np.asarray(self.full.element_dofs[element], dtype=np.int64), np.asarray(basis[0], dtype=np.float64)

    def _centroid_values(self, coefficients) -> np.ndarray:
        return centroid_values_3d(self.full, coefficients)

    def _centroid_basis(self) -> np.ndarray:
        return lagrange_basis_3d(self.order, np.full((1, 3), 0.25))[0][0]   # centroid_values_3d's basis

    def _centroid_gradients(self, coefficients) -> np.ndarray:
        count = self.mesh.n_elements
        _, gradients = evaluate_lagrange_3d(self.full, coefficients, np.arange(count, dtype=np.int64),
                                            np.full((count, 3), 0.25))
        return np.nan_to_num(gradients)
