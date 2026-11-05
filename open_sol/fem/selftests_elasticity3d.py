"""Self-test of fem.elasticity3d (CPU, specs() empty, check({}) on a graded structured Kuhn box, no gmsh): stiffness
symmetry and rigid-mode null space, the anisotropic patch test against consistent face tractions, free thermal
expansion, element strains, mass / body / traction totals and the nodal average, P1 and P2. error = number of failed
sub-checks (tolerance 0), values in details."""
import numpy as np

from ..model.schema import SelfTestCase
from ..model.selftest_helpers import checks_result

_LENGTH = 1e-6        # box side scale [m]
_RTOL = 1e-10         # exact identities in float64 (first estimates)
_DENSITY = 2300.0
_FORCE = (3e9, -2e9, 5e9)       # N/m^3
_TRACTION = (1e6, 2e6, -3e6)    # Pa
_GRADIENT = np.array([[1.0, 0.4, -0.3], [0.2, -0.7, 0.5], [-0.6, 0.1, 0.9]]) * 1e-3   # linear field u = G x + c
_ALPHA = np.array([2e-6, 5e-6, 1e-6])
_RISE = 40.0


def graded_box_mesh():
    """Graded 3 x 2 x 2 Kuhn box (no shell)."""
    from .mesh3d import structured_box_mesh
    x = _LENGTH * np.array([0.0, 0.3, 0.75, 1.2])
    y = _LENGTH * np.array([0.0, 0.45, 0.8])
    z = _LENGTH * np.array([0.0, 0.25, 0.7])
    return structured_box_mesh(x, y, z)


def _anisotropic_matrix():
    """A fixed SPD 6 x 6 Voigt stiffness [Pa] with every coupling nonzero."""
    rng = np.random.default_rng(3)
    factor = rng.standard_normal((6, 6))
    return 1e10 * (factor @ factor.T + 6.0 * np.eye(6))


def _voigt_strain(gradient):
    """Engineering Voigt strain (xx, yy, zz, yz, xz, xy) of a displacement gradient G[c, d] = d u_c / d x_d."""
    symmetric = gradient + gradient.T
    return np.array([gradient[0, 0], gradient[1, 1], gradient[2, 2], symmetric[1, 2], symmetric[0, 2],
                     symmetric[0, 1]])


def _stress_tensor(voigt):
    xx, yy, zz, yz, xz, xy = voigt
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def _check_elasticity3d(_):
    from .assembly3d import face_areas_3d
    from .dofs3d import lagrange_dofs_3d
    from .elasticity3d import (assemble_body_load_3d, assemble_elastic_mass_3d, assemble_elastic_stiffness_3d,
                               assemble_face_traction_3d, assemble_stress_load_3d, element_strains_3d,
                               isotropic_matrix_3d, local_dof_points_3d, nodal_average_3d, voigt_matrix_3d)

    checks = {}

    def add(name, value, tolerance=_RTOL):
        checks[name] = {'value': float(value), 'tolerance': tolerance, 'passed': bool(value <= tolerance)}

    mesh = graded_box_mesh()
    n_elements = mesh.n_elements
    volume = float(mesh.volumes().sum())
    lame_lambda, mu = 6e10, 4e10
    isotropic = isotropic_matrix_3d(np.full(n_elements, lame_lambda), np.full(n_elements, mu))
    entries = {'c11': lame_lambda + 2 * mu, 'c22': lame_lambda + 2 * mu, 'c33': lame_lambda + 2 * mu,
               'c12': lame_lambda, 'c13': lame_lambda, 'c23': lame_lambda, 'c44': mu, 'c55': mu, 'c66': mu}
    add('isotropic_matrix', np.abs(voigt_matrix_3d(entries, n_elements) - isotropic).max() / (lame_lambda + 2 * mu))
    anisotropic = np.broadcast_to(_anisotropic_matrix(), (n_elements, 6, 6))
    for order in (1, 2):
        dofs = lagrange_dofs_3d(mesh, order)
        points = dofs.dof_points()
        stiffness = assemble_elastic_stiffness_3d(dofs, anisotropic)
        scale = float(np.abs(stiffness.data).max())
        add(f'symmetry_p{order}', float(np.abs((stiffness - stiffness.T).data).max(initial=0.0)) / scale)
        rigid = np.zeros((6, dofs.n_dofs, 3))
        for axis in range(3):
            rigid[axis, :, axis] = 1.0
            rotation = np.zeros(3)
            rotation[axis] = 1.0
            rigid[3 + axis] = np.cross(rotation, points) / _LENGTH
        add(f'rigid_null_space_p{order}', max(float(np.abs(stiffness @ mode.ravel()).max()) for mode in rigid) / scale)
        displacement = points @ _GRADIENT.T + 1e-9
        strain = _voigt_strain(_GRADIENT)
        stress = _stress_tensor(_anisotropic_matrix() @ strain)
        tractions = np.zeros(3 * dofs.n_dofs)
        for face, faces in mesh.face_tags.items():
            normal = np.zeros(3)
            normal['xyz'.index(face[0])] = 1.0 if face.endswith('hi') else -1.0
            tractions += assemble_face_traction_3d(dofs, faces, tuple(stress @ normal))
        internal = stiffness @ displacement.ravel()
        add(f'patch_tractions_p{order}', float(np.abs(internal - tractions).max()) / float(np.abs(internal).max()))
        expansion = points * (_ALPHA * _RISE)
        thermal = np.zeros((n_elements, 6))
        thermal[:] = _anisotropic_matrix() @ np.concatenate([_ALPHA * _RISE, np.zeros(3)])
        thermal_load = assemble_stress_load_3d(dofs, thermal)
        add(f'free_expansion_p{order}', float(np.abs(stiffness @ expansion.ravel() - thermal_load).max())
            / float(np.abs(thermal_load).max()))
        reference = np.random.default_rng(5).dirichlet(np.ones(4), size=7)[:, 1:]
        values, strains = element_strains_3d(dofs, displacement, reference)
        add(f'element_strains_p{order}', float(np.abs(strains - strain).max()) / float(np.abs(strain).max()))
        barycentric = np.column_stack([1.0 - reference.sum(axis=1), reference])   # (q, 4)
        physical = np.einsum('qk,mkd->mqd', barycentric, mesh.points[mesh.tets])
        exact_values = physical @ _GRADIENT.T + 1e-9
        add(f'element_values_p{order}', float(np.abs(values - exact_values).max()) / float(np.abs(exact_values).max()))
        local = np.broadcast_to(strain, (n_elements, dofs.n_local, 6))
        add(f'nodal_average_p{order}', float(np.abs(nodal_average_3d(dofs, local) - strain).max())
            / float(np.abs(strain).max()))
        _, local_strains = element_strains_3d(dofs, displacement, local_dof_points_3d(order))
        add(f'local_dof_strains_p{order}', float(np.abs(local_strains - strain).max()) / float(np.abs(strain).max()))
        mass = assemble_elastic_mass_3d(dofs, _DENSITY)
        for axis in range(3):
            unit = np.zeros((dofs.n_dofs, 3))
            unit[:, axis] = 1.0
            add(f'mass_total_{"xyz"[axis]}_p{order}', abs(unit.ravel() @ (mass @ unit.ravel()) / (_DENSITY * volume)
                                                          - 1.0))
        body = assemble_body_load_3d(dofs, _FORCE).reshape(-1, 3).sum(axis=0)
        add(f'body_total_p{order}', float(np.abs(body / (np.array(_FORCE) * volume) - 1.0).max()))
        top = mesh.face_tags['z_hi']
        traction = assemble_face_traction_3d(dofs, top, _TRACTION).reshape(-1, 3).sum(axis=0)
        area = float(face_areas_3d(mesh, top).sum())
        add(f'traction_total_p{order}', float(np.abs(traction / (np.array(_TRACTION) * area) - 1.0).max()))
    return checks_result(checks, elements=mesh.n_elements)


ELASTICITY_3D_SELFTEST_CASES = (
    SelfTestCase(
        name='fem_elasticity3d_identities',
        label='FEM tetrahedra: vector elasticity identities',
        description='CPU checks of fem.elasticity3d on a graded 3 x 2 x 2 Kuhn box (no gmsh), P1 and P2: isotropic '
                    'Voigt matrix = the c_ij form; stiffness of a full anisotropic SPD 6 x 6 matrix symmetric with the '
                    'six rigid modes in its null space; patch test: K u of a linear field u = G x + c equals the '
                    'consistent face tractions sigma n of its constant stress on all six outer faces; free thermal '
                    'expansion u = alpha dT x: K u = the stress load C alpha dT; element strains / values of the '
                    'linear field at random points and at the local dof points; nodal average of a constant; mass, '
                    'body force and face traction totals rho V, f V, t A. All 1e-10 (first estimates, not yet run). '
                    'error = number of failed sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_elasticity3d,
        max_wall_time=120.0),
)
