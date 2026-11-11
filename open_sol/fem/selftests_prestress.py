"""Self-tests of the prestress pieces of fem.elasticity / fem.elasticity3d (new_modules_plan K1; CPU, specs() empty,
check({}) on structured meshes, no gmsh): geometric stiffness identities, initial-stress force patch tests, stress
sampling, Green-Lagrange internal force / tangent, and tensioned-string frequencies of (K + K_sigma, M). error =
number of failed sub-checks (tolerance 0), values in details."""
import numpy as np

from ..model.schema import SelfTestCase
from ..model.selftest_helpers import SubChecks, max_relative_error
from .selftests_elasticity3d import graded_box_mesh

_LENGTH = 1e-6         # mesh scale [m]
_RTOL = 1e-10          # exact identities in float64 (first estimates)
_FD_RTOL = 1e-6        # central difference of the internal force against the tangent
_STRING_RTOL = 1e-3    # P2 string frequencies, 16 cells (dispersion error ~1e-4 at n = 3)
_DENSITY = 2300.0      # kg/m^3
_TENSION = 1e9         # Pa: the string's axial prestress
_LAME_LAMBDA = 6e8     # Pa: deliberately soft, so the string check sees K and K_sigma at a similar size
_SHEAR = 4e8           # Pa
_STRING_CELLS = 16
_FD_STEP = 1e-4
_ROTATION_ANGLE = 0.4  # rad: the finite rotation of the Green-Lagrange checks


def _recorder():
    checks = SubChecks()

    def add(name, value, tolerance=_RTOL):
        checks.add_bound(name, value, tolerance)
    return checks, add


def _matrix_relative(matrix, reference) -> float:
    return float(abs(matrix - reference).max()) / max(float(abs(reference).max()), 1e-300)


def _rectangle_mesh(x_edges, y_edges):
    """Structured triangle mesh of a tensor grid, every cell split into two right triangles."""
    from .mesh import build_triangle_mesh
    n_x, n_y = len(x_edges), len(y_edges)
    grid_x, grid_y = np.meshgrid(x_edges, y_edges, indexing='ij')
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    triangles = []
    for i in range(n_x - 1):
        for j in range(n_y - 1):
            a, b, c, d = i * n_y + j, (i + 1) * n_y + j, (i + 1) * n_y + j + 1, i * n_y + j + 1
            triangles += [(a, b, c), (a, c, d)]
    return build_triangle_mesh(points, np.array(triangles),
                               window=(float(x_edges[0]), float(y_edges[0]), float(x_edges[-1]), float(y_edges[-1])))


def _spd(size, seed, scale=1e10) -> np.ndarray:
    """A fixed SPD matrix [Pa] with every coupling nonzero."""
    factor = np.random.default_rng(seed).standard_normal((size, size))
    return scale * (factor @ factor.T + size * np.eye(size))


def _planar_coefficients():
    matrix = _spd(3, 21)
    return {'c11': matrix[0, 0], 'c12': matrix[0, 1], 'c16': matrix[0, 2], 'c22': matrix[1, 1],
            'c26': matrix[1, 2], 'c66': matrix[2, 2]}


def _axisymmetric_coefficients():
    matrix = _spd(3, 22)   # (rr, zz, tt) block; the shear row is uncoupled in the axisymmetric form
    return {'c11': matrix[0, 0], 'c12': matrix[0, 1], 'c13': matrix[0, 2], 'c22': matrix[1, 1],
            'c23': matrix[1, 2], 'c33': matrix[2, 2], 'c66': 0.4 * matrix[0, 0]}


def _rotation_matrix(axis, angle) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * cross @ cross


def _voigt_strain_3d(gradient):
    symmetric = gradient + gradient.T
    return np.array([gradient[0, 0], gradient[1, 1], gradient[2, 2], symmetric[1, 2], symmetric[0, 2],
                     symmetric[0, 1]])


def _tangent_checks(add, label, force, tangent, n_vector, scale, rng):
    """Central difference of force(u) against tangent(u) @ v at a random u, and the tangent's symmetry."""
    displacement = 1e-2 * scale * rng.standard_normal(n_vector)
    direction = 1e-2 * scale * rng.standard_normal(n_vector)
    difference = (force(displacement + _FD_STEP * direction) - force(displacement - _FD_STEP * direction)) / (
        2.0 * _FD_STEP)
    matrix = tangent(displacement)
    add(f'tangent_difference_{label}', max_relative_error(difference, matrix @ direction), _FD_RTOL)
    add(f'tangent_symmetry_{label}', _matrix_relative(matrix, matrix.T) if matrix.nnz else 0.0)


# ---- identities ----
def _check_identities_3d(add):
    from .dofs3d import lagrange_dofs_3d
    from .elasticity import StressField
    from .elasticity3d import (CENTROID_3D, assemble_elastic_stiffness_3d, assemble_face_traction_3d,
                               assemble_geometric_stiffness_3d, assemble_stress_load_3d, assemble_tangent_stiffness_3d,
                               green_lagrange_strains_3d, initial_stress_force_3d, internal_force_3d,
                               sample_stress_3d, voigt_tensor_3d)

    mesh = graded_box_mesh()
    n_elements = mesh.n_elements
    volumes = mesh.volumes()
    rng = np.random.default_rng(11)
    stress = 1e8 * rng.standard_normal((n_elements, 6))
    thermal = 1e7 * rng.standard_normal((n_elements, 6))
    voigt = _spd(6, 3)
    material = np.broadcast_to(voigt, (n_elements, 6, 6))
    uniform = 1e8 * np.array([1.0, -0.6, 0.4, 0.3, -0.2, 0.5])
    gradient = np.array([[1.0, 0.4, -0.3], [0.2, -0.7, 0.5], [-0.6, 0.1, 0.9]]) * 1e-3
    omega = np.array([0.3, -0.5, 0.8]) / _LENGTH
    rotation = _rotation_matrix(omega, _ROTATION_ANGLE)
    stretch_value = 1.5 - 0.4j
    reference = rng.dirichlet(np.ones(4), size=5)[:, 1:]
    for order in (1, 2):
        dofs = lagrange_dofs_3d(mesh, order)
        points = dofs.dof_points()
        n_vector = 3 * dofs.n_dofs
        geometric = assemble_geometric_stiffness_3d(dofs, stress)
        scale = float(abs(geometric).max())
        add(f'geometric_symmetry_3d_p{order}', _matrix_relative(geometric, geometric.T))
        translation = max(float(np.abs(geometric @ np.tile(np.eye(3)[axis], dofs.n_dofs)).max()) for axis in range(3))
        add(f'geometric_translation_3d_p{order}', translation / scale)
        # infinitesimal rotation u = omega x x: u K_sigma u = int |omega|^2 tr sigma - omega . sigma omega dV
        spin = np.cross(omega, points).ravel()
        energy = float(spin @ (geometric @ spin))
        tensors = voigt_tensor_3d(stress)
        terms = volumes * (omega @ omega * np.trace(tensors, axis1=1, axis2=2)
                           - np.einsum('i,mij,j->m', omega, tensors, omega))
        bound = float(np.sum(volumes * np.abs(stress).sum(axis=1))) * float(omega @ omega)
        add(f'rotation_energy_3d_p{order}', abs(energy - float(terms.sum())) / bound)
        # the work of sigma0 on the quadratic Green-Lagrange strain of that rotation = half that energy
        quadratic = green_lagrange_strains_3d(dofs, spin, CENTROID_3D)[:, 0]
        add(f'rotation_work_3d_p{order}', abs(float(np.sum(volumes[:, None] * stress * quadratic)) - 0.5 * energy)
            / bound)
        # initial-stress force: uniform sigma0 = - its boundary tractions; any sigma0 has no net force / moment
        force = initial_stress_force_3d(dofs, np.tile(uniform, (n_elements, 1)))
        tractions = np.zeros(n_vector)
        uniform_tensor = voigt_tensor_3d(uniform)
        for face, faces in mesh.face_tags.items():
            normal = np.zeros(3)
            normal['xyz'.index(face[0])] = 1.0 if face.endswith('hi') else -1.0
            tractions += assemble_face_traction_3d(dofs, faces, tuple(uniform_tensor @ normal))
        add(f'initial_force_patch_3d_p{order}', max_relative_error(-force, tractions))
        random_force = initial_stress_force_3d(dofs, stress).reshape(-1, 3)
        total = float(np.abs(random_force).sum())
        add(f'initial_force_balance_3d_p{order}', float(np.abs(random_force.sum(axis=0)).max()) / total)
        add(f'initial_force_moment_3d_p{order}',
            float(np.abs(np.cross(points, random_force).sum(axis=0)).max()) / (total * _LENGTH))
        add(f'initial_force_stress_load_3d_p{order}',
            max_relative_error(initial_stress_force_3d(dofs, stress), -assemble_stress_load_3d(dofs, stress)))
        # stress sampling: sigma0 + C eps(u) - beta dT of a linear field
        displacement = points @ gradient.T + 1e-9
        field = StressField(prescribed=stress, material=material, displacement=displacement, thermal=thermal)
        expected = stress[:, None, :] + (voigt @ _voigt_strain_3d(gradient))[None, None, :] - thermal[:, None, :]
        add(f'sampled_stress_3d_p{order}', max_relative_error(sample_stress_3d(dofs, field, reference), expected))
        from_displacement = assemble_geometric_stiffness_3d(dofs, StressField(material=material,
                                                                              displacement=displacement))
        constant = np.tile(voigt @ _voigt_strain_3d(gradient), (n_elements, 1))
        add(f'geometric_from_displacement_3d_p{order}',
            _matrix_relative(from_displacement, assemble_geometric_stiffness_3d(dofs, constant)))
        stretched = assemble_geometric_stiffness_3d(
            dofs, stress, stretch=lambda where, _: np.full(where.shape, stretch_value, dtype=np.complex128))
        add(f'geometric_uniform_stretch_3d_p{order}', _matrix_relative(stretched, stretch_value * geometric))
        # Green-Lagrange: a finite rotation has no strain and no internal force
        turned = (points @ (rotation - np.eye(3)).T).ravel()
        add(f'green_lagrange_rotation_3d_p{order}',
            float(np.abs(green_lagrange_strains_3d(dofs, turned, reference)).max()) / _ROTATION_ANGLE)
        stiffness = assemble_elastic_stiffness_3d(dofs, material)
        add(f'internal_force_rotation_3d_p{order}',
            float(np.abs(internal_force_3d(dofs, turned, material)).max()) / float(np.abs(stiffness @ turned).max()))
        zero = np.zeros(n_vector)
        add(f'internal_force_initial_3d_p{order}',
            max_relative_error(internal_force_3d(dofs, zero, material, initial_stress=stress),
                               -initial_stress_force_3d(dofs, stress)))
        add(f'tangent_at_rest_3d_p{order}',
            _matrix_relative(assemble_tangent_stiffness_3d(dofs, zero, material, initial_stress=stress),
                             stiffness + geometric))
        _tangent_checks(add, f'3d_p{order}',
                        lambda u: internal_force_3d(dofs, u, material, initial_stress=stress),
                        lambda u: assemble_tangent_stiffness_3d(dofs, u, material, initial_stress=stress),
                        n_vector, _LENGTH, rng)


def _check_identities_2d(add):
    from .assembly import element_areas
    from .dofs import lagrange_dofs
    from .elasticity import (CENTROID, RING, StressField, assemble_edge_traction, assemble_elastic_stiffness,
                             assemble_geometric_stiffness, assemble_stress_load, assemble_tangent_stiffness,
                             element_strains, green_lagrange_strains, initial_stress_force, internal_force,
                             material_matrix, sample_stress)

    rng = np.random.default_rng(12)
    for axisymmetric in (False, True):
        kind = 'axisymmetric' if axisymmetric else 'planar'
        offset = 1.0 if axisymmetric else 0.0      # the axisymmetric section r in [1, 2.2] L stays off the axis
        mesh = _rectangle_mesh(_LENGTH * (np.array([0.0, 0.3, 0.75, 1.2]) + offset),
                               _LENGTH * np.array([0.0, 0.45, 0.8]))
        n_elements = mesh.n_elements
        rows = 4 if axisymmetric else 3
        stress = 1e8 * rng.standard_normal((n_elements, rows))
        thermal = 1e7 * rng.standard_normal((n_elements, rows))
        coefficients = _axisymmetric_coefficients() if axisymmetric else _planar_coefficients()
        voigt = material_matrix(coefficients, np.zeros(1, dtype=np.int64), axisymmetric)[0]
        # uniform sigma0 in equilibrium (axisymmetric: sigma_rr = sigma_tt, sigma_rz = 0)
        uniform = 1e8 * (np.array([0.7, -0.4, 0.7, 0.0]) if axisymmetric else np.array([0.7, -0.4, 0.3]))
        uniform_tensor = np.array([[uniform[0], uniform[-1]], [uniform[-1], uniform[1]]])
        reference = rng.dirichlet(np.ones(3), size=5)[:, 1:]
        for order in (1, 2):
            label = f'{kind}_p{order}'
            dofs = lagrange_dofs(mesh, order)
            points = dofs.dof_points()
            n_vector = 2 * dofs.n_dofs
            measures = element_areas(dofs, axisymmetric) * (RING if axisymmetric else 1.0)
            local_stress = stress[dofs.elements]
            geometric = assemble_geometric_stiffness(dofs, stress, axisymmetric=axisymmetric)
            scale = float(abs(geometric).max())
            add(f'geometric_symmetry_{label}', _matrix_relative(geometric, geometric.T))
            axes = (1,) if axisymmetric else (0, 1)
            translation = max(float(np.abs(geometric @ np.tile(np.eye(2)[axis], dofs.n_dofs)).max()) for axis in axes)
            add(f'geometric_translation_{label}', translation / scale)
            # planar: rotation u = theta (-y, x), u K_sigma u = theta^2 int sigma_xx + sigma_yy dA; axisymmetric: radial
            # expansion u_r = e r, e^2 int sigma_rr + sigma_tt dV (the hoop term)
            rate = 0.7 / _LENGTH
            if axisymmetric:
                motion = np.column_stack([rate * points[:, 0], np.zeros(len(points))])
                weights = local_stress[:, 0] + local_stress[:, 2]
            else:
                motion = rate * np.column_stack([-points[:, 1], points[:, 0]])
                weights = local_stress[:, 0] + local_stress[:, 1]
            motion = motion.ravel()
            energy = float(motion @ (geometric @ motion))
            bound = rate ** 2 * float(np.sum(measures * np.abs(local_stress).sum(axis=1)))
            add(f'motion_energy_{label}', abs(energy - rate ** 2 * float(np.sum(measures * weights))) / bound)
            quadratic = (green_lagrange_strains(dofs, motion, CENTROID, axisymmetric=axisymmetric)
                         - element_strains(dofs, motion, CENTROID, axisymmetric=axisymmetric)[1])[:, 0]
            add(f'motion_work_{label}', abs(float(np.sum(measures[:, None] * local_stress * quadratic)) - 0.5 * energy)
                / bound)
            # initial-stress force: equilibrated uniform sigma0 = - its boundary tractions; no net force / moment
            force = initial_stress_force(dofs, np.tile(uniform, (n_elements, 1)), axisymmetric=axisymmetric)
            tractions = np.zeros(n_vector)
            for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi'):
                normal = np.zeros(2)
                normal['xy'.index(face[0])] = 1.0 if face.endswith('hi') else -1.0
                traction = uniform_tensor @ normal
                tractions += assemble_edge_traction(dofs, mesh.edge_tags[face], traction[0], traction[1],
                                                    axisymmetric=axisymmetric)
            add(f'initial_force_patch_{label}', max_relative_error(-force, tractions))
            random_force = initial_stress_force(dofs, stress, axisymmetric=axisymmetric).reshape(-1, 2)
            total = float(np.abs(random_force).sum())
            add(f'initial_force_balance_{label}', float(np.abs(random_force.sum(axis=0)[list(axes)]).max()) / total)
            if not axisymmetric:
                moment = float(np.sum(points[:, 0] * random_force[:, 1] - points[:, 1] * random_force[:, 0]))
                add(f'initial_force_moment_{label}', abs(moment) / (total * _LENGTH))
            add(f'initial_force_stress_load_{label}',
                max_relative_error(initial_stress_force(dofs, stress, axisymmetric=axisymmetric),
                                   -assemble_stress_load(dofs, stress, axisymmetric=axisymmetric)))
            # stress sampling of a field with constant strain (axisymmetric: u_r = a r, u_z = d r + e z + f)
            if axisymmetric:
                a, d, e = 1e-3, -4e-4, 6e-4
                displacement = np.column_stack([a * points[:, 0], d * points[:, 0] + e * points[:, 1] + 1e-9])
                strain = np.array([a, e, a, d])
            else:
                gradient = np.array([[1.0, 0.4], [-0.6, 0.7]]) * 1e-3
                displacement = points @ gradient.T + 1e-9
                strain = np.array([gradient[0, 0], gradient[1, 1], gradient[0, 1] + gradient[1, 0]])
            field = StressField(prescribed=stress, material=coefficients, displacement=displacement, thermal=thermal)
            expected = (local_stress[:, None, :] + (voigt @ strain)[None, None, :]
                        - thermal[dofs.elements][:, None, :])
            sampled = sample_stress(dofs, field, reference, axisymmetric=axisymmetric)
            add(f'sampled_stress_{label}', max_relative_error(sampled, expected))
            from_displacement = assemble_geometric_stiffness(
                dofs, StressField(material=coefficients, displacement=displacement), axisymmetric=axisymmetric)
            constant = assemble_geometric_stiffness(dofs, np.tile(voigt @ strain, (n_elements, 1)),
                                                    axisymmetric=axisymmetric)
            add(f'geometric_from_displacement_{label}', _matrix_relative(from_displacement, constant))
            stiffness = assemble_elastic_stiffness(dofs, coefficients, axisymmetric=axisymmetric)
            if not axisymmetric:
                angle = _ROTATION_ANGLE
                turn = np.array([[np.cos(angle) - 1.0, -np.sin(angle)], [np.sin(angle), np.cos(angle) - 1.0]])
                turned = (points @ turn.T).ravel()
                add(f'green_lagrange_rotation_{label}',
                    float(np.abs(green_lagrange_strains(dofs, turned, reference)).max()) / angle)
                add(f'internal_force_rotation_{label}',
                    float(np.abs(internal_force(dofs, turned, coefficients)).max())
                    / float(np.abs(stiffness @ turned).max()))
            zero = np.zeros(n_vector)
            add(f'internal_force_initial_{label}',
                max_relative_error(
                    internal_force(dofs, zero, coefficients, axisymmetric=axisymmetric, initial_stress=stress),
                    -initial_stress_force(dofs, stress, axisymmetric=axisymmetric)))
            add(f'tangent_at_rest_{label}',
                _matrix_relative(assemble_tangent_stiffness(dofs, zero, coefficients, axisymmetric=axisymmetric,
                                                            initial_stress=stress), stiffness + geometric))
            _tangent_checks(add, label,
                            lambda u: internal_force(dofs, u, coefficients, axisymmetric=axisymmetric,
                                                     initial_stress=stress),
                            lambda u: assemble_tangent_stiffness(dofs, u, coefficients, axisymmetric=axisymmetric,
                                                                 initial_stress=stress),
                            n_vector, _LENGTH, rng)


def _check_identities(_):
    checks, add = _recorder()
    _check_identities_3d(add)
    _check_identities_2d(add)
    return checks.result()


# ---- tensioned strings ----
def _tie(coordinates, spacing, component, components):
    """(components n, 2 cells - 1) prolongation u_component = v at the interior half-cell stations of coordinates
    (ends held): a P2 field of x (or z) only, so the 3D / 2D operators reduce to the 1D P2 string exactly."""
    station = np.rint(np.asarray(coordinates) / (0.5 * spacing)).astype(np.int64)
    n_stations = 2 * _STRING_CELLS
    prolongation = np.zeros((components * len(station), n_stations - 1))
    free = np.flatnonzero((station > 0) & (station < n_stations))
    prolongation[components * free + component, station[free] - 1] = 1.0
    return prolongation


def _frequencies(stiffness, mass, prolongation, count=3) -> np.ndarray:
    """Lowest frequencies [Hz] of the tied pencil (P^T K P, P^T M P)."""
    from scipy.linalg import eigh
    reduced_stiffness = prolongation.T @ (stiffness @ prolongation)
    reduced_mass = prolongation.T @ (mass @ prolongation)
    eigenvalues = eigh(0.5 * (reduced_stiffness + reduced_stiffness.T), 0.5 * (reduced_mass + reduced_mass.T),
                       eigvals_only=True)
    return np.sqrt(np.abs(eigenvalues[:count])) / (2.0 * np.pi)


def _string_expected(length, tension, count=3) -> np.ndarray:
    return np.arange(1, count + 1) / (2.0 * length) * np.sqrt(tension / _DENSITY)


def _check_strings(_):
    from .dofs import lagrange_dofs
    from .dofs3d import lagrange_dofs_3d
    from .elasticity import (assemble_elastic_mass, assemble_elastic_stiffness, assemble_geometric_stiffness)
    from .elasticity3d import (assemble_elastic_mass_3d, assemble_elastic_stiffness_3d,
                               assemble_geometric_stiffness_3d, isotropic_matrix_3d)
    from .mesh3d import structured_box_mesh

    checks, add = _recorder()
    spacing = _LENGTH
    length = _STRING_CELLS * spacing
    width = 0.5 * _LENGTH
    along = np.linspace(0.0, length, _STRING_CELLS + 1)
    measured = {}

    def compare(name, frequencies, tension):
        expected = _string_expected(length, tension)
        measured[name] = frequencies.tolist()
        add(name, float(np.abs(frequencies / expected - 1.0).max()), _STRING_RTOL)

    # 3D bar along x: sigma_xx alone sets the u_y string; the other stress rows must not enter
    mesh = structured_box_mesh(along, np.array([0.0, width]), np.array([0.0, width]))
    n_elements = mesh.n_elements
    dofs = lagrange_dofs_3d(mesh, 2)
    stress = np.tile(_TENSION * np.array([1.0, 0.3, -0.2, 0.1, 0.25, -0.15]), (n_elements, 1))
    geometric = assemble_geometric_stiffness_3d(dofs, stress)
    stiffness = assemble_elastic_stiffness_3d(dofs, isotropic_matrix_3d(np.full(n_elements, _LAME_LAMBDA),
                                                                        np.full(n_elements, _SHEAR)))
    mass = assemble_elastic_mass_3d(dofs, _DENSITY)
    tie = _tie(dofs.dof_points()[:, 0], spacing, 1, 3)
    compare('string_geometric_3d', _frequencies(geometric, mass, tie), _TENSION)
    compare('string_total_3d', _frequencies(stiffness + geometric, mass, tie), _TENSION + _SHEAR)   # gamma_xy = v'
    # planar strip along x, u_y = v(x)
    mesh = _rectangle_mesh(along, np.array([0.0, width]))
    n_elements = mesh.n_elements
    dofs = lagrange_dofs(mesh, 2)
    stress = np.tile(_TENSION * np.array([1.0, 0.3, 0.2]), (n_elements, 1))
    coefficients = {'c11': _LAME_LAMBDA + 2.0 * _SHEAR, 'c12': _LAME_LAMBDA, 'c22': _LAME_LAMBDA + 2.0 * _SHEAR,
                    'c66': _SHEAR}
    geometric = assemble_geometric_stiffness(dofs, stress)
    mass = assemble_elastic_mass(dofs, _DENSITY)
    tie = _tie(dofs.dof_points()[:, 0], spacing, 1, 2)
    compare('string_geometric_planar', _frequencies(geometric, mass, tie), _TENSION)
    compare('string_total_planar', _frequencies(assemble_elastic_stiffness(dofs, coefficients) + geometric, mass,
                                                tie), _TENSION + _SHEAR)
    # axisymmetric tube r in [w, 2 w] along z, u_r = v(z), sigma_zz the tension (sigma_tt = 0: no hoop term)
    mesh = _rectangle_mesh(np.array([width, 2.0 * width]), along)
    n_elements = mesh.n_elements
    dofs = lagrange_dofs(mesh, 2)
    stress = np.tile(_TENSION * np.array([0.4, 1.0, 0.0, 0.2]), (n_elements, 1))
    geometric = assemble_geometric_stiffness(dofs, stress, axisymmetric=True)
    mass = assemble_elastic_mass(dofs, _DENSITY, axisymmetric=True)
    tie = _tie(dofs.dof_points()[:, 1], spacing, 0, 2)
    compare('string_geometric_axisymmetric', _frequencies(geometric, mass, tie), _TENSION)
    return checks.result(frequencies=measured,
                         expected={'geometric': _string_expected(length, _TENSION).tolist(),
                                   'total': _string_expected(length, _TENSION + _SHEAR).tolist()})


PRESTRESS_SELFTEST_CASES = (
    SelfTestCase(
        name='fem_prestress_identities',
        label='FEM: geometric stiffness and initial-stress identities',
        description='CPU checks of the K1 prestress pieces of fem.elasticity3d (graded 3 x 2 x 2 Kuhn box) and '
                    'fem.elasticity (planar and axisymmetric structured triangles), P1 and P2: K_sigma symmetric with '
                    'translations in its null space; u K_sigma u of an infinitesimal rotation = int |w|^2 tr s - w s w '
                    'dV (planar theta^2 (s_xx + s_yy); axisymmetric radial expansion e^2 (s_rr + s_tt), the hoop term) '
                    'and = 2 x the work of s0 on the quadratic Green-Lagrange strain; initial-stress force of a '
                    'uniform equilibrated s0 = - its boundary tractions, no net force / moment for any s0, = - the '
                    'stress load; StressField sampling s0 + C eps - beta dT of a constant-strain field; K_sigma from '
                    'a displacement = from its constant stress; uniform complex stretch s: K_sigma(s) = s K_sigma; '
                    'Green-Lagrange strain and internal force of a finite rotation 0; internal force at rest = - '
                    'initial-stress force; tangent at rest = K + K_sigma(s0); central difference of the internal '
                    'force = tangent (1e-6) and tangent symmetry. Others 1e-10 (first estimates, not yet run). '
                    'error = number of failed sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_identities,
        max_wall_time=120.0),
    SelfTestCase(
        name='fem_prestress_string',
        label='FEM: tensioned-string frequencies from K + K_sigma',
        description='CPU check of the geometric stiffness against the string formula f_n = n / (2 L) sqrt(sigma / '
                    'rho), n = 1..3, on 16-cell P2 bars with held ends: a 3D Kuhn bar and a planar strip (u_y = v(x)) '
                    'and an axisymmetric tube (u_r = v(z)), the transverse field tied to a function of the axial '
                    'coordinate (so the pencil is the 1D P2 string exactly; the other stress rows must not enter). '
                    '(K_sigma, M) gives sqrt(sigma / rho); (K + K_sigma, M) with an isotropic K gives sqrt((sigma + '
                    'mu) / rho) (the shear strain v\'). 1e-3 relative (P2 dispersion ~1e-4; first estimate, not yet '
                    'run). error = number of failed sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_strings,
        max_wall_time=60.0),
)
