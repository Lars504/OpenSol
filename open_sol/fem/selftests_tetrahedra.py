"""Self-tests of the tetrahedral FEM core (igor3d_plan section 7, T1 / T2; CPU, specs() empty, check({}) on structured
Kuhn meshes, no gmsh): quadrature exactness, mesh conformity, Nedelec tangential continuity, point location and the
degree-1 reproduction of constant fields; the Lagrange assembly / face integral / interface-face patch checks. error =
number of failed sub-checks (tolerance 0), values in details."""
import math

import numpy as np

from ..model.schema import SelfTestCase
from ..model.selftest_helpers import checks_result

_LENGTH = 1e-6                     # box side [m]
_QUADRATURE_RTOL = 1e-13
_CONTINUITY_RTOL = 1e-12           # tangential jump across interior faces / max |E|
_REPRODUCTION_RTOL = 1e-12
_VOLUME_RTOL = 1e-12
_N_RANDOM_POINTS = 300
_SEED = 5


def _graded_edges(count):
    """count + 1 increasing coordinates over [0, _LENGTH], graded so cells differ (no accidental symmetry)."""
    t = np.linspace(0.0, 1.0, count + 1)
    return _LENGTH * (t + 0.15 * np.sin(math.pi * t) * t)


def _quadrature_errors():
    """{degree: largest relative error of the monomials of total degree <= degree}."""
    from .quadrature3d import TETRAHEDRON_MAX_DEGREE, tetrahedron_rule
    errors = {}
    for degree in range(1, TETRAHEDRON_MAX_DEGREE + 1):
        rule = tetrahedron_rule(degree)
        worst = 0.0
        for a in range(degree + 1):
            for b in range(degree + 1 - a):
                for c in range(degree + 1 - a - b):
                    exact = math.factorial(a) * math.factorial(b) * math.factorial(c) / math.factorial(a + b + c + 3)
                    value = float(np.sum(rule.weights * rule.points[:, 0] ** a * rule.points[:, 1] ** b
                                         * rule.points[:, 2] ** c))
                    worst = max(worst, abs(value / exact - 1.0))
        errors[degree] = worst
    return errors


def _tangential_jump(mesh, degree, coefficients):
    """Largest tangential |E_0 - E_1| on interior faces / max |E| for a Nedelec field (face quadrature points)."""
    from .dofs3d import nedelec_dofs_3d
    from .elements3d import REFERENCE_VERTICES_3D, nedelec_basis_3d, tetrahedron_geometry
    from .quadrature3d import triangle_face_rule

    dofs = nedelec_dofs_3d(mesh, degree)
    interior = np.flatnonzero(mesh.face_elements[:, 1] >= 0)
    face_nodes = mesh.faces[interior]
    face_bary, _ = triangle_face_rule(4)
    corners = mesh.points[face_nodes]
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    sides = []
    for side in range(2):
        elements = mesh.face_elements[interior, side].astype(np.int64)
        local = np.argmax(mesh.tets[elements][:, None, :] == face_nodes[:, :, None], axis=2)   # (k, 3)
        reference = np.einsum('qi,kic->kqc', face_bary, REFERENCE_VERTICES_3D[local])
        vectors, _ = nedelec_basis_3d(degree, reference)   # (k, q, n, 3)
        geometry = tetrahedron_geometry(mesh, elements)
        physical = np.einsum('kji,kqnj->kqni', geometry.inverse, vectors)
        field = np.einsum('kn,kqni->kqi', coefficients[dofs.element_dofs[elements]], physical)
        sides.append(field - np.einsum('kqi,ki->kq', field, normal)[..., None] * normal[:, None, :])
    scale = max(float(np.abs(sides[0]).max()), 1e-300)
    return float(np.abs(sides[0] - sides[1]).max()) / scale


def _check_core(_):
    from .dofs3d import nedelec_dofs_3d
    from .mesh3d import structured_box_mesh
    from .sampling3d import PointLocator3D, evaluate_nedelec_3d

    checks, details = {}, {}

    def add(name, value, tolerance):
        checks[name] = {'value': float(value), 'tolerance': tolerance, 'passed': bool(value <= tolerance)}

    for degree, error in _quadrature_errors().items():
        add(f'quadrature_degree_{degree}', error, _QUADRATURE_RTOL)
    edges = _graded_edges(3)
    mesh = structured_box_mesh(edges, edges * 0.8, edges * 1.2)
    details['mesh'] = {'nodes': mesh.n_points, 'edges': mesh.n_edges, 'faces': mesh.n_faces,
                       'elements': mesh.n_elements, 'smallest_radius_ratio': mesh.quality.smallest}
    add('euler_characteristic', abs(mesh.n_points - mesh.n_edges + mesh.n_faces - mesh.n_elements - 1), 0.0)
    box_volume = _LENGTH ** 3 * 0.8 * 1.2
    add('volume_sum', abs(float(mesh.volumes().sum()) / box_volume - 1.0), _VOLUME_RTOL)
    n_boundary = int((mesh.face_elements[:, 1] < 0).sum())
    add('outer_faces_tagged', abs(sum(len(faces) for faces in mesh.face_tags.values()) - n_boundary), 0.0)
    rng = np.random.default_rng(_SEED)
    for degree in (1, 2):
        dofs = nedelec_dofs_3d(mesh, degree)
        coefficients = rng.standard_normal(dofs.n_dofs)
        add(f'tangential_continuity_degree_{degree}', _tangential_jump(mesh, degree, coefficients),
            _CONTINUITY_RTOL)
    low, high = np.array(mesh.window[:3]), np.array(mesh.window[3:])
    points = low + (high - low) * rng.random((_N_RANDOM_POINTS, 3))
    elements, reference = PointLocator3D(mesh).locate(points)
    add('points_located', float((elements < 0).sum()), 0.0)
    dofs = nedelec_dofs_3d(mesh, 1)
    constant = np.array([0.3, -1.1, 0.7])
    along = mesh.points[mesh.edges[:, 1]] - mesh.points[mesh.edges[:, 0]]   # Whitney dof: int E . t = E . (x_hi - x_lo)
    values, curls = evaluate_nedelec_3d(dofs, along @ constant, elements, reference)
    found = elements >= 0
    add('constant_field_reproduced', float(np.abs(values[found] - constant).max()) / float(np.abs(constant).max()),
        _REPRODUCTION_RTOL)
    add('constant_field_curl_free', float(np.abs(curls[found]).max()) * _LENGTH / float(np.abs(constant).max()),
        _REPRODUCTION_RTOL)
    return checks_result(checks, **details)


_PATCH_RTOL = 1e-10                # Lagrange patch solves (spsolve on a few hundred dofs)
_INTEGRAL_RTOL = 1e-12             # areas, volumes, moments
_CONDUCTIVITY = np.array([[2.0, 0.3, -0.2], [0.3, 1.5, 0.4], [-0.2, 0.4, 1.0]])   # SPD, full
_LINEAR_GRADIENT = np.array([0.7, -1.3, 0.4]) / _LENGTH
_QUADRATIC_FORM = np.array([[1.0, 0.2, -0.3], [0.2, -0.5, 0.1], [-0.3, 0.1, 0.8]]) / _LENGTH ** 2
_OUTWARD_NORMALS = {'x_lo': (-1, 0, 0), 'x_hi': (1, 0, 0), 'y_lo': (0, -1, 0), 'y_hi': (0, 1, 0),
                    'z_lo': (0, 0, -1), 'z_hi': (0, 0, 1)}


def _linear(points):
    return 1.0 + points @ _LINEAR_GRADIENT


def _quadratic(points):
    return np.einsum('pi,ij,pj->p', points, _QUADRATIC_FORM, points) + points @ _LINEAR_GRADIENT


def _constant_tensor_at(points, elements):
    """_CONDUCTIVITY at every rule point (m, q, 3, 3): a callable, so the assembler integrates point by point."""
    return np.broadcast_to(_CONDUCTIVITY, points.shape[:2] + (3, 3))


def _sparse_relative(matrix, reference) -> float:
    """max |matrix - reference| / max |reference| of two CSRs with the same stored pattern (inf when it differs)."""
    if not (np.array_equal(matrix.indptr, reference.indptr) and np.array_equal(matrix.indices, reference.indices)):
        return math.inf
    return float(np.abs(matrix.data - reference.data).max()) / max(float(np.abs(reference.data).max()), 1e-300)


def _check_lagrange(_):
    """Lagrange P1 / P2 on a graded Kuhn box: Dirichlet and Neumann patch tests with a full tensor, face / volume
    integrals, interface and window faces, point evaluation, point loads, element integrals."""
    from scipy.sparse.linalg import spsolve

    from .assembly3d import (PATTERN_REUSE_CALLS, SplitCoefficient, assemble_face_load_3d, assemble_face_mass_3d,
                             assemble_lagrange_load_3d, assemble_lagrange_mass_3d, assemble_lagrange_stiffness_3d,
                             assemble_point_load_3d, eliminate_dirichlet, element_integrals_3d, face_areas_3d)
    from .dofs3d import lagrange_dofs_3d
    from .mesh3d import interface_faces, structured_box_mesh, window_faces
    from .sampling3d import PointLocator3D, centroid_values_3d, evaluate_lagrange_3d

    checks = {}

    def add(name, value, tolerance):
        checks[name] = {'value': float(value), 'tolerance': tolerance, 'passed': bool(value <= tolerance)}

    def relative(values, reference):
        return float(np.abs(values - reference).max()) / max(float(np.abs(reference).max()), 1e-300)

    x_edges, y_edges, z_edges = (_graded_edges(3) * scale for scale in (1.0, 0.8, 1.2))
    mesh = structured_box_mesh(x_edges, y_edges, z_edges)
    sides = (x_edges[-1], y_edges[-1], z_edges[-1])
    volume = float(np.prod(sides))
    surface = 2.0 * (sides[0] * sides[1] + sides[1] * sides[2] + sides[2] * sides[0])
    boundary = np.flatnonzero(mesh.face_elements[:, 1] < 0)
    tensor = np.broadcast_to(_CONDUCTIVITY, (mesh.n_elements, 3, 3))
    rng = np.random.default_rng(_SEED)
    points = (np.array(mesh.window[:3]) + (np.array(mesh.window[3:]) - np.array(mesh.window[:3]))
              * rng.random((_N_RANDOM_POINTS, 3)))
    elements, reference = PointLocator3D(mesh).locate(points)
    for order in (1, 2):
        dofs = lagrange_dofs_3d(mesh, order)
        dof_points = dofs.dof_points()
        exact = _quadratic(dof_points) if order == 2 else _linear(dof_points)
        source = -2.0 * float(np.sum(_CONDUCTIVITY * _QUADRATIC_FORM)) if order == 2 else 0.0   # -div(C grad u)
        stiffness = assemble_lagrange_stiffness_3d(dofs, tensor)
        fixed = dofs.face_dofs(boundary)
        system = eliminate_dirichlet(stiffness, assemble_lagrange_load_3d(dofs, source), fixed, exact[fixed])
        add(f'dirichlet_patch_p{order}', relative(system.expand(spsolve(system.matrix.tocsc(), system.rhs)), exact),
            _PATCH_RTOL)
        linear = _linear(dof_points)   # Neumann patch: x_lo held, outward flux n . C g on the other faces
        flux = np.zeros(mesh.n_faces)
        for face, normal in _OUTWARD_NORMALS.items():
            flux[mesh.face_tags[face]] = np.asarray(normal, dtype=np.float64) @ (_CONDUCTIVITY @ _LINEAR_GRADIENT)
        neumann_faces = np.setdiff1d(boundary, mesh.face_tags['x_lo'])
        held = dofs.face_dofs(mesh.face_tags['x_lo'])
        system = eliminate_dirichlet(stiffness, assemble_face_load_3d(dofs, neumann_faces, flux), held, linear[held])
        add(f'neumann_patch_p{order}', relative(system.expand(spsolve(system.matrix.tocsc(), system.rhs)), linear),
            _PATCH_RTOL)
        ones = np.ones(dofs.n_dofs)
        add(f'face_mass_area_p{order}', abs(ones @ (assemble_face_mass_3d(dofs, boundary) @ ones) / surface - 1.0),
            _INTEGRAL_RTOL)
        add(f'face_load_area_p{order}', abs(assemble_face_load_3d(dofs, boundary).sum() / surface - 1.0),
            _INTEGRAL_RTOL)
        add(f'volume_mass_p{order}', abs(ones @ (assemble_lagrange_mass_3d(dofs) @ ones) / volume - 1.0),
            _INTEGRAL_RTOL)
        integrals = element_integrals_3d(dofs, linear)
        centre = 0.5 * np.array(sides)
        add(f'element_integrals_p{order}',
            max(abs(integrals['measure'].sum() / volume - 1.0),
                abs(integrals['value'].sum() / (volume * float(_linear(centre[None, :])[0])) - 1.0),
                relative(integrals['gradient'].sum(axis=0), _LINEAR_GRADIENT * volume)), _INTEGRAL_RTOL)
        add(f'centroid_values_p{order}', relative(centroid_values_3d(dofs, linear), _linear(mesh.centroids())),
            _INTEGRAL_RTOL)
        values, gradients = evaluate_lagrange_3d(dofs, exact, elements, reference)
        expected_gradient = (points @ (_QUADRATIC_FORM + _QUADRATIC_FORM.T) + _LINEAR_GRADIENT if order == 2
                             else np.broadcast_to(_LINEAR_GRADIENT, points.shape))
        add(f'point_values_p{order}', max(relative(values, _quadratic(points) if order == 2 else _linear(points)),
                                          relative(gradients, expected_gradient)), _INTEGRAL_RTOL * 100.0)
        load = assemble_point_load_3d(dofs, elements[:5], reference[:5], np.arange(1.0, 6.0))
        add(f'point_load_moments_p{order}', max(abs(load.sum() / 15.0 - 1.0),
                                                relative(load @ dof_points, np.arange(1.0, 6.0) @ points[:5])),
            _INTEGRAL_RTOL)
        # S4 paths: the affine reference form (stiffness above) = point-by-point quadrature of the same tensor, also
        # split over every other element; a kept slot map (repeated assemblies of one element set) = COO -> CSR
        quadrature = assemble_lagrange_stiffness_3d(dofs, _constant_tensor_at)
        varying = np.arange(mesh.n_elements) % 2 == 0
        split = assemble_lagrange_stiffness_3d(dofs, SplitCoefficient(tensor, _constant_tensor_at, varying))
        add(f'affine_forms_p{order}', max(_sparse_relative(stiffness, quadrature), _sparse_relative(split, quadrature)),
            _INTEGRAL_RTOL)
        reference_mass = assemble_lagrange_mass_3d(lagrange_dofs_3d(mesh, order), 2.0)
        repeated = lagrange_dofs_3d(mesh, order)
        kept = [assemble_lagrange_mass_3d(repeated, 2.0) for _ in range(PATTERN_REUSE_CALLS + 1)][-1]
        add(f'kept_pattern_p{order}', _sparse_relative(kept, reference_mass), _INTEGRAL_RTOL)
    member = mesh.centroids()[:, 0] < x_edges[2]
    half = (x_edges[2], sides[1], sides[2])
    half_surface = 2.0 * (half[0] * half[1] + half[1] * half[2] + half[2] * half[0])
    add('interface_area', abs(face_areas_3d(mesh, interface_faces(mesh, member)).sum() / half_surface - 1.0),
        _INTEGRAL_RTOL)
    shelled = structured_box_mesh(x_edges, y_edges, z_edges,
                                  window=(x_edges[1], y_edges[0], z_edges[0], *sides))
    add('window_faces_area', max(abs(face_areas_3d(shelled, window_faces(shelled, 'x_lo')).sum()
                                     / (sides[1] * sides[2]) - 1.0),
                                 abs(face_areas_3d(shelled, window_faces(shelled, 'y_lo')).sum()
                                     / ((sides[0] - x_edges[1]) * sides[2]) - 1.0)), _INTEGRAL_RTOL)
    return checks_result(checks, elements=mesh.n_elements)


TETRAHEDRA_SELFTEST_CASES = (
    SelfTestCase(
        name='fem_tetrahedra_core',
        label='FEM tetrahedra: quadrature, conformity, Nedelec continuity',
        description='CPU checks of the tetrahedral FEM core (igor3d_plan section 7) on a graded 3 x 3 x 3 Kuhn box '
                    'mesh (162 tetrahedra, no gmsh): the reference-tetrahedron rules of degree 1-8 integrate every '
                    'monomial up to their degree (1e-13); V - E + F - T = 1, element volumes sum to the box (1e-12), '
                    'every boundary face carries one outer-face tag; random Nedelec degree 1 and 2 fields have equal '
                    'tangential traces on both sides of every interior face (1e-12: the ascending-node orientation); '
                    '300 random points are all located and a constant field interpolated by Whitney dofs is '
                    'reproduced with zero curl (1e-12). error = number of failed sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_core,
        max_wall_time=120.0),
    SelfTestCase(
        name='fem_tetrahedra_lagrange',
        label='FEM tetrahedra: Lagrange assembly, face integrals, interface faces',
        description='CPU checks of the shared Lagrange pieces on tetrahedra (fem.assembly3d / sampling3d / mesh3d) on '
                    'the graded 3 x 3 x 3 Kuhn box (no gmsh), P1 and P2: a full SPD conductivity tensor reproduces a '
                    'linear (P1) / quadratic (P2, constant source) field with Dirichlet faces and a linear field with '
                    'x_lo held and the outward flux n . C g as face loads elsewhere (1e-10); face mass / load of 1 = '
                    'the box surface, volume mass = the box volume, element integrals (volume, int u, int grad u) and '
                    'centroid '
                    'values of a linear field, values / gradients at 300 located points, point-load sum and first '
                    'moment (1e-12, point values 1e-10); the affine reference-form stiffness equals point-by-point '
                    'quadrature of the same tensor, also split over every other element (SplitCoefficient), and a '
                    'mass assembled until its slot map is kept equals COO -> CSR (pattern and values, 1e-12); '
                    'interface faces of the elements left of an inner grid plane '
                    'sum to the half-box surface, window faces of a mesh with an x_lo shell to the window face areas '
                    '(1e-12). Tolerances are first estimates, not yet run. error = number of failed sub-checks, '
                    'tolerance 0.',
        specs=lambda: {},
        check=_check_lagrange,
        max_wall_time=120.0),
)
