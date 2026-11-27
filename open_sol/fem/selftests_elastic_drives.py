"""Self-tests of fem.elastic_drives (CPU, specs() empty, check({}) on the shared disc meshes): the stress drive is the
transpose of the strain operator (planar, axisymmetric m = 0 / 2, P1 / P2), constant stresses and tractions equal the
elasticity loads, the order-2 drive equals a phi quadrature of the 3D work for both parities, and the mixed u-p
projection of the mean normal stress (identity on P1, blind to ker B, the mass solve). error = number of failed
sub-checks (tolerance 0)."""
import math

import numpy as np

from ..model.schema import SelfTestCase
from ..model.selftest_helpers import SubChecks, no_runs, random_complex
from .selftests_mesher import _DISC_SIZES, disc_mesh

_UM = 1e-6
_TRANSPOSE_RTOL = 1e-13   # w^T u vs the point sum of sigma . eps(u) (same rule points: round-off)
_LOAD_RTOL = 1e-13        # constant stress / traction vs assemble_stress_load / assemble_edge_traction (exact rules)
_PHI_RTOL = 1e-12         # order-2 drive vs the 64-point phi quadrature (exact for the trigonometric degree 4)
_PHI_POINTS = 64
_FORMS = (('planar', False, 0), ('axisymmetric', True, 0), ('azimuthal_2', True, 2))   # (name, axisymmetric, order)
_SELECTION_X = 1.2 * _UM   # drive selection: elements with centroid x below this (part of the disc and the window)
_STRESS_SCALE = 1e6        # Pa
_DISPLACEMENT_SCALE = 1e-9   # m
_PROJECTION_RTOL = 1e-10   # projected vs plain drive of a P1 mean stress (CG residual 1e-13 x condition 4: round-off)
# |w^T u| / sum |w_i u_i| of the projected drive on u with B u = 0: round-off x cond(B B^T) of the kernel projection
# (the ring weights near the axis raise it; first estimate)
_KERNEL_RTOL = 1e-9
_KERNEL_SEEN = 1e-6        # the plain drive's reading of that u must exceed this share (first estimate: ~1e-2)


def _mesh():
    return disc_mesh(_DISC_SIZES[0])


def _selection(mesh) -> np.ndarray:
    return mesh.points[mesh.triangles].mean(axis=1)[:, 0] < _SELECTION_X


def _relative_to_terms(value, expected, terms) -> float:
    """|value - expected| / sum |terms| (no cancellation in the scale)."""
    scale = float(np.sum(np.abs(terms)))
    return abs(value - expected) / scale if scale > 0.0 else abs(value - expected)


def _point_strains(dofs, displacement, rule, positions, axisymmetric, order) -> np.ndarray:
    """(m, q, k) strains of the map's elements at positions at a rule (fem.elasticity's own strain evaluation)."""
    from .elasticity import azimuthal_element_strains, element_strains
    if order:
        return azimuthal_element_strains(dofs, displacement, rule.points, order)[1][positions]
    return element_strains(dofs, displacement, rule.points, axisymmetric=axisymmetric)[1][positions]


def _check_transpose(_):
    from .dofs import lagrange_dofs
    from .elastic_drives import assemble_stress_drive, drive_points

    checks = SubChecks()
    mesh = _mesh()
    rng = np.random.default_rng(41)
    for element_order in (1, 2):
        dofs = lagrange_dofs(mesh, element_order)
        for name, axisymmetric, order in _FORMS:
            points = drive_points(dofs, axisymmetric=axisymmetric, azimuthal_order=order, elements=_selection(mesh))
            stress = random_complex(rng, (points.rows, points.n_points), _STRESS_SCALE)
            displacement = _DISPLACEMENT_SCALE * rng.standard_normal((dofs.n_dofs, points.components))
            drive = assemble_stress_drive(dofs, points, stress)
            terms = []
            for index, (positions, rule) in enumerate(points.groups):
                start, stop = int(points.offsets[index]), int(points.offsets[index + 1])
                strains = _point_strains(dofs, displacement, rule, positions, axisymmetric, order)
                strains = strains.reshape(-1, points.rows)
                terms.append(points.weights[start:stop] * np.einsum('kp,pk->p', stress[:, start:stop], strains))
            terms = np.concatenate(terms)
            measured = complex(drive @ displacement.reshape(-1))
            checks.add_bound(f'transpose_p{element_order}_{name}', _relative_to_terms(measured, terms.sum(), terms),
                             _TRANSPOSE_RTOL)
        checks.add_raises(f'empty_selection_raises_p{element_order}',
                          lambda dofs=dofs: drive_points(dofs, elements=np.zeros(mesh.n_elements, dtype=bool)))
    return checks.result()


def _check_constant_loads(_):
    from .dofs import lagrange_dofs
    from .elasticity import assemble_edge_traction, assemble_stress_load
    from .elastic_drives import assemble_edge_drive, assemble_stress_drive, drive_points, edge_drive_points

    checks = SubChecks()
    mesh = _mesh()
    rng = np.random.default_rng(43)
    traction = (3.0e5, -2.0e5)
    for element_order in (1, 2):
        dofs = lagrange_dofs(mesh, element_order)
        edges = dofs.boundary_edges()   # the window's straight outer edges: both rules exact
        for name, axisymmetric, _ in _FORMS[:2]:
            points = drive_points(dofs, axisymmetric=axisymmetric)
            per_element = _STRESS_SCALE * rng.standard_normal((mesh.n_elements, points.rows))
            drive = assemble_stress_drive(dofs, points, per_element[points.elements].T)
            load = assemble_stress_load(dofs, per_element, axisymmetric=axisymmetric)
            checks.add_bound(f'stress_load_p{element_order}_{name}',
                             float(np.max(np.abs(drive - load))) / float(np.max(np.abs(load))), _LOAD_RTOL)
            edge_points = edge_drive_points(dofs, edges, axisymmetric=axisymmetric)
            values = np.broadcast_to(np.array(traction), edge_points.weights.shape + (2,))
            edge_drive = assemble_edge_drive(dofs, edge_points, values)
            edge_load = assemble_edge_traction(dofs, edges, traction[0], traction[1], axisymmetric=axisymmetric)
            checks.add_bound(f'edge_traction_p{element_order}_{name}',
                             float(np.max(np.abs(edge_drive - edge_load))) / float(np.max(np.abs(edge_load))),
                             _LOAD_RTOL)
            unit = np.einsum('kqc,kqc->kq', edge_points.normals, edge_points.normals)
            checks.add_bound(f'edge_normals_unit_p{element_order}_{name}', float(np.max(np.abs(unit - 1.0))), 1e-12)
        checks.add_raises(f'other_map_raises_p{element_order}',
                          lambda points=points, other=lagrange_dofs(mesh, 3 - element_order): assemble_stress_drive(
                              other, points, np.zeros((points.rows, points.n_points))))
    return checks.result()


def _phi_work(values, gradients, radius, stress, weights, odd) -> complex:
    """sum over points and _PHI_POINTS angles of the 3D work sigma . S of an order-2 field in cylindrical calculus:
    even u = (U_r cos, U_z cos, U_phi sin), stress normal rows x cos, shear rows x sin; odd u = (U_r sin, U_z sin,
    -U_phi cos), stress normal x sin, shear x cos (values (m, q, 3), gradients (m, q, 3, 2) d U_c / d (r, z), stress
    (m, q, 6) solid rows, weights w det J r (m, q))."""
    order = 2
    total = 0.0 + 0.0j
    u_r, u_z, u_phi = values[..., 0], values[..., 1], values[..., 2]
    for index in range(_PHI_POINTS):
        phi = 2.0 * math.pi * index / _PHI_POINTS
        c, s = math.cos(order * phi), math.sin(order * phi)
        if odd:   # (radial / axial factor, its phi derivative), (azimuthal factor, its phi derivative)
            a, da, b, db = s, order * c, -c, order * s
        else:
            a, da, b, db = c, -order * s, s, order * c
        s_rr = gradients[..., 0, 0] * a
        s_zz = gradients[..., 1, 1] * a
        s_tt = (u_r * a + u_phi * db) / radius
        g_rz = (gradients[..., 0, 1] + gradients[..., 1, 0]) * a
        g_zt = gradients[..., 2, 1] * b + u_z * da / radius
        g_rt = u_r * da / radius + gradients[..., 2, 0] * b - u_phi * b / radius
        normal_factor, shear_factor = (s, c) if odd else (c, s)
        density = (normal_factor * (stress[..., 0] * s_rr + stress[..., 1] * s_zz + stress[..., 2] * s_tt
                                    + stress[..., 3] * g_rz)
                   + shear_factor * (stress[..., 4] * g_zt + stress[..., 5] * g_rt))
        total += complex(np.sum(weights * density)) * (2.0 * math.pi / _PHI_POINTS)
    return total


def _check_azimuthal_quadrature(_):
    from .dofs import lagrange_dofs
    from .elasticity import AZIMUTHAL_RING
    from .elastic_drives import assemble_stress_drive, drive_points
    from .elements import element_geometry, lagrange_basis, physical_gradients

    checks = SubChecks()
    mesh = _mesh()
    rng = np.random.default_rng(47)
    for element_order in (1, 2):
        dofs = lagrange_dofs(mesh, element_order)
        points = drive_points(dofs, axisymmetric=True, azimuthal_order=2, elements=_selection(mesh))
        stress = random_complex(rng, (points.rows, points.n_points), _STRESS_SCALE)
        amplitudes = _DISPLACEMENT_SCALE * rng.standard_normal((dofs.n_dofs, 3))
        flipped = stress.copy()
        flipped[4:] *= -1.0   # the odd parity in the even convention: shear rows negated
        expected = {'even': 0.0j, 'odd': 0.0j}
        for index, (positions, rule) in enumerate(points.groups):
            start, stop = int(points.offsets[index]), int(points.offsets[index + 1])
            geometry = element_geometry(mesh, dofs.elements[positions], rule.points, dofs.geometry_order)
            basis, reference_gradients = lagrange_basis(dofs.order, rule.points)
            gradients = physical_gradients(reference_gradients, geometry.inverse)        # (m, q, n, 2)
            local = amplitudes[dofs.element_dofs[positions]]                             # (m, n, 3)
            values = np.einsum('qn,mnc->mqc', basis, local)
            derivatives = np.einsum('mqnd,mnc->mqcd', gradients, local)
            shape = values.shape[:2]
            point_stress = stress[:, start:stop].T.reshape(shape + (6,))
            weights = points.weights[start:stop].reshape(shape) / AZIMUTHAL_RING   # w det J r: the phi sum adds 2 pi
            radius = geometry.points[..., 0]
            for parity in ('even', 'odd'):
                expected[parity] += _phi_work(values, derivatives, radius, point_stress, weights, parity == 'odd')
        for parity, rows in (('even', stress), ('odd', flipped)):
            measured = complex(assemble_stress_drive(dofs, points, rows) @ amplitudes.reshape(-1))
            checks.add_bound(f'phi_quadrature_p{element_order}_{parity}',
                             abs(measured - expected[parity]) / abs(expected[parity]), _PHI_RTOL,
                             measured=str(measured), expected=str(expected[parity]))
    return checks.result()


def _p1_at_points(dofs, points, nodal) -> np.ndarray:
    """(n_points,) values of a continuous P1 nodal field (mesh nodes) at the DrivePoints (the pressure basis)."""
    from .elements import lagrange_basis
    values = []
    for positions, rule in points.groups:
        corners = dofs.mesh.triangles[dofs.elements[positions]]
        values.append((nodal[corners] @ lagrange_basis(1, rule.points)[0].T).reshape(-1))
    return np.concatenate(values)


def _check_pressure_projection(_):
    import scipy.sparse.linalg

    from .dofs import lagrange_dofs
    from .elastic_drives import (PROJECTION_TOLERANCE, PressureProjection, _pressure_mass_solve,
                                 assemble_stress_drive, drive_points)
    from .mixed_elasticity import assemble_pressure_terms, pressure_space

    checks = SubChecks()
    mesh = _mesh()
    rng = np.random.default_rng(53)
    dofs = lagrange_dofs(mesh, 2, geometry_order=1)   # straight elements: drive, B and M_p rules all exact
    centroids_x = mesh.points[mesh.triangles].mean(axis=1)[:, 0]
    two_groups = np.where(centroids_x < _SELECTION_X, 0, 1).astype(np.int64)   # disjoint pressure blocks
    for name, axisymmetric, order in _FORMS:
        space = pressure_space(dofs, two_groups, 2)
        coupling, masses = assemble_pressure_terms(dofs, space, axisymmetric=axisymmetric, azimuthal_order=order)
        projection = PressureProjection(space=space, coupling=coupling, masses=tuple(masses),
                                        free=np.ones(space.n_dofs, dtype=bool))
        points = drive_points(dofs, axisymmetric=axisymmetric, azimuthal_order=order)
        normal = 3 if axisymmetric else 2
        # a P1 mean normal stress is its own projection: the projected drive equals the plain one
        stress = random_complex(rng, (points.rows, points.n_points), _STRESS_SCALE)
        stress[:normal] -= stress[:normal].mean(axis=0)
        stress[:normal] += _p1_at_points(dofs, points, random_complex(rng, (mesh.n_points,), _STRESS_SCALE))
        plain = assemble_stress_drive(dofs, points, stress)
        projected = assemble_stress_drive(dofs, points, stress, projection=projection)
        checks.add_bound(f'p1_mean_identity_{name}',
                         float(np.max(np.abs(projected - plain))) / float(np.max(np.abs(plain))), _PROJECTION_RTOL)
        # a rough mean stress: the drive is B^T pi with M_p pi = int N_p s dV, so it reads nothing of a u with B u = 0
        rough = np.zeros((points.rows, points.n_points), dtype=np.complex128)
        rough[:normal] = random_complex(rng, (points.n_points,), _STRESS_SCALE)
        displacement = _DISPLACEMENT_SCALE * rng.standard_normal(coupling.shape[1])
        gram = (coupling @ coupling.T).tocsc()
        kernel = displacement - coupling.T @ scipy.sparse.linalg.spsolve(gram, coupling @ displacement)
        for label, drive in (('projected', assemble_stress_drive(dofs, points, rough, projection=projection)),
                             ('plain', assemble_stress_drive(dofs, points, rough))):
            reading = _relative_to_terms(complex(drive @ kernel), 0.0, drive * kernel)
            if label == 'projected':
                checks.add_bound(f'kernel_reading_{label}_{name}', reading, _KERNEL_RTOL)
            else:   # the unprojected drive must see the bulk-free modes, else the check above proves nothing
                checks.add_bound(f'kernel_reading_{label}_{name}_seen', _KERNEL_SEEN / max(reading, 1e-300), 1.0,
                                 reading=reading)
        # the mass solve on the free dofs (some held): M_p pi = load there, pi = 0 on the held ones
        free = rng.random(space.n_dofs) > 0.1
        held_projection = PressureProjection(space=space, coupling=coupling, masses=tuple(masses), free=free)
        load = random_complex(rng, (space.n_dofs,), 1.0)
        solved = _pressure_mass_solve(held_projection, load)
        residual = (sum(mass @ solved for mass in masses) - load)[free]
        checks.add_bound(f'mass_solve_residual_{name}', float(np.linalg.norm(residual))
                         / float(np.linalg.norm(load[free])), 10.0 * PROJECTION_TOLERANCE)
        checks.add_bound(f'mass_solve_held_zero_{name}', float(np.max(np.abs(solved[~free]), initial=0.0)), 0.0)
    ungrouped = np.where(centroids_x < _SELECTION_X, -1, 0).astype(np.int64)
    space = pressure_space(dofs, ungrouped, 1)
    coupling, masses = assemble_pressure_terms(dofs, space)
    projection = PressureProjection(space=space, coupling=coupling, masses=tuple(masses),
                                    free=np.ones(space.n_dofs, dtype=bool))
    points = drive_points(dofs)
    checks.add_raises('drive_outside_pressure_groups_raises', lambda: assemble_stress_drive(
        dofs, points, np.ones((points.rows, points.n_points)), projection=projection))
    return checks.result()


ELASTIC_DRIVES_SELFTEST_CASES = (
    SelfTestCase(
        name='fem_elastic_drive_transpose',
        label='FEM: complex stress drive = transpose of the strain operator',
        description='CPU check of fem.elastic_drives on the 2 x 2 um disc mesh (h 0.12 um), P1 and P2, planar, '
                    'axisymmetric m = 0 and azimuthal order 2, a drive selection of part of the mesh: w^T u of '
                    'assemble_stress_drive equals the sum over the drive points of weight x sigma . eps(u) with eps '
                    'from fem.elasticity element_strains / azimuthal_element_strains at the same rule points (complex '
                    'random stress, real random u; 1e-13 of the summed term magnitudes: round-off); an empty '
                    'selection raises. error = number of failed sub-checks, tolerance 0.',
        specs=no_runs,
        check=_check_transpose,
        max_wall_time=60.0),
    SelfTestCase(
        name='fem_elastic_drive_constant_stress',
        label='FEM: constant stress / traction drives equal the elasticity loads',
        description='CPU check on the disc mesh, P1 and P2, planar and axisymmetric: a real stress constant per '
                    'element (random) assembled as a point drive equals elasticity.assemble_stress_load (the same or '
                    'exact rules, 1e-13), and a constant traction on the window edges as an edge drive equals '
                    'assemble_edge_traction (straight edges, exact rules, 1e-13); edge normals are unit; points of '
                    'another dof map raise. error = number of failed sub-checks, tolerance 0.',
        specs=no_runs,
        check=_check_constant_loads,
        max_wall_time=60.0),
    SelfTestCase(
        name='fem_elastic_drive_azimuthal_quadrature',
        label='FEM: order-2 drive vs a phi quadrature of the 3D work',
        description='CPU check on the disc mesh, P1 and P2, azimuthal order 2: w^T U of a random complex stress '
                    'drive equals the 64-point phi quadrature of int sigma . S dV with S from the 3D cylindrical '
                    'strains of u = (U_r cos 2 phi, U_z cos 2 phi, U_phi sin 2 phi) and normal stress rows x cos, '
                    'shear rows x sin (pins the pi r ring weight and the rows); the odd parity (u = (U_r sin, U_z sin, '
                    '-U_phi cos), normal x sin, shear x cos) equals the even drive with the shear rows negated '
                    '(1e-12; the trapezoid in phi is exact). error = number of failed sub-checks, tolerance 0.',
        specs=no_runs,
        check=_check_azimuthal_quadrature,
        max_wall_time=60.0),
    SelfTestCase(
        name='fem_elastic_drive_pressure_projection',
        label='FEM: mixed u-p stress drive with its mean normal stress projected onto the pressure space',
        description='CPU check of assemble_stress_drive(projection=) on the disc mesh, P2 on straight elements, two '
                    'pressure groups (disjoint M_p blocks), planar, axisymmetric m = 0 and azimuthal order 2: a mean '
                    'normal stress that is a P1 field (random deviatoric rows on top) gives the plain drive (1e-10: CG '
                    'residual 1e-13 x condition 4); a rough mean stress reads nothing of a displacement with B u = 0 '
                    '(the bulk-free modes of the mixed form; 1e-9 of sum |w_i u_i|) while the plain drive reads them '
                    '(> 1e-6: the check bites); the Jacobi-CG mass solve meets M_p pi = load on the free dofs (10 x '
                    'its 1e-13 tolerance) with pi = 0 on held ones; drive elements outside every pressure group '
                    'raise. error = number of failed sub-checks, tolerance 0.',
        specs=no_runs,
        check=_check_pressure_projection,
        max_wall_time=60.0),
)
