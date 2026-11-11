"""Self-tests of fem.pml, fem.harmonic, fem.harmonic3d and fem.harmonic_resonances (new_modules_plan S1, S2, S8; CPU,
specs() empty, check({}) on structured triangle / Kuhn meshes, no gmsh): stretch values and the pml3d identity, exact
polynomial and convergent plane-wave fields, Robin / PML absorption, axisymmetric Bessel / Hankel fields, closed,
lossy and PML-terminated cavity resonances; on the GPU the cuDSS device-pencil sweep against host SuperLU. error = number
of failed sub-checks (tolerance 0), values in details."""
import math

import numpy as np

from ..model.schema import SelfTestCase
from ..model.selftest_helpers import SubChecks

# manufactured problems: complex symmetric tensors, b, damping, mass at omega, Robin admittance / damping
_TENSOR_2D = np.array([[2.0 + 0.3j, 0.4 - 0.1j], [0.4 - 0.1j, 1.2 + 0.2j]])
_TENSOR_3D = np.array([[2.0 + 0.3j, 0.4 - 0.1j, 0.2], [0.4 - 0.1j, 1.2 + 0.2j, -0.3 + 0.05j],
                       [0.2, -0.3 + 0.05j, 1.6 - 0.1j]])
_B = 0.5 + 0.2j
_DAMPING = 0.3
_MASS = 1.1
_OMEGA = 1.7                       # rad/s of the manufactured solves (unit-free problems on 1 m boxes)
_ADMITTANCE = 0.8 - 0.5j
_ROBIN_DAMPING = 0.25
_EXACT_RTOL = 1e-10                # polynomial fields the element space holds: exact through a host SuperLU solve
# waves: sound-like speed and frequency, 16 P2 cells per wavelength in 2D (12 on tetrahedra)
_SPEED = 340.0
_FREQUENCY = 1000.0
_PML_REFLECTION = 1e-6
_REFLECTION_TOLERANCE = 1e-3       # plan S2: PML reflection < 1e-3 of a 1D-like strip
# resonances: a membrane-like rectangle (c = sqrt(T / rho) = 100 m/s) and an impedance-step strip cavity
_TENSION = 1.0
_DENSITY = 1e-4
_LOSS_FACTOR = 0.02


def _rectangle_mesh(x_edges, y_edges, window=None):
    """Structured TriangleMesh of the tensor grid (each cell split along its 00-11 diagonal); cells whose centroid lies
    outside `window` form the PML layer ring (pml_faces bits by side)."""
    from .mesh import FACE_BITS, build_triangle_mesh
    x_edges, y_edges = np.asarray(x_edges, dtype=np.float64), np.asarray(y_edges, dtype=np.float64)
    n_y = len(y_edges)
    grid_x, grid_y = np.meshgrid(x_edges, y_edges, indexing='ij')
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    i, j = (index.ravel() for index in np.meshgrid(np.arange(len(x_edges) - 1), np.arange(n_y - 1), indexing='ij'))
    n00, n10, n01, n11 = i * n_y + j, (i + 1) * n_y + j, i * n_y + j + 1, (i + 1) * n_y + j + 1
    triangles = np.concatenate([np.column_stack([n00, n10, n11]), np.column_stack([n00, n11, n01])])
    window = (x_edges[0], y_edges[0], x_edges[-1], y_edges[-1]) if window is None else window
    x0, y0, x1, y1 = window
    centroids = points[triangles].mean(axis=1)
    pml_faces = np.zeros(len(triangles), dtype=np.uint8)
    for face, outside in (('x_lo', centroids[:, 0] < x0), ('x_hi', centroids[:, 0] > x1),
                          ('y_lo', centroids[:, 1] < y0), ('y_hi', centroids[:, 1] > y1)):
        pml_faces[outside] |= np.uint8(FACE_BITS[face])
    return build_triangle_mesh(points, triangles, pml_faces=pml_faces, window=tuple(float(v) for v in window))


def _graded(count, length=1.0):
    """count increasing coordinates over [0, length], smoothly graded (cells differ: no accidental symmetry)."""
    t = np.linspace(0.0, 1.0, count)
    return length * (t + 0.15 * np.sin(np.pi * t) * t)


def _checker():
    checks = SubChecks()
    return checks, checks.add_bound


# ---- S1: stretches ----
def _check_pml(_):
    from ..physics.mechanics.pml3d import elastic_pml_shell_3d
    from .mesh3d import structured_box_mesh
    from .pml import PML_GRADING_ORDER, pml_ring, pml_shell, pml_strength, ring_widths

    checks, add = _checker()
    omega = 2.0 * math.pi * _FREQUENCY
    x_edges = np.concatenate([np.linspace(0.0, 1.0, 11), np.linspace(1.0, 1.3, 7)[1:]])
    y_edges = np.concatenate([np.linspace(-0.2, 0.0, 5)[:-1], np.linspace(0.0, 0.5, 6)])
    mesh = _rectangle_mesh(x_edges, y_edges, window=(0.0, 0.0, 1.0, 0.5))
    widths = ring_widths(mesh)
    expected_widths = {'x_lo': 0.0, 'x_hi': 0.3, 'y_lo': 0.2, 'y_hi': 0.0}
    add('ring_widths', max(abs(widths[face] - expected_widths[face]) for face in widths), 1e-12)
    pml = pml_ring(mesh, _SPEED, omega, _PML_REFLECTION)
    order = PML_GRADING_ORDER
    beta_x = (order + 1) * math.log(1.0 / _PML_REFLECTION) * _SPEED / (2.0 * omega * 0.3)
    beta_y = pml_strength(0.2, _SPEED, omega, _PML_REFLECTION)
    add('strength_formula', max(abs(pml.strength['x_hi'] / beta_x - 1.0),
                                abs(beta_y * 2.0 * omega * 0.2 / ((order + 1) * math.log(1e6) * _SPEED) - 1.0)), 1e-12)
    points = np.array([[0.5, 0.25], [1.15, 0.25], [1.3, 0.25], [0.5, -0.1], [1.2, -0.15]])
    s_x = np.where(points[:, 0] > 1.0, 1.0 - 1j * beta_x * ((points[:, 0] - 1.0) / 0.3) ** order, 1.0)
    s_y = np.where(points[:, 1] < 0.0, 1.0 - 1j * beta_y * ((0.0 - points[:, 1]) / 0.2) ** order, 1.0)
    stretches = pml.stretches(points)
    add('stretch_values', max(np.abs(stretches[:, 0] - s_x).max(), np.abs(stretches[:, 1] - s_y).max()), 1e-13)
    step = 1e-6
    interior = points[[1, 3, 4]]   # inside a layer, away from its ends: central differences of x~ = s
    derivative_errors = []
    for axis in range(2):
        shifted = np.zeros(2)
        shifted[axis] = step
        difference = (pml.stretched_coordinates(interior + shifted)[:, axis]
                      - pml.stretched_coordinates(interior - shifted)[:, axis]) / (2.0 * step)
        derivative_errors.append(np.abs(difference - pml.stretches(interior)[:, axis]).max())
    add('coordinate_derivative', max(derivative_errors), 1e-6)
    tensor = pml.tensor_factor(points)
    jacobian = s_x * s_y
    add('tensor_factor', max(np.abs(tensor[:, 0, 0] - s_y / s_x).max(), np.abs(tensor[:, 1, 1] - s_x / s_y).max(),
                             np.abs(tensor[:, 0, 1] - 1.0).max()), 1e-13)
    add('volume_factor', np.abs(pml.volume_factor(points) - jacobian).max(), 1e-13)
    stretched_radius = np.where(points[:, 0] > 1.0, points[:, 0] - 1j * beta_x * 0.3 * (
        np.clip((points[:, 0] - 1.0) / 0.3, 0.0, 1.0)) ** (order + 1) / (order + 1), points[:, 0])
    add('axisymmetric_volume_factor', np.abs(pml.volume_factor(points, axisymmetric=True)
                                             - jacobian * stretched_radius / points[:, 0]).max(), 1e-13)
    add('in_layer', float(np.any(pml.in_layer(points) != np.array([False, True, True, True, True]))), 0.0)

    shell_mesh = structured_box_mesh(np.concatenate([np.linspace(0.0, 1.0, 5), [1.15, 1.3]]), np.linspace(0.0, 0.5, 3),
                                     np.concatenate([[-0.2, -0.1], np.linspace(0.0, 0.5, 3)]),
                                     window=(0.0, 0.0, 0.0, 1.0, 0.5, 0.5))
    shell = pml_shell(shell_mesh, _SPEED, omega, _PML_REFLECTION)
    add('shell_faces', float(shell.faces != ('x_hi', 'z_lo')), 0.0)
    reference = elastic_pml_shell_3d(shell_mesh, shell.faces, omega, _SPEED, _PML_REFLECTION)
    samples = np.concatenate([shell_mesh.centroids(), shell_mesh.points])
    add('pml3d_stretches', np.abs(shell.stretches(samples) - reference.stretches(samples)).max(), 1e-13)
    add('pml3d_jacobian', np.abs(shell.jacobian(samples) - reference.jacobian(samples)).max(), 1e-12)
    checks.add_raises('shell_face_mismatch_refused',
                      lambda: pml_shell(shell_mesh, _SPEED, omega, _PML_REFLECTION, faces=('x_hi',)))
    return checks.result(beta_x_hi=float(pml.strength['x_hi']), beta_y_lo=float(pml.strength['y_lo']))


# ---- S2: manufactured fields ----
def _polynomial_field(constant, linear, hessian, tensor):
    """f(points (..., d)) -> (u, grad u (..., d), -div(tensor grad u)) of u = c0 + c . x + x^T H x / 2."""
    linear, hessian = np.asarray(linear, dtype=np.complex128), np.asarray(hessian, dtype=np.complex128)
    operator = -np.sum(tensor * hessian)

    def field(points):
        points = np.asarray(points, dtype=np.float64)
        u = constant + points @ linear + 0.5 * np.einsum('...i,ij,...j->...', points, hessian, points)
        return u, linear + points @ hessian, operator * np.ones(points.shape[:-1])
    return field


def _plane_wave_field(wavevector, tensor):
    """f(points) -> (u, grad u, -div(tensor grad u)) of u = exp(j k . x)."""
    wavevector = np.asarray(wavevector, dtype=np.float64)
    factor = wavevector @ tensor @ wavevector

    def field(points):
        u = np.exp(1j * (np.asarray(points, dtype=np.float64) @ wavevector))
        return u, 1j * u[..., None] * wavevector, factor * u
    return field


def _manufactured(system, field, tensor, n_elements, dirichlet_faces, neumann_face, robin_face):
    """Set the manufactured problem of `field` on a harmonic system (Dirichlet / Neumann / Robin faces) and solve."""
    from .harmonic import HarmonicCondition
    b_total = _B + 1j * _OMEGA * _DAMPING - _OMEGA ** 2 * _MASS
    robin_total = _ADMITTANCE + 1j * _OMEGA * _ROBIN_DAMPING
    system.set_coefficients(a=np.broadcast_to(tensor, (n_elements,) + tensor.shape), b=_B, mass=_MASS,
                            damping=_DAMPING)

    def source(points, elements):
        u, _, operator = field(points)
        return operator + b_total * u

    def normal_flux(points, face):
        axis, sign = 'xyz'.index(face[0]), (1.0 if face.endswith('hi') else -1.0)
        _, gradient, _ = field(points)
        return sign * np.einsum('ij,...j->...i', tensor, gradient)[..., axis]

    system.set_source(source)
    for face in dirichlet_faces:
        system.set_face_condition(face, HarmonicCondition('dirichlet', value=lambda points: field(points)[0]))
    system.set_face_condition(neumann_face, HarmonicCondition(
        'neumann', value=lambda points, entities: normal_flux(points, neumann_face)))
    system.set_face_condition(robin_face, HarmonicCondition(
        'robin', value=lambda points, entities: normal_flux(points, robin_face) + robin_total * field(points)[0],
        admittance=_ADMITTANCE, damping=_ROBIN_DAMPING))
    system.solve(_OMEGA)
    return system


def _nodal_error(system, field) -> float:
    """max |u_h - u| / max |u| at the full-mesh dof points."""
    exact = field(system.full.dof_points())[0]
    return float(np.abs(system.values - exact).max() / np.abs(exact).max())


def _check_harmonic_2d(_):
    from .assembly import l2_error
    from .harmonic import HarmonicCondition, ScalarHarmonicSystem

    checks, add = _checker()
    faces = (('x_lo', 'y_lo'), 'x_hi', 'y_hi')
    graded = _rectangle_mesh(_graded(6), _graded(5, 0.8))
    linear = _polynomial_field(0.7 - 0.2j, [0.4 + 0.1j, -0.3 + 0.5j], np.zeros((2, 2)), _TENSOR_2D)
    quadratic = _polynomial_field(0.7 - 0.2j, [0.4 + 0.1j, -0.3 + 0.5j],
                                  [[0.6 + 0.2j, -0.25 + 0.1j], [-0.25 + 0.1j, 0.9 - 0.4j]], _TENSOR_2D)
    for order, field in ((1, linear), (2, quadratic)):
        system = _manufactured(ScalarHarmonicSystem(graded, order), field, _TENSOR_2D, graded.n_elements, *faces)
        add(f"p{order}_{'linear' if order == 1 else 'quadratic'}_exact", _nodal_error(system, field), _EXACT_RTOL)
    wave = _plane_wave_field([2.0, 1.5], _TENSOR_2D)
    errors = {}
    for order in (1, 2):
        for cells in (8, 16):
            mesh = _rectangle_mesh(np.linspace(0.0, 1.0, cells + 1), np.linspace(0.0, 1.0, cells + 1))
            system = _manufactured(ScalarHarmonicSystem(mesh, order), wave, _TENSOR_2D, mesh.n_elements, *faces)
            exact = lambda x, y: wave(np.stack([x, y], axis=-1))[0]   # noqa: E731
            errors[order, cells] = (l2_error(system.full, system.values, exact)
                                    / l2_error(system.full, np.zeros(system.full.n_dofs), exact))
        add(f'plane_wave_rate_p{order}', math.log2(errors[order, 8] / errors[order, 16]), 1.8 if order == 1 else 2.7,
            at_least=True)
    add('plane_wave_error_p2', errors[2, 16], 1e-3)
    first, second = system.operator(1.7), system.operator(3.1)
    add('operator_pattern_fixed', float(not (np.array_equal(first.indices, second.indices)
                                             and np.array_equal(first.indptr, second.indptr))), 0.0)
    for order in (1, 2):   # point loads: the P1 / P2 basis is a partition of unity
        loaded = ScalarHarmonicSystem(graded, order)
        loaded.set_face_condition('x_lo', HarmonicCondition('dirichlet', value=0.0))
        loaded.add_point_load([0.37, 0.41], 2.0 - 1.0j)
        loaded.add_point_load(graded.points[7], 0.5j)
        add(f'point_load_total_p{order}', abs(loaded.load_vector().sum() - (2.0 - 0.5j)), 1e-12)
    return checks.result(relative_l2_errors={f'p{order}_{cells}': float(value)
                                             for (order, cells), value in errors.items()})


# ---- S2: waves (Robin absorber, PML) ----
def _wave_fit(x, values, wavenumber) -> tuple:
    """(forward, backward) amplitudes of A exp(-j k x) + B exp(j k x) by least squares."""
    matrix = np.column_stack([np.exp(-1j * wavenumber * x), np.exp(1j * wavenumber * x)])
    coefficients, *_ = np.linalg.lstsq(matrix, values, rcond=None)
    return complex(coefficients[0]), complex(coefficients[1])


def _driven_strip(system, wavenumber, length):
    """Robin drive of a unit wave exp(-j k x) into x_lo (it also absorbs the returning wave), solve; (x, u) of the
    window dofs, the fitted (forward, backward)."""
    from .harmonic import HarmonicCondition
    tension = 1.0
    system.set_coefficients(a=tension, mass=tension / _SPEED ** 2)
    system.set_face_condition('x_lo', HarmonicCondition('robin', value=2j * wavenumber * tension,
                                                        damping=tension / _SPEED))
    system.solve(2.0 * math.pi * _FREQUENCY)
    x = system.full.dof_points()[:, 0]
    inside = (x >= 0.0) & (x <= length * (1.0 + 1e-12))
    forward, backward = _wave_fit(x[inside], system.values[inside], wavenumber)
    return x[inside], system.values[inside], forward, backward


def _check_harmonic_waves(_):
    from .harmonic import HarmonicCondition, ScalarHarmonicSystem
    from .pml import pml_ring

    checks, add = _checker()
    omega = 2.0 * math.pi * _FREQUENCY
    wavenumber, wavelength = omega / _SPEED, _SPEED / _FREQUENCY
    length, width = 3.0 * wavelength, wavelength / 8.0
    x_window = np.linspace(0.0, length, 49)
    y_edges = np.linspace(0.0, width, 3)
    plain = _rectangle_mesh(x_window, y_edges)
    system = ScalarHarmonicSystem(plain, 2)
    system.set_face_condition('x_hi', HarmonicCondition('robin', damping=1.0 / _SPEED))   # a / c: exact absorber
    x, values, forward, backward = _driven_strip(system, wavenumber, length)
    add('robin_field_error', np.abs(values - np.exp(-1j * wavenumber * x)).max(), 2e-3)
    add('robin_reflection', abs(backward) / abs(forward), 1e-4)
    ringed = _rectangle_mesh(np.concatenate([x_window, length + wavelength * np.arange(1, 17) / 16.0]), y_edges,
                             window=(0.0, 0.0, length, width))
    pml = pml_ring(ringed, _SPEED, omega, _PML_REFLECTION)
    _, _, forward, backward = _driven_strip(ScalarHarmonicSystem(ringed, 2, pml=pml), wavenumber, length)
    add('pml_reflection', abs(backward) / abs(forward), _REFLECTION_TOLERANCE)
    add('pml_incident_amplitude', abs(forward - 1.0), 2e-3)
    return checks.result(pml_reflection=abs(backward) / abs(forward), beta=float(pml.strength['x_hi']))


# ---- S2: axisymmetric ----
def _check_harmonic_axisymmetric(_):
    from scipy.special import hankel2, jv

    from .assembly import l2_error
    from .harmonic import HarmonicCondition, ScalarHarmonicSystem
    from .pml import pml_ring

    checks, add = _checker()
    tension = 1.3 + 0.2j
    radial, axial = 3.1 + 0.4j, 1.2   # u = J0(kappa r) exp(-j k_z z): -div(a grad u) - omega^2 m u = 0
    omega = 1.0
    mass = tension * (radial ** 2 + axial ** 2) / omega ** 2

    def bessel(points):
        points = np.asarray(points, dtype=np.float64)
        return jv(0, radial * points[..., 0]) * np.exp(-1j * axial * points[..., 1])

    errors = {}
    for order in (1, 2):
        for cells in (8, 16):
            mesh = _rectangle_mesh(np.linspace(0.0, 1.0, cells + 1), np.linspace(0.0, 1.0, cells + 1))
            system = ScalarHarmonicSystem(mesh, order, axisymmetric=True)
            system.set_coefficients(a=tension, mass=mass)
            for face in ('x_hi', 'y_lo', 'y_hi'):   # the axis (x_lo) is natural
                system.set_face_condition(face, HarmonicCondition('dirichlet', value=bessel))
            system.solve(omega)
            exact = lambda x, y: bessel(np.stack([x, y], axis=-1))   # noqa: E731
            errors[order, cells] = (l2_error(system.full, system.values, exact)
                                    / l2_error(system.full, np.zeros(system.full.n_dofs), exact))
        add(f'bessel_rate_p{order}', math.log2(errors[order, 8] / errors[order, 16]), 1.8 if order == 1 else 2.7,
            at_least=True)
    add('bessel_error_p2', errors[2, 16], 1e-3)
    # outgoing cylindrical wave H0^(2)(k r) between r0 and R, radial PML beyond R (the r~ / r weight)
    frequency_omega = 2.0 * math.pi * _FREQUENCY
    wavenumber, wavelength = frequency_omega / _SPEED, _SPEED / _FREQUENCY
    inner, outer, height = 0.5 * wavelength, 2.5 * wavelength, wavelength / 8.0
    x_edges = np.concatenate([np.linspace(inner, outer, 33), outer + wavelength * np.arange(1, 17) / 16.0])
    mesh = _rectangle_mesh(x_edges, np.linspace(0.0, height, 3), window=(inner, 0.0, outer, height))
    pml = pml_ring(mesh, _SPEED, frequency_omega, _PML_REFLECTION)
    system = ScalarHarmonicSystem(mesh, 2, axisymmetric=True, pml=pml)
    system.set_coefficients(a=1.0, mass=1.0 / _SPEED ** 2)
    system.set_face_condition('x_lo', HarmonicCondition('dirichlet', value=complex(hankel2(0, wavenumber * inner))))
    system.solve(frequency_omega)
    points = system.full.dof_points()
    inside = points[:, 0] <= outer * (1.0 + 1e-12)
    exact = hankel2(0, wavenumber * points[inside, 0])
    add('axisymmetric_pml_hankel', np.abs(system.values[inside] - exact).max() / np.abs(exact).max(), 5e-3)
    return checks.result(relative_l2_errors={f'p{order}_{cells}': float(value)
                                             for (order, cells), value in errors.items()})


# ---- S2: tetrahedra ----
def _check_harmonic_3d(_):
    from .harmonic3d import ScalarHarmonicSystem3D
    from .mesh3d import structured_box_mesh
    from .pml import pml_shell

    checks, add = _checker()
    box = structured_box_mesh(_graded(4), _graded(4, 0.8), _graded(4, 0.9))
    faces = (('x_lo', 'y_lo', 'z_lo', 'z_hi'), 'x_hi', 'y_hi')
    linear = _polynomial_field(0.7 - 0.2j, [0.4 + 0.1j, -0.3 + 0.5j, 0.2 - 0.1j], np.zeros((3, 3)), _TENSOR_3D)
    quadratic = _polynomial_field(0.7 - 0.2j, [0.4 + 0.1j, -0.3 + 0.5j, 0.2 - 0.1j],
                                  [[0.6 + 0.2j, -0.25 + 0.1j, 0.15], [-0.25 + 0.1j, 0.9 - 0.4j, 0.3j],
                                   [0.15, 0.3j, -0.5 + 0.2j]], _TENSOR_3D)
    for order, field in ((1, linear), (2, quadratic)):
        system = _manufactured(ScalarHarmonicSystem3D(box, order), field, _TENSOR_3D, box.n_elements, *faces)
        add(f"p{order}_{'linear' if order == 1 else 'quadratic'}_exact_3d", _nodal_error(system, field), _EXACT_RTOL)
    omega = 2.0 * math.pi * _FREQUENCY
    wavenumber, wavelength = omega / _SPEED, _SPEED / _FREQUENCY
    length, width = 3.0 * wavelength, wavelength / 8.0
    x_edges = np.concatenate([np.linspace(0.0, length, 37), length + wavelength * np.arange(1, 13) / 12.0])
    strip = structured_box_mesh(x_edges, [0.0, width], [0.0, width], window=(0.0, 0.0, 0.0, length, width, width))
    pml = pml_shell(strip, _SPEED, omega, _PML_REFLECTION)
    _, _, forward, backward = _driven_strip(ScalarHarmonicSystem3D(strip, 2, pml=pml), wavenumber, length)
    add('pml_reflection_3d', abs(backward) / abs(forward), _REFLECTION_TOLERANCE)
    add('pml_incident_amplitude_3d', abs(forward - 1.0), 2e-3)
    return checks.result(elements=box.n_elements, pml_reflection=abs(backward) / abs(forward))


# ---- S8: resonances ----
def _check_resonances(_):
    from .harmonic import HarmonicCondition, ScalarHarmonicSystem
    from .harmonic_resonances import solve_system_resonances
    from .pml import pml_ring

    checks, add = _checker()
    speed = math.sqrt(_TENSION / _DENSITY)
    length_x, length_y, target = 1.0, 0.7, 150.0
    mesh = _rectangle_mesh(np.linspace(0.0, length_x, 31), np.linspace(0.0, length_y, 22))

    def clamped(tension):
        system = ScalarHarmonicSystem(mesh, 2)
        system.set_coefficients(a=tension, mass=_DENSITY)
        for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi'):
            system.set_face_condition(face, HarmonicCondition('dirichlet', value=0.0))
        return system

    exact = sorted((0.5 * speed * math.hypot(m / length_x, n / length_y) for m in range(1, 9) for n in range(1, 9)),
                   key=lambda frequency: abs(frequency - target))
    closed = solve_system_resonances(clamped(_TENSION), target, 4)
    found = np.array([resonance.frequency for resonance in closed.resonances])
    add('closed_count', 4 - len(found), 0.0)
    add('closed_frequencies', max((np.min(np.abs(found - frequency)) / frequency for frequency in exact[:4]),
                                  default=math.inf), 1e-4)
    add('closed_real_pencil', float(closed.info['pencil'] != 'real_symmetric'), 0.0)
    add('closed_lossless', max((abs(resonance.frequency_imag) for resonance in closed.resonances), default=0.0), 0.0)
    lossy = solve_system_resonances(clamped(_TENSION * (1.0 + 1j * _LOSS_FACTOR)), target, 4)
    scale = complex(np.sqrt(1.0 + 1j * _LOSS_FACTOR))   # K (1 + j eta): lambda (1 + j eta) exactly
    expected = np.array([resonance.omega for resonance in closed.resonances]) * scale
    lossy_omegas = np.array([resonance.omega for resonance in lossy.resonances])
    add('lossy_omegas', max((np.min(np.abs(lossy_omegas - value)) / abs(value) for value in expected),
                            default=math.inf), 1e-8)
    exact_q = scale.real / (2.0 * scale.imag)
    add('lossy_q_factor', max((abs(resonance.q_factor / exact_q - 1.0) for resonance in lossy.resonances),
                              default=math.inf), 1e-6)
    # impedance-step cavity: x in [0, L1] (a1, m1), [L1, 1.5 L1] (a2, m2), PML beyond; x_lo held. Outgoing beyond L1:
    # tan(k L1) = j r, r = sqrt(a1 m1 / (a2 m2)), so omega_n = (c / L1) (n pi + j atanh(r)) (equal speeds c)
    cavity, ratio = 1.0, 5.0
    x_edges = np.concatenate([np.linspace(0.0, cavity, 41), np.linspace(cavity, 1.5 * cavity, 21)[1:],
                              1.5 * cavity + 0.5 * cavity * np.arange(1, 17) / 16.0])
    strip = _rectangle_mesh(x_edges, np.linspace(0.0, 0.1, 3), window=(0.0, 0.0, 1.5 * cavity, 0.1))
    outside = strip.centroids()[:, 0] > cavity
    pml = pml_ring(strip, speed, 2.0 * math.pi * target, _PML_REFLECTION)
    system = ScalarHarmonicSystem(strip, 2, pml=pml)
    system.set_coefficients(a=np.where(outside, ratio * _TENSION, _TENSION),
                            mass=np.where(outside, ratio * _DENSITY, _DENSITY))
    system.set_face_condition('x_lo', HarmonicCondition('dirichlet', value=0.0))
    leaky = solve_system_resonances(system, target, 5)
    leaky_omegas = np.array([resonance.omega for resonance in leaky.resonances])
    damping = math.atanh(1.0 / ratio)   # r = sqrt(a1 m1 / (a2 m2)) = 1 / ratio
    closed_form = {n: speed / cavity * complex(n * math.pi, damping) for n in (2, 3, 4)}
    add('leaky_omegas', max((np.min(np.abs(leaky_omegas - value)) / abs(value) if leaky_omegas.size else math.inf)
                            for value in closed_form.values()), 1e-3)
    nearest = leaky.resonances[int(np.argmin(np.abs(leaky_omegas - closed_form[3])))] if leaky_omegas.size else None
    add('leaky_q_factor', abs(nearest.q_factor / (3.0 * math.pi / (2.0 * damping)) - 1.0) if nearest else math.inf,
        1e-2)
    add('leaky_complex_pencil', float(leaky.info['pencil'] != 'complex_symmetric'), 0.0)
    return checks.result(closed=closed.summary(), lossy=lossy.summary(), leaky=leaky.summary(),
                         leaky_omegas=[[float(value.real), float(value.imag)] for value in leaky_omegas],
                         closed_form={n: [value.real, value.imag] for n, value in closed_form.items()})


# ---- cuDSS device pencil (sparse_direct.DevicePencil) ----
_PENCIL_FREQUENCIES = (900.0, 950.0, 1000.0, 1050.0, 1100.0)   # Hz around the PML reference _FREQUENCY
_PENCIL_RTOL = 1e-8                # relative field difference to the host SuperLU solve, double and mixed
_ASYMMETRIC_TENSOR = np.array([[2.0 + 0.1j, 0.4], [-0.1, 1.2 + 0.05j]])   # a != a^T: the general LU


def _pencil_square(cells_per_wavelength):
    """(mesh, pml) of a 1.5 x 1.5 wavelength square with a one-wavelength fem.pml ring on x_hi / y_hi."""
    from .pml import pml_ring
    wavelength = _SPEED / _FREQUENCY
    length = 1.5 * wavelength
    window = np.linspace(0.0, length, int(round(1.5 * cells_per_wavelength)) + 1)
    ring = length + wavelength * np.arange(1, 17) / 16.0
    mesh = _rectangle_mesh(np.concatenate([window, ring]), np.concatenate([window, ring]),
                           window=(0.0, 0.0, length, length))
    return mesh, pml_ring(mesh, _SPEED, 2.0 * math.pi * _FREQUENCY, _PML_REFLECTION)


def _pencil_system(mesh, pml, linear_solver, a=1.0):
    """P2 ScalarHarmonicSystem on the square: a Robin wave drive into x_lo, PML outer sides held."""
    from .harmonic import HarmonicCondition, ScalarHarmonicSystem
    wavenumber = 2.0 * math.pi * _FREQUENCY / _SPEED
    system = ScalarHarmonicSystem(mesh, 2, linear_solver, pml=pml)
    system.set_coefficients(a=a, mass=1.0 / _SPEED ** 2)
    system.set_face_condition('x_lo', HarmonicCondition('robin', value=2j * wavenumber, damping=1.0 / _SPEED))
    return system


def _check_device_pencil(_):
    import time

    from ..core.backend import gpu_available, to_host
    from ..solvers.cudss_factor import cudss_available
    from ..solvers.sparse_direct import CUDSS_BACK_ENDS, SparseDirectSolver
    from .harmonic import HarmonicCondition

    checks, add = _checker()
    if not (gpu_available() and cudss_available()):
        add('skipped', 0.0, 0.0)
        return checks.result(skipped='no usable GPU or nvmath-python (cuDSS) not installed')
    mesh, pml = _pencil_square(64)
    reference = _pencil_system(mesh, pml, None)   # host SuperLU
    omegas = [2.0 * math.pi * frequency for frequency in _PENCIL_FREQUENCIES]
    references = [reference.solve(omega).copy() for omega in omegas]
    add('superlu_device_values_none', float(reference.device_values() is not None), 0.0)   # R6: host solves keep none
    timing = {}
    for back_end in CUDSS_BACK_ENDS:
        runtime = SparseDirectSolver(back_end=back_end, backend='gpu')
        system = _pencil_system(mesh, pml, runtime)
        device_seconds = []   # factorise + drive solve (cudss_mixed: the solve is the deferred accuracy check)
        for omega, expected in zip(omegas, references):
            started = time.perf_counter()
            values = system.solve(omega)
            device_seconds.append(time.perf_counter() - started)
            add(f'{back_end}_{omega / (2.0 * math.pi):.0f}_hz', np.linalg.norm(values - expected)
                / np.linalg.norm(expected), _PENCIL_RTOL)
        kept = system.device_values()   # R6: the last direct solve's full values on the device = the host values
        add(f'{back_end}_device_values', math.inf if kept is None else
            float(np.max(np.abs(to_host(kept) - values)) / np.max(np.abs(values))), 1e-14)
        device, _ = system._device_terms()
        add(f'{back_end}_device_pencil_used', float(device is None), 0.0)
        add(f'{back_end}_pencil_symmetric', float(device is None or not device.symmetric), 0.0)
        add(f'{back_end}_ldlt', float(not system.factor.symmetric), 0.0)
        # a re-assembly (new mass) keeps the pattern: a new DevicePencil, the same cuDSS plan
        system.set_coefficients(a=1.0, mass=1.1 / _SPEED ** 2)
        reference.set_coefficients(a=1.0, mass=1.1 / _SPEED ** 2)
        expected = reference.solve(omegas[0]).copy()
        add(f'{back_end}_reassembled', np.linalg.norm(system.solve(omegas[0]) - expected) / np.linalg.norm(expected),
            _PENCIL_RTOL)
        reference.set_coefficients(a=1.0, mass=1.0 / _SPEED ** 2)
        counters = runtime.work_counters()
        add(f'{back_end}_one_plan', abs(counters['cudss_plans'] - 1), 0.0)
        # the host path of the same factor (before DevicePencil): host A(omega), canonicalise, asymmetry, lower take
        host_seconds = []
        drive = system.load_vector()[system.pencil.free]
        for omega in omegas:
            started = time.perf_counter()
            factor = runtime.factorize_host(system.operator(omega), symmetric=True, previous=system.factor)
            factor.solve_host(drive)
            host_seconds.append(time.perf_counter() - started)
        add(f'{back_end}_host_path_one_plan', abs(runtime.work_counters()['cudss_plans'] - 1), 0.0)
        factor.release()
        # first device factorisation planned; the host loop reused that plan throughout
        device_ms, host_ms = 1e3 * float(np.median(device_seconds[1:])), 1e3 * float(np.median(host_seconds))
        timing[back_end] = {'unknowns': int(system.pencil.free.size), 'entries': int(system.pencil.free_pattern[0].size),
                            'device_path_ms_per_frequency': device_ms, 'host_path_ms_per_frequency': host_ms,
                            'speedup': host_ms / device_ms if device_ms > 0.0 else None}
    # a non-symmetric tensor a: the DevicePencil finds the terms unsymmetric and cuDSS factors the general LU
    small, small_pml = _pencil_square(16)
    tensor = np.broadcast_to(_ASYMMETRIC_TENSOR, (small.n_elements, 2, 2))
    asymmetric_reference = _pencil_system(small, small_pml, None, a=tensor)
    runtime = SparseDirectSolver(back_end='cudss', backend='gpu')
    asymmetric = _pencil_system(small, small_pml, runtime, a=tensor)
    for system in (asymmetric_reference, asymmetric):
        system.set_face_condition('y_lo', HarmonicCondition('dirichlet', 0.5))
    expected = asymmetric_reference.solve(omegas[2]).copy()
    add('asymmetric_field', np.linalg.norm(asymmetric.solve(omegas[2]) - expected) / np.linalg.norm(expected),
        _PENCIL_RTOL)
    device, _ = asymmetric._device_terms()
    add('asymmetric_pencil_detected', float(device is None or device.symmetric), 0.0)
    add('asymmetric_general_lu', float(asymmetric.factor.symmetric), 0.0)
    asymmetric.factor.release()
    return checks.result(timing=timing)


HARMONIC_SELFTEST_CASES = (
    SelfTestCase(
        name='fem_pml_stretch',
        label='FEM: physics-neutral PML stretches',
        description='CPU checks of fem.pml (no model): the layer widths of a ringed triangle mesh (x_hi 0.3, y_lo 0.2), '
                    'beta = 3 ln(1 / R) c / (2 omega w), s = 1 - j beta (d / w)^2 at points in the window, the layers '
                    'and a corner, d x~ / dx = s by central differences, the scalar-operator factors (a: s_y / s_x, '
                    's_x / s_y; b: s_x s_y; axisymmetric x r~ / r), the layer flags, and on a Kuhn box with an x_hi / '
                    'z_lo shell the stretches and Jacobian of physics/mechanics/pml3d (the same map) and the refusal '
                    'of a face list that disagrees with the shell. Exact identities at 1e-12 / 1e-13 (differences '
                    '1e-6). error = number of failed sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_pml,
        max_wall_time=60.0),
    SelfTestCase(
        name='fem_harmonic_manufactured',
        label='FEM triangles: complex harmonic system, manufactured fields',
        description='CPU checks of fem.harmonic.ScalarHarmonicSystem (host SuperLU) with a complex symmetric tensor a, '
                    'complex b, damping and mass at omega = 1.7 rad/s, Dirichlet x_lo / y_lo, Neumann x_hi and Robin '
                    '(complex admittance + damping) y_hi faces: a linear field exact on P1 and a quadratic one on P2 '
                    '(graded structured mesh, 1e-10 at the dofs); the plane wave exp(j (2 x + 1.5 y)) on 8 / 16 cell '
                    'unit squares converges at L2 rate >= 1.8 (P1) / >= 2.7 (P2) with P2 error <= 1e-3 at 16; the '
                    'reduced operator keeps one pattern across frequencies; point loads total their values (partition '
                    'of unity, 1e-12). error = number of failed sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_harmonic_2d,
        max_wall_time=120.0),
    SelfTestCase(
        name='fem_harmonic_waves',
        label='FEM triangles: Robin absorber and PML reflection',
        description='CPU checks of fem.harmonic on a 1D-like strip (c = 340 m/s, 1 kHz, 3 wavelengths, 16 P2 cells per '
                    'wavelength): a Robin drive at x_lo (value 2 j k a, damping a / c) launches exp(-j k x); a Robin '
                    'x_hi absorber (damping a / c, exact at normal incidence) gives the field within 2e-3 and a fitted '
                    'reflection |B / A| <= 1e-4; a one-wavelength fem.pml ring (16 cells, R = 1e-6 at the drive '
                    'frequency, outer side held) reflects <= 1e-3 (plan S2) with |A - 1| <= 2e-3. error = number of '
                    'failed sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_harmonic_waves,
        max_wall_time=120.0),
    SelfTestCase(
        name='fem_harmonic_axisymmetric',
        label='FEM triangles: axisymmetric harmonic fields',
        description='CPU checks of fem.harmonic axisymmetric (r, z) solves: u = J0(kappa r) exp(-j k_z z) with complex '
                    'a and kappa (Dirichlet r = R, z faces; natural axis) converges at L2 rate >= 1.8 (P1) / >= 2.7 '
                    '(P2) on 8 / 16 cell squares with P2 error <= 1e-3; the outgoing wave H0^(2)(k r) from r0 = '
                    'lambda / 2 to 2.5 lambda into a radial fem.pml ring (the stretched-radius weight r~ / r) matches '
                    'within 5e-3 of its peak (P2, 16 cells per wavelength). error = number of failed sub-checks, '
                    'tolerance 0.',
        specs=lambda: {},
        check=_check_harmonic_axisymmetric,
        max_wall_time=120.0),
    SelfTestCase(
        name='fem_harmonic3d_identities',
        label='FEM tetrahedra: complex harmonic system',
        description='CPU checks of fem.harmonic3d.ScalarHarmonicSystem3D (host SuperLU) on a graded 3 x 3 x 3 Kuhn box: '
                    'a full complex symmetric 3 x 3 tensor, complex b, damping and mass, Dirichlet x_lo / y_lo / z_lo / '
                    'z_hi, Neumann x_hi and Robin y_hi faces: a linear field exact on P1, a quadratic one on P2 (1e-10 '
                    'at the dofs); a 1D-like Kuhn strip (12 P2 cells per wavelength, Robin drive at x_lo) into a '
                    'one-wavelength fem.pml shell reflects <= 1e-3 with |A - 1| <= 2e-3. error = number of failed '
                    'sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_harmonic_3d,
        max_wall_time=120.0),
    SelfTestCase(
        name='fem_harmonic_resonances',
        label='FEM: neutral complex resonance solve',
        description='CPU checks of fem.harmonic_resonances on fem.harmonic pencils (P2): a clamped 1 x 0.7 m rectangle '
                    '(c = 100 m/s) gives the 4 closed-form f_mn = c / 2 sqrt((m / Lx)^2 + (n / Ly)^2) nearest 150 Hz '
                    'within 1e-4, real pencil, lossless; with a (1 + 0.02 j) the same omegas x sqrt(1 + 0.02 j) '
                    '(1e-8) and Q (1e-6); an impedance-step strip cavity (r = 0.2 between x = 1 m and a PML-terminated '
                    'outside) gives the leaky omega_n = (c / L) (n pi + j atanh r), n = 2, 3, 4 within 1e-3 and Q_3 = '
                    '3 pi / (2 atanh r) within 1e-2, PML modes dropped by their energy share. error = number of failed '
                    'sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_resonances,
        max_wall_time=120.0),
    SelfTestCase(
        name='fem_harmonic_cudss_device_pencil',
        label='FEM triangles: harmonic sweep factorised from device pencil values (cuDSS)',
        description='GPU checks of fem.harmonic on a cuDSS back end (sparse_direct.DevicePencil: K, C, M values kept on '
                    'the device, each frequency combined, equilibrated and refactorised there): a 1.5 x 1.5 wavelength '
                    'P2 square (64 cells per wavelength, ~5e4 unknowns) with a one-wavelength PML ring on x_hi / y_hi '
                    'and a Robin wave drive at x_lo, solved at 900 - 1100 Hz by cudss and cudss_mixed, each field '
                    f'within {_PENCIL_RTOL:g} of the host SuperLU solve; the pencil found symmetric (LDL^T), one cuDSS '
                    'plan for every frequency and a re-assembly with a new mass; a non-symmetric tensor a on a 16-cell '
                    'square (Dirichlet y_lo) found unsymmetric and solved by the general LU; device_values() of the '
                    'last cuDSS solve equal to the host values (1e-14 relative), None after a SuperLU solve (R6). '
                    'Details: median ms per '
                    'frequency (factorise + drive solve; cudss_mixed checks its reused-plan factors by that solve) of '
                    'the device path (system.solve, host assembly of the rhs included) and of the former host path '
                    '(factorize_host of the host A(omega) on the same plan, probe included, + solve_host) and their '
                    'ratio (machine-dependent, not judged). Passes as skipped without a GPU or '
                    'nvmath-python. error = number of failed sub-checks, tolerance 0. Written, not run.',
        specs=lambda: {},
        check=_check_device_pencil,
        max_wall_time=180.0,
        backends=('gpu',),
        heavy=True),
)
