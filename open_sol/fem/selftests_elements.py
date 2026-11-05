"""Self-tests of the FEM core (mesh plan stage T2, CPU; specs() empty, check({}) assembles directly): Lagrange patch
tests, Nedelec / Lagrange pipe cutoffs, tensor <-> triangle transfer totals. Each case
reports error = number of failed sub-checks (tolerance 0) with the measured values in details."""
import math

import numpy as np

from ..model.schema import SelfTestCase
from ..model.selftest_helpers import SubChecks, checks_result, relative_error
from .selftests_mesher import _DISC, _DISC_SIZES, disc_mesh, disc_region_mesh

_UM = 1e-6
_LENGTH = 1.0 * _UM                 # coordinate scale of the patch fields [m]
_PATCH_H = 0.2 * _UM                # also the display and diffusion mesh and the transfer target (one cached mesh)
_EXACT_RTOL = 1e-11                 # direct solves of exactly representable fields (cond x eps)
_EVALUATION_RTOL = 1e-10            # point evaluation through located (Newton) reference coordinates
_TENSOR = np.array([[2.0, 0.5], [0.5, 1.0]])   # anisotropic conductivity of the patch tests
_POISSON_SIZES = _DISC_SIZES        # 0.12 / 0.06 um: the mesher's P2 area pair (shared cached meshes)
_ORDER_SLACK = 0.25                 # observed L2 / eigenvalue order may fall this far below the expected one
# a / h = 10 (the cached 0.06 um mesh); degree 2 eigenvalue errors ~ h^4: recorded at a / h = 16 TE11 1.1e-8, TM01
# 5.0e-8, so ~7e-8 / ~3.3e-7 expected here (x (16/10)^4 = 6.6), about 3x under _PIPE_RTOL; a straight boundary ~1e-3
_PIPE_H = _POISSON_SIZES[1]
_TE11 = 1.8411837813406593          # first zero of J1' (k_c a of TE11)
_TM01 = 2.404825557695773           # first zero of J0 (k_c a of TM01)
_PIPE_RTOL = 1e-6
_TOTAL_RTOL = 1e-12
_QUADRATURE_RTOL = 1e-3             # triangles -> triangles conservative total (non-nested meshes)


def _solve(system):
    from scipy.sparse.linalg import spsolve

    return system.expand(spsolve(system.matrix.tocsc(), system.rhs))


# ---- fem_lagrange_patch ----
def _patch_field(quadratic):
    """(u(x, y), grad u (x, y) -> (gx, gy), -div(K grad u)) of 1 + 3 X - 2 Y (+ 0.7 X^2 - 0.4 X Y + 0.9 Y^2)."""
    scale = 1.0 if quadratic else 0.0

    def value(x, y):
        X, Y = x / _LENGTH, y / _LENGTH
        return 1.0 + 3.0 * X - 2.0 * Y + scale * (0.7 * X * X - 0.4 * X * Y + 0.9 * Y * Y)

    def gradient(x, y):
        X, Y = x / _LENGTH, y / _LENGTH
        return (3.0 + scale * (1.4 * X - 0.4 * Y)) / _LENGTH, (-2.0 + scale * (-0.4 * X + 1.8 * Y)) / _LENGTH

    hessian = scale * np.array([[1.4, -0.4], [-0.4, 1.8]]) / _LENGTH ** 2
    return value, gradient, -float((_TENSOR * hessian).sum())


def _patch_solve(mesh, order, geometry_order, quadratic, neumann=False):
    """Nodal error / max |u| of -div(K grad u) = f with u exact on the outer boundary (neumann: on x_lo only, the
    flux n . K grad u on the other faces); returns (relative error, dofs, solution)."""
    from ..fem.assembly import (assemble_edge_load, assemble_load, assemble_stiffness, eliminate_dirichlet,
                                interpolate)
    from ..fem.dofs import lagrange_dofs

    value, gradient, source = _patch_field(quadratic)
    dofs = lagrange_dofs(mesh, order, geometry_order=geometry_order)
    stiffness = assemble_stiffness(dofs, np.broadcast_to(_TENSOR, (mesh.n_elements, 2, 2)))   # per-element form
    load = assemble_load(dofs, source)
    exact = interpolate(dofs, value)
    if neumann:
        fixed = dofs.edge_dofs(mesh.edge_tags['x_lo'])
        for face, normal in (('x_hi', (1.0, 0.0)), ('y_lo', (0.0, -1.0)), ('y_hi', (0.0, 1.0))):
            def flux(points, edges, normal=normal):
                gx, gy = gradient(points[..., 0], points[..., 1])
                return (normal[0] * (_TENSOR[0, 0] * gx + _TENSOR[0, 1] * gy)
                        + normal[1] * (_TENSOR[1, 0] * gx + _TENSOR[1, 1] * gy))
            load = load + assemble_edge_load(dofs, mesh.edge_tags[face], flux)
    else:
        fixed = dofs.edge_dofs(dofs.boundary_edges())
    solution = _solve(eliminate_dirichlet(stiffness, load, fixed, exact[fixed]))
    return float(np.abs(solution - exact).max() / np.abs(exact).max()), dofs, solution


def _check_patch(results_by_name):
    from ..fem.assembly import assemble_mass, element_areas, integrate
    from ..fem.dofs import lagrange_dofs
    from ..fem.sampling import evaluate

    checks = SubChecks()
    add = checks.add
    mesh = disc_mesh(_PATCH_H)
    add('mesh_has_curved_edges', bool((mesh.edge_curve >= 0).any()), int((mesh.edge_curve >= 0).sum()))
    p2_solves = {}   # (geometry order, quadratic) -> (dofs, solution), evaluated below
    for name, order, geometry_order, quadratic in (('p1_linear', 1, 1, False), ('p2_quadratic_straight', 2, 1, True),
                                                   ('p2_linear_curved', 2, 2, False)):
        error, dofs, solution = _patch_solve(mesh, order, geometry_order, quadratic)
        add(name, error <= _EXACT_RTOL, error)
        if order == 2:
            p2_solves[(geometry_order, quadratic)] = (dofs, solution)
    error, _, _ = _patch_solve(mesh, 2, 2, False, neumann=True)
    add('p2_linear_flux_faces', error <= _EXACT_RTOL, error)

    rng = np.random.default_rng(2)
    side = 2.0 * _UM
    points = rng.uniform(0.02 * side, 0.98 * side, size=(400, 2))
    for name, geometry_order, quadratic in (('evaluate_p2_quadratic_straight', 1, True),
                                            ('evaluate_p2_linear_curved', 2, False)):
        dofs, solution = p2_solves[(geometry_order, quadratic)]
        value, gradient, _ = _patch_field(quadratic)
        sampled, sampled_gradient = evaluate(dofs, solution, points, gradient=True)
        exact = value(points[:, 0], points[:, 1])
        exact_gradient = np.column_stack(gradient(points[:, 0], points[:, 1]))
        value_error = float(np.abs(sampled - exact).max() / np.abs(exact).max())
        gradient_error = float(np.abs(sampled_gradient - exact_gradient).max() / np.abs(exact_gradient).max())
        add(name, max(value_error, gradient_error) <= _EVALUATION_RTOL,
            {'value': value_error, 'gradient': gradient_error})

    window_area = side * side
    curved_areas = element_areas(lagrange_dofs(mesh, 0, geometry_order=2))
    add('areas_sum_to_window', abs(curved_areas.sum() / window_area - 1.0) <= _TOTAL_RTOL,
        float(curved_areas.sum() / window_area - 1.0))
    disc = mesh.region == mesh.region_names.index('disc')
    exact_disc = math.pi * _DISC[2] ** 2
    straight_error = abs(element_areas(lagrange_dofs(mesh, 0, disc, geometry_order=1)).sum() / exact_disc - 1.0)
    curved_error = abs(element_areas(lagrange_dofs(mesh, 0, disc, geometry_order=2)).sum() / exact_disc - 1.0)
    add('isoparametric_disc_area', curved_error <= straight_error / 50.0,
        {'straight': straight_error, 'curved': curved_error})
    p2 = lagrange_dofs(mesh, 2)
    ones = np.ones(p2.n_dofs)
    mass_area = float(ones @ (assemble_mass(p2) @ ones))
    add('mass_matrix_area', abs(mass_area / window_area - 1.0) <= _TOTAL_RTOL, mass_area / window_area - 1.0)
    ring_moment = float(integrate(lagrange_dofs(mesh, 1), np.ones(mesh.n_points), axisymmetric=True))
    exact_moment = side * side ** 2 / 2.0
    add('axisymmetric_weight', abs(ring_moment / exact_moment - 1.0) <= _TOTAL_RTOL, ring_moment / exact_moment - 1.0)
    return checks_result(checks)


# ---- fem_nedelec_pipe_cutoff ----
def _pipe_eigenvalues(mesh, family, order, geometry_order, target):
    """k_c a of the eigenvalue nearest (target / a)^2 of the PEC pipe (disc region): Nedelec curl-curl (TE) or the
    Lagrange Dirichlet Laplacian (TM)."""
    from scipy.sparse.linalg import eigsh

    from ..fem.assembly import (assemble_curl_curl, assemble_mass, assemble_stiffness, assemble_vector_mass,
                                restrict)
    from ..fem.dofs import lagrange_dofs, nedelec_dofs

    radius = _DISC[2]
    disc = mesh.region == mesh.region_names.index('disc')
    if family == 'nedelec':
        dofs = nedelec_dofs(mesh, order, disc, geometry_order)
        stiffness, mass = assemble_curl_curl(dofs), assemble_vector_mass(dofs)
    else:
        dofs = lagrange_dofs(mesh, order, disc, geometry_order)
        stiffness, mass = assemble_stiffness(dofs), assemble_mass(dofs)
    keep = np.ones(dofs.n_dofs, dtype=bool)
    keep[dofs.edge_dofs(dofs.boundary_edges())] = False
    n_modes = 2 if family == 'nedelec' else 1   # TE11 is a degenerate pair; TM01 is single (next: j11 = 3.83)
    eigenvalues = eigsh(restrict(stiffness, keep), k=n_modes, M=restrict(mass, keep),
                        sigma=(1.05 * target / radius) ** 2, which='LM', return_eigenvectors=False)
    cutoffs = np.sqrt(np.abs(eigenvalues)) * radius
    return float(cutoffs[np.argmin(np.abs(cutoffs - target))]), dofs.n_dofs


def _check_pipe(results_by_name):
    checks = SubChecks()
    add = checks.add
    fine = disc_region_mesh(_PIPE_H)
    te11, n_te = _pipe_eigenvalues(fine, 'nedelec', 2, 2, _TE11)
    te_error = abs(te11 / _TE11 - 1.0)
    add('te11_nedelec2_curved', te_error <= _PIPE_RTOL, {'k_c_a': te11, 'relative_error': te_error, 'dofs': n_te})
    tm01, n_tm = _pipe_eigenvalues(fine, 'lagrange', 2, 2, _TM01)
    tm_error = abs(tm01 / _TM01 - 1.0)
    add('tm01_lagrange2_curved', tm_error <= _PIPE_RTOL, {'k_c_a': tm01, 'relative_error': tm_error, 'dofs': n_tm})
    errors = [abs(_pipe_eigenvalues(disc_region_mesh(h), 'nedelec', 1, 1, _TE11)[0] / _TE11 - 1.0)
              for h in _POISSON_SIZES]
    observed = math.log2(errors[0] / errors[1])
    add('te11_nedelec1_order', observed >= 2.0 - _ORDER_SLACK, {'errors': errors, 'observed_order': observed})
    return checks_result(checks)


# ---- fem_transfer_totals ----
def _tensor_edges(rng, window_cells, padding, side):
    """(full-grid edges with the window from 0 to side, window slice): random window widths, padding (lo, hi)
    cells of 0.05 side."""
    widths = rng.uniform(0.5, 1.5, window_cells)
    widths *= side / widths.sum()
    pad_lo, pad_hi = padding
    all_widths = np.concatenate([np.full(pad_lo, 0.05 * side), widths, np.full(pad_hi, 0.05 * side)])
    edges = np.concatenate([[0.0], np.cumsum(all_widths)]) - 0.05 * side * pad_lo
    edges[pad_lo] = 0.0
    edges[pad_lo + window_cells] = side
    return edges, slice(pad_lo, pad_lo + window_cells)


def _window_total(values, x_edges, y_edges, window, axisymmetric=False):
    window_x, window_y = window
    edges_x, edges_y = x_edges[window_x.start:window_x.stop + 1], y_edges[window_y.start:window_y.stop + 1]
    radial = 0.5 * np.diff(edges_x ** 2) if axisymmetric else np.diff(edges_x)
    return float((values[window] * radial[:, None] * np.diff(edges_y)[None, :]).sum())


def _check_transfers(results_by_name):
    from ..fem.assembly import element_areas, integrate, interpolate
    from ..fem.dofs import lagrange_dofs
    from ..fem.mesher import MERGE_TOLERANCE
    from ..fem.transfer import tensor_to_triangles, triangles_to_tensor, triangles_to_triangles

    checks = SubChecks()
    add = checks.add
    rng = np.random.default_rng(5)
    side = 2.0 * _UM
    mesh = disc_mesh(_POISSON_SIZES[0], ring=True)
    x_edges, window_x = _tensor_edges(rng, 37, (3, 2), side)
    y_edges, window_y = _tensor_edges(rng, 29, (2, 0), side)
    window = (window_x, window_y)
    shape = (len(x_edges) - 1, len(y_edges) - 1)
    inside = mesh.pml_faces == 0
    p0 = lagrange_dofs(mesh, 0)
    areas = element_areas(p0)
    areas_r = element_areas(p0, axisymmetric=True)

    density = rng.uniform(0.5, 1.5, shape)
    for axisymmetric, measure in ((False, areas), (True, areas_r)):
        transfer = tensor_to_triangles('conservative', x_edges, y_edges, window, p0, axisymmetric)
        moved = transfer.apply(density)
        error = relative_error(float((moved * measure).sum()), _window_total(density, x_edges, y_edges, window,
                                                                             axisymmetric))
        add(f'tensor_to_triangles_total{"_axisymmetric" if axisymmetric else ""}',
            error <= _TOTAL_RTOL and not moved[~inside].any(), error)

    element_density = rng.uniform(0.5, 1.5, mesh.n_elements)
    moved = triangles_to_tensor('conservative', p0, x_edges, y_edges, window).apply(element_density)
    padding = np.ones(shape, dtype=bool)
    padding[window] = False
    error = relative_error(_window_total(moved, x_edges, y_edges, window),
                           float((element_density * areas)[inside].sum()))
    add('triangles_to_tensor_total_p0', error <= _TOTAL_RTOL and not moved[padding].any(), error)
    p2_window = lagrange_dofs(mesh, 2, inside)
    smooth = interpolate(p2_window, lambda x, y: 1.0 + np.sin(3.0 * x / side) * np.cos(2.0 * y / side))
    moved = triangles_to_tensor('conservative', p2_window, x_edges, y_edges, window).apply(smooth)
    error = relative_error(_window_total(moved, x_edges, y_edges, window), float(integrate(p2_window, smooth)))
    add('triangles_to_tensor_total_p2', error <= _TOTAL_RTOL, error)

    def linear(x, y):
        return 0.3 + 2.0 * x / side - 1.5 * y / side

    centres_x = 0.5 * (x_edges[:-1] + x_edges[1:])
    centres_y = 0.5 * (y_edges[:-1] + y_edges[1:])
    window_cx, window_cy = centres_x[window_x], centres_y[window_y]
    grid_x = np.clip(centres_x, window_cx[0], window_cx[-1])[:, None]
    grid_y = np.clip(centres_y, window_cy[0], window_cy[-1])[None, :]
    tensor_linear = linear(grid_x, grid_y) * np.ones(shape)
    p2 = lagrange_dofs(mesh, 2)
    moved = tensor_to_triangles('interpolate', x_edges, y_edges, window, p2).apply(tensor_linear)
    points = p2.dof_points()
    interior = ((points[:, 0] >= window_cx[0]) & (points[:, 0] <= window_cx[-1])
                & (points[:, 1] >= window_cy[0]) & (points[:, 1] <= window_cy[-1]))
    error = float(np.abs(moved - linear(points[:, 0], points[:, 1]))[interior].max())
    add('tensor_to_triangles_linear_exact', error <= _EVALUATION_RTOL, error)
    moved = triangles_to_tensor('interpolate', p2, x_edges, y_edges, window).apply(interpolate(p2, linear))
    error = float(np.abs(moved - tensor_linear).max())
    add('triangles_to_tensor_linear_exact', error <= _EVALUATION_RTOL, error)

    target_mesh = disc_mesh(_PATCH_H)   # 0.2 um: not nested with the 0.12 um source
    target_p0 = lagrange_dofs(target_mesh, 0)
    source_density = interpolate(p0, lambda x, y: 1.0 + 0.5 * np.sin(4.0 * x / side + 1.0) * np.cos(3.0 * y / side))
    moved = triangles_to_triangles('conservative', p0, target_p0).apply(source_density)
    error = relative_error(float((moved * element_areas(target_p0)).sum()),
                           float((source_density * areas)[inside].sum()))
    add('triangles_to_triangles_total', error <= _QUADRATURE_RTOL, error)
    target_p1 = lagrange_dofs(target_mesh, 1)
    moved = triangles_to_triangles('interpolate', p2, target_p1).apply(interpolate(p2, linear))
    inset = MERGE_TOLERANCE * side   # the transfer evaluates window-side points this far inside (keeps off the ring)
    clamped = np.clip(target_p1.dof_points(), inset, side - inset)
    error = float(np.abs(moved - linear(clamped[:, 0], clamped[:, 1])).max())
    add('triangles_to_triangles_linear_exact', error <= _EVALUATION_RTOL, error)
    moved32 = tensor_to_triangles('conservative', x_edges, y_edges, window, p0).apply(density.astype(np.float32))
    add('float32_apply', moved32.dtype == np.float32 and moved32.shape == (mesh.n_elements,), str(moved32.dtype))
    return checks_result(checks)


ELEMENT_SELFTEST_CASES = (
    SelfTestCase(
        name='fem_lagrange_patch',
        label='FEM Lagrange elements: patch tests',
        description='CPU patch tests of the FEM core (mesh plan T2) on the 2 x 2 um disc mesh (h 0.2 um, curved '
                    'interior edges): -div(K grad u) = f with an anisotropic constant K and u exact on the outer '
                    'boundary reproduces linear u with P1, quadratic u with P2 on straight elements and linear u with '
                    'isoparametric P2 (nodal error 1e-11), also with flux (edge load) faces; point evaluation of '
                    'values and gradients at 400 random points (1e-10); element areas sum to the window (1e-12), the '
                    'isoparametric disc area beats the polygon 50x, the P2 mass matrix integrates 1, the '
                    'axisymmetric weight integrates r. error = number of failed sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_patch,
        max_wall_time=300.0),
    SelfTestCase(
        name='fem_nedelec_pipe_cutoff',
        label='FEM Nedelec elements: circular pipe cutoffs',
        description='CPU: PEC circular pipe (the disc region, a = 0.6 um) at cutoff by shift-invert eigsh: TE11 '
                    'k_c a = 1.841184 from the Nedelec degree-2 curl-curl operator and TM01 2.404826 from the P2 '
                    'Dirichlet Laplacian, both isoparametric at a / h = 10 (h 0.06 um; relative 1e-6, expected errors '
                    '~7e-8 / ~3e-7 from the h^4 scaling of the a / h = 16 run, a straight boundary ~1e-3); Nedelec '
                    'degree 1 (straight elements) TE11 converges with order >= 1.75 (h 0.12 / 0.06 um). error = '
                    'number of failed sub-checks, tolerance 0. a / h 16 -> 10 on 2026-10-20 (margin 20x -> ~3x), '
                    'not yet run at that size.',
        specs=lambda: {},
        check=_check_pipe,
        max_wall_time=600.0),
    SelfTestCase(
        name='fem_transfer_totals',
        label='FEM transfers: tensor <-> triangles totals',
        description='CPU transfers between a graded tensor grid with padding and the disc mesh with a PML layer '
                    'ring: conservative tensor -> P0 (planar and axisymmetric r weights) and P0 / P2 -> tensor keep '
                    'the window integral (1e-12) and leave the ring / padding at 0; interpolation of a linear field '
                    'is exact both ways and triangles -> triangles (1e-10); triangles -> triangles conservative '
                    '(h 0.12 um source onto the non-nested h 0.2 um disc mesh) keeps the total to quadrature accuracy '
                    '(1e-3); float32 arrays stay float32. error = number of failed sub-checks, tolerance 0. Target '
                    'h 0.09 -> 0.2 um on 2026-10-20, not yet run with that target.',
        specs=lambda: {},
        check=_check_transfers,
        max_wall_time=300.0),
)
