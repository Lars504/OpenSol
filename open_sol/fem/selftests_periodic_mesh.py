"""Self-tests of periodic / Bloch triangle meshes (new_modules_plan G3; CPU, specs() empty, check({}) meshes directly):
matching opposite window sides (built-in mesher, mirror + periodic, gmsh when installed), P1 / P2 periodic dof
partners and Bloch masters, and the P2 Bloch Laplacian band of a square cell vs |k + G|^2. Written, not run:
tolerances are first estimates."""
import math

import numpy as np

from ..model.schema import SelfTestCase
from ..model.selftest_helpers import SubChecks, checks_result
from ..model.spec import DomainSpec, GeometrySpec, MeshSpec, ModelSpec

_UM = 1e-6
_CELL = 1.0 * _UM               # square cell side a [m]
_H = 0.1 * _UM                  # maximum element size [m]
_GROWTH = 1.3
_FACES = ('x_lo', 'x_hi', 'y_lo', 'y_hi')
_NODE_TOLERANCE = 1e-12         # x a: partner displacement minus the period
_PHASE_TOLERANCE = 1e-12
_HERMITIAN_TOLERANCE = 1e-12    # x the largest reduced-operator entry
_BAND_TOLERANCE = 2e-3          # |lambda_h - |k + G|^2| / (2 pi / a)^2, P2, h = a / 10 with a refined disc
_N_BANDS = 6
_K_POINTS = (('gamma', 0.0, 0.0), ('x', 1.0, 0.0), ('m', 1.0, 1.0), ('generic', 0.3, 0.17))   # (k_x, k_y) x pi / a


def _geometry():
    """An off-centre refined disc (finer side sampling near x_hi), a bar on x_lo only and a point on y_hi only: the
    window sides start unmatched."""
    um = _UM
    return [GeometrySpec('disc', 'circle', {'cx': 0.62 * um, 'cy': 0.41 * um, 'r': 0.2 * um}),
            GeometrySpec('bar', 'rectangle', {'x0': 0.0, 'y0': 0.23 * um, 'x1': 0.3 * um, 'y1': 0.37 * um}),
            GeometrySpec('spot', 'point', {'x': 0.55 * um, 'y': 1.0 * um})]


def _spec() -> ModelSpec:
    spec = ModelSpec(title='Self-test: periodic triangle mesh', domain=DomainSpec(size_x=_CELL, size_y=_CELL))
    spec.geometry = _geometry()
    spec.meshes = [MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=_H)]
    return spec


def _sizing(periodic_axes, mesher='builtin', mirror_axes=(), ring=0.0):
    """Explicit meshing.TriangleSizing: h _H, the disc at h / 3, a ring of `ring` m on every face."""
    from ..model.meshing import TriangleSizing

    return TriangleSizing(h_max=_H, h_min=_H / 100.0, growth=_GROWTH, curvature_factor=0.3,
                          object_sizes={'disc': _H / 3.0}, pml_widths={face: ring for face in _FACES}, pml_layers=4,
                          mesher=mesher, periodic_axes=tuple(periodic_axes), mirror_axes=tuple(mirror_axes))


def _mesh(periodic_axes, **options):
    from ..model.triangle_meshing import triangulate

    return triangulate(_spec(), {}, sizing=_sizing(periodic_axes, **options))


def _raises(function) -> bool:
    try:
        function()
    except ValueError:
        return True
    return False


def _partner_checks(mesh, label, axes, add):
    """Per periodic axis: node partners displaced by exactly the period, P2 dof partners (nodes + side edges) displaced
    alike; a P1 map on the x > a / 2 elements only refused (its partners lie outside the subset)."""
    from .dofs import lagrange_dofs, periodic_dof_partners
    from .mesh import periodic_node_partners

    add(f'{label}_periodic_axes', tuple(mesh.periodic_axes) == tuple(axes), list(mesh.periodic_axes))
    quadratic = lagrange_dofs(mesh, 2)
    dof_points = quadratic.dof_points()
    for axis in (0, 1):
        name = 'xy'[axis]
        if name not in axes:
            continue
        period = np.zeros(2)
        period[axis] = _CELL
        partners = periodic_node_partners(mesh, axis)
        hi = np.flatnonzero(partners >= 0)
        offset = float(np.abs(mesh.points[hi] - mesh.points[partners[hi]] - period).max()) if len(hi) else math.inf
        add(f'{label}_{name}_node_partners', len(hi) > 1 and offset <= _NODE_TOLERANCE * _CELL,
            {'pairs': int(len(hi)), 'offset': offset})
        dof_partners = periodic_dof_partners(quadratic, partners)
        paired = np.flatnonzero(dof_partners >= 0)
        dof_offset = (float(np.abs(dof_points[paired] - dof_points[dof_partners[paired]] - period).max())
                      if len(paired) else math.inf)
        add(f'{label}_{name}_p2_partners', len(paired) == 2 * len(hi) - 1 and dof_offset <= _NODE_TOLERANCE * _CELL,
            {'pairs': int(len(paired)), 'expected': int(2 * len(hi) - 1), 'offset': dof_offset})
    right_half = mesh.centroids()[:, 0] > 0.5 * _CELL
    if 'x' in axes:
        add(f'{label}_subset_refused', _raises(lambda: periodic_dof_partners(
            lagrange_dofs(mesh, 1, elements=right_half), periodic_node_partners(mesh, 0))))


def _bloch_corner_checks(mesh, add):
    """P1 Bloch masters with both axes: the top-right corner stands for the bottom-left one with both phases, an x_hi
    node for its x_lo partner with e^{-j k_x a}; no master lies on a hi side."""
    from .dofs import bloch_masters, lagrange_dofs
    from .mesh import periodic_node_partners

    k_x, k_y = 0.3 * math.pi / _CELL, 0.17 * math.pi / _CELL
    dofs = lagrange_dofs(mesh, 1)   # full map: dof = node
    master, phase = bloch_masters(dofs, {0: k_x, 1: k_y})
    points = mesh.points
    corner = int(np.argmin(np.abs(points - _CELL).sum(axis=1)))
    origin = int(np.argmin(np.abs(points).sum(axis=1)))
    corner_ok = (master[corner] == origin
                 and abs(phase[corner] - np.exp(-1j * (k_x + k_y) * _CELL)) <= _PHASE_TOLERANCE)
    add('bloch_corner_master', corner_ok, {'master': int(master[corner]), 'origin': origin,
                                          'phase': [float(phase[corner].real), float(phase[corner].imag)]})
    partners_x, partners_y = periodic_node_partners(mesh, 0), periodic_node_partners(mesh, 1)
    side = np.flatnonzero((partners_x >= 0) & (partners_y < 0))
    side_error = float(np.abs(phase[side] - np.exp(-1j * k_x * _CELL)).max()) if len(side) else math.inf
    add('bloch_side_phase', len(side) > 0 and side_error <= _PHASE_TOLERANCE
        and bool(np.array_equal(master[side], partners_x[side])), side_error)
    add('bloch_masters_on_lo_sides', not bool(((partners_x[master] >= 0) | (partners_y[master] >= 0)).any()))


def _check_periodic_mesh(results_by_name):
    """Every sub-check pass / fail; error = number of failed sub-checks, tolerance 0."""
    from .gmsh_io import find_gmsh
    from .mesh import periodic_node_partners

    checks = SubChecks()
    add = checks.add

    plain = _mesh(())
    add('unmatched_mesh_refused', _raises(lambda: periodic_node_partners(plain, 0)))
    both = _mesh(('x', 'y'))
    _partner_checks(both, 'builtin_xy', ('x', 'y'), add)
    _bloch_corner_checks(both, add)
    _partner_checks(_mesh(('x',)), 'builtin_x', ('x',), add)
    _partner_checks(_mesh(('x', 'y'), mirror_axes=('x',)), 'mirror_x_periodic_xy', ('x', 'y'), add)
    add('ring_on_periodic_axis_refused', _raises(lambda: _mesh(('x',), ring=0.2 * _UM)))
    notes = []
    if find_gmsh() is None:
        notes.append('gmsh not found: the gmsh variant is skipped')
    else:
        _partner_checks(_mesh(('x', 'y'), mesher='gmsh'), 'gmsh_xy', ('x', 'y'), add)
    return checks_result(checks, notes=notes, elements=int(both.n_elements))


# ---- Bloch Laplacian band of the square cell ----
def _exact_band(k_x, k_y) -> np.ndarray:
    """The _N_BANDS smallest |k + G|^2 [1/m^2] over the square reciprocal lattice G = 2 pi / a (m, n)."""
    orders = np.arange(-3, 4)
    g_x, g_y = np.meshgrid(2.0 * math.pi / _CELL * orders, 2.0 * math.pi / _CELL * orders, indexing='ij')
    return np.sort(((k_x + g_x) ** 2 + (k_y + g_y) ** 2).ravel())[:_N_BANDS]


def _band(stiffness, mass, dofs, k_x, k_y) -> tuple:
    """(the _N_BANDS smallest eigenvalues of P^H K P x = lambda P^H M P x, Hermitian defect of P^H K P / its largest
    entry) of the Bloch-reduced P2 Laplacian."""
    import scipy.sparse.linalg
    from .dofs import bloch_masters, bloch_prolongation

    master, phase = bloch_masters(dofs, {0: k_x, 1: k_y})
    _, prolongation = bloch_prolongation(master, phase)
    adjoint = prolongation.conj().T.tocsr()
    reduced_stiffness = (adjoint @ stiffness @ prolongation).tocsc()
    reduced_mass = (adjoint @ mass @ prolongation).tocsc()
    largest = float(abs(reduced_stiffness).max())
    defect = float(abs(reduced_stiffness - reduced_stiffness.conj().T).max()) / largest if largest else math.inf
    shift = -0.1 * (2.0 * math.pi / _CELL) ** 2   # below the band: K - shift M is definite
    values = scipy.sparse.linalg.eigsh(reduced_stiffness, k=_N_BANDS, M=reduced_mass, sigma=shift, which='LM',
                                       return_eigenvectors=False)
    return np.sort(np.real(values)), defect


def _check_bloch_band(results_by_name):
    """Worst |lambda_h - |k + G|^2| / (2 pi / a)^2 over the k points and the _N_BANDS lowest bands <= _BAND_TOLERANCE;
    every reduced stiffness Hermitian to _HERMITIAN_TOLERANCE (else the error is inf)."""
    from .assembly import assemble_mass, assemble_stiffness
    from .dofs import lagrange_dofs

    mesh = _mesh(('x', 'y'))
    dofs = lagrange_dofs(mesh, 2)
    stiffness, mass = assemble_stiffness(dofs), assemble_mass(dofs)
    unit = (2.0 * math.pi / _CELL) ** 2
    details, worst, worst_defect = {}, 0.0, 0.0
    for name, fraction_x, fraction_y in _K_POINTS:
        k_x, k_y = fraction_x * math.pi / _CELL, fraction_y * math.pi / _CELL
        values, defect = _band(stiffness, mass, dofs, k_x, k_y)
        exact = _exact_band(k_x, k_y)
        error = float(np.abs(values - exact).max()) / unit
        worst, worst_defect = max(worst, error), max(worst_defect, defect)
        details[name] = {'computed': (values / unit).tolist(), 'exact': (exact / unit).tolist(), 'error': error,
                         'hermitian_defect': defect}
    measured = worst if worst_defect <= _HERMITIAN_TOLERANCE else math.inf
    return {'measured': measured, 'expected': 0.0, 'error': measured, 'tolerance': _BAND_TOLERANCE,
            'details': {'k_points': details, 'dofs': int(dofs.n_dofs), 'elements': int(mesh.n_elements),
                        'units': 'eigenvalues / (2 pi / a)^2'}}


PERIODIC_MESH_SELFTEST_CASES = (
    SelfTestCase(
        name='fem_periodic_triangle_mesh',
        label='Periodic triangle mesh: matching sides, partners, Bloch masters',
        description=(
            'CPU checks of TriangleSizing.periodic_axes (new_modules_plan G3: fem.mesher.periodic_pslg + paired '
            'window-side splits, gmsh Transfinite side lines): a 1 um square cell, h 0.1 um, an off-centre disc '
            'refined to h / 3, a bar on x_lo only and a point on y_hi only (the sides start unmatched). Unperiodic '
            'mesh: periodic_node_partners refuses it. Periodic x + y, x only, and mirror_x + periodic x + y: node and '
            'P2 dof partners (nodes + side edges) displaced by exactly the period (1e-12 a), a P1 map on the right '
            'half refused; Bloch masters of a generic k: the top-right corner stands for the origin with e^{-j (k_x '
            '+ k_y) a}, x_hi nodes for their x_lo partners with e^{-j k_x a}, no master on a hi side; an absorbing '
            'ring on a periodic axis refused. gmsh variant skipped (noted) without gmsh. error = failed sub-checks, '
            'tolerance 0. Written, not run.'),
        specs=lambda: {},
        check=_check_periodic_mesh,
        max_wall_time=300.0),
    SelfTestCase(
        name='fem_bloch_laplacian_band',
        label='Periodic triangle mesh: Bloch Laplacian band vs |k + G|^2',
        description=(
            'CPU: the fem_periodic_triangle_mesh cell (periodic x + y, built-in mesher), P2 stiffness / mass of the '
            'Laplacian (homogeneous: the disc and bar only shape the mesh), reduced by fem.dofs.bloch_masters / '
            'bloch_prolongation (P^H K P, P^H M P) at Gamma, X, M and k = (0.3, 0.17) pi / a; the 6 lowest '
            'eigenvalues (scipy eigsh shift-invert below the band) vs the 6 smallest |k + G|^2: worst error / (2 pi / '
            f'a)^2 <= {_BAND_TOLERANCE:g} (first estimate), reduced stiffness Hermitian to '
            f'{_HERMITIAN_TOLERANCE:g}. Written, not run.'),
        specs=lambda: {},
        check=_check_bloch_band,
        max_wall_time=300.0),
)
