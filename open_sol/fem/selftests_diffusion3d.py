"""Self-test of fem.diffusion3d (FEMDiffusionSystem3D; CPU, specs() empty, check({}) on structured Kuhn meshes, no gmsh):
linear fields with Dirichlet / Robin faces (P1, P2) incl. the PML shell extension, face reactions, point and sheet
source balances, one implicit-Euler loss step. error = number of failed sub-checks (tolerance 0), values in details."""
import numpy as np

from ..model.schema import SelfTestCase
from ..model.selftest_helpers import checks_result

_LENGTH = 1e-6                     # box side [m]
_RTOL = 1e-10                      # exact identities through a host SuperLU solve
_SHELL_RTOL = 1e-8                 # shell dofs read the field 1e-9 x side inside the window (mesh3d.PLANE_TOLERANCE)
_CONDUCTIVITY = (2.0, 0.7, 1.3)    # k_xx, k_yy, k_zz
_LOSS = 4e12                       # a [W/(m^3 K)]: (a + c / dt) h^2 / k ~ 1 on the ~0.3 um cells (well conditioned)
_CAPACITY_OVER_DT = 12e12          # c / dt [W/(m^3 K)]
_ROBIN_H = 3e6                     # W/(m^2 K) on the 1 um box: h L / k ~ 1.5
_POINT_POWER = 2.5e-3
_SHEET_FLUX = 4e5


def _box_mesh():
    """Graded 3 x 3 x 3 Kuhn box with one shell cell beyond the z_hi window face."""
    from .mesh3d import structured_box_mesh
    t = np.linspace(0.0, 1.0, 4)
    edges = _LENGTH * (t + 0.15 * np.sin(np.pi * t) * t)
    z_edges = np.append(edges, edges[-1] + 0.4 * _LENGTH)
    return structured_box_mesh(edges, edges, z_edges, window=(0.0, 0.0, 0.0, edges[-1], edges[-1], edges[-1]))


def _check_diffusion3d(_):
    from .diffusion import EdgeCondition
    from .diffusion3d import FEMDiffusionSystem3D, shell_dof_locations
    from .mesh3d import interface_faces
    from .sampling3d import PointLocator3D

    checks = {}

    def add(name, value, tolerance=_RTOL):
        checks[name] = {'value': float(value), 'tolerance': tolerance, 'passed': bool(value <= tolerance)}

    mesh = _box_mesh()
    side = mesh.window[3]
    tensor = np.broadcast_to(np.diag(_CONDUCTIVITY), (mesh.n_elements, 3, 3))
    window_faces_all = ('x_lo', 'x_hi', 'y_lo', 'y_hi', 'z_lo', 'z_hi')
    for order in (1, 2):
        system = FEMDiffusionSystem3D(mesh, order)
        points = system.full.dof_points()
        in_shell = np.zeros(system.full.n_dofs, dtype=bool)
        in_shell[shell_dof_locations(system.full)[0]] = True
        system.set_face_condition('x_lo', EdgeCondition('dirichlet', value=1.0))
        system.set_face_condition('x_hi', EdgeCondition('dirichlet', value=3.0))
        system.assemble(tensor)
        system.solve()
        error = np.abs(system.values - (1.0 + 2.0 * points[:, 0] / side)) / 3.0   # z-independent: shell dofs too
        add(f'linear_dirichlet_p{order}', error[~in_shell].max())
        add(f'linear_dirichlet_shell_p{order}', error[in_shell].max(), _SHELL_RTOL)
        inflow = _CONDUCTIVITY[0] * 2.0 / side * side ** 2
        add(f'face_reactions_p{order}', max(abs(system.face_flux('x_lo') + system.face_flux('x_hi')) / inflow,
                                            abs(abs(system.face_flux('x_hi')) / inflow - 1.0)))
        system.set_face_condition('x_hi', EdgeCondition('robin', h=_ROBIN_H, external=5.0))
        system.assemble(tensor)
        system.solve()
        slope = _ROBIN_H * (5.0 - 1.0) / (_CONDUCTIVITY[0] + _ROBIN_H * side)
        error = np.abs(system.values - (1.0 + slope * points[:, 0])) / 5.0
        add(f'linear_robin_p{order}', error[~in_shell].max())
        add(f'linear_robin_shell_p{order}', error[in_shell].max(), _SHELL_RTOL)

        balance = FEMDiffusionSystem3D(mesh, order)   # held window faces: point source + sheet source on a box
        for face in window_faces_all:
            balance.set_face_condition(face, EdgeCondition('dirichlet', value=0.0))
        elements, reference = PointLocator3D(mesh).locate(np.array([[0.4, 0.55, 0.45]]) * side)
        balance.add_point_source(elements[0], reference[0], _POINT_POWER)
        centroids = mesh.centroids()
        inner = np.all((centroids > 0.3 * side) & (centroids < 0.7 * side), axis=1)
        sheet = interface_faces(mesh, inner)
        corners = mesh.points[mesh.faces[sheet]]
        sheet_area = 0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
                                          axis=1).sum()
        balance.add_surface_condition(sheet, EdgeCondition('neumann', value=_SHEET_FLUX))
        balance.assemble(tensor)
        balance.solve()
        total = _POINT_POWER + _SHEET_FLUX * sheet_area
        reactions = sum(balance.face_flux(face) for face in window_faces_all)
        add(f'source_balance_p{order}', abs(reactions + total) / total)

        decay = FEMDiffusionSystem3D(mesh, order)   # insulated, uniform loss: u1 = c u0 / (c + a dt)
        decay.set_element_values(np.ones(mesh.n_elements, dtype=bool), 2.0)
        decay.add_loss(np.ones(mesh.n_elements, dtype=bool), _LOSS, 0.0)
        decay.assemble(tensor, capacity_over_dt=_CAPACITY_OVER_DT)
        decay.solve()
        stepped = 2.0 * _CAPACITY_OVER_DT / (_CAPACITY_OVER_DT + _LOSS)
        add(f'implicit_loss_step_p{order}', np.abs(decay.values - stepped).max() / stepped)
    return checks_result(checks, elements=mesh.n_elements)


DIFFUSION_3D_SELFTEST_CASES = (
    SelfTestCase(
        name='fem_diffusion3d_identities',
        label='FEM tetrahedra: scalar diffusion system identities',
        description='CPU checks of fem.diffusion3d.FEMDiffusionSystem3D (host SuperLU) on a graded 3 x 3 x 3 Kuhn box '
                    'with one PML shell cell beyond z_hi (no gmsh), P1 and P2, diagonal k (2, 0.7, 1.3): a linear '
                    'field between Dirichlet x_lo / x_hi faces (shell dofs by the clamped extension) and with a Robin '
                    'x_hi face; the x_lo / x_hi reactions balance and equal k dT / L x area; with every window face '
                    'held, a point source plus a sheet flux on the interface faces of an inner element box are '
                    'returned by the face reactions; an insulated uniform field with loss a and capacity c / dt '
                    '((a + c / dt) h^2 / k ~ 1, so the uniform mode is well conditioned) steps to c u0 / (c + a dt). '
                    'All 1e-10, except the shell dofs at 1e-8: they read the field 1e-9 x side inside the window. '
                    'error = number of failed sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_diffusion3d,
        max_wall_time=120.0),
)
