"""Self-test of fem.diffusion (mesh plan stage T4, CPU; specs() empty, check({}) solves directly): FEMDiffusionSystem
identities on the curved disc mesh (selftests_mesher.disc_mesh), error = number of failed sub-checks (tolerance 0)."""
import numpy as np

from ..model.schema import SelfTestCase
from ..model.selftest_helpers import SubChecks, checks_result
from .selftests_elements import _PATCH_H
from .selftests_mesher import disc_mesh

_UM = 1e-6
_H = _PATCH_H                      # element size of the 2 x 2 um disc window (0.2 um, the cached patch mesh)
_SIDE = 2.0 * _UM
_TENSOR = (2.0, 1.0)               # diagonal conductivity (xx, yy)
_SLOPE = 3.0 / _UM                 # du/dx of the linear fields
_EXACT_RTOL = 1e-10                # direct solves of exactly representable fields


def _system(mesh, order, axisymmetric=False):
    from .diffusion import FEMDiffusionSystem

    return FEMDiffusionSystem(mesh, order, None, backend='cpu', axisymmetric=axisymmetric)


def _tensor(mesh):
    from .diffusion import coefficient_tensor

    count = mesh.n_elements
    return coefficient_tensor(np.full(count, _TENSOR[0]), np.full(count, _TENSOR[1]))


def _nodal_error(system, exact):
    points = system.full.dof_points()
    expected = exact(points[:, 0], points[:, 1])
    return float(np.abs(system.values - expected).max() / np.abs(expected).max())


def _check_diffusion(results_by_name):
    from .diffusion import EdgeCondition

    checks = SubChecks()
    add = checks.add
    mesh = disc_mesh(_H)
    linear_x = lambda x, y: 1.0 + _SLOPE * x
    face_flux = _TENSOR[0] * _SLOPE * _SIDE   # |K_xx du/dx| x face length [per m]
    for order in (1, 2):
        # Dirichlet x faces, insulated y faces: the linear field is exact, reactions are the face fluxes
        system = _system(mesh, order)
        system.set_face_condition('x_lo', EdgeCondition('dirichlet', value=1.0))
        system.set_face_condition('x_hi', EdgeCondition('dirichlet', value=linear_x(_SIDE, 0.0)))
        system.assemble(_tensor(mesh))
        system.solve()
        add(f'p{order}_linear_dirichlet', _nodal_error(system, linear_x) <= _EXACT_RTOL, _nodal_error(system, linear_x))
        reactions = [system.inward_flux(system.face_edges(face), system.face_conditions[face][0])
                     for face in ('x_lo', 'x_hi')]
        errors = [abs(reactions[0] + face_flux) / face_flux, abs(reactions[1] - face_flux) / face_flux]
        add(f'p{order}_face_reactions', max(errors) <= _EXACT_RTOL, {'reactions': reactions, 'expected': face_flux})
        # Robin x_hi with h (external - u) = K du/dx: the same linear field
        h = 1e6
        external = linear_x(_SIDE, 0.0) + _TENSOR[0] * _SLOPE / h
        system = _system(mesh, order)
        system.set_face_condition('x_lo', EdgeCondition('dirichlet', value=1.0))
        system.set_face_condition('x_hi', EdgeCondition('robin', h=h, external=external))
        system.assemble(_tensor(mesh))
        system.solve()
        add(f'p{order}_linear_robin', _nodal_error(system, linear_x) <= _EXACT_RTOL, _nodal_error(system, linear_x))
        # value-only changes (Dirichlet value, Robin external at the same h): no new factor, the shifted field exact
        shifted_x = lambda x, y: 2.0 + _SLOPE * x
        factorizations = system.linear_solver.runtime.n_factorizations
        system.set_face_condition('x_lo', EdgeCondition('dirichlet', value=2.0))
        system.set_face_condition('x_hi', EdgeCondition('robin', h=h, external=external + 1.0))
        system.solve()
        refactored = system.linear_solver.runtime.n_factorizations - factorizations
        add(f'p{order}_value_change_keeps_factor', refactored == 0 and _nodal_error(system, shifted_x) <= _EXACT_RTOL,
            {'new_factorizations': refactored, 'nodal_error': _nodal_error(system, shifted_x)})

    # axisymmetric (r, z): u linear in z between Dirichlet y faces, the axis x_lo insulated; reaction per radian
    linear_z = lambda x, y: 1.0 + _SLOPE * y
    system = _system(mesh, 2, axisymmetric=True)
    system.set_face_condition('y_lo', EdgeCondition('dirichlet', value=1.0))
    system.set_face_condition('y_hi', EdgeCondition('dirichlet', value=linear_z(0.0, _SIDE)))
    system.assemble(_tensor(mesh))
    system.solve()
    expected = _TENSOR[1] * _SLOPE * _SIDE ** 2 / 2.0   # K_yy du/dz x int r dr over the face
    reaction = system.inward_flux(system.face_edges('y_lo'), system.face_conditions['y_lo'][0])
    add('axisymmetric_linear_z', _nodal_error(system, linear_z) <= _EXACT_RTOL, _nodal_error(system, linear_z))
    add('axisymmetric_reaction', abs(reaction + expected) / expected <= _EXACT_RTOL,
        {'reaction': reaction, 'expected': -expected})

    # point source between four held faces: the held reactions take all of it in
    power = 2.5
    centre = np.argmin(np.hypot(mesh.points[:, 0] - 0.37 * _SIDE, mesh.points[:, 1] - 0.41 * _SIDE))
    system = _system(mesh, 2)
    for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi'):
        system.set_face_condition(face, EdgeCondition('dirichlet', value=0.0))
    system.add_point_source(centre, power)
    system.assemble(_tensor(mesh))
    system.solve()
    total = sum(system.face_flux(face) for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi'))   # corners counted once
    add('point_source_balance', abs(total + power) / power <= _EXACT_RTOL, {'held_outflow': total, 'power': power})

    # implicit Euler with a volumetric loss on an insulated uniform field: u1 = c / (c + a dt) exactly
    capacity_over_dt, loss = 3.0, 0.5
    system = _system(mesh, 2)
    system.values[:] = 1.0
    system.add_loss(np.ones(mesh.n_elements, dtype=bool), loss, 0.0)
    system.assemble(np.zeros((mesh.n_elements, 2, 2)), np.full(mesh.n_elements, capacity_over_dt))
    system.solve()
    step_value = capacity_over_dt / (capacity_over_dt + loss)
    error = float(np.abs(system.values - step_value).max() / step_value)
    add('implicit_euler_loss_step', error <= _EXACT_RTOL, error)
    return checks_result(checks)


DIFFUSION_SELFTEST_CASES = (
    SelfTestCase(
        name='fem_diffusion_identities', label='FEM: diffusion system identities on triangles',
        description=('fem.diffusion.FEMDiffusionSystem on the curved 2 x 2 um disc mesh (h = 0.2 um, the disc edges '
                     'isoparametric for P2), diagonal conductivity (2, 1), cpu SuperLU: P1 / P2 reproduce u = 1 + 3 '
                     'x / um between Dirichlet x faces and with a Robin x_hi face whose h (external - u) equals the '
                     'flux, and after a Dirichlet value / Robin external change at the same h the shifted field '
                     'without a new factorisation; the x-face reactions equal K_xx du/dx x the face length; '
                     'axisymmetric (r, z) u linear in z '
                     'with its per-radian reaction K_yy du/dz R^2 / 2; a point source between four held faces is '
                     'balanced by their face_flux reactions (a corner dof counted once); one implicit-Euler step of a uniform field with a volumetric loss '
                     'gives c / (c + a dt). error = number of failed sub-checks (relative tolerance 1e-10 each, values '
                     'in details); tolerance 0. h 0.25 -> 0.2 um on 2026-10-20 (exactness checks: size-independent), '
                     'not yet run at that size, nor with the topology anchors and pattern_sum assembly.'),
        specs=lambda: {}, check=_check_diffusion, max_wall_time=300.0),
)
