"""Thermoelastic damping (TED) of the vibration modes of solid_mechanics (2D tensor and triangle meshes) and
solid_mechanics_3d (tetrahedra): per mode the harmonic heat equation of its strain-rate heat source on the solid, Q_TED
from the energy dissipated per cycle; the runtime and 'table' result of the thermoelastic_damping feature."""
# Perturbative, one-way (Zener / Lifshitz-Roukes): the isothermal mode (omega = Re omega, displacement u, Voigt strain
# eps, engineering shear) is a fixed drive, exp(+j omega t). Constitutive sigma = C eps - beta theta with the thermal
# stress per kelvin beta = C : alpha (stiffness.thermal_coefficients / thermal_stress_3d; plane stress the reduced beta_I
# - c_I3 beta_3 / c_33 and no eps_zz term, plane strain no zz term, axisymmetric with the hoop row); the linearised heat
# equation j omega rho cp theta - div(k grad theta) = q, q = -j omega T0 (beta : eps), adiabatic (no flux) on every
# boundary of the solid. Discretised on the solid's own Lagrange space (tensor mesh: Q1 nodes of the cell corners,
# 2 x 2 Gauss; triangles / tetrahedra: the solid map's P1 / P2): b_a = int phi_a beta : eps dV (fem.elasticity
# assemble_strain_source*, the transpose of the thermal stress load), (K_T + j omega C_T) theta = -j omega T0 b with K_T
# = int k grad phi . grad phi (diagonal k_xx, k_yy (, k_zz)), C_T = int rho cp phi phi, I = b^H theta = int beta : eps*
# theta dV. The thermal stress then adds the complex stiffness d = -I to the modal stiffness u^H K u = omega^2 u^H M u
# (twice the peak stored energy), so 1 / Q_TED = Im d / (u^H K u) = -Im I / (omega^2 u^H M u) (> 0: the stress power
# -beta theta : d eps / dt averages to -(omega / 2) Im I per volume) and the relative frequency shift Re d / (2 u^H K u)
# (> 0: towards the adiabatic stiffness). One host SuperLU (solvers.sparse_direct, private runtime) per mode: each mode
# has its own omega. Axisymmetric integrals are whole rings (x 2 pi r on both sides: the ratio is per ring), planar ones
# per m of depth (thickness cancels). Other losses of the same mode: the Q of a damped pencil (PML / anchor radiation,
# mechanical_loss_factor on triangles / tetrahedra), else (tensor meshes, undamped pencil) the perturbative loss factor
# u^H (eta K) u / u^H K u, and on tetrahedra the squeeze-film gas Q; 1 / Q_total = 1 / Q_TED + 1 / Q_other.
import math

import numpy as np

from ...core.backend import to_host

TABLE_COLUMNS = ('resonance_number', 'frequency', 'q_ted', 'ted_frequency_shift', 'q_other', 'q_total')
TABLE_UNITS = ('1', 'Hz', '1', '1', '1', '1')
_GAUSS = (0.5 - 0.5 / math.sqrt(3.0), 0.5 + 0.5 / math.sqrt(3.0))   # 2-point Gauss on [0, 1]
_RING = 2.0 * math.pi
_THERMAL_ARRAYS = ('k_xx', 'k_yy', 'k_zz', 'cp')


class ThermalProblem:
    """Host pieces of one solid's TED evaluation: source(u) -> b over the scalar dofs, conductance K_T and capacity C_T
    (CSR over the same dofs), the active (thermal) dofs, the mechanical consistent mass M over the mode dofs and an
    optional loss operator eta K (None: no perturbative loss factor)."""

    def __init__(self, source, conductance, capacity, active, mass, loss=None):
        self.source = source
        self.conductance = conductance
        self.capacity = capacity
        self.active = np.flatnonzero(active)
        self.mass = mass
        self.loss = loss


def _host(values):
    return np.asarray(to_host(values), dtype=np.float64)


def _material_values(solver, label, view=None):
    """(k_xx, k_yy, k_zz, cp) host float64 of the solver's domain (view: a window slice of the tensor grid); an
    unallocated array (no material sets it) reads 0."""
    materials = solver.domain.materials
    names = set(materials.property_names())
    values = []
    for name in _THERMAL_ARRAYS:
        if name in names:
            array = getattr(materials, name)
            values.append(_host(array if view is None else array[view]))
        else:
            values.append(None)
    if values[3] is None:
        raise ValueError(f'{label}: thermoelastic damping needs the specific heat capacity cp (and the density rho) of '
                         'the solid: no material sets cp (library silicon / silica do)')
    zeros = np.zeros_like(values[3])
    return tuple(zeros if value is None else value for value in values)


def _capacity(solid, density, cp, label):
    """rho cp [J/(m^3 K)] on the solid (0 elsewhere); ValueError when a solid cell has none."""
    capacity = np.where(solid, density * cp, 0.0)
    missing = solid & ~(capacity > 0.0)
    if missing.any():
        raise ValueError(f'{label}: {int(missing.sum())} solid cell(s) / element(s) have no heat capacity rho cp > 0: '
                         'thermoelastic damping needs the density rho and the specific heat cp of every solid cell')
    return capacity


# ---- 2D tensor mesh (Q1 on the cell corners) ----
def _q1_tables():
    """(N (4 points, 4 nodes), dN / dxi, dN / deta, (xi, eta) points) of the 2 x 2 Gauss rule; local nodes (i, j),
    (i + 1, j), (i, j + 1), (i + 1, j + 1)."""
    points = [(xi, eta) for xi in _GAUSS for eta in _GAUSS]
    shape, d_xi, d_eta = [], [], []
    for xi, eta in points:
        shape.append([(1 - xi) * (1 - eta), xi * (1 - eta), (1 - xi) * eta, xi * eta])
        d_xi.append([-(1 - eta), 1 - eta, -eta, eta])
        d_eta.append([-(1 - xi), -xi, 1 - xi, xi])
    return np.array(shape), np.array(d_xi), np.array(d_eta), np.array(points)


def _tensor_problem(solver):
    """ThermalProblem of a 2D SolidMechanicsPhysics (tensor mesh) after its eigenfrequency solve."""
    system = solver.system
    label = system.label()
    if getattr(solver, 'guided', False) or getattr(solver, 'piezoelectric', False):
        raise ValueError(f'{label}: thermoelastic damping is not available for guided (2.5D) or piezoelectric modes')
    import scipy.sparse
    nx, ny = system.element_shape
    ny_nodes = system.node_shape[1]
    solid = np.array(to_host(solver._solid), dtype=bool)
    k_xx, k_yy, _, cp = _material_values(solver, label, system.window)
    density = _host(solver.domain.materials.rho[system.window])
    capacity = _capacity(solid, density, cp, label)
    plane_stress = solver.out_of_plane == 'plane_stress'
    beta_x, beta_y, beta_z, beta_xy = (_host(values) for values in solver._thermal_coefficients(
        plane_stress=plane_stress))
    axisymmetric = solver.axisymmetric
    shape, d_xi, d_eta, points = _q1_tables()
    i_index, j_index = np.nonzero(solid)
    nodes = np.stack([i_index * ny_nodes + j_index, (i_index + 1) * ny_nodes + j_index,
                      i_index * ny_nodes + j_index + 1, (i_index + 1) * ny_nodes + j_index + 1], axis=1)   # (m, 4)
    dx = system.dx_window[i_index][:, None]
    dy = system.dy_window[j_index][:, None]
    weights = np.broadcast_to(dx * dy / 4.0, (len(i_index), len(points))).copy()                 # (m, q)
    radius = None
    if axisymmetric:
        radius = system.r_nodes[i_index][:, None] + points[None, :, 0] * dx
        weights *= _RING * radius
    grad_x = d_xi[None] / dx[:, :, None]                                                         # (m, q, 4)
    grad_y = d_eta[None] / dy[:, :, None]
    betas = tuple(values[i_index, j_index][:, None] for values in (beta_x, beta_y, beta_xy, beta_z))
    n_nodes = system.node_shape[0] * ny_nodes

    def source(displacement):
        u = np.asarray(displacement).reshape(n_nodes, 2)
        u_x, u_y = u[nodes, 0], u[nodes, 1]                                                       # (m, 4)
        strain_xx = np.einsum('mqa,ma->mq', grad_x, u_x)
        strain_yy = np.einsum('mqa,ma->mq', grad_y, u_y)
        shear = np.einsum('mqa,ma->mq', grad_y, u_x) + np.einsum('mqa,ma->mq', grad_x, u_y)
        value = betas[0] * strain_xx + betas[1] * strain_yy + betas[2] * shear
        if axisymmetric:
            value = value + betas[3] * np.einsum('qa,ma->mq', shape, u_x) / radius
        local = np.einsum('mq,qa->ma', value * weights, shape)
        result = np.zeros(n_nodes, dtype=local.dtype)
        np.add.at(result, nodes, local)
        return result

    k_x = k_xx[i_index, j_index][:, None, None, None]
    k_y = k_yy[i_index, j_index][:, None, None, None]
    stiffness_blocks = np.einsum('mq,mqab->mab', weights, k_x * grad_x[..., :, None] * grad_x[..., None, :]
                                 + k_y * grad_y[..., :, None] * grad_y[..., None, :])
    capacity_blocks = np.einsum('mq,qa,qb->mab', weights * capacity[i_index, j_index][:, None], shape, shape)
    rows = np.broadcast_to(nodes[:, :, None], stiffness_blocks.shape).ravel()
    cols = np.broadcast_to(nodes[:, None, :], stiffness_blocks.shape).ravel()

    def csr(blocks):
        return scipy.sparse.coo_matrix((blocks.ravel(), (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()

    active = np.zeros(n_nodes, dtype=bool)
    active[nodes.ravel()] = True
    operators = solver._harmonic_operators
    loss = None if operators is None else operators[1]
    return ThermalProblem(source, csr(stiffness_blocks), csr(capacity_blocks), active, solver.modal_mass, loss)


# ---- triangle meshes (the solid map's P1 / P2) ----
def _triangle_problem(solver):
    """ThermalProblem of a 2D SolidMechanicsFEMPhysics (triangle mesh) after its eigenfrequency solve; the elastic PML
    ring takes no part (its elements are not physical)."""
    from ...fem.assembly import assemble_mass, assemble_stiffness
    from ...fem.elasticity import assemble_strain_source
    label = solver._label()
    system = solver.system
    if solver.azimuthal_order:
        raise ValueError(f'{label}: thermoelastic damping needs azimuthal_order 0 (the cos(m phi) modes are not '
                         'modelled)')
    dofs = system.dofs
    axisymmetric = solver.axisymmetric
    solid = np.asarray(solver._solid, dtype=bool) & np.asarray(solver._window_elements, dtype=bool)
    k_xx, k_yy, _, cp = _material_values(solver, label)
    capacity = _capacity(solid, _host(solver.domain.materials.rho), cp, label)
    plane_stress = solver.out_of_plane == 'plane_stress'
    beta_x, beta_y, beta_z, beta_xy = (np.broadcast_to(_host(values), solid.shape)
                                       for values in solver._thermal_coefficients(plane_stress=plane_stress))
    rows = (beta_x, beta_y, beta_z, beta_xy) if axisymmetric else (beta_x, beta_y, beta_xy)
    beta = np.where(solid[:, None], np.stack(rows, axis=1), 0.0)
    elements = solid[dofs.elements]   # per dof map element
    scale = _RING if axisymmetric else 1.0
    conductance = scale * assemble_stiffness(dofs, np.stack([k_xx, k_yy], axis=1), axisymmetric=axisymmetric,
                                             elements=elements)
    capacity_matrix = scale * assemble_mass(dofs, capacity, axisymmetric=axisymmetric, elements=elements)
    active = np.zeros(dofs.n_dofs, dtype=bool)
    active[np.unique(dofs.element_dofs[elements])] = True
    components = system.components

    def source(displacement):
        return assemble_strain_source(dofs, np.asarray(displacement).reshape(-1, components), beta,
                                      axisymmetric=axisymmetric, elements=elements)
    return ThermalProblem(source, conductance.tocsr(), capacity_matrix.tocsr(), active, solver.modal_mass)


# ---- tetrahedron meshes (solid_mechanics_3d) ----
def _tetrahedra_problem(solver):
    """ThermalProblem of a SolidMechanicsFEM3DPhysics (tetrahedra) after its eigenfrequency solve; PML shell elements
    take no part."""
    from ...fem.assembly3d import assemble_lagrange_mass_3d, assemble_lagrange_stiffness_3d
    from ...fem.elasticity3d import assemble_strain_source_3d
    label = solver._label()
    system = solver.system
    if solver.piezoelectric:
        raise ValueError(f'{label}: thermoelastic damping is not available for a piezoelectric solid (its modes carry '
                         'the potential as a fourth dof per node)')
    if system.bloch is not None:
        raise ValueError(f'{label}: thermoelastic damping is not available in a Bloch cell (periodic axes)')
    dofs = system.full
    solid = np.asarray(to_host(solver._solid), dtype=bool) & np.asarray(solver.domain.mesh.window_elements(),
                                                                         dtype=bool)
    k_xx, k_yy, k_zz, cp = _material_values(solver, label)
    capacity = _capacity(solid, _host(solver.domain.materials.rho), cp, label)
    beta = np.where(solid[:, None], np.asarray(solver._thermal_stress(), dtype=np.float64), 0.0)
    conductance = assemble_lagrange_stiffness_3d(dofs, np.stack([k_xx, k_yy, k_zz], axis=1), elements=solid)
    capacity_matrix = assemble_lagrange_mass_3d(dofs, capacity, elements=solid)
    active = np.zeros(dofs.n_dofs, dtype=bool)
    active[np.unique(dofs.element_dofs[solid])] = True

    def source(displacement):
        return assemble_strain_source_3d(dofs, np.asarray(displacement).reshape(-1, 3), beta, elements=solid)
    return ThermalProblem(source, conductance.tocsr(), capacity_matrix.tocsr(), active, solver.modal_mass)


def thermal_problem(solver) -> ThermalProblem:
    """ThermalProblem of the solid after an eigenfrequency solve, by its mesh kind; ValueError for a solid or mesh
    kind TED does not model (3D tensor meshes: matrix-free, no assembled K)."""
    if getattr(solver, 'eigen_solve', None) is None or getattr(solver, 'modal_mass', None) is None:
        raise ValueError('not solved yet: thermoelastic damping needs an eigenfrequency (or optomechanics) study')
    kind = getattr(solver.domain, 'grid_kind', 'tensor')
    if kind == 'triangles':
        return _triangle_problem(solver)
    if kind == 'tetrahedra':
        return _tetrahedra_problem(solver)
    if hasattr(solver.system, 'node_shape') and len(solver.system.node_shape) == 2:
        return _tensor_problem(solver)
    raise ValueError('thermoelastic damping of solid_mechanics_3d needs a tetrahedron mesh (the tensor-mesh modes are '
                     'matrix-free)')


# ---- evaluation ----
def mode_damping(problem, resonance, temperature, linear_solver) -> tuple:
    """(1 / Q_TED, relative frequency shift, 1 / Q of the perturbative loss factor) of one resonance (module header)."""
    omega = float(resonance.omega)
    displacement = np.asarray(resonance.displacement)
    stiffness_energy = omega * omega * float(np.real(np.vdot(displacement, problem.mass @ displacement)))
    loss_inverse = 0.0
    if problem.loss is not None and not resonance.frequency_imag and stiffness_energy > 0.0:
        loss_inverse = float(np.real(np.vdot(displacement, problem.loss @ displacement))) / stiffness_energy
    active = problem.active
    coupling = problem.source(displacement)[active]
    if omega <= 0.0 or stiffness_energy <= 0.0 or not np.any(coupling):
        return 0.0, 0.0, loss_inverse
    operator = (problem.conductance[active][:, active] + 1j * omega * problem.capacity[active][:, active]).tocsc()
    factor = linear_solver.factorize_host(operator.astype(np.complex128))
    try:
        temperature_field = factor.solve_host(-1j * omega * temperature * coupling.astype(np.complex128))
    finally:
        getattr(factor, 'release', lambda: None)()
    overlap = complex(np.vdot(coupling, temperature_field))
    return -overlap.imag / stiffness_energy, -overlap.real / (2.0 * stiffness_energy), loss_inverse


def _inverse(q_factor) -> float:
    return 0.0 if not math.isfinite(q_factor) or q_factor == 0.0 else 1.0 / q_factor


def _q(inverse) -> float:
    return math.inf if inverse <= 0.0 else 1.0 / inverse


class ThermoelasticDamping:
    """Runtime of one thermoelastic_damping feature: the reference temperature T0 [K] and the rows of the last
    eigenfrequency solve (evaluated on the first result read, kept while that solve is the solver's)."""

    def __init__(self, name, temperature):
        self.name = name
        self.temperature = float(temperature)
        self._kept = None   # (ModalSolve, rows)
        self.linear_solver = None

    def rows(self, solver) -> list:
        """[[number, f, Q_TED, df / f, Q_other, Q_total]] per resonance of solver.eigen_solve (cached per solve)."""
        solve = getattr(solver, 'eigen_solve', None)
        if self._kept is not None and self._kept[0] is solve:
            return self._kept[1]
        problem = thermal_problem(solver)
        if self.linear_solver is None:
            from ...solvers.sparse_direct import SparseDirectSolver
            self.linear_solver = SparseDirectSolver(back_end='scipy_splu', backend='cpu',
                                                    label=lambda: f'thermoelastic damping {self.name}')
        gas = _gas_inverse(solver, len(solve.resonances))
        rows = []
        for number, resonance in enumerate(solve.resonances, start=1):
            ted_inverse, shift, loss_inverse = mode_damping(problem, resonance, self.temperature, self.linear_solver)
            other_inverse = _inverse(resonance.q_factor) + loss_inverse + gas[number - 1]
            rows.append([number, resonance.frequency, _q(ted_inverse), shift, _q(other_inverse),
                         _q(ted_inverse + other_inverse)])
        self._kept = (solve, rows)
        return rows

    def table(self, solver) -> dict:
        """'table' result (TABLE_COLUMNS) with the extras T0 and selected_resonance."""
        return {'kind': 'table', 'columns': list(TABLE_COLUMNS), 'units': list(TABLE_UNITS), 'rows': self.rows(solver),
                'T0': self.temperature, 'selected_resonance': getattr(solver, 'selected_resonance', None)}


def _gas_inverse(solver, count) -> list:
    """1 / Q_gas per resonance summed over the evaluated squeeze films of a tetrahedra solid (0 without)."""
    films = [film for film in getattr(solver, 'squeeze_films', ()) if getattr(film, 'rows', None) is not None]
    totals = [0.0] * count
    for film in films:
        if len(film.rows) != count:
            continue
        for index, row in enumerate(film.rows):
            totals[index] += _inverse(float(row['q_gas']))
    return totals
