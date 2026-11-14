"""Problem classes of the solver profiler: each assembles one representative system with open_sol's own operator
code (2D FDFD Helmholtz, finite-volume and P2 finite-element heat conduction, 3D brick elasticity) at a size knob and
names the physics type, solver role, study and grid kind whose solver choices it stands for. open_sol is imported
inside the builders only (the parent process reads the metadata alone)."""
# Unknown order is open_sol's: FDFD / finite volume flat i * ny (* nz) + j over the full grid (PML included), FEM the
# P2 dof numbering (nodes, then edge midpoints) with the Dirichlet dofs eliminated (fem.assembly.eliminate_dirichlet).
# Operators are host scipy CSR (the cpu backend's): FDFD complex128 (general; D A complex symmetric with D the stretched
# cell areas, the symmetrizer the FDFD physics hands MUMPS / PARDISO / cuDSS), heat float64 SPD.
# Matrix-free problems (kind 'modes': solid_mechanics_3d on bricks) have no operator: the builder returns host
# constants only and elastic_modes_system assembles the physics' own ElasticSystem3D on the run backend (called by
# runners.backend_arrays, so before the memory baseline, as the build of a model would).
import math
from dataclasses import dataclass, field

import numpy as np

# 2D FDFD (em_fdfd_ez): a dielectric disc in vacuum, CPML on every side, the window refined by the size knob
FDFD_WAVELENGTH = 1.55e-6       # [m] drive / eigen target
FDFD_WINDOW = 4.0e-6            # [m] full grid side (PML included)
FDFD_DISC_RADIUS = 0.9e-6       # [m]
FDFD_DISC_EPS = 12.25           # silicon-like relative permittivity
FDFD_PML_CELLS = 10
FDFD_RESONANCES = 3             # resonances asked of the eigenfrequency solve
# heat conduction: a cube / square / segment of HEAT_WINDOW with a central inclusion HEAT_CONTRAST times more
# conducting, T = 0 on x_lo and 1 on x_hi, the other faces insulated
HEAT_WINDOW = 1.0e-3            # [m]
HEAT_INCLUSION_RADIUS = 0.25e-3   # [m]
HEAT_CONTRAST = 100.0
FEM_ORDER = 2                   # Lagrange order of heat_fem_3d (open_sol's FEM heat default)
# 3D vibration modes (solid_mechanics_3d, tensor mesh, eigenfrequency study): a steel box of uniform cubic cells,
# (n + 1) x n x (n - 1) cells for an even size n (pairwise coprime counts: no two axes share a phase p pi / N, so the
# near-cube has no exact cross-axis degeneracy, only the split triplets of the cube), rollers on all six faces
# (normal displacement 0). u_a = A_a sin(p_a pi i_a / N_a) prod_{d != a} cos(p_d pi i_d / N_d) are then exact discrete
# modes of the trilinear stiffness / consistent mass (the closed form of physics/mechanics/selftests_eigenfrequency's
# 3D roller box, there on 9 x 5 x 4 cells), so every run is also compared with the exact discrete frequencies.
MODES_PHYSICS_TYPE = 'solid_mechanics_3d'
MODES_STUDY = 'eigenfrequency'
MODES_GRID = 'tensor_3d'
MODES_SIDE = 10e-3              # [m] the n cells of the middle axis (y); x has one cell more, z one less
MODES_YOUNGS_MODULUS = 200e9    # [Pa] steel-like
MODES_POISSON_RATIO = 0.3
MODES_DENSITY = 7850.0          # [kg/m^3]
MODES_RESONANCES = 6            # n_resonances of the eigenfrequency study
# orders searched per axis by the closed form: the lowest modes have orders <= 1 (continuum estimate: shear (1, 1, 0)
# triplet, (1, 1, 1) pair, longitudinal (1, 0, 0) triplet, then (2, 1, 0) at 2.5x the lowest omega^2), so 6 is margin
MODES_REFERENCE_MAX_ORDER = 6


@dataclass(frozen=True, eq=False)
class Problem:
    """One assembled system: host scipy CSR operator, rhs (None for an eigen problem), the grid shape (FDFD: (nx, ny)
    of SparseDirectSolver.factorize / the line-block layout; finite volume: the window shape, DiffusionDirectSolver's
    nested-dissection shape in 3D; FEM: None; modes: the window cells), the FDFD symmetrizer D (None elsewhere), the
    longest axis in cells (pcg auto iteration limit) and the eigen data (FDFD: mass diagonal, target omega, energy
    weights, window rows, resonances; modes: the box constants, free dofs and the closed-form reference omegas)."""
    kind: str                     # 'complex' (FDFD drive) | 'spd' (heat) | 'eigen' (FDFD shift-invert) | 'modes'
    operator: object              # None for a matrix-free problem (kind 'modes')
    rhs: object = None
    grid_shape: tuple | None = None
    symmetrizer: object = None
    longest_axis: int = 0
    eigen: dict = field(default_factory=dict)

    @property
    def n_unknowns(self) -> int:
        """Rows of the operator; a matrix-free problem: its free dofs."""
        return int(self.eigen['n_free']) if self.operator is None else int(self.operator.shape[0])

    @property
    def nnz(self):
        """Stored entries of the operator, None for a matrix-free problem."""
        return None if self.operator is None else int(self.operator.nnz)


@dataclass(frozen=True, eq=False)
class ProblemClass:
    """Metadata of one problem class: the physics type / role / study / grid kind whose solver choices
    (solver_catalog) it profiles, the size knob and its default / --quick sizes, and builder(size) -> Problem;
    ignored_settings {SolverType name: (setting, ...)}: choice settings the problem's consumer ignores (the catalogue
    pins them to what 'auto' / the default resolves to instead of one identical row per value)."""
    name: str
    description: str
    physics_type: str
    role: str
    study: str
    grid: str
    dimensions: int
    size_text: str
    sizes: tuple
    quick_sizes: tuple
    builder: object
    kind: str = 'synthetic'   # model_cases.ModelTarget: 'model'
    ignored_settings: dict = field(default_factory=dict)


def _fdfd_parts(size):
    """(n, cell [m], omega [rad/s], eps (n, n) complex128, operator, symmetrizer D, in_window (n, n) bool) of the
    em_fdfd_ez Helmholtz operator at FDFD_WAVELENGTH (operators.yee_operators + helmholtz_operator as
    fdfd_2d._assemble, without pinned cells: no PEC / wall / mirror rows)."""
    from open_sol.core.constants import C0
    from open_sol.physics.em.boundaries import BoundaryConfig
    from open_sol.physics.em.operators import helmholtz_operator, stretched_cell_areas, yee_operators
    n = int(size)
    if n < 2 * FDFD_PML_CELLS + 8:
        raise ValueError(f'FDFD problems need at least {2 * FDFD_PML_CELLS + 8} cells per axis, got {n}')
    cell = FDFD_WINDOW / n
    widths = (np.full(n, cell), np.full(n, cell))
    config = BoundaryConfig.all_pml(FDFD_PML_CELLS)
    omega = 2.0 * math.pi * C0 / FDFD_WAVELENGTH
    ops = yee_operators(widths, config, cell, omega=omega)   # pml_dx = the (uniform) PML cell size
    centres = (np.arange(n) + 0.5) * cell - 0.5 * FDFD_WINDOW
    x, y = np.meshgrid(centres, centres, indexing='ij')
    eps = np.where(x * x + y * y <= FDFD_DISC_RADIUS ** 2, FDFD_DISC_EPS, 1.0).astype(np.complex128)
    face_weights = np.ones((n, n), dtype=np.complex128)   # ez: 1 / mu on both face sets (non-magnetic)
    operator = helmholtz_operator(ops, face_weights, face_weights, (omega / C0) ** 2 * eps).tocsr()
    symmetrizer = stretched_cell_areas(widths, config, cell, omega)
    in_window = np.zeros((n, n), dtype=bool)
    in_window[FDFD_PML_CELLS:n - FDFD_PML_CELLS, FDFD_PML_CELLS:n - FDFD_PML_CELLS] = True
    return n, cell, omega, eps, operator, symmetrizer, in_window


def fdfd_driven(size) -> Problem:
    """em_fdfd_ez driven solve: the Helmholtz operator and a unit point current inside the disc near its rim."""
    n, cell, _, _, operator, symmetrizer, _ = _fdfd_parts(size)
    rhs = np.zeros(n * n, dtype=np.complex128)
    i = n // 2 + int(0.6 * FDFD_DISC_RADIUS / cell)
    rhs[i * n + n // 2] = 1.0
    return Problem('complex', operator, rhs=rhs, grid_shape=(n, n), symmetrizer=symmetrizer, longest_axis=n)


def fdfd_eigen(size) -> Problem:
    """em_fdfd_ez eigenfrequency solve: FDFD_RESONANCES resonances of the disc nearest FDFD_WAVELENGTH (the arguments
    fdfd_2d hands eigenfrequency.solve_resonances: mass = eps cells, energy weights = cell areas, PML-free window)."""
    n, cell, omega, eps, operator, symmetrizer, in_window = _fdfd_parts(size)
    eigen = {'mass': eps.ravel().copy(), 'omega': omega, 'energy_weights': np.full(n * n, cell * cell),
             'in_window': in_window.ravel(), 'n_resonances': FDFD_RESONANCES}
    return Problem('eigen', operator, grid_shape=(n, n), symmetrizer=symmetrizer, longest_axis=n, eigen=eigen)


def _inclusion(axes_centres):
    """Bool window mask of the cells inside the central inclusion (disc / sphere; a line: a segment)."""
    grids = np.meshgrid(*axes_centres, indexing='ij')
    return sum(grid * grid for grid in grids) <= HEAT_INCLUSION_RADIUS ** 2


def _heat_fv(cells) -> Problem:
    """Finite-volume heat window of `cells` cells per axis ((n, 1) = a 1D line, (n, n) 2D, (n, n, n) 3D): the
    core.fv_diffusion operator (2D: build_anisotropic_laplacian, 3D: fv_diffusion_3d.build_laplacian) and its boundary
    rhs for T = 0 on x_lo, 1 on x_hi."""
    from open_sol.core.fv_diffusion import BCSide, ScalarBoundaryConfig, build_anisotropic_laplacian
    cells = tuple(int(count) for count in cells)
    cell = HEAT_WINDOW / cells[0]
    widths = [np.full(count, cell) for count in cells]
    centres = [(np.arange(count) + 0.5) * cell - 0.5 * count * cell for count in cells]
    conductivity = np.where(_inclusion(centres), HEAT_CONTRAST, 1.0)
    bcs = ScalarBoundaryConfig(x_lo=BCSide('dirichlet', value=0.0), x_hi=BCSide('dirichlet', value=1.0))
    if len(cells) == 3:
        from open_sol.core.fv_diffusion_3d import FaceGeometry, build_laplacian
        from open_sol.core.metric import window_metric_3d
        geometry = FaceGeometry(window_metric_3d(*widths), np.float64, np)
        operator, rhs, *_ = build_laplacian((conductivity,) * 3, geometry, bcs, dtype=np.float64)
    else:
        operator, rhs, *_ = build_anisotropic_laplacian(conductivity, conductivity, np.zeros_like(conductivity),
                                                        widths[0], widths[1], bcs, dtype=np.float64,
                                                        kxy_active=False)
    return Problem('spd', operator.tocsr(), rhs=np.asarray(rhs, dtype=np.float64).ravel(), grid_shape=cells,
                   longest_axis=max(cells))


def heat_line(size) -> Problem:
    return _heat_fv((size, 1))


def heat_2d(size) -> Problem:
    return _heat_fv((size, size))


def heat_3d(size) -> Problem:
    return _heat_fv((size, size, size))


def heat_fem_3d(size) -> Problem:
    """P2 FEM heat on the Kuhn tetrahedra of a size^3 brick grid (fem.mesh3d.structured_box_mesh, assembly3d
    stiffness with per-element conductivity), T = 0 / 1 on the x_lo / x_hi planes eliminated."""
    from open_sol.fem.assembly3d import assemble_lagrange_stiffness_3d, eliminate_dirichlet
    from open_sol.fem.dofs3d import lagrange_dofs_3d
    from open_sol.fem.mesh3d import structured_box_mesh
    n = int(size)
    edges = np.linspace(-0.5 * HEAT_WINDOW, 0.5 * HEAT_WINDOW, n + 1)
    mesh = structured_box_mesh(edges, edges, edges)
    dofs = lagrange_dofs_3d(mesh, FEM_ORDER)
    centroids = mesh.centroids()
    conductivity = np.where(np.sum(centroids * centroids, axis=1) <= HEAT_INCLUSION_RADIUS ** 2, HEAT_CONTRAST, 1.0)
    stiffness = assemble_lagrange_stiffness_3d(dofs, conductivity)
    positions = mesh.points
    if FEM_ORDER == 2:   # dof n_points + e = the midpoint of edge e
        positions = np.concatenate([positions, 0.5 * (mesh.points[mesh.edges[:, 0]] + mesh.points[mesh.edges[:, 1]])])
    tolerance = 1e-9 * HEAT_WINDOW
    on_lo = np.flatnonzero(positions[:, 0] <= edges[0] + tolerance)
    on_hi = np.flatnonzero(positions[:, 0] >= edges[-1] - tolerance)
    fixed = np.concatenate([on_lo, on_hi])
    values = np.concatenate([np.zeros(on_lo.size), np.ones(on_hi.size)])
    system = eliminate_dirichlet(stiffness, np.zeros(dofs.n_dofs), fixed, values)
    return Problem('spd', system.matrix.tocsr(), rhs=np.asarray(system.rhs, dtype=np.float64), grid_shape=None,
                   longest_axis=FEM_ORDER * n)


@dataclass(frozen=True, eq=False)
class ModesSystem:
    """A 'modes' problem on the run backend: the assembled ElasticSystem3D (its 'linear' PCGSolver = the Automatic
    linear solver of the eigenfrequency study on that backend, linear_values its resolved settings) and the window
    element densities [kg/m^3] (backend float64), as SolidMechanics3DPhysics hands them to lowest_modes."""
    system: object
    density: object
    linear_values: dict


def _roller_box_reference(cells, cell, lame_lambda, mu, density, count) -> list:
    """The `count` lowest omega [rad/s] of the roller box (module notes) in ascending order: per order triple (orders
    up to MODES_REFERENCE_MAX_ORDER per axis) the 3 x 3 symbol of the 1D linear-element symbols K(t) = (2 / h)(1 - cos
    t), M(t) = (h / 3)(2 + cos t), S(t) = sin t (B_aa = c11 K_a M_b M_c + c66 M_a (K_b M_c + M_b K_c), B_ab = (c12 +
    c66) S_a S_b M_c) over rho M_x M_y M_z, component a present for 0 < p_a < N_a (selftests_eigenfrequency's
    _discrete_modes_3d, parametrised)."""
    c11, c12, c66 = lame_lambda + 2.0 * mu, lame_lambda, mu
    eigenvalues = []
    for orders in np.ndindex(*(min(count_a, MODES_REFERENCE_MAX_ORDER) + 1 for count_a in cells)):
        present = [axis for axis in range(3) if 0 < orders[axis] < cells[axis]]
        if not present:
            continue
        phases = [order * math.pi / count_a for order, count_a in zip(orders, cells)]
        stiff = [2.0 / cell * (1.0 - math.cos(phase)) for phase in phases]
        mass = [cell / 3.0 * (2.0 + math.cos(phase)) for phase in phases]
        slope = [math.sin(phase) for phase in phases]
        symbol = np.zeros((3, 3))
        for a in range(3):
            b, c = (axis for axis in range(3) if axis != a)
            symbol[a, a] = c11 * stiff[a] * mass[b] * mass[c] + c66 * mass[a] * (stiff[b] * mass[c]
                                                                                 + mass[b] * stiff[c])
            for b in range(3):
                if b != a:
                    symbol[a, b] = (c12 + c66) * slope[a] * slope[b] * mass[3 - a - b]
        eigenvalues.extend(np.linalg.eigvalsh(symbol[np.ix_(present, present)])
                           / (density * mass[0] * mass[1] * mass[2]))
    return [math.sqrt(value) for value in sorted(eigenvalues)[:count]]


def solid_modes_3d(size) -> Problem:
    """solid_mechanics_3d eigenfrequency (tensor mesh): the MODES_RESONANCES lowest modes of the steel roller box of
    (size + 1) x size x (size - 1) cells of MODES_SIDE / size (module notes); host constants only (matrix-free: the
    system is assembled on the run backend by elastic_modes_system), with the free dofs and the closed-form omegas."""
    from open_sol.physics.mechanics.materials import lame_coefficients
    n = int(size)
    if n < 4 or n % 2:
        raise ValueError(f'solid_modes_3d sizes are even and >= 4 (cells (n + 1, n, n - 1) pairwise coprime), got '
                         f'{size}')
    cells = (n + 1, n, n - 1)
    cell = MODES_SIDE / n
    lame_lambda, mu = lame_coefficients(MODES_YOUNGS_MODULUS, MODES_POISSON_RATIO)
    nodes = [count + 1 for count in cells]
    # rollers hold component a on the two node planes normal to axis a: (N_a - 1) free node planes of it
    n_free = sum((cells[axis] - 1) * math.prod(nodes[other] for other in range(3) if other != axis)
                 for axis in range(3))
    eigen = {'cells': cells, 'cell': cell, 'lame_lambda': float(lame_lambda), 'mu': float(mu),
             'density': MODES_DENSITY, 'n_resonances': MODES_RESONANCES, 'n_dofs': 3 * math.prod(nodes),
             'n_free': n_free,
             'reference_omegas': _roller_box_reference(cells, cell, float(lame_lambda), float(mu), MODES_DENSITY,
                                                       MODES_RESONANCES)}
    return Problem('modes', None, grid_shape=cells, longest_axis=max(cells), eigen=eigen)


def automatic_linear_values(backend) -> dict:
    """Resolved settings of the 'linear' role's Automatic of solid_mechanics_3d in the eigenfrequency study on a tensor
    3D mesh on the backend (SolverRole.automatic over the SolverType's param defaults: pcg, gpu mixed / cpu double,
    block Jacobi); ValueError when Automatic is no longer pcg (this problem then needs another linear solver)."""
    from open_sol.model.schema import get_registry
    registry = get_registry()
    solver_role = registry.physics_type(MODES_PHYSICS_TYPE).solver_roles['linear']
    type_name, settings = solver_role.automatic(backend, MODES_GRID, MODES_STUDY)
    if type_name != 'pcg':
        raise ValueError(f"solid_modes_3d builds its system with a pcg 'linear' solver; Automatic is now {type_name}")
    values = {param.name: param.default for param in registry.solver_type(type_name).params}
    values.update(settings)
    return values


def elastic_modes_system(problem, backend) -> ModesSystem:
    """ModesSystem of a 'modes' problem on the backend, as the solid_mechanics_3d build makes it: a Domain3D of the
    box cells (no PML), the Automatic 'linear' PCGSolver (pcg create + solid_linear_solver: max_iterations 0 = twice
    the auto limit), ElasticSystem3D.prepare, rollers (hold_face of the normal component on every face, as the roller
    feature), assemble with the element Lame values (float64 constants: the model paints float32 E / nu and derives
    them per cell) and the density."""
    from open_sol.core.constants import FACES_3D
    from open_sol.core.domain3d import Domain3D
    from open_sol.physics.mechanics.system3d import ElasticSystem3D
    from open_sol.solvers.pcg import PCGSolver, resolve_max_iterations
    eigen = problem.eigen
    cells, cell = eigen['cells'], eigen['cell']
    domain = Domain3D.from_widths(*(np.full(count, cell) for count in cells), {},
                                  *(count * cell for count in cells), backend=backend)
    values = automatic_linear_values(backend)
    max_iterations = int(values['max_iterations']) or 2 * resolve_max_iterations(0, domain)
    linear_solver = PCGSolver(precision=values['precision'], tolerance=values['tolerance'],
                              max_iterations=max_iterations, label=_modes_label, backend=backend,
                              preconditioner=values['preconditioner'])
    # precision / tol / max_iter are the linear solver's (ElasticSystemBase takes them from it)
    system = ElasticSystem3D('float64', values['tolerance'], max_iterations, label=_modes_label,
                             linear_solver=linear_solver, backend=backend)
    system.prepare(domain)
    for face in FACES_3D:
        system.hold_face(face, **{f'u_{face[0]}': 0.0})
    xp = system.xp
    shape = system.element_shape
    system.assemble(xp.full(shape, eigen['lame_lambda'], dtype=np.float64),
                    xp.full(shape, eigen['mu'], dtype=np.float64), xp.ones(shape, dtype=bool))
    density = xp.full(shape, eigen['density'], dtype=system.dtype)
    return ModesSystem(system, density, dict(values, max_iterations=max_iterations))


def _modes_label():
    return 'profiler solid_modes_3d'


PROBLEMS = {problem.name: problem for problem in (
    ProblemClass('fdfd_driven', '2D FDFD E_z drive of a dielectric disc (complex Helmholtz, CPML; MUMPS / PARDISO '
                 'take the LDL^T of the symmetrised D A)', 'em_fdfd_ez', 'linear', 'stationary', 'tensor', 2,
                 f'cells per axis (window {FDFD_WINDOW * 1e6:g} um, {FDFD_PML_CELLS} PML cells per side)',
                 (100, 200, 400), (60, 120), fdfd_driven),
    ProblemClass('fdfd_eigen', f'2D FDFD E_z eigenfrequency: {FDFD_RESONANCES} disc resonances nearest '
                 f'{FDFD_WAVELENGTH * 1e9:g} nm (shift-invert ARPACK, eigenfrequency.solve_resonances)',
                 'em_fdfd_ez', 'eigen', 'eigenfrequency', 'tensor', 2,
                 f'cells per axis (window {FDFD_WINDOW * 1e6:g} um, {FDFD_PML_CELLS} PML cells per side)',
                 (80, 160, 240), (60,), fdfd_eigen),
    ProblemClass('heat_2d', '2D finite-volume heat conduction (real SPD 5-point, contrast inclusion)', 'heat', 'linear',
                 'stationary', 'tensor', 2, 'cells per axis', (128, 256, 512), (64, 128), heat_2d),
    ProblemClass('heat_3d', '3D finite-volume heat conduction (real SPD 7-point, contrast inclusion)', 'heat',
                 'linear', 'stationary', 'tensor_3d', 3, 'cells per axis', (16, 24, 32), (12, 16), heat_3d),
    ProblemClass('heat_fem_3d', f'3D P{FEM_ORDER} FEM heat conduction on Kuhn tetrahedra (real SPD, contrast '
                 'inclusion)', 'heat', 'linear', 'stationary', 'tetrahedra', 3, 'brick cells per axis (6 tetrahedra '
                 'each)', (8, 12, 16), (6, 8), heat_fem_3d),
    ProblemClass('heat_line', '1D model heat conduction (finite volume, one cell across: tridiagonal SPD)', 'heat',
                 'linear', 'stationary', 'line', 1, 'cells', (1000, 10000, 50000), (1000, 10000), heat_line),
    # arpack's factorization: no factor on brick meshes (eigsh_inverse with 'linear' PCG solves), one row
    ProblemClass('solid_modes_3d', f'3D brick elasticity vibration modes: the {MODES_RESONANCES} lowest modes of a '
                 'steel roller box (ElasticSystem3D.lowest_modes: matrix-free shift-invert Lanczos with Automatic '
                 "'linear' PCG solves, or block_eigen; frequencies against the exact discrete closed form; the "
                 'arpack factorization is ignored here)', MODES_PHYSICS_TYPE, 'eigen', MODES_STUDY, MODES_GRID, 3,
                 f'cells along y (x: one more, z: one less; even; y side {MODES_SIDE * 1e3:g} mm)', (16, 32, 48, 64),
                 (8, 16), solid_modes_3d, ignored_settings={'arpack_shift_invert': ('factorization',)}),
)}


def build(name, size) -> Problem:
    """Problem of class `name` at `size` (KeyError for an unknown class)."""
    return PROBLEMS[name].builder(size)
