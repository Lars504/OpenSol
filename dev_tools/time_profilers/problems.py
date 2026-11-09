"""Problem classes of the CPU solver profiler: each assembles one representative system with open_sol's own operator
code (2D FDFD Helmholtz, finite-volume and P2 finite-element heat conduction) at a size knob and names the physics
type, solver role, study and grid kind whose solver choices it stands for. open_sol is imported inside the builders
only (the parent process reads the metadata alone)."""
# Unknown order is open_sol's: FDFD / finite volume flat i * ny (* nz) + j over the full grid (PML included), FEM the
# P2 dof numbering (nodes, then edge midpoints) with the Dirichlet dofs eliminated (fem.assembly.eliminate_dirichlet).
# Operators are host scipy CSR (the cpu backend's): FDFD complex128 (general; D A complex symmetric with D the stretched
# cell areas, the symmetrizer the FDFD physics hands MUMPS / PARDISO / cuDSS), heat float64 SPD.
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


@dataclass(frozen=True, eq=False)
class Problem:
    """One assembled system: host scipy CSR operator, rhs (None for an eigen problem), the grid shape (FDFD: (nx, ny)
    of SparseDirectSolver.factorize / the line-block layout; finite volume: the window shape, DiffusionDirectSolver's
    nested-dissection shape in 3D; FEM: None), the FDFD symmetrizer D (None elsewhere), the longest axis in cells (pcg
    auto iteration limit) and the eigen data (mass diagonal, target omega, energy weights, window rows, resonances)."""
    kind: str                     # 'complex' (FDFD drive) | 'spd' (heat) | 'eigen' (FDFD shift-invert)
    operator: object
    rhs: object = None
    grid_shape: tuple | None = None
    symmetrizer: object = None
    longest_axis: int = 0
    eigen: dict = field(default_factory=dict)

    @property
    def n_unknowns(self) -> int:
        return int(self.operator.shape[0])

    @property
    def nnz(self) -> int:
        return int(self.operator.nnz)


@dataclass(frozen=True, eq=False)
class ProblemClass:
    """Metadata of one problem class: the physics type / role / study / grid kind whose solver choices
    (solver_catalog) it profiles, the size knob and its default / --quick sizes, and builder(size) -> Problem."""
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
)}


def build(name, size) -> Problem:
    """Problem of class `name` at `size` (KeyError for an unknown class)."""
    return PROBLEMS[name].builder(size)
