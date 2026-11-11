"""2D FDFD (em_fdfd_ez / em_fdfd_hz) on triangle meshes: the scalar Helmholtz equation of E_z or H_z on Lagrange
elements of degree 1 or 2, the PML ring as a complex coordinate stretch at the quadrature points, driven point loads
and the eigen pencil of eigenfrequency.solve_resonances. Host numpy (scipy inside functions)."""
# Formulation (exp(+j omega t), FDTD-sign currents J_F / M_F as harmonic.py; unconjugated Galerkin, so complex
# symmetric with loss / PML: D = I for the cuDSS LDL^T):
#   ez: K = int (nu grad E_z) . grad v,  M = int eps_r E_z v,  nu = 1 / mu;  (K - k0^2 M) E_z = j omega mu0 int J_zF v
#   hz: K = int (T grad H_z) . grad v,   M = int mu H_z v,     T = eps_t / det(eps_t) (the in-plane tensor with eps_xy:
#       the curl of eta R grad H with R v = (v_y, -v_x) gives R eta R^T = eps_t / det eps_t);
#       (K - k0^2 M) H_z = j omega eps0 int M_zF v + int (R eta J_F) . grad v  (an electric current by parts)
# The FDFD (Yee) operator is -(K - k0^2 M) of the same continuum problem, so a point current of I radiates E_z =
# (omega mu0 I / 4) H0^(2)(k r) and a magnetic current K_m radiates H_z = (omega eps0 eps_r K_m / 4) H0^(2)(k r), the
# amplitudes of the Yee sources. In-plane fields: ez H = R grad E_z / (-j omega mu0 mu); hz E = eta R grad H_z /
# (j omega eps0) (unstretched values in the PML ring: display only).
# PML ring (fem_modes.pml_ring): s_x, s_y of the tensor CPML grading at the quadrature points; K takes T_xx s_y / s_x,
# T_yy s_x / s_y, T_xy unchanged, M takes s_x s_y. The ring's outer side is pec (the industry-standard exterior, as
# fem_modes). Walls (the Yee _WALL_SIGNS as essential / natural conditions; mirror planes = the window edge):
#   ez: E_z = 0 on pec / antisymmetric faces, the ring's outer side and the closure of PEC elements (vertex contacts
#       too); pmc / symmetric faces natural.
#   hz: H_z = 0 on pmc / symmetric faces; pec / antisymmetric faces, the ring's outer side and PEC outlines natural.
# Eigen pencil (solve_resonances' A = L + k_t^2 M with L = -K: the negated A = K - k_t^2 M and mass -M, as fem_bor):
# 'shift_invert' (no gradient null space; an hz Neumann constant has omega 0 and is dropped there).
import math
import time
from dataclasses import dataclass, field

import numpy as np

from ...core.constants import C0, EPS0, MU0
from ...core.memory_estimate import host_lu_bytes
from ...fem.assembly import CHUNK_ELEMENTS, assemble_mass, assemble_stiffness, restrict
from ...fem.dofs import lagrange_dofs
from ...fem.elements import element_geometry, lagrange_basis, physical_gradients
from ...fem.fixed_pattern import scatter_add
from ...fem.indicators import CHUNK_ITEMS, add_edge_terms, edge_sides, lagrange_at
from ...fem.mesh import FACE_BITS
from ...fem.quadrature import TRIANGLE_DEGREES, triangle_rule
from ...fem.sampling import locator
from ...solvers.arpack_shift_invert import CUDSS_FACTORIZATIONS, ArpackShiftInvertSolver
from ...solvers.sparse_direct import SPLU_LIMIT_UNKNOWNS, SPLU_WARN_UNKNOWNS
from .boundaries import BCType
from .eigenfrequency import solve_resonances
from .fem_modes import _LOCAL_NODES, PERIODIC_TEXT, host_solver_label, pml_ring

POLARIZATIONS = ('ez', 'hz')
# faces whose condition holds the out-of-plane unknown at 0 (a cpml face: the ring's pec outer side); others natural
_ESSENTIAL_FACES = {'ez': frozenset({BCType.PEC, BCType.ANTISYMMETRIC, BCType.CPML}),
                    'hz': frozenset({BCType.PMC, BCType.SYMMETRIC})}
_PERIODIC_TYPES = frozenset({BCType.PERIODIC, BCType.BLOCH})
FEM_FDFD_DEGREE = TRIANGLE_DEGREES[-1]   # rule of every assembly: the PML stretch is no polynomial
SAMPLE_DEGREE = 4                        # rule of the element means (state fields, q_abs)
_CENTROID = np.array([[1.0 / 3.0, 1.0 / 3.0]])
_WINDOW_TOLERANCE = 1e-9                 # x the longest window side: a dof point on the window edge is inside
# nonzeros per row of the assembled scalar operator and SuperLU L + U over the 5-point heuristic of
# core.memory_estimate.host_lu_bytes: uncalibrated first figures (compare the summary 'solvers' factor_host_bytes)
FEM_FDFD_ENTRIES_PER_ROW = {1: 7, 2: 13}
FEM_FDFD_LU_FILL = {1: 1.5, 2: 3.0}
FEM_FDFD_SIZE_REMEDY = ('coarsen the triangle mesh (larger maximum element size), use discretization linear, or halve '
                        'the domain by a mirror face')


# ---- unknown counts and sizes (CPU, also the estimators) ----
def scalar_unknowns(n_nodes, n_edges, order) -> int:
    """Lagrange unknowns of a mesh of these counts (walls and PEC elements remove some): nodes, + edges at degree 2."""
    return int(n_nodes + (n_edges if order == 2 else 0))


def fem_fdfd_lu_bytes(n_unknowns, order) -> int:
    """Heuristic host bytes of the complex SuperLU factor of the scalar operator."""
    return int(FEM_FDFD_LU_FILL[order] * host_lu_bytes(n_unknowns))


def fem_fdfd_size_issue(n_unknowns, order, factorization='superlu'):
    """(level, text) when the host SuperLU factor of n_unknowns is near ('warning') or over ('error') the sparse_direct
    limits (unknowns x FEM_FDFD_LU_FILL as grid unknowns; a mumps / pardiso factorization or back end alike, texts
    naming it), else None; None for a cuDSS factorization or back end."""
    if factorization in CUDSS_FACTORIZATIONS:
        return None
    solver = host_solver_label(factorization)
    equivalent = float(n_unknowns) * FEM_FDFD_LU_FILL[order]
    text = (f'{int(n_unknowns):,} FEM unknowns (about {FEM_FDFD_LU_FILL[order]:g}x the SuperLU fill of a grid unknown, '
            f'so ~{int(equivalent):,} grid unknowns)')
    if equivalent > SPLU_LIMIT_UNKNOWNS:
        return 'error', (f'{text} exceed the {SPLU_LIMIT_UNKNOWNS:,} limit of the host {solver}: its '
                         f'factors would fill the host memory; {FEM_FDFD_SIZE_REMEDY}')
    if equivalent > SPLU_WARN_UNKNOWNS:
        return 'warning', (f'{text} exceed ~{SPLU_WARN_UNKNOWNS:,}, the practical size of the host {solver}; '
                           f'{FEM_FDFD_SIZE_REMEDY}')
    return None


# ---- layout: dofs and walls ----
@dataclass(frozen=True, eq=False)
class FEMFDFDLayout:
    """Unknowns of one mesh, element degree, polarisation, wall set and PEC element set: the Lagrange map of the non-PEC
    elements, its free dofs, the PML ring, the window flag and the unit mass of the free dofs (the PML share)."""
    mesh: object
    order: int
    polarization: str
    ring: object                  # fem_modes.PMLRing
    dofs: object                  # fem.dofs.DofMap (Lagrange, geometry order = order)
    free: np.ndarray              # int64 free dof indices
    in_window: np.ndarray         # bool (n_free,): dof point inside the window
    energy: object                # real scipy CSR int u v dA on the free dofs
    # point (x, y) -> _basis_at result (located once per layout: sources / probes re-read it every solve; read-only)
    point_bases: dict = field(default_factory=dict, repr=False)

    @property
    def n_unknowns(self) -> int:
        return int(self.free.size)

    def coefficients(self, free_values) -> np.ndarray:
        """Complex128 (dofs.n_dofs,) coefficients of free-dof values, fixed dofs 0."""
        coefficients = np.zeros(self.dofs.n_dofs, dtype=np.complex128)
        coefficients[self.free] = free_values
        return coefficients


def fem_fdfd_layout(mesh, order, boundary_config, polarization, pec=None) -> FEMFDFDLayout:
    """FEMFDFDLayout of element degree order (1 / 2) with the walls of boundary_config (faces, CPML settings) and the bool
    per-element PEC mask (None = none); ValueError for periodic / Bloch faces or when every element is PEC."""
    if order not in (1, 2):
        raise ValueError(f'fem_fdfd_layout: order must be 1 or 2, got {order!r}')
    if polarization not in POLARIZATIONS:
        raise ValueError(f"fem_fdfd_layout: polarization must be 'ez' or 'hz', got {polarization!r}")
    faces = {face: getattr(boundary_config, face) for face in FACE_BITS}
    if any(kind in _PERIODIC_TYPES for kind in faces.values()):
        raise ValueError(f'em_fdfd_{polarization} on triangles: {PERIODIC_TEXT}')
    ring = pml_ring(mesh, boundary_config)
    pec = np.zeros(mesh.n_elements, dtype=bool) if pec is None else np.asarray(pec, dtype=bool)
    if pec.shape != (mesh.n_elements,):
        raise ValueError(f'fem_fdfd_layout: PEC mask shape {pec.shape} != ({mesh.n_elements},)')
    if pec.all():
        raise ValueError(f'em_fdfd_{polarization} on triangles: every element is PEC')
    dofs = lagrange_dofs(mesh, order, ~pec, geometry_order=order)
    essential = _ESSENTIAL_FACES[polarization]
    empty = np.zeros(0, dtype=np.int64)

    def tagged(kinds_wanted):
        edges = [np.asarray(mesh.edge_tags.get(face, ()), dtype=np.int64) for face, kind in faces.items()
                 if kinds_wanted(kind)]
        return np.concatenate(edges) if edges else empty

    if polarization == 'ez':   # every outline of the non-PEC set but the natural faces, plus the PEC closure
        fixed = dofs.edge_dofs(np.setdiff1d(dofs.boundary_edges(), tagged(lambda kind: kind not in essential)))
        if pec.any():
            fixed = np.union1d(fixed, dofs.node_dofs(np.unique(mesh.triangles[pec])))
    else:                      # only the essential outer faces (PEC outlines are natural)
        fixed = dofs.edge_dofs(tagged(lambda kind: kind in essential))
    keep = np.ones(dofs.n_dofs, dtype=bool)
    keep[fixed] = False
    free = np.flatnonzero(keep)
    x0, y0, x1, y1 = (float(value) for value in mesh.window)
    tolerance = _WINDOW_TOLERANCE * max(x1 - x0, y1 - y0)
    points = dofs.dof_points()[free]
    in_window = ((points[:, 0] >= x0 - tolerance) & (points[:, 0] <= x1 + tolerance)
                 & (points[:, 1] >= y0 - tolerance) & (points[:, 1] <= y1 + tolerance))
    energy = restrict(assemble_mass(dofs, 1.0), free)
    return FEMFDFDLayout(mesh=mesh, order=int(order), polarization=polarization, ring=ring, dofs=dofs, free=free,
                         in_window=in_window, energy=energy)


# ---- materials and assembly ----
@dataclass(frozen=True, eq=False)
class FEMFDFDMaterials:
    """Host per-element values at one omega (sigma_e, sigma_m and poles included, Im < 0 lossy): eps = complex128 (n,)
    eps_r of E_z (ez) or the (n, 2, 2) in-plane tensor of E_x / E_y (hz), mu complex128 (n,), bool PEC elements."""
    eps: np.ndarray
    mu: np.ndarray
    pec: np.ndarray

    def __post_init__(self):
        pec = np.asarray(self.pec)
        if pec.dtype != np.bool_ or pec.ndim != 1:
            raise TypeError(f'FEMFDFDMaterials pec must be a 1D bool array, got {pec.dtype} {pec.shape}')
        n = pec.size
        eps, mu = np.asarray(self.eps), np.asarray(self.mu)
        if eps.shape not in ((n,), (n, 2, 2)) or mu.shape != (n,):
            raise ValueError(f'FEMFDFDMaterials eps {eps.shape} / mu {mu.shape} do not match {n} elements')
        usable = ~pec
        determinant = eps[usable] if eps.ndim == 1 else (eps[usable, 0, 0] * eps[usable, 1, 1]
                                                         - eps[usable, 0, 1] * eps[usable, 1, 0])
        for name, values in (('eps', determinant), ('mu', mu[usable])):
            if not (np.all(np.isfinite(values)) and np.all(values != 0)):
                raise ValueError(f'FEMFDFDMaterials {name} must be finite and non-singular on non-PEC elements')

    @property
    def tensor(self) -> bool:
        return np.ndim(self.eps) == 3

    def inverse_eps(self, elements) -> np.ndarray:
        """Complex (k, 2, 2) eta = eps_t^-1 of hz elements (closed-form 2 x 2 inverse in components: no per-matrix
        LAPACK loop of np.linalg.inv)."""
        eps = np.asarray(self.eps)[elements]
        xx, xy, yx, yy = eps[:, 0, 0], eps[:, 0, 1], eps[:, 1, 0], eps[:, 1, 1]
        inverse_determinant = 1.0 / (xx * yy - xy * yx)   # non-singular: checked in __post_init__
        return np.stack((np.stack((yy, -xy), axis=-1), np.stack((-yx, xx), axis=-1)), axis=-2) * \
            inverse_determinant[:, None, None]


def _coefficients(layout, materials, omega):
    """(stiffness tensor callable -> (m, q, 2, 2), mass callable -> (m, q)) of the layout's polarisation at omega with
    the PML stretch (module header)."""
    ring = layout.ring
    usable = ~np.asarray(materials.pec)
    base = np.zeros((usable.size, 2, 2), dtype=np.complex128)
    if layout.polarization == 'ez':
        nu = np.ones(usable.size, dtype=np.complex128)
        nu[usable] = 1.0 / np.asarray(materials.mu, dtype=np.complex128)[usable]
        base[:, 0, 0] = base[:, 1, 1] = nu
        mass_material = np.asarray(materials.eps, dtype=np.complex128)
    else:
        eps = np.asarray(materials.eps, dtype=np.complex128)
        determinant = eps[:, 0, 0] * eps[:, 1, 1] - eps[:, 0, 1] * eps[:, 1, 0]
        base[usable] = eps[usable] / determinant[usable, None, None]
        mass_material = np.asarray(materials.mu, dtype=np.complex128)

    def stiffness(points, elements):
        s_x, s_y = ring.stretch(points, omega)
        tensor = np.array(np.broadcast_to(base[elements][:, None], s_x.shape + (2, 2)))
        tensor[..., 0, 0] *= s_y / s_x
        tensor[..., 1, 1] *= s_x / s_y
        return tensor

    def mass(points, elements):
        s_x, s_y = ring.stretch(points, omega)
        return mass_material[elements][:, None] * s_x * s_y

    return stiffness, mass


@dataclass(frozen=True, eq=False)
class FEMFDFDSystem:
    """One assembled pencil at omega on the free dofs: operator A = K - k0^2 M (complex symmetric scipy CSR), mass M,
    equilibration (|diag K| + k0^2 |diag M|)^-1/2 (the cuDSS row scale; diag A may nearly cancel), its materials."""
    layout: FEMFDFDLayout
    materials: FEMFDFDMaterials
    omega: float
    operator: object
    mass: object
    equilibration: np.ndarray
    assembly_seconds: float


def assemble_fem_fdfd(layout, materials, omega) -> FEMFDFDSystem:
    """FEMFDFDSystem of the layout at omega [rad/s] with per-element FEMFDFDMaterials (the layout's PEC elements)."""
    started = time.perf_counter()
    omega = float(omega)
    if not (math.isfinite(omega) and omega > 0.0):
        raise ValueError(f'assemble_fem_fdfd needs a finite angular frequency omega > 0 [rad/s], got {omega!r}')
    if np.shape(materials.pec) != (layout.mesh.n_elements,):
        raise ValueError(f'materials of {np.shape(materials.pec)} elements on a mesh of {layout.mesh.n_elements}')
    if materials.tensor != (layout.polarization == 'hz'):
        raise ValueError(f'em_fdfd_{layout.polarization} needs eps as ' +
                         ('an (n, 2, 2) in-plane tensor' if layout.polarization == 'hz' else 'an (n,) array'))
    stiffness_coefficient, mass_coefficient = _coefficients(layout, materials, omega)
    stiffness = restrict(assemble_stiffness(layout.dofs, stiffness_coefficient, degree=FEM_FDFD_DEGREE), layout.free)
    mass = restrict(assemble_mass(layout.dofs, mass_coefficient, degree=FEM_FDFD_DEGREE), layout.free)
    k0_squared = (omega / C0) ** 2
    scale = np.abs(stiffness.diagonal()) + k0_squared * np.abs(mass.diagonal())
    equilibration = 1.0 / np.sqrt(np.where(scale > 0.0, scale, 1.0))
    return FEMFDFDSystem(layout=layout, materials=materials, omega=omega,
                         operator=(stiffness - k0_squared * mass).tocsr(), mass=mass, equilibration=equilibration,
                         assembly_seconds=time.perf_counter() - started)


def solve_fem_fdfd_resonances(system, n_resonances, eigen_solver=None, warnings=(), in_region=None,
                              min_region_fraction=0.0):
    """eigenfrequency.EigenSolve of up to n_resonances resonances nearest system.omega (shift-invert by eigen_solver,
    None = default settings: host SuperLU, or a cuDSS LDL^T of the complex symmetric pencil, cudss_mixed with the
    float64 Rayleigh polish); Resonance.field = free-dof values (layout.coefficients maps them); in_region (n_free,)
    bool / min_region_fraction: solve_resonances' region filter."""
    eigen_solver = ArpackShiftInvertSolver() if eigen_solver is None else eigen_solver
    if system.layout.n_unknowns < 4:
        raise ValueError(f'solve_fem_fdfd_resonances: only {system.layout.n_unknowns} unknowns remain inside the walls')
    return solve_resonances(system.operator, -system.mass, system.omega, system.layout.energy, system.layout.in_window,
                            n_resonances, warnings=warnings, eigen_solver=eigen_solver, symmetric=True,
                            equilibration=system.equilibration, in_region=in_region,
                            min_region_fraction=min_region_fraction)


# ---- point loads and point values ----
def _basis_at(layout, point):
    """(element, its dofs (n,), basis values (n,), physical gradients (n, 2)) at a point [m], None when it lies outside
    the mesh or in a PEC element; kept per layout (layout.point_bases)."""
    key = (float(point[0]), float(point[1]))
    if key not in layout.point_bases:
        layout.point_bases[key] = _located_basis(layout, key)
    return layout.point_bases[key]


def _located_basis(layout, point):
    """_basis_at of one point, located afresh."""
    location = locator(layout.mesh, layout.dofs.geometry_order).locate(np.asarray(point, dtype=np.float64).reshape(1, 2))
    element = int(location.element[0])
    if element < 0 or layout.dofs.element_position[element] < 0:
        return None
    reference = location.reference[:1][:, None, :]   # (1, 1, 2)
    values, reference_gradients = lagrange_basis(layout.order, reference)
    geometry = element_geometry(layout.mesh, np.array([element]), reference, layout.dofs.geometry_order)
    gradients = physical_gradients(reference_gradients, geometry.inverse)
    return (element, layout.dofs.element_dofs[layout.dofs.element_position[element]],
            np.asarray(values).reshape(-1), np.asarray(gradients).reshape(-1, 2))


def _free_load(layout, dofs, values):
    """Complex (n_free,) of local values scattered onto dofs."""
    full = np.zeros(layout.dofs.n_dofs, dtype=np.complex128)
    scatter_add(full, dofs, values)
    return full[layout.free]


def point_load(layout, point, weight) -> np.ndarray:
    """Complex (n_free,) weight phi_i(point) (a point current; zeros outside the non-PEC elements)."""
    basis = _basis_at(layout, point)
    if basis is None:
        return np.zeros(layout.n_unknowns, dtype=np.complex128)
    _, dofs, values, _ = basis
    return _free_load(layout, dofs, complex(weight) * values)


def dipole_load(layout, materials, point, moment) -> np.ndarray:
    """Complex (n_free,) (R eta p) . grad phi_i(point) of an in-plane FDTD-sign electric current moment p = (p_x, p_y)
    [A] (hz only; eta of the containing element, unstretched; zeros outside the non-PEC elements)."""
    basis = _basis_at(layout, point)
    if basis is None:
        return np.zeros(layout.n_unknowns, dtype=np.complex128)
    element, dofs, _, gradients = basis
    field = materials.inverse_eps(np.array([element]))[0] @ np.asarray(moment, dtype=np.complex128)
    turned = np.array([field[1], -field[0]])
    return _free_load(layout, dofs, gradients @ turned)


def _in_plane(layout, materials, elements, gradients, omega):
    """Complex (k, q, 2) in-plane field of out-of-plane gradients (k, q, 2) on elements (k,): ez (H_x, H_y), hz (E_x,
    E_y) (module header; unstretched)."""
    turned = np.stack([gradients[..., 1], -gradients[..., 0]], axis=-1)   # R grad u
    if layout.polarization == 'ez':
        return turned / (-1j * omega * MU0 * np.asarray(materials.mu)[elements][:, None, None])
    return np.einsum('kab,kqb->kqa', materials.inverse_eps(elements), turned) / (1j * omega * EPS0)


def point_fields(system, coefficients, point) -> dict:
    """{out-of-plane, in-plane x, in-plane y} complex phasors at a point [m]: ez {'ez', 'hx', 'hy'}, hz {'hz', 'ex',
    'ey'} of full dof coefficients; zeros outside the non-PEC elements."""
    layout = system.layout
    names = ('ez', 'hx', 'hy') if layout.polarization == 'ez' else ('hz', 'ex', 'ey')
    basis = _basis_at(layout, point)
    if basis is None:
        return dict.fromkeys(names, 0j)
    element, dofs, values, gradients = basis
    local = coefficients[dofs]
    in_plane = _in_plane(layout, system.materials, np.array([element]), (gradients.T @ local)[None, None, :],
                         system.omega)[0, 0]
    return {names[0]: complex(values @ local), names[1]: complex(in_plane[0]), names[2]: complex(in_plane[1])}


# ---- element and nodal fields ----
def element_state(system, coefficients) -> tuple:
    """(complex (n_elements,) out-of-plane value at each element centroid, float64 (n_elements,) element mean of q_abs =
    1/2 omega eps0 (-Im eps) |E|^2 [W/m^3] (hz: per in-plane component of a diagonal-loss eps)); 0 on PEC elements."""
    layout, materials, omega = system.layout, system.materials, system.omega
    mesh, dofs = layout.mesh, layout.dofs
    centroid_values = np.zeros(mesh.n_elements, dtype=np.complex128)
    absorbed = np.zeros(mesh.n_elements)
    rule = triangle_rule(SAMPLE_DEGREE)
    centroid_basis, _ = lagrange_basis(layout.order, _CENTROID)
    eps = np.asarray(materials.eps)
    for start in range(0, len(dofs.elements), CHUNK_ELEMENTS):
        elements = dofs.elements[start:start + CHUNK_ELEMENTS]
        local = coefficients[dofs.element_dofs[start:start + CHUNK_ELEMENTS]]
        centroid_values[elements] = local @ np.asarray(centroid_basis).reshape(-1)
        reference = np.broadcast_to(rule.points, (len(elements),) + rule.points.shape)
        values, gradients = lagrange_at(dofs, coefficients, elements, reference)
        weights = rule.weights[None, :] * np.abs(element_geometry(mesh, elements, reference,
                                                                  dofs.geometry_order).determinant)
        if layout.polarization == 'ez':
            density = -eps[elements][:, None].imag * np.abs(values) ** 2
        else:
            field = _in_plane(layout, materials, elements, gradients, omega)
            density = (-eps[elements, 0, 0][:, None].imag * np.abs(field[..., 0]) ** 2
                       - eps[elements, 1, 1][:, None].imag * np.abs(field[..., 1]) ** 2)
        absorbed[elements] = 0.5 * omega * EPS0 * (density * weights).sum(axis=1) / weights.sum(axis=1)
    return centroid_values, absorbed


def nodal_fields(system, coefficients, nodal_dofs) -> dict:
    """Complex (nodal_dofs.n_dofs,) display fields on a full-mesh Lagrange map of the layout's degree: 'u' the
    out-of-plane unknown (exact, 0 at nodes of PEC elements only), 'fx' / 'fy' the in-plane field averaged over the
    non-PEC elements sharing a node."""
    layout = system.layout
    dofs = layout.dofs
    u = np.zeros(nodal_dofs.n_dofs, dtype=np.complex128)
    u[np.searchsorted(nodal_dofs.global_ids, dofs.global_ids)] = coefficients
    local_nodes = _LOCAL_NODES[layout.order]
    totals = np.zeros((nodal_dofs.n_dofs, 2), dtype=np.complex128)
    counts = np.zeros(nodal_dofs.n_dofs)
    for start in range(0, len(dofs.elements), CHUNK_ELEMENTS):
        elements = dofs.elements[start:start + CHUNK_ELEMENTS]
        reference = np.broadcast_to(local_nodes, (len(elements),) + local_nodes.shape)
        _, gradients = lagrange_at(dofs, coefficients, elements, reference)
        field = _in_plane(layout, system.materials, elements, gradients, system.omega)
        targets = nodal_dofs.element_dofs[nodal_dofs.element_position[elements]].ravel()
        counts += np.bincount(targets, minlength=nodal_dofs.n_dofs)
        for component in (0, 1):
            scatter_add(totals[:, component], targets, field[..., component])
    averaged = np.where(counts[:, None] > 0, totals / np.maximum(counts, 1.0)[:, None], 0.0)
    return {'u': u, 'fx': averaged[:, 0], 'fy': averaged[:, 1]}


def fem_fdfd_error_indicator(system, coefficients) -> np.ndarray:
    """eta^2 per mesh element (float64 (n_elements,)) for mesh refinement: Kelly jumps of the tangential in-plane field
    (ez H . t, hz E . t) across interior window edges between non-PEC elements, 1/2 h_E int_E w |[F_t]|^2 ds to each side
    (w = mu0 |mu| for ez, eps0 |eps| for hz); outer, PEC and PML edges carry none."""
    layout, materials = system.layout, system.materials
    mesh = layout.mesh
    inside = np.zeros(mesh.n_elements, dtype=bool)
    inside[layout.dofs.elements] = True
    inside &= mesh.pml_faces == 0
    if layout.polarization == 'ez':
        weight = MU0 * np.abs(np.asarray(materials.mu))
    else:
        eps = np.asarray(materials.eps)
        weight = EPS0 * 0.5 * (np.abs(eps[:, 0, 0]) + np.abs(eps[:, 1, 1]))
    eta_squared = np.zeros(mesh.n_elements)
    candidates = np.unique(mesh.element_edges[inside])
    for start in range(0, candidates.size, CHUNK_ITEMS):
        sides = edge_sides(mesh, candidates[start:start + CHUNK_ITEMS], inside, layout.order)
        sides = sides.select(sides.two_sided)
        if not len(sides.edges):
            continue
        tangent = np.stack([-sides.normals[..., 1], sides.normals[..., 0]], axis=-1)
        along = []
        for side in (0, 1):
            elements = sides.elements[:, side]
            _, gradients = lagrange_at(layout.dofs, coefficients, elements, sides.reference_points(side))
            field = _in_plane(layout, materials, elements, gradients, system.omega)
            along.append(np.einsum('kqc,kqc->kq', field, tangent))
        jump = (np.abs(along[0] - along[1]) ** 2 * sides.weights).sum(axis=1)
        terms = np.stack([weight[sides.elements[:, side]] * jump for side in (0, 1)], axis=1)
        add_edge_terms(eta_squared, sides, terms)
    return eta_squared
