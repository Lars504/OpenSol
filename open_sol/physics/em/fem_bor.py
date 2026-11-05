"""Body-of-revolution resonances on triangle meshes (em_fdfd_bor on a 'triangles' mesh): mixed Nedelec (E_r, E_z) /
Lagrange (w = r u) elements of degree 1 or 2 per azimuthal order, a cylindrical PML ring, squared-Cayley shift-invert
through eigenfrequency.solve_resonances. Host numpy (scipy inside functions)."""
# Formulation (r = x - the window's x_lo edge = the axis, z = y, fields ~ exp(j omega t - j m phi), u = -j E_phi as in
# body_of_revolution: every entry is real without loss / PML). Unknowns e = (E_r, E_z) (Nedelec, covariant Piola) and
# w = r u (Lagrange of the same degree: the mixed axisymmetric element of the literature, r E_phi nodal and E_t on
# edges). With G = grad w + m e and rot e = dE_z/dr - dE_r/dz, curl E = (-j G_z / r, -rot e, j G_r / r) in (r, phi, z),
# so per radian
#   K(m) = int [ nu G . G / r + nu r rot(e)^2 ] dr dz,   M = int [ r (eps_rr E_r^2 + eps_zz E_z^2) + eps_phi w^2 / r ],
# K x = k^2 M x (unconjugated Galerkin: complex symmetric with loss / PML, D = I). K(m) = K0 + m K1 + m^2 K2 as the Yee
# operator: K0 = (e, e) nu r rot rot + (w, w) nu grad . grad / r, K1 = (e, w) nu e . grad / r + its transpose, K2 =
# (e, e) nu e . e / r. Discrete gradients (e = grad p, w = -m p, p Lagrange of the degree: grad P_k lies in N_k) give
# G = 0 and rot e = 0 exactly: the static null space, sent to 0 by the squared Cayley map and dropped.
# Axis (outer edges on r = 0): w = 0 at every order (its axis dofs fixed); |m| >= 1 also E_z = 0 (tangential axis dofs).
# The 1 / r integrands are log-singular on elements with an axis edge unless G_r -> 0 there (the regularity u(0) = -m
# E_r(0) at |m| = 1, dw/dr = 0 at m = 0 and |m| >= 2); the Dunavant points are interior, so the large 1 / r weight acts
# as that constraint (a penalty, not exact: degree-1 elements lock w and rot e in the axis layer, hence quadratic by
# default; fields of |m| >= 2 vanish near the axis anyway).
# Walls: pec faces and the outer side of the PML ring (COMSOL's exterior, as fem_modes; the Yee path ends a cpml layer
# in a pmc edge instead) fix tangential e and w (E_phi is tangential to every (r, z) boundary); pmc faces are natural;
# PEC elements leave the unknowns, their closure fixes tangential e and w (vertex contacts too). No mirror / periodic
# faces (pec / pmc on z_lo are the mirrors).
# PML (fem_modes.pml_ring: the r_hi and z rings): s_r(r), s_z(z) and the stretched radius r~ = int_0^r s_r dr'
# (Gauss-Legendre over the ring depth). In stretched coordinates with e = (s_r E~_r, s_z E~_z) (covariant) and w as is:
#   K = int [ G^T diag(nu s_z / s_r, nu s_r / s_z) G / r~ + nu r~ rot(e)^2 / (s_r s_z) ],
#   M = int [ r~ (eps_rr s_z / s_r e_r^2 + eps_zz s_r / s_z e_z^2) + eps_phi s_r s_z w^2 / r~ ],
# the cylindrical PML (r~ in every metric factor, as body_of_revolution). Recovery (stretched values in the ring, H at
# the target omega): E_r = e_r / s_r, E_z = e_z / s_z, E_phi = j w / r~, H_r = G_z / (s_z r~ omega mu0 mu), H_z = -G_r /
# (s_r r~ omega mu0 mu), H_phi = -j rot e / (s_r s_z omega mu0 mu).
# PML share (eigenfrequency.solve_resonances): energy matrix W = int [ r e . e + w^2 / r ] unstretched, in_window = dof
# points inside the window.
import math
import time
from dataclasses import dataclass

import numpy as np

from ...core.constants import C0, EPS0, MU0
from ...fem.assembly import (assemble_curl_curl, assemble_gradient_coupling, assemble_mass, assemble_stiffness,
                             assemble_vector_mass, restrict)
from ...fem.dofs import lagrange_dofs, nedelec_dofs
from ...fem.elements import element_geometry
from ...fem.fixed_pattern import scatter_add
from ...fem.indicators import lagrange_at, nedelec_at
from ...fem.mesh import FACE_BITS
from ...fem.quadrature import TRIANGLE_DEGREES, triangle_rule
from ...solvers.arpack_shift_invert import ArpackShiftInvertSolver
from .body_of_revolution import axis_class, order_eigen_solve
from .boundaries import MIRROR_TYPES, BCType, _grading_values
from .eigenfrequency import solve_resonances
from .fem_modes import _LOCAL_NODES, PERIODIC_TEXT, pml_ring

FEM_BOR_DEGREE = TRIANGLE_DEGREES[-1]   # rule of every assembly: the 1 / r, r~ and PML factors are no polynomials
FEM_BOR_SIZE_REMEDY = ('coarsen the triangle mesh (larger maximum element size), use discretization linear, or halve the '
                       '(r, z) domain by a pec / pmc z_lo face on the equatorial plane of a symmetric structure')
# Arnoldi basis per wanted eigenpair (as em3d's ARNOLDI_BASIS_PER_WANTED): with a PML ring about half the wanted
# eigenvalues are clustered PML modes, and scipy's default basis (2 k + 1) restarts ~7-9 applications per basis vector
FEM_BOR_ARNOLDI_BASIS_PER_WANTED = 5
AXIS_NODE_SHRINK = 1e-3   # display: a local node on the axis is sampled this share of the way toward the centroid
_RADIUS_POINTS = 16       # Gauss-Legendre points of the stretched-radius integral over the ring depth
_AXIS_TOLERANCE = 1e-9    # x within this times the longest window side of the axis: on the axis
_SAMPLE_DEGREE = 4        # rule of the element means (state fields)
_CHUNK_ELEMENTS = 16384   # elements per field-evaluation block
_NATURAL_WALLS = frozenset({BCType.PMC})
_PERIODIC_TYPES = frozenset({BCType.PERIODIC, BCType.BLOCH})
_CENTROID = np.array([1.0 / 3.0, 1.0 / 3.0])


# ---- PML metric ----
def stretched_radius(ring, radius, omega):
    """Complex r~ = int_0^r s_r dr' (any shape) at radii [m] from the axis: r inside the window, plus int (s_r - 1) over
    the depth into the r_hi ring of a fem_modes.PMLRing (Gauss-Legendre)."""
    radius = np.asarray(radius, dtype=np.float64)
    result = radius.astype(np.complex128)
    width = float(ring.widths['x_hi'])
    if width <= 0.0:
        return result
    depth = radius - (ring.window[2] - ring.window[0])
    inside = depth > 0.0
    if not inside.any():
        return result
    nodes, weights = np.polynomial.legendre.leggauss(_RADIUS_POINTS)
    span = np.minimum(depth[inside], width)[:, None]
    sigma, kappa, alpha = _grading_values(span * (0.5 * (nodes + 1.0)) / width, ring.sigma_max['x_hi'],
                                          ring.kappa_max, ring.alpha_max, ring.grading_order)
    excess = kappa + sigma / (alpha + 1j * float(omega) * EPS0) - 1.0
    result[inside] += 0.5 * span[:, 0] * (excess @ weights)
    return result


def _metric(layout, points, omega):
    """(r, r~, s_r, s_z) at physical points (..., 2) [m] of the layout's mesh (module header)."""
    s_r, s_z = layout.ring.stretch(points, omega)
    radius = points[..., 0] - layout.axis_x
    return radius, stretched_radius(layout.ring, radius, omega), s_r, s_z


# ---- layout: dofs, walls, axis ----
@dataclass(frozen=True, eq=False)
class FEMBORLayout:
    """Unknowns of one mesh, element degree, wall set and PEC element set: Nedelec (e) and Lagrange (w) dof maps on the
    non-PEC elements, the free entries of the stacked [e; w] vector per axis class (0: m = 0, 1: |m| >= 1), the PML ring,
    the axis position and the window flag of every stacked dof."""
    mesh: object
    order: int
    ring: object                  # fem_modes.PMLRing
    axis_x: float                 # x of the axis = the window's x_lo edge [m]
    axis_tolerance: float         # [m]
    nedelec: object               # fem.dofs.DofMap
    lagrange: object
    free_by_class: tuple          # (int64 free stacked indices of class 0, of class 1)
    in_window: np.ndarray         # bool (n_stacked,)

    @property
    def n_stacked(self) -> int:
        return self.nedelec.n_dofs + self.lagrange.n_dofs

    def free(self, azimuthal_order) -> np.ndarray:
        """Int64 free stacked indices of an integer azimuthal order."""
        return self.free_by_class[axis_class(azimuthal_order)]


def fem_bor_layout(mesh, order, boundary_config, pec=None) -> FEMBORLayout:
    """FEMBORLayout of element degree order (1 / 2) with the walls of boundary_config (r_hi / z faces, CPML settings;
    x_lo = the axis) and the bool per-element PEC mask (None = none); ValueError for periodic / mirror faces, a mesh
    reaching below r = 0 or when every element is PEC."""
    if order not in (1, 2):
        raise ValueError(f'fem_bor_layout: order must be 1 or 2, got {order!r}')
    faces = {face: getattr(boundary_config, face) for face in FACE_BITS}
    for face, kind in faces.items():
        if kind in _PERIODIC_TYPES:
            raise ValueError(f'em_fdfd_bor on triangles: {PERIODIC_TEXT}')
        if kind in MIRROR_TYPES:
            raise ValueError(f'em_fdfd_bor on triangles: {face} is a {kind.value} mirror; use pec / pmc walls (a pec / '
                             'pmc z_lo wall is the mirror plane)')
    if boundary_config.cpml_sides['x_lo']:
        raise ValueError('em_fdfd_bor on triangles: x_lo is the axis r = 0, so it must be no cpml face')
    ring = pml_ring(mesh, boundary_config)
    x0, y0, x1, y1 = (float(value) for value in mesh.window)
    tolerance = _AXIS_TOLERANCE * max(x1 - x0, y1 - y0)
    if float(mesh.points[:, 0].min()) < x0 - tolerance:
        raise ValueError('em_fdfd_bor on triangles: the mesh reaches below r = 0 (the x_lo window edge is the axis)')
    pec = np.zeros(mesh.n_elements, dtype=bool) if pec is None else np.asarray(pec, dtype=bool)
    if pec.shape != (mesh.n_elements,):
        raise ValueError(f'fem_bor_layout: PEC mask shape {pec.shape} != ({mesh.n_elements},)')
    if pec.all():
        raise ValueError('em_fdfd_bor on triangles: every element is PEC')
    nedelec = nedelec_dofs(mesh, order, ~pec)
    lagrange = lagrange_dofs(mesh, order, ~pec, geometry_order=order)
    outer = np.flatnonzero(mesh.edge_elements[:, 1] < 0)
    axis_edges = outer[(np.abs(mesh.points[mesh.edges[outer], 0] - x0) <= tolerance).all(axis=1)]
    natural = [np.asarray(mesh.edge_tags.get(face, ()), dtype=np.int64) for face, kind in faces.items()
               if face != 'x_lo' and kind in _NATURAL_WALLS]
    wall_edges = np.setdiff1d(nedelec.boundary_edges(), np.concatenate([axis_edges] + natural))
    fixed_w = np.union1d(lagrange.edge_dofs(wall_edges), lagrange.edge_dofs(axis_edges))
    if pec.any():   # w = 0 on the closure of the PEC region (vertex contacts too)
        fixed_w = np.union1d(fixed_w, lagrange.node_dofs(np.unique(mesh.triangles[pec])))
    n_tangential = nedelec.n_dofs
    free_by_class = []
    for fixed_axis_e in (np.zeros(0, dtype=np.int64), nedelec.edge_dofs(axis_edges)):   # m = 0: E_z free on the axis
        keep = np.ones(n_tangential + lagrange.n_dofs, dtype=bool)
        keep[nedelec.edge_dofs(wall_edges)] = False
        keep[fixed_axis_e] = False
        keep[n_tangential + fixed_w] = False
        free_by_class.append(np.flatnonzero(keep))
    points = np.concatenate((nedelec.dof_points(), lagrange.dof_points()))
    in_window = ((points[:, 0] <= x1 + tolerance) & (points[:, 1] >= y0 - tolerance)
                 & (points[:, 1] <= y1 + tolerance))
    return FEMBORLayout(mesh=mesh, order=int(order), ring=ring, axis_x=x0, axis_tolerance=tolerance, nedelec=nedelec,
                        lagrange=lagrange, free_by_class=tuple(free_by_class), in_window=in_window)


# ---- assembly and solve ----
@dataclass(frozen=True, eq=False)
class FEMBORSystem:
    """Pencil of one omega and azimuthal order on the free stacked dofs: operator A = K(m) - k0^2 M, mass M, energy W
    (scipy CSR), in_window (n,), equilibration (n,) = (|diag K| + k0^2 |diag M|)^-1/2 (the cuDSS row scale)."""
    omega: float
    azimuthal_order: int
    free: np.ndarray
    operator: object
    mass: object
    energy: object
    in_window: np.ndarray
    equilibration: np.ndarray

    @property
    def n_unknowns(self) -> int:
        return int(self.free.size)


def _check_materials(materials, n_elements):
    """(eps_rr, eps_phiphi, eps_zz, mu, pec) complex128 / bool per element of BORMaterials over the mesh elements."""
    pec = np.asarray(materials.pec, dtype=bool)
    if pec.shape != (n_elements,):
        raise ValueError(f'BORMaterials of shape {pec.shape} on a mesh of {n_elements} elements')
    values = [materials.component(name) for name in ('r', 'phi', 'z')] + [np.asarray(materials.mu, np.complex128)]
    for name, value in zip(('eps_rr', 'eps', 'eps_zz', 'mu'), values):
        if value.shape != (n_elements,):
            raise ValueError(f'BORMaterials {name} shape {value.shape} != ({n_elements},)')
        usable = value[~pec]
        if not (np.all(np.isfinite(usable)) and np.all(usable != 0)):
            raise ValueError(f'BORMaterials {name} must be finite and non-zero on non-PEC elements')
    return (*values, pec)


class FEMBORAssembly:
    """Order-independent parts of the triangle BOR pencil at one omega and material state (module header): stacked
    [e; w] CSR K0, K1, K2, M and the real energy matrix W on every dof of the layout; system(m) restricts K(m) - k0^2 M
    to the order's free set (an m sweep assembles once)."""

    def __init__(self, layout, materials, omega):
        """layout = FEMBORLayout of the materials' PEC elements, materials = body_of_revolution.BORMaterials of
        (n_elements,) host arrays at omega [rad/s]."""
        import scipy.sparse
        started = time.perf_counter()
        omega = float(omega)
        if not (math.isfinite(omega) and omega > 0.0):
            raise ValueError(f'FEMBORAssembly needs a finite angular frequency omega > 0 [rad/s], got {omega!r}')
        eps_rr, eps_phi, eps_zz, mu, _ = _check_materials(materials, layout.mesh.n_elements)
        nu = 1.0 / np.where(mu != 0, mu, 1.0)

        def diagonal(xx, yy):
            tensor = np.zeros(xx.shape + (2, 2), dtype=np.complex128)
            tensor[..., 0, 0], tensor[..., 1, 1] = xx, yy
            return tensor

        def rot_coefficient(points, elements):     # nu r~ / (s_r s_z)
            _, stretched, s_r, s_z = _metric(layout, points, omega)
            return nu[elements][:, None] * stretched / (s_r * s_z)

        def gradient_tensor(points, elements):     # diag(nu s_z / s_r, nu s_r / s_z) / r~
            _, stretched, s_r, s_z = _metric(layout, points, omega)
            element_nu = nu[elements][:, None]
            return diagonal(element_nu * s_z / (s_r * stretched), element_nu * s_r / (s_z * stretched))

        def tangential_mass(points, elements):     # r~ diag(eps_rr s_z / s_r, eps_zz s_r / s_z)
            _, stretched, s_r, s_z = _metric(layout, points, omega)
            return diagonal(eps_rr[elements][:, None] * stretched * s_z / s_r,
                            eps_zz[elements][:, None] * stretched * s_r / s_z)

        def azimuthal_mass(points, elements):      # eps_phiphi s_r s_z / r~
            _, stretched, s_r, s_z = _metric(layout, points, omega)
            return eps_phi[elements][:, None] * s_r * s_z / stretched

        def tangential_energy(points, elements):
            return points[..., 0] - layout.axis_x

        def azimuthal_energy(points, elements):
            return 1.0 / (points[..., 0] - layout.axis_x)

        nedelec, lagrange, degree = layout.nedelec, layout.lagrange, FEM_BOR_DEGREE
        rot = assemble_curl_curl(nedelec, rot_coefficient, degree=degree)
        gradient = assemble_stiffness(lagrange, gradient_tensor, degree=degree)
        coupling = assemble_gradient_coupling(nedelec, lagrange, gradient_tensor, degree=degree)
        order_mass = assemble_vector_mass(nedelec, gradient_tensor, degree=degree)
        bmat = scipy.sparse.bmat
        empty = scipy.sparse.csr_matrix((lagrange.n_dofs, lagrange.n_dofs), dtype=np.complex128)
        self.k0_part = bmat([[rot, None], [None, gradient]], format='csr')
        self.k1_part = bmat([[None, coupling], [coupling.T, None]], format='csr')
        self.k2_part = bmat([[order_mass, None], [None, empty]], format='csr')
        self.mass = bmat([[assemble_vector_mass(nedelec, tangential_mass, degree=degree), None],
                          [None, assemble_mass(lagrange, azimuthal_mass, degree=degree)]], format='csr')
        self.energy = bmat([[assemble_vector_mass(nedelec, tangential_energy, degree=degree), None],
                            [None, assemble_mass(lagrange, azimuthal_energy, degree=degree)]], format='csr')
        self.layout, self.materials, self.omega, self.mu = layout, materials, omega, mu
        self.assembly_seconds = time.perf_counter() - started

    def system(self, azimuthal_order) -> FEMBORSystem:
        """FEMBORSystem of an integer azimuthal order: K0 + m K1 + m^2 K2 - k0^2 M on its axis class's free dofs."""
        axis_class(azimuthal_order)
        order = int(azimuthal_order)
        free = self.layout.free(order)
        stiffness = self.k0_part if order == 0 else self.k0_part + order * self.k1_part + order ** 2 * self.k2_part
        stiffness, mass, energy = (restrict(matrix, free) for matrix in (stiffness, self.mass, self.energy))
        target_squared = (self.omega / C0) ** 2
        equilibration = 1.0 / np.sqrt(np.abs(stiffness.diagonal()) + target_squared * np.abs(mass.diagonal()))
        return FEMBORSystem(omega=self.omega, azimuthal_order=order, free=free,
                            operator=(stiffness - target_squared * mass).tocsr(), mass=mass, energy=energy,
                            in_window=self.layout.in_window[free], equilibration=equilibration)


def solve_fem_bor_resonances(system, n_resonances, eigen_solver=None, warnings=()):
    """eigenfrequency.EigenSolve of up to n_resonances body_of_revolution.BORResonances nearest system.omega
    (squared-Cayley shift-invert by eigen_solver, None = default settings: host SuperLU, or a cuDSS LDL^T of the
    complex symmetric pencil, cudss_mixed with the float64 Rayleigh polish); fields = free stacked vectors."""
    eigen_solver = ArpackShiftInvertSolver() if eigen_solver is None else eigen_solver
    if system.n_unknowns < 4:
        raise ValueError(f'solve_fem_bor_resonances: only {system.n_unknowns} unknowns remain inside the walls')
    # solve_resonances' A = L + k_t^2 M' with L = K, M' = -M (body_of_revolution.solve_bor_resonances alike)
    solve = solve_resonances(system.operator, -system.mass, system.omega, system.energy, system.in_window,
                             n_resonances, spectral_map='squared_cayley', warnings=warnings, eigen_solver=eigen_solver,
                             symmetric=True, equilibration=system.equilibration,
                             basis_per_wanted=FEM_BOR_ARNOLDI_BASIS_PER_WANTED)
    return order_eigen_solve(solve, system.azimuthal_order, eigen_solver)


def resonance_coefficients(layout, azimuthal_order, field) -> np.ndarray:
    """Complex128 (n_stacked,) [e; w] coefficients of a resonance field (free vector of its order), fixed dofs 0."""
    coefficients = np.zeros(layout.n_stacked, dtype=np.complex128)
    coefficients[layout.free(azimuthal_order)] = field
    return coefficients


# ---- fields of a resonance ----
def _fields_at(layout, coefficients, azimuthal_order, omega, mu, elements, reference):
    """(complex (k, q) er, ephi, ez, hr, hphi, hz at per-element reference points (k, q, 2) of non-PEC elements (k,),
    the element geometry there): module header recovery, stretched values in the PML ring."""
    n_tangential = layout.nedelec.n_dofs
    e_t, rot = nedelec_at(layout.nedelec, coefficients[:n_tangential], elements, reference)
    w, grad_w = lagrange_at(layout.lagrange, coefficients[n_tangential:], elements, reference)
    geometry = element_geometry(layout.mesh, elements, reference, layout.nedelec.geometry_order)
    _, stretched, s_r, s_z = _metric(layout, geometry.points, omega)
    g = grad_w + azimuthal_order * e_t
    scale = omega * MU0 * mu[elements][:, None]
    fields = {'er': e_t[..., 0] / s_r, 'ephi': 1j * w / stretched, 'ez': e_t[..., 1] / s_z,
              'hr': g[..., 1] / (s_z * stretched * scale), 'hphi': -1j * rot / (s_r * s_z * scale),
              'hz': -g[..., 0] / (s_r * stretched * scale)}
    return fields, geometry


def electric_field_at(assembly, coefficients, azimuthal_order, elements, reference) -> tuple:
    """(complex cylindrical (E_r, E_phi, E_z) (k, q, 3) [V/m], physical points (k, q, 2) [m]) of stacked coefficients
    at per-element reference points (k, q, 2) of non-PEC elements (k,): the public E part of _fields_at."""
    fields, geometry = _fields_at(assembly.layout, coefficients, azimuthal_order, assembly.omega, assembly.mu,
                                  np.asarray(elements, dtype=np.int64), reference)
    return np.stack([fields['er'], fields['ephi'], fields['ez']], axis=-1), geometry.points


def element_means(assembly, coefficients, azimuthal_order) -> tuple:
    """Float64 (n_elements,) ring-weighted element means of |E|^2 [V^2/m^2] and q_abs = 1/2 omega eps0 sum_c (-Im
    eps_c) |E_c|^2 [W/m^3] of stacked coefficients (0 on PEC elements; the ring in stretched coordinates)."""
    layout, omega = assembly.layout, assembly.omega
    rule = triangle_rule(_SAMPLE_DEGREE)
    eps = {name: assembly.materials.component(name) for name in ('r', 'phi', 'z')}
    intensity, absorbed = np.zeros(layout.mesh.n_elements), np.zeros(layout.mesh.n_elements)
    for start in range(0, len(layout.nedelec.elements), _CHUNK_ELEMENTS):
        elements = layout.nedelec.elements[start:start + _CHUNK_ELEMENTS]
        reference = np.broadcast_to(rule.points, (len(elements),) + rule.points.shape)
        fields, geometry = _fields_at(layout, coefficients, azimuthal_order, omega, assembly.mu, elements, reference)
        weights = rule.weights[None, :] * np.abs(geometry.determinant) * (geometry.points[..., 0] - layout.axis_x)
        total = weights.sum(axis=1)
        squared = {name: np.abs(fields[key]) ** 2 for name, key in (('r', 'er'), ('phi', 'ephi'), ('z', 'ez'))}
        intensity[elements] = (sum(squared.values()) * weights).sum(axis=1) / total
        loss = sum(-eps[name][elements][:, None].imag * squared[name] for name in squared)
        absorbed[elements] = 0.5 * omega * EPS0 * (loss * weights).sum(axis=1) / total
    return intensity, absorbed


def nodal_fields(assembly, coefficients, azimuthal_order, nodal_dofs) -> dict:
    """Complex (nodal_dofs.n_dofs,) er .. hz at the nodes of a full-mesh Lagrange map of the layout's degree: means over
    the non-PEC elements sharing each node (0 at nodes of PEC elements only); a local node on the axis is sampled
    AXIS_NODE_SHRINK of the way toward its element's centroid (E_phi = j w / r and H_r / H_z divide by r)."""
    layout = assembly.layout
    local = _LOCAL_NODES[layout.order]
    totals = {name: np.zeros(nodal_dofs.n_dofs, dtype=np.complex128)
              for name in ('er', 'ephi', 'ez', 'hr', 'hphi', 'hz')}
    counts = np.zeros(nodal_dofs.n_dofs)
    for start in range(0, len(layout.nedelec.elements), _CHUNK_ELEMENTS):
        elements = layout.nedelec.elements[start:start + _CHUNK_ELEMENTS]
        node_x = element_geometry(layout.mesh, elements, local, layout.nedelec.geometry_order).points[..., 0]
        reference = np.array(np.broadcast_to(local, (len(elements),) + local.shape))
        on_axis = np.abs(node_x - layout.axis_x) <= layout.axis_tolerance
        reference[on_axis] += AXIS_NODE_SHRINK * (_CENTROID - reference[on_axis])
        fields, _ = _fields_at(layout, coefficients, azimuthal_order, assembly.omega, assembly.mu, elements, reference)
        dofs = nodal_dofs.element_dofs[elements].ravel()
        counts += np.bincount(dofs, minlength=nodal_dofs.n_dofs)
        for name, values in fields.items():
            scatter_add(totals[name], dofs, values)
    return {name: np.where(counts > 0, total / np.maximum(counts, 1.0), 0.0) for name, total in totals.items()}
