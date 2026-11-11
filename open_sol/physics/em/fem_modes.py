"""Cross-section eigenmodes on triangle meshes (mesh plan 2.6: em_modes on a 'triangles' mesh): Lee-Sun-Cendes mixed
Nedelec (E_t) / Lagrange (E_z) elements of degree 1 or 2 for beta^2, the PML ring as a complex coordinate stretch at
the quadrature points, shift-invert on the host through the 'eigen' role runtime. xy = cross-section, fields ~ exp(j
omega t - j beta z), lossy Im n_eff < 0. Host numpy (scipy inside functions)."""
# Formulation (test fields W e^{+j beta z}, unconjugated: complex symmetric with PML / loss), x = [e_t; e_z] with
# E_t = e_t / beta, E_z = -j e_z:   A x = -beta^2 B x,
#   A = [[S - k0^2 T_eps, 0], [0, 0]],   B = [[T_nu, -G], [-G^T, K_nu - k0^2 M_zz]],
#   S = int nu_zz curl N_i curl N_j, T_eps = int (eps_t N_j) . N_i, T_nu = int (nu_t N_j) . N_i,
#   G = int (nu_t grad phi_j) . N_i, K_nu = int (nu_t grad phi_j) . grad phi_i, M_zz = int eps_zz phi_i phi_j,
#   nu_t = diag(1 / mu_yy, 1 / mu_xx) (the transverse curl turned by 90 deg), nu_zz = 1 / mu_zz. Spurious-free: grad P_p
#   lies in N_p. Uniform eps, mu: beta^2 = k0^2 eps mu - (a discrete cutoff independent of eps) for TE and TM alike.
# PML ring (mesh.pml_faces): s = kappa + sigma / (alpha + j omega eps0) of the tensor CPML grading
# (boundaries._grading_values at depth = distance beyond the window / ring width, sigma_max at pml_dx = width /
# cpml_npml), folded into eps and mu as Lambda = diag(s_y / s_x, s_x / s_y, s_x s_y) at the quadrature points.
# Shift-invert: (A + sigma B)^-1 B x = theta x, beta^2 = sigma - 1 / theta, sigma = (k0 n_guess)^2 (1 + 1e-6). A + sigma
# B is complex symmetric (unconjugated Galerkin, diagonal Lambda): a cudss / mumps / pardiso eigen node factors an
# LDL^T, equilibrated by the parts' |diagonals| (S - k0^2 T_eps + sigma T_nu nearly cancels on its diagonal, as fem3d's
# curl-curl minus mass). A mixed LDL^T (cudss_mixed on the device, mumps_mixed / pardiso_mixed on the host; complex64)
# runs ARPACK on loose solves (eigenfrequency.MIXED_EIGS_REFINEMENTS float64 steps, ARPACK
# to MIXED_ARPACK_TOLERANCE) and then replaces beta^2 by the float64 Rayleigh quotient sigma - x^T (A + sigma B) x /
# x^T B x of each vector: stationary for a complex symmetric pencil (left = right eigenvector), error ~ (vector error)^2.
# Walls: pec / antisymmetric faces, the outer side of a PML ring (pec, the usual default exterior) and the closure of
# PEC elements: tangential E_t and E_z fixed 0; pmc / symmetric faces natural; mirror planes = the window edge.
# Periodic faces need a periodic mesh (not built): refused.
# Recovery (coefficients stored physical: E_t = e_t, E_z = -j beta e_z): H = curl E / (-j omega mu0 mu'):
#   H_x = (dE_z/dy + j beta E_y) / (-j omega mu0 mu'_xx), H_y = (dE_z/dx + j beta E_x) / (j omega mu0 mu'_yy),
#   H_z = (dE_y/dx - dE_x/dy) / (-j omega mu0 mu'_zz), mu' = mu Lambda.
# FEMMode carries the cross_section_modes.mode_overlap attributes (ex, ey, hx, hy at the window quadrature samples,
# x_face_area = y_face_area = their weights), so mode_overlap, mode_group_indices and the ModePhysics table /
# dispersion monitors serve it unchanged.
import math
import time
from dataclasses import dataclass, replace

import numpy as np

from ...core.constants import C0, EPS0, MU0
from ...core.memory_estimate import COMPLEX128_BYTES, FLOAT64_BYTES, host_lu_bytes
from ...fem.assembly import (assemble_curl_curl, assemble_gradient_coupling, assemble_mass, assemble_stiffness,
                             assemble_vector_mass, restrict)
from ...fem.dofs import lagrange_dofs, nedelec_dofs
from ...fem.elements import (LAGRANGE_COUNTS, NEDELEC_COUNTS, element_geometry, lagrange_basis, nedelec_basis,
                             physical_gradients, piola_vectors)
from ...fem.fixed_pattern import scatter_add
from ...fem.indicators import CHUNK_ITEMS, add_edge_terms, edge_sides, lagrange_at, nedelec_at
from ...fem.mesh import FACE_BITS
from ...fem.quadrature import triangle_rule
from ...solvers.arpack_shift_invert import CUDSS_FACTORIZATIONS, ArpackShiftInvertSolver, device_factor_matvec
from ...solvers.cudss_factor import operator_copy_bytes, solve_buffer_bytes, solve_peak_bytes
from ...solvers.host_direct import HOST_DIRECT_FACTORIZATIONS, factor_label
from ...solvers.sparse_direct import SPLU_LIMIT_UNKNOWNS, SPLU_WARN_UNKNOWNS
from .boundaries import MIRROR_TYPES, BCType, _grading_values, cpml_sigma_max
from .cross_section_modes import (_PML_MODE_FRACTION, _POWER_FLOOR, _RANK_DECIMALS, _SHIFT_RELATIVE_OFFSET,
                                  _START_VECTOR_SEED, _THETA_FLOOR, CrossSectionSolve)
from .eigenfrequency import MIXED_ARPACK_TOLERANCE, MIXED_EIGS_REFINEMENTS
from .slab_modes import _transverse_wavenumber, peak_cell

DISCRETIZATION_ORDERS = {'quadratic': 2, 'linear': 1}   # em_modes setting discretization -> element degree
SAMPLE_DEGREE = 4               # quadrature rule of the mode samples (power, overlaps, metrics, element means)
SAMPLE_POINTS = 6               # points of that rule per element
# reference coordinates of the Lagrange local nodes (fem.elements order: corners, then midsides of local edges 0..2)
_LOCAL_NODES = {1: np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
                2: np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])}
# SuperLU L + U of the FEM pencil over the 5-point heuristic of core.memory_estimate.host_lu_bytes (wider stencils):
# uncalibrated first figure (mesh plan 2.10); the summary 'solvers' factor_host_bytes calibrates it
FEM_LU_FILL_FACTOR = 4.0
FEM_ENTRIES_PER_ROW = {1: 12, 2: 30}   # nonzeros per row of the assembled blocks (estimate)
_NATURAL_WALLS = frozenset({BCType.PMC, BCType.SYMMETRIC})
_PERIODIC_TYPES = frozenset({BCType.PERIODIC, BCType.BLOCH})
PERIODIC_TEXT = ('periodic faces need matching nodes on opposite window sides, which triangle meshes do not have '
                 'yet; use a tensor mesh, or pec / pmc / mirror / cpml faces')
_SIZE_REMEDY = ('coarsen the triangle mesh (larger maximum element size), use discretization linear, or cut the '
                'cross-section by a symmetric / antisymmetric mirror face')


# ---- unknown counts and sizes (CPU, also the estimators) ----
def estimated_unknowns(n_elements, n_edges, n_nodes, order) -> int:
    """[e_t; e_z] unknowns of a mesh of these counts: degree 1 one per edge + nodes, degree 2 two per edge + two per
    element + nodes + edges."""
    if order == 1:
        return int(n_edges + n_nodes)
    return int(3 * n_edges + 2 * n_elements + n_nodes)


def fem_lu_bytes(n_unknowns) -> int:
    """Heuristic host bytes of the complex SuperLU factor of the FEM shift-invert operator."""
    return int(FEM_LU_FILL_FACTOR * host_lu_bytes(n_unknowns))


def operator_bytes(n_unknowns, order) -> int:
    """Host bytes of the six assembled blocks, A, B (free part) and the shifted operator (complex128 CSR, estimate)."""
    return int(4 * FEM_ENTRIES_PER_ROW[order] * n_unknowns * (COMPLEX128_BYTES + 4))


def cudss_operator_bytes(n_unknowns, order, mixed) -> int:
    """Device bytes beside an own cuDSS factor for its solves (estimate): the LDL^T CSR copies, rhs buffer / row scale
    and one solve's vectors (cudss_factor helpers); the factor itself is unknown before planning."""
    entries = FEM_ENTRIES_PER_ROW[order] * int(n_unknowns)
    return int(operator_copy_bytes(n_unknowns, entries, mixed=mixed) + solve_buffer_bytes(n_unknowns, mixed=mixed)
               + solve_peak_bytes(n_unknowns, COMPLEX128_BYTES, mixed))


def sample_basis_bytes(n_elements, order) -> int:
    """Host bytes of ModeSamples: points, weights, Nedelec vectors / curls, Lagrange gradients per element."""
    n_tangential, n_axial = NEDELEC_COUNTS[order], LAGRANGE_COUNTS[order]
    per_point = 2 + 1 + 3 * n_tangential + 2 * n_axial
    return int(n_elements * SAMPLE_POINTS * per_point * FLOAT64_BYTES)


def host_solver_label(factorization) -> str:
    """Host solver of a size-rule text: 'SuperLU solver', or for a mumps / pardiso (+ _mixed) factorization or
    back end its library under SuperLU's limits (host_direct factors keep the SuperLU size rule, uncalibrated)."""
    if factorization in HOST_DIRECT_FACTORIZATIONS:
        return f"{factor_label(factorization)} solver (SuperLU's limits, uncalibrated for it)"
    return 'SuperLU solver'


def size_issue_for_unknowns(n_unknowns, factorization='superlu', remedy=_SIZE_REMEDY):
    """(level, text) when the FEM pencil of n_unknowns is near ('warning') or over ('error') the host SuperLU size rule
    (unknowns x FEM_LU_FILL_FACTOR against the 5-point limits of solvers.sparse_direct; MUMPS / PARDISO alike, texts
    naming them), else None; None for a cuDSS eigen factorization (its own device-memory guard after planning). remedy:
    the closing advice (em_fdfd_bor: its own)."""
    if factorization in CUDSS_FACTORIZATIONS:
        return None
    solver = host_solver_label(factorization)
    equivalent = float(n_unknowns) * FEM_LU_FILL_FACTOR
    text = (f'{int(n_unknowns):,} FEM unknowns (about {FEM_LU_FILL_FACTOR:g}x the SuperLU fill of a grid unknown, '
            f'so ~{int(equivalent):,} grid unknowns)')
    if equivalent > SPLU_LIMIT_UNKNOWNS:
        return 'error', (f'{text} exceed the {SPLU_LIMIT_UNKNOWNS:,} limit of the host {solver}: its '
                         f'factors would fill the host memory; {remedy}')
    if equivalent > SPLU_WARN_UNKNOWNS:
        return 'warning', (f'{text} exceed ~{SPLU_WARN_UNKNOWNS:,}, the practical size of the host {solver}; '
                           f'{remedy}')
    return None


# ---- PML ring ----
@dataclass(frozen=True, eq=False)
class PMLRing:
    """Complex coordinate stretch of the mesh's PML layer ring: thickness and peak conductivity per face (0 without a
    ring) with the tensor CPML grading; stretch() = (s_x, s_y) at points."""
    window: tuple                 # (x0, y0, x1, y1) [m]
    widths: dict                  # FACE_BITS face -> ring thickness [m]
    sigma_max: dict               # face -> peak conductivity [S/m]
    kappa_max: float
    alpha_max: float              # [S/m]
    grading_order: int

    def stretch(self, points, omega) -> tuple:
        """Complex128 (s_x, s_y) of shape points.shape[:-1] at points [m] (1 inside the window)."""
        points = np.asarray(points, dtype=np.float64)
        x0, y0, x1, y1 = self.window
        return (self._axis(points[..., 0], x0, x1, 'x_lo', 'x_hi', omega),
                self._axis(points[..., 1], y0, y1, 'y_lo', 'y_hi', omega))

    def _axis(self, coordinate, low, high, lo_face, hi_face, omega):
        stretch = np.ones(coordinate.shape, dtype=np.complex128)
        for face, depth in ((lo_face, low - coordinate), (hi_face, coordinate - high)):
            width = self.widths[face]
            if width <= 0.0:
                continue
            inside = depth > 0.0
            if not inside.any():
                continue
            sigma, kappa, alpha = _grading_values(np.clip(depth[inside] / width, 0.0, 1.0), self.sigma_max[face],
                                                  self.kappa_max, self.alpha_max, self.grading_order)
            stretch[inside] = kappa + sigma / (alpha + 1j * float(omega) * EPS0)
        return stretch


def pml_ring(mesh, boundary_config) -> PMLRing:
    """PMLRing of the mesh's layer ring with the CPML settings of boundary_config; ValueError when the ring faces differ
    from its cpml faces (the mesh was built for another boundary configuration)."""
    x0, y0, x1, y1 = mesh.window
    window_edges = {'x_lo': x0, 'x_hi': x1, 'y_lo': y0, 'y_hi': y1}
    cpml = boundary_config.cpml_sides
    widths, sigma_max = {}, {}
    for face, bit in FACE_BITS.items():
        ring = (mesh.pml_faces & bit) != 0
        width = 0.0
        if ring.any():
            coordinates = mesh.points[mesh.triangles[ring]][..., 0 if face[0] == 'x' else 1]
            width = float(window_edges[face] - coordinates.min() if face.endswith('lo')
                          else coordinates.max() - window_edges[face])
        if (width > 0.0) != bool(cpml[face]):
            raise ValueError(f"triangle mesh PML ring on {face}: {'present' if width > 0.0 else 'missing'}, but the "
                             f"face is {getattr(boundary_config, face).value}; mesh and boundary settings disagree")
        widths[face] = width
        sigma_max[face] = (cpml_sigma_max(boundary_config, width / int(boundary_config.cpml_npml)) if width > 0.0
                           else 0.0)
    return PMLRing(window=tuple(float(value) for value in mesh.window), widths=widths, sigma_max=sigma_max,
                   kappa_max=float(boundary_config.cpml_kappa_max), alpha_max=float(boundary_config.cpml_alpha_max),
                   grading_order=int(boundary_config.cpml_order))


# ---- layout: dofs, walls, samples ----
@dataclass(frozen=True, eq=False)
class ModeSamples:
    """Points of the layout's elements (dof map order: degree-SAMPLE_DEGREE quadrature, or the Lagrange local nodes)
    with the basis there."""
    points: np.ndarray              # (m, q, 2) [m]
    weights: np.ndarray             # (m, q) w det J [m^2] (quadrature samples; 0 at nodes)
    in_window: np.ndarray           # bool (m,): element inside the window (not in the PML ring)
    nedelec_vectors: np.ndarray     # (m, q, n_t, 2) physical (covariant Piola)
    nedelec_curls: np.ndarray       # (m, q, n_t)
    lagrange_values: np.ndarray     # (q, n_z)
    lagrange_gradients: np.ndarray  # (m, q, n_z, 2)


def _mode_samples(nedelec, lagrange, nodes=False) -> ModeSamples:
    """ModeSamples at the quadrature points, or (nodes) at the Lagrange local nodes of every element."""
    mesh = nedelec.mesh
    rule = triangle_rule(SAMPLE_DEGREE)
    reference = _LOCAL_NODES[lagrange.order] if nodes else rule.points
    geometry = element_geometry(mesh, nedelec.elements, reference, nedelec.geometry_order)
    vectors, curls = nedelec_basis(nedelec.order, reference, mesh.edge_signs[nedelec.elements][:, None, :])
    values, gradients = lagrange_basis(lagrange.order, reference)
    weights = np.zeros(geometry.determinant.shape) if nodes else rule.weights[None, :] * geometry.determinant
    return ModeSamples(points=geometry.points, weights=weights, in_window=mesh.pml_faces[nedelec.elements] == 0,
                       nedelec_vectors=piola_vectors(vectors, geometry.inverse),
                       nedelec_curls=curls / geometry.determinant[..., None],
                       lagrange_values=values, lagrange_gradients=physical_gradients(gradients, geometry.inverse))


@dataclass(frozen=True, eq=False)
class FEMModeLayout:
    """Unknowns of one mesh, element degree, wall set and PEC element set: Nedelec (e_t) and Lagrange (e_z) dof maps on
    the non-PEC elements, the free entries of the stacked [e_t; e_z] vector, the PML ring and the mode samples."""
    mesh: object
    order: int
    ring: PMLRing
    nedelec: object                 # fem.dofs.DofMap
    lagrange: object
    free: np.ndarray                # int64 indices into the stacked vector
    samples: ModeSamples
    mirror_factor: float            # 2 per mirror face: the mesh holds 1 / mirror_factor of the mode

    @property
    def n_stacked(self) -> int:
        return self.nedelec.n_dofs + self.lagrange.n_dofs

    @property
    def n_unknowns(self) -> int:
        return len(self.free)


def fem_mode_layout(mesh, order, boundary_config, pec=None) -> FEMModeLayout:
    """FEMModeLayout of element degree order (1 / 2) with the walls of boundary_config (faces, CPML settings) and the
    bool per-element PEC mask (None = none); ValueError for periodic faces or when every element is PEC."""
    if order not in (1, 2):
        raise ValueError(f'fem_mode_layout: order must be 1 or 2, got {order!r}')
    faces = {face: getattr(boundary_config, face) for face in FACE_BITS}
    if any(kind in _PERIODIC_TYPES for kind in faces.values()):
        raise ValueError(f'em_modes on triangles: {PERIODIC_TEXT}')
    ring = pml_ring(mesh, boundary_config)
    pec = np.zeros(mesh.n_elements, dtype=bool) if pec is None else np.asarray(pec, dtype=bool)
    if pec.shape != (mesh.n_elements,):
        raise ValueError(f'fem_mode_layout: PEC mask shape {pec.shape} != ({mesh.n_elements},)')
    if pec.all():
        raise ValueError('em_modes on triangles: every element is PEC')
    nedelec = nedelec_dofs(mesh, order, ~pec)
    lagrange = lagrange_dofs(mesh, order, ~pec, geometry_order=order)
    natural = [np.asarray(mesh.edge_tags.get(face, ()), dtype=np.int64) for face, kind in faces.items()
               if kind in _NATURAL_WALLS]
    wall_edges = np.setdiff1d(nedelec.boundary_edges(),
                              np.concatenate(natural) if natural else np.zeros(0, dtype=np.int64))
    fixed_axial = lagrange.edge_dofs(wall_edges)
    if pec.any():   # E_z = 0 on the closure of the PEC region (vertex contacts too)
        fixed_axial = np.union1d(fixed_axial, lagrange.node_dofs(np.unique(mesh.triangles[pec])))
    keep = np.ones(nedelec.n_dofs + lagrange.n_dofs, dtype=bool)
    keep[nedelec.edge_dofs(wall_edges)] = False
    keep[nedelec.n_dofs + fixed_axial] = False
    mirrors = sum(kind in MIRROR_TYPES for kind in faces.values())
    return FEMModeLayout(mesh=mesh, order=int(order), ring=ring, nedelec=nedelec, lagrange=lagrange,
                         free=np.flatnonzero(keep), samples=_mode_samples(nedelec, lagrange),
                         mirror_factor=2.0 ** mirrors)


# ---- materials and modes ----
@dataclass(frozen=True, eq=False)
class FEMModeMaterials:
    """Host per-element values at one omega: complex128 (n_elements,) relative eps_xx, eps_yy, eps_zz and mu (sigma and
    poles included, Im < 0 lossy) and the bool PEC elements."""
    eps_xx: np.ndarray
    eps_yy: np.ndarray
    eps_zz: np.ndarray
    mu: np.ndarray
    pec: np.ndarray

    def __post_init__(self):
        shape = np.shape(self.pec)
        if np.asarray(self.pec).dtype != np.bool_ or len(shape) != 1:
            raise TypeError(f'FEMModeMaterials pec must be a 1D bool array, got {np.asarray(self.pec).dtype} {shape}')
        for name in ('eps_xx', 'eps_yy', 'eps_zz', 'mu'):
            values = np.asarray(getattr(self, name))
            if values.shape != shape:
                raise ValueError(f'FEMModeMaterials {name} shape {values.shape} != PEC shape {shape}')
            usable = values[~np.asarray(self.pec)]
            if not (np.all(np.isfinite(usable)) and np.all(usable != 0)):
                raise ValueError(f'FEMModeMaterials {name} must be finite and non-zero on non-PEC elements')


@dataclass(frozen=True, eq=False)
class FEMMode:
    """One mode carrying |power| = 1 W (the whole mode when mirrors cut the mesh): physical dof coefficients of the
    layout it was solved on ([E_t; E_z], fixed entries 0) and the cross_section_modes.mode_overlap attributes."""
    beta: complex                   # [1/m]: Re > 0 when Re beta^2 > 0, else Im < 0
    n_eff: complex
    omega: float
    coefficients: np.ndarray        # complex128 (layout.n_stacked,)
    ex: np.ndarray                  # complex128 (n_window_samples,) [V/m]
    ey: np.ndarray
    hx: np.ndarray                  # [A/m]
    hy: np.ndarray
    x_face_area: np.ndarray         # sample weights [m^2] (mode_overlap reads both)
    y_face_area: np.ndarray
    power: float                    # 1/2 Re int (E x H*).z dA of the whole mode: +-1, ~0 when evanescent
    te_fraction: float              # int |E_x|^2 / int (|E_x|^2 + |E_y|^2) over the window
    effective_area: float           # [m^2] (int |E|^2)^2 / int |E|^4, whole mode
    pml_fraction: float             # share of int |H_t|^2 dA in the PML ring
    mirror_factor: float
    n_group: complex | None = None


def sample_permeability(layout, mu, omega, samples=None) -> tuple:
    """(mu'_xx, mu'_yy, mu'_zz) complex (m, q) at ModeSamples (None = the layout's): per-element mu times the PML
    Lambda."""
    s_x, s_y = layout.ring.stretch((layout.samples if samples is None else samples).points, omega)
    element_mu = np.asarray(mu, dtype=np.complex128)[layout.nedelec.elements][:, None]
    return element_mu * s_y / s_x, element_mu * s_x / s_y, element_mu * s_x * s_y


def _faraday(e_t, curl, e_z, grad_z, beta, omega, permeability) -> dict:
    """Complex ex, ey, ez, hx, hy, hz of physical E_t, curl E_t, E_z and grad E_z (module header)."""
    mu_xx, mu_yy, mu_zz = permeability
    j_omega_mu0 = 1j * float(omega) * MU0
    j_beta = 1j * complex(beta)
    return {'ex': e_t[..., 0], 'ey': e_t[..., 1], 'ez': e_z,
            'hx': (grad_z[..., 1] + j_beta * e_t[..., 1]) / (-j_omega_mu0 * mu_xx),
            'hy': (grad_z[..., 0] + j_beta * e_t[..., 0]) / (j_omega_mu0 * mu_yy),
            'hz': curl / (-j_omega_mu0 * mu_zz)}


def _sample_fields(layout, coefficients, beta, omega, permeability, samples=None) -> dict:
    """Complex (m, q) ex .. hz at ModeSamples (None = the layout's) of physical coefficients [E_t; E_z]."""
    samples = layout.samples if samples is None else samples
    n_tangential = layout.nedelec.n_dofs
    tangential = coefficients[:n_tangential][layout.nedelec.element_dofs]
    axial = coefficients[n_tangential:][layout.lagrange.element_dofs]
    return _faraday(np.einsum('mqnc,mn->mqc', samples.nedelec_vectors, tangential),
                    np.einsum('mqn,mn->mq', samples.nedelec_curls, tangential),
                    np.einsum('qn,mn->mq', samples.lagrange_values, axial),
                    np.einsum('mqnc,mn->mqc', samples.lagrange_gradients, axial), beta, omega, permeability)


def _build_mode(layout, raw, beta, omega, k0, permeability) -> FEMMode:
    """Normalised FEMMode of a raw stacked eigenvector [e_t; e_z] at beta: physical coefficients, 1 W (bilinear for
    evanescent modes), dominant transverse E real > 0 at its peak sample, metrics."""
    coefficients = np.array(raw, dtype=np.complex128)
    coefficients[layout.nedelec.n_dofs:] *= -1j * complex(beta)
    fields = _sample_fields(layout, coefficients, beta, omega, permeability)
    samples = layout.samples
    ring_energy = (np.abs(fields['hx']) ** 2 + np.abs(fields['hy']) ** 2) * samples.weights
    total_energy = float(ring_energy.sum())
    pml_fraction = float(ring_energy[~samples.in_window].sum()) / total_energy if total_energy > 0.0 else 1.0
    weights = samples.weights[samples.in_window].ravel()
    window = {name: values[samples.in_window].ravel() for name, values in fields.items()}
    mirror = layout.mirror_factor

    def cross(conjugate):
        hx = np.conj(window['hx']) if conjugate else window['hx']
        hy = np.conj(window['hy']) if conjugate else window['hy']
        return 0.5 * mirror * complex(np.sum((window['ex'] * hy - window['ey'] * hx) * weights))

    power = cross(True).real
    magnitude = 0.5 * mirror * float(np.sum((np.abs(window['ex'] * window['hy'])
                                             + np.abs(window['ey'] * window['hx'])) * weights))
    norm = abs(power) if abs(power) >= _POWER_FLOOR * magnitude else abs(cross(False))
    if norm > 0.0:
        scale = 1.0 / math.sqrt(norm)
    else:
        peak_e = max(float(np.max(np.abs(window[name]), initial=0.0)) for name in ('ex', 'ey', 'ez'))
        scale = 1.0 / peak_e if peak_e > 0.0 else 1.0
    energy_x = float(np.sum(np.abs(window['ex']) ** 2 * weights))
    energy_y = float(np.sum(np.abs(window['ey']) ** 2 * weights))
    reference = window['ex'] if energy_x >= energy_y else window['ey']
    phase = 1.0
    if np.any(reference != 0):
        peak_value = reference[peak_cell(reference)]
        phase = np.conj(peak_value) / abs(peak_value)
    factor = scale * phase
    coefficients *= factor
    for name in window:
        window[name] = window[name] * factor
    intensity = np.abs(window['ex']) ** 2 + np.abs(window['ey']) ** 2 + np.abs(window['ez']) ** 2
    quartic = float(np.sum(intensity ** 2 * weights))
    transverse = energy_x + energy_y
    return FEMMode(beta=complex(beta), n_eff=complex(beta / k0), omega=float(omega), coefficients=coefficients,
                   ex=window['ex'], ey=window['ey'], hx=window['hx'], hy=window['hy'], x_face_area=weights,
                   y_face_area=weights, power=cross(True).real,
                   te_fraction=energy_x / transverse if transverse > 0.0 else 0.0,
                   effective_area=(mirror * float(np.sum(intensity * weights)) ** 2 / quartic if quartic > 0.0
                                   else float('nan')),
                   pml_fraction=pml_fraction, mirror_factor=mirror)


# ---- assembly and solve ----
@dataclass(frozen=True, eq=False)
class FEMModeProblem:
    """Assembled pencil of one omega on the free unknowns: shifted operator A + shift B (CSC, complex symmetric), B, the
    shift and index guess, the sample permeability of the recovery, the cuDSS equilibration row_scale."""
    omega: float
    k0: float
    operator: object
    b_free: object
    shift: complex
    guess: complex
    permeability: tuple
    assembly_seconds: float
    row_scale: np.ndarray   # (sum of the parts' |diagonals|)^-1/2: diag(A + shift B) may nearly cancel


def _default_index(materials):
    """max Re sqrt(eps mu) over the non-PEC elements and the three permittivity components."""
    usable = ~np.asarray(materials.pec)
    mu = np.asarray(materials.mu, dtype=np.complex128)[usable]
    return max(float(np.max(np.real(np.sqrt(np.asarray(eps, dtype=np.complex128)[usable] * mu))))
               for eps in (materials.eps_xx, materials.eps_yy, materials.eps_zz))


def _coefficients(layout, materials, omega):
    """Quadrature-point coefficient callables (fem.assembly) of the pencil: nu_zz, eps_t, nu_t (tensors), eps_zz,
    each material per element times the PML Lambda."""
    ring = layout.ring
    eps_xx, eps_yy, eps_zz, mu = (np.asarray(values, dtype=np.complex128) for values in
                                  (materials.eps_xx, materials.eps_yy, materials.eps_zz, materials.mu))

    def diagonal(xx, yy):
        tensor = np.zeros(xx.shape + (2, 2), dtype=np.complex128)
        tensor[..., 0, 0], tensor[..., 1, 1] = xx, yy
        return tensor

    def inverse_mu_zz(points, elements):
        s_x, s_y = ring.stretch(points, omega)
        return 1.0 / (mu[elements][:, None] * s_x * s_y)

    def eps_transverse(points, elements):
        s_x, s_y = ring.stretch(points, omega)
        return diagonal(eps_xx[elements][:, None] * s_y / s_x, eps_yy[elements][:, None] * s_x / s_y)

    def nu_transverse(points, elements):   # diag(1 / mu'_yy, 1 / mu'_xx)
        s_x, s_y = ring.stretch(points, omega)
        element_mu = mu[elements][:, None]
        return diagonal(s_y / (element_mu * s_x), s_x / (element_mu * s_y))

    def eps_axial(points, elements):
        s_x, s_y = ring.stretch(points, omega)
        return eps_zz[elements][:, None] * s_x * s_y

    return inverse_mu_zz, eps_transverse, nu_transverse, eps_axial


def prepare_fem_modes(layout, materials, omega, n_eff_guess=None) -> FEMModeProblem:
    """FEMModeProblem of the layout at omega [rad/s] (assembly, free part, shifted operator) before any factor."""
    import scipy.sparse

    started = time.perf_counter()
    omega = float(omega)
    if not (math.isfinite(omega) and omega > 0.0):
        raise ValueError(f'solve_fem_modes needs a finite angular frequency omega > 0 [rad/s], got {omega!r}')
    if np.shape(materials.pec) != (layout.mesh.n_elements,):
        raise ValueError(f'materials of {np.shape(materials.pec)} elements on a mesh of {layout.mesh.n_elements}')
    k0 = omega / C0
    nedelec, lagrange = layout.nedelec, layout.lagrange
    guess = _default_index(materials) if n_eff_guess is None else complex(n_eff_guess)
    if not (np.isfinite(guess) and guess != 0):
        raise ValueError(f'n_eff_guess must be finite and non-zero, got {n_eff_guess!r}')
    shift = (k0 * guess) ** 2 * (1.0 + _SHIFT_RELATIVE_OFFSET)
    inverse_mu_zz, eps_transverse, nu_transverse, eps_axial = _coefficients(layout, materials, omega)
    curl_curl = assemble_curl_curl(nedelec, inverse_mu_zz)
    eps_mass = assemble_vector_mass(nedelec, eps_transverse)
    nu_mass = assemble_vector_mass(nedelec, nu_transverse)
    gradient = assemble_gradient_coupling(nedelec, lagrange, nu_transverse)
    axial_stiffness = assemble_stiffness(lagrange, nu_transverse)
    axial_mass = assemble_mass(lagrange, eps_axial)
    zero = scipy.sparse.csr_matrix((lagrange.n_dofs, lagrange.n_dofs), dtype=np.complex128)
    a_matrix = scipy.sparse.bmat([[curl_curl - k0 ** 2 * eps_mass, None], [None, zero]], format='csr')
    b_matrix = scipy.sparse.bmat([[nu_mass, -gradient], [-gradient.T, axial_stiffness - k0 ** 2 * axial_mass]],
                                 format='csr')
    a_free, b_free = restrict(a_matrix, layout.free), restrict(b_matrix, layout.free)
    magnitude = abs(shift)
    scale_diagonal = np.concatenate((
        np.abs(curl_curl.diagonal()) + k0 ** 2 * np.abs(eps_mass.diagonal()) + magnitude * np.abs(nu_mass.diagonal()),
        magnitude * (np.abs(axial_stiffness.diagonal()) + k0 ** 2 * np.abs(axial_mass.diagonal()))))[layout.free]
    row_scale = 1.0 / np.sqrt(np.where(scale_diagonal > 0.0, scale_diagonal, 1.0))
    return FEMModeProblem(omega=omega, k0=k0, operator=(a_free + shift * b_free).tocsc(), b_free=b_free, shift=shift,
                          guess=guess, permeability=sample_permeability(layout, materials.mu, omega),
                          assembly_seconds=time.perf_counter() - started, row_scale=row_scale)


def _start_vector(start_vector, layout):
    """ARPACK v0 on the free unknowns: None, a flat array of their count, or the summed raw [e_t; e_z] of a FEMMode /
    sequence of them (None when that sum vanishes)."""
    if start_vector is None:
        return None
    modes = [start_vector] if isinstance(start_vector, FEMMode) else start_vector
    if isinstance(modes, (list, tuple)) and all(isinstance(mode, FEMMode) for mode in modes):
        total = np.zeros(layout.n_unknowns, dtype=np.complex128)
        n_tangential = layout.nedelec.n_dofs
        for mode in modes:
            if mode.coefficients.shape != (layout.n_stacked,):
                raise ValueError(f'start_vector mode has {mode.coefficients.size} coefficients, the layout '
                                 f'{layout.n_stacked}')
            raw = np.array(mode.coefficients, dtype=np.complex128)
            raw[n_tangential:] /= -1j * mode.beta
            total += raw[layout.free]
        return total if np.any(total != 0) else None
    vector = np.asarray(start_vector, dtype=np.complex128).ravel()
    if vector.size != layout.n_unknowns:
        raise ValueError(f'start_vector needs {layout.n_unknowns} entries, got {vector.size}')
    return vector


def solve_fem_modes(layout, materials, omega, n_modes, n_eff_guess=None, start_vector=None, eigen_solver=None,
                    drop_pml_modes=True) -> CrossSectionSolve:
    """CrossSectionSolve of up to n_modes FEMModes found by shift-invert near (k0 n_eff_guess)^2 (default max Re sqrt(eps
    mu)), ranked by descending Re beta^2 (ties by Im), modes with > 1/2 of int |H_t|^2 in the PML ring dropped;
    eigen_solver: solvers.arpack_shift_invert runtime (None = default settings; cudss_mixed / mumps_mixed /
    pardiso_mixed: Rayleigh-polished)."""
    if isinstance(n_modes, bool) or not isinstance(n_modes, (int, np.integer)) or n_modes < 1:
        raise ValueError(f'n_modes must be an integer >= 1, got {n_modes!r}')
    problem = prepare_fem_modes(layout, materials, omega, n_eff_guess)
    started = time.perf_counter()
    eigen_solver = ArpackShiftInvertSolver() if eigen_solver is None else eigen_solver
    n_unknowns = layout.n_unknowns
    if n_unknowns < 4:
        raise ValueError(f'solve_fem_modes: only {n_unknowns} unknowns remain inside the walls')
    warnings = []
    # complex symmetric pencil: a cudss / mumps / pardiso node factors an LDL^T (SuperLU ignores symmetric / row_scale)
    factorization = eigen_solver.factorize_host(problem.operator, symmetric=True, row_scale=problem.row_scale)
    factorized = time.perf_counter()
    # module header: loose solves, then float64 Rayleigh quotients, for any mixed LDL^T (cuDSS on the device, MUMPS /
    # PARDISO on the host: solve_host(refinement_steps=)); SuperLU has no factor_precision
    polish = getattr(factorization, 'factor_precision', 'double') == 'mixed' and getattr(factorization, 'symmetric',
                                                                                          False)
    steps = MIXED_EIGS_REFINEMENTS if polish else None
    n_wanted = min(2 * int(n_modes) + 10, n_unknowns - 2)
    initial = _start_vector(start_vector, layout)
    if initial is None:   # deterministic (reproducible solves and run times), not ARPACK's process-wide random state
        random = np.random.default_rng(_START_VECTOR_SEED)
        initial = random.standard_normal(n_unknowns) + 1j * random.standard_normal(n_unknowns)
    if steps is None:
        def apply(vector):
            return factorization.solve_host(problem.b_free @ vector)
    else:
        def apply(vector):
            return factorization.solve_host(problem.b_free @ vector, refinement_steps=steps)
    theta, vectors, applications, converged = eigen_solver.eigs(   # cuDSS / a gpu runtime: device Krylov-Schur
        apply, n_unknowns, n_wanted, initial, tolerance_floor=MIXED_ARPACK_TOLERANCE if polish else 0.0,
        device_matvec=device_factor_matvec(factorization, problem.b_free, refinement_steps=steps))
    if not converged:
        warnings.append(f'ARPACK converged {np.size(theta)} of {n_wanted} eigenvalues near n_eff = {problem.guess:.6g}; '
                        f'modes may be missing (set the effective index guess near the wanted mode)')
    eigen_done = time.perf_counter()
    theta = np.asarray(theta, dtype=np.complex128)
    usable = np.isfinite(theta) & (np.abs(theta) > _THETA_FLOOR * float(np.max(np.abs(theta), initial=0.0)))
    beta_squared = problem.shift - 1.0 / theta[usable]
    vectors = np.asarray(vectors)[:, usable]
    rayleigh_change = None
    if polish:
        beta_squared, rayleigh_change = _rayleigh_beta_squared(problem, vectors, beta_squared)
    beta = _transverse_wavenumber(beta_squared)
    candidates, n_pml_dropped = [], 0
    for column in range(beta.size):
        if beta[column] == 0:
            continue
        raw = np.zeros(layout.n_stacked, dtype=np.complex128)
        raw[layout.free] = vectors[:, column]
        mode = _build_mode(layout, raw, beta[column], problem.omega, problem.k0, problem.permeability)
        if drop_pml_modes and mode.pml_fraction > _PML_MODE_FRACTION:
            n_pml_dropped += 1
            continue
        candidates.append((beta_squared[column], mode))
    values = np.array([candidate[0] for candidate in candidates], dtype=np.complex128)
    order = np.lexsort((-values.imag, -np.round(values.real / problem.k0 ** 2, _RANK_DECIMALS)))[:int(n_modes)]
    finished = time.perf_counter()
    info = {'n_unknowns': n_unknowns, 'n_elements': int(layout.mesh.n_elements), 'order': layout.order,
            'factor_host_bytes': factorization.host_bytes or 0, 'assembly_seconds': problem.assembly_seconds,
            'factor_seconds': factorized - started, 'eigen_seconds': eigen_done - factorized,
            'seconds': problem.assembly_seconds + finished - started, 'operator_applications': applications,
            'eigenvalues_requested': n_wanted, 'eigenvalues_converged': int(theta.size),
            'pml_modes_dropped': n_pml_dropped, 'shift_index': problem.guess,
            'factor_kind': getattr(factorization, 'description', 'SuperLU'),
            'factor_device_bytes': int(getattr(factorization, 'device_bytes', 0)),
            'arnoldi': eigen_solver.last_arnoldi, 'warnings': warnings}
    if getattr(factorization, 'single_residual', None) is not None:   # *_mixed: the complex64 factor alone
        info['single_residual'] = float(factorization.single_residual)
    if rayleigh_change is not None:
        info['rayleigh_change'] = rayleigh_change
    return CrossSectionSolve(modes=[candidates[index][1] for index in order], info=info)


def fem_group_indices(layout, materials_at, omega, modes, relative_step) -> tuple:
    """(copies of modes with n_group = C0 d beta / d omega, warnings) by first-order perturbation of the complex
    symmetric pencil (A + beta^2 B) x = 0 (the unconjugated x is its own left vector): d beta^2 / d omega = -x^T (A' +
    beta^2 B') x / x^T B x, A' and B' central differences of the pencil assembled (no factor, no eigen solve) at omega
    (1 +- relative_step) on materials_at(omega') (material dispersion and the PML stretch included), B their mean;
    NaN where x^T B x vanishes."""
    omega, step = float(omega), float(relative_step)

    def pencil(at):   # (A, B) of the free unknowns at omega `at`
        problem = prepare_fem_modes(layout, materials_at(at), at)
        return problem.operator - problem.shift * problem.b_free, problem.b_free

    a_plus, b_plus = pencil(omega * (1.0 + step))
    a_minus, b_minus = pencil(omega * (1.0 - step))
    d_a, d_b = (a_plus - a_minus) / (2.0 * step * omega), (b_plus - b_minus) / (2.0 * step * omega)
    b_mean = 0.5 * (b_plus + b_minus)
    n_tangential = layout.nedelec.n_dofs
    with_group, warnings = [], []
    for number, mode in enumerate(modes, start=1):
        beta = complex(mode.beta)
        raw = np.array(mode.coefficients, dtype=np.complex128)
        raw[n_tangential:] /= -1j * beta   # the eigenvector's own E_z scaling (_build_mode)
        vector = raw[layout.free]
        denominator = complex(vector @ (b_mean @ vector))
        if denominator == 0 or beta == 0:
            warnings.append(f'mode {number} (n_eff = {mode.n_eff:.6g}): x^T B x vanishes, n_group NaN')
            n_group = complex(float('nan'), float('nan'))
        else:
            derivative = -complex(vector @ (d_a @ vector) + beta ** 2 * (vector @ (d_b @ vector))) / denominator
            n_group = C0 * derivative / (2.0 * beta)
        with_group.append(replace(mode, n_group=complex(n_group)))
    return with_group, warnings


def _rayleigh_beta_squared(problem, vectors, beta_squared) -> tuple:
    """(beta^2 = shift - x^T (A + shift B) x / x^T B x per vector column (ARPACK's value where x^T B x is 0), largest
    change / k0^2): the float64 polish of loose cudss_mixed solves (module header)."""
    numerators = np.sum(vectors * (problem.operator @ vectors), axis=0)
    denominators = np.sum(vectors * (problem.b_free @ vectors), axis=0)
    polished = np.array(beta_squared, dtype=np.complex128)
    usable = denominators != 0
    polished[usable] = problem.shift - numerators[usable] / denominators[usable]
    change = float(np.max(np.abs(polished - beta_squared), initial=0.0)) / problem.k0 ** 2
    return polished, change


# ---- fields of a solved mode ----
def element_means(layout, mode, mu) -> tuple:
    """Float64 (n_elements,) element means of |E|^2 [V^2/m^2] and S_z = 1/2 Re(E_x H_y* - E_y H_x*) [W/m^2] (0 on PEC
    elements; PML ring elements in stretched coordinates)."""
    fields = _sample_fields(layout, mode.coefficients, mode.beta, mode.omega,
                            sample_permeability(layout, mu, mode.omega))
    weights = layout.samples.weights
    area = weights.sum(axis=1)
    intensity = sum(np.abs(fields[name]) ** 2 for name in ('ex', 'ey', 'ez'))
    flow = 0.5 * np.real(fields['ex'] * np.conj(fields['hy']) - fields['ey'] * np.conj(fields['hx']))
    results = []
    for values in (intensity, flow):
        full = np.zeros(layout.mesh.n_elements)
        full[layout.nedelec.elements] = (values * weights).sum(axis=1) / area
        results.append(full)
    return tuple(results)


def mode_error_indicator(layout, mode, materials) -> np.ndarray:
    """eta^2 per mesh element (float64 (n_elements,)) of a mode (mesh plan A3): Kelly jumps of tangential H (H . t and
    H_z) and normal D across interior window edges between non-PEC elements, 1/2 h_E int_E (mu0 |mu| |[H_tan]|^2 +
    |[D_n]|^2 / (eps0 |eps|)) ds to each side (that side's energy weights); outer, PEC and PML edges carry none."""
    mesh = layout.mesh
    inside = np.zeros(mesh.n_elements, dtype=bool)
    inside[layout.nedelec.elements] = True
    inside &= mesh.pml_faces == 0
    eps_xx, eps_yy, mu = (np.asarray(values, dtype=np.complex128) for values in
                          (materials.eps_xx, materials.eps_yy, materials.mu))
    eps_scale = 0.5 * (np.abs(eps_xx) + np.abs(eps_yy))
    n_tangential = layout.nedelec.n_dofs
    tangential, axial = mode.coefficients[:n_tangential], mode.coefficients[n_tangential:]
    eta_squared = np.zeros(mesh.n_elements)
    candidates = np.unique(mesh.element_edges[inside])
    for start in range(0, candidates.size, CHUNK_ITEMS):
        sides = edge_sides(mesh, candidates[start:start + CHUNK_ITEMS], inside, layout.order)
        sides = sides.select(sides.two_sided)
        if not len(sides.edges):
            continue
        normal = sides.normals
        tangent = np.stack([-normal[..., 1], normal[..., 0]], axis=-1)
        values = []
        for side in (0, 1):
            elements = sides.elements[:, side]
            reference = sides.reference_points(side)
            e_t, curl = nedelec_at(layout.nedelec, tangential, elements, reference)
            e_z, grad_z = lagrange_at(layout.lagrange, axial, elements, reference)
            element_mu = np.broadcast_to(mu[elements][:, None], e_z.shape)
            fields = _faraday(e_t, curl, e_z, grad_z, mode.beta, mode.omega, (element_mu,) * 3)
            normal_d = EPS0 * (eps_xx[elements][:, None] * e_t[..., 0] * normal[..., 0]
                               + eps_yy[elements][:, None] * e_t[..., 1] * normal[..., 1])
            along = fields['hx'] * tangent[..., 0] + fields['hy'] * tangent[..., 1]
            values.append((along, fields['hz'], normal_d))
        jump_h = (np.abs(values[0][0] - values[1][0]) ** 2 + np.abs(values[0][1] - values[1][1]) ** 2) * sides.weights
        jump_d = np.abs(values[0][2] - values[1][2]) ** 2 * sides.weights
        jump_h, jump_d = jump_h.sum(axis=1), jump_d.sum(axis=1)
        terms = np.stack([MU0 * np.abs(mu[sides.elements[:, side]]) * jump_h
                          + jump_d / (EPS0 * eps_scale[sides.elements[:, side]]) for side in (0, 1)], axis=1)
        add_edge_terms(eta_squared, sides, terms)
    return eta_squared


def nodal_fields(layout, mode, mu, nodal_dofs) -> dict:
    """Complex (nodal_dofs.n_dofs,) ex .. hz of a mode at the nodes of a full-mesh Lagrange map of the layout's degree:
    the mean over the non-PEC elements sharing each node (fields jump across material interfaces), 0 at nodes of PEC
    elements only; PML ring nodes in stretched coordinates."""
    samples = _mode_samples(layout.nedelec, layout.lagrange, nodes=True)
    fields = _sample_fields(layout, mode.coefficients, mode.beta, mode.omega,
                            sample_permeability(layout, mu, mode.omega, samples), samples)
    dofs = nodal_dofs.element_dofs[nodal_dofs.element_position[layout.nedelec.elements]].ravel()
    counts = np.bincount(dofs, minlength=nodal_dofs.n_dofs)
    result = {}
    for name, values in fields.items():
        total = np.zeros(nodal_dofs.n_dofs, dtype=np.complex128)
        scatter_add(total, dofs, values)
        result[name] = np.where(counts > 0, total / np.maximum(counts, 1), 0.0)
    return result
