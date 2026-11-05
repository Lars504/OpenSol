"""Conical (fixed out-of-plane k_z) resonances on triangle meshes (em_fdfd_conical on a 'triangles' mesh): fem_modes'
Nedelec (E_x, E_y) / Lagrange (w) spaces, walls and PML ring, squared-Cayley shift-invert through
eigenfrequency.solve_resonances. Host numpy (scipy inside functions)."""
# Formulation (fields ~ exp(j omega t - j k_z z), test fields W e^{+j k_z z}, unconjugated Galerkin: complex symmetric
# with loss / PML, D = I). Unknowns e = (E_x, E_y) (Nedelec, covariant Piola) and w = -j E_z (Lagrange of the same
# degree; trial E_z = j w, test -j w', so E_z E_z' -> w w'). With G = grad w + k_z e and rot e = dE_y/dx - dE_x/dy,
# curl E = (j G_y, -j G_x, rot e), hence
#   K(k_z) = int [ nu_zz rot(e)^2 + G . nu_t G ],   M = int [ e . eps_t e + eps_zz w^2 ],   K x = k0^2 M x,
# nu_t = diag(1 / mu_yy, 1 / mu_xx) (G_x carries H_y): K = blkdiag(S, K_nu) + k_z (G + G^T) + k_z^2 blkdiag(T_nu, 0)
# with fem_modes' parts S, K_nu, G (e rows, w columns), T_nu (the planar fem_bor pencil: r -> 1, m -> k_z). PML:
# fem_modes._coefficients (Lambda = diag(s_y / s_x, s_x / s_y, s_x s_y) on eps and mu) holds for the k_z terms too (z is
# not stretched, e covariant, w scalar). Discrete gradients (e = grad p, w = -k_z p, p Lagrange of the degree) give G =
# 0 and rot e = 0 exactly: the static null space, sent to 0 by the squared Cayley map and dropped.
# Walls, PEC elements, mirrors: fem_modes.fem_mode_layout (pec / antisymmetric faces, the outer side of the PML ring and
# the closure of PEC elements fix tangential e and w; pmc / symmetric faces natural); no periodic / Bloch faces (no
# periodic mesh). Recovery: physical [E_t; E_z] = [e; j w], then fem_modes' Faraday law at beta = k_z (H at the target
# omega, stretched values in the ring): FEMConicalField duck-types the FEMMode attributes fem_modes.element_means /
# nodal_fields / mode_error_indicator read. PML share: energy matrix W = int e . e + w^2 (unstretched), in_window = dof
# points inside the window.
import math
import time
from dataclasses import dataclass

import numpy as np

from ...core.constants import C0, EPS0
from ...fem.assembly import (assemble_curl_curl, assemble_gradient_coupling, assemble_mass, assemble_stiffness,
                             assemble_vector_mass, restrict)
from ...fem.mesh import FACE_BITS
from ...solvers.arpack_shift_invert import ArpackShiftInvertSolver
from .boundaries import BCType
from .eigenfrequency import solve_resonances
from .fem_modes import PERIODIC_TEXT, _coefficients, _sample_fields, fem_mode_layout, sample_permeability

FEM_CONICAL_SIZE_REMEDY = ('coarsen the triangle mesh (larger maximum element size), use discretization linear, or cut '
                           'the cross-section by a symmetric / antisymmetric mirror face')
_PERIODIC_TYPES = frozenset({BCType.PERIODIC, BCType.BLOCH})
_WINDOW_TOLERANCE = 1e-9   # dof points within this times the longest window side of the window count as inside


@dataclass(frozen=True, eq=False)
class FEMConicalSystem:
    """Pencil of one omega and k_z on the layout's free stacked [e; w] dofs: operator A = K - k0^2 M, mass M, energy W
    (scipy CSR), in_window (n,), equilibration (n,) = (|diag K| + k0^2 |diag M|)^-1/2 (the cuDSS row scale), the
    layout and FEMModeMaterials it was assembled from."""
    omega: float
    k_z: float
    layout: object
    materials: object
    operator: object
    mass: object
    energy: object
    in_window: np.ndarray
    equilibration: np.ndarray
    assembly_seconds: float

    @property
    def n_unknowns(self) -> int:
        return int(self.layout.free.size)


@dataclass(frozen=True, eq=False)
class FEMConicalField:
    """Physical [E_t; E_z] coefficients (complex128 (n_stacked,)) of one resonance, k_z [rad/m] as beta and the target
    omega [rad/s] of its H: the FEMMode attributes fem_modes.element_means / nodal_fields / mode_error_indicator read."""
    coefficients: np.ndarray
    beta: float
    omega: float


def fem_conical_layout(mesh, order, boundary_config, pec=None):
    """fem_modes.FEMModeLayout of element degree order (1 / 2) with the walls / PML ring of boundary_config and the
    per-element PEC mask; ValueError for periodic / Bloch faces."""
    for face in FACE_BITS:
        if getattr(boundary_config, face) in _PERIODIC_TYPES:
            raise ValueError(f'em_fdfd_conical on triangles: {PERIODIC_TEXT}')
    return fem_mode_layout(mesh, order, boundary_config, pec)


def assemble_fem_conical(layout, materials, omega, k_z) -> FEMConicalSystem:
    """FEMConicalSystem of fem_modes.FEMModeMaterials (per element, at omega) at omega [rad/s] (PML stretch, k0) and real
    k_z [rad/m] (module header)."""
    import scipy.sparse
    started = time.perf_counter()
    omega, k_z = float(omega), float(k_z)
    if not (math.isfinite(omega) and omega > 0.0):
        raise ValueError(f'assemble_fem_conical needs a finite angular frequency omega > 0 [rad/s], got {omega!r}')
    if not math.isfinite(k_z):
        raise ValueError(f'assemble_fem_conical needs a finite k_z [rad/m], got {k_z!r}')
    if np.shape(materials.pec) != (layout.mesh.n_elements,):
        raise ValueError(f'materials of {np.shape(materials.pec)} elements on a mesh of {layout.mesh.n_elements}')
    inverse_mu_zz, eps_transverse, nu_transverse, eps_axial = _coefficients(layout, materials, omega)
    nedelec, lagrange = layout.nedelec, layout.lagrange
    coupling = assemble_gradient_coupling(nedelec, lagrange, nu_transverse)
    bmat = scipy.sparse.bmat
    stiffness = bmat([[assemble_curl_curl(nedelec, inverse_mu_zz)
                       + k_z ** 2 * assemble_vector_mass(nedelec, nu_transverse), k_z * coupling],
                      [k_z * coupling.T, assemble_stiffness(lagrange, nu_transverse)]], format='csr')
    mass = bmat([[assemble_vector_mass(nedelec, eps_transverse), None],
                 [None, assemble_mass(lagrange, eps_axial)]], format='csr')
    energy = bmat([[assemble_vector_mass(nedelec, 1.0), None], [None, assemble_mass(lagrange, 1.0)]], format='csr')
    free = layout.free
    stiffness, mass, energy = (restrict(matrix, free) for matrix in (stiffness, mass, energy))
    target_squared = (omega / C0) ** 2
    scale = np.abs(stiffness.diagonal()) + target_squared * np.abs(mass.diagonal())
    x0, y0, x1, y1 = (float(value) for value in layout.mesh.window)
    tolerance = _WINDOW_TOLERANCE * max(x1 - x0, y1 - y0)
    points = np.concatenate((nedelec.dof_points(), lagrange.dof_points()))[free]
    in_window = ((points[:, 0] >= x0 - tolerance) & (points[:, 0] <= x1 + tolerance)
                 & (points[:, 1] >= y0 - tolerance) & (points[:, 1] <= y1 + tolerance))
    return FEMConicalSystem(omega=omega, k_z=k_z, layout=layout, materials=materials,
                            operator=(stiffness - target_squared * mass).tocsr(), mass=mass, energy=energy,
                            in_window=in_window, equilibration=1.0 / np.sqrt(np.where(scale > 0.0, scale, 1.0)),
                            assembly_seconds=time.perf_counter() - started)


def solve_fem_conical_resonances(system, n_resonances, eigen_solver=None, warnings=()):
    """eigenfrequency.EigenSolve of up to n_resonances resonances nearest system.omega (squared-Cayley shift-invert by
    eigen_solver, None = default settings: host SuperLU, or a cuDSS LDL^T of the complex symmetric pencil, cudss_mixed
    with the float64 Rayleigh polish); fields = free [e; w] vectors."""
    eigen_solver = ArpackShiftInvertSolver() if eigen_solver is None else eigen_solver
    if system.n_unknowns < 4:
        raise ValueError(f'solve_fem_conical_resonances: only {system.n_unknowns} unknowns remain inside the walls')
    # solve_resonances' A = L + k_t^2 M' with L = K, M' = -M (fem_bor alike): its own factor is of K - s M
    return solve_resonances(system.operator, -system.mass, system.omega, system.energy, system.in_window,
                            n_resonances, spectral_map='squared_cayley', warnings=warnings, eigen_solver=eigen_solver,
                            symmetric=True, equilibration=system.equilibration)


def resonance_field(system, field) -> FEMConicalField:
    """FEMConicalField of a resonance's free [e; w] vector: physical [E_t; E_z = j w], fixed dofs 0."""
    layout = system.layout
    coefficients = np.zeros(layout.n_stacked, dtype=np.complex128)
    coefficients[layout.free] = field
    coefficients[layout.nedelec.n_dofs:] *= 1j
    return FEMConicalField(coefficients=coefficients, beta=system.k_z, omega=system.omega)


def absorbed_power_means(system, field) -> np.ndarray:
    """Float64 (n_elements,) element means of q_abs = 1/2 omega eps0 sum_c (-Im eps_c) |E_c|^2 [W/m^3] of a
    FEMConicalField (0 on PEC elements; the ring in stretched coordinates)."""
    layout, materials = system.layout, system.materials
    fields = _sample_fields(layout, field.coefficients, field.beta, field.omega,
                            sample_permeability(layout, materials.mu, field.omega))
    elements = layout.nedelec.elements
    loss = sum(-np.asarray(getattr(materials, name), dtype=np.complex128)[elements][:, None].imag
               * np.abs(fields[component]) ** 2
               for name, component in (('eps_xx', 'ex'), ('eps_yy', 'ey'), ('eps_zz', 'ez')))
    weights = layout.samples.weights
    means = np.zeros(layout.mesh.n_elements)
    means[elements] = 0.5 * field.omega * EPS0 * (loss * weights).sum(axis=1) / weights.sum(axis=1)
    return means
