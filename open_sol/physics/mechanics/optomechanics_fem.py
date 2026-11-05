"""Optomechanical overlaps on FEM meshes against an axisymmetric (em_fdfd_bor) optical resonance: moving-boundary edge
integrals and the cylindrical photoelastic volume integral with 2 pi r weights, and the g0 table of a solved triangles
resonance against axisymmetric triangle or tetrahedra solids (host numpy)."""
# Conventions (optomechanics.py header for the 2D Yee version; new_modules_plan.md O1). The (r, z) cross-section is the
# drawing plane of em_fdfd_bor on triangles: r = x - the window's x_lo edge (the axis), z = y. Optical fields are
# cylindrical (E_r, E_phi, E_z) samples [V/m] of one resonance ~ exp(j omega t - j m phi): every product E_a* E_b is
# phi-independent, so an integral over the body of revolution is 2 pi int r (...) dA (both numerator and energy carry
# the 2 pi r weight). Mechanical inputs are the azimuthally uniform (m = 0, or phi-averaged) parts of the mode: q_n =
# q . n on the interface samples, strains as cylindrical tensor components (rr, phiphi, zz, phiz, rz, rphi).
# Moving boundary (Johnson et al., PRE 65, 066611) per interface sample with n the unit normal out of side 0:
#   2 pi r w eps0 [Delta eps |E_par|^2 - Delta(1/eps) |D_n / eps0|^2],  Delta = side 0 - side 1 (relative, isotropic),
# |E_par|^2 = |E_phi|^2 + |t . (E_r, E_z)|^2 and |D_n / eps0|^2 = |eps n . (E_r, E_z)|^2, each the mean of the two
# sides' values (continuous for the exact field; FEM D_n is not). Swapping the sides flips n and Delta: no orientation
# rule. I_mb = sum q_n x weight. Photoelastic: I_pe = sum 2 pi r w eps0 E* . Delta eps E, Delta eps_ij = -eps_ii eps_jj
# Delta(1/eps)_ij, Nye Delta(1/eps)_I = p_IJ S_J in cylindrical Voigt order (rr, phiphi, zz, phiz, rz, rphi;
# engineering shear), isotropic p11 / p12 / p44 (unset p44 = (p11 - p12) / 2: optomechanics.shear_photoelastic_coefficient;
# a rotated crystal tensor is not phi-invariant: not supported). The hoop strain S_phiphi = u_r / r. Rate g_om =
# -(omega / 2) Re(I / U), U = sum 2 pi r w eps0 sum_c eps_c |E_c|^2 over the window (PML ring excluded) [rad/s/m for q in
# m of the scaled mode shape].
# 3D solids (tetrahedra) -> (r, z): the mode is sampled on a ring of n_phi points per (r, z) point about the 3D z axis
# through axis_point (3D point = axis_point + (r cos phi, r sin phi, z)); the phi-average of q . n (n = n_r e_r(phi) +
# n_z e_z) is exact for the travelling resonance, so the crown (m != 0) parts drop out. The uniform phi grid averages a
# displacement of azimuthal orders up to n_phi - 2 exactly (q . n carries orders m +- 1).
# Model wiring (O2, axisymmetric_coupling_table; OptomechanicalCoupling.evaluate dispatches here for em_fdfd_bor
# optics): the optical samples come from the solved FEMBORPhysics (its `assembly`, fem_bor.electric_field_at) on the
# window, non-PEC elements (interfaces touching PEC / the PML ring dropped); the solid is sampled by point location in
# its own mesh, so it may sit on another mesh of the same drawing. An interface sample is located SIDE_SHIFT x its edge
# length off the edge into each side: a side counts as solid when that point lies in a stiff element, q_n comes from
# side 0 when it is solid, else side 1, else 0 (the zero-padded mode would move void interfaces next to the solid).
# Prescribed q_n (no mode): value x (solid(side 0) - solid(side 1)) = the outward normal motion of the solid, phi-
# averaged for a 3D solid. An axisymmetric solid of azimuthal order m >= 1 (cos(m phi) amplitudes U_r, U_z, U_phi)
# couples through the ring mean of cos(m phi) over RING_POINTS angles (round-off for 0 < m < RING_POINTS), its
# photoelastic part is 0 likewise. Rows: Q scaled to |Q| = 1 at the peak dof (or the solid dof nearest the reference
# point, triangles only), m_eff = Q^T M Q (whole rings / the 3D body: kg, no length, mirror factor 1), complex (damped)
# modes rotated to a real peak and taken real.
import math
from dataclasses import dataclass

import numpy as np

from ...core.constants import C0, EPS0, HBAR
from ...fem.elements import element_geometry
from ...fem.indicators import edge_sides
from ...fem.quadrature import triangle_rule
from ...fem.sampling import LocatedPoints, evaluate, locator
from .optomechanics import (OPTOMECHANICS_TABLE_COLUMNS, OPTOMECHANICS_TABLE_UNITS, inverse_permittivity_changes,
                            permittivity_changes, shear_photoelastic_coefficient)

INTERFACE_DEGREE = 6   # Gauss rule along interface edges (+2 on curved edges): |E|^2 of P2 fields times r
VOLUME_DEGREE = 6      # Dunavant rule of the photoelastic / energy volume samples
RING_POINTS = 64       # suggested phi samples per ring of the 3D -> (r, z) reduction
SIDE_SHIFT = 1e-7      # interface samples are located this share of their edge length into each side's element
_CHUNK_ELEMENTS = 8192   # optical elements per field-evaluation block
_CHUNK_RINGS = 2048      # (r, z) samples per ring-location block of a 3D solid
_TWO_PI = 2.0 * math.pi


# ---- mesh sampling (TriangleMesh of the optical physics) ----
def interface_sides(mesh, element_eps, region=None, excluded=None, geometry_order=2, degree=INTERFACE_DEGREE):
    """fem.indicators.EdgeSides of the interior edges whose two elements differ in element_eps (complex (n_elements,)
    relative permittivity, isotropic): side 0 = edge_elements[:, 0], normals out of side 0; region (bool per element,
    None = all) keeps edges touching it, excluded (bool, e.g. PEC) drops edges touching it. geometry_order = the field
    map's (fem_bor: layout.nedelec.geometry_order), so the sample points are where the fields are evaluated."""
    element_eps = np.asarray(element_eps, dtype=np.complex128)
    if element_eps.shape != (mesh.n_elements,):
        raise ValueError(f'interface_sides: element_eps shape {element_eps.shape} != ({mesh.n_elements},)')
    interior = np.flatnonzero(mesh.edge_elements[:, 1] >= 0)
    pairs = mesh.edge_elements[interior].astype(np.int64)
    first, second = element_eps[pairs[:, 0]], element_eps[pairs[:, 1]]
    keep = np.abs(first - second) > 1e-12 * np.maximum(np.abs(first), np.abs(second))
    if region is not None:
        region = np.asarray(region, dtype=bool)
        keep &= region[pairs[:, 0]] | region[pairs[:, 1]]
    if excluded is not None:
        excluded = np.asarray(excluded, dtype=bool)
        keep &= ~(excluded[pairs[:, 0]] | excluded[pairs[:, 1]])
    return edge_sides(mesh, interior[keep], np.ones(mesh.n_elements, dtype=bool), geometry_order, degree)


def element_rule(mesh, elements, degree=VOLUME_DEGREE, geometry_order=2) -> tuple:
    """(reference points (k, q, 2), physical points (k, q, 2) [m], weights (k, q) [m^2] = Dunavant weight x |det J|) of
    mesh elements (int indices, k of them); the photoelastic / energy volume samples."""
    elements = np.asarray(elements, dtype=np.int64)
    rule = triangle_rule(degree)
    reference = np.broadcast_to(rule.points, (len(elements),) + rule.points.shape)
    geometry = element_geometry(mesh, elements, reference, geometry_order)
    return reference, geometry.points, rule.weights[None, :] * np.abs(geometry.determinant)


# ---- optical side ----
def _diagonal_eps(eps, field_shape):
    """Complex (..., 3) per-component eps of a scalar (...) or diagonal (..., 3) relative permittivity."""
    eps = np.asarray(eps, dtype=np.complex128)
    if eps.ndim == len(field_shape) and eps.shape[-1] == 3:
        return np.broadcast_to(eps, field_shape)
    return np.broadcast_to(eps[..., None], field_shape)


def electric_energy(radius, weights, field, eps) -> complex:
    """U = sum 2 pi r w eps0 sum_c eps_c |E_c|^2 [J-like, V^2 F/m^2 m^3] of samples: radius (...) [m] from the axis,
    area weights (...) [m^2], cylindrical field (..., 3) [V/m], eps scalar (...) or diagonal (..., 3) (rr, phiphi, zz)."""
    field = np.asarray(field, dtype=np.complex128)
    components = _diagonal_eps(eps, field.shape)
    density = np.sum(components * np.abs(field) ** 2, axis=-1)
    return complex(np.sum(_TWO_PI * np.asarray(radius, dtype=np.float64) * np.asarray(weights, dtype=np.float64)
                          * EPS0 * density))


def moving_boundary_weights(radius, weights, normals, field_side0, field_side1, eps_side0, eps_side1) -> np.ndarray:
    """Complex (...) per interface sample 2 pi r w eps0 [Delta eps |E_par|^2 - Delta(1/eps) |D_n / eps0|^2] (header):
    radius / arc weights (...) [m], unit (n_r, n_z) normals (..., 2) out of side 0, cylindrical fields (..., 3) [V/m] of
    each side at the sample, isotropic relative eps (...) per side; I_mb = sum q_n x weights."""
    normals = np.asarray(normals, dtype=np.float64)
    normal_r, normal_z = normals[..., 0], normals[..., 1]
    eps_side0 = np.asarray(eps_side0, dtype=np.complex128)
    eps_side1 = np.asarray(eps_side1, dtype=np.complex128)

    def parts(field, eps):
        field = np.asarray(field, dtype=np.complex128)
        normal = normal_r * field[..., 0] + normal_z * field[..., 2]
        tangential = normal_r * field[..., 2] - normal_z * field[..., 0]
        return np.abs(tangential) ** 2 + np.abs(field[..., 1]) ** 2, np.abs(eps * normal) ** 2

    parallel_0, displacement_0 = parts(field_side0, eps_side0)
    parallel_1, displacement_1 = parts(field_side1, eps_side1)
    parallel = 0.5 * (parallel_0 + parallel_1)
    displacement = 0.5 * (displacement_0 + displacement_1)
    jump = (eps_side0 - eps_side1) * parallel - (1.0 / eps_side0 - 1.0 / eps_side1) * displacement
    return (_TWO_PI * np.asarray(radius, dtype=np.float64) * np.asarray(weights, dtype=np.float64) * EPS0 * jump)


# ---- mechanical side ----
def normal_displacement(displacement, normals) -> np.ndarray:
    """q_n = u_r n_r + u_z n_z (...) of (u_r, u_z) samples (..., 2) [m] and unit normals (..., 2)."""
    displacement, normals = np.asarray(displacement), np.asarray(normals, dtype=np.float64)
    return displacement[..., 0] * normals[..., 0] + displacement[..., 1] * normals[..., 1]


def prescribed_normal_displacement(value, radius, height) -> np.ndarray:
    """q_n (radius.shape) of a prescribed boundary motion: a number [m] or a callable(radius, height) of the (r, z)
    sample arrays [m] (e.g. 1 for the uniform-expansion benchmark, no mechanics solve)."""
    radius = np.asarray(radius, dtype=np.float64)
    values = value(radius, np.asarray(height, dtype=np.float64)) if callable(value) else value
    return np.array(np.broadcast_to(np.asarray(values), radius.shape))


def axisymmetric_strains(radius, displacement, gradient) -> np.ndarray:
    """Cylindrical tensor strains (..., 6) (rr, phiphi, zz, phiz, rz, rphi) of an m = 0 (u_r, u_z) mode: samples (...,
    2) [m], gradients (..., 2, 2) [c, d] = d u_c / d (r, z)_d, radius (...) [m]; hoop u_r / r (d u_r / dr at r = 0)."""
    radius = np.asarray(radius, dtype=np.float64)
    displacement, gradient = np.asarray(displacement), np.asarray(gradient)
    off_axis = radius > 0.0
    hoop = np.where(off_axis, displacement[..., 0] / np.where(off_axis, radius, 1.0), gradient[..., 0, 0])
    zero = np.zeros_like(hoop)
    return np.stack([gradient[..., 0, 0], hoop, gradient[..., 1, 1], zero,
                     0.5 * (gradient[..., 0, 1] + gradient[..., 1, 0]), zero], axis=-1)


def cylindrical_photoelastic_changes(eps, strains, p11, p12, p44=None) -> tuple:
    """(Delta eps_rr, _phiphi, _zz, _phiz, _rz, _rphi) complex (...) of diagonal eps (..., 3) and cylindrical tensor
    strains (..., 6) with isotropic p11 / p12 / p44 (scalars or (...); p44 None / NaN = (p11 - p12) / 2)."""
    strains = np.asarray(strains)
    s_rr, s_pp, s_zz, s_pz, s_rz, s_rp = (strains[..., index] for index in range(6))
    voigt = (s_rr, s_pp, s_zz, 2.0 * s_pz, 2.0 * s_rz, 2.0 * s_rp)
    inverse = inverse_permittivity_changes(voigt, p11, p12, shear_photoelastic_coefficient(p11, p12, p44))
    eps = np.asarray(eps)
    return permittivity_changes(eps[..., 0], eps[..., 1], eps[..., 2], inverse)


def photoelastic_integral(radius, weights, field, eps, strains, p11, p12, p44=None) -> complex:
    """I_pe = sum 2 pi r w eps0 E* . Delta eps E (header) of volume samples: radius / area weights (...) [m, m^2],
    cylindrical field (..., 3) [V/m], eps scalar (...) or diagonal (..., 3), strains (..., 6) (0 off the solid)."""
    field = np.asarray(field, dtype=np.complex128)
    change_rr, change_pp, change_zz, change_pz, change_rz, change_rp = cylindrical_photoelastic_changes(
        _diagonal_eps(eps, field.shape), strains, p11, p12, p44)
    e_r, e_phi, e_z = field[..., 0], field[..., 1], field[..., 2]
    contraction = (change_rr * np.abs(e_r) ** 2 + change_pp * np.abs(e_phi) ** 2 + change_zz * np.abs(e_z) ** 2
                   + change_pz * 2.0 * (np.conj(e_phi) * e_z).real + change_rz * 2.0 * (np.conj(e_r) * e_z).real
                   + change_rp * 2.0 * (np.conj(e_r) * e_phi).real)
    return complex(np.sum(_TWO_PI * np.asarray(radius, dtype=np.float64) * np.asarray(weights, dtype=np.float64)
                          * EPS0 * contraction))


# ---- 3D solids: ring sampling and the phi-average ----
def ring_points(radius, height, n_phi, axis_point=(0.0, 0.0, 0.0)) -> tuple:
    """(3D points (..., n_phi, 3) [m], phi (n_phi,) [rad]) of a uniform ring per (r, z) sample (radius / height (...)):
    axis_point + (r cos phi, r sin phi, z), phi from +x about the 3D z axis."""
    count = int(n_phi)
    if count < 1:
        raise ValueError(f'ring_points: n_phi must be >= 1, got {n_phi!r}')
    phi = _TWO_PI * np.arange(count) / count
    radius = np.asarray(radius, dtype=np.float64)[..., None]
    height = np.asarray(height, dtype=np.float64)[..., None]
    x0, y0, z0 = (float(value) for value in axis_point)
    x, y, z = np.broadcast_arrays(x0 + radius * np.cos(phi), y0 + radius * np.sin(phi), z0 + height + 0.0 * phi)
    return np.stack([x, y, z], axis=-1), phi


def cylindrical_vectors(vectors, phi) -> np.ndarray:
    """(u_r, u_phi, u_z) (..., n_phi, 3) of Cartesian vectors (..., n_phi, 3) on the ring angles phi (n_phi,)."""
    vectors = np.asarray(vectors)
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    return np.stack([cos_phi * vectors[..., 0] + sin_phi * vectors[..., 1],
                     cos_phi * vectors[..., 1] - sin_phi * vectors[..., 0], vectors[..., 2]], axis=-1)


def cylindrical_strains(strains, phi) -> np.ndarray:
    """Cylindrical tensor strains (..., n_phi, 6) (rr, phiphi, zz, phiz, rz, rphi) of Cartesian tensor strains (...,
    n_phi, 6) (xx, yy, zz, yz, xz, xy) on the ring angles phi (n_phi,)."""
    strains = np.asarray(strains)
    s_xx, s_yy, s_zz, s_yz, s_xz, s_xy = (strains[..., index] for index in range(6))
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    return np.stack([cos_phi ** 2 * s_xx + sin_phi ** 2 * s_yy + 2.0 * cos_phi * sin_phi * s_xy,
                     sin_phi ** 2 * s_xx + cos_phi ** 2 * s_yy - 2.0 * cos_phi * sin_phi * s_xy,
                     s_zz,
                     cos_phi * s_yz - sin_phi * s_xz,
                     cos_phi * s_xz + sin_phi * s_yz,
                     cos_phi * sin_phi * (s_yy - s_xx) + (cos_phi ** 2 - sin_phi ** 2) * s_xy], axis=-1)


def ring_average_normal_displacement(ring_displacement, normals, phi) -> np.ndarray:
    """<q . n>_phi (...) of Cartesian displacements (..., n_phi, 3) [m] sampled on ring_points, with the (n_r, n_z) unit
    normals (..., 2) of each (r, z) sample: the m = 0 part the travelling resonance sees."""
    cylindrical = cylindrical_vectors(ring_displacement, phi)
    normals = np.asarray(normals, dtype=np.float64)
    return np.mean(cylindrical[..., 0] * normals[..., None, 0] + cylindrical[..., 2] * normals[..., None, 1], axis=-1)


def ring_average_strains(ring_strains, phi) -> np.ndarray:
    """<cylindrical strains>_phi (..., 6) of Cartesian tensor strains (..., n_phi, 6) sampled on ring_points."""
    return np.mean(cylindrical_strains(ring_strains, phi), axis=-2)


# ---- coupling rate ----
@dataclass(frozen=True, eq=False)
class AxisymmetricOverlap:
    """Optical side of g_om for one em_fdfd_bor resonance: omega [rad/s], energy U (electric_energy), per-sample
    moving_boundary_weights (None: no interface) and the photoelastic volume samples (radius, weights, field, eps) or
    None; per mechanical mode only q_n / strains change."""
    omega: float
    energy: complex
    boundary_weights: object = None   # complex (...) of moving_boundary_weights
    volume_samples: tuple = None      # (radius (...), weights (...), field (..., 3), eps (...) or (..., 3))

    def __post_init__(self):
        if complex(self.energy) == 0.0:
            raise ValueError('the optical resonance field vanishes on the window: no perturbation formula')

    def moving_boundary(self, normal_displacements) -> complex:
        """I_mb = sum q_n x boundary weight of q_n samples (the weights' shape) [m]."""
        if self.boundary_weights is None:
            raise ValueError('AxisymmetricOverlap has no interface samples (boundary_weights None)')
        return complex(np.sum(np.asarray(normal_displacements) * self.boundary_weights))

    def photoelastic(self, strains, p11, p12, p44=None) -> complex:
        """I_pe of cylindrical tensor strains (..., 6) at the volume samples (0 off the solid), isotropic p."""
        if self.volume_samples is None:
            raise ValueError('AxisymmetricOverlap has no volume samples (volume_samples None)')
        radius, weights, field, eps = self.volume_samples
        return photoelastic_integral(radius, weights, field, eps, strains, p11, p12, p44)

    def rate(self, integral) -> float:
        """g_om = -(omega / 2) Re(I / U) [rad/s/m] of an overlap integral I (I_mb, I_pe or their sum)."""
        return float((-0.5 * self.omega * complex(integral) / complex(self.energy)).real)


# ---- model wiring (O2): the optical side of a solved em_fdfd_bor physics on triangles ----
@dataclass(frozen=True, eq=False)
class BOROpticalSamples:
    """One em_fdfd_bor (triangles) resonance for the g0 table: its AxisymmetricOverlap and the drawing-plane points the
    solid is sampled at (interface points / normals / edge lengths, volume points), the axis x, order, wavelength, Q."""
    overlap: AxisymmetricOverlap
    edge_points: np.ndarray       # (k, q, 2) [m]
    edge_normals: np.ndarray      # (k, q, 2) unit, out of side 0
    edge_lengths: np.ndarray      # (k,) [m]
    volume_points: np.ndarray     # (v, q, 2) [m], the samples of overlap.volume_samples
    axis_x: float                 # [m] x of the axis (the optical window's x_lo edge)
    azimuthal_order: int
    wavelength: float             # [m] at Re omega
    q_factor: float


def bor_optical_samples(optical, number) -> BOROpticalSamples:
    """BOROpticalSamples of resonance `number` (1-based) of a solved em_fdfd_bor physics on a triangles mesh (duck-typed:
    `assembly`, `eigen_solve`); ValueError on a tensor mesh or for a missing resonance."""
    from ..em.fem_bor import FEMBORAssembly, electric_field_at, resonance_coefficients
    if not isinstance(getattr(optical, 'assembly', None), FEMBORAssembly):
        raise ValueError(f"the optomechanical overlap with em_fdfd_bor '{optical.name}' needs it on a triangles mesh "
                         '(FEMBORPhysics.assembly: the finite-element resonance); the rectangular-cell body-of-revolution '
                         'solve has no element interfaces to integrate over')
    solve, assembly = optical.eigen_solve, optical.assembly
    if solve is None or assembly is None:
        raise ValueError(f"optical physics '{optical.name}' has no eigenfrequency solve")
    if not 1 <= number <= len(solve.resonances):
        raise ValueError(f"optical resonance {number} of '{optical.name}' does not exist: the eigen-solve found "
                         f'{len(solve.resonances)} (raise n_optical_resonances)')
    resonance = solve.resonances[number - 1]
    order = int(resonance.azimuthal_order)
    layout = assembly.layout
    mesh = layout.mesh
    coefficients = resonance_coefficients(layout, order, resonance.field)
    materials = assembly.materials
    element_eps = np.stack([materials.component(name) for name in ('r', 'phi', 'z')], axis=-1)   # rr, phiphi, zz
    usable = (np.asarray(mesh.pml_faces) == 0) & ~np.asarray(materials.pec, dtype=bool)   # window, not PEC
    elements = np.flatnonzero(usable)
    if not elements.size:
        raise ValueError(f"optical physics '{optical.name}': no window element outside PEC")
    axis_x = float(layout.axis_x)
    geometry_order = layout.nedelec.geometry_order

    parts = {name: [] for name in ('radius', 'weights', 'field', 'eps', 'points')}
    for start in range(0, len(elements), _CHUNK_ELEMENTS):
        chunk = elements[start:start + _CHUNK_ELEMENTS]
        reference, points, weights = element_rule(mesh, chunk, geometry_order=geometry_order)
        field, _ = electric_field_at(assembly, coefficients, order, chunk, reference)
        parts['radius'].append(points[..., 0] - axis_x)
        parts['weights'].append(weights)
        parts['field'].append(field)
        parts['eps'].append(np.broadcast_to(element_eps[chunk][:, None, :], field.shape))
        parts['points'].append(points)
    radius, weights, field, eps, points = (np.concatenate(parts[name])
                                           for name in ('radius', 'weights', 'field', 'eps', 'points'))
    energy = electric_energy(radius, weights, field, eps)

    sides = interface_sides(mesh, element_eps[:, 1], excluded=~usable, geometry_order=geometry_order)
    if len(sides.edges):
        side_fields = [electric_field_at(assembly, coefficients, order, sides.elements[:, side],
                                         sides.reference_points(side))[0] for side in (0, 1)]
        side_eps = [np.broadcast_to(element_eps[sides.elements[:, side], 1][:, None], sides.weights.shape)
                    for side in (0, 1)]
        boundary = moving_boundary_weights(sides.points[..., 0] - axis_x, sides.weights, sides.normals, *side_fields,
                                           *side_eps)
    else:
        boundary = np.zeros(sides.weights.shape, dtype=np.complex128)
    omega = float(complex(resonance.omega).real)
    overlap = AxisymmetricOverlap(omega=omega, energy=energy, boundary_weights=boundary,
                                  volume_samples=(radius, weights, field, eps))
    return BOROpticalSamples(overlap=overlap, edge_points=sides.points, edge_normals=sides.normals,
                             edge_lengths=sides.lengths, volume_points=points, axis_x=axis_x, azimuthal_order=order,
                             wavelength=_TWO_PI * C0 / omega, q_factor=float(resonance.q_factor))


# ---- the solid side: modes of an axisymmetric triangle solid or a tetrahedra solid at the optical samples ----
def _shifted(samples, sign):
    """Interface sample points moved sign x SIDE_SHIFT x edge length along the normal (-1: into side 0, +1: side 1)."""
    return samples.edge_points + sign * SIDE_SHIFT * samples.edge_lengths[:, None, None] * samples.edge_normals


def ring_factor(azimuthal_order, n_phi=RING_POINTS) -> float:
    """Mean of cos(m phi) over n_phi uniform ring angles: the share of a cos(m phi) mode the travelling resonance sees."""
    count = int(n_phi)
    return float(np.mean(np.cos(int(azimuthal_order) * _TWO_PI * np.arange(count) / count)))


class _TriangleSolid:
    """Axisymmetric triangle solid (modal_overlap_data kind 'triangles'): its modes on the solid Lagrange map, located at
    the shifted interface points and the volume points of the optical samples (mode-independent, kept)."""

    def __init__(self, data, samples):
        if not data.get('axisymmetric'):
            raise ValueError('em_fdfd_bor optics pair with an axisymmetric solid (geometry axisymmetric); this triangle '
                             'solid is planar')
        self.dofs = data['dofs']
        self.azimuthal_order = int(data.get('azimuthal_order') or 0)
        solid = np.asarray(data['solid'], dtype=bool)
        finder = locator(self.dofs.mesh, self.dofs.geometry_order)

        def located(points):   # LocatedPoints: the basis is computed once for every mode and component
            location = finder.locate(points.reshape(-1, 2))
            inside = location.found & solid[np.maximum(location.element, 0)]
            return LocatedPoints(location), inside.reshape(points.shape[:-1])

        self.normals = samples.edge_normals
        self.sides = [located(_shifted(samples, sign)) for sign in (-1.0, 1.0)]
        self.volume, self.volume_inside = located(samples.volume_points)
        self.radius = samples.volume_points[..., 0] - samples.axis_x
        element = np.maximum(self.volume.location.element, 0)
        flat_inside = self.volume_inside.reshape(-1)
        self.photoelastic_coefficients = tuple(
            np.where(flat_inside, np.asarray(data[name], dtype=np.float64)[element], 0.0).reshape(self.radius.shape)
            for name in ('p11', 'p12', 'p44'))   # p44 NaN = unset: isotropic

    def solid_fractions(self) -> tuple:
        """(side 0, side 1) 1 / 0 per interface sample: that side lies in a stiff element."""
        return tuple(inside.astype(np.float64) for _, inside in self.sides)

    def nearest_dof(self, point) -> int:
        """Solid dof whose interpolation point is nearest the (x, y) point [m]."""
        positions = self.dofs.dof_points()
        return int(np.argmin(np.sum((positions - np.asarray(point, dtype=np.float64)) ** 2, axis=1)))

    def _values(self, mode, location, shape, gradient=False):
        """((U_r, U_z) (..., 2), gradients (..., 2, 2) [c, d] or None) of a mode (n_dofs, c) at LocatedPoints (NaN:
        0)."""
        values, gradients = [], []
        for component in (0, 1):
            result = evaluate(self.dofs, mode[:, component], location=location, gradient=gradient)
            value, slope = result if gradient else (result, None)
            values.append(np.nan_to_num(value).reshape(shape))
            if gradient:
                gradients.append(np.nan_to_num(slope).reshape(shape + (2,)))
        return np.stack(values, axis=-1), (np.stack(gradients, axis=-2) if gradient else None)

    def normal_displacements(self, mode) -> np.ndarray:
        """q_n (k, q) [m] of a scaled mode: side 0 where solid, else side 1, else 0 (azimuthal order: x ring_factor)."""
        shape = self.normals.shape[:-1]
        result = np.zeros(shape)
        for location, inside in reversed(self.sides):   # side 0 written last: it wins
            values, _ = self._values(mode, location, shape)
            result = np.where(inside, normal_displacement(values, self.normals), result)
        return result * ring_factor(self.azimuthal_order) if self.azimuthal_order else result

    def photoelastic(self, mode, overlap) -> complex:
        """I_pe of a scaled m = 0 mode (axisymmetric strains at the volume samples inside the solid); 0 for m >= 1."""
        if self.azimuthal_order:
            return 0.0j
        values, gradients = self._values(mode, self.volume, self.radius.shape, gradient=True)
        strains = np.where(self.volume_inside[..., None], axisymmetric_strains(self.radius, values, gradients), 0.0)
        return overlap.photoelastic(strains, *self.photoelastic_coefficients)


class _TetrahedronSolid:
    """Tetrahedra solid (kind 'tetrahedra'): its modes on the full Lagrange map, sampled on rings of RING_POINTS angles
    about the 3D z axis through axis_point at the shifted interface points and at the volume samples inside the solid's
    (r, z) extent (locations kept)."""

    def __init__(self, data, samples, axis_point):
        from ...fem.sampling3d import PointLocator3D
        self.dofs = data['dofs']
        mesh = self.dofs.mesh
        self.solid = np.asarray(data['solid'], dtype=bool)
        if not self.solid.any():
            raise ValueError('the 3D solid has no element with stiffness')
        self.axis_point = tuple(float(value) for value in axis_point)
        self.finder = PointLocator3D(mesh)
        self.p11 = np.asarray(data['p11'], dtype=np.float64)
        self.p12 = np.asarray(data['p12'], dtype=np.float64)
        self.normals = samples.edge_normals
        self.sides = []
        for sign in (-1.0, 1.0):
            shifted = _shifted(samples, sign)
            self.sides.append(self._ring(shifted[..., 0] - samples.axis_x, shifted[..., 1]))
        nodes = mesh.points[np.unique(mesh.tets[self.solid])] - np.asarray(self.axis_point)
        node_radius, node_height = np.hypot(nodes[:, 0], nodes[:, 1]), nodes[:, 2]
        radius = (samples.volume_points[..., 0] - samples.axis_x).reshape(-1)
        height = samples.volume_points[..., 1].reshape(-1)
        self.volume_rows = np.flatnonzero((radius >= node_radius.min()) & (radius <= node_radius.max())
                                          & (height >= node_height.min()) & (height <= node_height.max()))
        self.volume = self._ring(radius[self.volume_rows], height[self.volume_rows])

    def _ring(self, radius, height):
        """(elements (N,), reference (N, 3), inside (..., n_phi), phi) of rings at (r, z) arrays (...) [m]."""
        points, phi = ring_points(radius, height, RING_POINTS, self.axis_point)
        flat = points.reshape(-1, 3)
        elements = np.full(len(flat), -1, dtype=np.int64)
        reference = np.zeros((len(flat), 3))
        block = _CHUNK_RINGS * RING_POINTS
        for start in range(0, len(flat), block):
            found, local = self.finder.locate(flat[start:start + block])
            elements[start:start + block] = found
            reference[start:start + block] = np.nan_to_num(local)
        inside = (elements >= 0) & self.solid[np.maximum(elements, 0)]
        return elements, reference, inside.reshape(points.shape[:-1]), phi

    def _evaluate(self, mode, elements, reference, gradient=False):
        """Cartesian values (N, 3) and gradients (N, 3, 3) [c, d] (or None) of a mode (n_dofs, 3) at located points (0
        outside the mesh): one basis (and with gradient one geometry) per chunk for all three components."""
        from ...fem.elements3d import lagrange_basis_3d, tetrahedron_geometry
        dofs = self.dofs
        values = np.zeros((len(elements), 3))
        gradients = np.zeros((len(elements), 3, 3)) if gradient else None
        block = _CHUNK_RINGS * RING_POINTS
        for start in range(0, len(elements), block):
            found = start + np.flatnonzero(elements[start:start + block] >= 0)
            if not found.size:
                continue
            basis, reference_gradients = lagrange_basis_3d(dofs.order, reference[found])   # (p, n), (p, n, 3)
            local = mode[dofs.element_dofs[elements[found]]]   # (p, n, 3)
            values[found] = np.einsum('pn,pnc->pc', basis, local)
            if gradient:
                inverse = tetrahedron_geometry(dofs.mesh, elements[found]).inverse
                physical = np.einsum('pji,pnj->pni', inverse, reference_gradients)
                gradients[found] = np.einsum('pnc,pni->pci', local, physical)
        return values, gradients

    def solid_fractions(self) -> tuple:
        """(side 0, side 1) share of each interface sample's ring inside stiff elements."""
        return tuple(inside.mean(axis=-1) for _, _, inside, _ in self.sides)

    def nearest_dof(self, point) -> int:
        raise ValueError("normalisation 'point' is not available for a solid_mechanics_3d solid: use 'max'")

    def normal_displacements(self, mode) -> np.ndarray:
        """<q . n>_phi (k, q) [m] of a scaled mode: per ring angle side 0 where solid, else side 1, else 0."""
        result = np.zeros(self.normals.shape[:-1] + (RING_POINTS,))
        for elements, reference, inside, phi in reversed(self.sides):   # side 0 written last: it wins
            values, _ = self._evaluate(mode, elements, reference)
            cylindrical = cylindrical_vectors(values.reshape(inside.shape + (3,)), phi)
            normal = (cylindrical[..., 0] * self.normals[..., None, 0]
                      + cylindrical[..., 2] * self.normals[..., None, 1])
            result = np.where(inside, normal, result)
        return result.mean(axis=-1)

    def photoelastic(self, mode, overlap) -> complex:
        """I_pe of a scaled mode: cylindrical strains and p per ring angle inside the solid, averaged over the ring."""
        rows = self.volume_rows
        if not rows.size:
            return 0.0j
        elements, reference, inside, phi = self.volume
        _, gradients = self._evaluate(mode, elements, reference, gradient=True)
        g = gradients.reshape(inside.shape + (3, 3))
        cartesian = np.stack([g[..., 0, 0], g[..., 1, 1], g[..., 2, 2], 0.5 * (g[..., 1, 2] + g[..., 2, 1]),
                              0.5 * (g[..., 0, 2] + g[..., 2, 0]), 0.5 * (g[..., 0, 1] + g[..., 1, 0])], axis=-1)
        strains = np.where(inside[..., None], cylindrical_strains(cartesian, phi), 0.0)
        element = np.maximum(elements, 0).reshape(inside.shape)
        p11, p12 = (np.where(inside, values[element], 0.0) for values in (self.p11, self.p12))
        radius, weights, field, eps = overlap.volume_samples

        def selected(values):
            return values.reshape((-1,) + values.shape[radius.ndim:])[rows]

        return photoelastic_integral(selected(radius)[:, None], selected(weights)[:, None] / RING_POINTS,
                                     selected(field)[:, None, :], selected(eps)[:, None, :], strains, p11, p12)


# ---- the g0 table ----
def _real_mode(displacement, components) -> np.ndarray:
    """Float64 (n_dofs, components) mode shape; a complex (damped) mode rotated to a real peak dof, its real part."""
    mode = np.asarray(displacement).reshape(-1, components)
    if np.iscomplexobj(mode):
        peak = np.unravel_index(int(np.argmax(np.abs(mode))), mode.shape)
        mode = (mode * np.exp(-1j * np.angle(mode[peak]))).real
    return np.asarray(mode, dtype=np.float64)


def _coupling_row(number, resonance, overlap, sampler, mass, components, reference, moving_boundary, photoelastic):
    """OPTOMECHANICS_TABLE_COLUMNS row of one mechanical mode (module header; dn_eff_dx and brillouin_gain NaN)."""
    mode = _real_mode(resonance.displacement, components)
    norms = np.sqrt(np.sum(mode ** 2, axis=1))
    scale = float(np.max(norms)) if reference is None else float(norms[reference])
    if not scale > 0.0:
        raise ValueError(f'mechanical mode {number} does not move the reference dof: choose another reference point')
    scaled = mode / scale
    moving = overlap.moving_boundary(sampler.normal_displacements(scaled)) if moving_boundary else 0.0j
    photo = sampler.photoelastic(scaled, overlap) if photoelastic else 0.0j
    flat = scaled.reshape(-1)
    effective_mass = float(np.real(flat @ (mass @ flat)))
    omega_mechanical = float(complex(resonance.omega).real)
    zero_point = math.sqrt(HBAR / (2.0 * effective_mass * omega_mechanical))
    rate_moving, rate_photo = overlap.rate(moving), overlap.rate(photo)
    rate = rate_moving + rate_photo
    to_hz = zero_point / _TWO_PI
    return [number, omega_mechanical / _TWO_PI, effective_mass, zero_point, rate_moving, rate_photo, rate, math.nan,
            rate_moving * to_hz, rate_photo * to_hz, rate * to_hz, math.nan]


def axisymmetric_coupling_table(coupling, optical, solid, data) -> dict:
    """'table' of an OptomechanicalCoupling against an em_fdfd_bor (triangles) resonance: one row per mode of the solid's
    last eigen-solve (modal_overlap_data kind 'triangles' (axisymmetric) or 'tetrahedra'), the prescribed q_n rate."""
    if coupling.backward or data.get('axial_wavenumber') is not None:
        raise ValueError(f"optomechanical_coupling '{coupling.name}': em_fdfd_bor optics take forward scattering of "
                         'standing vibration modes only (no guided k_z)')
    kind = data.get('kind', 'tensor')
    if kind not in ('triangles', 'tetrahedra'):
        raise ValueError(f"optomechanical_coupling '{coupling.name}': em_fdfd_bor optics need the solid on a triangles "
                         f'mesh (geometry axisymmetric) or a tetrahedra mesh, not a {kind} one')
    samples = bor_optical_samples(optical, coupling.optical_mode)
    sampler = (_TriangleSolid(data, samples) if kind == 'triangles'
               else _TetrahedronSolid(data, samples, coupling.axis_point))
    overlap = samples.overlap
    reference = None if coupling.reference_point is None else sampler.nearest_dof(coupling.reference_point)
    rows = [_coupling_row(number, resonance, overlap, sampler, data['mass'], int(data['components']), reference,
                          coupling.moving_boundary, coupling.photoelastic)
            for number, resonance in enumerate(solid.eigen_solve.resonances, start=1)]
    prescribed = coupling.prescribed_normal_displacement
    prescribed_rate = None
    if prescribed is not None:
        side0, side1 = sampler.solid_fractions()
        prescribed_rate = overlap.rate(overlap.moving_boundary(float(prescribed) * (side0 - side1)))
    return {'kind': 'table', 'columns': list(OPTOMECHANICS_TABLE_COLUMNS), 'units': list(OPTOMECHANICS_TABLE_UNITS),
            'rows': rows, 'optical_physics': coupling.optical_physics, 'optical_mode': coupling.optical_mode,
            'optical_frequency': overlap.omega / _TWO_PI, 'optical_wavelength': samples.wavelength,
            'optical_azimuthal_order': samples.azimuthal_order, 'optical_q_factor': samples.q_factor,
            'n_eff': None, 'n_group': None, 'length': None, 'mirror_factor': 1.0, 'scattering': 'forward',
            'axial_wavenumber': None, 'normalisation': 'max' if reference is None else 'point',
            'reference_node': reference, 'solid_mesh': kind,
            'mechanical_azimuthal_order': int(data.get('azimuthal_order') or 0),
            'prescribed_normal_displacement': prescribed, 'prescribed_g_om': prescribed_rate}
