"""EME mode sets (host numpy): slab line modes (2D) or cross-section modes (3D) rescaled to the bilinear norm
<e_n, h_n> = 1, PML modes ranked after the others, the field of a section from its amplitudes, the 2D bend line map
and the Gauss cell means of the absorbed power."""
# 2D: fields of mode n toward +axis: u (out-of-plane: ez E_z, hz H_z) and its transverse partner t = sign_n g u (ez H,
# hz E along the line; slab_modes header), both tangential to an interface plane. Matching vectors: e = u (continuum)
# or e = cos(k D / 2) u (discrete propagation: the mean of the two cells beside the interface face), h = g u; the
# unconjugated product <e_a, h_b> = sum e_a h_b w s over the line cells (w s = stretched widths) vanishes between
# distinct modes of one line (slab_modes' bilinear orthogonality), so the scale 1 / sqrt(<e_n, h_n>) makes the set
# orthonormal. 3D (propagation z): e = [E_x; -E_y], h = [H_y; H_x] on their shared Yee points (E_x / H_y hi-x faces,
# E_y / H_x hi-y faces), so sum e h w = sum (E x H).z dA with the stretched dual areas w = A s_x s_y (the summation by
# parts of the stretched Yee differences: discrete Lorentz reciprocity makes distinct modes orthogonal); degenerate
# modes, which the eigensolver returns as arbitrary combinations, are recombined bilinear-orthogonal. power = the flux
# [W/m in 2D, W in 3D] one unit of amplitude carries (1/2 for propagating lossless modes). Ranking: the solver's
# (descending Re k^2), modes with more than half their weight on complex-stretched points (PML modes) moved last, so
# port mode k is the k-th physical mode as in the mode ports.
import math
from dataclasses import dataclass

import numpy as np

from ...core.constants import C0, EPS0, MU0

_GAUSS_MEAN_WEIGHTS = (5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0)   # 3-point Gauss-Legendre, normalised to a cell mean
_PML_MODE_FRACTION = 0.5          # share of the mode weight on complex-stretched points above which a mode is a PML mode
_DEGENERATE_GAP = 1e-8            # |k_a^2 - k_b^2| / max |k^2| below which two 3D modes count as degenerate
_ORTHOGONAL_TOLERANCE = 1e-6      # |<e_a, h_b>| / sqrt(|<e_a, h_a> <e_b, h_b>|) above which they are recombined
INPUT_POWER_FLOOR = 1e-12         # flux of unit amplitude below which an input mode carries no power (evanescent)


@dataclass(frozen=True, eq=False)
class ModeSet:
    """Bilinear-normalised modes of one section, physical modes first, then PML modes."""
    k: np.ndarray           # (N,) complex128 propagation constant toward +axis [1/m] (discrete k with a spacing)
    n_eff: np.ndarray       # (N,) complex128
    profiles: np.ndarray    # 2D (N, m) u of unit amplitude; 3D (N, 3, P) E_x, E_y, E_z at the cell centres (P = nx ny)
    e: np.ndarray           # (N, M) complex128 matching vectors (2D: of u; 3D: [E_x; -E_y])
    h: np.ndarray           # (N, M) complex128 matching vectors (2D: g u over sign_n; 3D: [H_y; H_x])
    weights: np.ndarray     # (M,) complex128 stretched cell widths w s (2D) / dual areas A s_x s_y (3D)
    power: np.ndarray       # (N,) float64 flux of unit amplitude [W/m in 2D, W in 3D]
    pml: np.ndarray         # (N,) bool PML mode (radiation carried in the transverse PML)
    discrete_spacing: float | None

    @property
    def n_modes(self) -> int:
        return int(self.k.size)

    @property
    def nbytes(self) -> int:
        return int(sum(array.nbytes for array in (self.k, self.n_eff, self.profiles, self.e, self.h, self.weights,
                                                  self.power, self.pml)))


def _physical_first(pml_flags):
    """Stable order putting the non-PML modes first."""
    return np.argsort(np.asarray(pml_flags, dtype=np.int8), kind='stable')


def mode_set(line, modes, discrete_spacing=None):
    """ModeSet of SlabModes solved on line (with normal_spacing = discrete_spacing for the discrete variant);
    ValueError without modes or for a mode with a vanishing bilinear self-product."""
    if not modes:
        raise ValueError('the section line has no mode (every cell PEC?)')
    profiles = np.array([np.asarray(mode.profile, dtype=np.complex128) for mode in modes])
    factors = np.array([np.asarray(mode.transverse_factor, dtype=np.complex128) for mode in modes])
    k = np.array([complex(mode.k) for mode in modes], dtype=np.complex128)
    cell_stretch = np.asarray(line.cell_stretch, dtype=np.complex128)
    weights = np.asarray(line.cell_widths, dtype=np.float64) * cell_stretch
    e = profiles if discrete_spacing is None else np.cos(0.5 * k * float(discrete_spacing))[:, None] * profiles
    h = factors * profiles
    norms = np.sum(e * h * weights, axis=1)
    if np.any(norms == 0):
        raise ValueError('a section mode has a vanishing bilinear self-product <e, h> (a mode exactly at cutoff)')
    scale = 1.0 / np.sqrt(norms)
    power = np.array([float(mode.power) for mode in modes]) * np.abs(scale) ** 2
    magnitude = np.abs(profiles) ** 2 * np.abs(weights)[None, :]
    stretched = cell_stretch.imag != 0.0
    total = np.sum(magnitude, axis=1)
    pml = np.sum(magnitude[:, stretched], axis=1) > _PML_MODE_FRACTION * np.where(total > 0.0, total, np.inf)
    order = _physical_first(pml)
    return ModeSet(k=k[order], n_eff=np.array([complex(mode.n_eff) for mode in modes], dtype=np.complex128)[order],
                   profiles=(profiles * scale[:, None])[order], e=(e * scale[:, None])[order],
                   h=(h * scale[:, None])[order], weights=weights, power=power[order], pml=pml[order],
                   discrete_spacing=None if discrete_spacing is None else float(discrete_spacing))


def stretched_areas(cell_widths, boundary_config, pml_dx, omega):
    """((nx, ny) s_x s_y at the E_x / H_y points (hi-x faces), at the E_y / H_x points (hi-y faces), at the E_z points
    (cell centres)): the complex CPML stretch the cross-section operator of these arguments uses (1 outside the PML)."""
    from ..em.boundaries import cpml_axis_profiles
    from ..em.operators import axis_stretch
    (x_centre, x_face), (y_centre, y_face) = (
        axis_stretch(cpml_axis_profiles(boundary_config, np.asarray(widths).size, axis, pml_dx), omega)
        for axis, widths in zip(('x', 'y'), cell_widths))
    return np.outer(x_face, y_centre), np.outer(x_centre, y_face), np.outer(x_centre, y_centre)


def _clusters(links):
    """Connected components (lists of indices, ascending) of a symmetric bool adjacency matrix."""
    unassigned, clusters = set(range(links.shape[0])), []
    while unassigned:
        stack, cluster = [min(unassigned)], set()
        while stack:
            index = stack.pop()
            if index in cluster:
                continue
            cluster.add(index)
            stack.extend(int(other) for other in np.flatnonzero(links[index]) if other not in cluster)
        unassigned -= cluster
        clusters.append(sorted(cluster))
    return clusters


def _orthogonalise_degenerate(k, e, h, profiles, weights):
    """(k, e, h, profiles) with every cluster of degenerate, non-orthogonal modes recombined so that their bilinear
    Gram matrix is diagonal (slab_modes' pivoted bilinear Gram-Schmidt); each new mode takes the cluster's mean k."""
    from ..em.slab_modes import _bilinear_orthogonal_combinations
    gram = (e * weights[None, :]) @ h.T
    self_products = np.abs(np.diag(gram))
    relative = np.abs(gram) / np.sqrt(np.outer(self_products, self_products) + 1e-300)
    squared = k ** 2
    tie = np.abs(squared[:, None] - squared[None, :]) <= _DEGENERATE_GAP * float(np.max(np.abs(squared)))
    links = tie & (relative > _ORTHOGONAL_TOLERANCE)
    np.fill_diagonal(links, False)
    k, e, h, profiles = k.copy(), e.copy(), h.copy(), profiles.copy()
    for cluster in _clusters(links):
        if len(cluster) < 2:
            continue
        block = gram[np.ix_(cluster, cluster)]
        combination = _bilinear_orthogonal_combinations(0.5 * (block + block.T))
        e[cluster] = combination.T @ e[cluster]
        h[cluster] = combination.T @ h[cluster]
        profiles[cluster] = np.tensordot(combination.T, profiles[cluster], axes=1)
        k[cluster] = np.mean(k[cluster])
    return k, e, h, profiles


def cross_section_mode_set(modes, area_stretch, periodic):
    """ModeSet of CrossSectionModes of one cross-section (cross_section_modes, PML modes kept), area_stretch =
    stretched_areas of its operator, periodic = (periodic_x, periodic_y); ValueError without modes or for a mode with a
    vanishing bilinear self-product."""
    from ..em.cross_section_modes import centre_fields
    if not modes:
        raise ValueError('the cross-section has no mode')
    first = modes[0]
    real_area = np.concatenate((first.x_face_area.ravel(), first.y_face_area.ravel()))
    weights = real_area * np.concatenate((area_stretch[0].ravel(), area_stretch[1].ravel()))
    e = np.array([np.concatenate((mode.ex.ravel(), -mode.ey.ravel())) for mode in modes])
    h = np.array([np.concatenate((mode.hy.ravel(), mode.hx.ravel())) for mode in modes])
    profiles = []
    for mode in modes:
        centres = centre_fields(mode, *periodic)
        profiles.append([centres[name].ravel() for name in ('ex', 'ey', 'ez')])
    profiles = np.array(profiles, dtype=np.complex128)
    k = np.array([complex(mode.beta) for mode in modes], dtype=np.complex128)
    k, e, h, profiles = _orthogonalise_degenerate(k, e, h, profiles, weights)
    norms = np.sum(e * h * weights[None, :], axis=1)
    if np.any(norms == 0):
        raise ValueError('a cross-section mode has a vanishing bilinear self-product <e, h> (a mode exactly at cutoff)')
    scale = 1.0 / np.sqrt(norms)
    e, h, profiles = e * scale[:, None], h * scale[:, None], profiles * scale[:, None, None]
    power = 0.5 * float(first.mirror_factor) * np.real(np.sum(e * np.conj(h) * real_area[None, :], axis=1))
    pml = np.array([mode.pml_fraction > _PML_MODE_FRACTION for mode in modes])
    order = _physical_first(pml)
    n_eff = k * C0 / float(first.omega)
    return ModeSet(k=k[order], n_eff=n_eff[order], profiles=profiles[order], e=e[order], h=h[order], weights=weights,
                   power=power[order], pml=pml[order], discrete_spacing=None)


def longitudinal_flux(line, values):
    """(n, m + 1) complex128 F_k = face_weight face_inverse_spacing / face_stretch (u_k - u_(k-1)) on the line faces of
    rows u (n, m), the end faces by the line's end conditions; |F| / (omega X0) = the in-plane field component normal to
    the line (along the propagation axis), X0 = MU0 (ez) / EPS0 (hz)."""
    u = np.asarray(values, dtype=np.complex128)
    conductance = (np.asarray(line.face_weight, dtype=np.complex128)
                   * np.asarray(line.face_inverse_spacing, dtype=np.float64)
                   / np.asarray(line.face_stretch, dtype=np.complex128))
    n_cells = u.shape[1]
    flux = np.zeros((u.shape[0], n_cells + 1), dtype=np.complex128)
    flux[:, 1:n_cells] = conductance[1:n_cells] * (u[:, 1:] - u[:, :-1])
    if line.lo_end == 'periodic':
        flux[:, 0] = conductance[0] * (u[:, 0] - u[:, -1])
        flux[:, n_cells] = flux[:, 0]
        return flux
    end_factor = {'dirichlet': 1.0, 'neumann': 0.0, 'odd': 2.0}   # ghost 0, zero flux, ghost -u
    flux[:, 0] = end_factor[line.lo_end] * conductance[0] * u[:, 0]
    flux[:, n_cells] = -end_factor[line.hi_end] * conductance[n_cells] * u[:, -1]
    return flux


def _propagation_terms(modes, forward, backward, centres, length):
    """((n, N) a e^{-j k s}, (n, N) b e^{-j k (L - s)}) at distances centres [m] into a section of `length`."""
    centres = np.asarray(centres, dtype=np.float64)
    forward_terms = np.exp(-1j * np.outer(centres, modes.k)) * np.asarray(forward)[None, :]
    backward_terms = np.exp(-1j * np.outer(float(length) - centres, modes.k)) * np.asarray(backward)[None, :]
    return forward_terms, backward_terms


def gauss_positions(widths):
    """((3 n,) distances [m] into a run of cells of `widths` (n,): the 3-point Gauss-Legendre points of every cell, lo
    points first, then the centres, then the hi points), (3,) weights of a cell mean (sum 1))."""
    widths = np.asarray(widths, dtype=np.float64)
    centres = np.cumsum(widths) - 0.5 * widths
    offset = 0.5 * math.sqrt(0.6) * widths
    return np.concatenate((centres - offset, centres, centres + offset)), _GAUSS_MEAN_WEIGHTS


def gauss_mean(values):
    """(n, ...) cell means of (3 n, ...) values at gauss_positions (lo, centre, hi blocks)."""
    n_cells = values.shape[0] // 3
    return sum(weight * values[k * n_cells:(k + 1) * n_cells] for k, weight in enumerate(_GAUSS_MEAN_WEIGHTS))


def section_fields(line, modes, forward, backward, centres, length, omega, face_scale=None):
    """(u, in-plane |field|^2) (n, m) at distances centres [m] into a 2D section of `length` from its forward
    amplitudes (at its lo end) and backward ones (at its hi end): u = sum (a e^{-jks} + b e^{-jk(L - s)}) u_n; the
    in-plane part = |t|^2 of the transverse partner plus the face mean of the component along the propagation axis
    |F| / (face_scale omega X0) (ez: |H|^2, hz: |E|^2; face_scale (m + 1,) = the bend's r / R at the line faces)."""
    forward_terms, backward_terms = _propagation_terms(modes, forward, backward, centres, length)
    u = (forward_terms + backward_terms) @ modes.profiles
    transverse = (forward_terms - backward_terms) @ modes.h
    flux = longitudinal_flux(line, u)
    if face_scale is not None:
        flux = flux / np.asarray(face_scale)[None, :]
    field_constant = MU0 if line.polarization == 'ez' else EPS0
    along_axis = 0.5 * (flux[:, :-1] + flux[:, 1:]) / (float(omega) * field_constant)
    return u, np.abs(transverse) ** 2 + np.abs(along_axis) ** 2


def bent_line(line, radius, centre_positions, face_positions, reference):
    """(SlabLine of the radial line of a bend of `radius` [m] at the line coordinate `reference`, (m + 1,) r / R at
    its faces): cell_weight x r / R, face_weight x r_f / R, index_term x (r / R)^2 with r = R + (position - reference),
    the exact cylindrical operator (r / R^2) d/dr (r du/dr) + k0^2 q (r / R)^2 (ez; hz with its eps weights), so k is
    beta at r = R (propagation along the arc length R theta); ValueError when r <= 0 on the line."""
    from dataclasses import replace
    radius, reference = float(radius), float(reference)
    centre_scale = 1.0 + (np.asarray(centre_positions, dtype=np.float64) - reference) / radius
    face_scale = 1.0 + (np.asarray(face_positions, dtype=np.float64) - reference) / radius
    if np.min(face_scale) <= 0.0:
        raise ValueError(f'bend radius {radius:g} m at reference {reference:g} m: the centre of curvature lies on the '
                         f'section line (r <= 0); use a larger |radius|')
    bent = replace(line, cell_weight=np.asarray(line.cell_weight) * centre_scale,
                   face_weight=np.asarray(line.face_weight) * face_scale,
                   index_term=np.asarray(line.index_term) * centre_scale ** 2)
    return bent, face_scale


def cross_section_fields(modes, forward, backward, centres, length):
    """(n, 3, P) complex128 E_x, E_y, E_z at the cell centres at distances centres [m] into a 3D section (amplitudes as
    section_fields): the transverse E of a backward mode equals the forward one's, its E_z the opposite."""
    forward_terms, backward_terms = _propagation_terms(modes, forward, backward, centres, length)
    n_modes, _, n_points = modes.profiles.shape
    transverse = ((forward_terms + backward_terms) @ modes.profiles[:, :2, :].reshape(n_modes, 2 * n_points))
    longitudinal = (forward_terms - backward_terms) @ modes.profiles[:, 2, :]
    return np.concatenate((transverse.reshape(-1, 2, n_points), longitudinal[:, None, :]), axis=1)


def is_propagating(modes, relative_floor=1e-3):
    """(N,) bool: modes carrying more than relative_floor x the largest power of the set (lossless: the real-k ones)."""
    largest = float(np.max(modes.power)) if modes.n_modes else 0.0
    return modes.power > relative_floor * largest if largest > 0.0 else np.zeros(modes.n_modes, dtype=bool)
