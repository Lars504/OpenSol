"""One-sided focused-beam incident fields of the 2D FDFD (em_fdfd_ez / _hz): a Gaussian angular spectrum of discrete
plane waves of the uniform line medium focused at a point, its values on the TFSF rows of a line and its exact discrete
incident power, the line placement, the continuum references, and the DriveFields record of a solved optical field;
host numpy."""
# Line frame: s along the line (its `axis`), t along its normal (the beam axis); the beam travels toward +t
# ('positive', sigma = +1) or -t ('negative', sigma = -1). exp(+j omega t) (harmonic.py header).
#
# Discrete waves. The FDFD rows (PlaneWavePhasor comment) are the face-weighted second differences plus the cell term
# c: w_t D_t^2 u + w_s D_s^2 u + c u = 0 (ez: w_t = w_s = 1 / mu, c = k0^2 eps; hz: w_t = 1 / eps of the faces the t
# difference crosses, w_s = 1 / eps of the s faces, c = k0^2 mu). On uniform cells (h_s along, h_t across) the plane
# wave exp(-j (k s + sigma kappa t)) solves them exactly when
#     w_t (2 / h_t)^2 sin^2(kappa h_t / 2) + w_s (2 / h_s)^2 sin^2(k h_s / 2) = c,
# kappa = (2 / h_t) asin(h_t sqrt(q) / 2), q = (c - w_s (2 / h_s)^2 sin^2(k h_s / 2)) / w_t (principal root: Re >= 0,
# Im <= 0, so a lossy wave decays along its direction). Waves with Re q <= 0 (evanescent) are dropped: back-propagated
# from the focus to the line they would grow.
#
# Beam. u_inc(s, t) = sum_k A(k) exp(-j (k (s - s_f) + sigma kappa(k) (t - t_f))), A = exp(-(k w0 / 2)^2) (a Gaussian
# focus of 1/e field radius w0, all waves in phase at the focus (s_f, t_f), so |u| peaks there in a uniform medium),
# |k| <= k0 NA (NA None: no cap), A >= SPECTRUM_FLOOR, |k| < pi / h_s. k runs over the grid 2 pi m / (P n h_s) of a
# period P = SPECTRAL_PERIOD_LINES times the line length (n span cells), so the periodic images of the beam lie at
# least (P - 1) line lengths from the line and the focus may sit anywhere on it; nothing depends on the focus but the
# phases, so a moved focus is the same set of waves (FocusedBeamPhasor.moved: no snapping to cells).
#
# Power. The TFSF line (fdfd_sources.FocusedBeamPhasor) injects u_inc on its two rows (line cell c = the first
# total-field row, upstream cell c' = c - sigma). Its power is the discrete Yee flux of u_inc through the interface face
# between them, the expression HarmonicFluxMonitor evaluates there: with u_l / u_h the lower / higher row and m the face
# material (ez mu0 mu, hz eps0 eps of the interface face),
#     S_t = -1/2 Re(conj(u_l) (u_h - u_l) / (h_t j omega m))   [W/m^2, toward +t],
# summed over the span times h_s; a lossless uniform medium conserves it face by face, so a downstream flux monitor
# reads it exactly (minus what the structure or the PML send back). The amplitude is scaled to `power` in the unfolded
# structure: a line ending on a mirror plane through the focus carries power / mirror_factor.
import math
from dataclasses import dataclass

import numpy as np

from ...core.constants import C0, EPS0, MU0
from ...core.mesh import has_uniform_spacing
from ...core.mirror import MIRROR_SIGNS, plane_wave_mirror_issue
from .placement import port_cells

DIRECTIONS = ('positive', 'negative')
SPECTRAL_PERIOD_LINES = 4.0   # period of the transverse wavenumber grid in line lengths (images >= 3 lengths away)
SPECTRUM_FLOOR = 1e-16        # relative angular amplitude below which a wave is dropped (Gaussian tail)
NORMAL_RANGE = 2              # cells on either side of the line cell that must be uniform, in the window and homogeneous
MIN_SPAN_CELLS = 3            # cells along the line (the medium check reads the faces between them)
CONTINUUM_POINTS = 2049       # transverse wavenumbers of the continuum references (trapezoid over the spectrum)
FOCUS_TOLERANCE = 1e-6        # x the along spacing: largest distance of a focus "on" a mirror plane


@dataclass(frozen=True, eq=False)
class FocusedBeamLine:
    """Cells of a focused-beam line on the full Yee grid (axis 'x' = along x at row line_cell of y): the two TFSF rows
    (line_cell = first total-field row, upstream_cell = its scattered-field neighbour), the uniform spacings, the
    drawing-frame coordinates (window lower edges 0) of the span centres and the two rows, the flat i * ny + j indices of
    both rows over the span, and the even mirror planes the span ends on (focus there; mirror_factor = 2 per plane)."""
    axis: str                  # along the line, 'x' | 'y'
    normal: str                # the beam axis, the other one
    direction: str             # 'positive' (toward +normal) | 'negative'
    line_cell: int
    upstream_cell: int
    span: tuple                # half-open cells along the line
    along_centres: np.ndarray  # (n,) float64 [m]
    along_spacing: float       # [m]
    normal_spacing: float      # [m]
    line_coordinate: float     # [m] normal centre of the line row
    upstream_coordinate: float
    line_index: np.ndarray     # (n,) int64 flat indices of the line row
    upstream_index: np.ndarray
    shape: tuple               # (nx, ny)
    mirror_planes: tuple       # along coordinates [m] of the even mirror planes the span reaches
    mirror_factor: float       # unfolded / drawn power
    label: str

    @property
    def sign(self) -> float:
        """+1 toward +normal ('positive'), -1 toward -normal."""
        return 1.0 if self.direction == 'positive' else -1.0

    def along_focus(self, focus) -> float:
        """Along-line coordinate [m] of a drawing-frame focus (x, y)."""
        return float(focus[0 if self.axis == 'x' else 1])

    def normal_focus(self, focus) -> float:
        """Normal coordinate [m] of a drawing-frame focus (x, y)."""
        return float(focus[1 if self.axis == 'x' else 0])

    def rows_index(self, normal_cells):
        """Flat indices (len(normal_cells), n) of the span cells in the given normal cells."""
        return flat_rows(self.shape, self.axis, self.span, normal_cells)


def flat_rows(shape, axis, span, normal_cells):
    """int64 flat i * ny + j indices (len(normal_cells), n) of the span cells [span) along a line (axis 'x' = along x)
    in the given normal cells of an (nx, ny) grid."""
    ny = int(shape[1])
    along = np.arange(*span, dtype=np.int64)
    rows = np.asarray(normal_cells, dtype=np.int64)
    if axis == 'x':
        return along[None, :] * ny + rows[:, None]
    return rows[:, None] * ny + along[None, :]


def focused_beam_line(domain, boundary_config, polarization, axis, position, span, direction, label):
    """FocusedBeamLine of a line (axis 'x' = along x at y = position, span = (start, end) along it [m]); ValueError for a
    periodic axis, non-uniform cells along the span or across the line (cells line_cell +- NORMAL_RANGE, which must lie
    in the physical window), a mirror face normal to the beam axis, and a reached mirror face of the wrong parity."""
    if direction not in DIRECTIONS:
        raise ValueError(f"{label} direction must be one of {DIRECTIONS}, got {direction!r}")
    if polarization not in ('ez', 'hz'):
        raise ValueError(f"{label} polarization must be 'ez' or 'hz', got {polarization!r}")
    line_cell, span_cells, _ = port_cells(domain, axis, position, span, label)
    normal = 'y' if axis == 'x' else 'x'
    for name in (axis, normal):
        if getattr(boundary_config, f'periodic_{name}'):
            raise ValueError(f"{label}: the {name} axis is periodic, which a focused beam does not support (its "
                             'Gaussian spectrum is not periodic); use cpml or wall faces on that axis')
    if span_cells[1] - span_cells[0] < MIN_SPAN_CELLS:
        raise ValueError(f"{label}: the line covers {span_cells[1] - span_cells[0]} cell(s) along {axis}; a beam needs "
                         f'at least {MIN_SPAN_CELLS} (and should span it to 1e-6 of its peak)')
    along_axis, normal_axis = 'xy'.index(axis), 'xy'.index(normal)
    widths, centres, edges = domain.cell_widths(), domain.cell_centers(), domain.cell_edges()
    window = domain.physical_window()
    along_widths = np.asarray(widths[along_axis][slice(*span_cells)], dtype=np.float64)
    if not has_uniform_spacing(along_widths):
        raise ValueError(f"{label}: the cells along the line are not uniform ({along_widths.min():.4g} .. "
                         f"{along_widths.max():.4g} m): the beam is a sum of plane waves of uniform cells; keep the "
                         'line in a uniform part of the mesh')
    rows = np.arange(line_cell - NORMAL_RANGE, line_cell + NORMAL_RANGE + 1)
    normal_window = window[normal_axis]
    if rows[0] < normal_window.start or rows[-1] >= normal_window.stop:
        raise ValueError(f"{label}: needs {NORMAL_RANGE} cells between the line and the window edges along {normal} "
                         f'(line cell {line_cell}, window cells [{normal_window.start}, {normal_window.stop})); move '
                         'the line inside')
    normal_widths = np.asarray(widths[normal_axis][rows], dtype=np.float64)
    if not has_uniform_spacing(normal_widths):
        raise ValueError(f"{label}: the cells across the line (cells {rows[0]} .. {rows[-1]} along {normal}) are not "
                         f'uniform ({normal_widths.min():.4g} .. {normal_widths.max():.4g} m): the one-sided launch '
                         'leaks upstream on graded rows; keep the line in a uniform part of the mesh')
    mirrors = tuple(getattr(boundary_config, 'ghost_faces', ()))
    polarization_axis = 'z' if polarization == 'ez' else axis   # the incident E: E_z, or in-plane across the beam
    for face in mirrors:
        if face[0] == normal:
            raise ValueError(f"{label}: {face} is a mirror face normal to the beam axis {normal}: its image beam "
                             'would travel the other way; use cpml or a wall face there')
    along_window = window[along_axis]
    planes = []
    for side, reached, index in (('lo', span_cells[0] <= along_window.start, along_window.start),
                                 ('hi', span_cells[1] >= along_window.stop, along_window.stop)):
        face = f'{axis}_{side}'
        if not reached or face not in mirrors:
            continue
        sign = MIRROR_SIGNS[getattr(boundary_config, face).value]
        if plane_wave_mirror_issue(face, sign, normal, polarization_axis) is not None:
            wanted = ('symmetric' if plane_wave_mirror_issue(face, 1, normal, polarization_axis) is None
                      else 'antisymmetric')
            raise ValueError(f"{label}: its line ends on the {face} mirror plane, about which the beam (E along "
                             f"{polarization_axis}) is even, so {face} must be {wanted}; else end the line inside")
        planes.append(float(edges[along_axis][index]))
    upstream_cell = line_cell - 1 if direction == 'positive' else line_cell + 1
    shape = (int(domain.grid.nx), int(domain.grid.ny))
    along = slice(*span_cells)
    indices = flat_rows(shape, axis, span_cells, (line_cell, upstream_cell))
    return FocusedBeamLine(
        axis=axis, normal=normal, direction=direction, line_cell=int(line_cell), upstream_cell=int(upstream_cell),
        span=(int(span_cells[0]), int(span_cells[1])),
        along_centres=np.array(centres[along_axis][along], dtype=np.float64),
        along_spacing=float(along_widths[0]), normal_spacing=float(normal_widths[NORMAL_RANGE]),
        line_coordinate=float(centres[normal_axis][line_cell]),
        upstream_coordinate=float(centres[normal_axis][upstream_cell]),
        line_index=indices[0].copy(), upstream_index=indices[1].copy(),
        shape=shape, mirror_planes=tuple(planes), mirror_factor=float(2 ** len(planes)), label=label)


@dataclass(frozen=True)
class LineMedium:
    """Background of a focused-beam line at one omega [rad/s]: the face weights of the normal (w_t) and along (w_s)
    differences and the cell term c of the FDFD rows, and the face material m of the interface flux (ez mu0 mu, hz eps0
    eps of the interface face), all complex."""
    omega: float
    normal_weight: complex
    along_weight: complex
    cell_term: complex
    flux_material: complex

    @property
    def index(self) -> complex:
        """Complex refractive index along the beam axis, sqrt(c / (k0^2 w_t)) (sqrt(eps mu) when isotropic)."""
        k0_squared = (self.omega / float(C0)) ** 2
        return complex(np.sqrt(self.cell_term / (k0_squared * self.normal_weight)))


def line_medium(polarization, omega, normal_weight, along_weight, cell_material) -> LineMedium:
    """LineMedium from the uniform values of the assembled operator: its face weights on the faces normal to the beam
    axis and along the line (ez 1 / mu, hz 1 / eps of the face) and the cell material of its cell term (ez eps, hz
    mu); the interface flux material is mu0 / w_t (ez: the H face's mu) or eps0 / w_t (hz: the E face's eps)."""
    omega = float(omega)
    normal_weight = complex(normal_weight)
    vacuum = float(MU0) if polarization == 'ez' else float(EPS0)
    return LineMedium(omega, normal_weight, complex(along_weight), (omega / float(C0)) ** 2 * complex(cell_material),
                      vacuum / normal_weight)


@dataclass(frozen=True, eq=False)
class BeamWaves:
    """Discrete plane waves of one beam spectrum: transverse wavenumbers k [rad/m], real amplitudes A, complex normal
    wavenumbers kappa [rad/m] (all (m,))."""
    wavenumbers: np.ndarray
    amplitudes: np.ndarray
    normal_wavenumbers: np.ndarray


def wave_count(n_along, along_spacing, waist):
    """Upper bound of the waves of a beam on n_along cells of along_spacing [m] with waist [m] (no NA or medium cut):
    the Gaussian and Nyquist caps on the SPECTRAL_PERIOD_LINES grid (memory estimates)."""
    step = 2.0 * math.pi / (SPECTRAL_PERIOD_LINES * n_along * along_spacing)
    cap = min(2.0 / waist * math.sqrt(math.log(1.0 / SPECTRUM_FLOOR)), math.pi / along_spacing)
    return 2 * int(cap / step) + 1


def discrete_waves(n_along, along_spacing, normal_spacing, medium, waist, numerical_aperture, label='Focused beam'):
    """BeamWaves of the discrete spectrum (module comment); ValueError for an NA above the line index, a mesh of pi or
    fewer cells per wavelength along the beam axis, or no propagating wave."""
    step = 2.0 * math.pi / (SPECTRAL_PERIOD_LINES * n_along * along_spacing)
    cap = min(2.0 / waist * math.sqrt(math.log(1.0 / SPECTRUM_FLOOR)), math.pi / along_spacing)
    k0 = medium.omega / float(C0)
    if numerical_aperture is not None:
        index = medium.index.real
        if numerical_aperture > index:
            raise ValueError(f'{label}: numerical_aperture {numerical_aperture:g} exceeds the index {index:.6g} of the '
                             'line medium (NA = n sin(theta) <= n); lower it')
        cap = min(cap, k0 * numerical_aperture)
    count = int(cap / step)
    wavenumbers = step * np.arange(-count, count + 1, dtype=np.float64)
    wavenumbers = wavenumbers[np.abs(wavenumbers) <= cap]
    amplitudes = np.exp(-(0.5 * waist * wavenumbers) ** 2)
    along_term = (2.0 / along_spacing * np.sin(0.5 * along_spacing * wavenumbers)) ** 2
    normal_squared = (medium.cell_term - medium.along_weight * along_term) / medium.normal_weight
    axial = complex(medium.cell_term / medium.normal_weight)   # q of the k = 0 wave
    if 0.5 * normal_spacing * np.sqrt(axial).real >= 1.0:
        raise ValueError(f'{label}: {math.pi / (0.5 * normal_spacing * np.sqrt(axial).real):.3g} cells per wavelength '
                         f'along the beam axis (cell {normal_spacing:.4g} m) is at most pi, where the FDFD grid carries '
                         'no propagating wave; refine the mesh (>= 10 cells per wavelength in the medium)')
    keep = (normal_squared.real > 0.0) & (amplitudes >= SPECTRUM_FLOOR)
    if not np.any(keep):
        raise ValueError(f'{label}: no propagating plane wave in the beam spectrum (waist {waist:.4g} m); check the '
                         'waist, numerical_aperture and the line medium')
    wavenumbers, amplitudes, normal_squared = wavenumbers[keep], amplitudes[keep], normal_squared[keep]
    normal_wavenumbers = 2.0 / normal_spacing * np.arcsin(0.5 * normal_spacing * np.sqrt(normal_squared))
    # principal branches: sqrt Re >= 0; arcsin of Re in (0, 1) keeps Re > 0 and Im of the sign of Im sqrt (<= 0 lossy)
    return BeamWaves(wavenumbers, amplitudes, normal_wavenumbers.astype(np.complex128))


def beam_rows(waves, along_centres, row_coordinates, focus_along, focus_normal, sign):
    """complex128 (len(row_coordinates), n) of sum_k A e^{-j (k (s - s_f) + sign kappa (t - t_f))} on the along centres
    s [m] of each row coordinate t [m] (unit spectrum, no power scaling)."""
    along_phases = np.exp(-1j * np.outer(np.asarray(along_centres, dtype=np.float64) - focus_along,
                                         waves.wavenumbers))   # (n, m)
    offsets = np.asarray(row_coordinates, dtype=np.float64) - focus_normal
    row_weights = waves.amplitudes[None, :] * np.exp(-1j * sign * np.outer(offsets, waves.normal_wavenumbers))
    return row_weights @ along_phases.T   # (rows, n)


def interface_power(line, medium, line_values, upstream_values):
    """Unfolded power [W/m] along the beam direction of incident values on the line / upstream rows through their
    interface face (module comment; x line.mirror_factor)."""
    lower, upper = ((upstream_values, line_values) if line.direction == 'positive'
                    else (line_values, upstream_values))
    density = -0.5 * (np.conj(lower) * (upper - lower) / (line.normal_spacing * 1j * medium.omega
                                                          * medium.flux_material)).real
    return line.sign * line.mirror_factor * float(density.sum()) * line.along_spacing


# ---- continuum references (validation, self-tests) ----
def continuum_waves(omega, index, waist, numerical_aperture, n_points=CONTINUUM_POINTS):
    """(k, weights dk / (2 pi), A, kappa) of the continuum spectrum in a medium of complex index (kappa = sqrt((k0
    n)^2 - k^2), |k| below the propagating, Gaussian and NA caps), trapezoid weights over [-cap, cap]."""
    k0 = float(omega) / float(C0)
    medium_wavenumber = k0 * complex(index)
    cap = min(2.0 / waist * math.sqrt(math.log(1.0 / SPECTRUM_FLOOR)), medium_wavenumber.real)
    if numerical_aperture is not None:
        cap = min(cap, k0 * numerical_aperture)
    wavenumbers = np.linspace(-cap, cap, n_points)
    weights = np.full(n_points, (wavenumbers[1] - wavenumbers[0]) / (2.0 * math.pi))
    weights[[0, -1]] *= 0.5
    amplitudes = np.exp(-(0.5 * waist * wavenumbers) ** 2)
    normal = np.sqrt(medium_wavenumber ** 2 - wavenumbers.astype(np.complex128) ** 2)
    return wavenumbers, weights, amplitudes, normal


def continuum_field(spectrum, along_offsets, normal_offsets, sign=1.0):
    """complex (1/2 pi) int A e^{-j (k s + sign kappa t)} dk at the points (s, t) relative to the focus [m] (equal-
    shape arrays), unit spectrum."""
    wavenumbers, weights, amplitudes, normal = spectrum
    s = np.asarray(along_offsets, dtype=np.float64).ravel()
    t = np.asarray(normal_offsets, dtype=np.float64).ravel()
    phases = np.exp(-1j * (np.outer(s, wavenumbers) + sign * np.outer(t, normal)))
    return (phases @ (weights * amplitudes)).reshape(np.shape(along_offsets))


def continuum_unit_power(spectrum, omega, flux_material):
    """Power [W/m] (1/2 pi) int |A|^2 Re(kappa / (omega m)) / 2 dk of the unit continuum spectrum through a plane
    normal to the beam (m = mu0 mu for E_z, eps0 eps for H_z)."""
    wavenumbers, weights, amplitudes, normal = spectrum
    return float(np.sum(weights * amplitudes ** 2 * 0.5 * (normal / (float(omega) * complex(flux_material))).real))


def continuum_focus_amplitude(omega, index, waist, numerical_aperture, flux_material, power):
    """|u| at the focus [V/m for E_z, A/m for H_z] of a continuum beam carrying `power` [W/m] (the reference of the
    discrete one)."""
    spectrum = continuum_waves(omega, index, waist, numerical_aperture)
    unit_power = continuum_unit_power(spectrum, omega, flux_material)
    return math.sqrt(power / unit_power) * abs(complex(continuum_field(spectrum, np.zeros(1), np.zeros(1))[0]))


def continuum_end_levels(omega, index, waist, numerical_aperture, along_ends, line_normal, focus_along, focus_normal,
                         sign, n_samples=401):
    """(level at the lo end, level at the hi end) = |u| / max |u| on the line of the continuum beam (validation: the line
    must span the beam), the line from along_ends[0] to along_ends[1] at normal coordinate line_normal [m]."""
    spectrum = continuum_waves(omega, index, waist, numerical_aperture)
    along = np.concatenate((np.linspace(along_ends[0], along_ends[1], n_samples), np.asarray(along_ends, float)))
    values = np.abs(continuum_field(spectrum, along - focus_along, np.full(along.shape, line_normal - focus_normal),
                                    sign))
    peak = float(values.max())
    if not peak > 0.0:
        return math.nan, math.nan
    return float(values[-2]) / peak, float(values[-1]) / peak


# ---- drive fields of a solved optical physics (brillouin_scattering) ----
@dataclass(frozen=True, eq=False)
class DriveFields:
    """Host window samples of one solved optical field, read by a solid's drive (brillouin_scattering): omega [rad/s];
    x / y the window cell centres [m] in the drawing frame (window lower edge 0; BOR: r / z); fields component ->
    complex128 phasors [V/m] at that component's own Yee points, grids component -> (x coordinates, y coordinates) [m]
    of them in the same frame (along each axis the window cell centres (mx / my) or the window cell edges (mx + 1 / my +
    1, both window edges included): 2D ez {'ez'} at the centres; hz {'ex'} on the y faces (x centres, y edges), {'ey'}
    on the x faces (x edges, y centres), never averaged along the beam; BOR its own components, see fdfd_bor); eps
    component -> complex128 (mx, my) relative cell permittivity that component sees (the staircase: no subpixel deltas;
    loss and poles included); pec bool (mx, my) PEC cells or None; polarization of the physics."""
    omega: float
    x: np.ndarray
    y: np.ndarray
    fields: dict
    grids: dict
    eps: dict
    pec: np.ndarray | None
    polarization: str
