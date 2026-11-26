"""Focused-beam incident field of em_fdfd_bor: the Debye (Richards-Wolf) field of an aplanatic lens focusing an
x-polarised Gaussian pupil to a point on the axis, as the azimuthal order 1 standing wave (E_r, u, E_z) on the BOR Yee
points of two TFSF slices, its power (the discrete interface flux; the continuum pupil power as the reference), the
line placement and the BORFocusedBeam source; host numpy, scipy.special inside functions."""
# Conventions: exp(+j omega t), cylindrical (r, phi, z) right-handed with z the axis (drawing y), the beam toward +z
# ('positive', sigma = +1) or -z ('negative', sigma = -1), focus (r = 0, z_f), medium of index n = sqrt(eps mu) and
# wavenumber k = k0 n at the line (one homogeneous isotropic medium on the line rows; the focal region may hold any
# structure: the TFSF line launches the beam of the unbounded line medium).
#
# Debye field (Novotny & Hecht, Principles of Nano-Optics, eqs. 3.66-3.68, converted from exp(-i omega t) by complex
# conjugation, so e^{+ikz cos} -> e^{-jkz cos} and -2i I01 -> +2j I01). With d = sigma (z - z_f), s_m = sin theta_max =
# NA / Re n, the pupil l(theta) = exp(-(sin theta / (f0 s_m))^2) (f0 = fill_factor) and the aplanatic sqrt(cos theta):
#   I00 = int_0^theta_max l sqrt(cos) sin (1 + cos) J0(k rho sin) e^{-j kappa d} dtheta
#   I01 = int l sqrt(cos) sin^2 J1(k rho sin) e^{-j kappa d} dtheta
#   I02 = int l sqrt(cos) sin (1 - cos) J2(k rho sin) e^{-j kappa d} dtheta
# the Cartesian field of an x-polarised pupil is C (I00 + I02 cos 2 phi, I02 sin 2 phi, 2 j sigma I01 cos phi); in
# cylindrical components it is exactly the m = 1 standing wave E = (A_r cos phi, A_phi sin phi, A_z cos phi) with
#   A_r = C (I00 + I02),   A_phi = C (I02 - I00),   A_z = 2 j sigma C I01
# (each plane wave of the angular spectrum carries e_inf = cos phi_k n_theta - sin phi_k n_phi, transverse to k: so
# E_z = -tan(theta) cos(phi_k) E_x per wave, which the phi_k integral turns into +2 j I01 cos phi; the mirror z -> -z of
# the +z beam is the -z beam, A_z odd). The solve at azimuthal order m = 1 IS this standing wave (body_of_revolution
# header: E_r, E_z ~ cos phi, E_phi ~ u sin phi with u = -j E_phi of the exp(-j phi) solve), so the unknowns take E_r =
# A_r, u = A_phi, E_z = A_z, and brillouin.azimuthal_parts reads (A_r, A_phi = -j E_phi, A_z) back (DriveFields).
# At the focus (rho = 0, d = 0) A_r = C I00 with I00 > 0: phase = the phase of E_x there.
#
# Semi-discrete axial wavenumber. Each angular component varies along z as e^{-j kappa d}; with kappa = k cos theta
# (the continuum Debye field) the Yee z differences see (2 / h) sin(kappa h / 2) instead of k cos theta, an O((kh)^2)
# launch error. The slices use kappa = (2 / h) asin(h k cos theta / 2) (h = the uniform z width of the line rows), for
# which every staggered z first difference of the operator is exactly -j k cos theta: the incident field solves the
# z-discrete, r-continuous equations, so the leakage upstream comes from the radial discretisation of J_n alone (O((k
# s_m h)^2), largest at high NA; bor_ports discrete modes would make it exact: open). At the focus (d = 0) both agree.
#
# Power. The physical (linearly polarised) beam is the standing wave, whose cos^2 / sin^2 rings carry
# core.metric.ring_angle(1) = pi where the travelling exp(-j phi) wave of the solve carries 2 pi: its power is
# ring_angle(1) / (2 pi) = 1/2 of the travelling-wave flux that flux_monitor reports. `power` is that standing-wave
# power, measured as the exact discrete Yee flux of the injected incident field through the interface face
# (bor_ports.slice_interface_power x ring_angle(1) / (2 pi)), so a flux_monitor downstream in the uniform lossless line
# medium reads 2 x power (to the launch error) and the drive of brillouin_scattering (kappa with incident_power, the
# standing-wave amplitudes with ring_angle(m) solid rings) is consistent with it. The continuum pupil power
#   P = (4 pi |C|^2 / (eta k^2)) int_0^theta_max l^2 sin cos dtheta = pi f0^2 s_m^2 (1 - e^{-2 / f0^2}) |C|^2 / (eta k^2)
# (eta = eta0 mu / n; the angular-spectrum flux of the Debye field, = the pupil integral n / (2 eta0) int |E_p|^2 dA of
# an aplanatic lens) differs from it by O((kh)^2) (E at row centres, H on the face: ~ cos(kappa h / 2)).
#
# Quadrature: Gauss-Legendre over theta in [0, theta_max], MIN_THETA_NODES + THETA_NODES_PER_RADIAN x the largest
# integrand phase excursion k (rho_max s_m + |d|_max (1 - cos theta_max)) nodes (ValueError above MAX_THETA_NODES),
# vectorised over the points in blocks of CHUNK_VALUES (point, node) values.
import math
import weakref
from dataclasses import dataclass

import numpy as np

from ...core.constants import C0
from ...core.mesh import has_uniform_spacing
from ...core.metric import ring_angle
from .boundaries import BCType
from .bor_ports import key_positions, line_keys, slice_interface_power, slice_tfsf_rhs
from .fdfd_sources import FocusedBeamSource, uniform_value
from .focused_beam import DIRECTIONS, MIN_SPAN_CELLS, NORMAL_RANGE
from .placement import port_cells

BEAM_ORDER = 1                      # the x-polarised beam on the axis is the m = 1 standing wave
MIN_THETA_NODES = 32
THETA_NODES_PER_RADIAN = 4.0 / math.pi   # Gauss-Legendre nodes per radian of the integrand's phase excursion
MAX_THETA_NODES = 16384
CHUNK_VALUES = 1 << 18              # (point, node) values per vectorised block
QUADRATURE_BYTES_PER_VALUE = 56     # Bessel argument (8, complex 16), J_n (8), phases (16), product (16) per value
QUADRATURE_FLOPS_PER_VALUE = 200    # one or two J_n, the complex phase and the weighted sum per value (rough)
_Z_WALLS = (BCType.PEC, BCType.PMC)  # z faces the beam axis may not end on
_COMPONENT_CODES = {0: 'r', 1: 'phi', 2: 'z'}   # bor_ports.line_keys codes: E_r, u, E_z


# ---- Debye field (module header) ----
def aperture_sine(numerical_aperture, index, label='Focused beam') -> float:
    """sin theta_max = NA / Re n of a numerical aperture in a medium of complex index; ValueError unless 0 < NA < Re n
    (theta_max < 90 deg)."""
    numerical_aperture = float(numerical_aperture)
    index = complex(index).real
    if not (math.isfinite(numerical_aperture) and numerical_aperture > 0.0):
        raise ValueError(f'{label}: numerical_aperture must be finite and > 0, got {numerical_aperture!r}')
    if not numerical_aperture < index:
        raise ValueError(f'{label}: numerical_aperture {numerical_aperture:g} must stay below the index {index:.6g} of '
                         'the line medium (NA = n sin(theta_max), theta_max < 90 deg); lower it')
    return numerical_aperture / index


def theta_node_count(phase_span) -> int:
    """Gauss-Legendre nodes for an integrand whose phase varies by phase_span [rad] over the aperture (module header);
    ValueError above MAX_THETA_NODES (a line far too long for the beam: shorten it)."""
    count = MIN_THETA_NODES + int(math.ceil(THETA_NODES_PER_RADIAN * max(float(phase_span), 0.0)))
    if count > MAX_THETA_NODES:
        raise ValueError(f'focused beam quadrature: {count} angular nodes needed (phase excursion {phase_span:.4g} rad '
                         f'over the line), above {MAX_THETA_NODES}; shorten the line or move the focus closer to it')
    return count


@dataclass(frozen=True, eq=False)
class DebyeRule:
    """Angular quadrature of one beam: sin / cos theta of the nodes, weights (Gauss-Legendre x l(theta) sqrt(cos) sin,
    the common factor of the three integrals), axial wavenumbers kappa(theta) [rad/m] (complex in a lossy medium)."""
    sin: np.ndarray
    cos: np.ndarray
    weights: np.ndarray
    axial: np.ndarray


def pupil_amplitudes(sin_theta, sin_max, fill_factor):
    """Pupil field l(theta) = exp(-(sin theta / (f0 s_m))^2) of a Gaussian of waist f0 x the pupil radius."""
    return np.exp(-(np.asarray(sin_theta) / (float(fill_factor) * float(sin_max))) ** 2)


def debye_rule(wavenumber, sin_max, fill_factor, rho_max, offset_max, z_spacing=None) -> DebyeRule:
    """DebyeRule of a beam of complex medium wavenumber [rad/m] and aperture sine, for points within rho_max [m] of the
    axis and |d| <= offset_max [m] of the focal plane; z_spacing [m]: the semi-discrete kappa of a uniform Yee z row
    width (module header; ValueError at pi or fewer cells per wavelength), None: the continuum k cos theta."""
    fill_factor = float(fill_factor)
    if not (math.isfinite(fill_factor) and fill_factor > 0.0):
        raise ValueError(f'focused beam fill_factor must be finite and > 0, got {fill_factor!r}')
    wavenumber = complex(wavenumber)
    theta_max = math.asin(float(sin_max))
    span = abs(wavenumber.real) * (float(rho_max) * float(sin_max) + float(offset_max) * (1.0 - math.cos(theta_max)))
    nodes, weights = np.polynomial.legendre.leggauss(theta_node_count(span))
    theta = 0.5 * theta_max * (nodes + 1.0)
    sin, cos = np.sin(theta), np.cos(theta)
    weights = 0.5 * theta_max * weights * pupil_amplitudes(sin, sin_max, fill_factor) * np.sqrt(cos) * sin
    if z_spacing is None:
        axial = wavenumber * cos
    else:
        h = float(z_spacing)
        if not 0.5 * h * wavenumber.real < 1.0:
            raise ValueError(f'focused beam: {math.pi / (0.5 * h * wavenumber.real):.3g} cells per wavelength along z '
                             f'(cell {h:.4g} m) is at most pi, where the Yee grid carries no propagating wave; refine '
                             'the mesh (>= 15 cells per wavelength in the medium)')
        axial = 2.0 / h * np.arcsin(0.5 * h * wavenumber * cos.astype(np.complex128))
    if wavenumber.imag == 0.0:
        axial = np.asarray(axial).real.astype(np.complex128)   # lossless: kappa real (arcsin of real args < 1)
    return DebyeRule(sin=sin, cos=cos, weights=weights, axial=np.asarray(axial, dtype=np.complex128))


_RADIAL_FACTORS = {0: lambda rule: 1.0 + rule.cos, 1: lambda rule: rule.sin, 2: lambda rule: 1.0 - rule.cos}


def debye_integrals(rule, wavenumber, rho, offsets, orders) -> dict:
    """{n: complex128 (p,) I0n} of the orders n in (0, 1, 2) at the points rho [m] (>= 0) and signed distances d =
    sigma (z - z_f) [m] (module header), blocks of CHUNK_VALUES values."""
    import scipy.special
    rho = np.asarray(rho, dtype=np.float64).ravel()
    offsets = np.asarray(offsets, dtype=np.float64).ravel()
    if rho.shape != offsets.shape:
        raise ValueError(f'debye_integrals: rho and offsets differ in shape ({rho.shape}, {offsets.shape})')
    wavenumber = complex(wavenumber)
    scale = wavenumber.real if wavenumber.imag == 0.0 else wavenumber
    factors = {n: _RADIAL_FACTORS[n](rule) for n in orders}
    values = {n: np.empty(rho.size, dtype=np.complex128) for n in orders}
    step = max(1, CHUNK_VALUES // rule.sin.size)
    for start in range(0, rho.size, step):
        part = slice(start, start + step)
        argument = np.multiply.outer(scale * rho[part], rule.sin)                  # (p, t): k rho sin theta
        phases = np.exp(-1j * np.multiply.outer(offsets[part], rule.axial))         # e^{-j kappa d}
        phases *= rule.weights
        for n in orders:
            values[n][part] = (scipy.special.jv(n, argument) * phases) @ factors[n]
        del argument, phases
    return values


def beam_amplitudes(rule, wavenumber, rho, z_offsets, sign, component):
    """complex128 (p,) unit (C = 1) standing-wave amplitude of component 'r' (A_r) | 'phi' (A_phi = u) | 'z' (A_z) at
    the points rho [m], z - z_f [m] of a beam toward sign (+1: +z, -1: -z) (module header)."""
    offsets = float(sign) * np.asarray(z_offsets, dtype=np.float64)
    if component == 'z':
        return 2j * float(sign) * debye_integrals(rule, wavenumber, rho, offsets, (1,))[1]
    integrals = debye_integrals(rule, wavenumber, rho, offsets, (0, 2))
    if component == 'r':
        return integrals[0] + integrals[2]
    if component == 'phi':
        return integrals[2] - integrals[0]
    raise ValueError(f"beam component must be 'r', 'phi' or 'z', got {component!r}")


def line_end_level(omega, index, numerical_aperture, fill_factor, r_end, z_offset, sign, n_samples=401) -> float:
    """|E| at r = r_end over its largest value on [0, r_end] (max over phi: sqrt(max(|A_r|^2 + |A_z|^2, |A_phi|^2)))
    of the continuum beam on a line at z - z_f = z_offset [m] (validation: the line must hold the beam); NaN without a
    field."""
    index = complex(index).real
    wavenumber = float(omega) / float(C0) * index
    sin_max = aperture_sine(numerical_aperture, index)
    rule = debye_rule(wavenumber, sin_max, fill_factor, r_end, abs(float(z_offset)))
    rho = np.linspace(0.0, float(r_end), int(n_samples))
    offsets = np.full(rho.shape, float(z_offset))
    radial, azimuthal, axial = (beam_amplitudes(rule, wavenumber, rho, offsets, sign, name)
                                for name in ('r', 'phi', 'z'))
    magnitude = np.sqrt(np.maximum(np.abs(radial) ** 2 + np.abs(axial) ** 2, np.abs(azimuthal) ** 2))
    peak = float(magnitude.max())
    return float(magnitude[-1]) / peak if peak > 0.0 else math.nan


# ---- placement ----
@dataclass(frozen=True, eq=False)
class BORBeamLine:
    """Cells of a BOR focused-beam line (z = const, from the axis): line_row = the first total-field z row,
    upstream_row = its scattered-field neighbour, n_line r cells [0, n_line) from the axis (uniform), r_edges (n_line +
    1,) / r_centres (n_line,) [m] from the axis, the drawing-frame z [m] of the row centres (E_r, u points) and lo
    edges (E_z points) of (line row, upstream row), the uniform z row width [m]."""
    direction: str
    line_row: int
    upstream_row: int
    n_line: int
    r_edges: np.ndarray
    r_centres: np.ndarray
    row_centres: tuple
    row_edges: tuple
    z_spacing: float
    label: str

    @property
    def sign(self) -> float:
        """+1 toward +z ('positive'), -1 toward -z."""
        return 1.0 if self.direction == 'positive' else -1.0


def bor_focused_beam_line(domain, boundary_config, axis, position, span, direction, label) -> BORBeamLine:
    """BORBeamLine of a line (axis 'x' = along r at z = position [m], span = (start, end) along r [m]); ValueError for
    a vertical line, a line not starting on the axis, fewer than MIN_SPAN_CELLS or non-uniform r cells along it, rows
    within NORMAL_RANGE of the z window edges or non-uniform across it, and a pec / pmc z face (a beam along z is no
    mirror image of itself about a z plane)."""
    if direction not in DIRECTIONS:
        raise ValueError(f'{label} direction must be one of {DIRECTIONS}, got {direction!r}')
    if axis != 'x':
        raise ValueError(f'{label}: the line must be horizontal (z = const, axis along r): the beam travels along the '
                         'rotation axis')
    for face in ('y_lo', 'y_hi'):
        if getattr(boundary_config, face) in _Z_WALLS:
            raise ValueError(f'{label}: the {face} face is a {getattr(boundary_config, face).value} wall, which the '
                             'body of revolution treats as the mirror plane of a z-symmetric structure, but a beam '
                             'along z is not symmetric about any z plane; use cpml on both z faces (a reflector: a '
                             'pec_domain)')
    row, span_cells, _ = port_cells(domain, 'x', position, span, label)
    window_r, window_z = domain.physical_window()
    if window_r.start != 0 or span_cells[0] != 0:
        raise ValueError(f'{label}: the line must start on the axis (r = 0, the first r cell; it starts in r cell '
                         f'{span_cells[0]}): the beam is focused on the axis')
    n_line = int(span_cells[1])
    if n_line < MIN_SPAN_CELLS:
        raise ValueError(f'{label}: the line covers {n_line} r cell(s); a beam needs at least {MIN_SPAN_CELLS} (and '
                         'should hold the whole beam)')
    widths_r, widths_z = domain.cell_widths()
    if not has_uniform_spacing(widths_r[:n_line]):
        raise ValueError(f'{label}: the r cells along the line are not uniform ({float(np.min(widths_r[:n_line])):.4g} '
                         f'.. {float(np.max(widths_r[:n_line])):.4g} m); keep the line in a uniform part of the mesh')
    rows = np.arange(row - NORMAL_RANGE, row + NORMAL_RANGE + 1)
    if rows[0] < window_z.start or rows[-1] >= window_z.stop:
        raise ValueError(f'{label}: needs {NORMAL_RANGE} z rows between the line and the window edges (line row {row}, '
                         f'window rows [{window_z.start}, {window_z.stop})); move the line inside')
    if not has_uniform_spacing(widths_z[rows]):
        raise ValueError(f'{label}: the z rows around the line (rows {rows[0]} .. {rows[-1]}) are not uniform '
                         f'({float(widths_z[rows].min()):.4g} .. {float(widths_z[rows].max()):.4g} m); keep the line in '
                         'a uniform part of the mesh')
    edges_r = np.asarray(domain.cell_edges()[0], dtype=np.float64)
    centres_z = np.asarray(domain.cell_centers()[1], dtype=np.float64)
    edges_z = np.asarray(domain.cell_edges()[1], dtype=np.float64)
    r_edges = edges_r[:n_line + 1] - edges_r[0]   # from the axis (the x_lo grid edge, as BORAssembly)
    upstream = row - 1 if direction == 'positive' else row + 1
    return BORBeamLine(direction=direction, line_row=int(row), upstream_row=int(upstream), n_line=n_line,
                       r_edges=r_edges, r_centres=0.5 * (r_edges[:-1] + r_edges[1:]),
                       row_centres=(float(centres_z[row]), float(centres_z[upstream])),
                       row_edges=(float(edges_z[row]), float(edges_z[upstream])),
                       z_spacing=float(widths_z[row]), label=label)


# ---- the source ----
class BORFocusedBeam(FocusedBeamSource):
    """Focused-beam source of BORPhysics (module header): the x-polarised Debye beam of numerical_aperture and
    fill_factor focused on the axis at drawing-frame z = focus_z [m], launched one-sided from a BORBeamLine toward its
    direction with `power` [W] in the standing wave (ring weight: module header "Power"), E_x at the focus = |C| e^{j
    phase}. The unit slices (Debye values on the two TFSF slices and their power) are kept per omega and line medium
    (read per right-hand side from the assembly's materials on the cells r [0, n_line) x rows line_row +-
    NORMAL_RANGE: one isotropic medium, no PEC, ValueError otherwise), their active positions per system. Azimuthal
    order 1 only, no order sums (term_rhs raises). Attributes the Brillouin drive (physics/mechanics/brillouin_drive)
    reads: name, incident_power, wavelength; axis / position / span are the line glyph (as the mode sources)."""
    power_unit = 'W'
    offset_names = ('dr', 'dz')

    def __init__(self, line, power, numerical_aperture, focus_z, fill_factor, phase, polarization, name):
        self.line = line
        self.label = line.label
        self.name = str(name)
        self.polarization = polarization
        self.direction = line.direction
        self.axis, self.position, self.span = 'x', line.line_row, (0, line.n_line)
        numerical_aperture = float(numerical_aperture)
        if not (math.isfinite(numerical_aperture) and numerical_aperture > 0.0):
            raise ValueError(f'{self.label} numerical_aperture must be finite and > 0, got {numerical_aperture!r}')
        self.numerical_aperture = numerical_aperture
        fill_factor = float(fill_factor)
        if not (math.isfinite(fill_factor) and fill_factor > 0.0):
            raise ValueError(f'{self.label} fill_factor must be finite and > 0, got {fill_factor!r}')
        self.fill_factor = fill_factor
        focus_z = float(focus_z)
        if not math.isfinite(focus_z):
            raise ValueError(f'{self.label} focus_z must be finite [m], got {focus_z!r}')
        self.focus_z = focus_z
        self.solved_omega = None   # rad/s of the last right-hand side
        self._unit = None          # ((omega, eps, mu), codes, indices, line values, upstream values, unit power)
        self._positions = None     # (weakref of system.unknown_index, line positions, upstream positions)
        self.set_amplitude(power, phase)

    @property
    def focus(self) -> tuple:
        """Drawing-frame (r, z) [m] of the focus: (0, focus_z)."""
        return 0.0, self.focus_z

    def moved(self, offset):
        """New BORFocusedBeam on the same line with the focus moved by the drawing-frame offset (dr, dz) [m] (no
        snapping: only the phases of the angular components change); ValueError for dr != 0 (off the axis the beam is
        no m = 1 field)."""
        dr, dz = self._focus_offset(offset)
        if dr != 0.0:
            raise ValueError(f'{self.label}: a radial focus offset ({dr:.6g} m) moves the focus off the axis, where the '
                             'beam is no longer an azimuthal order 1 field; scan along z only')
        moved = BORFocusedBeam(self.line, self.power, self.numerical_aperture, self.focus_z + dz, self.fill_factor,
                               self.phase, self.polarization, self.name)
        moved._positions = self._positions   # same slices: no new position lookup per scan position
        return moved

    def _medium(self, materials):
        """(eps, mu) complex of the one isotropic medium on the cells the beam reads (BORMaterials of the assembly);
        ValueError for PEC, anisotropy or several media there."""
        line = self.line
        cells = (slice(0, line.n_line), slice(line.line_row - NORMAL_RANGE, line.line_row + NORMAL_RANGE + 1))
        where = (f'on the cells r [0, {line.n_line}) x z rows [{cells[1].start}, {cells[1].stop}) around the line')
        if np.any(np.asarray(materials.pec, dtype=bool)[cells]):
            raise ValueError(f'{self.label}: PEC cells {where}; keep the line in one dielectric medium')
        blocks = {'eps_phiphi': materials.component('phi')[cells], 'eps_rr': materials.component('r')[cells],
                  'eps_zz': materials.component('z')[cells]}
        eps = complex(blocks['eps_phiphi'].flat[0])
        mu_block = np.asarray(materials.mu, dtype=np.complex128)[cells]
        mu = complex(mu_block.flat[0])
        for name, block, reference in (*((name, block, eps) for name, block in blocks.items()), ('mu_r', mu_block, mu)):
            spread, uniform = uniform_value(block, reference)
            if not uniform:
                raise ValueError(f'{self.label}: the line medium is not one homogeneous isotropic medium {where} ({name} '
                                 f'differs by {spread:.3g} from {reference:.6g}): the Debye beam is the field of one '
                                 'unbounded isotropic medium; move the line into a uniform region')
        return eps, mu

    def _unit_slices(self, physics, system):
        """(codes, indices, unit line values, unit upstream values, unit standing-wave power [W]) of the system's omega
        and line medium (recomputed when either changed; module header)."""
        eps, mu = self._medium(physics.last_materials)
        key = (float(system.omega), eps, mu)
        if self._unit is not None and self._unit[0] == key:
            return self._unit[1:]
        line = self.line
        index = complex(np.sqrt(eps * mu))
        if not index.real > 0.0:
            raise ValueError(f'{self.label}: the line medium (eps {eps:.6g}, mu {mu:.6g}) carries no propagating wave')
        wavenumber = float(system.omega) / float(C0) * index
        sin_max = aperture_sine(self.numerical_aperture, index, self.label)
        codes, indices = line_keys(line.n_line)
        radial = codes == 0
        rho = np.where(radial, line.r_edges[indices], line.r_centres[np.minimum(indices, line.n_line - 1)])
        offsets = [np.where(codes == 2, edge, centre) - self.focus_z
                   for centre, edge in zip(line.row_centres, line.row_edges)]
        rule = debye_rule(wavenumber, sin_max, self.fill_factor, float(line.r_edges[-1]),
                          max(float(np.max(np.abs(values))) for values in offsets), z_spacing=line.z_spacing)
        slices = []
        for values in offsets:
            amplitudes = np.empty(codes.size, dtype=np.complex128)
            for code, component in _COMPONENT_CODES.items():
                keys = codes == code
                amplitudes[keys] = beam_amplitudes(rule, wavenumber, rho[keys], values[keys], line.sign, component)
            slices.append(amplitudes)
        line_values, upstream_values = slices
        line_positions, upstream_positions = self._slice_positions(system, codes, indices)
        flux = slice_interface_power(system, line_positions, line_values, upstream_positions, upstream_values)
        unit_power = ring_angle(BEAM_ORDER) / (2.0 * math.pi) * flux   # the standing wave (module header)
        if not (math.isfinite(unit_power) and unit_power > 0.0):
            raise ValueError(f'{self.label}: the incident beam carries no power toward {self.direction} '
                             f'({unit_power:.3g} W per unit amplitude); check the line medium (lossless, eps > 0)')
        self._unit = (key, codes, indices, line_values, upstream_values, unit_power)
        return self._unit[1:]

    def _slice_positions(self, system, codes, indices):
        """(line positions, upstream positions) int64 of the slice keys in `system` (bor_ports.key_positions), kept
        while the system's unknown_index array is the same object (a new active set recomputes them)."""
        kept = self._positions
        if kept is not None and kept[0]() is system.unknown_index:
            return kept[1], kept[2]
        line = self.line
        positions = (key_positions(system, codes, indices, line.line_row),
                     key_positions(system, codes, indices, line.upstream_row))
        self._positions = (weakref.ref(system.unknown_index), *positions)
        return positions

    def rhs(self, physics, system):
        """Active rhs of the one-sided launch (bor_ports.slice_tfsf_rhs) at the system's omega and power / phase."""
        if system.azimuthal_order != BEAM_ORDER:
            raise ValueError(f'{self.label}: a focused beam on the axis is the azimuthal order {BEAM_ORDER} field (an '
                             f'x-polarised beam), but the system solves order {system.azimuthal_order}; set '
                             f'azimuthal_order {BEAM_ORDER}')
        codes, indices, line_values, upstream_values, unit_power = self._unit_slices(physics, system)
        scale = math.sqrt(self.power / unit_power) * self._phasor
        line = self.line
        self.solved_omega = float(system.omega)
        line_positions, upstream_positions = self._slice_positions(system, codes, indices)
        return slice_tfsf_rhs(system, line.line_row, line.direction, line_positions, scale * line_values,
                              upstream_positions, scale * upstream_values)

    def term_rhs(self, system, order=None):
        """Refused: the beam is one m = 1 standing wave, not a sum of exp(-j m phi) terms of a point source."""
        raise ValueError(f'{self.label}: a focused beam is one azimuthal order {BEAM_ORDER} field and cannot join a '
                         'driven order sum (azimuthal_orders); clear azimuthal_orders and set azimuthal_order '
                         f'{BEAM_ORDER}')
