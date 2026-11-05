"""Straight membrane ports and mode projections (new_modules_plan S7): the strip profiles of an outer face (clamped
walls sin(n pi s / W), one free mirror wall sin / cos((n - 1/2) pi s / W)), their propagation constants, the exact
Dirichlet-to-Neumann rank-n update with an incident mode, outgoing amplitudes / powers, and the projection a_n of a field
onto the profiles along a guide (film ports; a face with bending_stiffness takes plate_ports.PlatePort). Host numpy."""
# exp(+j omega t); d = outward distance beyond the port face, s = the coordinate along it from the guide's first wall,
# W = the guide width (extent of the port's active edges). Outside the domain the guide is assumed straight, uniform
# and clamped at s = 0, W: w = sum_n a_n phi_n(s) exp(-j beta_n d), phi_n = sin(n pi s / W), beta_n^2 = (rho_s omega^2
# - T_ss k_n^2) / T_nn (T (1 + j eta)), k_n = n pi / W, branch Im beta <= 0 (outgoing or decaying). A free wall (an even
# mirror plane, `free_walls`) at s = W gives phi_n = sin(k_n s), at s = 0 phi_n = cos(k_n s), both with
# k_n = (n - 1/2) pi / W: the odd-numbered modes 2n - 1 of the unfolded guide of width 2 W, same
# a_n (int over the half of an even field = half the full one). int phi_n^2 ds = W / 2 for every case, so the (2 / W)
# below and the power hold unchanged; two free walls (a constant mode) are refused. With an incident mode m of
# amplitude A at the port plane (A phi_m exp(+j beta_m d)) the exact boundary flux is T_nn dw/dn = sum_n -j beta_n T_nn
# a_n phi_n + 2 j beta_m T_nn A phi_m with a_n = (2 / W) int w phi_n ds: the operator gains sum_n j beta_n T_nn (2 / W) c_n
# c_n^T (c_n = int phi_n v ds, complex symmetric), the load 2 j beta_m T_nn A c_m. Modes beyond mode_count are left
# natural (keep the port where their near field has decayed). Outgoing amplitude b_n = a_n - A delta_nm, power P_n =
# W omega Re(T_nn beta_n) |b_n|^2 / 4 [W] (= 1/2 T omega Re beta |a|^2 for a unit-normalised profile); the discrete power
# balance is exact (lossless: sum of the outgoing powers = the incident power to round-off). The medium is the mean over
# the port's elements (the cut); a tension varying along it (e.g. coupled from a solid) is accepted up to the port's
# tension_tolerance, the modes then being those of the mean guide (reflections of the order of the spread).
# Mirror walls (a guide wall on a model mirror face: +1 a free symmetry face, -1 a held antisymmetry face): the half
# guide's mode k is mode unfolded_modes(...) of the unfolded guide (2k - 1 / 2k, signed (-1)^(k-1) / (-1)^k when the
# mirror is the wall at s = 0, whose image lies before it); incident modes are given in that unfolded numbering
# (folded_mode) and the powers carry the touch factor 2 per mirror wall (power_factor: the unfolded guide's power).
import math

import numpy as np

from ...core.harmonic_probe import FREQUENCY_MATCH_RTOL, harmonic_probe_result
from ...fem.assembly import assemble_edge_load
from ...fem.diffusion import window_face_edges
from ...fem.elements import lagrange_values
from ...fem.mesher import MERGE_TOLERANCE
from ...fem.quadrature import gauss_line
from ...fem.sampling import locator

PORT_RULE_DEGREE = 10            # Gauss degree of int phi_n(s) v ds per edge (P2 v, smooth profile)
# relative spread of the medium over a port's elements still called uniform (density and loss always; the tension
# parts up to the port's tension_tolerance, whose default this is: above it the cut's mean is used with a warning)
PORT_UNIFORM_RTOL = 1e-6
PROJECTION_SEGMENTS_PER_MODE = 4  # mode_projection: composite Gauss segments across the guide per projected mode
PROJECTION_MIN_SEGMENTS = 16
PROJECTION_DEGREE = 5            # Gauss degree per segment (3 points)
CLAMPED_WALLS = (False, False)


def transverse_wavenumbers(numbers, width, free_walls=CLAMPED_WALLS) -> np.ndarray:
    """k_n [rad/m] of profile numbers n (1-based) across a guide of width W [m] with free_walls (at start, at end)."""
    numbers = np.asarray(numbers, dtype=np.float64)
    if all(free_walls):
        raise ValueError('a guide with two free walls (symmetry on both sides) has a constant mode: not supported, '
                         'clamp or hold one wall')
    return (numbers - (0.5 if any(free_walls) else 0.0)) * math.pi / float(width)


def strip_profiles(coordinates, start, width, numbers, free_walls=CLAMPED_WALLS) -> np.ndarray:
    """phi_n(s) (len(numbers), *coordinates.shape) at s [m]: sin(k_n (s - start)), cos for a free wall at start."""
    numbers = np.asarray(numbers, dtype=np.float64)
    wavenumbers = transverse_wavenumbers(numbers, width, free_walls)
    phase = wavenumbers.reshape(-1, *([1] * np.ndim(coordinates))) * (np.asarray(coordinates) - float(start))
    return np.cos(phase) if free_walls[0] else np.sin(phase)


def unfolded_modes(mirror_walls, mode_count) -> tuple:
    """(unfolded mode numbers, signs) (mode_count,) of the half-guide modes k = 1.. for mirror_walls (sign or None at
    start, at end): one mirror wall +1 -> 2k - 1, -1 -> 2k (signed when the mirror is the start wall); else k, +1."""
    numbers = np.arange(1, int(mode_count) + 1)
    signs = np.ones(int(mode_count), dtype=np.int64)
    start, end = mirror_walls
    if (start is None) == (end is None):   # no mirror wall, or two (no single unfolding): the drawn numbering
        return numbers, signs
    even = (start if start is not None else end) > 0
    unfolded = 2 * numbers - 1 if even else 2 * numbers
    if start is not None:   # the image lies before s = 0: sin(m pi (s + W) / (2 W)) = sign x the half profile
        signs = np.where((numbers - (1 if even else 0)) % 2 == 0, 1, -1).astype(np.int64)
    return unfolded, signs


def folded_mode(mirror_walls, number):
    """(half-guide mode k, sign) of unfolded mode `number` for mirror_walls, None when that mode has the other parity
    about the mirror wall (it cannot enter the half guide)."""
    numbers, signs = unfolded_modes(mirror_walls, int(number))
    hits = np.nonzero(numbers == int(number))[0]
    if not hits.size:
        return None
    return int(hits[0]) + 1, int(signs[hits[0]])


class PortBase:
    """The guide fields MembranePort and PlatePort share (_init_guide): name, face, edges, start / width along the face,
    mode_count, merged incident modes, tension tolerance, mirror walls and their power factor, the medium record."""

    def _init_guide(self, name, face, edges, start, width, mode_count, incidents, tension_tolerance, mirror_walls):
        """Check and keep the shared guide fields (ValueError naming the port)."""
        if not 1 <= int(mode_count):
            raise ValueError(f"port '{name}': mode_count must be >= 1, got {mode_count!r}")
        merged = {}
        for number, value in incidents:
            if not 1 <= int(number) <= int(mode_count):
                raise ValueError(f"port '{name}': an incident mode must lie in [1, mode_count = {mode_count}], got "
                                 f'{number!r}')
            merged[int(number)] = merged.get(int(number), 0.0) + complex(value)
        if not (math.isfinite(float(tension_tolerance)) and float(tension_tolerance) > 0.0):
            raise ValueError(f"port '{name}': tension_tolerance must be a finite value > 0, got {tension_tolerance!r}")
        self.name = str(name)
        self.face = face
        self.edges = np.asarray(edges, dtype=np.int64)
        self.start = float(start)
        self.width = float(width)
        self.tangent_axis = 1 if face[0] == 'x' else 0
        self.mode_count = int(mode_count)
        self.incidents = sorted(merged.items())   # [(mode, complex amplitude [m])]
        self.tension_tolerance = float(tension_tolerance)
        self.mirror_walls = tuple(None if wall is None else int(wall) for wall in mirror_walls)
        self.power_factor = 2.0 ** sum(wall is not None for wall in self.mirror_walls)   # touch factor
        self.tension_spread = math.nan   # largest relative tension spread over the port's elements (set with the medium)
        self.medium_warning = ''         # text when that spread exceeds PORT_UNIFORM_RTOL but is accepted
        self.positions = None            # int64 system dofs of the port (set at assembly)


class MembranePort(PortBase):
    """Runtime of one port: its face edges, guide start / width along the face [m], modes, incident modes [(half-guide
    mode, complex amplitude [m])], tension tolerance (relative), free walls (at start, at end: even mirror faces),
    mirror walls (sign or None; power_factor 2 per mirror wall) and the sector drive scale (result extras);
    positions / loads (system numbering) are set at assembly, the medium (and its tension spread / warning) by the
    physics."""

    def __init__(self, name, face, edges, start, width, mode_count, incident_mode=0, amplitude=0.0,
                 tension_tolerance=PORT_UNIFORM_RTOL, free_walls=CLAMPED_WALLS, incidents=None,
                 mirror_walls=(None, None), drive_scale=1.0):
        if incidents is None:
            incidents = [(int(incident_mode), amplitude)] if int(incident_mode) else []
        self._init_guide(name, face, edges, start, width, mode_count, incidents, tension_tolerance, mirror_walls)
        self.free_walls = tuple(bool(free) for free in free_walls)
        self.drive_scale = float(drive_scale)
        self.wavenumbers = transverse_wavenumbers(np.arange(1, self.mode_count + 1), self.width, self.free_walls)
        self.loads = None           # (mode_count, len(positions)) int phi_n v ds
        self.tension_normal = None  # complex T_nn (1 + j eta) of the port elements [N/m]
        self.tension_transverse = None
        self.density = None         # rho_s [kg/m^2]

    def profile(self, coordinates, number) -> np.ndarray:
        """phi_n at coordinates s [m] along the face (strip_profiles with the port's free walls)."""
        return strip_profiles(coordinates, self.start, self.width, [number], self.free_walls)[0]

    def assemble(self, dofs) -> None:
        """Positions (the system numbering = the active DofMap's) and the profile loads int phi_n v ds on the port
        edges (other dofs have no trace there)."""
        loads = []
        for number in range(1, self.mode_count + 1):
            def profile(points, _edges, number=number):
                return self.profile(points[..., self.tangent_axis], number)
            loads.append(assemble_edge_load(dofs, self.edges, profile, degree=PORT_RULE_DEGREE))
        self.positions = np.unique(dofs.edge_dofs(self.edges)).astype(np.int64)
        self.loads = np.array(loads)[:, self.positions]

    def set_medium(self, tension_normal, tension_transverse, density) -> None:
        self.tension_normal = complex(tension_normal)
        self.tension_transverse = complex(tension_transverse)
        self.density = float(density)

    def propagation_constants(self, omega) -> np.ndarray:
        """Complex beta_n (mode_count,) [rad/m], Im beta <= 0 (module header)."""
        if self.tension_normal is None:
            raise RuntimeError(f"port '{self.name}': medium not set (coefficients not applied yet)")
        squared = (self.density * omega ** 2 - self.tension_transverse * self.wavenumbers ** 2) / self.tension_normal
        beta = np.sqrt(np.asarray(squared, dtype=np.complex128))
        return np.where(beta.imag > 0.0, -beta, beta)

    def terms(self, omega) -> tuple:
        """(dense (k, k) DtN block on positions, (k,) incident load) at omega [rad/s]."""
        beta = self.propagation_constants(omega)
        coefficient = 1j * beta * self.tension_normal * 2.0 / self.width
        block = self.loads.T @ (coefficient[:, None] * self.loads)
        load = np.zeros(len(self.positions), dtype=np.complex128)
        for number, amplitude in self.incidents:
            load += 2j * beta[number - 1] * self.tension_normal * amplitude * self.loads[number - 1]
        return block, load

    def amplitudes(self, solution) -> np.ndarray:
        """Complex a_n (mode_count,) = (2 / W) int w phi_n ds of a system-numbering solution (total field)."""
        return (2.0 / self.width) * (self.loads @ np.asarray(solution)[self.positions])

    def outputs(self, omega, solution) -> dict:
        """{'beta', 'total', 'outgoing' (b_n), 'power' (P_n [W]), 'incident_power' [W]} at omega [rad/s] (powers of the
        unfolded guide: power_factor)."""
        beta = self.propagation_constants(omega)
        total = self.amplitudes(solution)
        outgoing = total.copy()
        incident_power = 0.0
        for number, amplitude in self.incidents:
            outgoing[number - 1] -= amplitude
            incident_power += self._power(omega, beta[number - 1], amplitude)
        power = np.array([self._power(omega, beta[n], outgoing[n]) for n in range(self.mode_count)])
        return {'beta': beta, 'total': total, 'outgoing': outgoing, 'power': power, 'incident_power': incident_power}

    def _power(self, omega, beta, amplitude) -> float:
        return float(0.25 * self.power_factor * self.width * omega * (self.tension_normal * beta).real
                     * abs(amplitude) ** 2)


def port_result(amplitude_record, power_record) -> dict:
    """'xy' result vs frequency [Hz] of a port: harmonic_probe_result series mode_<n>_re / _im / _abs / _phase of the
    outgoing amplitudes [m], plus power_<n> [W], power_fraction_<n> = power_<n> / incident_power (NaN without an
    incident mode anywhere), incident_power [W] (every port's incident modes) and port_incident_power [W] (this
    port's, when recorded)."""
    result = harmonic_probe_result(amplitude_record)
    frequencies, powers = power_record.recorded()
    if not np.array_equal(frequencies, result['x']):
        raise RuntimeError('port result: the amplitude and power records hold different frequencies')
    incident = powers['incident_power'].real
    with np.errstate(divide='ignore', invalid='ignore'):
        for component in power_record.components:
            if component in ('incident_power', 'port_incident_power'):
                continue
            number = component.removeprefix('power_')
            values = powers[component].real
            result['series'][component] = values
            result['series_units'][component] = 'W'
            result['series'][f'power_fraction_{number}'] = np.where(incident > 0.0, values / incident, np.nan)
            result['series_units'][f'power_fraction_{number}'] = '1'
    result['series']['incident_power'] = incident
    result['series_units']['incident_power'] = 'W'
    if 'port_incident_power' in powers:
        result['series']['port_incident_power'] = powers['port_incident_power'].real
        result['series_units']['port_incident_power'] = 'W'
    result['y_label'] = amplitude_record.label
    return result


def port_edges(mesh, face, active) -> tuple:
    """(window edges of face with an active element, start [m], width [m]) of a port; ValueError when the active edges
    do not form one contiguous cross-section."""
    edges = window_face_edges(mesh, face)
    pairs = mesh.edge_elements[edges]
    window = mesh.pml_faces == 0
    element = np.where(window[pairs[:, 0]], pairs[:, 0], pairs[:, 1])
    keep = (element >= 0) & np.asarray(active, dtype=bool)[np.maximum(element, 0)]
    edges = edges[keep]
    if not len(edges):
        raise ValueError(f'port on {face}: no membrane element (tension / areal density) touches that face')
    tangent = 1 if face[0] == 'x' else 0
    ends = mesh.points[mesh.edges[edges]][:, :, tangent]
    start, end = float(ends.min()), float(ends.max())
    covered = float(np.abs(ends[:, 1] - ends[:, 0]).sum())
    x0, y0, x1, y1 = mesh.window
    tolerance = 1e3 * MERGE_TOLERANCE * max(x1 - x0, y1 - y0)
    if abs(covered - (end - start)) > tolerance:
        raise ValueError(f'port on {face}: the membrane meets that face in more than one piece (covered '
                         f'{covered:.6g} m of the span {end - start:.6g} m); a port is one clamped guide cross-section')
    return edges, start, end - start


class ModeProjector:
    """Located sample points of one mode projection (built once): apply(values) gives the complex (mode_count,
    len(positions)) a_n(p) of full-numbering Lagrange values; set_free_walls switches the profiles (solver, before the
    solve: the symmetry features may be applied after the projection)."""

    def __init__(self, positions, basis, dofs, cross_axis, cross, cross_weights, cross_start, width, mode_count):
        self.positions = positions                    # [m] along the guide axis
        self.cross_axis = int(cross_axis)             # 0 x / 1 y: the axis across the guide
        self._basis = basis                           # (points x cross, element basis) at the sample points
        self._dofs = dofs                             # full-numbering dofs of each sample point's element
        self._cross, self._cross_weights = cross, cross_weights   # [m] quadrature points / weights across the guide
        self.cross_start, self.width, self.mode_count = float(cross_start), float(width), int(mode_count)
        self.mirror_walls = (None, None)              # sign or None at cross_start / cross_end (set by the solver)
        self.free_walls = None
        self._weighted_profiles = None                # (mode_count, cross): (2 / W) phi_n(s) x quadrature weight
        self.set_free_walls(CLAMPED_WALLS)

    def set_free_walls(self, free_walls) -> None:
        """Profiles of a guide with free_walls (at cross_start, at cross_end): ports.strip_profiles."""
        free_walls = tuple(bool(free) for free in free_walls)
        if free_walls == self.free_walls:
            return
        profiles = strip_profiles(self._cross, self.cross_start, self.width, np.arange(1, self.mode_count + 1),
                                  free_walls)   # (n, c)
        self._weighted_profiles = (2.0 / self.width) * profiles * self._cross_weights[None, :]
        self.free_walls = free_walls

    def apply(self, values) -> np.ndarray:
        field = np.einsum('kn,kn->k', self._basis, np.asarray(values)[self._dofs])
        field = field.reshape(self.positions.size, -1)
        return np.einsum('nc,pc->np', self._weighted_profiles, field)


class ProjectionRecord:
    """Complex (mode_count, points) projections per solved frequency [Hz] (a repeated frequency replaces its row, as
    core.harmonic_probe)."""

    def __init__(self):
        self._frequencies = []
        self._rows = []

    def __len__(self):
        return len(self._frequencies)

    def clear(self):
        self._frequencies.clear()
        self._rows.clear()

    def record(self, frequency, projection):
        frequency = float(frequency)
        for index, recorded in enumerate(self._frequencies):
            if abs(recorded - frequency) <= FREQUENCY_MATCH_RTOL * max(abs(recorded), abs(frequency)):
                self._rows[index] = np.array(projection, dtype=np.complex128)
                return
        self._frequencies.append(frequency)
        self._rows.append(np.array(projection, dtype=np.complex128))

    def recorded(self) -> tuple:
        """(frequencies float64 ascending [Hz], complex128 (n_frequencies, mode_count, points))."""
        order = np.argsort(np.asarray(self._frequencies, dtype=np.float64), kind='stable')
        return (np.asarray(self._frequencies, dtype=np.float64)[order],
                np.stack([self._rows[index] for index in order]))


def mode_projector(mesh, full_dofs, guide_axis, cross_start, cross_end, positions, mode_count) -> ModeProjector:
    """ModeProjector of a_n(p) = (2 / W) int_cross w(p, s) phi_n(s) ds at positions p [m] along guide_axis (0 x / 1 y),
    W = cross_end - cross_start, clamped-wall profiles until set_free_walls; composite Gauss across (exact for the P2
    trace up to the profile's smoothness). ValueError when a sample point lies outside the mesh."""
    width = float(cross_end) - float(cross_start)
    if not width > 0.0:
        raise ValueError(f'mode projection: cross-section end {cross_end!r} must exceed its start {cross_start!r}')
    segments = max(PROJECTION_MIN_SEGMENTS, PROJECTION_SEGMENTS_PER_MODE * int(mode_count))
    t, weights = gauss_line(PROJECTION_DEGREE)
    edges = np.linspace(float(cross_start), float(cross_end), segments + 1)
    cross = (edges[:-1, None] + np.diff(edges)[:, None] * t[None, :]).ravel()
    cross_weights = (np.diff(edges)[:, None] * weights[None, :]).ravel()
    positions = np.asarray(positions, dtype=np.float64).ravel()
    points = np.empty((positions.size, cross.size, 2))
    points[:, :, guide_axis] = positions[:, None]
    points[:, :, 1 - guide_axis] = cross[None, :]
    location = locator(mesh, full_dofs.geometry_order).locate(points.reshape(-1, 2))
    elements = np.asarray(location.element, dtype=np.int64)
    if (elements < 0).any():
        raise ValueError(f'mode projection: {int((elements < 0).sum())} sample points lie outside the mesh')
    basis = lagrange_values(full_dofs.order, location.reference)
    positions_of = full_dofs.element_position[elements]
    return ModeProjector(positions, basis, full_dofs.element_dofs[positions_of], 1 - guide_axis, cross, cross_weights,
                         cross_start, width, mode_count)
