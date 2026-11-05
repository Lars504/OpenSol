"""Slab-mode ports of the FDFD physics (A4b-3): the one-sided ModeSourcePhasor, the HarmonicModeMonitor and their
results; port geometry, mode selection, warnings and the mode table are shared with the FDTD ports (mode_ports)."""
# Line axis = direction ALONG the line, normal = the other axis (mode_ports conventions). FDFDPhysics re-solves both
# classes through its port hook (solve(physics, omega) at finalize and after every reassembly: new omega, materials, PEC
# or poles), so unlike the frozen FDTD ports they follow material changes. Modes are FDFD-consistent (dt None, normal
# spacing of the interface / measuring face): a mode of a guide invariant along the normal is then an exact discrete
# solution of the 2D Helmholtz operator. Neither class adds face currents, so face_fields() is the plain curl: a monitor
# must not sit on a mode source's interface face.
import cmath
import math

import numpy as np
from ...core.backend import array_module_of, to_host

from ...core.constants import C0
from .fdfd_monitors import _ascending_wavelength
from .fdfd_sources import phase_factor
from .mode_ports import _DIRECTIONS, PortLine, _check_mode_number, mode_table_rows
from .results import _MODE_TABLE_COLUMNS, _MODE_TABLE_UNITS, mode_power_series
from .slab_modes import align_phase, mode_amplitudes, track_mode


def _check_bloch_line(physics, port):
    """ValueError for a port line spanning the whole period of a Bloch axis with k != 0: its periodic slab mode would
    need the Bloch phase on the wrap (slab_modes solves k = 0 lines only)."""
    config = physics.boundary_config
    if port.lo_kind == 'periodic' and getattr(config, f'bloch_{port.axis}') and \
            getattr(config, f'bloch_k{port.axis}') != 0.0:
        raise ValueError(f"{port.label}: the line spans the whole {port.axis} period with Bloch k{port.axis} = "
                         f"{getattr(config, f'bloch_k{port.axis}'):.6g} rad/m, whose port mode is not available; "
                         f"shorten the line to the guide (truncated ends) or set bloch_k{port.axis} = 0")


def _total_field_mask(port, n_normal, periodic_normal, direction_sign, xp):
    """xp bool Q along the normal, shaped to broadcast over the (nx, ny) grid: the line cell and the cells downstream
    of it (toward +normal for direction_sign +1)."""
    downstream_distance = (np.arange(n_normal) - port.line_cell) * int(direction_sign)
    if periodic_normal:
        # A periodic normal axis has no half plane: Q = the half period downstream of the line cell (distance < n/2, up
        # to the cell opposite it). b depends only on Q holding the line cell and its downstream neighbour and not the
        # two upstream cells, which needs n >= 4.
        if n_normal < 4:
            raise ValueError(f"{port.label}: a one-sided mode source needs at least 4 cells along the periodic "
                             f"{port.normal} axis, got {n_normal}")
        downstream_distance %= n_normal
        inside = 2 * downstream_distance < n_normal
    else:
        inside = downstream_distance >= 0
    shape = (n_normal, 1) if port.normal == 'x' else (1, n_normal)
    return xp.asarray(inside.reshape(shape))


class ModeSourcePhasor:
    """One-sided slab-mode phasor source on a line: total field from the line cell downstream, scattered field upstream;
    injects `power` [W/m] of mode `mode_number` toward +normal ('positive') or -normal ('negative') at `phase` [deg]."""
    # Right-hand side (TFSF sign of spec section 0): b += A (Q u_inc) - Q (A u_inc), Q = the total-field half plane and
    # u_inc the incident mode on the line cells c and their upstream neighbours c' (c - 1 'positive', c + 1 'negative')
    # only, so b is non-zero on those two rows: b[c] = -A[c, c'] u_inc[c'], b[c'] = +A[c', c] u_inc[c]. The mode varies
    # as e^{-j k s} toward its direction: u_inc[c'] = u_inc[c] e^{+j k D_f}, D_f = the dual spacing between c' and c.

    def __init__(self, physics, axis, line_cell, span_cells, power, mode_number, direction, phase, label):
        if direction not in _DIRECTIONS:
            raise ValueError(f"{label} direction must be one of {_DIRECTIONS}, got {direction!r}")
        port = PortLine(physics, axis, line_cell, span_cells, label)
        _check_bloch_line(physics, port)
        self.port = port
        self.axis = axis
        self.position = port.line_cell
        self.span = port.solve_range
        self.power = float(power)
        self.mode_number = _check_mode_number(mode_number)
        self.direction = direction
        self.phase = float(phase)
        self.incident_amplitude = math.sqrt(self.power) * phase_factor(phase, label)   # sqrt(P) e^{j phase}
        self.warnings = []
        self.modes = None          # SlabModes of the last solve; mode = the selected one
        self.mode = None
        self.solved_omega = None   # rad/s of the last solve

        positive = direction == 'positive'
        # Interface face (scattered field upstream of it): lo face of the line cell for 'positive', its hi face for
        # 'negative'; normal_face wraps on a periodic normal axis and refuses a missing / never-updated face.
        self.interface_face = port.normal_face(self.position - 1 if positive else self.position)
        self._normal_spacing = port.normal_spacing(self.interface_face)
        grid = physics.domain.grid
        n_normal = grid.nx if port.normal == 'x' else grid.ny
        upstream_cell = self.interface_face if positive else (self.interface_face + 1) % n_normal
        config = physics.boundary_config
        periodic_normal = bool(config.periodic_x if port.normal == 'x' else config.periodic_y)
        # An upstream cell wrapped across a Bloch normal axis stores the mode one period away: positive from cell 0
        # (upstream = the last cell) e^{-j k L}, negative from the last cell (upstream = cell 0) e^{+j k L}.
        wrap_factor = physics.lower_wrap_factors[0 if port.normal == 'x' else 1]
        if positive and self.position == 0:
            self._upstream_wrap = 1.0 / wrap_factor
        elif not positive and self.position == n_normal - 1:
            self._upstream_wrap = wrap_factor
        else:
            self._upstream_wrap = 1.0
        self._line = port.line_index(self.position)
        self._upstream_line = port.line_index(upstream_cell)
        self._xp = physics.domain.xp
        self._total_field_mask = _total_field_mask(port, n_normal, periodic_normal, 1 if positive else -1, self._xp)
        self._line_incident = None       # complex128 u_inc (run backend) on the line cells (solve range) once solved
        self._upstream_incident = None   # and on the upstream cells
        self._table = None

    def solve(self, physics, omega):
        """Solve the modes at omega [rad/s] on the current materials (dt None, normal spacing D_f), select mode_number
        and store the incident line phasors on the device."""
        port = self.port
        omega = float(omega)
        line, modes = port.solve_modes(physics, omega, self.mode_number, None, self._normal_spacing)
        mode, self.warnings = port.select(line, modes, self.mode_number, self.interface_face, 'source')
        line_incident = self.incident_amplitude * np.asarray(mode.profile, dtype=np.complex128)
        self._line_incident = self._xp.asarray(line_incident)
        self._upstream_incident = self._xp.asarray(line_incident * (cmath.exp(1j * complex(mode.k) * self._normal_spacing)
                                                              * self._upstream_wrap))
        self.modes, self.mode, self.solved_omega = modes, mode, omega
        self._table = None

    def add_to_rhs(self, rhs, physics, system):
        omega = float(system.omega)
        if self.mode is None or self.solved_omega != omega:
            raise ValueError(f"mode not solved yet: {self.port.label} has no mode at omega = {omega:.6g} rad/s "
                             f"(FDFDPhysics solves port modes at finalize and after every reassembly)")
        operator = system.operator
        incident = self._xp.zeros(rhs.shape, dtype=self._xp.complex128)
        incident[self._line] = self._line_incident   # Q u_inc: the line cells are the incident cells inside Q
        inside_product = (operator @ incident.ravel()).reshape(rhs.shape)
        incident[self._upstream_line] = self._upstream_incident
        masked_product = (operator @ incident.ravel()).reshape(rhs.shape)
        masked_product *= self._total_field_mask
        rhs += inside_product
        rhs -= masked_product

    def mode_table(self):
        """Rows [mode number, Re n_eff, Im n_eff, loss dB/cm, Re n_group, edge field, selected] of the last solve's
        modes (n_group computed at the first call per solve); ValueError before the first solve."""
        if self.modes is None:
            raise ValueError(f"mode not solved yet: {self.port.label} (finalize, step or solve_harmonic first)")
        if self._table is None:
            self._table = mode_table_rows(self.port, self.modes, self.solved_omega, self.mode_number, None,
                                          self._normal_spacing)
        return self._table


class HarmonicModeMonitor:
    """Slab-mode decomposition on a line per FDFD solve: amplitudes (a_positive, a_negative) of mode `mode_number`
    toward +normal / -normal from the solution on the line cells and the transverse field on their hi faces."""
    # The first solve selects mode_number and sets the warnings; later solves (a new omega or new materials) track the
    # previous selection by overlap and align its phase, as ModeMonitor does outward from its middle frequency.

    def __init__(self, physics, axis, line_cell, span_cells, mode_number, label):
        port = PortLine(physics, axis, line_cell, span_cells, label)
        _check_bloch_line(physics, port)
        self.port = port
        self.axis = axis
        self.position = port.line_cell
        self.span = port.solve_range
        self.mode_number = _check_mode_number(mode_number)
        self.warnings = []
        self.measuring_face = port.normal_face(self.position)
        self._normal_spacing = port.normal_spacing(self.measuring_face)
        self._line = port.line_index(self.position)   # the line cells; the same index addresses their hi faces
        self._face_field_index = 1 if port.normal == 'x' else 0   # face_fields(): hi-x faces (normal x), else hi-y
        self.line = None           # SlabLine and selected SlabMode of the last solve
        self.mode = None
        self.solved_omega = None   # rad/s of the last solve
        self._records = {}         # omega [rad/s] -> (a_positive, a_negative, n_eff) complex

    def solve(self, physics, omega):
        """Solve the modes at omega [rad/s] on the current materials (dt None, normal spacing D_c): the first solve
        selects mode_number, later ones track the previous selection."""
        port = self.port
        omega = float(omega)
        line, modes = port.solve_modes(physics, omega, self.mode_number, None, self._normal_spacing)
        if self.mode is None:
            mode, self.warnings = port.select(line, modes, self.mode_number, self.measuring_face, 'monitor')
        else:
            tracked = modes[track_mode(line, self.mode, modes)]
            mode = align_phase(line, tracked, self.mode)
        self.line, self.mode, self.solved_omega = line, mode, omega

    def record(self, physics, system):
        """Mode amplitudes of the last solve at system.omega (modes solved there first when needed; one host transfer
        of the two line rows)."""
        omega = float(system.omega)
        if self.solved_omega != omega:
            self.solve(physics, omega)
        out_of_plane = physics.solution[self._line]
        transverse = physics.face_fields()[self._face_field_index][self._line]
        out_of_plane_row, transverse_row = to_host(array_module_of(out_of_plane).stack((out_of_plane, transverse)))
        a_positive, a_negative = mode_amplitudes(self.line, self.mode, out_of_plane_row,
                                                 transverse_row / self.port.normal_sign, 0.5 * self._normal_spacing)
        self._records[omega] = (a_positive, a_negative, complex(self.mode.n_eff))

    def recorded(self):
        """(omega float64 in ascending wavelength order, a_positive, a_negative, n_eff complex128 per omega); ValueError
        before the first record."""
        omegas, records = _ascending_wavelength(self._records, self.port.label)
        values = np.array(records, dtype=np.complex128)
        return omegas, values[:, 0], values[:, 1], values[:, 2]


def harmonic_mode_source_result(source) -> dict:
    """Mode table (kind 'table') of a ModeSourcePhasor at its last solved wavelength: the first max(mode_number + 2, 4)
    modes; extras wavelength [m], power [W/m], direction, mode_number, phase [deg]; ValueError before the solve."""
    rows = source.mode_table()
    return {'kind': 'table', 'columns': list(_MODE_TABLE_COLUMNS), 'units': list(_MODE_TABLE_UNITS),
            'rows': [list(row) for row in rows],
            'wavelength': float(2.0 * math.pi * C0 / source.solved_omega), 'power': float(source.power),
            'direction': source.direction, 'mode_number': int(source.mode_number), 'phase': float(source.phase)}


def harmonic_mode_monitor_result(monitor, physics) -> dict:
    """Mode powers (kind 'xy') of a HarmonicModeMonitor vs ascending wavelength: |a|^2 [W/m] per direction (modes carry
    1 W/m); with exactly one ModeSourcePhasor in `physics` also fractions / phases of a / (sqrt(P) e^{j phase})."""
    omegas, a_positive, a_negative, n_eff = monitor.recorded()
    frequency = omegas / (2.0 * math.pi)
    wavelength = C0 / frequency
    mode_sources = [source for source in physics.sources if isinstance(source, ModeSourcePhasor)]
    normalized = len(mode_sources) == 1
    series, series_units = mode_power_series(dict(zip(_DIRECTIONS, (a_positive, a_negative))), 1.0, 'W/m',
                                             incident=mode_sources[0].incident_amplitude if normalized else None)
    return {'kind': 'xy', 'x': wavelength, 'x_label': 'Wavelength', 'x_unit': 'm', 'y_label': 'Mode power',
            'series': series, 'series_units': series_units, 'frequency': frequency, 'wavelength': wavelength.copy(),
            'n_eff': n_eff.real.copy(), 'n_eff_imag': n_eff.imag.copy(),
            'mode_number': int(monitor.mode_number), 'normalized': normalized}
