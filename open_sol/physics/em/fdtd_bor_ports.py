"""Mode ports of the body-of-revolution FDTD (em_fdtd_bor, symmetry 2.3): the bor_ports line modes launched through a
one-sided TFSF interface in time and measured by DFT on two slices, with their results; host numpy / scipy."""
# Interface (a single-order fdtd_bor stepper): z-edge e = row ('positive': total field on the rows >= row and the
# z-edges >= row) or row + 1 ('negative': rows <= row, z-edges <= row + 1, so E_z on the interface edge is total in both
# directions). H is stored total on the interface points H_r(i, e), H_phi(k, e); only d_z entries cross the interface,
# so the H update there adds gain (S C)_z x_inc of the upstream row's E_r / u, and the upstream row's E_r / u updates add
# -gain (V^-1 C^T S V_a)_z h_inc of the interface points (BORTimeStepper.port_operators, add_field_injection).
# Incident phasors: the mode at omega_tilde = (2 / dt) sin(omega_c dt / 2) of the carrier omega_c, the leap-frog's own
# frequency (exact for the stepped operator; line materials as stepped: BORFDTDPhysics.stepped_materials), slices X_s =
# psi^(s - row) X (the backward partner at 1 / psi for 'negative'), the interface H = -S C X / (j omega_tilde mu0 mu) from
# the three-row port strip (complex stretch at omega_tilde in an r_hi cpml). In time Re(U) s(t) + Im(U) s_q(t) (s_q =
# the waveform's quadrature), E samples at n dt, H at (n + 1/2) dt, so DFT / S(omega) of the launched field is the 1 W
# mode times sqrt(P) e^{j phase} at the carrier (at other frequencies the carrier profile leaks a little backward).
# Monitor: DFT of the E points of slices row and row + 1; per frequency the modes at its omega_tilde, mode_number at the
# centre frequency followed outward by overlap, amplitudes by the bor_ports reciprocity form (exact for the stepped
# operator). Modes are solved once, on the materials at the build (no re-solve after material changes).
import cmath
import copy
import math

import numpy as np

from ...core.constants import C0, MU0
from .body_of_revolution import BORMaterials
from .bor_ports import (_MODE_TABLE_COLUMNS, _MODE_TABLE_UNITS, aligned_mode, full_slice_indices, mode_table_rows,
                        port_geometry, slice_amplitudes, solve_port_modes, track_mode)
from .fdtd_bor import layout_sizes
from .results import _DIRECTIONS

_STRIP_ROWS = 3   # bor_ports strip: rows row - 1, row, row + 1


def stepped_frequency(omega, dt) -> float:
    """omega_tilde = (2 / dt) sin(omega dt / 2) [rad/s] of the leap-frog."""
    return 2.0 / float(dt) * math.sin(0.5 * float(omega) * float(dt))


def _line_modes(physics, geometry, row, omega, n_modes):
    """PortModes of the port row at the stepped frequency of omega [rad/s] (materials as stepped at omega)."""
    start, n_line, end_kind, n_window, truncated = geometry
    materials = physics.stepped_materials(omega)

    def line(values):
        return None if values is None else np.asarray(values)[:n_line, row:row + 1]

    line_materials = BORMaterials(eps=line(materials.eps), mu=line(materials.mu), pec=line(materials.pec),
                                  eps_rr=line(materials.eps_rr), eps_zz=line(materials.eps_zz))
    widths_r, widths_z = physics.domain.cell_widths()
    strip_config = copy.copy(physics.boundary_config)
    strip_config.cpml_alpha_max = 0.0   # the stepper's r_hi layer stretches at alpha = 0 (fdtd_bor header)
    return solve_port_modes(widths_r[:n_line], float(widths_z[row]), end_kind, strip_config,
                            physics.pml_dx, line_materials, stepped_frequency(omega, physics.engine.dt),
                            physics.azimuthal_order, n_modes, n_window, truncated=truncated, start=start)


def _check_rows(physics, rows, label):
    """ValueError unless every z row of `rows` lies in the window (no z CPML across the port)."""
    window = physics.domain.physical_window()[1]
    outside = [row for row in rows if not window.start <= row < window.stop]
    if outside:
        raise ValueError(f'{label}: z rows {outside} lie outside the window rows [{window.start}, {window.stop}) (a '
                         'port and its neighbour rows must stay out of the z cpml layers)')


def _check_mode_number(mode_number, label) -> int:
    if isinstance(mode_number, bool) or int(mode_number) != mode_number or mode_number < 1:
        raise ValueError(f'{label}: mode_number must be an integer >= 1, got {mode_number!r}')
    return int(mode_number)


class BORFDTDModeSource:
    """One-sided time-domain port source on a z = const line (from the axis or annular): `power` [W] of mode
    `mode_number` toward +z ('positive') or -z ('negative') at `phase` [deg], amplitude referred to the line row, with
    the waveform `temporal` (needs a quadrature); attached to the stepper at finalize."""

    def __init__(self, physics, row, span_cells, temporal, power, mode_number, direction, phase, label):
        if direction not in _DIRECTIONS:
            raise ValueError(f'{label} direction must be one of {_DIRECTIONS}, got {direction!r}')
        if not (math.isfinite(float(power)) and float(power) >= 0.0):
            raise ValueError(f'{label} power must be finite and >= 0 [W], got {power!r}')
        if not hasattr(temporal, 'quadrature'):
            raise TypeError(f'{label} needs a waveform with a quadrature signal (gaussian or cw), got '
                            f'{type(temporal).__name__}')
        self.label = label
        self.row = int(row)
        self.geometry = port_geometry(physics, self.row, span_cells, label)
        _check_rows(physics, (self.row - 1, self.row, self.row + 1), label)
        self.physics = physics
        self.temporal = temporal
        self.power = float(power)
        self.mode_number = _check_mode_number(mode_number, label)
        self.direction = direction
        self.phase = float(phase)
        self.incident_amplitude = math.sqrt(self.power) * cmath.exp(1j * math.radians(self.phase))
        self.port_modes = None
        self.mode = None
        self.injections = ()

    def attach(self, engine):
        """Solve the modes at the carrier's stepped frequency and add the TFSF field injections (module header)."""
        omega = 2.0 * math.pi * float(self.temporal.f0)
        port_modes = _line_modes(self.physics, self.geometry, self.row, omega, max(self.mode_number + 2, 4))
        if len(port_modes.modes) < self.mode_number:
            raise ValueError(f'{self.label}: {len(port_modes.modes)} guided mode(s) found at the carrier lambda = '
                             f'{C0 / float(self.temporal.f0) * 1e9:.6g} nm, mode_number {self.mode_number} does not '
                             f'exist (azimuthal order {self.physics.azimuthal_order})')
        self.port_modes, self.mode = port_modes, port_modes.modes[self.mode_number - 1]
        h_rows, h_phasor, e_rows, e_phasor = self._incident_terms(engine.shape, engine.port_operators)
        self.injections = (engine.add_field_injection('h', h_rows, h_phasor, self.temporal),
                           engine.add_field_injection('e', e_rows, e_phasor, self.temporal))

    def _incident_terms(self, shape, port_operators):
        """(H rows, H update phasors, E rows, E update phasors) of the TFSF interface (module header)."""
        port_modes, mode = self.port_modes, self.mode
        positive = self.direction == 'positive'
        vector = self.incident_amplitude * (mode.vector if positive else mode.backward)
        ratio = mode.psi if positive else 1.0 / mode.psi   # X_{s+1} = ratio X_s
        upstream = self.row - 1 if positive else self.row + 1
        edge, strip_edge = (self.row, 1) if positive else (self.row + 1, 2)
        nz = shape[1]
        e_sizes, a_sizes = layout_sizes(shape)

        # incident E_r / u of the upstream row (the scattered side the interface H points read)
        incident = np.zeros(sum(e_sizes), dtype=np.complex128)
        planar = port_modes.codes < 2
        incident[full_slice_indices(shape, port_modes, upstream)[planar]] = vector[planar] * ratio ** (upstream - self.row)

        # interface H from the strip (strip rows row - 1, row, row + 1)
        strip = port_modes.strip
        strip_vector = np.zeros(strip.full_size, dtype=np.complex128)
        for strip_slice in range(_STRIP_ROWS):
            strip_vector[full_slice_indices(strip.shape, port_modes, strip_slice)] = vector * ratio ** (strip_slice - 1)
        curl = (strip.curl @ strip_vector[strip.unknown_index]) * strip.a_active
        strip_nr, strip_nz = strip.shape
        strip_ar, strip_az = strip_nr * (strip_nz + 1), (strip_nr + 1) * strip_nz
        sign = np.ones(curl.size)
        sign[strip_ar:] = -1.0   # -S: +1 on a_r, -1 on a_z and a_phi
        strip_h = sign * curl / (1j * port_modes.omega * MU0 * strip.a_mu)
        cells, edges = np.arange(strip_nr), np.arange(strip_nr + 1)
        h_rows = np.concatenate((cells * (nz + 1) + edge, a_sizes[0] + a_sizes[1] + edges * (nz + 1) + edge))
        interface_h = np.zeros(sum(a_sizes), dtype=np.complex128)
        interface_h[h_rows] = np.concatenate((strip_h[cells * (strip_nz + 1) + strip_edge],
                                              strip_h[strip_ar + strip_az + edges * (strip_nz + 1) + strip_edge]))

        z_h, z_e = port_operators
        h_phasor = (z_h @ incident)[h_rows]
        e_rows = np.concatenate((edges * nz + upstream, e_sizes[0] + cells * nz + upstream))
        e_phasor = -(z_e @ interface_h)[e_rows]
        h_keep, e_keep = h_phasor != 0.0, e_phasor != 0.0
        return h_rows[h_keep], h_phasor[h_keep], e_rows[e_keep], e_phasor[e_keep]

    def mode_table(self):
        """Rows [mode number, Re n_eff, Im n_eff, loss dB/cm, n_group, edge field, selected] of the carrier solve."""
        if self.port_modes is None:
            raise ValueError(f'mode not solved yet: {self.label} (the build solves it)')
        return mode_table_rows(self.port_modes, self.mode_number - 1)


class BORFDTDModeMonitor:
    """Time-domain port monitor on a z = const line (from the axis or annular): DFT of the E points of the line row and
    the row above at `frequencies` [Hz]; amplitudes (a_positive, a_negative) of mode `mode_number` per frequency."""

    def __init__(self, physics, row, span_cells, frequencies, mode_number, label):
        self.label = label
        self.row = int(row)
        self.geometry = port_geometry(physics, self.row, span_cells, label)
        _check_rows(physics, (self.row, self.row + 1), label)
        self.physics = physics
        self.frequencies = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
        self.mode_number = _check_mode_number(mode_number, label)
        self.spectra = None
        self._modes = None   # [(PortModes, PortMode)] per frequency once solved

    def attach(self, engine):
        n_line, nz = self.geometry[1], engine.shape[1]
        e_sizes, _ = layout_sizes(engine.shape)
        cells, edges = np.arange(n_line), np.arange(n_line + 1)
        e_index = np.concatenate([np.concatenate((edges * nz + s, e_sizes[0] + cells * nz + s,
                                                  e_sizes[0] + e_sizes[1] + cells * (nz + 1) + s))
                                  for s in (self.row, self.row + 1)])
        self.spectra = engine.add_spectra(self.frequencies, e_index=e_index, a_index=np.zeros(1, dtype=np.int64))

    def solve_modes(self):
        """Modes at every frequency (mode_number at the centre one, followed outward by overlap, phase aligned)."""
        if self._modes is not None:
            return self._modes
        n_modes = max(self.mode_number + 2, 4)
        order = np.argsort(self.frequencies)
        centre = int(order[order.size // 2])
        omegas = 2.0 * math.pi * self.frequencies
        selected = [None] * self.frequencies.size
        port_modes = _line_modes(self.physics, self.geometry, self.row, omegas[centre], n_modes)
        if len(port_modes.modes) < self.mode_number:
            raise ValueError(f'{self.label}: {len(port_modes.modes)} guided mode(s) found at lambda = '
                             f'{C0 / self.frequencies[centre] * 1e9:.6g} nm, mode_number {self.mode_number} does not '
                             'exist')
        selected[centre] = (port_modes, port_modes.modes[self.mode_number - 1])
        for neighbours in (order[order.size // 2 + 1:], order[:order.size // 2][::-1]):
            reference = selected[centre][1]
            for index in neighbours:
                port_modes = _line_modes(self.physics, self.geometry, self.row, omegas[index], n_modes)
                tracked = track_mode(reference, port_modes)
                if tracked is None:
                    raise ValueError(f'{self.label}: no guided mode left at lambda = '
                                     f'{C0 / self.frequencies[index] * 1e9:.6g} nm to follow mode {self.mode_number}')
                reference = aligned_mode(port_modes.modes[tracked], reference)
                selected[index] = (port_modes, reference)
        self._modes = selected
        return selected

    def amplitudes(self, scales=None):
        """(a_positive, a_negative, n_eff) complex128 (n_f,) of the DFT phasors times scales (None: 1)."""
        shape = self.physics.engine.shape
        a_positive, a_negative, n_eff = (np.zeros(self.frequencies.size, dtype=np.complex128) for _ in range(3))
        for index, (port_modes, mode) in enumerate(self.solve_modes()):
            scale = 1.0 if scales is None else scales[index]
            e_values, _ = self.spectra.full_phasors(index, scale)
            here = e_values[full_slice_indices(shape, port_modes, self.row)]
            beyond = e_values[full_slice_indices(shape, port_modes, self.row + 1)]
            a_positive[index], a_negative[index] = slice_amplitudes(port_modes, mode, here, beyond)
            n_eff[index] = complex(mode.n_eff)
        return a_positive, a_negative, n_eff


def fdtd_bor_mode_source_result(source) -> dict:
    """'table' of a BORFDTDModeSource at its carrier: one row per mode found; extras wavelength [m] (carrier), power
    [W], direction, mode_number, phase [deg]; ValueError before the build."""
    return {'kind': 'table', 'columns': list(_MODE_TABLE_COLUMNS), 'units': list(_MODE_TABLE_UNITS),
            'rows': source.mode_table(), 'wavelength': C0 / float(source.temporal.f0), 'power': source.power,
            'direction': source.direction, 'mode_number': source.mode_number, 'phase': source.phase}


def fdtd_bor_mode_monitor_result(monitor, physics, shared) -> dict:
    """'xy' of a BORFDTDModeMonitor vs ascending wavelength: power_* = |a|^2 of the dt-weighted DFT spectra (a.u.);
    with one shared source waveform (shared = (S, valid)) power_per_cw_* = |a / S|^2 [W] (NaN outside the spectrum) and,
    with exactly one BORFDTDModeSource, fraction_* / phase_* of a / (S sqrt(P) e^{j phase}); extras n_eff, n_eff_imag,
    mode_number, normalized."""
    a_positive, a_negative, n_eff = monitor.amplitudes()
    order = np.argsort(C0 / monitor.frequencies)
    raw = {'positive': a_positive[order], 'negative': a_negative[order]}
    series = {f'power_{direction}': np.abs(amplitude) ** 2 for direction, amplitude in raw.items()}
    units = dict.fromkeys(series, 'a.u.')
    mode_sources = [source for source in physics.sources if isinstance(source, BORFDTDModeSource)]
    normalized = shared is not None and len(mode_sources) == 1
    if shared is not None:
        spectrum, valid = shared[0][order], shared[1][order]
        for direction, amplitude in raw.items():
            per_cw = np.where(valid, amplitude / np.where(valid, spectrum, 1.0), np.nan)
            series[f'power_per_cw_{direction}'] = np.abs(per_cw) ** 2
            units[f'power_per_cw_{direction}'] = 'W'
            if normalized:
                relative = per_cw / mode_sources[0].incident_amplitude
                series[f'fraction_{direction}'] = np.abs(relative) ** 2
                series[f'phase_{direction}'] = np.angle(relative)
                units[f'fraction_{direction}'], units[f'phase_{direction}'] = '1', 'rad'
    frequency = monitor.frequencies[order]
    wavelength = C0 / frequency
    return {'kind': 'xy', 'x': wavelength, 'x_label': 'Wavelength', 'x_unit': 'm', 'y_label': 'Mode power',
            'series': series, 'series_units': units, 'frequency': frequency, 'wavelength': wavelength.copy(),
            'n_eff': n_eff[order].real.copy(), 'n_eff_imag': n_eff[order].imag.copy(),
            'mode_number': monitor.mode_number, 'normalized': normalized}
