"""Harmonic monitors of the FDFD physics: record(physics, system) after every solve keeps one value per omega (a
repeated omega overwrites); the result functions list the recorded wavelengths in ascending order."""
# Monitors read the solution (cell centres) and physics.face_fields() (the in-plane field on the hi faces: discrete curl
# plus the H_z sources' face currents; right-hand-side-only sources such as the TFSF / mode phasors are not in it, so a
# monitor must not sit on their total / scattered interface). The mode monitor (fdfd_ports.HarmonicModeMonitor)
# follows the same record / recorded / result protocol.
import math

import numpy as np
from ...core.backend import array_module_of, to_host

from ...core.constants import C0, ETA0
from .monitors import near_to_far_geometry, near_to_far_pattern
from .placement import line_mirror_factor
from .results import _FAR_FIELD_QUANTITY, far_field_angles   # _FAR_FIELD_QUANTITY: 'quantity' per polarisation
from .yee_grid import lower_face

_COMPONENT_LABELS = {'ez': 'E_z', 'hx': 'H_x', 'hy': 'H_y', 'hz': 'H_z', 'ex': 'E_x', 'ey': 'E_y'}
# face_fields() index of each in-plane component: 0 = hi-y faces (ez H_x, hz E_x), 1 = hi-x faces (ez H_y, hz E_y).
_FACE_FIELD_INDEX = {'hx': 0, 'ex': 0, 'hy': 1, 'ey': 1}


def _ascending_wavelength(records, label):
    """(omega float64 descending = ascending wavelength, values in that order) of an omega -> value dict; ValueError
    before the first record."""
    if not records:
        raise ValueError(f"not solved yet: {label} has no recorded wavelength (run a stationary or frequency_domain "
                         f"study, or call step / solve_harmonic, first)")
    omegas = sorted(records, reverse=True)
    return np.array(omegas, dtype=np.float64), [records[omega] for omega in omegas]


def _stacked_on_host(values):
    """Host numpy stack of per-omega records of the run backend (one transfer on the gpu)."""
    return to_host(array_module_of(values[0]).stack(values))


class HarmonicProbe:
    """Complex phasor of one field component in cell (ix, iy) per solve: the out-of-plane field at the centre, ez H_x /
    H_y raw on the +y / +x face (the FDTD probe position), hz E_x / E_y averaged over the cell's two faces."""

    def __init__(self, ix, iy, field_component, unit):
        if field_component not in _COMPONENT_LABELS:
            raise ValueError(f"HarmonicProbe field_component must be one of {tuple(_COMPONENT_LABELS)}, got "
                             f"{field_component!r}")
        self.ix = int(ix)
        self.iy = int(iy)
        self.field_component = field_component
        self.unit = unit
        self._values = {}   # omega [rad/s] -> 0-d complex128 of the run backend (moved to the host by the result)

    def record(self, physics, system):
        component, ix, iy = self.field_component, self.ix, self.iy
        if component in ('ez', 'hz'):
            value = physics.solution[ix, iy].copy()   # a view would follow the next solve
        elif component in ('hx', 'hy'):
            value = physics.face_fields()[_FACE_FIELD_INDEX[component]][ix, iy].copy()
        elif component == 'ex':
            # E_x on hi-y faces: mean of the hi face and yee_grid.lower_face (scalar: no full-grid temporary per record);
            # a lo face wrapped across a Bloch axis takes physics.lower_wrap_factors.
            ex_faces = physics.face_fields()[_FACE_FIELD_INDEX['ex']]
            lower = ex_faces[ix, lower_face(iy, system.periodic_y)]
            value = 0.5 * (ex_faces[ix, iy] + (lower * physics.lower_wrap_factors[1] if iy == 0 else lower))
        else:
            ey_faces = physics.face_fields()[_FACE_FIELD_INDEX['ey']]
            lower = ey_faces[lower_face(ix, system.periodic_x), iy]
            value = 0.5 * (ey_faces[ix, iy] + (lower * physics.lower_wrap_factors[0] if ix == 0 else lower))
        self._values[float(system.omega)] = value

    def recorded(self):
        """(omega float64 in ascending wavelength order, complex128 host phasors); ValueError before the first solve."""
        omegas, values = _ascending_wavelength(
            self._values, f"harmonic probe {self.field_component} at cell ({self.ix}, {self.iy})")
        return omegas, _stacked_on_host(values).astype(np.complex128)


class HarmonicFluxMonitor:
    """Time-averaged Poynting power [W/m] through the hi faces of a line of cells per solve (axis 'x' = line at x-cell
    `position` spanning y; > 0 toward +x / +y), summed with the along-line cell widths [m]."""

    def __init__(self, axis, position, span, along_widths, polarization):
        if axis not in ('x', 'y'):
            raise ValueError(f"HarmonicFluxMonitor axis must be 'x' or 'y', got {axis!r}")
        if polarization not in ('ez', 'hz'):
            raise ValueError(f"HarmonicFluxMonitor polarization must be 'ez' or 'hz', got {polarization!r}")
        span = (int(span[0]), int(span[1]))
        if not 0 <= span[0] < span[1]:
            raise ValueError(f"HarmonicFluxMonitor span must be a half-open cell range lo < hi, got {span}")
        along_widths = np.asarray(along_widths, dtype=np.float64)
        if along_widths.ndim != 1 or along_widths.size < span[1]:
            raise ValueError(f"HarmonicFluxMonitor along_widths (shape {along_widths.shape}) must cover span {span}")
        self.axis = axis
        self.position = int(position)
        self.span = span
        self.polarization = polarization
        self._line = (self.position, slice(*span)) if axis == 'x' else (slice(*span), self.position)
        # float64 [m] on the run backend (use_array_module, FDFDPhysics._add_monitor; adjoint.FluxObjective reads it)
        self._along_widths = along_widths[span[0]:span[1]].copy()
        # Normal flux = flux_factor Re(u conj(t)), u = out-of-plane, t = transverse field: ez S_x = -1/2 Re(E_z H_y*),
        # S_y = +1/2 Re(E_z H_x*); hz S_x = +1/2 Re(E_y H_z*), S_y = -1/2 Re(E_x H_z*) (Re(a b*) = Re(a* b)).
        if polarization == 'ez':
            self._flux_factor = -0.5 if axis == 'x' else 0.5
        else:
            self._flux_factor = 0.5 if axis == 'x' else -0.5
        self._face_index = 1 if axis == 'x' else 0   # transverse field on the hi-x (axis 'x') or hi-y faces
        self._power = {}   # omega [rad/s] -> 0-d float64 W/m of the run backend (moved to the host by the result)

    def use_array_module(self, xp):
        """Move the line weights to the run backend's array module (numpy / cupy) before the first record."""
        self._along_widths = xp.asarray(self._along_widths)

    def record(self, physics, system):
        out_of_plane = physics.solution[self._line]
        transverse = physics.face_fields()[self._face_index][self._line]
        normal_flux = (out_of_plane * transverse.conj()).real   # float64 on the device: no host sync per solve
        self._power[float(system.omega)] = (normal_flux * self._along_widths).sum() * self._flux_factor

    def recorded(self):
        """(omega float64 in ascending wavelength order, power float64 [W/m]); ValueError before the first solve."""
        omegas, powers = _ascending_wavelength(
            self._power, f"harmonic flux monitor (axis {self.axis!r}, line cell {self.position})")
        return omegas, _stacked_on_host(powers).astype(np.float64)


class HarmonicNearToFar:
    """Near-to-far-field box sampled like the FDTD NearToFarField (the closed centre contour of NearToFarGeometry):
    columns x0, x1 (ez E_z / H_y, hz E_y / H_z) and rows y0, y1 (E_z / H_x, E_x / H_z), all phasors of one solve."""

    def __init__(self, x0, x1, y0, y1, cell_centers, polarization, mirrors=None):
        if polarization not in ('ez', 'hz'):
            raise ValueError(f"HarmonicNearToFar polarization must be 'ez' or 'hz', got {polarization!r}")
        self.geometry = near_to_far_geometry(x0, x1, y0, y1, cell_centers, 'HarmonicNearToFar', mirrors)
        self.x0, self.x1, self.y0, self.y1 = int(x0), int(x1), int(y0), int(y1)
        self.polarization = polarization
        # Sample counts of the (E, H) pairs left, right, bottom, top, concatenated in that order per record.
        n_side, n_row = self.y1 - self.y0 + 1, self.x1 - self.x0 + 1
        self._sample_counts = (n_side, n_side, n_side, n_side, n_row, n_row, n_row, n_row)
        self._face_phasors = {}   # omega [rad/s] -> 1D complex128 of the run backend (moved to the host by the result)

    def record(self, physics, system):
        solution = physics.solution
        hi_y_faces, hi_x_faces = physics.face_fields()   # ez (H_x, H_y), hz (E_x, E_y)
        face_pairs = zip(self.geometry.face_cells, self.geometry.lower_face_cells,
                         (hi_x_faces, hi_x_faces, hi_y_faces, hi_y_faces))
        samples = []
        for cells, lower_cells, face_field in face_pairs:
            out_of_plane = solution[cells]
            # The hi-face field averaged onto the centres (as harmonic.face_mean_at_centres): ez H, hz E.
            tangential = 0.5 * (face_field[cells] + face_field[lower_cells])
            samples.extend((out_of_plane, tangential) if self.polarization == 'ez' else (tangential, out_of_plane))
        self._face_phasors[float(system.omega)] = array_module_of(solution).concatenate(samples)   # a copy

    def recorded(self):
        """(omega float64 in ascending wavelength order, ((E, H) left, right, bottom, top) complex128 host (n_omega,
        n_side) face phasors); ValueError before the first solve."""
        omegas, values = _ascending_wavelength(
            self._face_phasors, f"harmonic near-to-far box [{self.x0}, {self.x1}) x [{self.y0}, {self.y1})")
        host = _stacked_on_host(values)
        parts = np.split(host, np.cumsum(self._sample_counts)[:-1], axis=1)
        return omegas, tuple(zip(parts[0::2], parts[1::2]))


def probe_phasor_result(probe, y_label) -> dict:
    """'xy' result of a phasor probe (recorded() -> omegas, complex values; field_component, unit) vs ascending
    wavelength [m]: series 'abs' (probe unit) and 'phase' [rad]; extras frequency, wavelength, component, unit."""
    omegas, values = probe.recorded()
    frequency = omegas / (2.0 * math.pi)
    wavelength = C0 / frequency
    return {'kind': 'xy', 'x': wavelength, 'x_label': 'Wavelength', 'x_unit': 'm', 'y_label': y_label,
            'series': {'abs': np.abs(values), 'phase': np.angle(values)},
            'series_units': {'abs': probe.unit, 'phase': 'rad'},
            'frequency': frequency, 'wavelength': wavelength.copy(), 'component': probe.field_component,
            'unit': probe.unit}


def harmonic_probe_result(probe) -> dict:
    """probe_phasor_result of a HarmonicProbe (y label: the component)."""
    return probe_phasor_result(probe, _COMPONENT_LABELS[probe.field_component])


class CentrePhasorProbe:
    """Complex centre phasor of one field component in cell (ix, iy) per driven solve, read from
    physics.centre_phasors (em_fdfd_bor, em_fdfd_conical; an order sum records its shown phi = 0 sum)."""

    def __init__(self, ix, iy, field_component, unit):
        self.ix, self.iy = int(ix), int(iy)
        self.field_component = field_component
        self.unit = unit
        self._values = {}   # omega [rad/s] -> complex

    def record_order(self, physics, system, terms):
        """Order-sum terms are not needed: record reads the shown phi = 0 sum."""
        return

    def record(self, physics):
        self._values[float(physics.solved_omega)] = complex(physics.centre_phasors[self.field_component][self.ix,
                                                                                                         self.iy])

    def recorded(self):
        """(omega float64 in ascending wavelength order, complex128 phasors); ValueError before the first solve."""
        omegas, values = _ascending_wavelength(self._values, f'the field probe {self.field_component} at cell '
                                                             f'({self.ix}, {self.iy})')
        return omegas, np.array(values, dtype=np.complex128)


def harmonic_flux_result(monitor, physics=None) -> dict:
    """'flux_spectrum' of a HarmonicFluxMonitor: time-averaged power per unit z length [W/m] vs ascending wavelength,
    > 0 toward +x (axis 'x') or +y (axis 'y'); with `physics`, a line reaching a mirror face counts its image too:
    x 'mirror_factor' (2 per such face)."""
    omegas, powers = monitor.recorded()
    frequency = omegas / (2.0 * math.pi)
    mirror_factor = 1.0 if physics is None else line_mirror_factor(physics.domain, physics.boundary_config,
                                                                   monitor.axis, monitor.span)
    return {'kind': 'flux_spectrum', 'frequency': frequency, 'wavelength': C0 / frequency,
            'power': mirror_factor * powers, 'unit': 'W/m', 'mirror_factor': mirror_factor}


def harmonic_far_field_result(monitor, n_angles=360) -> dict:
    """'far_field' of a HarmonicNearToFar vs ascending wavelength, keys as results.far_field_result: 'power' = radiated
    power per unit z length and angle |pattern|^2 / (2 eta0) [W/(m rad)] (n_wavelengths, n_angles) at 'angles' =
    |amplitude|^2 (amplitude_re / _im: pattern / sqrt(2 eta0), combine_sector_far_fields)."""
    angles = far_field_angles(n_angles)
    omegas, faces = monitor.recorded()
    frequency = omegas / (2.0 * math.pi)
    pattern = near_to_far_pattern(monitor.polarization, frequency, angles, faces, monitor.geometry)
    amplitude = np.asarray(pattern, dtype=np.complex128) / math.sqrt(2.0 * float(ETA0))
    return {'kind': 'far_field', 'frequency': frequency, 'wavelength': C0 / frequency, 'angles': angles,
            'power': np.abs(amplitude) ** 2, 'amplitude_re': amplitude.real, 'amplitude_im': amplitude.imag,
            'unit': 'W/(m rad)', 'quantity': _FAR_FIELD_QUANTITY[monitor.polarization]}
