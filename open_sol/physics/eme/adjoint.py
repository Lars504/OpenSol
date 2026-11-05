"""EME cascade adjoint (host numpy, 2D ez and hz, 3D cross-sections): objective terms on the port power fractions and
dJ/d eps per cell from the forward and one adjoint cascade field (Lorentz reciprocity), no extra mode solve."""
# Section equation (continuum along the axis s, line discretised): (rho u')' + S u = 0, S = diag(rho) A complex
# symmetric, rho = w s / cell_weight (slab_modes header); interfaces keep u and q = rho u' continuous (the e / h matching
# of a complete basis: h w s = q / (-j omega X0), X0 = MU0 (ez) / EPS0 (hz)). For two solutions the Wronskian
# W(du, v) = v^T dq - du^T rho v' (dq = rho du' + drho u') is continuous and obeys dW/ds = v'^T drho u' - v^T dS u when
# du is the first-order response to (drho, dS). With du purely outgoing at both ports and v_m = the field of unit
# incidence in port mode m at face X, W(f_m, b_m) = -2 j omega X0 (bilinear norm u^T rho u = omega X0 / k) gives the
# change of the outgoing amplitude of mode m at X: dS_m = int (v_m^T dS u - v_m'^T drho u') ds / (2 j omega X0). For
# J = sum weight * |S_p|^2 p_p / p_in (cotangents c_p = weight conj(S_p) p_p / p_in, dJ = 2 Re(c . dS)) the field v
# becomes Lambda = the cascade field of incident amplitudes c at the faces.
# ez: drho = 0, dS = diag(w s k0^2 cs) d eps_r (cs = r / R on a bent line, 1 straight), so dJ/d eps_r = 2 Re(-j omega
# eps0 / 2 w s cs int Lambda u ds) per cell. hz: rho = w s / (eps_t cs) and S = -D^T C D + diag(w s mu cs k0^2) (the
# index term eps_t mu cancels eps_t), C_f = the face conductance (face_weight = face_scale / eps_l of the face), so
# dJ/d eps_t = 2 Re(-j / (2 omega eps0) rho / eps_t int u' Lambda' ds) per cell and dJ/d eps_face = 2 Re(-j / (2 omega
# eps0) C_f / eps_face int (D u)_f (D Lambda)_f ds) per face (face_differences: the end faces by their share of the
# quadratic form), the faces mapped to the cells by the transposed face mean (face_gradient_to_cells).
# 3D (propagation z, the cross_section_modes Yee cross-section, continuum along z): Lorentz reciprocity of the
# semi-discrete Maxwell equations (the transverse curls are adjoint under the stretched dual areas W = A s_x s_y of
# each E point, the weights of the bilinear norm) with v_m as above gives dS_m = -j omega eps0 / 2 sum_p W_p int
# Lambda_p d eps_p E_p dz over the E points p (E_x hi-x faces, E_y hi-y faces, E_z centres; transverse E of a backward
# mode = the forward one's, E_z opposite). eps_p = the point permittivity of the operator (cross_section_modes
# ModePointMaterials: bend factor x width-weighted face mean of the ghost-filled cells), so the cell gradients are the
# point gradients through the transposed bend factor, face mean and ghost fill (cross_section_gradient).
# Exact for a complete basis (the EME then solves the section equation); with a truncated basis the same formula on the
# truncated fields (approximate). Periodic block of n_periods > 1: a drawn cell slice stands for every period, so its
# integrand sums the products of each period's fields (amplitude_pairs over PeriodicAdjointFields: the amplitudes at the
# planes between periods from partial cascades extended one cell at a time, then the cell cascade inside each period).
# Held fixed: the port modes (first / last slice). Per slice the integral of the products of e^{-j k s} /
# e^{-j k (L - s)} terms is closed form, each term written as exp(start + rate t) with Re <= 0 at both ends (stable
# with evanescent modes); u' multiplies a by -j k and b by +j k.
import math

import numpy as np

from ...core.constants import C0, EPS0
from ..em.yee_grid import face_shares
from .cascade import period_plane_amplitudes
from .mode_sets import INPUT_POWER_FLOOR

EME_OBJECTIVE_QUANTITIES = ('transmission_mode', 'reflection_mode', 'transmission', 'reflection')
_SERIES_LIMIT = 0.05         # |rate width| below which int exp uses the series of (e^z - 1) / z (cancellation)


def port_objective(quantity, mode_number, reflection, transmission, column, input_modes, output_modes, unit_power):
    """(value [1], role 'input' | 'output', (N_face,) complex cotangent c with d value = 2 Re(c . dS[:, column])) of a
    power-fraction quantity of the excitation column; ValueError for a missing mode."""
    if quantity not in EME_OBJECTIVE_QUANTITIES:
        raise ValueError(f"EME objective quantity must be one of {', '.join(EME_OBJECTIVE_QUANTITIES)}, got "
                         f'{quantity!r}')
    transmitted = quantity.startswith('transmission')
    block, modes, role = (transmission, output_modes, 'output') if transmitted else (reflection, input_modes, 'input')
    amplitudes = np.asarray(block)[:, column]
    ratio = np.asarray(modes.power, dtype=np.float64) / float(unit_power)
    selected = np.ones(amplitudes.size, dtype=bool)
    if quantity.endswith('_mode'):
        if not 1 <= int(mode_number) <= amplitudes.size:
            raise ValueError(f'EME objective mode_number {mode_number} exceeds the {amplitudes.size} modes of the '
                             f'{role} face')
        selected[:] = False
        selected[int(mode_number) - 1] = True
    value = float(np.sum(np.abs(amplitudes[selected]) ** 2 * ratio[selected]))
    return value, role, np.where(selected, np.conj(amplitudes) * ratio, 0.0).astype(np.complex128)


def objective_cotangents(objectives, input_face, blocks, column, unit_power):
    """(J = sum weight * value, {face: (N_face,) cotangent}) of objectives (quantity, mode_number, weight attributes)
    with blocks = (reflection, transmission, input ModeSet, output ModeSet); zero weights add no cotangent."""
    reflection, transmission, input_modes, output_modes = blocks
    output_face = 'hi' if input_face == 'lo' else 'lo'
    total, cotangents = 0.0, {}
    for objective in objectives:
        value, role, cotangent = port_objective(objective.quantity, objective.mode_number, reflection, transmission,
                                                column, input_modes, output_modes, unit_power)
        total += objective.weight * value
        if objective.weight != 0.0:
            face = input_face if role == 'input' else output_face
            cotangents[face] = cotangents.get(face, 0.0) + objective.weight * cotangent
    return total, cotangents


def _exponential_integrals(start_exponent, rate, width):
    """Elementwise int_0^width exp(start_exponent + rate t) dt (series of (e^z - 1) / z for small |rate width|)."""
    z = rate * width
    small = np.abs(z) < _SERIES_LIMIT
    start = np.exp(start_exponent)
    series = 1.0 + z * (1.0 / 2.0 + z * (1.0 / 6.0 + z * (1.0 / 24.0 + z * (1.0 / 120.0 + z * (1.0 / 720.0
                                                                                                   + z / 5040.0)))))
    direct = (np.exp(start_exponent + z) - start) / np.where(small, 1.0, rate)
    return np.where(small, start * width * series, direct)


def amplitude_pairs(amplitude_sets):
    """(4,) (N, N) outer products a a', a b', b a', b b' summed over the (forward, backward, adjoint forward, adjoint
    backward) amplitude sets (one per period of a periodic cell, else one)."""
    pairs = [0.0, 0.0, 0.0, 0.0]
    for forward, backward, adjoint_forward, adjoint_backward in amplitude_sets:
        for slot, (left, right) in enumerate(((forward, adjoint_forward), (forward, adjoint_backward),
                                              (backward, adjoint_forward), (backward, adjoint_backward))):
            pairs[slot] = pairs[slot] + np.outer(left, right)
    return pairs


def slice_products(k, profiles, pairs, slice_widths, length, start_offset=0.0):
    """(n_slices, M) complex int over each slice of u_t Lambda_t ds in a section of `length` [m] whose slices have
    slice_widths (lo to hi, the first starting start_offset [m] into the section): u = sum (a e^{-jks} + b e^{-jk(L -
    s)}) profile, profiles (N, M), Lambda likewise (amplitudes as Cascade.section_amplitudes), pairs = amplitude_pairs
    of the amplitude sets."""
    widths = np.asarray(slice_widths, dtype=np.float64)
    starts = float(start_offset) + np.cumsum(widths) - widths
    length = float(length)
    k_u, k_adjoint = k[:, None], k[None, :]
    total, difference = k_u + k_adjoint, k_u - k_adjoint
    products = np.empty((widths.size, profiles.shape[1]), dtype=np.complex128)
    for index, (start, width) in enumerate(zip(starts, widths)):
        remaining = length - start
        kernel = (pairs[0] * _exponential_integrals(-1j * total * start, -1j * total, width)
                  + pairs[1] * _exponential_integrals(-1j * (k_u * start + k_adjoint * remaining), -1j * difference,
                                                      width)
                  + pairs[2] * _exponential_integrals(-1j * (k_u * remaining + k_adjoint * start), 1j * difference,
                                                      width)
                  + pairs[3] * _exponential_integrals(-1j * total * remaining, 1j * total, width))
        products[index] = np.sum(profiles * (kernel @ profiles), axis=0)
    return products


def ez_cell_factor(omega, line, centre_scale=None):
    """(m,) complex -j omega eps0 / 2 w s cs: dJ/d eps_r = 2 Re(factor int Lambda u ds) per line cell."""
    factor = (-0.5j * float(omega) * EPS0) * np.asarray(line.cell_widths, dtype=np.float64) \
        * np.asarray(line.cell_stretch, dtype=np.complex128)
    return factor if centre_scale is None else factor * np.asarray(centre_scale, dtype=np.float64)


def hz_factors(omega, line, centre_scale=None, face_scale=None):
    """((m,) -j / (2 omega eps0) rho / eps_t of the active cells, (m + 1,) -j / (2 omega eps0) C_f / eps_face of the
    faces): dJ/d eps_t = 2 Re(cell factor int u' Lambda' ds), dJ/d eps_face = 2 Re(face factor int P_f ds)."""
    prefactor = -0.5j / (float(omega) * EPS0)
    active = np.asarray(line.active)
    cell_weight = np.where(active, np.asarray(line.cell_weight, dtype=np.complex128), 1.0)   # inactive: any value
    widths = np.asarray(line.cell_widths, dtype=np.float64) * np.asarray(line.cell_stretch, dtype=np.complex128)
    scale = 1.0 if centre_scale is None else np.asarray(centre_scale, dtype=np.float64)
    cell_factor = np.where(active, prefactor * widths * scale / cell_weight ** 2, 0.0)
    face_weight = np.asarray(line.face_weight, dtype=np.complex128)
    conductance = (face_weight * np.asarray(line.face_inverse_spacing, dtype=np.float64)
                   / np.asarray(line.face_stretch, dtype=np.complex128))
    faces = 1.0 if face_scale is None else np.asarray(face_scale, dtype=np.float64)
    return cell_factor, prefactor * conductance * face_weight / faces


def face_differences(line, profiles):
    """(N, m + 1) complex face differences of rows p (N, m) in the quadratic form of the line's faces: interior p_k -
    p_(k-1), the end faces sqrt(their diagonal share) p (dirichlet 1, neumann 0, odd 2), periodic: face 0 only."""
    p = np.asarray(profiles, dtype=np.complex128)
    n_cells = p.shape[1]
    differences = np.zeros((p.shape[0], n_cells + 1), dtype=np.complex128)
    differences[:, 1:n_cells] = p[:, 1:] - p[:, :-1]
    if line.lo_end == 'periodic':   # faces 0 and m are one face: counted once
        differences[:, 0] = p[:, 0] - p[:, -1]
        return differences
    end_share = {'dirichlet': 1.0, 'neumann': 0.0, 'odd': 2.0}   # = mode_sets.longitudinal_flux end factors
    differences[:, 0] = math.sqrt(end_share[line.lo_end]) * p[:, 0]
    differences[:, n_cells] = math.sqrt(end_share[line.hi_end]) * p[:, -1]
    return differences


def face_gradient_to_cells(face_gradient, face_index, along_widths, periodic):
    """(n, n_along) float64 dJ/d eps of the grid-line cells from (n, m + 1) dJ/d eps of a line's faces, local face f =
    the hi face of grid cell face_index[f] (the transposed yee_grid.face_average)."""
    widths = np.asarray(along_widths, dtype=np.float64)
    n_along = widths.size
    grid_faces = _sum_columns(face_gradient, face_index, n_along)
    lower_share, upper_share = face_shares(widths, periodic, np.float64)
    partner = np.arange(1, n_along + 1) % n_along if periodic else np.minimum(np.arange(1, n_along + 1), n_along - 1)
    return grid_faces * lower_share[None, :] + _sum_columns(grid_faces * upper_share[None, :], partner, n_along)


def _sum_columns(values, columns, n_columns):
    """(n, n_columns) float64 sums of the columns of real values (n, m) into columns (m,) (repeats summed: one bincount,
    not np.add.at)."""
    values = np.asarray(values, dtype=np.float64)
    n_rows = values.shape[0]
    flat = (np.arange(n_rows)[:, None] * n_columns + np.asarray(columns)[None, :]).ravel()
    return np.bincount(flat, weights=values.ravel(), minlength=n_rows * n_columns).reshape(n_rows, n_columns)


def _section_amplitudes(cascade, index, n_modes, input_face, incident, cotangents):
    """(forward, backward, adjoint forward, adjoint backward) amplitudes of section `index` (Cascade.section_amplitudes):
    the field of `incident` at input_face and the adjoint field of the cotangents {face: incident amplitudes}."""
    forward, backward = cascade.section_amplitudes(index, input_face, incident)
    adjoint_forward = np.zeros(n_modes, dtype=np.complex128)
    adjoint_backward = np.zeros(n_modes, dtype=np.complex128)
    for face, vector in cotangents.items():
        face_forward, face_backward = cascade.section_amplitudes(index, face, vector)
        adjoint_forward += face_forward
        adjoint_backward += face_backward
    return forward, backward, adjoint_forward, adjoint_backward


class PeriodicAdjointFields:
    """Forward and adjoint amplitudes of the cell sections of a periodic block of n_periods > 1 in periods 2..n
    (period 1 = the drawn cell of the main cascade): the plane amplitudes between periods, then the cell cascade."""

    def __init__(self, cascade, block, cell, exit_interface, n_periods, input_face, incident, cotangents):
        self.first, self.last = block
        self.cell = cell   # cascade.cell_cascade of the unit cell
        self.n_periods = int(n_periods)
        self.planes = period_plane_amplitudes(cascade, self.last, cell, exit_interface, self.n_periods,
                                              ([(input_face, incident)], list(cotangents.items())))

    def _cell_amplitudes(self, cell_index, lo_incident, hi_incident):
        lo_forward, lo_backward = self.cell.section_amplitudes(cell_index, 'lo', lo_incident)
        hi_forward, hi_backward = self.cell.section_amplitudes(cell_index, 'hi', hi_incident)
        return lo_forward + hi_forward, lo_backward + hi_backward

    def amplitude_sets(self, index):
        """[(forward, backward, adjoint forward, adjoint backward)] of section `index` in periods 2..n ([] outside the
        cell): period p + 1 lies between planes P_p (forward incident) and P_(p+1) (backward incident)."""
        if not self.first <= index <= self.last:
            return []
        cell_index = index - self.first + 1
        (forward, backward), (adjoint_forward, adjoint_backward) = self.planes
        return [self._cell_amplitudes(cell_index, forward[period - 1], backward[period])
                + self._cell_amplitudes(cell_index, adjoint_forward[period - 1], adjoint_backward[period])
                for period in range(1, self.n_periods)]


def section_gradient(cascade, index, line, modes, slice_widths, length, omega, input_face, incident, cotangents,
                     centre_scale=None, face_scale=None, periodic=None):
    """((n_slices, m) float64 dJ/d eps of the line cells, (n_slices, m + 1) dJ/d eps of the line faces or None) of every
    slice of section `index`: ez eps_r of the cells (faces None); hz eps_t of the cells and eps_l of the faces;
    periodic = PeriodicAdjointFields (a cell section sums its periods) or None."""
    amplitude_sets = [_section_amplitudes(cascade, index, modes.n_modes, input_face, incident, cotangents)]
    if periodic is not None:
        amplitude_sets += periodic.amplitude_sets(index)
    pairs = amplitude_pairs(amplitude_sets)
    if line.polarization == 'ez':
        products = slice_products(modes.k, modes.profiles, pairs, slice_widths, length)
        return 2.0 * np.real(ez_cell_factor(omega, line, centre_scale)[None, :] * products), None
    rate = -1j * modes.k   # u' = sum (rate a e^{-jks} - rate b e^{-jk(L - s)}) u_n
    rates = np.outer(rate, rate)
    derivative_pairs = [rates * pairs[0], -rates * pairs[1], -rates * pairs[2], rates * pairs[3]]
    derivatives = slice_products(modes.k, modes.profiles, derivative_pairs, slice_widths, length)
    face_products = slice_products(modes.k, face_differences(line, modes.profiles), pairs, slice_widths, length)
    cell_factor, face_factor = hz_factors(omega, line, centre_scale, face_scale)
    return (2.0 * np.real(cell_factor[None, :] * derivatives), 2.0 * np.real(face_factor[None, :] * face_products))


def _face_mean_transpose(face_gradient, axis, periodic, cell_widths):
    """Cell gradient of a gradient on the hi faces along `axis` (other axes free): the transposed yee_grid.face_average."""
    lower_share, upper_share = face_shares(cell_widths, periodic, np.float64)
    shape = [1] * face_gradient.ndim
    shape[axis] = -1
    cells = face_gradient * lower_share.reshape(shape)
    upper = face_gradient * upper_share.reshape(shape)   # the part of face k that cell k + 1 (its partner) supplied
    if periodic:
        return cells + np.roll(upper, 1, axis=axis)
    cell_view, upper_view = np.moveaxis(cells, axis, 0), np.moveaxis(upper, axis, 0)
    cell_view[1:] += upper_view[:-1]
    cell_view[-1] += upper_view[-1]   # the last face's partner is the last cell itself
    return cells


def _ghost_fill_transpose(cell_gradient, layout, axes):
    """In place (and returned): transpose of the cross_section_modes wall ghost fill on the array axes `axes` (x, y): a
    ghost cell's gradient moves into its inner neighbour, the ghost keeps 0."""
    for axis, axis_layout in reversed(tuple(zip(axes, (layout.x, layout.y)))):
        view = np.moveaxis(cell_gradient, axis, 0)
        if axis_layout.hi_sign is not None:
            view[-2] += view[-1]
            view[-1] = 0.0
        if axis_layout.lo_sign is not None:
            view[1] += view[0]
            view[0] = 0.0
    return cell_gradient


def section_amplitude_pairs(cascade, index, n_modes, input_face, incident, cotangents, periodic=None):
    """amplitude_pairs of 3D section `index` (forward / adjoint amplitudes, periodic images included): the same for
    every slice block of the section (cross_section_gradient's pairs)."""
    amplitude_sets = [_section_amplitudes(cascade, index, n_modes, input_face, incident, cotangents)]
    if periodic is not None:
        amplitude_sets += periodic.amplitude_sets(index)
    return amplitude_pairs(amplitude_sets)


def cross_section_gradient(cascade, index, modes, points, area_stretch, slice_widths, length, omega, input_face,
                           incident, cotangents, periodic=None, start_offset=0.0, pairs=None):
    """{eps_xx, eps_yy, eps_r: (n_slices, nx, ny) float64 dJ/d eps of the cells} of the slices of 3D section `index`
    with slice_widths starting start_offset [m] into it; points = its cross_section_modes.ModePointMaterials,
    area_stretch = mode_sets.stretched_areas, periodic = PeriodicAdjointFields or None; pairs = the section's
    section_amplitude_pairs (None: computed here)."""
    if pairs is None:
        pairs = section_amplitude_pairs(cascade, index, modes.n_modes, input_face, incident, cotangents, periodic)
    n_x, n_y = points.centre_area.shape
    n_points = n_x * n_y
    # E_x, E_y products from e = [E_x; -E_y] (the E_y sign squares out); E_z of the backward waves has the opposite sign
    transverse = slice_products(modes.k, modes.e, pairs, slice_widths, length, start_offset)
    longitudinal = slice_products(modes.k, modes.profiles[:, 2, :], [pairs[0], -pairs[1], -pairs[2], pairs[3]],
                                  slice_widths, length, start_offset)
    layout = points.layout
    factor = -0.5j * float(omega) * EPS0
    # stretched dual area x bend factor of the E_x (hi-x face), E_y (hi-y face) and E_z (centre) points
    weights = ((layout.x_face_area * area_stretch[0] * points.face_factor[:, None]).ravel(),
               (layout.y_face_area * area_stretch[1] * points.centre_factor[:, None]).ravel(),
               (points.centre_area * area_stretch[2] * points.z_centre_factor[:, None]).ravel())
    point_x, point_y, point_z = (
        2.0 * np.real(factor * weight[None, :] * products).reshape(-1, n_x, n_y)
        for weight, products in zip(weights, (transverse[:, :n_points], transverse[:, n_points:], longitudinal)))
    periodic_x, periodic_y = points.periodic
    return {'eps_xx': _ghost_fill_transpose(_face_mean_transpose(point_x, 1, periodic_x, points.dx_cells), layout,
                                            (1, 2)),
            'eps_yy': _ghost_fill_transpose(_face_mean_transpose(point_y, 2, periodic_y, points.dy_cells), layout,
                                            (1, 2)),
            'eps_r': _ghost_fill_transpose(point_z, layout, (1, 2))}


class EMEObjective:
    """Objective term weight * quantity (a port power fraction of the excitation) of an em_eme physics; records the
    value per solve."""

    def __init__(self, quantity, mode_number=1, weight=1.0):
        if quantity not in EME_OBJECTIVE_QUANTITIES:
            raise ValueError(f"EME objective quantity must be one of {', '.join(EME_OBJECTIVE_QUANTITIES)}, got "
                             f'{quantity!r}')
        if int(mode_number) != mode_number or mode_number < 1:
            raise ValueError(f'EME objective mode_number must be an integer >= 1, got {mode_number!r}')
        weight = float(weight)
        if not math.isfinite(weight):
            raise ValueError(f'EME objective weight must be finite, got {weight!r}')
        self.quantity, self.mode_number, self.weight = quantity, int(mode_number), weight
        self.values = {}   # omega [rad/s] -> value of the last solve there (NaN when it has none)

    def record(self, physics):
        _, unit_power = physics.input_amplitude()
        value = math.nan
        if unit_power is not None and unit_power > INPUT_POWER_FLOOR:
            reflection, transmission, input_modes, output_modes = physics.port_blocks()
            try:
                value = port_objective(self.quantity, self.mode_number, reflection, transmission,
                                       physics.excitation[1] - 1, input_modes, output_modes, unit_power)[0]
            except ValueError:
                value = math.nan
        self.values[float(physics.solved_omega)] = value


def objective_result(objective) -> dict:
    """'xy' result of an EMEObjective vs ascending wavelength: series value and weighted [1]; extras frequency,
    wavelength, quantity, mode_number, weight; ValueError before the first solve."""
    if not objective.values:
        raise ValueError('EME objective: no solve recorded yet')
    omegas = np.array(sorted(objective.values, reverse=True), dtype=np.float64)   # ascending wavelength
    values = np.array([objective.values[omega] for omega in omegas], dtype=np.float64)
    frequency = omegas / (2.0 * math.pi)
    wavelength = C0 / frequency
    return {'kind': 'xy', 'x': wavelength, 'x_label': 'Wavelength', 'x_unit': 'm', 'y_label': 'Objective term',
            'series': {'value': values, 'weighted': objective.weight * values},
            'series_units': {'value': '1', 'weighted': '1'}, 'frequency': frequency, 'wavelength': wavelength.copy(),
            'quantity': objective.quantity, 'mode_number': objective.mode_number, 'weight': objective.weight}
