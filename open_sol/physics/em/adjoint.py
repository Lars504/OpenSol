"""FDFD adjoint sensitivity: objective terms (weighted flux / mode-power monitors) and dJ/d eps per cell from one
transposed solve A^T lam = dJ/du on the forward factorisation (FDFDPhysics.solve_adjoint / eps_sensitivity)."""
# Derivation (exp(+j omega t); A u = b = the solved, pinned operator of fdfd_2d). A real J(u) changes by dJ = 2 Re(g^T
# du) with g = dJ/du the Wirtinger derivative (conj(u) held fixed), and du = -A^-1 (dA) u, so dJ/dp = -2 Re(lam^T
# (dA/dp) u) with A^T lam = g (unconjugated transpose: SuperLU trans='T'). Objectives read the filled solution (outer
# wall rows = sign x inner row, then PEC re-zeroed: fdfd_2d._fill_wall_rows), so g is first folded back through the
# transpose of that fill; lam and u then vanish on every pinned cell.
# ez: A = ... + diag(k0^2 eps), so dJ/d eps_r = -2 Re(k0^2 lam u) per cell.
# hz: A = pin(dx_e diag(V_x) dx_h + dy_e diag(V_y) dy_h + diag(k0^2 mu)), pin(M) = D (M + diag(M s)) D + (I - D) with
# the image signs s of the outer wall rows (operators.pin_image_cells), V = 1 / eps_face, eps_face = face_average of the
# cells: per face lam^T dA u = [(D_e^T lam)(D_h u) + (D_e^T (lam u))(D_h s)] dV, dV = -V^2 d eps_face, and the
# transposed face mean spreads d eps_face onto its two cells (faces forced to V = 0, PEC or even walls, drop out).
# Conformal PEC: the operator holds V f (f = HarmonicSystem.face_fractions, l / L), the face fields V: dA carries f too.
# hz with eps_xy (harmonic.py header): A also holds -(dx_e X_xy dy_h + dy_e X_yx dx_h), X = keep F diag(eta_xy) C keep
# (operators.off_diagonal_face_couplings), so p^T dX q = sum_c d eta_c (F^T keep p)_c (C keep q)_c per cell; V_y / V_x
# are 1 / face means of eps_xx - eps_xy^2 / eps_yy and eps_yy - eps_xy^2 / eps_xx and eta_xy = -eps_xy / (eps_xx eps_yy
# - eps_xy^2): the cell cotangents of these three chain onto d eps_xx, d eps_yy (eps_xy held: the design arrays) and d
# eps_xy (off_diagonal_density; 0 where the assembly zeroes eps_xy). The image-sign fold sees no X change (eps_xy is 0
# on the outer two wall rows), and face fields t = scale (V c - X c') make the adjoint source carry -scale X^T too.
# Held fixed (not differentiated): source currents and calibrations, TFSF / port incident fields, the port modes (line
# cells) and the objective's own face weights (hz: E on its faces is W (D u + J)), so the gradient is exact only off
# every source / port / objective line and the cells beyond their hi faces.
# Backends: every array is on the physics' run backend (`physics.domain.xp`: cupy + cupyx CSR or numpy + scipy CSR).
import math
from dataclasses import dataclass

import numpy as np

from ...core.backend import array_module_of
from ...core.constants import C0, DTYPE, EPS0, MU0
from .fdfd_monitors import HarmonicFluxMonitor
from .fdfd_ports import HarmonicModeMonitor
from .operators import active_coupling_faces, off_diagonal_mean_maps
from .slab_modes import mode_amplitude_weights
from .yee_grid import cell_row, face_shares, grid_cell_widths

OBJECTIVE_QUANTITIES = ('flux', 'mode_power_positive', 'mode_power_negative')
_MODE_POWER_PREFIX = 'mode_power_'


def _finite_weight(weight):
    """float weight; ValueError unless finite."""
    weight = float(weight)
    if not math.isfinite(weight):
        raise ValueError(f"objective weight must be finite, got {weight!r}")
    return weight


class FluxObjective(HarmonicFluxMonitor):
    """HarmonicFluxMonitor whose power [W/m] (> 0 toward +x / +y) enters the adjoint objective times `weight`;
    mirror_factor (placement.line_mirror_factor) unfolds it as harmonic_flux_result does."""

    quantity = 'flux'

    def __init__(self, axis, position, span, along_widths, polarization, weight, mirror_factor=1.0):
        super().__init__(axis, position, span, along_widths, polarization)
        self.weight = _finite_weight(weight)
        self.mirror_factor = float(mirror_factor)

    def recorded_objective(self):
        """(omega float64 in ascending wavelength order, unfolded flux [W/m] float64); ValueError before the first
        solve."""
        omegas, powers = self.recorded()
        return omegas, self.mirror_factor * powers

    def adjoint_terms(self, physics, system):
        """(value [W/m], line index, dJ/du on the line cells, dJ/dt on their hi faces, face index) of the last solve,
        J = mirror_factor f sum w Re(u conj(t)) (complex128 rows of the run backend)."""
        out_of_plane = physics.solution[self._line]
        transverse = physics.face_fields()[self._face_index][self._line]
        flux_factor = self.mirror_factor * self._flux_factor
        value = flux_factor * float(((out_of_plane * transverse.conj()).real * self._along_widths).sum())
        half_weights = (0.5 * flux_factor) * self._along_widths
        return value, self._line, half_weights * transverse.conj(), half_weights * out_of_plane.conj(), \
            self._face_index


class ModePowerObjective(HarmonicModeMonitor):
    """HarmonicModeMonitor whose mode power |a|^2 [W/m] toward `direction` enters the adjoint objective times
    `weight`."""

    def __init__(self, physics, axis, line_cell, span_cells, mode_number, label, direction, weight):
        if direction not in ('positive', 'negative'):
            raise ValueError(f"{label} direction must be 'positive' or 'negative', got {direction!r}")
        super().__init__(physics, axis, line_cell, span_cells, mode_number, label)
        self.direction = direction
        self.quantity = f'{_MODE_POWER_PREFIX}{direction}'
        self.weight = _finite_weight(weight)

    def recorded_objective(self):
        """(omega float64 in ascending wavelength order, |a|^2 [W/m] float64 of `direction`); ValueError before the
        first solve."""
        omegas, a_positive, a_negative, _ = self.recorded()
        amplitudes = a_positive if self.direction == 'positive' else a_negative
        return omegas, np.abs(amplitudes) ** 2

    def adjoint_terms(self, physics, system):
        """As FluxObjective.adjoint_terms for J = |a|^2, a = W_u U + W_t T' (slab_modes.mode_amplitude_weights) with
        the mode of the last record held fixed."""
        if self.mode is None or self.solved_omega != float(system.omega):
            raise ValueError(f"{self.port.label}: no mode recorded at the last solve (omega {system.omega:.6g} rad/s)")
        row = 0 if self.direction == 'positive' else 1
        u_weights, t_weights = mode_amplitude_weights(self.line, self.mode, 0.5 * self._normal_spacing)
        xp = physics.domain.xp
        u_weights = xp.asarray(u_weights[row])
        t_weights = xp.asarray(t_weights[row] / self.port.normal_sign)   # T' = t / sign_n
        out_of_plane = physics.solution[self._line]
        transverse = physics.face_fields()[self._face_field_index][self._line]
        amplitude = complex((u_weights * out_of_plane).sum() + (t_weights * transverse).sum())
        conjugate = amplitude.conjugate()
        return abs(amplitude) ** 2, self._line, conjugate * u_weights, conjugate * t_weights, self._face_field_index


def is_objective(monitor) -> bool:
    """True for an adjoint objective monitor (FluxObjective, ModePowerObjective)."""
    return isinstance(monitor, (FluxObjective, ModePowerObjective))


def objective_result(objective) -> dict:
    """'xy' result of an objective vs ascending wavelength: series value (the monitor quantity [W/m]) and weighted
    (weight * value [W/m]); extras frequency, wavelength, quantity, weight."""
    omegas, values = objective.recorded_objective()
    frequency = omegas / (2.0 * math.pi)
    wavelength = C0 / frequency
    return {'kind': 'xy', 'x': wavelength, 'x_label': 'Wavelength', 'x_unit': 'm', 'y_label': 'Objective term',
            'series': {'value': values, 'weighted': objective.weight * values},
            'series_units': {'value': 'W/m', 'weighted': 'W/m'}, 'frequency': frequency,
            'wavelength': wavelength.copy(), 'quantity': objective.quantity, 'weight': objective.weight}


@dataclass(frozen=True, eq=False)
class EpsSensitivity:
    """Adjoint gradient of J = sum weight * value over the objectives at one solve (zeros without one)."""
    objective: float         # J [W/m]
    omega: float | None      # rad/s of the solve; None = no objective or no solve
    cell_gradients: dict     # ez eps_r / hz eps_xx, eps_yy -> float64 (nx, ny) dJ/d eps per cell [W/m] (run backend)
    density: object          # float32 (nx, ny) dJ/d eps_r per cell area [W/m^3] (hz: eps_xx + eps_yy terms)
    off_diagonal_density: object = None   # hz: float32 (nx, ny) dJ/d eps_xy per cell area [W/m^3]; else None


def _face_operator(system, face_index):
    """(D_h, scale, weights) of face set face_index (0 hi-y, 1 hi-x faces): its in-plane field is scale * weights *
    (D_h u + face currents), as FDFDPhysics.face_fields."""
    ops = system.ops
    if system.polarization == 'ez':
        curl_scale = 1.0 / (1j * system.omega * MU0)
        scales = (-curl_scale, curl_scale)
    else:
        curl_scale = 1.0 / (1j * system.omega * EPS0)
        scales = (curl_scale, -curl_scale)
    if face_index == 0:
        return ops.dy_h, scales[0], system.weight_hi_y_faces
    return ops.dx_h, scales[1], system.weight_hi_x_faces


def _fold_filled_rows(source, wall_rows, pec_cells):
    """In place: transpose of the wall-row fill and PEC re-zero of fdfd_2d (rows in reverse order: inner += sign x
    outer, outer = 0), so the adjoint source lives on the solved unknowns."""
    if pec_cells is not None:
        array_module_of(source).copyto(source, source.dtype.type(0), where=pec_cells)
    for axis, outer, inner, _, sign in reversed(wall_rows):
        if sign != 0.0:
            cell_row(source, axis, inner)[...] += sign * cell_row(source, axis, outer)
        cell_row(source, axis, outer)[...] = 0


def _face_average_transpose(face_values, axis, periodic, cell_widths):
    """Transpose of yee_grid.face_average along `axis`: each hi-face value onto its lower cell (share w_k / (w_k +
    w_(k+1))) and its partner cell (periodic wrap; the last face of a non-periodic axis onto its own cell)."""
    xp = array_module_of(face_values)
    lower_share, upper_share = face_shares(cell_widths, periodic, np.float64)
    broadcast_shape = [1, 1]
    broadcast_shape[axis] = lower_share.size
    cell_values = face_values * xp.asarray(lower_share.reshape(broadcast_shape))
    upper_part = face_values * xp.asarray(upper_share.reshape(broadcast_shape))
    if periodic:
        cell_values += xp.roll(upper_part, 1, axis=axis)
        return cell_values
    following, preceding, last = [slice(None)] * 2, [slice(None)] * 2, [slice(None)] * 2
    following[axis], preceding[axis], last[axis] = slice(1, None), slice(None, -1), -1
    cell_values[tuple(following)] += upper_part[tuple(preceding)]
    cell_values[tuple(last)] += upper_part[tuple(last)]
    return cell_values


def objective_cotangents(physics):
    """(J [W/m], cell cotangent, [hi-y, hi-x face cotangents], any weight != 0) of an FDFDPhysics' objectives at its
    last solve: dJ = 2 Re(c du + f dt) on the filled solution u / face fields t (None where no objective acts)."""
    shape = physics.solution.shape
    xp = physics.domain.xp
    cell_part, face_parts, total, weighted = None, [None, None], 0.0, False
    for objective in (monitor for monitor in physics.monitors if is_objective(monitor)):
        value, line, cell_values, face_values, face_index = objective.adjoint_terms(physics, physics.system)
        total += objective.weight * value
        if objective.weight == 0.0:
            continue
        weighted = True
        if cell_part is None:
            cell_part = xp.zeros(shape, dtype=xp.complex128)
        cell_part[line] += objective.weight * cell_values
        if face_parts[face_index] is None:
            face_parts[face_index] = xp.zeros(shape, dtype=xp.complex128)
        face_parts[face_index][line] += objective.weight * face_values
    return total, cell_part, face_parts, weighted


def adjoint_source(physics, cell_part, face_parts, wall_rows):
    """Flat complex128 adjoint source g = c + D_h^T (scale W f) (hz eps_xy: - D_h'^T X^T (scale f)) of a filled-solution
    cotangent (None parts = 0), folded through the wall-row fill onto the solved unknowns."""
    system = physics.system
    shape = physics.solution.shape
    xp = physics.domain.xp
    source = xp.zeros(shape, dtype=xp.complex128) if cell_part is None else cell_part.astype(xp.complex128, copy=True)
    for face_index, face_part in enumerate(face_parts):
        if face_part is not None:   # g += D_h^T (scale W dJ/dt): the face field is linear in u
            difference, scale, weights = _face_operator(system, face_index)
            scaled_part = face_part * scale
            source += (difference.T @ (scaled_part * weights).ravel()).reshape(shape)
            if system.off_diagonal is not None:   # E_x -= scale X_yx dx_h u (hi-y), E_y -= scale X_xy dy_h u (hi-x)
                other_difference = system.ops.dx_h if face_index == 0 else system.ops.dy_h
                coupling = system.off_diagonal[face_index]
                source -= (other_difference.T @ (coupling.T @ scaled_part.ravel())).reshape(shape)
            del scaled_part
    _fold_filled_rows(source, wall_rows, system.pec_cells)   # the fixed unknowns (hz conformal: not the live cut cells)
    return source.ravel()


def _eta_cotangent(physics, system, x_from_y_pairs, y_from_x_pairs):
    """complex128 (nx, ny) h with dJ = 2 Re(h d eta_xy) per cell: sum of p^T dX q over the (p, q) face pairs of X_xy
    (p hi-x, q hi-y faces) and X_yx (p hi-y, q hi-x), X = keep F diag(eta_xy) C keep (operators)."""
    grid = physics.domain.grid
    xp = physics.domain.xp
    shape = (grid.nx, grid.ny)
    periodic = (system.periodic_x, system.periodic_y)
    centre_x, centre_y, face_x, face_y = off_diagonal_mean_maps(grid_cell_widths(grid), periodic,
                                                                physics.lower_wrap_factors, xp=xp)
    keep_x, keep_y = active_coupling_faces(system.weight_hi_x_faces, system.weight_hi_y_faces, *periodic)
    eta_cotangent = xp.zeros(shape, dtype=xp.complex128)
    for pairs, p_keep, q_keep, face_map, centre_map in ((x_from_y_pairs, keep_x, keep_y, face_x, centre_y),
                                                        (y_from_x_pairs, keep_y, keep_x, face_y, centre_x)):
        for p_faces, q_faces in pairs:
            eta_cotangent += ((face_map.T @ (p_faces * p_keep).ravel())
                              * (centre_map @ (q_faces * q_keep).ravel())).reshape(shape)
    return eta_cotangent


def _assembled_xy_cells(physics, wall_rows, shape):
    """float64 (nx, ny) 1 where the assembly keeps eps_xy, 0 in PEC cells and the outer two rows of wall / mirror faces
    (FDFDPhysics.assembly_eps_xy)."""
    xp = physics.domain.xp
    kept = xp.ones(shape, dtype=xp.float64)
    if physics.pec_cells is not None:
        xp.copyto(kept, 0.0, where=physics.pec_cells)
    for axis, outer, inner, _, _ in wall_rows:
        cell_row(kept, axis, outer)[...] = 0.0
        cell_row(kept, axis, inner)[...] = 0.0
    return kept


def permittivity_gradients(physics, adjoint, wall_rows, face_currents=None, face_cotangents=None,
                           off_diagonal=False):
    """{array: float64 (nx, ny) dJ/d eps} = -2 Re(lam^T dA u) for the adjoint lam (nx, ny); hz options: face_currents
    whose rhs W J is differentiated, face_cotangents of face fields (dt at fixed u), off_diagonal adds 'eps_xy'."""
    system = physics.system
    grid = physics.domain.grid
    shape = (grid.nx, grid.ny)
    forward = physics.solution.copy()   # the solved unknowns: outer wall rows back to 0
    for axis, outer, _, _, _ in wall_rows:
        cell_row(forward, axis, outer)[...] = 0
    if system.polarization == 'ez':
        return {'eps_r': (-2.0 * (system.omega / C0) ** 2) * (adjoint * forward).real}
    image_signs = physics.domain.xp.zeros(shape, dtype=np.complex128)
    for axis, outer, _, _, sign in wall_rows:
        cell_row(image_signs, axis, outer)[...] = sign
    flat_adjoint, flat_forward = adjoint.ravel(), forward.ravel()
    flat_product, flat_signs = (adjoint * forward).ravel(), image_signs.ravel()
    ops = system.ops
    cell_dx, cell_dy = grid_cell_widths(grid)
    currents = (None, None) if face_currents is None else face_currents
    cotangents = (None, None) if face_cotangents is None else face_cotangents
    # t = scale (W c - X c') of the filled solution's face curls c (incl. every source's face currents)
    full_curls = physics.face_curls() if any(part is not None for part in cotangents) else (None, None)
    coupled = off_diagonal or system.off_diagonal is not None
    along_cotangents = []   # complex cell cotangents of eps_xx - eps_xy^2 / eps_yy (E_x), eps_yy - eps_xy^2 / eps_xx
    residual_pieces, cotangent_pieces = [], []   # per face set: (D_e^T lam, curl), (-scale f or None, full curl)
    # (D_e, D_h, face weights V, axis, periodic, widths, face current (sign as in face_curls), face index): eps_xx
    # sets the hi-y (E_x) faces, eps_yy the hi-x
    fractions = (None, None) if system.face_fractions is None else system.face_fractions   # (hi-x, hi-y)
    for d_e, d_h, weights, axis, periodic, widths, current, index, fraction in (
            (ops.dy_e, ops.dy_h, system.weight_hi_y_faces, 1, system.periodic_y, cell_dy, currents[0], 0, fractions[1]),
            (ops.dx_e, ops.dx_h, system.weight_hi_x_faces, 0, system.periodic_x, cell_dx,
             None if currents[1] is None else -currents[1], 1, fractions[0])):
        d_e_transposed = d_e.T
        curl = d_h @ flat_forward
        if current is not None:   # A u - b holds D_e W (D_h u +- J): the rhs face currents see W (and X) as well
            curl = curl + current.ravel()
        adjoint_curl = d_e_transposed @ flat_adjoint
        face_term = adjoint_curl * curl
        if wall_rows:   # the image-sign fold of the wall rows (s = 0 without walls: two SpMVs saved)
            face_term += (d_e_transposed @ flat_product) * (d_h @ flat_signs)
        face_term = face_term.reshape(shape) * weights * weights   # dJ/d eps_face = 2 Re(face_term) per face
        if fraction is not None:   # conformal PEC: the operator (not the face fields) holds W l / L
            face_term *= fraction
        scaled_cotangent = None
        if cotangents[index] is not None:   # dJ = 2 Re(f scale c dW), dW = -W^2 d eps_face
            scaled_cotangent = cotangents[index] * _face_operator(system, index)[1]
            face_term -= scaled_cotangent * full_curls[index] * weights * weights
        along_cotangents.append(_face_average_transpose(face_term, axis, periodic, widths))
        del face_term
        if coupled:
            residual_pieces.append((adjoint_curl.reshape(shape), curl.reshape(shape)))
            cotangent_pieces.append((None if scaled_cotangent is None else -scaled_cotangent, full_curls[index]))
    along_xx, along_yy = along_cotangents
    if not coupled:
        return {'eps_xx': 2.0 * along_xx.real, 'eps_yy': 2.0 * along_yy.real}
    # dJ = -2 Re(lam^T dA u) holds +2 Re((D_e^T lam)^T dX curl); t holds -scale dX c': pair p and q of each X
    x_from_y_pairs = [(residual_pieces[1][0], residual_pieces[0][1])]
    y_from_x_pairs = [(residual_pieces[0][0], residual_pieces[1][1])]
    if cotangent_pieces[1][0] is not None:
        x_from_y_pairs.append((cotangent_pieces[1][0], cotangent_pieces[0][1]))
    if cotangent_pieces[0][0] is not None:
        y_from_x_pairs.append((cotangent_pieces[0][0], cotangent_pieces[1][1]))
    del residual_pieces, cotangent_pieces
    eta_cotangent = _eta_cotangent(physics, system, x_from_y_pairs, y_from_x_pairs)
    del x_from_y_pairs, y_from_x_pairs
    eps_xx, eps_yy, eps_xy = system.cell_eps_xx, system.cell_eps_yy, system.cell_eps_xy
    if eps_xy is None:   # eps_xy = 0: the diagonal terms decouple, d eta_xy / d eps_xy = -1 / (eps_xx eps_yy)
        gradients = {'eps_xx': 2.0 * along_xx.real, 'eps_yy': 2.0 * along_yy.real}
        xy_cotangent = -eta_cotangent / (eps_xx * eps_yy)
    else:
        squared = eps_xy * eps_xy
        determinant_squared = (eps_xx * eps_yy - squared) ** 2
        eta_share = eta_cotangent * eps_xy / determinant_squared   # d eta / d eps_xx = eps_xy eps_yy / det^2
        gradients = {'eps_xx': 2.0 * (along_xx + along_yy * squared / (eps_xx * eps_xx) + eta_share * eps_yy).real,
                     'eps_yy': 2.0 * (along_yy + along_xx * squared / (eps_yy * eps_yy) + eta_share * eps_xx).real}
        xy_cotangent = (-2.0 * eps_xy * (along_xx / eps_yy + along_yy / eps_xx)
                        - eta_cotangent * (eps_xx * eps_yy + squared) / determinant_squared)
        del squared, determinant_squared, eta_share
    if off_diagonal:
        gradients['eps_xy'] = 2.0 * xy_cotangent.real * _assembled_xy_cells(physics, wall_rows, shape)
    return gradients


def sensitivity_density(domain, gradients):
    """float32 (nx, ny) sum of the cell gradients per cell area [W/m^3] (the eps_sensitivity field)."""
    widths_x, widths_y = domain.cell_widths()
    cell_areas = domain.xp.asarray(np.outer(widths_x, widths_y))
    return (sum(gradients.values()) / cell_areas).astype(DTYPE)


def _summed_face_currents(physics):
    """(J_x on hi-y, J_y on hi-x faces) summed over the sources with face currents (as FDFDPhysics.face_curls adds
    them; a None part = none), None without any."""
    totals = [None, None]
    for source in physics.sources:
        if not hasattr(source, 'face_currents'):
            continue
        for index, part in enumerate(source.face_currents(physics, physics.system)):
            if part is not None:
                totals[index] = part if totals[index] is None else totals[index] + part
    return None if all(part is None for part in totals) else tuple(totals)


def eps_sensitivity(physics, wall_rows) -> EpsSensitivity:
    """EpsSensitivity of the objective monitors of an FDFDPhysics at its last solve (wall_rows = its outer wall rows; hz
    also dJ/d eps_xy); one physics.solve_adjoint unless every weight is 0."""
    grid = physics.domain.grid
    xp = physics.domain.xp
    shape = (grid.nx, grid.ny)
    objectives = [monitor for monitor in physics.monitors if is_objective(monitor)]
    system = physics.system
    hz = physics.polarization == 'hz'
    if not objectives or physics.solution is None or system is None:
        return EpsSensitivity(0.0, None, {}, xp.zeros(shape, dtype=DTYPE))
    total, cell_part, face_parts, weighted = objective_cotangents(physics)
    if not weighted:
        return EpsSensitivity(total, float(system.omega), {name: xp.zeros(shape, dtype=xp.float64)
                                                           for name in physics.permittivity_arrays},
                              xp.zeros(shape, dtype=DTYPE), xp.zeros(shape, dtype=DTYPE) if hz else None)
    source = adjoint_source(physics, cell_part, face_parts, wall_rows)
    del cell_part
    adjoint = physics.solve_adjoint(source).reshape(shape)
    del source
    # hz: the objective face fields t = scale V (D u + J) and the rhs D_e V J of face-current sources depend on eps
    # through V as well (ez face fields read mu only)
    gradients = permittivity_gradients(physics, adjoint, wall_rows,
                                       face_currents=_summed_face_currents(physics) if hz else None,
                                       face_cotangents=face_parts if hz else None, off_diagonal=hz)
    del face_parts
    off_diagonal_gradient = gradients.pop('eps_xy', None)
    off_diagonal_density = (None if off_diagonal_gradient is None
                            else sensitivity_density(physics.domain, {'eps_xy': off_diagonal_gradient}))
    del off_diagonal_gradient
    return EpsSensitivity(total, float(system.omega), gradients, sensitivity_density(physics.domain, gradients),
                          off_diagonal_density)
