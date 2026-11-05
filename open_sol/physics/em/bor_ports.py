"""Vector line-mode ports of the body-of-revolution FDFD (em_fdfd_bor, symmetry 2.3): guided modes ~ exp(-j beta z) of a
z = const line (from the axis or annular), the one-sided mode source, the mode monitor and results; host numpy / scipy."""
# Slices: slice s = E_r (edge k, row s), u (cell i, row s), E_z (cell i, z-edge s = the lo edge of row s), so the BOR
# operator of a z-invariant section on uniform rows (width h) is block tridiagonal over slices, A_- X_{s-1} + A_0 X_s +
# A_+ X_{s+1} = 0 with A_+ = A_-^T (complex symmetric operator). The blocks are slice 1 of assemble_bor on a three-row
# strip of the line cells (pec z ends; the line's own r end), so a mode is an exact discrete solution of the 2D operator
# of a straight guide. Modes X_s = psi^s X, psi = exp(-j beta h): (A_- + psi A_0 + psi^2 A_+) X = 0, linearised and
# solved by shift-invert ARPACK near psi of the largest line index. Mirror z -> -z: the backward partner of X is
# (E_r, u, -psi E_z) at 1 / psi. Power toward +z between slices s, s+1: pi / (omega mu0) Im(X_s^H A_+ X_{s+1})
# (body_of_revolution.box_flux, exact); modes carry 1 W. Reciprocity form W(G, F) = G_s^T A_+ F_{s+1} - G_{s+1}^T A_-
# F_s: independent of s for source-free solutions, 0 between modes unless partners, so a_+ = W(F_-, F) / W(F_-, F_+)
# and a_- = W(F_+, F) / W(F_+, F_-) from the field on slices q, q+1. Source (one-sided TFSF as fdfd_ports): b = A (Q u)
# - Q (A u), Q = slices >= q (positive) or <= q (negative), u = the incident mode on the two slices across the
# interface. Port lines start on the axis or at an inner cell edge (annular: the strip cells below it are PEC, so the
# start is a pec wall there and the line keys of those cells drop out); an end on the last window cell continues
# through the r_hi cpml (or ends on its pec / pmc wall), an end inside the domain is a pec wall on that cell edge. A
# metal end is exact when the model's cells beyond it are PEC too (e.g. the inner conductor of a coaxial guide).
# Vertical (r = const) ports are not possible this way: the operator is not invariant along r (the metric), so there
# is no psi^s ansatz; they would need per-radius block recurrences (discrete Hankel waves of the z slab modes).
import cmath
import math
from dataclasses import dataclass

import numpy as np

from ...core.constants import C0, MU0
from ...solvers.arpack_shift_invert import MAX_RESTARTS
from ...solvers.sparse_direct import factorize_host
from .body_of_revolution import _WALL_KINDS, BORMaterials, _full_offsets, assemble_bor
from .boundaries import BCType, BoundaryConfig
from .results import _DIRECTIONS, _MODE_TABLE_COLUMNS, _MODE_TABLE_UNITS, mode_power_series

_STRIP_ROWS = 3
_TARGET_INDEX_MARGIN = 1e-4     # shift at (1 + this) x the largest line index: never exactly on a mode
_PML_MODE_FRACTION = 0.5        # modes with more of their |X|^2 V on PML-stretched line points are dropped
_EDGE_FIELD_WARN = 1e-2         # mode |field| at a metal (truncated) line end above this share of its peak: warning
_THETA_FLOOR = 1e-12
_START_VECTOR_SEED = 11
_UNIFORM_TOLERANCE = 1e-2       # relative row-width change around a port line before the warning
_MATERIAL_TOLERANCE = 1e-9      # relative eps / mu change between the port rows before the warning
_DB_PER_CM = 0.2 / math.log(10.0)   # loss_db_per_cm = -_DB_PER_CM x Im beta [1/m] (power ~ exp(2 Im beta z))
_CPML_ATTRIBUTES = ('cpml_npml', 'cpml_order', 'cpml_sigma_factor', 'cpml_kappa_max', 'cpml_alpha_max')
_COMPONENT_CODES = {'E_r': 0, 'u': 1, 'E_z': 2}


@dataclass(frozen=True, eq=False)
class PortMode:
    """One guided mode: psi = exp(-j beta h) per row, beta [1/m], n_eff, complex slice vector carrying 1 W toward +z,
    its backward partner (1 W toward -z), share of |X|^2 V in PML points, |X| at a metal line end / peak, n_group."""
    psi: complex
    beta: complex
    n_eff: complex
    vector: np.ndarray
    backward: np.ndarray
    pml_fraction: float
    edge_field: float
    n_group: float


@dataclass(frozen=True, eq=False)
class PortModes:
    """Guided modes of one port line at one omega (descending Re n_eff): slice keys (component codes 0 E_r / 1 u / 2
    E_z, radial index), strip blocks A_- (lower), A_+ (upper), row width h [m], line end cell n_line, first line cell
    start (> 0: annular, metal inner end), truncated (metal outer end), the three-row strip BORSystem."""
    omega: float
    azimuthal_order: int
    row_width: float
    n_line: int
    start: int
    truncated: bool
    codes: np.ndarray
    indices: np.ndarray
    lower: object
    upper: object
    modes: list
    strip: object = None


def _full_indices(shape, codes, indices, slice_index):
    """Full E-vector indices of the slice keys at slice `slice_index` on an (nr, nz) grid."""
    nr, nz = shape
    n_er, n_u, _ = _full_offsets(shape)
    return np.where(codes == 0, indices * nz + slice_index,
                    np.where(codes == 1, n_er + indices * nz + slice_index,
                             n_er + n_u + indices * (nz + 1) + slice_index))


def full_slice_indices(shape, port_modes, slice_index):
    """Full E-vector indices on an (nr, nz) grid of the port's slice keys at slice `slice_index`."""
    return _full_indices(shape, port_modes.codes, port_modes.indices, slice_index)


def _inverse_index(system):
    """int (full_size,) active position of every full E-vector point, -1 when inactive."""
    inverse = np.full(system.full_size, -1, dtype=np.int64)
    inverse[system.unknown_index] = np.arange(system.n_unknowns)
    return inverse


def slice_positions(system, port_modes, slice_index):
    """int (n_s,) active positions in `system` of the port's slice keys at slice `slice_index` (-1: inactive there)."""
    if not 0 <= slice_index < system.shape[1]:
        raise ValueError(f'port slice {slice_index} is outside the {system.shape[1]} z rows')
    if port_modes.n_line > system.shape[0]:
        raise ValueError(f'port line of {port_modes.n_line} r cells exceeds the {system.shape[0]} grid cells')
    return _inverse_index(system)[_full_indices(system.shape, port_modes.codes, port_modes.indices, slice_index)]


def unknown_slices(system):
    """int (n,) slice of every active unknown (row of E_r / u, z-edge of E_z)."""
    nr, nz = system.shape
    n_er, n_u, _ = _full_offsets(system.shape)
    full = system.unknown_index
    return np.where(full < n_er, full % nz, np.where(full < n_er + n_u, (full - n_er) % nz,
                                                      (full - n_er - n_u) % (nz + 1)))


def _target_index(line_materials, n_window):
    """Largest Re sqrt(eps_c mu) over the components of the non-PEC window cells of the line."""
    keep = ~np.asarray(line_materials.pec[:n_window, 0], dtype=bool)
    if not np.any(keep):
        raise ValueError('port line: every window cell of the line is PEC, so it guides no mode')
    mu = line_materials.mu[:n_window, 0][keep]
    return max(float(np.max(np.sqrt(line_materials.component(name)[:n_window, 0][keep] * mu).real))
               for name in ('r', 'phi', 'z'))


def solve_port_modes(dr_cells, row_width, end_kind, boundary_config, pml_dx, line_materials, omega, azimuthal_order,
                     n_modes, n_window, truncated=False, start=0):
    """PortModes of a line over r cells [start, len(dr_cells)) (dr_cells from the axis; start > 0: annular, the
    cells below are PEC) with materials BORMaterials (n, 1) (the line row from the axis), end 'cpml' (boundary_config's
    grading) | 'pec' | 'pmc', rows of width row_width [m]; up to n_modes modes, the PML share over the first n_window
    cells."""
    import scipy.sparse
    import scipy.sparse.linalg
    dr_cells = np.asarray(dr_cells, dtype=np.float64)
    n_line, row_width, omega, start = dr_cells.size, float(row_width), float(omega), int(start)
    if end_kind not in ('cpml', 'pec', 'pmc'):
        raise ValueError(f"port line end must be 'cpml', 'pec' or 'pmc', got {end_kind!r}")
    if not 0 <= start < n_line:
        raise ValueError(f'port line start cell {start} must lie inside its {n_line} r cells')
    inner_backed = bool(np.all(np.asarray(line_materials.pec, dtype=bool)[:start]))   # PEC in the model below start
    if start > 0:   # the annular line's inner end: a pec wall on the edge of cell `start`
        inner_pec = np.asarray(line_materials.pec, dtype=bool).copy()
        inner_pec[:start] = True
        line_materials = BORMaterials(eps=line_materials.eps, mu=line_materials.mu, pec=inner_pec,
                                      eps_rr=line_materials.eps_rr, eps_zz=line_materials.eps_zz)
    strip_config = BoundaryConfig(x_lo='pmc', x_hi=end_kind, y_lo='pec', y_hi='pec')
    if end_kind == 'cpml':
        for name in _CPML_ATTRIBUTES:
            setattr(strip_config, name, getattr(boundary_config, name))

    def rows(values):
        return None if values is None else np.repeat(np.asarray(values)[:, :1], _STRIP_ROWS, axis=1)

    strip_materials = BORMaterials(eps=rows(line_materials.eps), mu=rows(line_materials.mu),
                                   pec=rows(line_materials.pec), eps_rr=rows(line_materials.eps_rr),
                                   eps_zz=rows(line_materials.eps_zz))
    strip = assemble_bor((dr_cells, np.full(_STRIP_ROWS, row_width)), strip_config, pml_dx, strip_materials, omega,
                         azimuthal_order, window=(slice(0, int(n_window)), slice(0, _STRIP_ROWS)))
    codes = np.concatenate((np.zeros(n_line + 1, dtype=np.int64), np.ones(n_line, dtype=np.int64),
                            np.full(n_line, 2, dtype=np.int64)))
    indices = np.concatenate((np.arange(n_line + 1), np.arange(n_line), np.arange(n_line)))
    inverse = _inverse_index(strip)
    positions = [inverse[_full_indices(strip.shape, codes, indices, s)] for s in range(_STRIP_ROWS)]
    keep = positions[1] >= 0
    if np.any(positions[0][keep] < 0) or np.any(positions[2][keep] < 0):
        raise ValueError('port strip: the active unknowns differ between rows (internal layout error)')
    codes, indices = codes[keep], indices[keep]
    lower_cols, centre, upper_cols = (position[keep] for position in positions)
    operator_rows = strip.operator.tocsr()[centre]
    lower = operator_rows[:, lower_cols].tocsr()
    middle = operator_rows[:, centre].tocsr()
    upper = operator_rows[:, upper_cols].tocsr()
    n_slice = int(centre.size)
    if n_slice < 2:
        raise ValueError(f'port line: only {n_slice} unknowns per row remain outside PEC and the axis rules')

    k0 = omega / C0
    shift = cmath.exp(-1j * k0 * _target_index(line_materials, int(n_window)) * (1.0 + _TARGET_INDEX_MARGIN)
                      * row_width)
    identity = scipy.sparse.identity(n_slice, dtype=np.complex128, format='csr')
    pencil_a = scipy.sparse.bmat([[None, identity], [-lower, -middle]], format='csc')
    pencil_b = scipy.sparse.bmat([[identity, None], [None, upper]], format='csc')
    factor = factorize_host(pencil_a - shift * pencil_b)
    linear_operator = scipy.sparse.linalg.LinearOperator(
        (2 * n_slice, 2 * n_slice), matvec=lambda vector: factor.solve_host(pencil_b @ vector), dtype=np.complex128)
    n_wanted = max(1, min(2 * int(n_modes) + 8, 2 * n_slice - 2))
    random = np.random.default_rng(_START_VECTOR_SEED)
    start_vector = random.standard_normal(2 * n_slice) + 1j * random.standard_normal(2 * n_slice)
    try:
        theta, vectors = scipy.sparse.linalg.eigs(linear_operator, k=n_wanted, which='LM', v0=start_vector,
                                                  maxiter=MAX_RESTARTS)
    except scipy.sparse.linalg.ArpackNoConvergence as exc:
        theta, vectors = exc.eigenvalues, exc.eigenvectors
    theta = np.asarray(theta, dtype=np.complex128)
    usable = np.isfinite(theta) & (np.abs(theta) > _THETA_FLOOR * float(np.max(np.abs(theta), initial=0.0)))

    mass = strip.mass[centre]
    weights = strip.energy_weights[centre]
    outside = ~strip.in_window[centre]
    end_points = indices >= n_line - 1   # next to the outer end; and next to an inner metal end the model lacks
    if not inner_backed:
        end_points |= indices <= start
    power_scale = math.pi / (omega * MU0)
    modes = []
    for column in np.flatnonzero(usable):
        psi = complex(shift + 1.0 / theta[column])
        if not (cmath.phase(psi) < 0.0 and abs(psi) <= 1.0 + 1e-9):   # forward, not growing
            continue
        vector = np.asarray(vectors[:n_slice, column], dtype=np.complex128)
        power = power_scale * float((psi * np.vdot(vector, upper @ vector)).imag)
        if not power > 0.0:
            continue
        vector = vector / math.sqrt(power)
        peak = int(np.argmax(np.abs(vector)))
        vector = vector * (abs(vector[peak]) / vector[peak])
        backward = vector.copy()
        backward[codes == 2] *= -psi
        energy = np.abs(vector) ** 2 * weights
        fraction = float(np.sum(energy[outside]) / np.sum(energy)) if np.sum(energy) > 0.0 else 1.0
        if fraction > _PML_MODE_FRACTION:
            continue
        beta = complex(-cmath.phase(psi), math.log(abs(psi))) / row_width
        # n_group: Hellmann-Feynman on the pencil, left vector = the backward partner; only the mass term depends on
        # omega here (material dispersion and the PML stretch are held at omega)
        d_pencil_d_psi = middle @ vector + 2.0 * psi * (upper @ vector)
        d_pencil_d_omega = -psi * 2.0 * omega / C0 ** 2 * mass * vector
        d_psi = -(backward @ d_pencil_d_omega) / (backward @ d_pencil_d_psi)
        n_group = float((C0 * 1j * d_psi / (row_width * psi)).real)
        edge_field = float(np.max(np.abs(vector[end_points])) / np.max(np.abs(vector)))
        modes.append(PortMode(psi=psi, beta=beta, n_eff=beta / k0, vector=vector, backward=backward,
                              pml_fraction=fraction, edge_field=edge_field, n_group=n_group))
    modes.sort(key=lambda mode: -mode.n_eff.real)
    return PortModes(omega=omega, azimuthal_order=int(azimuthal_order), row_width=row_width, n_line=n_line,
                     start=start, truncated=bool(truncated) or not inner_backed, codes=codes, indices=indices,
                     lower=lower, upper=upper, modes=modes[:max(int(n_modes), 1)], strip=strip)


def mode_source_rhs(system, port_modes, mode, row, direction, amplitude):
    """Active rhs (complex128 (n,)) launching `amplitude` x the 1 W mode from line row `row` toward +z ('positive':
    total field on slices >= row) or -z ('negative': slices <= row); one-sided TFSF of the module header."""
    if direction not in _DIRECTIONS:
        raise ValueError(f'mode source direction must be one of {_DIRECTIONS}, got {direction!r}')
    positive = direction == 'positive'
    upstream = row - 1 if positive else row + 1
    vector = mode.vector if positive else mode.backward
    inside_positions = slice_positions(system, port_modes, row)
    upstream_positions = slice_positions(system, port_modes, upstream)
    incident = np.zeros(system.n_unknowns, dtype=np.complex128)
    keep = inside_positions >= 0
    incident[inside_positions[keep]] = amplitude * vector[keep]
    inside_product = system.operator @ incident
    keep = upstream_positions >= 0
    incident[upstream_positions[keep]] = amplitude * vector[keep] / mode.psi   # one slice upstream: 1 / psi
    slices = unknown_slices(system)
    total_field = slices >= row if positive else slices <= row
    return inside_product - total_field * (system.operator @ incident)


def mode_amplitudes(system, port_modes, mode, solution, row):
    """(a_positive, a_negative) of the 1 W mode at slice `row` from the solution on slices row and row + 1 (the
    reciprocity form of the module header; other modes and the backward wave drop out)."""
    def sampled(slice_index):
        positions = slice_positions(system, port_modes, slice_index)
        values = np.zeros(positions.size, dtype=np.complex128)
        values[positions >= 0] = np.asarray(solution)[positions[positions >= 0]]
        return values

    return slice_amplitudes(port_modes, mode, sampled(row), sampled(row + 1))


def slice_amplitudes(port_modes, mode, here, beyond):
    """(a_positive, a_negative) of the 1 W mode from a field's slice-key values on a slice and the next one (the
    reciprocity form of the module header)."""
    lower, upper = port_modes.lower, port_modes.upper
    forward, backward, psi = mode.vector, mode.backward, mode.psi

    def reciprocity(g_here, g_beyond, f_here, f_beyond):
        return complex(g_here @ (upper @ f_beyond) - g_beyond @ (lower @ f_here))

    a_positive = (reciprocity(backward, backward / psi, here, beyond)
                  / reciprocity(backward, backward / psi, forward, psi * forward))
    a_negative = (reciprocity(forward, psi * forward, here, beyond)
                  / reciprocity(forward, psi * forward, backward, backward / psi))
    return a_positive, a_negative


def track_mode(previous, port_modes):
    """Index of the mode of port_modes overlapping `previous` (PortMode on the same keys) most, None when the keys
    changed or no mode is left."""
    if not port_modes.modes or previous.vector.size != port_modes.codes.size:
        return None
    overlaps = [abs(np.vdot(previous.vector, mode.vector)) / (np.linalg.norm(previous.vector)
                                                               * np.linalg.norm(mode.vector))
                for mode in port_modes.modes]
    return int(np.argmax(overlaps))


def aligned_mode(mode, previous):
    """`mode` with its phase turned so that previous^H mode is real positive (both directions)."""
    overlap = np.vdot(previous.vector, mode.vector)
    if abs(overlap) == 0.0:
        return mode
    turn = abs(overlap) / overlap
    return PortMode(psi=mode.psi, beta=mode.beta, n_eff=mode.n_eff, vector=mode.vector * turn,
                    backward=mode.backward * turn, pml_fraction=mode.pml_fraction, edge_field=mode.edge_field,
                    n_group=mode.n_group)


def mode_table_rows(port_modes, selected_index):
    """Rows [mode number, Re n_eff, Im n_eff, loss dB/cm, n_group, edge field, selected] of a PortModes."""
    return [[number, float(mode.n_eff.real), float(mode.n_eff.imag), -_DB_PER_CM * float(mode.beta.imag),
             mode.n_group, mode.edge_field, number - 1 == selected_index]
            for number, mode in enumerate(port_modes.modes, start=1)]


# ---- runtime ports of BORPhysics (solvers/fdfd_bor; fdtd_bor_ports for em_fdtd_bor) ----
def port_geometry(physics, row, span_cells, label):
    """(start cell, n_line, end kind, window cells on the line, truncated) of a port line in row `row` over r cells
    span_cells = [start, end) (start > 0: annular); ValueError unless it leaves the rows its slices need."""
    start, end = int(span_cells[0]), int(span_cells[1])
    nr, nz = physics.domain.grid.nx, physics.domain.grid.ny
    window_r = physics.domain.physical_window()[0]
    if not 1 <= row <= nz - 2:
        raise ValueError(f'{label}: row {row} needs a z row on both sides inside the {nz} grid rows')
    if not 0 <= start < end:
        raise ValueError(f'{label}: the line covers no r cell (cells [{start}, {end}))')
    config = physics.boundary_config
    if end >= window_r.stop:
        end_kind = 'cpml' if config.x_hi == BCType.CPML else _WALL_KINDS[config.x_hi]
        return start, nr, end_kind, window_r.stop, False
    return start, end, 'pec', end, True


def _port_warnings(physics, row, rows, n_line, label):
    """Warnings for a non-uniform row width or z-varying line materials over `rows` around port row `row`."""
    warnings = []
    widths = physics.domain.cell_widths()[1]
    if any(abs(widths[other] / widths[row] - 1.0) > _UNIFORM_TOLERANCE for other in rows):
        warnings.append(f'{label}: the row widths around the line differ by more than {_UNIFORM_TOLERANCE:.0%}; the '
                        'port mode is exact only on uniform rows (make the mesh uniform there)')
    materials = physics.last_materials
    reference = [materials.component(name)[:n_line, row] for name in ('r', 'phi', 'z')] + [materials.mu[:n_line, row]]
    for other in rows:
        values = [materials.component(name)[:n_line, other] for name in ('r', 'phi', 'z')] + [materials.mu[:n_line,
                                                                                                         other]]
        changed = any(np.any(np.abs(value - ref) > _MATERIAL_TOLERANCE * np.abs(ref)) for value, ref in
                      zip(values, reference))
        if changed or np.any(materials.pec[:n_line, other] != materials.pec[:n_line, row]):
            warnings.append(f'{label}: the materials of row {other} differ from the line row: keep the guide straight '
                            '(invariant along z) for a cell on each side of the line, else the launch / '
                            'decomposition is not exact')
            break
    return warnings


class _BORPort:
    """Shared mode solve of BORModeSource / BORModeMonitor: re-solved per omega and per assembly of the physics."""

    def __init__(self, physics, row, span_cells, mode_number, label, rows):
        if isinstance(mode_number, bool) or int(mode_number) != mode_number or mode_number < 1:
            raise ValueError(f'{label}: mode_number must be an integer >= 1, got {mode_number!r}')
        self.label = label
        self.row = int(row)
        self.mode_number = int(mode_number)
        self.start, self.n_line, self.end_kind, self.n_window, self.truncated = port_geometry(
            physics, self.row, span_cells, label)
        self._rows = rows
        self.port_modes = None
        self.mode = None
        self.selected_index = None
        self.warnings = []
        self._solved_key = None   # (omega, azimuthal order, systems built) of the last mode solve

    def ensure_modes(self, physics, system):
        """Solve the line modes when omega, the order or the system changed; the first solve selects mode_number, later
        ones track the previous mode by overlap (phase aligned; a new order selects mode_number afresh)."""
        key = (float(system.omega), system.azimuthal_order, physics.assembly_count)
        if key == self._solved_key:
            return
        if self.port_modes is not None and self.port_modes.azimuthal_order != system.azimuthal_order:
            self.mode = None   # another order: the previous mode is no reference
        dr_cells = physics.domain.cell_widths()[0][:self.n_line]
        row_width = float(physics.domain.cell_widths()[1][self.row])
        materials = physics.last_materials

        def line(values):
            return None if values is None else np.asarray(values)[:self.n_line, self.row:self.row + 1]

        line_materials = BORMaterials(eps=line(materials.eps), mu=line(materials.mu), pec=line(materials.pec),
                                      eps_rr=line(materials.eps_rr), eps_zz=line(materials.eps_zz))
        port_modes = solve_port_modes(dr_cells, row_width, self.end_kind, physics.boundary_config, physics.pml_dx,
                                      line_materials, system.omega, system.azimuthal_order,
                                      max(self.mode_number + 2, 4), self.n_window, truncated=self.truncated,
                                      start=self.start)
        index = None if self.mode is None else track_mode(self.mode, port_modes)
        if index is None:
            if len(port_modes.modes) < self.mode_number:
                raise ValueError(f'{self.label}: {len(port_modes.modes)} guided mode(s) found at lambda = '
                                 f'{2.0 * math.pi * C0 / system.omega * 1e9:.6g} nm, mode_number '
                                 f'{self.mode_number} does not exist (azimuthal order {system.azimuthal_order})')
            index = self.mode_number - 1
            mode = port_modes.modes[index]
        else:
            mode = aligned_mode(port_modes.modes[index], self.mode)
        self.selected_index = index
        warnings = _port_warnings(physics, self.row, self._rows, self.n_line, self.label)
        if port_modes.truncated and mode.edge_field > _EDGE_FIELD_WARN:
            warnings.append(f'{self.label}: the mode still has {mode.edge_field:.2g} of its peak field at a metal '
                            'end of the line: extend the line into the cladding (or toward the axis), or make the '
                            'cells beyond an inner end pec')
        self.warnings = warnings
        for warning in warnings:
            if warning not in physics.runtime_warnings:
                physics.runtime_warnings.append(warning)
        self.port_modes, self.mode, self._solved_key = port_modes, mode, key

    def mode_table(self):
        """Rows [mode number, Re n_eff, Im n_eff, loss dB/cm, n_group, edge field, selected] of the last solve."""
        if self.port_modes is None:
            raise ValueError(f'mode not solved yet: {self.label} (run a stationary or frequency_domain study)')
        return mode_table_rows(self.port_modes, self.selected_index)


class BORModeSource(_BORPort):
    """One-sided port source on a z = const line (from the axis or annular): `power` [W] of mode `mode_number` toward
    +z ('positive') or -z ('negative') at `phase` [deg], amplitude referred to the line row."""

    def __init__(self, physics, row, span_cells, power, mode_number, direction, phase, label):
        if direction not in _DIRECTIONS:
            raise ValueError(f'{label} direction must be one of {_DIRECTIONS}, got {direction!r}')
        if not (math.isfinite(float(power)) and float(power) >= 0.0):
            raise ValueError(f'{label} power must be finite and >= 0 [W], got {power!r}')
        super().__init__(physics, row, span_cells, mode_number, label,
                         (row - 1, row + 1) if direction == 'positive' else (row + 1, row - 1))
        self.power = float(power)
        self.direction = direction
        self.phase = float(phase)
        self.incident_amplitude = math.sqrt(self.power) * cmath.exp(1j * math.radians(self.phase))

    def rhs(self, physics, system):
        self.ensure_modes(physics, system)
        return mode_source_rhs(system, self.port_modes, self.mode, self.row, self.direction, self.incident_amplitude)


class BORModeMonitor(_BORPort):
    """Port monitor on a z = const line (from the axis or annular): amplitudes (a_positive, a_negative) of mode
    `mode_number` toward +z / -z per driven solve (|a|^2 in W) from the solution on the line row and the row above."""

    def __init__(self, physics, row, span_cells, mode_number, label):
        super().__init__(physics, row, span_cells, mode_number, label, (row + 1,))
        self._records = {}   # omega -> (a_positive, a_negative, n_eff)

    def record(self, physics):
        system = physics.system
        self.ensure_modes(physics, system)
        a_positive, a_negative = mode_amplitudes(system, self.port_modes, self.mode, physics.solution, self.row)
        self._records[float(system.omega)] = (a_positive, a_negative, complex(self.mode.n_eff))

    def recorded(self):
        """(omega float64 in ascending wavelength order, a_positive, a_negative, n_eff complex128); ValueError before
        the first solve."""
        if not self._records:
            raise ValueError(f'not solved yet: {self.label} has no driven solve (stationary or frequency_domain study)')
        omegas = sorted(self._records, reverse=True)
        values = np.array([self._records[omega] for omega in omegas], dtype=np.complex128)
        return np.array(omegas, dtype=np.float64), values[:, 0], values[:, 1], values[:, 2]


def bor_mode_source_result(source) -> dict:
    """'table' of a BORModeSource at its last solved wavelength: one row per mode found; extras wavelength [m], power
    [W], direction, mode_number, phase [deg]; ValueError before the solve."""
    rows = source.mode_table()
    return {'kind': 'table', 'columns': list(_MODE_TABLE_COLUMNS), 'units': list(_MODE_TABLE_UNITS), 'rows': rows,
            'wavelength': float(2.0 * math.pi * C0 / source.port_modes.omega), 'power': source.power,
            'direction': source.direction, 'mode_number': source.mode_number, 'phase': source.phase}


def bor_mode_monitor_result(monitor, physics) -> dict:
    """'xy' mode powers |a|^2 [W] per direction vs ascending wavelength; with exactly one BORModeSource in `physics`
    also fraction_* / phase_* of a / (sqrt(P) e^{j phase}); extras n_eff (real part) / n_eff_imag, mode_number, normalized."""
    omegas, a_positive, a_negative, n_eff = monitor.recorded()
    frequency = omegas / (2.0 * math.pi)
    wavelength = C0 / frequency
    mode_sources = [source for source in physics.sources if isinstance(source, BORModeSource)]
    normalized = len(mode_sources) == 1
    series, series_units = mode_power_series(dict(zip(_DIRECTIONS, (a_positive, a_negative))), 1.0, 'W',
                                             incident=mode_sources[0].incident_amplitude if normalized else None)
    return {'kind': 'xy', 'x': wavelength, 'x_label': 'Wavelength', 'x_unit': 'm', 'y_label': 'Mode power',
            'series': series, 'series_units': series_units, 'frequency': frequency, 'wavelength': wavelength.copy(),
            'n_eff': n_eff.real.copy(), 'n_eff_imag': n_eff.imag.copy(), 'mode_number': monitor.mode_number,
            'normalized': normalized}
