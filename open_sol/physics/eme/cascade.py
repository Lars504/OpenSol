"""EME S-matrix cascade (host numpy): Redheffer star products of sections (diag(exp(-j k L))) and interfaces, built
left to right, then the partial cascades from the hi end, so internal amplitudes stay stable with evanescent modes;
star powers of a unit cell (periodic blocks), its Bloch modes and the per-period amplitudes of the adjoint."""
# SMatrix ports: 1 = lo side, 2 = hi side; b1 = s11 a1 + s12 a2, b2 = s21 a1 + s22 a2 (a incident, b outgoing, in the
# bilinear-normalised mode amplitudes). Per section k the builder keeps L21 / L22 of the cascade from the lo window
# plane to the lo end of section k and R11 / R12 of the cascade from the hi end of section k to the hi window plane
# (4 N^2 per section); the interfaces live only until finish().
from dataclasses import dataclass

import numpy as np

PORT_FACES = ('lo', 'hi')


@dataclass(frozen=True, eq=False)
class SMatrix:
    s11: np.ndarray   # (n1, n1) reflection at port 1
    s12: np.ndarray   # (n1, n2) port 2 -> port 1
    s21: np.ndarray   # (n2, n1) port 1 -> port 2
    s22: np.ndarray   # (n2, n2) reflection at port 2


def identity_smatrix(n_modes):
    """Zero-length connection of n_modes modes."""
    zeros = np.zeros((n_modes, n_modes), dtype=np.complex128)
    eye = np.eye(n_modes, dtype=np.complex128)
    return SMatrix(zeros, eye, eye.copy(), zeros.copy())


def _solve_blocks(matrix, first, second):
    """(matrix^-1 first, matrix^-1 second) from one LU: the right-hand blocks side by side, leading batch axes
    broadcast."""
    batch = np.broadcast_shapes(matrix.shape[:-2], first.shape[:-2], second.shape[:-2])
    solved = np.linalg.solve(matrix, np.concatenate((np.broadcast_to(first, batch + first.shape[-2:]),
                                                     np.broadcast_to(second, batch + second.shape[-2:])), axis=-1))
    n_first = first.shape[-1]
    return solved[..., :n_first], solved[..., n_first:]


def star(left, right):
    """Redheffer star product: left (ports 1, 2) followed by right (ports 2, 3) -> ports 1, 3; blocks may carry
    leading batch axes (broadcast; the length sweep stacks its stretch factors)."""
    n_middle = left.s22.shape[-1]
    if right.s11.shape[-2] != n_middle:
        raise ValueError(f'star: {n_middle} modes on the left of the junction, {right.s11.shape[-2]} on the right')
    eye = np.eye(n_middle, dtype=np.complex128)
    toward_left = eye - right.s11 @ left.s22    # (I - B11 A22)
    toward_right = eye - left.s22 @ right.s11   # (I - A22 B11)
    reflected, passed_left = _solve_blocks(toward_left, right.s11 @ left.s21, right.s12)
    passed_right, returned = _solve_blocks(toward_right, left.s21, left.s22 @ right.s12)
    return SMatrix(s11=left.s11 + left.s12 @ reflected, s12=left.s12 @ passed_left,
                   s21=right.s21 @ passed_right, s22=right.s22 + right.s21 @ returned)


def append_phase(smatrix, phase):
    """smatrix followed by a section of propagation factors phase (..., N) = exp(-j k L)."""
    return SMatrix(smatrix.s11, smatrix.s12 * phase[..., None, :], phase[..., :, None] * smatrix.s21,
                   phase[..., :, None] * smatrix.s22 * phase[..., None, :])


def _prepend_phase(phase, smatrix):
    """A section of propagation factors phase (..., N) followed by smatrix (the star product with a zero-reflection
    section, written out)."""
    return SMatrix(phase[..., :, None] * smatrix.s11 * phase[..., None, :], phase[..., :, None] * smatrix.s12,
                   smatrix.s21 * phase[..., None, :], smatrix.s22)


def smatrix_power(smatrix, count):
    """smatrix cascaded `count` (>= 0) times with itself (star products by repeated squaring); equal port mode counts."""
    count = int(count)
    if count < 0 or smatrix.s11.shape[-2:] != smatrix.s22.shape[-2:]:
        raise ValueError(f'smatrix_power needs count >= 0 and equal port mode counts, got {count}, '
                         f'{smatrix.s11.shape[-1]} / {smatrix.s22.shape[-1]} modes')
    result, square = None, smatrix
    while count:
        if count & 1:
            result = square if result is None else star(result, square)
        count >>= 1
        if count:
            square = star(square, square)
    return identity_smatrix(smatrix.s11.shape[-1]) if result is None else result


def unit_cell(phases, interfaces, first, last, junction):
    """SMatrix of one period (the junction last -> first cell section, then sections first..last) between the hi ends
    of the cell's last section; interfaces[k] = the interface after section k (list or dict)."""
    cell = total_smatrix(list(phases[first:last + 1]), [interfaces[k] for k in range(first, last)])
    return star(junction, cell)


def repeated_link(cell, n_periods, exit_interface):
    """Interface after a drawn unit cell standing for n_periods - 1 more periods: cell^(n - 1) then exit_interface."""
    return star(smatrix_power(cell, n_periods - 1), exit_interface)


def bloch_modes(cell):
    """(multipliers lam (M,), forward amplitudes a (N, M), backward b (N, M)) of a unit-cell SMatrix with the same N
    modes at both ports: the fields one period on are lam times these (b2 = lam a, a2 = lam b), lam = exp(-j K period);
    the generalised eigenproblem [[S21, 0], [S11, -I]] v = lam [[I, -S22], [0, -S12]] v, infinite lam dropped."""
    n_modes = cell.s11.shape[0]
    if cell.s22.shape[0] != n_modes:
        raise ValueError(f'bloch_modes: {n_modes} modes at port 1, {cell.s22.shape[0]} at port 2')
    eye = np.eye(n_modes, dtype=np.complex128)
    zero = np.zeros((n_modes, n_modes), dtype=np.complex128)
    left = np.block([[cell.s21, zero], [cell.s11, -eye]])
    right = np.block([[eye, -cell.s22], [zero, -cell.s12]])
    try:
        from scipy.linalg import eig
        values, vectors = eig(left, right)
    except ImportError:
        values, vectors = np.linalg.eig(np.linalg.solve(right, left))
    finite = np.isfinite(values) & (values != 0)
    return values[finite], vectors[:n_modes, finite], vectors[n_modes:, finite]


def forward_bloch(multipliers, forward, backward, power, tolerance=1e-8):
    """(M,) bool: Bloch modes travelling toward +axis, decaying (|lam| < 1) or, on the unit circle, with net forward
    power sum p (|a|^2 - |b|^2) > 0 (p = the cell modes' unit-amplitude power)."""
    magnitude = np.abs(multipliers)
    flux = np.sum(np.asarray(power)[:, None] * (np.abs(forward) ** 2 - np.abs(backward) ** 2), axis=0)
    return (magnitude < 1.0 - tolerance) | ((np.abs(magnitude - 1.0) <= tolerance) & (flux > 0.0))


def total_smatrix(phases, interfaces):
    """Total SMatrix of sections with propagation factors phases[k] (..., N_k) and the interfaces between them (no
    partial cascades: the length-sweep re-cascade, batched over leading axes)."""
    if len(interfaces) != len(phases) - 1:
        raise ValueError(f'total_smatrix: {len(phases)} sections need {len(phases) - 1} interfaces, got '
                         f'{len(interfaces)}')
    if not interfaces:
        return append_phase(identity_smatrix(phases[0].shape[-1]), phases[0])
    total = append_phase(_prepend_phase(phases[0], interfaces[0]), phases[1])
    for interface, phase in zip(interfaces[1:], phases[2:]):
        total = append_phase(star(total, interface), phase)
    return total


@dataclass(eq=False)
class Cascade:
    """Total port S-matrix and the per-section partial cascades of one EME solve."""
    total: SMatrix
    phases: list          # (N_k,) exp(-j k L_k) per section
    left_s21: list        # (N_k, N_1) lo window plane -> lo end of section k
    left_s22: list        # (N_k, N_k) reflection seen from the lo end of section k looking back
    right_s11: list       # (N_k, N_k) reflection seen from the hi end of section k looking forward
    right_s12: list       # (N_k, N_last) hi window plane -> hi end of section k

    def section_amplitudes(self, index, face, incident):
        """(forward amplitudes at the lo end, backward ones at the hi end) of section `index` for the incident
        amplitudes (port modes of `face`, travelling into the device)."""
        return partial_amplitudes(self.phases[index], self.left_s21[index], self.left_s22[index],
                                  self.right_s11[index], self.right_s12[index], face, incident)


def partial_amplitudes(phase, left_s21, left_s22, right_s11, right_s12, face, incident):
    """(forward amplitudes at the lo end, backward ones at the hi end) of a section of propagation factors phase
    between the partial cascades lo plane -> its lo end (s21, s22) and its hi end -> hi plane (s11, s12)."""
    eye = np.eye(phase.size, dtype=np.complex128)
    if face == 'lo':
        matrix = eye - left_s22 @ (phase[:, None] * right_s11 * phase[None, :])
        forward = np.linalg.solve(matrix, left_s21 @ incident)
        return forward, right_s11 @ (phase * forward)
    if face != 'hi':
        raise ValueError(f"face must be one of {PORT_FACES}, got {face!r}")
    matrix = eye - right_s11 @ (phase[:, None] * left_s22 * phase[None, :])
    backward = np.linalg.solve(matrix, right_s12 @ incident)
    return left_s22 @ (phase * backward), backward


def cell_cascade(phases, interfaces, first, last, junction):
    """Cascade of one period between the hi ends of the cell's last section: a zero-length section (its basis), the
    junction, then sections first..last; section j >= 1 = cell section first + j - 1 (total = unit_cell)."""
    builder = CascadeBuilder()
    builder.add_section(np.ones(np.asarray(phases[last]).size, dtype=np.complex128))
    builder.add_interface(junction)
    for index in range(first, last + 1):
        if index > first:
            builder.add_interface(interfaces[index - 1])
        builder.add_section(phases[index])
    return builder.finish()


def _extend_left(left_s21, left_s22, smatrix):
    """(s21, s22) of the partial cascade (lo plane -> here: s21, s22) followed by smatrix (the two star product blocks
    it needs, one LU)."""
    toward_right = np.eye(left_s22.shape[0], dtype=np.complex128) - left_s22 @ smatrix.s11
    n_first = left_s21.shape[1]
    solved = np.linalg.solve(toward_right, np.concatenate((left_s21, left_s22 @ smatrix.s12), axis=1))
    return smatrix.s21 @ solved[:, :n_first], smatrix.s22 + smatrix.s21 @ solved[:, n_first:]


def _extend_right(smatrix, right_s11, right_s12):
    """(s11, s12) of smatrix followed by the partial cascade (here -> hi plane: s11, s12; one LU)."""
    toward_left = np.eye(right_s11.shape[0], dtype=np.complex128) - right_s11 @ smatrix.s22
    n_reflected = smatrix.s21.shape[1]
    solved = np.linalg.solve(toward_left, np.concatenate((right_s11 @ smatrix.s21, right_s12), axis=1))
    return smatrix.s11 + smatrix.s12 @ solved[:, :n_reflected], smatrix.s12 @ solved[:, n_reflected:]


def period_plane_amplitudes(cascade, last, cell, exit_interface, n_periods, fields):
    """Per field (a list of (face, incident) excitations, summed) the (n_periods, N) forward and backward amplitudes
    at the planes P_1..P_n (hi end of the cell's last section after period p) of a cascade whose interface after
    section `last` is repeated_link(unit cell, n_periods, exit_interface), cell = its cell_cascade (total = unit_cell);
    keeps 2 N^2 n_periods of partial cascades."""
    cell_smatrix = cell.total
    after = cascade.phases[last + 1]
    rights = [None] * n_periods   # P_p -> hi window plane (s11, s12)
    rights[-1] = _extend_right(exit_interface, after[:, None] * cascade.right_s11[last + 1] * after[None, :],
                               after[:, None] * cascade.right_s12[last + 1])
    for period in range(n_periods - 2, -1, -1):
        rights[period] = _extend_right(cell_smatrix, *rights[period + 1])
    phase = cascade.phases[last]
    left = (phase[:, None] * cascade.left_s21[last], phase[:, None] * cascade.left_s22[last] * phase[None, :])
    ones = np.ones(phase.size, dtype=np.complex128)
    planes = [(np.zeros((n_periods, phase.size), dtype=np.complex128),
               np.zeros((n_periods, phase.size), dtype=np.complex128)) for _ in fields]
    for period in range(n_periods):
        if period:
            left = _extend_left(*left, cell_smatrix)
        for (forward, backward), field in zip(planes, fields):
            for face, incident in field:
                plane_forward, plane_backward = partial_amplitudes(ones, *left, *rights[period], face, incident)
                forward[period] += plane_forward
                backward[period] += plane_backward
        rights[period] = None   # free as the sweep passes
    return planes


class CascadeBuilder:
    """Streams sections and interfaces lo to hi (add_section, add_interface alternately), then finish(). The forward
    pass keeps only the s21 / s22 blocks of the growing cascade and the backward pass only s11 / s12; the total is
    assembled from the end partials (s11 = phi_0 R11_0 phi_0, s12 = phi_0 R12_0, s21 / s22 = the forward blocks)."""

    def __init__(self):
        self._left = None   # (s21, s22) of the cascade lo plane -> the current point
        self._phases = []
        self._interfaces = []
        self._left_s21 = []
        self._left_s22 = []

    def add_section(self, phase):
        """Next section with propagation factors phase (N,) = exp(-j k L) of its modes."""
        phase = np.asarray(phase, dtype=np.complex128)
        if len(self._phases) != len(self._interfaces):
            raise ValueError('add_section: an interface must separate two sections')
        if self._left is None:
            self._left = (np.eye(phase.size, dtype=np.complex128), np.zeros((phase.size, phase.size),
                                                                            dtype=np.complex128))
        elif self._left[1].shape[0] != phase.size:
            raise ValueError(f'add_section: {phase.size} modes after an interface ending in {self._left[1].shape[0]}')
        left_s21, left_s22 = self._left
        self._left_s21.append(left_s21)
        self._left_s22.append(left_s22)
        self._phases.append(phase)
        self._left = (phase[:, None] * left_s21, phase[:, None] * left_s22 * phase[None, :])

    def add_interface(self, smatrix):
        """Interface S-matrix between the last added section and the next one."""
        if len(self._phases) != len(self._interfaces) + 1:
            raise ValueError('add_interface: needs a section before it')
        self._interfaces.append(smatrix)
        self._left = _extend_left(*self._left, smatrix)

    def finish(self):
        """Cascade with the total S-matrix and the hi-end partial cascades (backward pass over the interfaces)."""
        n_sections = len(self._phases)
        if n_sections == 0 or len(self._interfaces) != n_sections - 1:
            raise ValueError('finish: needs at least one section and no trailing interface')
        n_last = self._phases[-1].size
        right = (np.zeros((n_last, n_last), dtype=np.complex128), np.eye(n_last, dtype=np.complex128))
        right_s11, right_s12 = [None] * n_sections, [None] * n_sections
        right_s11[-1], right_s12[-1] = right
        for index in range(n_sections - 2, -1, -1):
            phase = self._phases[index + 1]
            right = _extend_right(self._interfaces[index], phase[:, None] * right[0] * phase[None, :],
                                  phase[:, None] * right[1])
            right_s11[index], right_s12[index] = right
        first = self._phases[0]
        total = SMatrix(s11=first[:, None] * right_s11[0] * first[None, :], s12=first[:, None] * right_s12[0],
                        s21=self._left[0], s22=self._left[1])
        cascade = Cascade(total=total, phases=self._phases, left_s21=self._left_s21, left_s22=self._left_s22,
                          right_s11=right_s11, right_s12=right_s12)
        self._interfaces = []
        return cascade
