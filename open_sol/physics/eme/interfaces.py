"""EME interfaces (host numpy): mode-matching S-matrix between two bilinear-normalised mode sets of one transverse
line, its power / reciprocity defects, and the cascade of a list of mode sets, plain or with a periodic cell (CPU
self-tests and EMEPhysics)."""
# Interface A | B (A on the lo side), incident a from A, reflected b, transmitted c; P = [<e_A,m, h_B,n>],
# Q = [<e_B,m, h_A,n>] (unconjugated, weights w s). Tangential u matched and projected on h_B: P^T (a + b) = c;
# tangential t matched and projected on e_A: a - b = P c. So G = P P^T, R_AA = (I + G)^-1 (I - G),
# T_BA = P^T (I + R_AA); incidence from B the same with Q (a sign flip of every t cancels). Complete bases give
# Q P = I, hence T_AB = T_BA^T (reciprocity) and, lossless, power conservation over the propagating modes; a truncated
# basis breaks both, which interface_defects reports. Discrete variant (matching vectors e = cos(k D / 2) u, mode_sets):
# e = the mean of the two cells beside the face, h ~ each side's own face flux w (u_+ - u_-) / D. ez (w = 1/mu, equal
# mu across the face): matching both is matching u on both cells, the em_fdfd_ez x stencil. hz (w = 1/eps_t): with
# flux F continuous, the actual difference across the face is D F / w_f and the two sides' virtual differences D F /
# w_A, D F / w_B, so mean continuity holds iff 1/w_f = (eps_t,A + eps_t,B) / 2: the em_fdfd_hz face rule (arithmetic
# face_average). So a complete-basis cascade solves the 2D Yee lattice of the sectioned guide exactly for both
# polarisations (self-tests eme_complete_basis_identities, eme_vs_fdfd_step_ez / _hz).
import numpy as np

from .cascade import CascadeBuilder, SMatrix, cell_cascade, repeated_link
from .mode_sets import is_propagating


def overlap_matrix(left, right):
    """(N_left, N_right) unconjugated <e_left,m, h_right,n> = sum e h w s over the line cells."""
    same_line = left.weights is right.weights or (left.weights.shape == right.weights.shape and (
        np.array_equal(left.weights, right.weights)
        or np.allclose(left.weights, right.weights, rtol=1e-12, atol=0.0)))
    if not same_line:
        raise ValueError('overlap_matrix: the two mode sets live on different transverse lines')
    return (left.e * left.weights[None, :]) @ right.h.T


def _reflection(overlap):
    """(I + P P^T)^-1 (I - P P^T) of an overlap matrix P."""
    gram = overlap @ overlap.T
    eye = np.eye(gram.shape[0], dtype=np.complex128)
    return np.linalg.solve(eye + gram, eye - gram)


def interface_smatrix(left, right):
    """SMatrix of the interface between ModeSets left (port 1, lo side) and right (port 2)."""
    to_right = overlap_matrix(left, right)
    to_left = overlap_matrix(right, left)
    reflection_left = _reflection(to_right)
    reflection_right = _reflection(to_left)
    eye_left = np.eye(left.n_modes, dtype=np.complex128)
    eye_right = np.eye(right.n_modes, dtype=np.complex128)
    return SMatrix(s11=reflection_left, s12=to_left.T @ (eye_right + reflection_right),
                   s21=to_right.T @ (eye_left + reflection_left), s22=reflection_right)


def interface_defects(smatrix, left, right):
    """(power defect, reciprocity defect) of an interface: max over the propagating incident modes of |1 - power
    reflected and transmitted into propagating modes| (meaningful for lossless lines without PML), and max |s12 -
    s21^T| / max |s21|."""
    largest = max(float(np.max(np.abs(smatrix.s21))), float(np.max(np.abs(smatrix.s12))), 1e-300)
    reciprocity = float(np.max(np.abs(smatrix.s12 - smatrix.s21.T))) / largest
    worst = -1.0
    for incident, reflection, transmission, other in ((left, smatrix.s11, smatrix.s21, right),
                                                      (right, smatrix.s22, smatrix.s12, left)):
        incident_mask, other_mask = is_propagating(incident), is_propagating(other)
        columns = np.flatnonzero(incident_mask)
        if columns.size == 0:
            continue
        carried = (incident.power[incident_mask] @ np.abs(reflection[np.ix_(incident_mask, columns)]) ** 2
                   + other.power[other_mask] @ np.abs(transmission[np.ix_(other_mask, columns)]) ** 2)
        worst = max(worst, float(np.max(np.abs(1.0 - carried / incident.power[columns]))))
    return (worst if worst >= 0.0 else float('nan')), reciprocity


def cascade_mode_sets(mode_sets, lengths):
    """Cascade of ModeSets with section lengths [m] (lo to hi), interfaces between consecutive sets."""
    if len(mode_sets) != len(lengths):
        raise ValueError(f'{len(mode_sets)} mode sets but {len(lengths)} section lengths')
    builder = CascadeBuilder()
    for index, (modes, length) in enumerate(zip(mode_sets, lengths)):
        if index:
            builder.add_interface(interface_smatrix(mode_sets[index - 1], modes))
        builder.add_section(np.exp(-1j * modes.k * float(length)))
    return builder.finish()


def periodic_cascade_mode_sets(mode_sets, lengths, first, last, n_periods):
    """(Cascade, cell Cascade (cascade.cell_cascade), exit interface) of ModeSets whose sections first..last are a
    unit cell repeated n_periods times (EMEBase._solve's link; a section must follow the cell)."""
    if len(mode_sets) != len(lengths) or not 0 <= first <= last < len(mode_sets) - 1:
        raise ValueError(f'{len(mode_sets)} mode sets, {len(lengths)} lengths, cell sections {first}..{last}: needs '
                         'equal counts and a section after the cell')
    phases = [np.exp(-1j * modes.k * float(length)) for modes, length in zip(mode_sets, lengths)]
    interfaces = [interface_smatrix(left, right) for left, right in zip(mode_sets, mode_sets[1:])]
    cell = cell_cascade(phases, interfaces, first, last, interface_smatrix(mode_sets[last], mode_sets[first]))
    builder = CascadeBuilder()
    for index, phase in enumerate(phases):
        if index:
            link = interfaces[index - 1]
            builder.add_interface(repeated_link(cell.total, n_periods, link) if index - 1 == last else link)
        builder.add_section(phase)
    return builder.finish(), cell, interfaces[last]
