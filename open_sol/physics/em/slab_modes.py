"""1D slab (line-cut) eigenmodes of the 2D Yee stencil for mode sources and monitors (A3): numpy, scipy imported lazily
with numpy fallbacks, never cupy. exp(+j omega t): a mode toward +normal varies as exp(-j k s), Re k > 0 (propagating)
or Im k < 0 (evanescent)."""
# Operator on the active cells of a SlabLine (face k = lo face of cell k; F_0 / F_m from the end conditions):
#   F_k = face_weight[k] face_inverse_spacing[k] / face_stretch[k] (u_k - u_{k-1})
#   (A u)_k = cell_weight[k] / (cell_widths[k] cell_stretch[k]) (F_{k+1} - F_k) + (omega_t / C0)^2 index_term[k] u_k
# A u = k_t^2 u follows from the kernels with d/dt -> j omega_t and d/ds of exp(-j k s) over the dual spacing D ->
# -j k_t. ez: j omega_t mu0 mu_r (H_x, H_y) = (-dE_z/dy, dE_z/dx), j omega_t eps0 eps E_z = dH_y/dx - dH_x/dy, so
# cell_weight mu_r, face_weight 1/mu_r, q = eps mu_r. hz: j omega_t eps0 (eps_x E_x, eps_y E_y) = (dH_z/dy, -dH_z/dx),
# j omega_t mu0 mu_r H_z = dE_x/dy - dE_y/dx, so cell_weight eps_t, face_weight 1/eps_l, q = eps_t mu_r.
# rho = cell_widths cell_stretch / cell_weight makes S = diag(rho) A complex symmetric; T = diag(rho)^(1/2) A
# diag(rho)^(-1/2) is diagonalised and u = v / sqrt(rho). Bilinear product <a, b> = sum a b rho (0 between distinct
# modes of one line; complex lines recombine (nearly) degenerate eigenvectors so that it vanishes there too, e.g. the
# +-k pairs of a lossy periodic line: _orthogonalise_degenerate).
# Branch of k_t = +-sqrt(lambda): Re k_t > 0 when Re lambda > 0 (propagating: the phase advances toward +normal), else
# Im k_t < 0 (evanescent: decaying toward +normal); Im k_t then follows Im lambda. Forcing Im k_t <= 0 instead would
# turn a guided mode whose evanescent tail reaches a PML (its eigenvalue gets a tiny Im of either sign, the end-wall
# reflection phase-rotated by the stretch) into a backward one and sort it last.
import cmath
import importlib
import math
from dataclasses import dataclass, replace

import numpy as np

from ...core.constants import C0, EPS0, MU0

END_CONDITIONS = ('dirichlet', 'neumann', 'odd', 'periodic')
_POLARIZATIONS = ('ez', 'hz')
# Diagonal share of an end face's conductance b: dirichlet (ghost 0) b, neumann (flux 0) 0, odd (ghost -u) 2 b,
# periodic b on both end cells plus the corner coupling b.
_END_DIAGONAL_FACTOR = {'dirichlet': 1.0, 'neumann': 0.0, 'odd': 2.0, 'periodic': 1.0}
DENSE_COMPLEX_MAX_CELLS = 600   # larger complex systems use scipy shift-invert (dense numpy eig without scipy)
_SHIFT_RELATIVE_OFFSET = 1e-6    # shift-invert sigma this far above (omega_t/C0)^2 max Re q: never exactly singular
_ARPACK_MAX_RESTARTS = 300        # sparse eigs restarts, then the exact dense / inverse-iteration fallback below
# Dense complex lines: numpy eigvals, then inverse iteration (scipy solve_banded, O(m) per step) for the ranked vectors
# only, at least two steps (one passes the residual test while the vector still carries ~1e-9 of its neighbours); a
# residual above _INVERSE_RESIDUAL ||T||_inf after _INVERSE_STEPS, or two eigenvalues closer than _DEGENERATE_GAP
# ||T||_inf (one shift cannot separate their vectors), falls back to the full numpy eig.
_INVERSE_STEPS = 3
_INVERSE_RESIDUAL = 1e-10
_INVERSE_SHIFT_OFFSET = 1e-13    # shift this far (relative to ||T||_inf) off the eigenvalue: never exactly singular
_DEGENERATE_GAP = 1e-8
# Eigenvectors of one such cluster count as bilinear-orthogonal when |v_a^T v_b| <= this x |v_a| |v_b| (exact
# eigenvectors of eigenvalues a gap g apart reach eps ||T|| / g <= ~1e-8); above it they are recombined.
_ORTHOGONAL_TOLERANCE = 1e-6
_POWER_FLOOR = 1e-6              # |Re sum conj(g)|u|^2 w| below this fraction of sum |g||u|^2 w: bilinear norm
# Phase reference: the first cell from the lo end with |u| >= (1 - _PEAK_TIE_RELATIVE) max |u|, so the mirror peaks of
# an odd mode on a symmetric line tie deterministically (argmax would pick one by round-off).
_PEAK_TIE_RELATIVE = 1e-6
# Modes are ranked by descending Re(k_t^2) (= descending Re n_eff for propagating modes, degenerate periodic pairs
# aside), ties by descending Im(k_t^2) (least lossy first). Ranking by Re n_eff would let evanescent modes with a
# complex eigenvalue, Re n_eff = |Im lambda| / (2 |k_t| k0) at Im n_eff of order -20 on PML lines, outrank weakly
# guided modes.
_EIGENVALUE_TIE_DECIMALS = 12    # Re(k_t^2) equal to this many decimals of its relative value counts as a tie
# PML modes (more than this share of sum |u|^2 |rho| on cells with a complex stretch) are dropped: localised in a CPML
# with |arg s| > 45 deg they reach Re(n_eff) far above every material index and would outrank the guided modes.
_PML_MODE_FRACTION = 0.5
# track_slab_mode: Rayleigh-quotient iteration from the neighbouring frequency's mode; it converges when
# ||T v - lambda v|| <= _TRACK_RESIDUAL ||T||_inf ||v|| (then one polishing step), else, or when the converged mode
# overlaps the reference below _TRACK_MIN_OVERLAP, the full solve_slab_modes + track_mode decides.
_TRACK_MAX_ITERATIONS = 8
_TRACK_RESIDUAL = 1e-12
_TRACK_MIN_OVERLAP = 0.9


@dataclass(frozen=True, eq=False)
class SlabLine:
    """One line cut at one frequency: m unknown cells (u at the centres) and m + 1 faces (face k = lo face of cell k);
    material values relative and complex at that frequency (Im < 0 lossy)."""
    polarization: str                 # 'ez' (u = E_z) | 'hz' (u = H_z)
    cell_widths: np.ndarray           # (m,) float64 [m] kernel widths along the line
    cell_stretch: np.ndarray          # (m,) complex128 PML stretch at the centres (1 outside the PML)
    cell_weight: np.ndarray           # (m,) complex128: ez mu_r; hz eps_t (permittivity of the transverse partner)
    index_term: np.ndarray            # (m,) complex128 q: ez eps_r mu_r; hz eps_t mu_r
    active: np.ndarray                # (m,) bool; False = PEC cell (u = 0, no row)
    face_inverse_spacing: np.ndarray  # (m + 1,) float64 [1/m]; periodic: face 0 (= face m) is the wrap face
    face_stretch: np.ndarray          # (m + 1,) complex128
    face_weight: np.ndarray           # (m + 1,) complex128: ez 1/mu_r of the face's lower cell; hz 1/eps_l; 0 = PEC
    lo_end: str                       # END_CONDITIONS
    hi_end: str

    def __post_init__(self):
        if self.polarization not in _POLARIZATIONS:
            raise ValueError(f"SlabLine polarization must be 'ez' or 'hz', got {self.polarization!r}")
        n_cells = np.shape(self.cell_widths)[0] if np.ndim(self.cell_widths) == 1 else 0
        if n_cells == 0:
            raise ValueError(f"SlabLine cell_widths must be a non-empty 1D array, got shape "
                             f"{np.shape(self.cell_widths)}")
        for name, size in (('cell_stretch', n_cells), ('cell_weight', n_cells), ('index_term', n_cells),
                           ('active', n_cells), ('face_inverse_spacing', n_cells + 1), ('face_stretch', n_cells + 1),
                           ('face_weight', n_cells + 1)):
            if np.shape(getattr(self, name)) != (size,):
                raise ValueError(f"SlabLine {name} must have shape ({size},), got {np.shape(getattr(self, name))}")
        if np.asarray(self.active).dtype != np.bool_:
            raise TypeError(f"SlabLine active must be a bool array, got dtype {np.asarray(self.active).dtype}")
        if self.lo_end not in END_CONDITIONS or self.hi_end not in END_CONDITIONS:
            raise ValueError(f"SlabLine ends must be one of {', '.join(END_CONDITIONS)}, got lo {self.lo_end!r}, "
                             f"hi {self.hi_end!r}")
        if (self.lo_end == 'periodic') != (self.hi_end == 'periodic'):
            raise ValueError(f"SlabLine: 'periodic' applies to both ends or neither, got lo {self.lo_end!r}, "
                             f"hi {self.hi_end!r}")
        lengths = np.concatenate((np.asarray(self.cell_widths, dtype=np.float64),
                                  np.asarray(self.face_inverse_spacing, dtype=np.float64)))
        if not (np.all(np.isfinite(lengths)) and np.all(lengths > 0.0)):
            raise ValueError("SlabLine cell_widths [m] and face_inverse_spacing [1/m] must be finite and > 0")
        active = np.asarray(self.active)
        divisors = np.concatenate((np.asarray(self.cell_stretch)[active], np.asarray(self.cell_weight)[active],
                                   np.asarray(self.face_stretch)))
        values = np.concatenate((divisors, np.asarray(self.index_term)[active], np.asarray(self.face_weight)))
        if not (np.all(np.isfinite(values)) and np.all(divisors != 0)):
            raise ValueError("SlabLine stretch, weight and index values must be finite; stretches and the cell_weight "
                             "of active cells non-zero")


@dataclass(frozen=True, eq=False)
class SlabMode:
    """One normalised mode of a SlabLine, varying as exp(-j k s) toward +normal."""
    n_eff: complex                  # k C0 / omega
    k_tilde: complex                # sqrt(eigenvalue) [1/m]: Re > 0 when Re(eigenvalue) > 0, else Im < 0
    k: complex                      # propagation constant along the normal [1/m] (= k_tilde without normal_spacing)
    omega: float                    # true angular frequency [rad/s]
    omega_tilde: float              # numerical angular frequency used [rad/s]
    profile: np.ndarray             # (m,) complex128 u, 0 on inactive cells, normalised (solve_slab_modes)
    transverse_factor: np.ndarray   # (m,) complex128 g = k_tilde / (omega_tilde X0 cell_weight), X0 = MU0 (ez) /
                                    #   EPS0 (hz), 0 on inactive cells; t = sign_n g u, (e x h).n = g u^2 per cell
    power: float                    # discrete flux 1/2 Re sum conj(g) |u|^2 cell_widths phi [W/m] (1 when propagating),
                                    #   phi = _discrete_flux_factor (1 without dt / normal_spacing)
    edge_amplitude: float           # max(|u| first active, |u| last active) / max |u|


def numerical_frequency(omega, dt=None):
    """FDTD numerical angular frequency (2/dt) sin(omega dt / 2) [rad/s] (omega itself when dt is None)."""
    omega = float(omega)
    if not (math.isfinite(omega) and omega > 0.0):
        raise ValueError(f"angular frequency omega must be finite and > 0 [rad/s], got {omega!r}")
    if dt is None:
        return omega
    dt = float(dt)
    if not (math.isfinite(dt) and dt > 0.0):
        raise ValueError(f"time step dt must be finite and > 0 [s], got {dt!r}")
    if omega * dt >= math.pi:
        raise ValueError(f"omega dt = {omega * dt:.6g} >= pi: the frequency is above the time-step Nyquist limit")
    return 2.0 / dt * math.sin(0.5 * omega * dt)


def _field_constant(line):
    """X0 of g = k_t / (omega_t X0 cell_weight): MU0 (ez) or EPS0 (hz)."""
    return MU0 if line.polarization == 'ez' else EPS0


def _stretched_widths(line):
    """(m,) complex128 cell_widths cell_stretch [m]."""
    return np.asarray(line.cell_widths, dtype=np.float64) * np.asarray(line.cell_stretch, dtype=np.complex128)


def _bilinear_weight(line):
    """(m,) complex128 rho = cell_widths cell_stretch / cell_weight on active cells, 0 on inactive ones."""
    active = np.asarray(line.active)
    rho = np.zeros(active.shape[0], dtype=np.complex128)
    rho[active] = _stretched_widths(line)[active] / np.asarray(line.cell_weight, dtype=np.complex128)[active]
    return rho


def _symmetric_system(line, omega_tilde):
    """S = diag(rho) A on the active cells: (active cell indices, rho, diagonal, coupling between consecutive active
    cells (0 where they are not neighbours), periodic corner coupling (0 when none))."""
    n_cells = np.shape(line.cell_widths)[0]
    conductance = (np.asarray(line.face_weight, dtype=np.complex128)
                   * np.asarray(line.face_inverse_spacing, dtype=np.float64)
                   / np.asarray(line.face_stretch, dtype=np.complex128))
    rho = _bilinear_weight(line)
    diagonal = rho * (omega_tilde / C0) ** 2 * np.asarray(line.index_term, dtype=np.complex128)
    interior = conductance[1:n_cells]                   # faces 1..m-1 act on both adjacent cells
    diagonal[1:] -= interior
    diagonal[:-1] -= interior
    periodic = line.lo_end == 'periodic'
    diagonal[0] -= _END_DIAGONAL_FACTOR[line.lo_end] * conductance[0]
    diagonal[-1] -= _END_DIAGONAL_FACTOR[line.hi_end] * conductance[0 if periodic else n_cells]
    active_cells = np.flatnonzero(line.active)
    coupling = np.where(np.diff(active_cells) == 1, conductance[active_cells[1:]], 0.0)
    corner = 0.0
    if periodic and active_cells[0] == 0 and active_cells[-1] == n_cells - 1:
        corner = conductance[0]
    return active_cells, rho[active_cells], diagonal[active_cells], coupling, corner


def _symmetrised_tridiagonal(rho, diagonal, coupling, corner):
    """(sqrt(rho), diagonal, coupling, corner) of T = diag(rho)^(-1/2) S diag(rho)^(-1/2) from S = diag(rho) A."""
    sqrt_rho = np.sqrt(rho)
    t_corner = corner / (sqrt_rho[0] * sqrt_rho[-1]) if corner != 0 else 0.0
    return sqrt_rho, diagonal / rho, coupling / (sqrt_rho[:-1] * sqrt_rho[1:]), t_corner


def _optional_scipy(submodule):
    """scipy.<submodule>, or None when scipy is not importable."""
    try:
        return importlib.import_module(f'scipy.{submodule}')
    except ImportError:
        return None


def _dense_matrix(diagonal, coupling, corner):
    """(n, n) symmetric tridiagonal matrix plus the periodic corner entries (added, so n <= 2 sums them)."""
    n_cells = diagonal.size
    matrix = np.diag(diagonal)
    index = np.arange(n_cells - 1)
    matrix[index, index + 1] += coupling
    matrix[index + 1, index] += coupling
    if corner != 0:
        matrix[0, n_cells - 1] += corner
        matrix[n_cells - 1, 0] += corner
    return matrix


def _real_symmetric_eigenpairs(diagonal, coupling, corner, n_wanted):
    """The n_wanted largest eigenpairs of a real symmetric T: scipy eigh_tridiagonal without a corner, else numpy eigh
    of the dense matrix."""
    n_cells = diagonal.size
    n_wanted = min(n_wanted, n_cells)
    scipy_linalg = _optional_scipy('linalg') if corner == 0 and n_cells > 1 else None
    if scipy_linalg is not None:
        return scipy_linalg.eigh_tridiagonal(diagonal, coupling, select='i',
                                             select_range=(n_cells - n_wanted, n_cells - 1))
    eigenvalues, eigenvectors = np.linalg.eigh(_dense_matrix(diagonal, coupling, corner))
    return eigenvalues[n_cells - n_wanted:], eigenvectors[:, n_cells - n_wanted:]


def _ranking_order(eigenvalues, eigenvalue_scale):
    """Indices by descending Re(eigenvalue) (rounded relative to eigenvalue_scale), ties by descending Im."""
    return np.lexsort((-eigenvalues.imag, -np.round(eigenvalues.real / eigenvalue_scale, _EIGENVALUE_TIE_DECIMALS)))


def _drop_pml_modes(eigenvalues, eigenvectors, complex_stretch):
    """The eigenpairs with at most _PML_MODE_FRACTION of sum |v|^2 (= |u|^2 |rho|) on the complex_stretch cells."""
    if not np.any(complex_stretch):
        return eigenvalues, eigenvectors
    weight = np.abs(eigenvectors) ** 2
    guided_or_radiating = np.sum(weight[complex_stretch], axis=0) <= _PML_MODE_FRACTION * np.sum(weight, axis=0)
    return eigenvalues[guided_or_radiating], eigenvectors[:, guided_or_radiating]


def _inverse_iteration_eigenpairs(diagonal, coupling, n_wanted, complex_stretch, eigenvalue_scale):
    """The n_wanted top-ranked non-PML eigenpairs of a complex symmetric tridiagonal T (no corner): dense eigvals, then
    inverse iteration per ranked vector; None when an iteration fails or an eigenvalue is (nearly) degenerate."""
    eigenvalues = np.linalg.eigvals(_dense_matrix(diagonal, coupling, 0.0))
    scale = _tridiagonal_scale(diagonal, coupling, 0.0)
    start_generator = np.random.default_rng(0)   # fixed start: never bilinear-orthogonal to a mode by symmetry
    start = start_generator.standard_normal(diagonal.size) + 1j * start_generator.standard_normal(diagonal.size)
    selected_values, selected_vectors = [], []
    for index in _ranking_order(eigenvalues, eigenvalue_scale):
        eigenvalue = eigenvalues[index]
        gaps = np.abs(np.delete(eigenvalues, index) - eigenvalue)
        if gaps.size > 0 and float(np.min(gaps)) <= _DEGENERATE_GAP * scale:
            return None
        vector = start / np.linalg.norm(start)
        shift = eigenvalue + _INVERSE_SHIFT_OFFSET * scale
        converged = False
        for step in range(_INVERSE_STEPS):
            try:
                solved = _shifted_solve(diagonal, coupling, 0.0, shift, vector)
            except np.linalg.LinAlgError:
                return None
            if not np.all(np.isfinite(solved)):
                return None
            vector = solved / np.linalg.norm(solved)
            if step == 0:   # one step leaves ~|shift - lambda| / gap ~ 1e-9 of the other modes: always take two
                continue
            residual = _tridiagonal_product(diagonal, coupling, 0.0, vector) - eigenvalue * vector
            if float(np.linalg.norm(residual)) <= _INVERSE_RESIDUAL * scale:
                converged = True
                break
        if not converged:
            return None
        kept_values, _ = _drop_pml_modes(np.array([eigenvalue]), vector[:, None], complex_stretch)
        if kept_values.size == 1:
            selected_values.append(eigenvalue)
            selected_vectors.append(vector)
            if len(selected_values) == n_wanted:
                break
    if not selected_values:
        return np.zeros(0, dtype=np.complex128), np.zeros((diagonal.size, 0), dtype=np.complex128)
    return np.array(selected_values, dtype=np.complex128), np.column_stack(selected_vectors)


def _complex_eigenpairs(diagonal, coupling, corner, n_wanted, shift, complex_stretch, eigenvalue_scale):
    """Eigenpairs of a complex symmetric T, PML modes (complex_stretch cells) dropped: above DENSE_COMPLEX_MAX_CELLS
    cells the 2 n_wanted + 10 nearest the real shift (scipy shift-invert), else (or when ARPACK fails) the n_wanted
    top-ranked by _inverse_iteration_eigenpairs; all by dense numpy eig with a corner, without scipy or when that fails."""
    n_cells = diagonal.size
    n_nearest = 2 * n_wanted + 10
    sparse_linalg = None
    if n_cells > DENSE_COMPLEX_MAX_CELLS and n_nearest < n_cells - 1:
        sparse_linalg = _optional_scipy('sparse.linalg')
    if sparse_linalg is not None:
        scipy_sparse = importlib.import_module('scipy.sparse')
        index = np.arange(n_cells)
        rows = [index, index[:-1], index[1:]]
        columns = [index, index[1:], index[:-1]]
        values = [diagonal, coupling, coupling]
        if corner != 0:
            rows.append(np.array([0, n_cells - 1]))
            columns.append(np.array([n_cells - 1, 0]))
            values.append(np.array([corner, corner]))
        matrix = scipy_sparse.coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
                                         shape=(n_cells, n_cells)).tocsc()
        try:
            return _drop_pml_modes(*sparse_linalg.eigs(matrix, k=n_nearest, sigma=shift, which='LM',
                                                       maxiter=_ARPACK_MAX_RESTARTS), complex_stretch)
        except RuntimeError:   # ARPACK non-convergence or a singular factorisation: the dense solve is equivalent
            pass
    if corner == 0 and n_cells > 1 and _optional_scipy('linalg') is not None:
        pairs = _inverse_iteration_eigenpairs(diagonal, coupling, n_wanted, complex_stretch, eigenvalue_scale)
        if pairs is not None:
            return pairs
    return _drop_pml_modes(*np.linalg.eig(_dense_matrix(diagonal, coupling, corner)), complex_stretch)


def _bilinear_orthogonal_combinations(gram):
    """(g, g) C with C^T gram C diagonal for a complex symmetric, non-singular gram (pivoted bilinear Gram-Schmidt; a
    sum of two columns replaces a quasi-null pivot such as e^{+jks} on a full period, whose v^T v vanishes)."""
    candidates = np.eye(gram.shape[0], dtype=np.complex128)
    chosen = []
    while candidates.shape[1] > 0:
        forms = candidates.T @ gram @ candidates
        diagonal = np.abs(np.diag(forms))
        best = int(np.argmax(diagonal))
        pivot, dropped = candidates[:, best], best
        off_diagonal = np.abs(forms - np.diag(np.diag(forms)))
        if candidates.shape[1] > 1 and float(off_diagonal.max()) > float(diagonal[best]):
            # |x_i +- x_j|^T G (x_i +- x_j) = |G_ii + G_jj +- 2 G_ij|: the larger sign is >= 2 |G_ij| > 0
            first, second = np.unravel_index(int(np.argmax(off_diagonal)), off_diagonal.shape)
            pair_sum = forms[first, first] + forms[second, second]
            cross_term = 2.0 * forms[first, second]
            sign = 1.0 if abs(pair_sum + cross_term) >= abs(pair_sum - cross_term) else -1.0
            pivot, dropped = candidates[:, first] + sign * candidates[:, second], first
        remaining = np.delete(candidates, dropped, axis=1)
        pivot_form = complex(pivot @ gram @ pivot)
        remaining = remaining - np.outer(pivot, (pivot @ gram @ remaining) / pivot_form)
        chosen.append(pivot)
        candidates = remaining
    return np.column_stack(chosen)


def _orthogonalise_degenerate(eigenvalues, eigenvectors, diagonal, coupling, corner):
    """(eigenvalues, eigenvectors) of T with every cluster of eigenvalues closer than _DEGENERATE_GAP ||T|| whose
    vectors are not bilinear-orthogonal (above _ORTHOGONAL_TOLERANCE) recombined so that v_a^T v_b = 0, each new
    vector taking its unconjugated Rayleigh quotient v^T T v / v^T v as eigenvalue."""
    eigenvalues = np.array(eigenvalues, dtype=np.complex128)
    eigenvectors = np.array(eigenvectors, dtype=np.complex128)
    gap = _DEGENERATE_GAP * _tridiagonal_scale(diagonal, coupling, corner)
    unassigned = list(range(eigenvalues.size))
    while unassigned:
        first = unassigned[0]
        cluster = [index for index in unassigned if abs(eigenvalues[index] - eigenvalues[first]) <= gap]
        unassigned = [index for index in unassigned if index not in cluster]
        if len(cluster) < 2:
            continue
        vectors = eigenvectors[:, cluster]
        gram = vectors.T @ vectors
        norms = np.sqrt(np.sum(np.abs(vectors) ** 2, axis=0))
        relative = np.abs(gram) / np.outer(norms, norms)
        if float(np.max(relative - np.diag(np.diag(relative)))) <= _ORTHOGONAL_TOLERANCE:
            continue
        vectors = vectors @ _bilinear_orthogonal_combinations(gram)
        eigenvectors[:, cluster] = vectors
        for position, index in enumerate(cluster):
            vector = vectors[:, position]
            eigenvalues[index] = (complex(vector @ _tridiagonal_product(diagonal, coupling, corner, vector))
                                  / complex(vector @ vector))
    return eigenvalues, eigenvectors


def _tridiagonal_product(diagonal, coupling, corner, vector):
    """T v of the symmetric tridiagonal T plus its periodic corner entries (added, as _dense_matrix)."""
    product = diagonal * vector
    product[:-1] += coupling * vector[1:]
    product[1:] += coupling * vector[:-1]
    if corner != 0:
        product[0] += corner * vector[-1]
        product[-1] += corner * vector[0]
    return product


def _tridiagonal_scale(diagonal, coupling, corner):
    """||T||_inf bound max |diagonal| + 2 max |coupling| + 2 |corner| (the residual tests' scale)."""
    return (float(np.max(np.abs(diagonal))) + 2.0 * float(np.max(np.abs(coupling), initial=0.0))
            + 2.0 * abs(corner))


def _shifted_solve(diagonal, coupling, corner, shift, rhs):
    """(T - shift I)^-1 rhs: scipy solve_banded (partial pivoting, O(m)) without a corner, else dense numpy solve."""
    scipy_linalg = _optional_scipy('linalg') if corner == 0 and diagonal.size > 1 else None
    if scipy_linalg is None:
        return np.linalg.solve(_dense_matrix(diagonal - shift, coupling, corner), rhs)
    bands = np.zeros((3, diagonal.size), dtype=np.complex128)
    bands[0, 1:] = coupling
    bands[1] = diagonal - shift
    bands[2, :-1] = coupling
    return scipy_linalg.solve_banded((1, 1), bands, rhs, check_finite=False)


def _rayleigh_quotient_iteration(diagonal, coupling, corner, eigenvalue, vector):
    """(eigenvalue, eigenvector) of the complex symmetric T reached from the start pair by Rayleigh-quotient iteration
    with the unconjugated quotient v^T T v / v^T v; None unless the residual converges."""
    scale = _tridiagonal_scale(diagonal, coupling, corner)
    vector = vector / np.linalg.norm(vector)
    converged = False
    for _ in range(_TRACK_MAX_ITERATIONS):
        try:
            solved = _shifted_solve(diagonal, coupling, corner, eigenvalue, vector)
        except np.linalg.LinAlgError:   # shift exactly on an eigenvalue
            break
        if not np.all(np.isfinite(solved)):
            break
        vector = solved / np.linalg.norm(solved)
        product = _tridiagonal_product(diagonal, coupling, corner, vector)
        self_product = complex(vector @ vector)
        if self_product == 0:
            return None
        eigenvalue = complex(vector @ product) / self_product
        if converged:
            return eigenvalue, vector   # the polishing step after the residual test
        converged = float(np.linalg.norm(product - eigenvalue * vector)) <= _TRACK_RESIDUAL * scale
    return (eigenvalue, vector) if converged else None


def _transverse_wavenumber(eigenvalues):
    """k_t = +-sqrt(eigenvalue) on the branch of the module header: Re k_t > 0 when Re(eigenvalue) > 0, else Im k_t < 0."""
    eigenvalues = np.asarray(eigenvalues, dtype=np.complex128)
    k_tilde = np.sqrt(eigenvalues)                                # principal root: Re >= 0
    return np.where((eigenvalues.real <= 0.0) & (k_tilde.imag > 0.0), -k_tilde, k_tilde)


def _propagation_constant(k_tilde, normal_spacing):
    """k = (2/D) arcsin(k_t D/2) on the branch continuous with k_t (the principal arcsin keeps the signs of Re and Im;
    real |k_t D/2| > 1 lies on its cut: take Im k < 0); k_t itself when normal_spacing D is None."""
    if normal_spacing is None:
        return k_tilde
    half_spacing = 0.5 * normal_spacing
    argument = np.asarray(k_tilde, dtype=np.complex128) * half_spacing
    phase = np.arcsin(argument)
    on_cut = (argument.imag == 0.0) & (np.abs(argument.real) > 1.0)
    phase = np.where(on_cut, phase.real - 1j * np.abs(phase.imag), phase)
    return phase / half_spacing


def _discrete_flux_factor(k, omega, normal_spacing, dt):
    """phi of the Yee flux 1/2 Re sum conj(g) |u|^2 w phi: e^{+j conj(k) D/2} (t sits D/2 downstream of u) times
    cos(omega dt/2) (the FDTD flux averages E over two steps to the H time); 1 without normal_spacing D / dt."""
    factor = 1.0 + 0.0j if normal_spacing is None else cmath.exp(0.5j * complex(k).conjugate() * normal_spacing)
    return factor if dt is None else factor * math.cos(0.5 * omega * dt)


def peak_cell(profile):
    """Phase-reference cell of a mode profile: the first cell from the lo end within 1e-6 (relative) of max |u|, so
    mirror-equal peaks tie deterministically."""
    magnitude = np.abs(profile)
    return int(np.argmax(magnitude >= (1.0 - _PEAK_TIE_RELATIVE) * np.max(magnitude)))


def _normalised_mode(line, active_cells, sqrt_rho, eigenvector, k_tilde, k, omega, omega_tilde, flux_factor):
    """SlabMode of one eigenvector (_normalised_modes of one column)."""
    return _normalised_modes(line, active_cells, sqrt_rho, np.asarray(eigenvector)[:, None], [k_tilde], [k], omega,
                             omega_tilde, [flux_factor])[0]


def _normalised_modes(line, active_cells, sqrt_rho, eigenvectors, k_tilde, k, omega, omega_tilde, flux_factors):
    """SlabModes of the (n_active, n) eigenvector columns, one row each in one pass: 1 W/m of discrete flux (bilinear
    |1/2 sum g u^2 w s| = 1 for evanescent / strongly lossy modes, unit peak if both vanish), real > 0 at peak_cell."""
    widths = np.asarray(line.cell_widths, dtype=np.float64)
    k_tilde = np.asarray(k_tilde, dtype=np.complex128)
    k = np.asarray(k, dtype=np.complex128)
    n_modes = k_tilde.size
    profiles = np.zeros((n_modes, widths.size), dtype=np.complex128)   # one mode per row (row sums as 1D sums)
    profiles[:, active_cells] = (np.asarray(eigenvectors) / np.asarray(sqrt_rho)[:, None]).T
    transverse_factors = np.zeros((n_modes, widths.size), dtype=np.complex128)
    transverse_factors[:, active_cells] = k_tilde[:, None] / (
        omega_tilde * _field_constant(line) * np.asarray(line.cell_weight, dtype=np.complex128)[active_cells])
    flux_weights = np.conj(transverse_factors) * np.asarray(flux_factors, dtype=np.complex128)[:, None]
    intensities = np.abs(profiles) ** 2 * widths
    fluxes = np.real(np.sum(flux_weights * intensities, axis=1))
    flux_floors = _POWER_FLOOR * np.sum(np.abs(transverse_factors) * intensities, axis=1)
    bilinear_norms = np.abs(0.5 * np.sum(transverse_factors * profiles ** 2 * _stretched_widths(line), axis=1))
    norms = np.where(np.abs(fluxes) >= flux_floors, 0.5 * np.abs(fluxes), bilinear_norms)
    scales = np.where(norms > 0.0, np.sqrt(norms), np.max(np.abs(profiles), axis=1))
    profiles /= scales[:, None]
    magnitudes = np.abs(profiles)   # peak_cell per row: lo-most cell within the tie of the row maximum
    peaks = np.argmax(magnitudes >= (1.0 - _PEAK_TIE_RELATIVE) * np.max(magnitudes, axis=1, keepdims=True), axis=1)
    rows = np.arange(n_modes)
    peak_values = profiles[rows, peaks]
    profiles *= (np.conj(peak_values) / np.abs(peak_values))[:, None]
    magnitudes = np.abs(profiles)
    powers = 0.5 * np.real(np.sum(flux_weights * magnitudes ** 2 * widths, axis=1))
    edge_amplitudes = (np.maximum(magnitudes[:, active_cells[0]], magnitudes[:, active_cells[-1]])
                       / magnitudes[rows, peaks])
    return [SlabMode(n_eff=complex(k[index] * C0 / omega), k_tilde=complex(k_tilde[index]), k=complex(k[index]),
                     omega=omega, omega_tilde=omega_tilde, profile=profiles[index],
                     transverse_factor=transverse_factors[index], power=float(powers[index]),
                     edge_amplitude=float(edge_amplitudes[index]))
            for index in range(n_modes)]


def solve_slab_modes(line, omega, n_modes, dt=None, normal_spacing=None, drop_pml_modes=True):
    """Up to n_modes SlabModes of line at angular frequency omega [rad/s] by descending Re(k_t^2) (descending Re n_eff of
    the propagating modes, evanescent ones last), PML modes dropped unless drop_pml_modes is False (EME keeps them as
    radiation), degenerate ones mutually orthogonal; dt [s] and the normal dual spacing D [m] make them FDTD-consistent
    (omega_t, k = (2/D) arcsin(k_t D/2), 1 W/m of Yee flux)."""
    omega_tilde = numerical_frequency(omega, dt)
    omega = float(omega)
    dt = None if dt is None else float(dt)
    if isinstance(n_modes, bool) or not isinstance(n_modes, (int, np.integer)) or n_modes < 1:
        raise ValueError(f"n_modes must be an integer >= 1, got {n_modes!r}")
    if normal_spacing is not None:
        normal_spacing = float(normal_spacing)
        if not (math.isfinite(normal_spacing) and normal_spacing > 0.0):
            raise ValueError(f"normal_spacing must be finite and > 0 [m], got {normal_spacing!r}")
    if not np.any(line.active):
        raise ValueError("solve_slab_modes: the line has no active (non-PEC) cell")
    active_cells, rho, diagonal, coupling, corner = _symmetric_system(line, omega_tilde)
    sqrt_rho, t_diagonal, t_coupling, t_corner = _symmetrised_tridiagonal(rho, diagonal, coupling, corner)
    real_symmetric = bool(np.all(rho.imag == 0.0) and np.all(rho.real > 0.0) and np.all(diagonal.imag == 0.0)
                          and np.all(coupling.imag == 0.0) and np.imag(corner) == 0.0)
    eigenvalue_scale = (omega_tilde / C0) ** 2     # k0^2: the rounding of the tie key is relative to it
    if real_symmetric:
        eigenvalues, eigenvectors = _real_symmetric_eigenpairs(
            t_diagonal.real, t_coupling.real, float(np.real(t_corner)), int(n_modes))
    else:
        max_index_term = float(np.max(np.real(np.asarray(line.index_term, dtype=np.complex128)[active_cells])))
        shift = eigenvalue_scale * max_index_term * (1.0 + _SHIFT_RELATIVE_OFFSET)
        complex_stretch = np.asarray(line.cell_stretch, dtype=np.complex128)[active_cells].imag != 0.0
        if not drop_pml_modes:   # no cell counts as PML: _drop_pml_modes keeps every eigenpair
            complex_stretch = np.zeros_like(complex_stretch)
        eigenvalues, eigenvectors = _complex_eigenpairs(t_diagonal, t_coupling, t_corner, int(n_modes), shift,
                                                        complex_stretch, eigenvalue_scale)
        eigenvalues, eigenvectors = _orthogonalise_degenerate(eigenvalues, eigenvectors, t_diagonal, t_coupling,
                                                              t_corner)
    eigenvalues = np.asarray(eigenvalues, dtype=np.complex128)
    k_tilde = _transverse_wavenumber(eigenvalues)
    k = _propagation_constant(k_tilde, normal_spacing)
    order = np.asarray(_ranking_order(eigenvalues, eigenvalue_scale)[:n_modes], dtype=np.intp)
    flux_factors = [_discrete_flux_factor(k[index], omega, normal_spacing, dt) for index in order]
    return _normalised_modes(line, active_cells, sqrt_rho, np.asarray(eigenvectors)[:, order], k_tilde[order],
                             k[order], omega, omega_tilde, flux_factors)


def _check_profiles(line, *modes):
    """ValueError unless every mode profile has the line's cell count."""
    n_cells = np.shape(line.cell_widths)[0]
    for mode in modes:
        if np.shape(mode.profile) != (n_cells,):
            raise ValueError(f"mode profile shape {np.shape(mode.profile)} does not match the line ({n_cells} cells)")


def mode_overlap(line, mode_a, mode_b):
    """<a, b> / sqrt(sum |a|^2 |rho| sum |b|^2 |rho|) with the unconjugated <a, b> = sum a b rho: linear in each
    profile, 0 between distinct modes of one line, |overlap| <= 1, 1 for a real-profile mode with itself."""
    _check_profiles(line, mode_a, mode_b)
    rho = _bilinear_weight(line)
    magnitude_rho = np.abs(rho)
    norm = math.sqrt(float(np.sum(np.abs(mode_a.profile) ** 2 * magnitude_rho))
                     * float(np.sum(np.abs(mode_b.profile) ** 2 * magnitude_rho)))
    return complex(np.sum(mode_a.profile * mode_b.profile * rho)) / norm


def track_mode(line, reference_mode, modes):
    """Index in modes of the largest |mode_overlap| with reference_mode (all profiles on line's cells)."""
    if not modes:
        raise ValueError("track_mode needs at least one mode")
    return int(np.argmax([abs(mode_overlap(line, reference_mode, mode)) for mode in modes]))


def track_slab_mode(line, omega, reference_mode, n_modes, dt=None, normal_spacing=None):
    """The mode of line at omega [rad/s] continuing reference_mode (same cells, nearby frequency), phase not aligned:
    Rayleigh-quotient iteration from it (O(m) per step), else solve_slab_modes(n_modes) + track_mode."""
    _check_profiles(line, reference_mode)
    omega_tilde = numerical_frequency(omega, dt)
    omega = float(omega)
    dt = None if dt is None else float(dt)
    normal_spacing = None if normal_spacing is None else float(normal_spacing)
    active_cells, rho, diagonal, coupling, corner = _symmetric_system(line, omega_tilde)
    sqrt_rho, t_diagonal, t_coupling, t_corner = _symmetrised_tridiagonal(rho, diagonal, coupling, corner)
    start_vector = np.asarray(reference_mode.profile, dtype=np.complex128)[active_cells] * sqrt_rho
    predicted_eigenvalue = complex(reference_mode.k_tilde) ** 2 * (omega_tilde / reference_mode.omega_tilde) ** 2
    pair = None
    if active_cells.size > 0 and np.any(start_vector != 0):
        pair = _rayleigh_quotient_iteration(t_diagonal, t_coupling, t_corner, predicted_eigenvalue, start_vector)
    if pair is not None:
        k_tilde = complex(_transverse_wavenumber(pair[0]))
        k = complex(_propagation_constant(k_tilde, normal_spacing))
        mode = _normalised_mode(line, active_cells, sqrt_rho, pair[1], k_tilde, k, omega, omega_tilde,
                                _discrete_flux_factor(k, omega, normal_spacing, dt))
        if abs(mode_overlap(line, reference_mode, mode)) >= _TRACK_MIN_OVERLAP:
            return mode
    modes = solve_slab_modes(line, omega, n_modes, dt, normal_spacing)
    return modes[track_mode(line, reference_mode, modes)]


def align_phase(line, mode, reference_mode):
    """Copy of mode with its profile rotated so mode_overlap(line, mode, reference_mode) is real > 0 (mode itself when
    the overlap vanishes)."""
    overlap = mode_overlap(line, mode, reference_mode)
    if overlap == 0:
        return mode
    return replace(mode, profile=mode.profile * (abs(overlap) / overlap))


def group_index(line_at, omega, mode, n_modes, dt=None, normal_spacing=None, relative_step=1e-3):
    """C0 dk/domega by a central difference over line_at(omega (1 +- relative_step)) -> SlabLine, the mode followed by
    track_slab_mode (n_modes for its full-solve fallback)."""
    relative_step = float(relative_step)
    if not 0.0 < relative_step < 1.0:
        raise ValueError(f"relative_step must be in (0, 1), got {relative_step!r}")
    omega = float(omega)
    wavenumbers = []
    for step in (relative_step, -relative_step):
        shifted_omega = omega * (1.0 + step)
        wavenumbers.append(track_slab_mode(line_at(shifted_omega), shifted_omega, mode, n_modes, dt,
                                           normal_spacing).k)
    return complex(C0 * (wavenumbers[0] - wavenumbers[1]) / (2.0 * relative_step * omega))


def mode_amplitude_weights(line, mode, offset):
    """complex128 (2, m) weights (W_u, W_t) of mode_amplitudes: (a_positive, a_negative) = W_u @ U + W_t @ T' (the
    linear map the FDFD adjoint differentiates)."""
    _check_profiles(line, mode)
    profile = np.asarray(mode.profile, dtype=np.complex128)
    rho = _bilinear_weight(line)
    self_product = complex(np.sum(profile * profile * rho))
    u_weights = profile * rho / self_product
    impedance_factor = mode.k_tilde / (mode.omega_tilde * _field_constant(line))
    t_weights = profile * _stretched_widths(line) / (impedance_factor * self_product)
    phase = mode.k * float(offset)
    forward = cmath.exp(1j * phase)
    denominator = 2.0 * cmath.cos(phase)
    return (np.stack((u_weights * (forward / denominator), u_weights / (forward * denominator))),
            np.stack((t_weights / denominator, -t_weights / denominator)))


def mode_amplitudes(line, mode, out_of_plane, transverse_over_sign, offset):
    """(a_positive, a_negative) of the normalised mode from true-time phasors U (m,) = u at the centres and T' (m,) =
    t / sign_n sampled offset [m] further along +normal: U = (a+ + a-) u, T' = g u (a+ e^{-j k h} - a- e^{j k h})."""
    u_weights, t_weights = mode_amplitude_weights(line, mode, offset)
    n_cells = u_weights.shape[1]
    measured_u = np.asarray(out_of_plane, dtype=np.complex128)
    measured_t = np.asarray(transverse_over_sign, dtype=np.complex128)
    if measured_u.shape != (n_cells,) or measured_t.shape != (n_cells,):
        raise ValueError(f"out_of_plane / transverse_over_sign must have shape ({n_cells},), got {measured_u.shape} / "
                         f"{measured_t.shape}")
    a_positive, a_negative = u_weights @ measured_u + t_weights @ measured_t
    return complex(a_positive), complex(a_negative)
