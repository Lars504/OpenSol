"""Solid-conductor circuit rows and the host solve of the time-harmonic magnetic system (host numpy / scipy; tensor and
triangle meshes): per conductor k an applied field v_k (planar V/m, axisymmetric V / (2 pi)) with J = sigma (v_k / r -
j omega A) on its cells; a current drive adds one bordered row (scaled by 1 / (j omega): the system stays complex
symmetric), a voltage drive moves v_k to the rhs. RecycledFactor keeps the factor of the 'linear' sparse_direct
runtime and its recycled-factor GMRES re-solves."""
import math
from dataclasses import dataclass

import numpy as np

from ...solvers.host_direct import factor_label
from ...solvers.sparse_direct import SPLU_LIMIT_UNKNOWNS, SPLU_WARN_UNKNOWNS

CONDUCTORS = ('stranded', 'solid')   # coil kinds of magnetic_fields_ac
DRIVES = ('current', 'voltage')       # solid-conductor drives (stranded coils are always current-driven)
# Per-coil results (table columns after 'feature', scalars <coil>_<key>) of magnetic_fields_ac.
COIL_RESULT_KEYS = ('current_re', 'current_im', 'voltage_re', 'voltage_im', 'resistance', 'inductance', 'power')
_SMALLER_TRIANGLE_MODEL = 'coarsen the mesh away from the conductors, use linear elements or shrink the domain'


@dataclass(frozen=True, eq=False)
class ConductorRow:
    """One solid conductor on the free unknowns: positions (free index), weights g = int sigma phi / r there (tensor:
    sigma A / r per cell) [c], drive 'current' (value = I [A]) or 'voltage' (value = v_k), both complex; conductance G
    = int sigma / r dA (tensor: sum of the weights) and row_offset g . u of the fixed unknowns (border rhs)."""
    name: str
    positions: np.ndarray
    weights: np.ndarray
    drive: str
    value: complex
    conductance: float
    row_offset: complex = 0.0


def bordered_system(matrix, rhs, conductors, omega):
    """(matrix, rhs) with one row / column per current-driven conductor appended (in conductor order): column
    -weights, row -weights, diagonal G / (j omega), rhs I / (j omega) + row_offset; voltage drives add weights v_k to the
    rhs of their unknowns."""
    import scipy.sparse
    rhs = np.array(rhs, dtype=np.complex128)
    driven = [conductor for conductor in conductors if conductor.drive == 'current']
    for conductor in conductors:
        if conductor.drive == 'voltage':
            rhs[conductor.positions] += conductor.weights * conductor.value
    if not driven:
        return matrix, rhs
    n_free = matrix.shape[0]
    rows, columns, values = [], [], []
    border_rhs = np.empty(len(driven), dtype=np.complex128)
    for k, conductor in enumerate(driven):
        border_index = n_free + k
        rows += [conductor.positions, np.full(conductor.positions.size, border_index), [border_index]]
        columns += [np.full(conductor.positions.size, border_index), conductor.positions, [border_index]]
        values += [-conductor.weights, -conductor.weights, [conductor.conductance / (1j * omega)]]
        border_rhs[k] = conductor.value / (1j * omega) + conductor.row_offset
    size = n_free + len(driven)
    border = scipy.sparse.csr_matrix(
        (np.concatenate(values).astype(np.complex128), (np.concatenate(rows), np.concatenate(columns))),
        shape=(size, size))
    padded = scipy.sparse.bmat([[matrix, None], [None, scipy.sparse.csr_matrix((len(driven), len(driven)))]],
                               format='csr')
    return (padded + border).tocsr(), np.concatenate((rhs, border_rhs))


def coil_result_values(current, voltage, omega, conductor, drive) -> dict:
    """COIL_RESULT_KEYS of a complex current and loop voltage: Z = V / I -> resistance Re Z, inductance Im Z / omega,
    power Re(V I*) / 2, plus conductor and drive."""
    impedance = voltage / current if current != 0 else complex(math.nan, math.nan)
    return {'current_re': current.real, 'current_im': current.imag, 'voltage_re': voltage.real,
            'voltage_im': voltage.imag, 'resistance': impedance.real, 'inductance': impedance.imag / omega,
            'power': 0.5 * (voltage * np.conj(current)).real, 'conductor': conductor, 'drive': drive}


def coil_drive(name, conductor, drive, current, voltage, phase, turns, fill_factor) -> tuple:
    """(drive, complex value) of an AC coil after the shared checks (ValueError): a stranded coil is current-driven;
    value = current (or a solid conductor's voltage) x e^(j phase deg)."""
    if conductor not in CONDUCTORS or drive not in DRIVES:
        raise ValueError(f"coil '{name}': conductor must be one of {CONDUCTORS} and drive one of {DRIVES}, got "
                         f'{conductor!r}, {drive!r}')
    if int(turns) < 1 or not 0.0 < float(fill_factor) <= 1.0:
        raise ValueError(f"coil '{name}': need turns >= 1 and 0 < fill_factor <= 1, got {turns}, {fill_factor}")
    drive = 'current' if conductor == 'stranded' else drive
    rotation = complex(np.exp(1j * math.radians(float(phase))))
    return drive, (float(current) if drive == 'current' else float(voltage)) * rotation


def stranded_resistance(turns, fill_factor, area, path, sigma) -> float:
    """R_dc = turns^2 / (fill_factor area^2) sum(path / sigma) over the coil's conducting cells / elements (path: area x
    loop length per cell; sigma 0 cells skipped)."""
    return (turns ** 2 / (fill_factor * area ** 2)
            * float(np.sum(np.where(sigma > 0.0, path / np.where(sigma > 0.0, sigma, 1.0), 0.0))))


def triangle_size_issue(n_unknowns, order, back_end='scipy_splu'):
    """(level, text) of the complex FEM system of n_unknowns (dofs + circuit rows, Lagrange order) on the host factor of
    `back_end` (SuperLU; a MUMPS / PARDISO back end keeps the same limits, uncalibrated, the text naming its library)
    near ('warning') or over ('error') its limits, else None. CPU only; a cuDSS back end skips it."""
    text = f'{int(n_unknowns):,} complex unknowns (P{order} vector potential dofs and circuit rows)'
    library = factor_label(back_end)
    factors = 'its L + U factors' if library == 'SuperLU' else f'its {library} factors'
    if n_unknowns > SPLU_LIMIT_UNKNOWNS:
        return 'error', (f'{text} exceed the {SPLU_LIMIT_UNKNOWNS:,} limit of the host {library} solver: {factors} '
                         f'would fill the host memory; {_SMALLER_TRIANGLE_MODEL}')
    if n_unknowns > SPLU_WARN_UNKNOWNS:
        return 'warning', (f'{text} exceed ~{SPLU_WARN_UNKNOWNS:,}, the practical size of host {library}; '
                           f'{_SMALLER_TRIANGLE_MODEL}')
    return None


class RecycledFactor:
    """The factorisation of a bordered host system through a 'linear' sparse_direct runtime and its reuse: direct
    solves while the operator revision matches, after a change GMRES preconditioned by the old factor (recyclable
    SuperLU factors of the same size), else refactorised (a cuDSS factor on its plan)."""

    def __init__(self, host_solver):
        self.host_solver = host_solver
        self.factorization = None
        self.revision = None          # operator revision the factorisation belongs to
        self.last_solution = None     # GMRES start of the next recycled solve
        self.factor_bytes = None      # host L + U of the last factorisation (0: a device factor)

    def solve(self, matrix, rhs, revision) -> np.ndarray:
        """Host complex solution of matrix x = rhs (matrix: the operator of `revision`)."""
        factorization = self.factorization
        solution = None
        if factorization is not None and self.revision == revision:
            solution = factorization.solve_host(rhs)
        elif (factorization is not None and getattr(factorization, 'recyclable', False)
              and factorization.n_unknowns == rhs.size):
            start = (self.last_solution if self.last_solution is not None and self.last_solution.size == rhs.size
                     else None)
            solution, _iterations = self.host_solver.solve_recycled_host(factorization, matrix, rhs, start)
        if solution is None:
            # a cuDSS factor (has release) is refreshed on its plan (previous=); the system is complex symmetric: LDL^T
            device_factor = factorization if getattr(factorization, 'release', None) is not None else None
            self.factorization = self.host_solver.factorize_host(matrix, symmetric=True, previous=device_factor)
            if device_factor is not None and self.factorization is not device_factor:
                device_factor.release()   # replaced (new options or a SuperLU fallback); release is idempotent
            self.revision = revision
            self.factor_bytes = self.factorization.host_bytes or 0
            solution = self.factorization.solve_host(rhs)
        self.last_solution = solution
        return solution
