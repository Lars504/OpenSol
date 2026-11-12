"""'sparse_direct' solver type: sparse direct solves on the host (SuperLU 'scipy_splu', one factorisation reused for
every right-hand side) or the device (cuSOLVER QR 'cusolver_qr', one factorisation per solve; cuDSS 'cudss' /
'cudss_mixed', a kept device factor, solvers.cudss_factor), size rules and the SparseDirectSolver runtime of the
'linear' and 'shifted' roles. numpy only at module level (cupy, scipy, nvmath inside)."""
# Unpreconditioned Krylov (cupyx cg / gmres) is not offered: it stalls on the indefinite Helmholtz matrix. GMRES runs
# only preconditioned by an earlier SuperLU factor (Factorization.solve_preconditioned). Host users (cross-section
# modes, eigenfrequency, thermal noise, magnetic AC) factorise through factorize_host / solve_host: the SuperLU options,
# the host-byte count and the size rule (size_issue with unknowns_per_cell) live here only.
# cuDSS back ends: offered to a role only when its setting_choices['back_end'] lists them (the SolverType's
# opt_in_settings), never by 'auto'; factorize / factorize_host return a cudss_factor.CudssFactorization (solve /
# solve_host, transposed solves of symmetric factors only, not recyclable: re-solves after a value change refactorise
# with previous=, reusing the cuDSS plan of an unchanged pattern). The factor lives outside the cupy pool and is guarded
# by cuDSS's own estimate after planning, so the host size rules do not apply to it.
# Host direct back ends (mumps / mumps_mixed / pardiso / pardiso_mixed, solvers.host_direct): opt-in like cuDSS (never
# 'auto'), on both backends; factorize / factorize_host / DiffusionDirectSolver return a
# host_direct.HostDirectFactorization (transposed solves, recyclable, inertia; previous= refactorises on the kept
# analysis). They keep the SuperLU size rules (library-named texts; uncalibrated, LDL^T / nested dissection usually
# allow more) plus the library's own memory estimate after the analysis; a refusal raises (no fallback).
import importlib.util
import inspect
import math
import time

import numpy as np

from .cudss_factor import csr_rows
from .host_direct import (HOST_DIRECT_BACK_ENDS, HOST_DIRECT_MIXED_BACK_ENDS, factor_label, host_direct_available,
                          host_direct_factorization, host_direct_values_factorization, is_host_direct_factor,
                          missing_text)

CUDSS_BACK_ENDS = ('cudss', 'cudss_mixed')
BACK_ENDS = ('scipy_splu', 'cusolver_qr') + CUDSS_BACK_ENDS + HOST_DIRECT_BACK_ENDS
LINEAR_SOLVER_CHOICES = ('auto',) + BACK_ENDS
SYMMETRIC_OPTION_BACK_ENDS = CUDSS_BACK_ENDS + HOST_DIRECT_BACK_ENDS   # factorize honours symmetric / symmetrizer /
                                                                       # row_scale / previous
MIXED_BACK_ENDS = ('cudss_mixed',) + HOST_DIRECT_MIXED_BACK_ENDS       # a float32 / complex64 factor + refinement
HOST_FACTOR_BACK_ENDS = ('scipy_splu',) + HOST_DIRECT_BACK_ENDS        # factors in host memory (SuperLU size rules)
# run-time recipe of the host direct back ends: the SuperLU items (HostLU / HostSolve on the cpu profile) times these
# (uncalibrated: the multicore speed-up of MUMPS / PARDISO over SuperLU is not measured; profile curves not yet), and
# solves of a *_mixed factor x HOST_DIRECT_MIXED_FULL_SOLVES (low-precision solves + refinement, as
# CUDSS_MIXED_FULL_SOLVES; uncalibrated)
HOST_DIRECT_FACTOR_TIME = 1.0
HOST_DIRECT_SOLVE_TIME = 1.0
HOST_DIRECT_MIXED_FULL_SOLVES = 4.0
# run-time recipe of the cuDSS back ends (2D: the profile's complex 5-point cudss curves; 3D: the arpack_shift_invert
# fit of complex 3D curl-curl factors), scaled for real operators by these value-size / flop ratios (uncalibrated
# guesses: scalar 7-point and FEM fills differ from the curl-curl one), and (3D only; the 2D mixed solve curve holds its
# refinement) complex64 / float32 solves per full-accuracy cudss_mixed solve (~3 refinement steps from an unrefined
# 1e-4 at ~3e3 per step; uncalibrated)
CUDSS_REAL_BYTES = 0.5
CUDSS_REAL_FLOPS = 0.25
CUDSS_MIXED_FULL_SOLVES = 4.0
# A plan that finds a kept ordering (cudss_factor "Ordering reuse": no host reordering, symbolic phase and upload only)
# over a fresh plan: romero2019 E2a 0.26 / 4.79 s, hirsch2025 E5 (R1a, upload included) 1.05 / 4.5 s; the larger share
# kept (uncalibrated: two cases)
CUDSS_KEPT_ORDERING_PLAN_FRACTION = 0.25
# Resolved-values key schema.auto_back_end_values sets when it replaced back_end 'auto' by a role's cuDSS back end: the
# runtime then falls back to host SuperLU when cuDSS refuses an operator (B1b; shown in summary 'settings')
AUTO_BACK_END_KEY = 'back_end_from_auto'
# Size guards (validation errors and solver refusals alike). An overrun does not fail cleanly: the cusolver_qr workspace
# bypasses the cupy pool budget, and overrunning the device freezes the display (Windows pages the desktop's GPU memory
# out) and can crash the driver; oversized SuperLU factors swap the host. So these sizes are refused, never attempted.
SPLU_WARN_UNKNOWNS = 1_000_000        # nx * ny incl. PML
SPLU_LIMIT_UNKNOWNS = 2_000_000       # L + U of order 10 GB of host memory
# Real 3D 7-point diffusion systems (DiffusionDirectSolver): fill ~ n^(4/3) (core.memory_estimate.host_real_lu_bytes),
# so SuperLU stops paying off far earlier than in 2D; above these Jacobi-PCG / multigrid is the faster solver
SPLU_3D_WARN_UNKNOWNS = 100_000
SPLU_3D_LIMIT_UNKNOWNS = 400_000
# Complex 3D curl-curl systems (em_fdfd_3d, E on the live Yee faces, 13 entries per row): no iterative alternative yet,
# so the limits follow the factor memory only; first values (igor3d_plan decision 11), to lower from measured factors
SPLU_VECTOR_3D_WARN_UNKNOWNS = 1_000_000
SPLU_VECTOR_3D_LIMIT_UNKNOWNS = 4_000_000
_TO_PCG = ("assign a pcg solver node as the physics' linear solver (preconditioner multigrid on graded or strongly "
           "contrasting meshes), or coarsen the mesh")
_SMALLER_3D_MODEL = ('coarsen the mesh, shrink the padding or the PML, or halve the model by mirror planes (each mirror '
                     'face halves the unknowns)')
QR_WARN_BYTES = 512 * 2 ** 20         # estimated QR workspace: solves take tens of seconds or more
QR_LIMIT_BYTES = 2 * 2 ** 30          # QR time ~ nx ny band^2: ~2 min per solve here on an RTX 3050 (extrapolated)
# Assumed fill of the sparse QR in index order (qr_band_order) per row and band column: R (2 b) + Householder vectors
# (b), complex128 values + int32 indices. By it the 36,100-unknown Hankel self-test (~0.4 GB) solves in ~15 s on an RTX 3050, and the
# 333k-unknown TFLN ring (~12 GB) froze that machine.
_QR_BYTES_PER_BAND_ENTRY = 64
_QR_DEVICE_FRACTION = 0.5             # largest share of the free device memory the QR workspace may claim
_SPLU_PIVOT_THRESHOLD = 0.1           # SuperLU takes the diagonal pivot while |a_jj| >= this * column max
# grid_nested_dissection: boxes of at most this many cells (or < 3 cells along every axis) stay in index order
ND_LEAF_CELLS = 64
# A SuperLU factor of an earlier operator at the same omega (thermo-optic Picard step: A changes slightly) serves as
# GMRES preconditioner while GMRES reaches RECYCLE_RTOL (true residual ||b - A x|| / ||b||) within this many iterations;
# the caller refactorises otherwise. 25 back-substitutions cost far less than one O(N^1.5) factorisation.
RECYCLE_MAX_ITERATIONS = 25
RECYCLE_RTOL = 1e-10
_SMALLER_MODEL = ('coarsen the mesh or shrink the domain (large models: em_fdtd_ez / em_fdtd_hz in a time_dependent '
                  'study, DFT-based monitors)')
_SCIPY_ALTERNATIVES = ("pip install scipy, or use a sparse_direct solver node with back_end 'cusolver_qr' (device QR, "
                       "up to ~1e5 unknowns)")


def cudss_precision(back_end) -> str:
    """CudssFactorization precision of a cuDSS back end: 'mixed' for cudss_mixed, else 'double'."""
    return 'mixed' if back_end == 'cudss_mixed' else 'double'


def require_backend_operator(label, backend, operator) -> None:
    """TypeError unless the CSR operator lives on `backend` ('cpu': scipy, 'gpu': cupyx)."""
    if isinstance(operator.data, np.ndarray) != (backend == 'cpu'):
        raise TypeError(f"{label}: the {backend} backend needs a {'scipy' if backend == 'cpu' else 'cupyx'} operator, "
                        f'got {type(operator).__module__}.{type(operator).__name__}')


def scipy_available() -> bool:
    """True when scipy is installed (find_spec only, nothing imported: safe on the GUI thread)."""
    return importlib.util.find_spec('scipy') is not None


def expected_back_end(choice, backend='gpu') -> str:
    """Back end a back_end choice resolves to on this machine, without raising (CPU, estimators / validation): 'auto' ->
    'scipy_splu' when scipy is installed or on the cpu backend, else 'cusolver_qr' (never a cuDSS back end)."""
    if choice == 'auto':
        return 'scipy_splu' if scipy_available() or backend == 'cpu' else 'cusolver_qr'
    return choice


def backend_misfit(values, backend) -> str:
    """SolverType.backend_misfit of sparse_direct: '' unless a GPU back end (cusolver_qr, cudss, cudss_mixed) is asked
    of the cpu backend (the host direct back ends run on both)."""
    back_end = values.get('back_end')
    if backend == 'cpu' and (back_end == 'cusolver_qr' or back_end in CUDSS_BACK_ENDS):
        return f"back_end '{back_end}' runs on the GPU only (use 'auto' or 'scipy_splu' on the CPU backend)"
    return ''


def qr_band_order(nx, ny, periodic_x, periodic_y):
    """(bandwidth, column_major) of the cusolver_qr unknown order: the narrower of i*ny + j (band ny, doubled by a
    periodic x wrap) and column-major j*nx + i (band nx, doubled by a periodic y wrap)."""
    band_row_major = int(ny) * (2 if periodic_x else 1)
    band_column_major = int(nx) * (2 if periodic_y else 1)
    if band_column_major < band_row_major:
        return band_column_major, True
    return band_row_major, False


def qr_workspace_bytes(nx, ny, periodic_x, periodic_y) -> int:
    """Estimated cusolver_qr device workspace [bytes] of an nx * ny system: band fill of a QR in the qr_band_order
    index order (cuSOLVER's device QR is believed not to reorder)."""
    band, _column_major = qr_band_order(nx, ny, periodic_x, periodic_y)
    return _QR_BYTES_PER_BAND_ENTRY * int(nx) * int(ny) * band


def _format_bytes(n) -> str:
    from ..core.gpu_memory import format_bytes   # core.gpu_memory imports cupy
    return format_bytes(n)


def size_issue(back_end, nx, ny, periodic_x, periodic_y, unknowns_per_cell=1, remedy=_SMALLER_MODEL):
    """(level, text) for a system of unknowns_per_cell * nx * ny unknowns (PML included) that `back_end` refuses
    ('error': validation and the solver alike) or that is close to its limit ('warning'); None otherwise (always for
    the cuDSS back ends: guarded by cuDSS's own device estimate after planning). The host direct back ends (MUMPS /
    PARDISO) keep the SuperLU limits (uncalibrated: LDL^T and nested dissection usually allow more), with
    unknowns_per_cell allowed. CPU only."""
    if back_end in CUDSS_BACK_ENDS:
        return None
    n_unknowns = int(unknowns_per_cell) * int(nx) * int(ny)
    scalar = unknowns_per_cell == 1
    components = '' if scalar else f' x {unknowns_per_cell} field components'
    cells = f'{n_unknowns:,} unknowns ({nx} x {ny} cells{components}, PML included)'
    if back_end in HOST_DIRECT_BACK_ENDS:
        label = factor_label(back_end)
        solver_text, factors, growth = (f"the sparse_direct back end '{back_end}' ({label})", f'its {label} factors',
                                        f'host memory for the {label} factors')
    else:
        solver_text = "the sparse_direct back end 'scipy_splu'" if scalar else 'the host SuperLU solver'
        factors, growth = 'its L + U factors', 'host memory for the SuperLU factors'
    if back_end in HOST_FACTOR_BACK_ENDS:
        if n_unknowns > SPLU_LIMIT_UNKNOWNS:
            return 'error', (f'{cells} exceed the {SPLU_LIMIT_UNKNOWNS:,} limit of {solver_text}: {factors} '
                             f'would fill the host memory and swap the machine to a halt; {remedy}')
        if n_unknowns > SPLU_WARN_UNKNOWNS:
            return 'warning', (f'{cells} exceed ~{SPLU_WARN_UNKNOWNS:,}, the practical size of {solver_text} '
                               f'({growth} grows faster than the unknowns); {remedy}')
        return None
    if unknowns_per_cell != 1:
        raise ValueError(f"size_issue: back end {back_end!r} sizes scalar (one unknown per cell) systems only")
    workspace = qr_workspace_bytes(nx, ny, periodic_x, periodic_y)
    if workspace <= QR_WARN_BYTES:
        return None
    estimate = (f"{cells} need an estimated {_format_bytes(workspace)} of GPU workspace with the sparse_direct back end "
                f"'cusolver_qr'")
    if workspace > QR_LIMIT_BYTES:
        return 'error', (f'{estimate}, over its {_format_bytes(QR_LIMIT_BYTES)} limit: the workspace bypasses the GPU '
                         f'memory budget, and overrunning the device freezes the display or crashes the driver; '
                         f'{_to_superlu_text()}, or {_SMALLER_MODEL}')
    return 'warning', (f'{estimate}, refactorised in every solve (tens of seconds or more per solve); '
                       f'{_to_superlu_text()}, or {_SMALLER_MODEL}')


def host_size_back_end(name) -> str:
    """The sparse_direct back end whose size rule applies to a back end / eigen factorization name: 'superlu', 'auto'
    and None -> 'scipy_splu' (host SuperLU), every other name itself (host direct names keep the SuperLU limits in
    size_issue, cuDSS names have none)."""
    return 'scipy_splu' if name in (None, 'auto', 'superlu') else name


def diffusion_size_issue(n_unknowns, dimensions=2, back_end='scipy_splu'):
    """(level, text) for a real DiffusionSystem of n_unknowns window cells (3D: voxels) on host SuperLU (the
    sparse_direct 'linear' solver of heat / potential / magnetostatics) near ('warning') or over ('error') its size
    rule; None otherwise. A host direct back_end (MUMPS / PARDISO) has the same rule, its text naming the library.
    CPU only."""
    n_unknowns = int(n_unknowns)
    warn, limit = ((SPLU_3D_WARN_UNKNOWNS, SPLU_3D_LIMIT_UNKNOWNS) if dimensions == 3
                   else (SPLU_WARN_UNKNOWNS, SPLU_LIMIT_UNKNOWNS))
    cells = f"{n_unknowns:,} window {'voxels' if dimensions == 3 else 'cells'}"
    if back_end in HOST_DIRECT_BACK_ENDS:
        label = factor_label(back_end)
        solver, factors, practical = (f"(host {label}, back_end '{back_end}')", f'its {label} factors',
                                      f"(host {label}, back_end '{back_end}')")
    else:
        solver, factors, practical = ('(host SuperLU, the Automatic choice on the cpu backend)', 'its L + U factors',
                                      '(host SuperLU)')
    if n_unknowns > limit:
        return 'error', (f'{cells} exceed the {limit:,} limit of the sparse_direct linear solver {solver}: {factors} '
                         f'would fill the host memory; {_TO_PCG}')
    if n_unknowns > warn:
        return 'warning', (f'{cells} exceed ~{warn:,}, the practical size of the sparse_direct linear solver '
                           f'{practical}: factor memory and time grow faster than the cells; {_TO_PCG}')
    return None


def vector_3d_size_issue(n_unknowns):
    """(level, text) for a complex 3D curl-curl system of n_unknowns E unknowns on host SuperLU (em_fdfd_3d) near
    ('warning') or over ('error') SPLU_VECTOR_3D_*; None otherwise. CPU only."""
    n_unknowns = int(n_unknowns)
    unknowns = f'{n_unknowns:,} E unknowns'
    if n_unknowns > SPLU_VECTOR_3D_LIMIT_UNKNOWNS:
        return 'error', (f'{unknowns} exceed the {SPLU_VECTOR_3D_LIMIT_UNKNOWNS:,} limit of the host SuperLU '
                         f'factor of the 3D curl-curl system: its L + U factors would fill the host memory; '
                         f'{_SMALLER_3D_MODEL}')
    if n_unknowns > SPLU_VECTOR_3D_WARN_UNKNOWNS:
        return 'warning', (f'{unknowns} exceed ~{SPLU_VECTOR_3D_WARN_UNKNOWNS:,}: the host SuperLU factor of the 3D '
                           f'curl-curl system grows ~ n^(4/3) (tens of GB and minutes here); {_SMALLER_3D_MODEL}')
    return None


def check_solver_size(back_end, nx, ny, periodic_x, periodic_y) -> None:
    """ValueError for a system size_issue refuses, or (cusolver_qr) whose estimated workspace exceeds half the free
    device memory; only above QR_WARN_BYTES, measured after freeing the cupy pool's cached blocks."""
    issue = size_issue(back_end, nx, ny, periodic_x, periodic_y)
    if issue is not None and issue[0] == 'error':
        raise ValueError(issue[1])
    if back_end != 'cusolver_qr':
        return
    workspace = qr_workspace_bytes(nx, ny, periodic_x, periodic_y)
    if workspace <= QR_WARN_BYTES:
        return
    import cupy as cp
    from ..core.gpu_memory import memory_status
    cp.get_default_memory_pool().free_all_blocks()   # e.g. the previous omega's dropped system, an earlier FDTD run
    device_free = memory_status().device_free
    if workspace > _QR_DEVICE_FRACTION * device_free:
        raise ValueError(f"the sparse_direct back end 'cusolver_qr' needs an estimated {_format_bytes(workspace)} of "
                         f'GPU workspace for {nx} x {ny} cells, over half the free device memory '
                         f'({_format_bytes(device_free)}): refused before it could overrun the device and freeze the '
                         f'display; {_to_superlu_text()}, or {_SMALLER_MODEL}')


def _to_superlu_text():
    """Remedy text pointing a cusolver_qr model to scipy_splu (installing scipy when it is missing)."""
    if scipy_available():
        return "use Automatic or a sparse_direct node with back_end 'auto' or 'scipy_splu' (host SuperLU, ~1e6 unknowns)"
    return "pip install scipy so that back_end 'auto' uses 'scipy_splu' (host SuperLU, ~1e6 unknowns)"


def resolve_back_end(choice, backend='gpu') -> str:
    """Back end of a back_end choice: 'auto' -> 'scipy_splu' when scipy is installed (always on the cpu backend), else
    'cusolver_qr'; ValueError for an unknown choice, 'scipy_splu' without scipy, a GPU back end on the cpu, a cuDSS
    one without nvmath-python or a host direct one without its wrapper (host_direct.missing_text)."""
    if choice not in LINEAR_SOLVER_CHOICES:
        raise ValueError(f"back_end must be one of {', '.join(LINEAR_SOLVER_CHOICES)}, got {choice!r}")
    misfit = backend_misfit({'back_end': choice}, backend)
    if misfit:
        raise ValueError(misfit)
    resolved = expected_back_end(choice, backend)
    if resolved == 'scipy_splu' and not scipy_available():
        alternatives = 'pip install scipy (the CPU backend needs it)' if backend == 'cpu' else _SCIPY_ALTERNATIVES
        raise ValueError(f"back_end 'scipy_splu' needs scipy, which is not installed: {alternatives}")
    if resolved in CUDSS_BACK_ENDS:
        from .cudss_factor import cudss_available
        if not cudss_available():
            raise ValueError(f"back_end '{resolved}' needs nvmath-python with cuDSS, which is not installed: pip install "
                             "nvmath-python[cu12] (or [cu13]), or use 'auto' / 'scipy_splu'")
    if resolved in HOST_DIRECT_BACK_ENDS and not host_direct_available(resolved):
        raise ValueError(missing_text(resolved))
    return resolved


class Factorization:
    """Factorised square operator of one back end; solve(rhs) maps a cupy complex128 (n,) right-hand side to the cupy
    complex128 (n,) solution, numpy to numpy when host_arrays (a host operator: the cpu backend); scipy_splu also
    solve_host (numpy in, numpy out; float64 for a real factorize_host). host_bytes = SuperLU L + U size."""

    def __init__(self, back_end, n_unknowns, host_bytes, host_lu=None, device_operator=None,
                 column_major_shape=None, host_arrays=False, value_dtype=np.complex128, ordering=None):
        self.back_end = back_end
        self.n_unknowns = n_unknowns
        self.host_bytes = host_bytes
        self.value_dtype = value_dtype   # solve_host dtype: complex128, float64 for a real host factor
        self.host_arrays = bool(host_arrays)   # solve / solve_preconditioned take and return numpy (cpu backend)
        self._host_lu = host_lu
        self._ordering = ordering        # host_lu factors A[ordering][:, ordering] (factorize_host ordering), or None
        self._device_operator = device_operator
        self._column_major_shape = column_major_shape   # (nx, ny) when device_operator is on the j*nx + i index

    def _lu_solve(self, rhs, trans='N'):
        """numpy x of A x = rhs (trans 'T': A^T) by the host SuperLU factor, through its symmetric pre-ordering."""
        if self._ordering is None:
            return self._host_lu.solve(rhs, trans=trans)
        solution = np.empty_like(rhs)
        solution[self._ordering] = self._host_lu.solve(rhs[self._ordering], trans=trans)
        return solution

    def solve(self, rhs, transpose=False):
        """Solution x of A x = rhs (transpose: A^T x = rhs, unconjugated, the adjoint solve); scipy_splu moves rhs to
        the host and x back (one transfer each; none with host_arrays)."""
        if self.host_arrays:   # solve_host: a real factor (factorize_host real=True) splits a complex rhs
            return self.solve_host(np.asarray(rhs, dtype=np.complex128), transpose=transpose)
        import cupy as cp
        rhs = cp.asarray(rhs, dtype=cp.complex128)
        if rhs.shape != (self.n_unknowns,):
            raise ValueError(f"rhs must be a flat ({self.n_unknowns},) array on the i*ny + j index, got shape "
                             f"{rhs.shape}")
        if self._host_lu is not None:
            return cp.asarray(self.solve_host(cp.asnumpy(rhs), transpose=transpose))
        import cupyx.scipy.sparse.linalg
        # Assumption (cupy 13.3): spsolve = cuSOLVER csrlsvqr whose Z variant takes complex128 CSR; the device variant
        # is believed not to reorder (fill follows the index order, hence qr_band_order); its workspace lives outside
        # the cupy pool. QR keeps no factor, so the transposed solve factorises A^T (same band: P A^T P^T).
        operator = self._device_operator.T.tocsr() if transpose else self._device_operator
        if self._column_major_shape is None:
            return cupyx.scipy.sparse.linalg.spsolve(operator, rhs)
        nx, ny = self._column_major_shape
        solution = cupyx.scipy.sparse.linalg.spsolve(operator, rhs.reshape(nx, ny).T.ravel())
        return solution.reshape(ny, nx).T.ravel()

    def solve_host(self, rhs, transpose=False):
        """numpy solution x (value_dtype) of A x = rhs (transpose: A^T, unconjugated) for a numpy (n,) rhs, all on the
        host (scipy_splu only); a real factor solves a complex rhs as two real solves (complex128 solution)."""
        if self._host_lu is None:
            raise ValueError("solve_host needs a 'scipy_splu' factorisation (cusolver_qr keeps no host factor)")
        rhs = np.asarray(rhs)
        if rhs.shape != (self.n_unknowns,):
            raise ValueError(f"rhs must be a flat ({self.n_unknowns},) array, got shape {rhs.shape}")
        trans = 'T' if transpose else 'N'
        if rhs.dtype.kind == 'c' and np.dtype(self.value_dtype).kind != 'c':   # a cast would drop the imaginary part
            return self._lu_solve(np.ascontiguousarray(rhs.real), trans) + 1j * self._lu_solve(
                np.ascontiguousarray(rhs.imag), trans)
        return self._lu_solve(rhs.astype(self.value_dtype, copy=False), trans)

    def inertia(self):
        """(n_positive, n_negative) of a symmetric / Hermitian A from its SuperLU factor (superlu_inertia); None
        without a host factor or after an off-diagonal pivot."""
        return None if self._host_lu is None else superlu_inertia(self._host_lu)

    @property
    def recyclable(self) -> bool:
        """True when solve_preconditioned can reuse this factorisation for a changed operator (scipy_splu only)."""
        return self._host_lu is not None

    def solve_preconditioned(self, operator, rhs, initial_guess=None, transpose=False):
        """(x, iterations) of operator x = rhs (transpose: operator^T x = rhs; cupyx CSR, cupy rhs) by host GMRES
        preconditioned with this factorisation of an earlier, nearby operator; x None above RECYCLE_RTOL."""
        if self._host_lu is None:
            raise ValueError("solve_preconditioned needs a 'scipy_splu' factorisation (cusolver_qr keeps no factor)")
        if self.host_arrays:   # scipy operator, numpy rhs / guess / solution
            return self.solve_preconditioned_host(operator, rhs, initial_guess, transpose=transpose)
        import cupy as cp
        host_rhs = cp.asnumpy(cp.asarray(rhs, dtype=cp.complex128))
        if host_rhs.shape != (self.n_unknowns,):
            raise ValueError(f"rhs must be a flat ({self.n_unknowns},) array on the i*ny + j index, got shape "
                             f"{host_rhs.shape}")
        host_operator = operator.get().tocsr().astype(np.complex128)
        if transpose:
            host_operator = host_operator.T.tocsr()
        initial = None if initial_guess is None else cp.asnumpy(cp.asarray(initial_guess, dtype=cp.complex128)).ravel()
        solution, iterations = self._recycled_gmres(host_operator, host_rhs, initial, 'T' if transpose else 'N')
        return (None if solution is None else cp.asarray(solution)), iterations

    def solve_preconditioned_host(self, matrix, rhs, initial_guess=None, transpose=False):
        """(x, iterations) of matrix x = rhs (transpose: matrix^T x = rhs), all host numpy / scipy, by GMRES
        preconditioned with this factorisation of an earlier, nearby matrix; x None above RECYCLE_RTOL (scipy_splu)."""
        if self._host_lu is None:
            raise ValueError("solve_preconditioned_host needs a 'scipy_splu' factorisation")
        rhs = np.asarray(rhs, dtype=np.complex128)
        if rhs.shape != (self.n_unknowns,):
            raise ValueError(f"rhs must be a flat ({self.n_unknowns},) array, got shape {rhs.shape}")
        initial = None if initial_guess is None else np.asarray(initial_guess, dtype=np.complex128).ravel()
        if transpose:
            matrix = matrix.T
        return self._recycled_gmres(matrix.tocsr().astype(np.complex128), rhs, initial, 'T' if transpose else 'N')

    def _recycled_gmres(self, host_operator, host_rhs, initial, trans):
        """recycled_gmres preconditioned by this SuperLU factor (_lu_solve)."""
        return recycled_gmres(self._lu_solve, self.n_unknowns, host_operator, host_rhs, initial, trans)


def recycled_gmres(lu_solve, n_unknowns, host_operator, host_rhs, initial, trans):
    """(host x or None, iterations) of host GMRES on host_operator (one restart cycle of RECYCLE_MAX_ITERATIONS)
    preconditioned by lu_solve(vector, trans=trans) of a factor of an earlier operator (SuperLU Factorization,
    host_direct.HostDirectFactorization); None when the true residual misses RECYCLE_RTOL."""
    import scipy.sparse.linalg
    rhs_norm = float(np.linalg.norm(host_rhs))
    if rhs_norm == 0.0:
        return np.zeros(n_unknowns, dtype=np.complex128), 0
    preconditioner = scipy.sparse.linalg.LinearOperator(
        host_operator.shape, matvec=lambda vector: lu_solve(vector, trans=trans), dtype=np.complex128)
    iteration_count = [0]

    def count_iteration(_residual_norm):
        iteration_count[0] += 1

    # scipy >= 1.12 names the relative tolerance rtol (tol removed in 1.14); asked 10x tighter because scipy may
    # test a preconditioned residual: the true residual below decides.
    tolerance_name = 'rtol' if 'rtol' in inspect.signature(scipy.sparse.linalg.gmres).parameters else 'tol'
    solution, _info = scipy.sparse.linalg.gmres(
        host_operator, host_rhs, x0=initial, restart=RECYCLE_MAX_ITERATIONS, maxiter=1, M=preconditioner,
        callback=count_iteration, callback_type='pr_norm', atol=0.0, **{tolerance_name: 0.1 * RECYCLE_RTOL})
    residual_norm = float(np.linalg.norm(host_rhs - host_operator @ solution))
    if not residual_norm <= RECYCLE_RTOL * rhs_norm:   # also NaN
        return None, iteration_count[0]
    return solution, iteration_count[0]


def factorize(operator, back_end, nx, ny, periodic_x, periodic_y, previous=None, symmetric=False,
              symmetrizer=None, orderings=None):
    """Factorization of a square cupyx CSR operator of an nx * ny grid (cast to complex128): scipy_splu factorises on
    the host now and keeps no device copy (orderings: a SuperLUOrdering reused / kept), cusolver_qr keeps the operator
    (in qr_band_order) and factorises per solve, cudss / cudss_mixed give a CudssFactorization (previous / symmetric /
    symmetrizer: cudss_factorization; ignored by the others), the host direct back ends a
    host_direct.HostDirectFactorization of a host copy (same options, host_arrays for a scipy operator). A scipy
    operator (cpu backend: scipy_splu or a host direct back end) gives numpy solves (host_arrays)."""
    if back_end not in BACK_ENDS:
        raise ValueError(f"back_end must be one of {', '.join(BACK_ENDS)} (resolve_back_end maps 'auto'), got "
                         f"{back_end!r}")
    n_rows, n_columns = operator.shape
    if n_rows != n_columns:
        raise ValueError(f"factorize needs a square operator, got shape {operator.shape}")
    if n_rows != int(nx) * int(ny):
        raise ValueError(f"factorize: operator of {n_rows} unknowns does not match the {nx} x {ny} grid")
    if back_end in CUDSS_BACK_ENDS:
        if not isinstance(operator.data, np.ndarray) and _refreshed_on_device(previous, operator, back_end, symmetric,
                                                                               symmetrizer):
            return previous
        # a first factor: cuDSS equilibrates and checks symmetry on a host copy
        host_operator = operator if isinstance(operator.data, np.ndarray) else operator.get()
        return cudss_factorization(host_operator, back_end, previous=previous, symmetric=symmetric,
                                   symmetrizer=symmetrizer)
    if back_end in HOST_DIRECT_BACK_ENDS:   # one device-to-host transfer of a cupyx operator, complex128 factor
        on_host = isinstance(operator.data, np.ndarray)
        factor = host_direct_factorization(operator if on_host else operator.get(), back_end, previous=previous,
                                           symmetric=symmetric, symmetrizer=symmetrizer)
        factor.host_arrays = on_host
        return factor
    if isinstance(operator.data, np.ndarray):   # host scipy operator
        if back_end != 'scipy_splu':
            raise ValueError(f"factorize: a host (scipy) operator needs back_end 'scipy_splu' or a host direct one "
                             f"({', '.join(HOST_DIRECT_BACK_ENDS)}), got {back_end!r}")
        factor = _ordered_superlu(operator, np.complex128, orderings)
        factor.host_arrays = True
        return factor
    import cupy as cp
    if operator.dtype != cp.complex128:
        operator = operator.astype(cp.complex128)
    if back_end == 'cusolver_qr':
        _band, column_major = qr_band_order(nx, ny, periodic_x, periodic_y)
        if not column_major:
            return Factorization(back_end, int(n_rows), None, device_operator=operator.tocsr())
        return Factorization(back_end, int(n_rows), None, device_operator=_column_major_operator(operator, nx, ny),
                             column_major_shape=(int(nx), int(ny)))
    # one device-to-host transfer, no device copy kept
    return _ordered_superlu(operator.get(), np.complex128, orderings)


def _refreshed_on_device(previous, operator, back_end, symmetric, symmetrizer) -> bool:
    """True when a valid cuDSS `previous` of the same options was refactorised from the device values of the cupyx
    `operator` (Sp3: rows x symmetrizer, equilibration, asymmetry check and lower take on the device; no host copy,
    scipy transpose or lower-triangle take: ~250 of ~370 ms per FDFD frequency on the host route), else False."""
    from .cudss_factor import CudssFactorization
    if not isinstance(previous, CudssFactorization) or previous.n_unknowns != operator.shape[0]:
        return False
    precision = cudss_precision(back_end)
    if not previous.can_refactorize(False, precision, symmetric, False, symmetrizer):
        return False
    if not operator.has_canonical_format:
        operator = operator.copy()
        operator.sum_duplicates()
    previous.refactorize_device_values(operator.data, *previous.host_pattern_of(operator), check_symmetry=True,
                                       symmetrizer=symmetrizer)
    return True


def factorize_host(host_operator, real=False, ordering=None, orderings=None) -> Factorization:
    """'scipy_splu' Factorization of a square scipy sparse operator already on the host (cast to complex128 CSC, float64
    with real: a real operator, e.g. a symmetric stiffness pencil), with the FDFD SuperLU options and host-byte count;
    solve it with solve_host. ordering: (n,) elimination order of the unknowns (grid_nested_dissection), None = MMD
    (orderings: a SuperLUOrdering whose kept order of the same pattern replaces the MMD pass, kept otherwise)."""
    n_rows, n_columns = host_operator.shape
    if n_rows != n_columns:
        raise ValueError(f"factorize_host needs a square operator, got shape {host_operator.shape}")
    if real and np.iscomplexobj(host_operator.data):
        raise ValueError('factorize_host(real=True) needs a real operator')
    value_dtype = np.float64 if real else np.complex128
    if ordering is None:
        return _ordered_superlu(host_operator, value_dtype, orderings)
    ordering = np.asarray(ordering, dtype=np.int64)
    if ordering.shape != (n_rows,) or not np.array_equal(np.bincount(ordering, minlength=n_rows), np.ones(n_rows)):
        raise ValueError(f'factorize_host: ordering must be a permutation of the {n_rows} unknowns')
    return _superlu_factorization(symmetric_permutation(host_operator, ordering).astype(value_dtype),
                                  value_dtype=value_dtype, ordering=ordering)


class SuperLUOrdering:
    """Elimination order of the last host SuperLU of one sparsity pattern, reused by later factorisations of that
    pattern (permc_spec NATURAL on the pre-permuted operator: no minimum-degree pass); any order is exact, only the fill
    depends on it."""

    def __init__(self):
        self._pattern = None    # (indptr, indices) copies of the CSR the order belongs to
        self.ordering = None    # (n,) int64 elimination order (factorize_host ordering)
        self.n_reused = 0

    def lookup(self, host_csr):
        """The kept order when host_csr has the remembered pattern, else None."""
        if self.ordering is None or self.ordering.size != host_csr.shape[0]:
            return None
        indptr, indices = self._pattern
        if not (np.array_equal(indptr, host_csr.indptr) and np.array_equal(indices, host_csr.indices)):
            return None
        self.n_reused += 1
        return self.ordering

    def remember(self, host_csr, ordering) -> None:
        """Keep `ordering` for host_csr's pattern (copied: callers may sort or reuse their index arrays)."""
        self._pattern = (host_csr.indptr.copy(), host_csr.indices.copy())
        self.ordering = np.asarray(ordering, dtype=np.int64)


def superlu_elimination_order(host_lu):
    """(n,) int64 elimination order of a SuperLU factor (pre-permutation reproducing its column order): perm_c maps an
    unknown to its position, so the order is its inverse."""
    return np.argsort(host_lu.perm_c).astype(np.int64)


def symmetric_permutation(host_operator, ordering):
    """CSC of A[ordering][:, ordering] of a host scipy operator (entries relabelled, duplicates summed)."""
    import scipy.sparse
    position = np.empty(ordering.size, dtype=np.int64)
    position[ordering] = np.arange(ordering.size, dtype=np.int64)
    entries = host_operator.tocoo()
    return scipy.sparse.csc_matrix((entries.data, (position[entries.row], position[entries.col])),
                                   shape=host_operator.shape)


def _ordered_superlu(host_operator, value_dtype, orderings=None) -> Factorization:
    """scipy_splu Factorization of a host operator in the order kept by `orderings` (SuperLUOrdering) for its pattern,
    else by MMD (the order then kept); orderings None: MMD, nothing kept."""
    if orderings is None:
        return _superlu_factorization(host_operator.tocsc().astype(value_dtype, copy=False), value_dtype=value_dtype)
    host_csr = host_operator.tocsr()
    ordering = orderings.lookup(host_csr)
    if ordering is not None:
        return _superlu_factorization(symmetric_permutation(host_csr, ordering).astype(value_dtype, copy=False),
                                      value_dtype=value_dtype, ordering=ordering)
    factor = _superlu_factorization(host_csr.tocsc().astype(value_dtype, copy=False), value_dtype=value_dtype)
    orderings.remember(host_csr, superlu_elimination_order(factor._host_lu))
    return factor


def cudss_factorization(host_operator, back_end, previous=None, real=False, symmetric=False, positive_definite=False,
                        symmetrizer=None, row_scale=None, symmetric_known=False):
    """cudss_factor.CudssFactorization of a host scipy operator on back_end 'cudss' (double) / 'cudss_mixed'; a valid
    `previous` CudssFactorization of the same options is refactorised in place (plan reused for an unchanged pattern),
    one of other options released first (it is being replaced: its device memory lies outside the cupy pool).
    symmetric_known: a canonical symmetric CSR (CudssFactorization.refactorize)."""
    from .cudss_factor import CudssFactorization
    if back_end not in CUDSS_BACK_ENDS:
        raise ValueError(f"cudss_factorization needs back_end {' or '.join(CUDSS_BACK_ENDS)}, got {back_end!r}")
    precision = cudss_precision(back_end)
    if isinstance(previous, CudssFactorization):
        if previous.n_unknowns == host_operator.shape[0] and previous.can_refactorize(
                real, precision, symmetric, positive_definite, symmetrizer):
            previous.refactorize(host_operator, row_scale=row_scale, symmetrizer=symmetrizer,
                                 symmetric_known=symmetric_known)
            return previous
        previous.release()
    return CudssFactorization(host_operator, real=real, symmetric=symmetric, precision=precision, row_scale=row_scale,
                              positive_definite=positive_definite, symmetrizer=symmetrizer,
                              symmetric_known=symmetric_known)


# DevicePencil pool bytes per stored entry of the full pattern beside its term values: the factor's device index maps
# (rows, columns, lower take, the lower triangle's rows / indices; refactorize_device_values / _lower) and one
# frequency's transients (the combined lower values, |E| of the lazy norm; complex128)
DEVICE_PENCIL_MAP_BYTES = 14
DEVICE_PENCIL_TRANSIENT_BYTES = 32
LOWER_STORED_FRACTION = 0.55   # lower triangle + diagonal / all entries of a symmetric FEM pattern (rounded up)


def device_pencil_bytes(entries, term_value_bytes, symmetric=False) -> int:
    """Estimated pool bytes of a DevicePencil of `entries` stored entries whose terms take term_value_bytes (one value
    size per term, e.g. (16, 8) for complex K and real M) plus its factor's maps and per-frequency transients; a
    symmetric pencil keeps its terms on the lower triangle only (R4)."""
    term_bytes = sum(int(size) for size in term_value_bytes) * (LOWER_STORED_FRACTION if symmetric else 1.0)
    return int(math.ceil(int(entries) * (term_bytes + DEVICE_PENCIL_MAP_BYTES + DEVICE_PENCIL_TRANSIENT_BYTES)))


class DevicePencil:
    """Terms X_k of a pencil sum_k c_k X_k on one canonical host CSR pattern (sorted, no duplicates), their values
    uploaded once to the device for SparseDirectSolver.factorize_pencil; symmetric = asked for and every term complex
    symmetric (max |X - X^T| <= SYMMETRY_TOLERANCE max |X| per term, checked once on the device; symmetric_known: the
    caller vouches for it, no check). A symmetric pencil keeps only the lower triangle (diagonal included) of each term,
    in the CSR order of the pattern's lower take (lower_stored, R4: ~45 % less term memory); full_term(k) gives a term
    on the full pattern. Callers never modify indptr / indices in place (factors match them by identity)."""

    def __init__(self, indptr, indices, term_values, symmetric=False, symmetric_known=False):
        import cupy as cp
        self.indptr, self.indices = np.asarray(indptr), np.asarray(indices)
        self.n_rows = int(self.indptr.size) - 1
        self._host_rows = csr_rows(self.indptr)   # host int64 row of every stored entry (slots, symmetry, mirror)
        terms = [cp.asarray(values).reshape(-1) for values in term_values]
        if any(term.size != self.indices.size for term in terms):
            raise ValueError(f'DevicePencil: every term needs {self.indices.size} values (one per stored entry)')
        self.symmetric = bool(symmetric) and (bool(symmetric_known) or self._terms_symmetric(terms))
        self.lower_stored = self.symmetric
        if self.lower_stored:
            lower = cp.asarray(np.flatnonzero(self.indices <= self._host_rows))
            terms = [term[lower] for term in terms]
        self.stored_terms = terms
        self.device_bytes = sum(int(term.nbytes) for term in self.stored_terms)
        self._full_take = None   # lower_stored: device (nnz,) stored-layout position of every full-pattern entry (lazy)

    def _terms_symmetric(self, terms) -> bool:
        """Every term complex symmetric on a structurally symmetric pattern (one device sort of the transposed keys)."""
        import cupy as cp
        from .cudss_factor import SYMMETRY_TOLERANCE
        rows = cp.asarray(self._host_rows)
        columns = cp.asarray(self.indices.astype(np.int64))
        transposed = cp.argsort(columns * self.n_rows + rows)   # X^T's entries in CSR order
        if not (bool(cp.array_equal(columns[transposed], rows)) and bool(cp.array_equal(rows[transposed], columns))):
            return False
        for term in terms:
            largest = float(cp.abs(term).max()) if term.size else 0.0
            if float(cp.abs(term - term[transposed]).max() if term.size else 0.0) > SYMMETRY_TOLERANCE * largest:
                return False
        return True

    def combined(self, coefficients):
        """Device values of sum_k c_k X_k in the stored layout (the lower take of a lower_stored pencil, else all nnz;
        complex128, float64 when every term and coefficient is real)."""
        import cupy as cp
        if len(coefficients) != len(self.stored_terms):
            raise ValueError(f'DevicePencil: {len(coefficients)} coefficients for {len(self.stored_terms)} terms')
        complex_values = any(term.dtype.kind == 'c' for term in self.stored_terms) or any(
            isinstance(coefficient, complex) or np.iscomplexobj(coefficient) for coefficient in coefficients)
        total = cp.zeros(self.stored_terms[0].size if self.stored_terms else 0,
                         dtype=np.complex128 if complex_values else np.float64)
        for coefficient, term in zip(coefficients, self.stored_terms):
            if coefficient != 0:
                total += coefficient * term
        return total

    def slots(self, rows, columns):
        """Host int64 positions in the stored layout of the entries (rows, columns) (E7-8: a small per-frequency block,
        e.g. a port DtN, added by factorize_pencil(extra=)); a lower_stored pencil takes lower entries only (rows >=
        columns); None when an entry is not in the pattern (or not lower)."""
        return _pattern_slots(self._host_rows, self.indices, self.n_rows, rows, columns, self.lower_stored)

    def full_term(self, index):
        """Device (nnz,) values of term `index` on the full pattern (a lower_stored pencil mirrors its lower triangle:
        a new array per call)."""
        return self.full_values(self.stored_terms[index])

    def full_values(self, term):
        """Device (nnz,) full-pattern values of device values in the stored layout (combined(c) or a stored term; a
        lower_stored pencil mirrors the lower triangle: a new array)."""
        if not self.lower_stored:
            return term
        if self._full_take is None:   # built once per pencil (the SuperLU fallback mirrors every frequency)
            import cupy as cp
            rows = cp.asarray(self._host_rows)
            columns = cp.asarray(self.indices.astype(np.int64))
            is_lower = columns <= rows
            lower_rank = cp.cumsum(is_lower) - 1   # an entry's position in the lower take
            mirror = cp.searchsorted(rows * self.n_rows + columns, columns * self.n_rows + rows)   # symmetric pattern
            self._full_take = cp.where(is_lower, lower_rank, lower_rank[mirror])
        return term[self._full_take]


def _pattern_slots(pattern_rows, indices, n_rows, rows, columns, lower_stored):
    """Host int64 positions of the entries (rows, columns) in a pencil's stored layout (the canonical pattern of
    pattern_rows / indices, or its lower take when lower_stored: lower entries only); None when one is not stored."""
    rows, columns = np.asarray(rows, dtype=np.int64), np.asarray(columns, dtype=np.int64)
    if lower_stored and np.any(columns > rows):
        return None
    keys = pattern_rows * n_rows + indices   # ascending: a canonical pattern
    wanted = rows * n_rows + columns
    where = np.minimum(np.searchsorted(keys, wanted), max(keys.size - 1, 0))
    if keys.size == 0 or not np.array_equal(keys[where], wanted):
        return None
    if lower_stored:   # full position -> position in the lower take
        where = (np.cumsum(indices <= pattern_rows) - 1)[where]
    return where


class HostPencil:
    """Terms X_k of a pencil sum_k c_k X_k on one canonical host CSR pattern (sorted, no duplicates) for
    SparseDirectSolver.factorize_pencil on a host direct back end (MUMPS / PARDISO, either backend): term values kept on
    the host (the caller's arrays when already flat, no copy), the full pattern always (lower_stored False); symmetric
    = asked for and every term complex symmetric (relative_asymmetry <= SYMMETRY_TOLERANCE per term, checked once;
    symmetric_known: vouched), so the factors skip the per-frequency check. Callers never modify indptr / indices (or
    the term arrays) in place: factors match the pattern by identity."""

    lower_stored = False

    def __init__(self, indptr, indices, term_values, symmetric=False, symmetric_known=False):
        self.indptr, self.indices = np.asarray(indptr), np.asarray(indices)
        self.n_rows = int(self.indptr.size) - 1
        self.stored_terms = [np.asarray(values).reshape(-1) for values in term_values]
        if any(term.size != self.indices.size for term in self.stored_terms):
            raise ValueError(f'HostPencil: every term needs {self.indices.size} values (one per stored entry)')
        self.symmetric = bool(symmetric) and (bool(symmetric_known) or self._terms_symmetric())
        self.host_bytes = sum(int(term.nbytes) for term in self.stored_terms)
        self._host_rows = None   # int64 row per stored entry (slots; built on first use)

    def _terms_symmetric(self) -> bool:
        """Every term complex symmetric (transpose, not conjugate) on this pattern."""
        import scipy.sparse
        from .cudss_factor import SYMMETRY_TOLERANCE, relative_asymmetry
        shape = (self.n_rows, self.n_rows)
        return all(relative_asymmetry(scipy.sparse.csr_matrix((term, self.indices, self.indptr), shape=shape))
                   <= SYMMETRY_TOLERANCE for term in self.stored_terms)

    def combined(self, coefficients) -> np.ndarray:
        """Host values of sum_k c_k X_k on the full pattern (a new array; complex128, float64 when every term and
        coefficient is real)."""
        if len(coefficients) != len(self.stored_terms):
            raise ValueError(f'HostPencil: {len(coefficients)} coefficients for {len(self.stored_terms)} terms')
        complex_values = any(term.dtype.kind == 'c' for term in self.stored_terms) or any(
            isinstance(coefficient, complex) or np.iscomplexobj(coefficient) for coefficient in coefficients)
        total = np.zeros(self.indices.size, dtype=np.complex128 if complex_values else np.float64)
        for coefficient, term in zip(coefficients, self.stored_terms):
            if coefficient != 0:
                total += coefficient * term
        return total

    def slots(self, rows, columns):
        """Host int64 positions of the entries (rows, columns) in the pattern (factorize_pencil(extra=): both triangles
        of a symmetric block); None when an entry is not in the pattern."""
        if self._host_rows is None:
            self._host_rows = csr_rows(self.indptr)
        return _pattern_slots(self._host_rows, self.indices, self.n_rows, rows, columns, False)

    def full_values(self, values):
        """Full-pattern values of stored-layout values: the same array (the full pattern is stored)."""
        return values

    def full_term(self, index):
        """Values of term `index` on the full pattern (the kept array)."""
        return self.stored_terms[index]


def _superlu_factorization(host_csc, value_dtype=np.complex128, ordering=None) -> Factorization:
    """scipy_splu Factorization of a complex128 (or float64) scipy CSC operator: SuperLU factor plus its L + U host
    bytes; with ordering, host_csc is already permuted by it and SuperLU keeps its index order."""
    host_lu = _host_superlu(host_csc, pre_ordered=ordering is not None)
    # SuperLU.nnz counts the L + U store as is; the .L / .U getters would build (and cache) CSC copies of both factors.
    # ~20 B per entry = complex128 value + int32 row index (12 B for float64).
    entry_bytes = 12 if value_dtype == np.float64 else 20
    return Factorization('scipy_splu', int(host_csc.shape[0]), int(host_lu.nnz) * entry_bytes, host_lu=host_lu,
                         value_dtype=value_dtype, ordering=ordering)


def grid_nested_dissection(shape, leaf_cells=ND_LEAF_CELLS):
    """(n_cells,) int64 flat C-order cell indices of a structured grid in geometric nested-dissection order: each box
    splits at the middle plane of its longest axis, both halves first and the plane last (recursively)."""
    # A one-cell plane separates the halves when a cell's unknowns couple only to cells within +-1 per axis (the 2D / 3D
    # Yee curl-curl and 5- / 7-point stencils). Periodic wraps and mirror images couple across it: still correct, just
    # more fill. For 3D grids the George bound (fill ~ n^(4/3), work ~ n^2) beats minimum degree on large grids.
    shape = tuple(int(count) for count in shape)
    flat_index = np.arange(math.prod(shape), dtype=np.int64).reshape(shape)
    pieces = []

    def dissect(box):
        extents = [part.stop - part.start for part in box]
        axis = int(np.argmax(extents))
        if math.prod(extents) <= leaf_cells or extents[axis] < 3:
            pieces.append(flat_index[box].ravel())
            return
        middle = box[axis].start + extents[axis] // 2
        for half in (slice(box[axis].start, middle), slice(middle + 1, box[axis].stop)):
            dissect(box[:axis] + (half,) + box[axis + 1:])
        pieces.append(flat_index[box[:axis] + (slice(middle, middle + 1),) + box[axis + 1:]].ravel())

    dissect(tuple(slice(0, count) for count in shape))
    return np.concatenate(pieces)


def _column_major_operator(operator, nx, ny):
    """P A P^T of an i*ny + j indexed cupyx sparse operator on the column-major j*nx + i index (CSR)."""
    import cupy as cp
    import cupyx.scipy.sparse
    column_major_index = cp.arange(int(nx) * int(ny), dtype=cp.int32).reshape(int(ny), int(nx)).T.ravel()
    entries = operator.tocoo()
    return cupyx.scipy.sparse.coo_matrix(
        (entries.data, (column_major_index[entries.row], column_major_index[entries.col])),
        shape=operator.shape).tocsr()


def _host_superlu(host_csc, pre_ordered=False):
    """scipy SuperLU of a host scipy CSC operator with the FDFD ordering / pivot options (pre_ordered: the caller's
    symmetric permutation, kept by permc_spec NATURAL)."""
    try:
        import scipy.sparse.linalg
    except ImportError as exc:
        raise ImportError(f"the 'scipy_splu' solver could not import scipy ({exc}); {_SCIPY_ALTERNATIVES}") from exc
    # The Helmholtz matrix is structurally symmetric (5-point stencil; pinned rows and columns alike) except for the
    # first cell of a non-periodic axis, whose row A1 leaves without that axis's coupling, so SuperLU's minimum-degree
    # ordering on A^T + A fits it better than the default COLAMD for unsymmetric patterns (the cross-section mode
    # operator A - sigma B is nearly structurally symmetric too). Symmetric mode applies that ordering to rows too and
    # the low threshold keeps the (dominant) diagonal pivots; with the default threshold 1 off-diagonal row swaps break
    # the ordering and add fill (SuperLU user guide, symmetric mode).
    return scipy.sparse.linalg.splu(host_csc, permc_spec='NATURAL' if pre_ordered else 'MMD_AT_PLUS_A',
                                    diag_pivot_thresh=_SPLU_PIVOT_THRESHOLD, options={'SymmetricMode': True})


def superlu_inertia(host_lu):
    """(n_positive, n_negative) signs of the U diagonal of a scipy SuperLU whose pivots stayed on the diagonal (perm_r
    == perm_c: P A P^T = L D L^T, Sylvester; real parts for Hermitian A), else None. Builds the U copy once."""
    if not np.array_equal(host_lu.perm_r, host_lu.perm_c):
        return None
    pivots = np.real(host_lu.U.diagonal())
    return int(np.count_nonzero(pivots > 0.0)), int(np.count_nonzero(pivots < 0.0))


def superlu_counting_inertia(host_matrix):
    """superlu_inertia of a SuperLU taken for counting only: symmetric mode, diagonal pivots whenever nonzero
    (diag_pivot_thresh 0); None when a zero diagonal forced a row swap. RuntimeError when singular."""
    import scipy.sparse.linalg
    host_lu = scipy.sparse.linalg.splu(host_matrix.tocsc(), permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0,
                                       options={'SymmetricMode': True})
    return superlu_inertia(host_lu)


class SparseDirectSolver:
    """LinearSolver / ShiftedSolver of the 'sparse_direct' type: factorisations of the resolved back end (factorize) or
    of a host matrix (factorize_host: host SuperLU, cuDSS on the cuDSS back ends), and GMRES re-solves on an earlier
    SuperLU factor, all counted. It keeps no factor itself, so the consumers sharing one (NLFD channels, thermal noise
    operators) hold their own. fallback_back_end ('scipy_splu' when 'auto' resolved to cuDSS, B1b): a cuDSS refusal
    (CUDSS_REFUSALS) switches the runtime to it for good and the factorisation is redone there (fall_back)."""

    def __init__(self, back_end='auto', label=None, backend='gpu', fallback_back_end=None):
        if back_end not in LINEAR_SOLVER_CHOICES:
            raise ValueError(f"back_end must be one of {', '.join(LINEAR_SOLVER_CHOICES)}, got {back_end!r}")
        if backend not in ('gpu', 'cpu'):
            raise ValueError(f"backend must be 'gpu' or 'cpu', got {backend!r}")
        if fallback_back_end not in (None, 'scipy_splu'):
            raise ValueError(f"fallback_back_end must be None or 'scipy_splu', got {fallback_back_end!r}")
        self.backend = backend   # 'cpu': factorize takes scipy operators, solves are numpy (host_arrays)
        self.back_end_choice = back_end
        self.label = label if label is not None else (lambda: 'sparse_direct')
        self._back_end = None
        self.fallback_back_end = fallback_back_end
        self.auto_fallback = None        # text of the cuDSS refusal that switched an 'auto' runtime to SuperLU, or None
        self.n_factorizations = 0
        self.total_factor_seconds = 0.0
        self.last_factor_seconds = None
        self.n_recycled_solves = 0
        self.total_recycled_iterations = 0
        self.host_bytes = 0   # L + U of the last host factorisation (the consumer keeps the factor)
        self.orderings = SuperLUOrdering()   # SuperLU elimination order of the last pattern (consumers alike: any is exact)
        self.n_cudss_plans = 0           # cuDSS analyses (a refactorisation of an unchanged pattern adds none)
        self.n_cudss_reused_orderings = 0   # of those, analyses on a kept ordering (cudss_factor "Ordering reuse")
        self.total_cudss_plan_seconds = 0.0
        self.cudss_device_bytes = 0       # last cuDSS factor + operator copies (outside the cupy pool)
        self.cudss_permanent_bytes = 0    # last plan's permanent / peak device estimates (outside the pool) and the
        self.cudss_peak_bytes = 0         # factor entries: the data of cudss_factor.factor_device_bytes' fit
        self.cudss_lu_entries = 0
        self.cudss_refinement_steps = 0   # last cuDSS factor: refinement steps per full-accuracy solve
        self.cudss_single_residual = None   # last cudss_mixed factor: unrefined probe residual
        self.n_cudss_symmetric = 0       # cuDSS factorisations that are an LDL^T / Cholesky (the rest: general LU)
        self.n_cudss_double_fallbacks = 0   # cudss_mixed factors rejected by their probe and replaced by double ones
        self.n_auto_fallbacks = 0        # B1b: an 'auto' cuDSS back end switched to SuperLU after a refusal (0 / 1)
        # host direct back ends (MUMPS / PARDISO): analyses (fallback re-analyses included) and those reused by a
        # refactorisation, the last factor's library estimate after its analysis, factor entries, refinement iterations
        # of its probe, unrefined probe residual (mixed), thread count; LDL^T / Cholesky factors and mixed rejections
        self.n_host_direct_analyses = 0
        self.n_host_direct_reused_analyses = 0
        self.host_direct_peak_bytes = 0
        self.host_direct_lu_entries = 0
        self.n_host_direct_symmetric = 0
        self.host_direct_refinement_steps = 0
        self.host_direct_single_residual = None
        self.n_host_direct_double_fallbacks = 0
        self.host_direct_threads = None

    @property
    def back_end(self) -> str:
        """Resolved back end (BACK_ENDS); ValueError naming 'pip install scipy' / nvmath when it cannot be."""
        if self._back_end is None:
            self._back_end = resolve_back_end(self.back_end_choice, self.backend)
        return self._back_end

    def check_size(self, nx, ny, periodic_x, periodic_y) -> None:
        """check_solver_size of an nx * ny grid system on the resolved back end."""
        check_solver_size(self.back_end, nx, ny, periodic_x, periodic_y)

    def factorize(self, operator, nx, ny, periodic_x, periodic_y, previous=None, symmetric=False, symmetrizer=None):
        """Factorization of a square CSR grid operator of the backend (cupyx; scipy on the cpu) on the resolved back end
        (module factorize; previous / symmetric / symmetrizer reach the cuDSS and host direct back ends only),
        counted."""
        require_backend_operator(self.label(), self.backend, operator)
        from .cudss_factor import CUDSS_REFUSALS
        start = time.perf_counter()
        try:
            factor = factorize(operator, self.back_end, nx, ny, periodic_x, periodic_y, previous=previous,
                               symmetric=symmetric, symmetrizer=symmetrizer, orderings=self.orderings)
        except CUDSS_REFUSALS as error:
            if not self.fall_back(error, operator.shape[0]):
                raise
            factor = factorize(operator, self.back_end, nx, ny, periodic_x, periodic_y, orderings=self.orderings)
        return self._counted(factor, start)

    def factorize_host(self, matrix, real=False, symmetric=False, positive_definite=False, symmetrizer=None,
                       row_scale=None, previous=None, symmetric_known=False):
        """Factorization (solve_host; float64 with real) of a host scipy sparse matrix, counted: cuDSS on the cuDSS back
        ends (cudss_factorization options; a refusal of an 'auto' runtime: SuperLU, fall_back), MUMPS / PARDISO on the
        host direct back ends (host_direct_factorization, the same options; symmetric_known: a canonical symmetric
        CSR, no asymmetry check; refusals raise), else host SuperLU whatever the back end (the options ignored)."""
        from .cudss_factor import CUDSS_REFUSALS
        start = time.perf_counter()
        factor = None
        if self.back_end in HOST_DIRECT_BACK_ENDS:
            factor = host_direct_factorization(matrix, self.back_end, previous=previous, real=real,
                                               symmetric=symmetric, positive_definite=positive_definite,
                                               symmetrizer=symmetrizer, row_scale=row_scale,
                                               symmetric_known=symmetric_known)
        elif self.back_end in CUDSS_BACK_ENDS:
            try:
                factor = cudss_factorization(matrix, self.back_end, previous=previous, real=real, symmetric=symmetric,
                                             positive_definite=positive_definite, symmetrizer=symmetrizer,
                                             row_scale=row_scale)
            except CUDSS_REFUSALS as error:
                if not self.fall_back(error, matrix.shape[0]):
                    raise
        if factor is None:
            factor = factorize_host(matrix, real=real, orderings=self.orderings)
        return self._counted(factor, start)

    def fall_back(self, error, n_unknowns) -> bool:
        """B1b: switch an 'auto' runtime (fallback_back_end set) from its cuDSS back end to SuperLU for good after a
        cuDSS refusal (CUDSS_REFUSALS: zero pivot, accuracy, device memory) of an operator within SPLU_LIMIT_UNKNOWNS;
        True when it switched (warned; the caller redoes its factorisation, SuperLU now), else False (re-raise)."""
        import warnings
        from .cudss_factor import CUDSS_REFUSALS
        if (self.fallback_back_end is None or self._back_end not in CUDSS_BACK_ENDS
                or not isinstance(error, CUDSS_REFUSALS) or int(n_unknowns) > SPLU_LIMIT_UNKNOWNS):
            return False
        self.auto_fallback = (f"{self.label()}: back_end 'auto' resolved to '{self._back_end}', which refused the "
                              f"operator ({error}); host SuperLU ('{self.fallback_back_end}') from now on")
        warnings.warn(self.auto_fallback, RuntimeWarning, stacklevel=2)
        self._back_end = self.fallback_back_end
        self.n_auto_fallbacks += 1
        return True

    def device_pencil(self, indptr, indices, term_values, symmetric=False, symmetric_known=False):
        """DevicePencil of these terms when factorize_pencil factorises on the device (gpu backend, a cuDSS back end),
        else None (the caller factorises host matrices by factorize_host); symmetric_known skips its symmetry check."""
        if self.backend != 'gpu' or self.back_end not in CUDSS_BACK_ENDS:
            return None
        return DevicePencil(indptr, indices, term_values, symmetric=symmetric, symmetric_known=symmetric_known)

    def pencil(self, indptr, indices, term_values, symmetric=False, symmetric_known=False):
        """Pencil of these terms for factorize_pencil: a HostPencil on a host direct back end (MUMPS / PARDISO, either
        backend), a DevicePencil on a cuDSS back end (gpu backend), else None (the caller factorises host matrices by
        factorize_host); symmetric_known skips the symmetry check."""
        if self.back_end in HOST_DIRECT_BACK_ENDS:
            return HostPencil(indptr, indices, term_values, symmetric=symmetric, symmetric_known=symmetric_known)
        return self.device_pencil(indptr, indices, term_values, symmetric=symmetric, symmetric_known=symmetric_known)

    def factorize_pencil(self, pencil, coefficients, row_scale=None, previous=None, extra=None, defer_check=True):
        """CudssFactorization of sum_k c_k X_k (+ extra) of a DevicePencil (symmetric request = pencil.symmetric),
        counted, all on the device (CudssFactorization.refactorize_device_values: values, equilibration, lower-triangle
        take, no host CSR or asymmetry check): a previous factor of the same options is refreshed (checked by its first
        solve with defer_check: a CudssAccuracyError may come from solve; defer_check False probes now, for factors
        read without a solve, e.g. inertia counts), else a new one is planned from the device values. A HostPencil:
        the HostDirectFactorization of the host values (_factorize_host_pencil, the same rules).
        extra: (pencil.slots positions, values) added to the combined values (E7-8: a port DtN block; symmetric with a
        symmetric pencil, its lower entries only for a lower_stored one, as the caller vouches)."""
        if isinstance(pencil, HostPencil):
            return self._factorize_host_pencil(pencil, coefficients, row_scale, previous, extra, defer_check)
        import cupy as cp
        from .cudss_factor import CUDSS_REFUSALS, CudssFactorization
        start = time.perf_counter()
        values = pencil.combined(coefficients)   # a symmetric pencil: its lower triangle (R4)
        if extra is not None:
            positions, added = cp.asarray(extra[0]), cp.asarray(extra[1])
            if added.dtype.kind == 'c' and values.dtype.kind != 'c':
                values = values.astype(np.complex128)
            values[positions] += added   # unique positions (one CSR entry each)
        real = values.dtype.kind != 'c'
        if self.back_end in CUDSS_BACK_ENDS:
            precision = cudss_precision(self.back_end)
            try:
                if isinstance(previous, CudssFactorization) and previous.valid:
                    if previous.n_unknowns == pencil.n_rows and previous.can_refactorize(real, precision,
                                                                                         symmetric=pencil.symmetric):
                        refresh = (previous.refactorize_device_lower if pencil.lower_stored
                                   else previous.refactorize_device_values)
                        refresh(values, pencil.indptr, pencil.indices, row_scale=row_scale, defer_check=defer_check)
                        return self._counted(previous, start)
                    previous.release()
                factor = CudssFactorization.from_device_values(values, pencil.indptr, pencil.indices, real=real,
                                                               symmetric=pencil.symmetric, precision=precision,
                                                               row_scale=row_scale,   # symmetric when requested
                                                               lower=pencil.lower_stored)
                factor.fallback_listener = self._count_deferred_fallback
                return self._counted(factor, start)
            except CUDSS_REFUSALS as error:
                if not self.fall_back(error, pencil.n_rows):
                    raise
        if isinstance(previous, CudssFactorization):   # after a fall_back: host SuperLU of the combined pencil
            previous.release()
        import scipy.sparse
        host = scipy.sparse.csr_matrix((cp.asnumpy(pencil.full_values(values)), pencil.indices, pencil.indptr),
                                       shape=(pencil.n_rows, pencil.n_rows))
        return self._counted(factorize_host(host, real=real, orderings=self.orderings), start)

    def _factorize_host_pencil(self, pencil, coefficients, row_scale, previous, extra, defer_check):
        """HostDirectFactorization of sum_k c_k X_k (+ extra) of a HostPencil, counted: a previous factor of the same
        options refactorised on its analysis (refactorize_values: the pattern matched by identity, no canonicalising
        copy, no asymmetry check: pencil.symmetric vouches; defer_check as factorize_pencil), else a new one."""
        if self.back_end not in HOST_DIRECT_BACK_ENDS:
            raise ValueError(f"{self.label()}: a HostPencil needs a host direct back end, got '{self.back_end}'")
        start = time.perf_counter()
        values = pencil.combined(coefficients)
        if extra is not None:
            positions, added = np.asarray(extra[0]), np.asarray(extra[1])
            if added.dtype.kind == 'c' and values.dtype.kind != 'c':
                values = values.astype(np.complex128)
            values[positions] += added   # unique positions (one CSR entry each)
        factor = host_direct_values_factorization(values, pencil.indptr, pencil.indices, self.back_end,
                                                  previous=previous, real=values.dtype.kind != 'c',
                                                  symmetric=pencil.symmetric, row_scale=row_scale,
                                                  symmetric_known=pencil.symmetric, defer_check=defer_check)
        return self._counted(factor, start)

    def factorize_device_csr(self, operator, real=False, symmetric=False, positive_definite=False, previous=None):
        """CudssFactorization of a cupyx CSR (cuDSS back ends) without a host copy of its values (H1), counted: the
        asymmetry check (when symmetry is asked for), equilibration and lower-triangle take on the device; a previous
        factor of the same options is refreshed (its pattern matched on the device: no index download), else a new one
        is planned from the device values. A non-canonical operator is canonicalised in a copy first. A refusal of an
        'auto' runtime: a host SuperLU Factorization of a host copy, the runtime on SuperLU from then on (B1b)."""
        import cupy as cp
        from .cudss_factor import CUDSS_REFUSALS, CudssFactorization
        if self.back_end not in CUDSS_BACK_ENDS:
            raise ValueError(f"{self.label()}: factorize_device_csr needs a cuDSS back end, got '{self.back_end}'")
        start = time.perf_counter()
        if not operator.has_canonical_format:
            operator = operator.copy()
            operator.sum_duplicates()
        precision = cudss_precision(self.back_end)
        try:
            if isinstance(previous, CudssFactorization) and previous.valid:
                if previous.n_unknowns == operator.shape[0] and previous.can_refactorize(real, precision, symmetric,
                                                                                         positive_definite):
                    indptr, indices = previous.host_pattern_of(operator)
                    previous.refactorize_device_values(operator.data, indptr, indices, check_symmetry=True)
                    return self._counted(previous, start)
                previous.release()
            factor = CudssFactorization.from_device_values(
                operator.data, cp.asnumpy(operator.indptr), cp.asnumpy(operator.indices), real=real,
                symmetric=symmetric, precision=precision, positive_definite=positive_definite, check_symmetry=True)
            return self._counted(factor, start)
        except CUDSS_REFUSALS as error:   # an 'auto' runtime: host SuperLU of a host copy from now on (B1b)
            if not self.fall_back(error, operator.shape[0]):
                raise
        return self._counted(factorize_host(operator.get(), real=real, orderings=self.orderings), start)

    def factorize_values(self, values, indptr, indices, real=True, previous=None, defer_check=True):
        """Factorization of an unsymmetric operator given as values on a fixed canonical host CSR pattern (indptr,
        indices: sorted, no duplicates, never modified in place: matched by identity), counted (Newton Jacobians):
        cuDSS refreshes a previous factor of the same options on the device (refactorize_device_values; values host
        or device; a refusal of an 'auto' runtime: SuperLU, B1b), MUMPS / PARDISO on their kept analysis
        (host_direct_values_factorization), else host SuperLU of the CSR; defer_check as factorize_pencil."""
        from .cudss_factor import CUDSS_REFUSALS, CudssFactorization
        start = time.perf_counter()
        n_rows = int(np.asarray(indptr).size) - 1
        if self.back_end in HOST_DIRECT_BACK_ENDS:
            from ..core.backend import to_host
            factor = host_direct_values_factorization(np.asarray(to_host(values)), indptr, indices, self.back_end,
                                                      previous=previous, real=real, defer_check=defer_check)
            return self._counted(factor, start)
        if self.back_end in CUDSS_BACK_ENDS and self.backend == 'gpu':
            import cupy as cp
            precision = cudss_precision(self.back_end)
            device_values = cp.asarray(values, dtype=np.float64 if real else np.complex128)
            try:
                if isinstance(previous, CudssFactorization) and previous.valid:
                    if previous.n_unknowns == n_rows and previous.can_refactorize(real, precision):
                        previous.refactorize_device_values(device_values, indptr, indices, defer_check=defer_check)
                        return self._counted(previous, start)
                    previous.release()
                factor = CudssFactorization.from_device_values(device_values, indptr, indices, real=real,
                                                               precision=precision)
                factor.fallback_listener = self._count_deferred_fallback
                return self._counted(factor, start)
            except CUDSS_REFUSALS as error:
                if not self.fall_back(error, n_rows):
                    raise
        if isinstance(previous, CudssFactorization):   # replaced by SuperLU after a fall_back
            previous.release()
        import scipy.sparse
        from ..core.backend import to_host
        host = scipy.sparse.csr_matrix((np.asarray(to_host(values)), indices, indptr), shape=(n_rows, n_rows))
        return self._counted(factorize_host(host, real=real, orderings=self.orderings), start)

    def _count_deferred_fallback(self, plans, plan_seconds, reused_orderings) -> None:
        """A deferred check inside a solve replaced a cudss_mixed factor by a double one (its plans, one fallback)."""
        self.n_cudss_plans += plans
        self.n_cudss_reused_orderings += reused_orderings
        self.total_cudss_plan_seconds += plan_seconds
        self.n_cudss_double_fallbacks += 1

    def solve_recycled(self, factorization, operator, rhs, initial_guess=None, transpose=False):
        """Factorization.solve_preconditioned on an earlier factor, counted; (x or None, iterations)."""
        solution, iterations = factorization.solve_preconditioned(operator, rhs, initial_guess, transpose=transpose)
        self.count_recycled(iterations, solution is not None)
        return solution, iterations

    def solve_recycled_host(self, factorization, matrix, rhs, initial_guess=None):
        """Factorization.solve_preconditioned_host (host scipy matrix, numpy rhs) on an earlier factor, counted; (x or
        None, iterations)."""
        solution, iterations = factorization.solve_preconditioned_host(matrix, rhs, initial_guess)
        self.count_recycled(iterations, solution is not None)
        return solution, iterations

    def count_recycled(self, iterations, converged) -> None:
        """Record one GMRES re-solve on an earlier factor (both solve_recycled paths)."""
        self.total_recycled_iterations += int(iterations)
        self.n_recycled_solves += bool(converged)

    def _counted(self, factor, start):
        self.last_factor_seconds = time.perf_counter() - start
        self.total_factor_seconds += self.last_factor_seconds
        self.n_factorizations += 1
        self.host_bytes = factor.host_bytes or 0
        if getattr(factor, 'back_end', None) == 'cudss':   # a CudssFactorization (both precisions)
            self.n_cudss_plans += factor.last_plans   # fallback re-plans of the call included
            self.n_cudss_reused_orderings += factor.last_reused_orderings
            self.total_cudss_plan_seconds += factor.last_plans_seconds
            self.cudss_device_bytes = factor.device_bytes
            self.cudss_permanent_bytes, self.cudss_peak_bytes = factor.permanent_bytes, factor.peak_bytes
            self.cudss_lu_entries = factor.lu_entries
            self.cudss_refinement_steps = factor.refinement_steps
            self.cudss_single_residual = factor.single_residual
            self.n_cudss_symmetric += int(factor.symmetric)
            self.n_cudss_double_fallbacks += int(factor.last_double_fallback)
        elif is_host_direct_factor(factor):   # MUMPS / PARDISO (host_bytes above = the library's factor bytes)
            self.n_host_direct_analyses += factor.last_analyses   # fallback re-analyses of the call included
            self.n_host_direct_reused_analyses += int(factor.last_plan_reused)
            self.host_direct_peak_bytes = factor.peak_bytes
            self.host_direct_lu_entries = factor.lu_entries
            self.n_host_direct_symmetric += int(factor.symmetric)
            self.host_direct_refinement_steps = factor.refinement_steps
            self.host_direct_single_residual = factor.single_residual
            self.n_host_direct_double_fallbacks += int(factor.last_double_fallback)
            self.host_direct_threads = factor.threads
        return factor

    def work_counters(self) -> dict:
        """Work so far (summary 'solvers' record of its (physics, role)); factor_seconds is host wall time (cuDSS: plan,
        device factorisation and probe, blocking)."""
        counters = {'back_end': self._back_end, 'factorizations': self.n_factorizations,
                    'factor_seconds': self.total_factor_seconds, 'recycled_solves': self.n_recycled_solves,
                    'gmres_iterations': self.total_recycled_iterations, 'host_factor_bytes': self.host_bytes,
                    'reused_orderings': self.orderings.n_reused}
        if self.fallback_back_end is not None:   # an 'auto' cuDSS runtime: whether it fell back to SuperLU (B1b)
            counters['auto_fallbacks'] = self.n_auto_fallbacks
        if self._back_end in CUDSS_BACK_ENDS or self.n_auto_fallbacks:
            counters.update(cudss_plans=self.n_cudss_plans, cudss_reused_orderings=self.n_cudss_reused_orderings,
                            cudss_plan_seconds=self.total_cudss_plan_seconds,
                            cudss_factor_device_bytes=self.cudss_device_bytes,
                            cudss_permanent_bytes=self.cudss_permanent_bytes, cudss_peak_bytes=self.cudss_peak_bytes,
                            cudss_lu_entries=self.cudss_lu_entries,
                            cudss_refinement_steps=self.cudss_refinement_steps,
                            cudss_single_residual=self.cudss_single_residual,
                            cudss_symmetric_factors=self.n_cudss_symmetric,
                            cudss_double_fallbacks=self.n_cudss_double_fallbacks)
        if self._back_end in HOST_DIRECT_BACK_ENDS:
            counters.update(host_direct_analyses=self.n_host_direct_analyses,
                            host_direct_reused_analyses=self.n_host_direct_reused_analyses,
                            host_direct_peak_bytes=self.host_direct_peak_bytes,
                            host_direct_lu_entries=self.host_direct_lu_entries,
                            host_direct_symmetric_factors=self.n_host_direct_symmetric,
                            host_direct_refinement_steps=self.host_direct_refinement_steps,
                            host_direct_single_residual=self.host_direct_single_residual,
                            host_direct_double_fallbacks=self.n_host_direct_double_fallbacks,
                            host_direct_threads=self.host_direct_threads)
        return counters


class ExactSolveRecord:
    """PCGSolver-protocol record of the exact (direct) DiffusionSystem solvers (BandedSolver, DiffusionDirectSolver):
    the stop-rule / count attributes and the host solve tail (solution write, counts, optional residual)."""

    precision = 'double'

    def _init_exact_record(self):
        self.tolerance = 0.0          # direct: no stop rule (DiffusionSystem tol / max_iter setters land here)
        self.max_iterations = 1
        self.compute_residual = False
        self.n_solves = 0
        self.last_iters = 0
        self.last_converged = True
        self.last_residual = float('nan')
        self.last_warning = None      # never set: a singular / indefinite operator raises at factorize
        self.n_unconverged = 0

    def _host_rhs(self, rhs):
        """Host float64 copy of a flat rhs of the backend."""
        return np.asarray(rhs.get() if self.backend == 'gpu' else rhs, dtype=np.float64)

    def _finish_host_solve(self, host_rhs, host_solution, solution, host_operator) -> None:
        """solution[...] = host_solution (one transfer on the gpu), the counts, and with compute_residual the relative
        residual of host_operator (None: not kept)."""
        if self.backend == 'gpu':
            solution.set(host_solution.astype(solution.dtype, copy=False))
        else:
            solution[...] = host_solution
        self.n_solves += 1
        self.last_iters = 1
        if self.compute_residual and host_operator is not None:
            rhs_norm = float(np.linalg.norm(host_rhs))
            residual = float(np.linalg.norm(host_rhs - host_operator @ host_solution))
            self.last_residual = residual / rhs_norm if rhs_norm > 0.0 else residual


class DiffusionDirectSolver(ExactSolveRecord):
    """The PCGSolver protocol of core.fv_diffusion.DiffusionSystem (and the FEM systems) over a SparseDirectSolver
    runtime: one real float64 factor per assembly (factorize), exact solves (solve); factorisations counted on the
    runtime. Host SuperLU (gpu backend: the CSR and rhs / solution move per call) or, back_end cudss / cudss_mixed, a
    real cuDSS Cholesky (LDL^T if it fails; cudss_mixed: a float32 LDL^T, never a float32 Cholesky) refactorised on the
    kept plan while the pattern holds, solves on the device; on the host direct back ends a MUMPS LDL^T / PARDISO
    Cholesky (LDL^T if it fails; mixed: a float32 LDL^T) refactorised on the kept analysis, solves on the host;
    cusolver_qr is refused."""

    def __init__(self, runtime):
        if not isinstance(runtime, SparseDirectSolver):
            raise TypeError(f'DiffusionDirectSolver needs a SparseDirectSolver, got {type(runtime).__name__}')
        self.runtime = runtime
        self.backend = runtime.backend
        self.label = runtime.label
        self._init_exact_record()
        self._factor = None
        self._host_operator = None
        self._device_operator = None   # cuDSS with compute_residual: the assembled cupyx CSR
        # window cell shape of a tensor-grid DiffusionSystem (its prepare; None elsewhere): a 3D window starts SuperLU
        # from grid_nested_dissection instead of minimum degree (fill ~ n^(4/3) on 7-point stencils)
        self.grid_shape = None

    def _refuse_cusolver_qr(self, back_end) -> None:
        if back_end == 'cusolver_qr':
            raise ValueError(f"{self.label()}: diffusion systems factorise with host SuperLU, cuDSS, MUMPS or PARDISO; "
                             "back_end 'cusolver_qr' is not supported here (use 'auto', 'scipy_splu', 'cudss', "
                             f"'cudss_mixed' or one of {', '.join(HOST_DIRECT_BACK_ENDS)})")

    def factorize(self, operator, preconditioner=None, n_unknowns=None) -> None:
        """Real factor of the assembled SPD CSR (scipy on the cpu backend, cupyx on the gpu): host SuperLU, cuDSS on
        its back ends or MUMPS / PARDISO on theirs (the previous factor refactorised: plan / analysis reuse);
        preconditioner ignored."""
        back_end = self.runtime.back_end
        self._refuse_cusolver_qr(back_end)
        require_backend_operator(self.label(), self.backend, operator)
        on_host = isinstance(operator.data, np.ndarray)
        if back_end in CUDSS_BACK_ENDS:   # gpu backend only (backend_misfit): no host round trip of the values (H1)
            self._factor = self.runtime.factorize_device_csr(operator, real=True, positive_definite=True,
                                                             previous=self._factor)
            self._device_operator = operator if self.compute_residual else None
            return
        if back_end in HOST_DIRECT_BACK_ENDS:
            self._factorize_host_direct(operator if on_host else operator.get())
            return
        self._factorize_superlu(operator if on_host else operator.get())

    def factorize_host_operator(self, host_operator, symmetric_known=False, defer_check=False) -> None:
        """factorize of a host scipy operator on either backend, no device copy made: cuDSS and the host direct back
        ends take it as it is (symmetric_known: a canonical symmetric CSR, the canonicalising copy and asymmetry check
        skipped; host direct: its values refresh the kept analysis, defer_check as refactorize_values), else SuperLU."""
        back_end = self.runtime.back_end
        self._refuse_cusolver_qr(back_end)
        if back_end in HOST_DIRECT_BACK_ENDS:
            self._factorize_host_direct(host_operator, symmetric_known=symmetric_known, defer_check=defer_check)
            return
        if back_end in CUDSS_BACK_ENDS:
            from .cudss_factor import CUDSS_REFUSALS
            start = time.perf_counter()
            try:
                factor = cudss_factorization(host_operator, back_end, previous=self._factor, real=True,
                                             positive_definite=True, symmetric_known=symmetric_known)
                self._factor = self.runtime._counted(factor, start)
                self._device_operator = None
                return
            except CUDSS_REFUSALS as error:   # an 'auto' runtime: host SuperLU from now on (B1b)
                if not self.runtime.fall_back(error, host_operator.shape[0]):
                    raise
        self._factorize_superlu(host_operator)

    def factorize_device_values(self, values, indptr, indices, defer_check=False) -> None:
        """factorize of a canonical symmetric CSR given as device values (cupy (nnz,)) on the host pattern (indptr,
        indices) (gpu backend, plan G5): cuDSS refreshes a factor of the same options on the device
        (CudssFactorization.refactorize_device_values: no host CSR once planned on that pattern; defer_check: the next
        solve checks it), else plans a new one from the device values; SuperLU factorises a host copy."""
        import cupy as cp
        import scipy.sparse
        from .cudss_factor import CudssFactorization
        back_end = self.runtime.back_end
        n_rows = len(indptr) - 1
        previous = self._factor
        if back_end in CUDSS_BACK_ENDS:
            from .cudss_factor import CUDSS_REFUSALS
            start = time.perf_counter()
            precision = cudss_precision(back_end)
            try:
                if (isinstance(previous, CudssFactorization) and previous.valid and previous.n_unknowns == n_rows
                        and previous.can_refactorize(True, precision, positive_definite=True)):
                    previous.refactorize_device_values(values, indptr, indices, defer_check=defer_check)
                    factor = previous
                else:
                    if isinstance(previous, CudssFactorization):
                        previous.release()
                    factor = CudssFactorization.from_device_values(values, indptr, indices, real=True,
                                                                   precision=precision, positive_definite=True)
                self._factor = self.runtime._counted(factor, start)
                self._device_operator = None
                return
            except CUDSS_REFUSALS as error:   # an 'auto' runtime: SuperLU of a host copy from now on (B1b)
                if not self.runtime.fall_back(error, n_rows):
                    raise
        host_operator = scipy.sparse.csr_matrix((cp.asnumpy(values), indices, indptr), shape=(n_rows, n_rows))
        self.factorize_host_operator(host_operator, symmetric_known=True, defer_check=defer_check)

    def _factorize_superlu(self, host_operator) -> None:
        """Host SuperLU of a host scipy operator, counted on the runtime: the runtime's kept order of this pattern,
        else (3D window) nested dissection, else minimum degree, the order kept for the next assembly."""
        n_unknowns = int(host_operator.shape[0])
        if n_unknowns > SPLU_LIMIT_UNKNOWNS:   # validation's diffusion_size_issue refuses 3D earlier
            raise ValueError(f'{self.label()}: {diffusion_size_issue(n_unknowns)[1]}')
        host_csr = host_operator.tocsr()
        start = time.perf_counter()
        orderings = self.runtime.orderings
        ordering = orderings.lookup(host_csr)
        if (ordering is None and self.grid_shape is not None and len(self.grid_shape) == 3
                and math.prod(self.grid_shape) == n_unknowns):
            ordering = grid_nested_dissection(self.grid_shape)
            orderings.remember(host_csr, ordering)
        if ordering is None:   # minimum degree, its order kept
            factor = _ordered_superlu(host_csr, np.float64, orderings)
        else:
            factor = _superlu_factorization(symmetric_permutation(host_csr, ordering).astype(np.float64, copy=False),
                                            value_dtype=np.float64, ordering=ordering)
        factor.host_arrays = True
        self._factor = self.runtime._counted(factor, start)
        self._host_operator = host_csr if self.compute_residual else None

    def _factorize_host_direct(self, host_operator, symmetric_known=False, defer_check=False) -> None:
        """MUMPS / PARDISO factor of a host scipy operator (real, positive_definite: PARDISO Cholesky, MUMPS LDL^T),
        counted on the runtime; the previous host direct factor refactorised on its kept analysis (symmetric_known: by
        its values on the identity-matched pattern, defer_check). SPLU_LIMIT_UNKNOWNS applies; refusals raise."""
        back_end = self.runtime.back_end
        n_unknowns = int(host_operator.shape[0])
        if n_unknowns > SPLU_LIMIT_UNKNOWNS:   # validation's diffusion_size_issue refuses 3D earlier
            raise ValueError(f'{self.label()}: {diffusion_size_issue(n_unknowns, back_end=back_end)[1]}')
        start = time.perf_counter()
        previous = self._factor if is_host_direct_factor(self._factor) else None
        if symmetric_known:   # a canonical symmetric CSR (FEM Newton tangents): no copy, compare or asymmetry check
            csr = host_operator.tocsr()
            factor = host_direct_values_factorization(csr.data, csr.indptr, csr.indices, back_end, previous=previous,
                                                      real=True, symmetric_known=True, defer_check=defer_check,
                                                      positive_definite=True)
        else:
            factor = host_direct_factorization(host_operator, back_end, previous=previous, real=True,
                                               positive_definite=True)
        factor.host_arrays = True
        self._factor = self.runtime._counted(factor, start)
        self._host_operator = host_operator.tocsr() if self.compute_residual else None
        self._device_operator = None

    def release_factor(self) -> None:
        """Drop the factor (a cuDSS factor's device memory, outside the cupy pool, or a MUMPS / PARDISO factor's
        library memory, freed now); the next solve needs a factorize. Counters stay."""
        factor, self._factor = self._factor, None
        release = getattr(factor, 'release', None)   # CudssFactorization / HostDirectFactorization (SuperLU: none)
        if release is not None:
            release()
        self._host_operator = self._device_operator = None

    def solve(self, rhs, solution, reference=None) -> None:
        """solution[...] = A^-1 rhs (flat arrays of the backend); reference ignored (exact solve)."""
        if self._factor is None:
            raise RuntimeError(f'{self.label()}: factorize before solve')
        if is_host_direct_factor(self._factor):   # MUMPS / PARDISO: host solve (one transfer each way on the gpu)
            host_rhs = self._host_rhs(rhs)
            self._finish_host_solve(host_rhs, self._factor.solve_host(host_rhs), solution, self._host_operator)
            return
        if not isinstance(self._factor, Factorization):   # cuDSS: device in, device out
            self._solve_on_device(rhs, solution)
            return
        host_rhs = self._host_rhs(rhs)
        host_solution = self._factor._lu_solve(host_rhs)   # through the factor's elimination order
        self._finish_host_solve(host_rhs, host_solution, solution, self._host_operator)

    def _solve_on_device(self, rhs, solution) -> None:
        """solution[...] = A^-1 rhs by the cuDSS factor (cupy arrays; float64 solve, cast to the solution dtype)."""
        import cupy as cp
        device_rhs = cp.asarray(rhs, dtype=cp.float64)
        device_solution = self._factor.solve(device_rhs)
        solution[...] = device_solution.astype(solution.dtype, copy=False)
        self.n_solves += 1
        self.last_iters = 1
        if self.compute_residual and self._device_operator is not None:
            rhs_norm = float(cp.linalg.norm(device_rhs))
            residual = float(cp.linalg.norm(device_rhs - self._device_operator.astype(cp.float64) @ device_solution))
            self.last_residual = residual / rhs_norm if rhs_norm > 0.0 else residual

    def work_counters(self) -> dict:
        """The runtime's counters plus the solves of this system."""
        return dict(self.runtime.work_counters(), solves=self.n_solves)


def time_estimate(values, work):
    """Run-time recipe of a sparse_direct role (core.time_estimate.SolverWork): host SuperLU factorisations (+ the
    CSR transfer on the gpu) and triangular solves (+ rhs / solution transfers), the grid curves times
    work.factor_scale / solve_scale; cuDSS by _cudss_time_items (2D: the profile's cudss curves, 3D: the fit); the
    host direct back ends by the SuperLU items x HOST_DIRECT_FACTOR_TIME / HOST_DIRECT_SOLVE_TIME (mixed solves x
    HOST_DIRECT_MIXED_FULL_SOLVES; uncalibrated); cusolver_qr is not estimated."""
    from ..core import time_estimate as te
    back_end = expected_back_end(values.get('back_end', 'auto'), work.backend)
    if back_end in CUDSS_BACK_ENDS and work.dimensions in (2, 3):
        return te.TimeEstimate(_cudss_time_items(back_end, work))
    if back_end not in HOST_FACTOR_BACK_ENDS:
        return te.TimeEstimate(not_estimated=(f"{work.role} solver (sparse_direct back_end '{back_end}', "
                                              f"{work.dimensions}D)",))
    host_direct = back_end in HOST_DIRECT_BACK_ENDS
    label = factor_label(back_end)   # 'SuperLU' for scipy_splu
    factor_scale = work.factor_scale * (HOST_DIRECT_FACTOR_TIME if host_direct else 1.0)
    solve_scale = work.solve_scale * ((HOST_DIRECT_SOLVE_TIME * (HOST_DIRECT_MIXED_FULL_SOLVES
                                                                 if back_end in MIXED_BACK_ENDS else 1.0))
                                      if host_direct else 1.0)
    gpu = work.backend == 'gpu'
    solve = (te.HostSolve(work.unknowns, work.dimensions, work.complex, work.rhs, scale=solve_scale),) + (
        (te.Transfer(2.0 * work.unknowns * work.value_bytes * work.rhs),) if gpu else ())
    factor = te.HostLU(work.unknowns, work.dimensions, work.complex, scale=factor_scale)
    items = [te.TimeItem(f'{label} factorisations', 'factorisation', float(work.factorizations),
                         work.factorizations_kind, (factor,),
                         per_step=work.factor_per_step, counter=work.counter('factorizations'),
                         seconds_counter=work.counter('factor_seconds')),   # timed around the factorisation only
             te.TimeItem(f'{label} solves', 'solve', float(work.solves), work.solves_kind, solve,
                         per_step=work.per_step)]
    if gpu:
        items.append(te.TimeItem(f'{label} operator transfers', 'factorisation', float(work.factorizations),
                                 work.factorizations_kind, (te.Transfer(work.nnz * (work.value_bytes + 4)),),
                                 per_step=work.factor_per_step))
    return te.TimeEstimate(tuple(items))


def cudss_plan_fraction(work) -> float:
    """Share of a fresh plan the work's plan costs: CUDSS_KEPT_ORDERING_PLAN_FRACTION when the caller marks it as
    finding a kept ordering (SolverWork.ordering_kept, R1c: a sweep's later runs on one mesh, an eigen role after a
    driven role of the same pattern), else 1."""
    return CUDSS_KEPT_ORDERING_PLAN_FRACTION if getattr(work, 'ordering_kept', False) else 1.0


def cudss_2d_ops(mixed, work) -> tuple:
    """(plan, factorisation, solve) ops of a 2D cuDSS factor on the profile's 5-point cudss curves (the device
    refactorisation curve with work.device_values), scaled by work.factor_scale / solve_scale (FEM fill) and, for a real
    operator, CUDSS_REAL_FLOPS / CUDSS_REAL_BYTES; the plan by cudss_plan_fraction too."""
    from ..core import time_estimate as te
    n_unknowns = float(work.unknowns)
    real_flops, real_bytes = (1.0, 1.0) if work.complex else (CUDSS_REAL_FLOPS, CUDSS_REAL_BYTES)
    return (te.DevicePlan(n_unknowns, 2, scale=work.factor_scale * cudss_plan_fraction(work)),
            te.DeviceFactor(n_unknowns, 2, mixed, scale=work.factor_scale * real_flops,
                            device_values=work.device_values),
            te.DeviceSolve(n_unknowns, 2, mixed, scale=work.solve_scale * real_bytes))


def _cudss_time_items(back_end, work):
    """TimeItems of a cuDSS back end: one host plan (later factorisations reuse it while the pattern holds: previous=;
    on a kept ordering: cudss_plan_fraction), device factorisations and solves (+ R3's deferred check of each refreshed
    double factor from device values: one product and two norms); 2D on the profile curves (cudss_2d_ops: the
    refactorisation curve holds the host route's preparation, or the device one with work.device_values; the mixed
    solve curve its refinement), 3D on the arpack_shift_invert fit scaled by CUDSS_REAL_* for real operators and
    CUDSS_MIXED_FULL_SOLVES (module constants), the values moved per refactorisation by its route."""
    from ..core import time_estimate as te
    from . import arpack_shift_invert as fit
    n_unknowns = float(work.unknowns)
    mixed = back_end == 'cudss_mixed'
    factorizations = float(work.factorizations)
    checks = ()
    if work.device_values and not mixed and factorizations > 1.0:   # R3: a refreshed double factor's first solve
        checks = (te.TimeItem('cuDSS deferred accuracy checks', 'factorisation', factorizations - 1.0,
                              work.factorizations_kind, (te.Spmv(n_unknowns, work.nnz, work.value_bytes),
                                                         (2.0, te.Reduction(n_unknowns, work.value_bytes, True))),
                              per_step=work.factor_per_step),)
    if work.dimensions == 2:
        plan, factor, solve = cudss_2d_ops(mixed, work)
        return (te.TimeItem('cuDSS plans', 'plan', 1.0 if factorizations else 0.0, 'guess', (plan,),
                            counter=work.counter('cudss_plans'), seconds_counter=work.counter('factor_seconds')),
                te.TimeItem('cuDSS factorisations', 'factorisation', factorizations, work.factorizations_kind,
                            (factor,), per_step=work.factor_per_step, counter=work.counter('factorizations'),
                            seconds_counter=work.counter('factor_seconds')),
                te.TimeItem('cuDSS solves', 'solve', float(work.solves) * float(work.rhs), work.solves_kind,
                            (solve,), per_step=work.per_step)) + checks
    real_bytes, real_flops = (1.0, 1.0) if work.complex else (CUDSS_REAL_BYTES, CUDSS_REAL_FLOPS)
    device_bytes = (fit.CUDSS_BYTES_PER_UNKNOWN * n_unknowns * (n_unknowns / fit.CUDSS_REFERENCE_UNKNOWNS)
                    ** fit.CUDSS_BYTES_GROWTH * (1.0 if mixed else fit.CUDSS_DOUBLE_BYTES) * real_bytes)
    flops = (fit.CUDSS_FACTOR_FLOPS * (n_unknowns / fit.CUDSS_REFERENCE_UNKNOWNS) ** fit.CUDSS_FACTOR_GROWTH
             * (1.0 if mixed else fit.CUDSS_DOUBLE_FACTOR_TIME) * real_flops)
    single_solve = te.Elementwise(fit.CUDSS_SOLVE_READS * device_bytes, work.value_bytes * n_unknowns)
    solve = (((CUDSS_MIXED_FULL_SOLVES, single_solve),
              (CUDSS_MIXED_FULL_SOLVES - 1.0, te.Spmv(n_unknowns, work.nnz, work.value_bytes))) if mixed
             else (single_solve,))
    # a refactorisation moves the values only: none from device values (DevicePencil, device CSR), one upload on the
    # host route's planned pattern (H1); the plan's full upload is in the plan item
    upload = () if work.device_values else (te.Transfer(work.nnz * work.value_bytes),)
    return (te.TimeItem('cuDSS plans', 'plan', 1.0 if factorizations else 0.0, 'guess',
                        (te.HostStream(fit.CUDSS_PLAN_HOST_BYTES * n_unknowns * cudss_plan_fraction(work)),
                         te.Transfer(work.nnz * (work.value_bytes + 4))),
                        counter=work.counter('cudss_plans'),
                        seconds_counter=work.counter('factor_seconds')),   # factor_seconds includes the plans
            te.TimeItem('cuDSS factorisations', 'factorisation', factorizations, work.factorizations_kind,
                        (te.DenseFlops(flops), *upload), per_step=work.factor_per_step,
                        counter=work.counter('factorizations'), seconds_counter=work.counter('factor_seconds')),
            te.TimeItem('cuDSS solves', 'solve', float(work.solves) * float(work.rhs), work.solves_kind, solve,
                        per_step=work.per_step)) + checks
