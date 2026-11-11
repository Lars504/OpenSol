"""Host multicore sparse direct factors by MUMPS (python-mumps) and MKL PARDISO (pydiso): back ends mumps / pardiso (+
_mixed) of sparse_direct and arpack_shift_invert. The wrappers are imported inside functions only (open_sol imports
and runs without them; never chosen by 'auto'); numpy and the cudss_factor helpers at module level, so the names and
helpers here are GUI-thread safe."""
# HostDirectFactorization is a sparse_direct.Factorization / CudssFactorization stand-in (solve / solve_host,
# transposed solves, recyclable GMRES re-solves, inertia, refactorize with the analysis reused, release). It factors
# E = R S R, S = D A (D the caller's symmetrizer, else S = A), R = |diag S|^-1/2 (or the caller's row_scale), entry
# by entry on the canonical CSR, as CudssFactorization does: LDL^T of a (complex) symmetric E (complex symmetric, not
# Hermitian) once relative_asymmetry passes SYMMETRY_TOLERANCE, else LU; PARDISO factors a real positive_definite
# request by Cholesky (matrix type 2; a failed one: LDL^T, sticky), MUMPS always by LDL^T (python-mumps exposes no
# SYM=1). Solves map back: A x = b as x = R E^-1 R (D) b; A^T x = b as x = R E^-T R b (no symmetrizer; E^T = E for a
# symmetric kind) or x = D R E^-T R b (symmetrizer: A^T = S^T D^-1). Both libraries solve A^T natively (MUMPS ICNTL(9),
# PARDISO iparm[11] = 2: transposed, not conjugated).
# Probe after every factorisation (seeded normal rhs of E, complex unless real): relative residual <= ACCEPT_RESIDUAL
# or normwise backward error <= BACKWARD_ERROR_TOLERANCE (cudss_factor gates). A double factor that misses switches
# to float64 GMRES-IR on its own factor (cudss_factor.gmres_refine, every later solve too; refinement_steps = the
# probe's iterations); still missing: released, HostDirectAccuracyError. The libraries' own iterative refinement
# (MUMPS ICNTL(10), PARDISO iparm[7]) is not used. PARDISO perturbs tiny pivots silently (1e-13 unsymmetric, 1e-8
# symmetric): only the probe catches them, so a singular operator usually surfaces as HostDirectAccuracyError there.
# precision 'mixed' (back ends *_mixed): float32 / complex64 factor of E, float64 E kept for residuals; a solve without
# a step count refines by gmres_refine to solve_backward_error (MIXED_SOLVE_BACKWARD_ERROR), an int takes that many
# plain steps (the eigen polish); each low-precision solve scales its rhs to unit norm. The mixed probe uses the stall
# test and the mixed gates; a miss refactors in double (sticky: factor_precision 'double', last_double_fallback,
# mixed_rejection). Remembered rejections per pattern (cuDSS H4) are not implemented.
# Memory: after the analysis the library's own estimate (MUMPS INFOG(17) MB, PARDISO max(iparm[14], iparm[15] +
# iparm[16]) KiB) above HOST_FRACTION of core.host_memory.available_bytes() is HostDirectMemoryError (no check when
# that is unknown). The factor lives in library memory outside numpy: holders release() it before replacing it.
# Refactorize: the same canonical pattern (indptr, indices) and kind refactors numerically on the kept analysis.
# Value refreshes (refactorize_values, the sparse_direct.HostPencil route of frequency sweeps): values of a canonical
# pattern the caller never modifies in place, matched by identity first (no canonicalising copy, no array compare);
# the per-entry rows and diagonal positions are kept per pattern, ||E|| is lazy (built only when a residual misses
# ACCEPT_RESIDUAL), and with defer_check a double factor on the kept analysis whose last probe passed unrefined skips
# its probe: the next solve is the check (a miss: GMRES-IR and solve again, a second miss releases and raises), so
# probe_residual / refinement_steps lag one factor (as cuDSS R3). Mixed factors always probe.
# PARDISO values: when pydiso's private arrays match the analysed pattern (checked once per analysis) a factorisation
# writes the kept values straight into the handle (`_data`, then `_factor`), skipping pydiso's refactor (sp.triu of a
# symmetric type, sort, sum_duplicates per call); otherwise the public refactor.
# Threads: library defaults (MUMPS: OMP_NUM_THREADS / BLAS; PARDISO: MKL_NUM_THREADS); both wrappers release the GIL
# in their library calls (python-mumps 0.0.6, pydiso 0.3.1), which are uninterruptible like SuperLU.
# PARDISO beside an OpenBLAS numpy / scipy (pip wheels, conda-forge defaults): two thread pools that both spin after a
# call, so MKL calls alternating with threaded BLAS calls (norms, Gram-Schmidt, ARPACK) oversubscribe the cores
# (measured, 4 cores: PARDISO solves 4 - 10 x slower, an ARPACK shift-invert run 5 - 25 x). quiet_openblas holds the
# OpenBLAS pools at one thread (threadpoolctl, optional: nothing without it) for the duration of every PARDISO
# factorisation / solve of a HostDirectFactorization, and consumers iterating around such a factor wrap their loop
# in it (arpack_shift_invert's ARPACK runs); MKL's own pool is never touched, MUMPS never limited (its BLAS may be
# that OpenBLAS).
# No automatic fallback: an explicit host node raises (HOST_DIRECT_REFUSALS).
# Library calls live only in the adapter section at the end (_MumpsFactor, _PardisoFactor): PARDISO run against
# pydiso 0.3.1 / oneMKL 2026.2, MUMPS checked against the sources of python-mumps 0.0.6 only (not run).
import contextlib
import functools
import importlib.metadata
import importlib.util
import math
import os
import platform
import time

import numpy as np

from .cudss_factor import (ACCEPT_RESIDUAL, BACKWARD_ERROR_TOLERANCE, GMRES_RESTART, MIXED_SOLVE_BACKWARD_ERROR,
                           MIXED_SOLVE_TOLERANCE, SYMMETRY_TOLERANCE, canonical_csr, csr_rows, gmres_refine,
                           relative_asymmetry)

HOST_DIRECT_BACK_ENDS = ('mumps', 'mumps_mixed', 'pardiso', 'pardiso_mixed')   # the one list (sparse_direct back_end)
HOST_DIRECT_FACTORIZATIONS = HOST_DIRECT_BACK_ENDS                           # arpack_shift_invert factorization values
HOST_DIRECT_DOUBLE_BACK_ENDS = tuple(name for name in HOST_DIRECT_BACK_ENDS if not name.endswith('_mixed'))
HOST_DIRECT_MIXED_BACK_ENDS = tuple(name for name in HOST_DIRECT_BACK_ENDS if name.endswith('_mixed'))
HOST_DIRECT_LIBRARIES = ('mumps', 'pardiso')            # factor.back_end values
LIBRARY_LABELS = {'mumps': 'MUMPS', 'pardiso': 'PARDISO'}
WRAPPERS = {'mumps': ('mumps', 'python-mumps'), 'pardiso': ('pydiso.mkl_solver', 'pydiso')}   # (module, distribution)
HOST_FRACTION = 0.8   # largest share of the available RAM the library's own estimate may claim (as cuDSS DEVICE_FRACTION)
# Memory heuristics relative to the SuperLU heuristics of core.memory_estimate (uncalibrated: refit from the counters
# host_direct_lu_entries / host_factor_bytes (sparse_direct) and eigen_lu_entries (arpack_shift_invert) of summary.json)
LU_FILL_OVER_SUPERLU = 1.0    # nested dissection (METIS) usually fills less than SuperLU's MMD: 1.0 is conservative
LDLT_FILL_OVER_SUPERLU = 0.5  # one triangle
WORKSPACE_OVER_FACTOR = 1.25  # MUMPS ICNTL(14) relaxation / PARDISO work arrays (guess)
MEGABYTE = 2 ** 20            # MUMPS reports memory in MB (read as MiB: at most 5 % high, the guard's safe side)
_PROBE_SEED = 11
_KIND_TEXTS = {'general': 'LU', 'symmetric': 'LDL^T', 'spd': 'Cholesky'}
_PARDISO_MACHINES = ('x86_64', 'amd64')
_SUPERLU_NAMES = (None, 'auto', 'superlu', 'scipy_splu')
_TO_SUPERLU = "use host SuperLU instead (sparse_direct back_end 'scipy_splu', eigen solver factorization 'superlu')"


class HostDirectSingularError(RuntimeError):
    """A MUMPS / PARDISO factorisation that failed with a zero pivot (MUMPS INFO(1) -10 / -6, PARDISO error -4 / -7)."""


class HostDirectAccuracyError(RuntimeError):
    """MUMPS / PARDISO solves that miss the probe accuracy after float64 refinement (perturbed pivots, an
    ill-conditioned or singular operator)."""


class HostDirectMemoryError(ValueError):
    """A MUMPS / PARDISO factorisation refused or failed for memory: its own estimate after the analysis over
    HOST_FRACTION of the available RAM, or an allocation failure of the library."""


HOST_DIRECT_REFUSALS = (HostDirectSingularError, HostDirectAccuracyError, HostDirectMemoryError)

# equilibration E = R S R of every (re)factorisation (next-gains N6): a numba row loop writing the scaled values into
# the new array from this many stored entries (below, or without numba: the numpy gathers)
HOST_SCALE_MIN_ENTRIES = 200_000


def _scale_host(out, values, indptr, indices, row_factor, row_scale) -> None:
    """out[k] = values[k] x row_factor[row] x row_scale[column] by numpy gathers."""
    np.multiply(values, np.repeat(row_factor, np.diff(indptr)), out=out)
    out *= row_scale[indices]


def _scale_numba(numba):
    @numba.njit(parallel=True, cache=True)
    def scale(out, values, indptr, indices, row_factor, row_scale):
        for prange_row in numba.prange(indptr.size - 1):
            row = np.int64(prange_row)
            factor = row_factor[row]
            for position in range(indptr[row], indptr[row + 1]):
                out[position] = values[position] * (factor * row_scale[indices[position]])
    return scale


def scaled_values(values, indptr, indices, row_scale, symmetrizer, value_dtype, numba=None) -> np.ndarray:
    """New value_dtype array of values x symmetrizer[row] x row_scale[row] x row_scale[column] on the CSR pattern
    (symmetrizer None: 1). numba None: the numba row loop from HOST_SCALE_MIN_ENTRIES (core.kernels.host_twin), True /
    False: forced."""
    from ..core.kernels import host_twin
    scale = host_twin('solvers.host_scale', _scale_host, _scale_numba)
    values = np.asarray(values).reshape(-1)
    if numba and scale.host_kind != 'numba':
        raise RuntimeError('host_direct.scaled_values(numba=True): the numba scaling is not available')
    if not (values.size >= HOST_SCALE_MIN_ENTRIES if numba is None else numba):
        scale = scale.cpu
    out = np.empty(values.size, dtype=value_dtype)
    scale(out, values.astype(value_dtype, copy=False), indptr, indices,
          row_scale if symmetrizer is None else symmetrizer * row_scale, row_scale)
    return out


def library_of(name) -> str:
    """Library ('mumps' / 'pardiso') of a host direct back end or factorization name ('mumps_mixed' -> 'mumps');
    ValueError for any other name."""
    library = name[:-len('_mixed')] if isinstance(name, str) and name.endswith('_mixed') else name
    if library not in HOST_DIRECT_LIBRARIES:
        raise ValueError(f"not a host direct back end: {name!r} (one of {', '.join(HOST_DIRECT_BACK_ENDS)})")
    return library


def host_direct_precision(name) -> str:
    """'mixed' for a *_mixed host direct name, else 'double'."""
    return 'mixed' if isinstance(name, str) and name.endswith('_mixed') else 'double'


def double_of(name):
    """The double-precision name of a host *_mixed name ('pardiso_mixed' -> 'pardiso'); any other name unchanged."""
    return name[:-len('_mixed')] if name in HOST_DIRECT_MIXED_BACK_ENDS else name


def is_host_direct_factor(factor) -> bool:
    """True for a HostDirectFactorization (its back_end is a library name of HOST_DIRECT_LIBRARIES)."""
    return getattr(factor, 'back_end', None) in HOST_DIRECT_LIBRARIES


def _is_host_name(name) -> bool:
    return name in HOST_DIRECT_BACK_ENDS or name in HOST_DIRECT_LIBRARIES


def _distribution_installed(names) -> bool:
    """True when one of these distribution names has installed metadata (nothing imported)."""
    for name in names:
        try:
            importlib.metadata.distribution(name)
            return True
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception:   # unreadable metadata: not counted as installed
            continue
    return False


@functools.lru_cache(maxsize=None)
def host_direct_available(name) -> bool:
    """True when the wrapper of a host direct back end / library name is installed (find_spec and distribution
    metadata only, nothing imported: GUI-thread safe): MUMPS = module 'mumps' of the distribution python-mumps (PyMUMPS
    installs another 'mumps'), PARDISO = pydiso on an x86-64 machine. False for any other name."""
    if not _is_host_name(name):
        return False
    library = library_of(name)
    try:
        if library == 'mumps':
            return (importlib.util.find_spec('mumps') is not None
                    and _distribution_installed(('python-mumps', 'python_mumps')))
        return (importlib.util.find_spec('pydiso') is not None
                and platform.machine().lower() in _PARDISO_MACHINES)
    except (ImportError, ValueError):
        return False


def install_hint(name) -> str:
    """How to install the wrapper of a host direct back end / library name."""
    if library_of(name) == 'mumps':
        return 'conda install -c conda-forge python-mumps (pip install python-mumps builds against an installed MUMPS)'
    return ('pip install pydiso (Linux / Windows x86-64 wheels, pulls MKL) or conda install -c conda-forge pydiso; '
            'x86-64 only')


def missing_text(name) -> str:
    """Validation / runtime text of a host direct back end or factorization name whose wrapper is not available
    (neutral: serves the sparse_direct back_end and the arpack_shift_invert factorization setting alike)."""
    library = library_of(name)
    distribution = WRAPPERS[library][1]
    if library == 'pardiso' and platform.machine().lower() not in _PARDISO_MACHINES:
        return (f"'{name}' needs {distribution} (Intel MKL PARDISO), which runs on x86-64 processors only (this "
                f"machine: {platform.machine() or 'unknown'}); use MUMPS ('mumps') or host SuperLU instead")
    return f"'{name}' needs {distribution}, which is not installed: {install_hint(name)}"


def factor_label(name) -> str:
    """Library label of a back end / factorization name for texts: 'MUMPS' / 'PARDISO' (host direct names), 'cuDSS'
    (cudss / cudss_mixed), 'cuSOLVER QR' (cusolver_qr), else 'SuperLU' (superlu, scipy_splu, auto, None)."""
    if _is_host_name(name):
        return LIBRARY_LABELS[library_of(name)]
    if name in ('cudss', 'cudss_mixed'):
        return 'cuDSS'
    if name == 'cusolver_qr':
        return 'cuSOLVER QR'
    return 'SuperLU'


def _host_scale(symmetric, complex_values, mixed) -> float:
    """Host direct factor bytes over the SuperLU heuristic: fill ratio x WORKSPACE_OVER_FACTOR x the mixed value share."""
    fill = LDLT_FILL_OVER_SUPERLU if symmetric else LU_FILL_OVER_SUPERLU
    scale = fill * WORKSPACE_OVER_FACTOR
    if mixed:
        value_bytes = 16.0 if complex_values else 8.0
        scale *= (value_bytes / 2.0 + 4.0) / (value_bytes + 4.0)   # values halve, the int32 index per entry stays
    return scale


def superlu_bytes_scale(name, symmetric=False, complex_values=True) -> float:
    """Factor over the booked SuperLU heuristic bytes for the back end / factorization `name` (estimators booking
    SuperLU items multiply by it): 1.0 for SuperLU names (and every non-host name), else the host direct fill ratio
    (LDL^T for symmetric) x WORKSPACE_OVER_FACTOR x the mixed value share (uncalibrated)."""
    if not _is_host_name(name):
        return 1.0
    return _host_scale(symmetric, complex_values, host_direct_precision(name) == 'mixed')


def factor_host_bytes(n_unknowns, dimensions=2, complex_values=True, symmetric=False, mixed=False,
                      counted_triangle_entries=None) -> int:
    """Heuristic host bytes of a MUMPS / PARDISO factor: the SuperLU heuristic of core.memory_estimate (complex:
    host_lu_bytes, 2D n log2 n or 3D curl-curl n^(4/3); real: host_real_lu_bytes; counted_triangle_entries: the
    counted factor triangle, host_counted_lu_bytes) x superlu_bytes_scale (uncalibrated)."""
    from ..core import memory_estimate
    n_unknowns = int(n_unknowns)
    if counted_triangle_entries is not None:
        superlu = memory_estimate.host_counted_lu_bytes(counted_triangle_entries, n_unknowns)
        if not complex_values:
            superlu = superlu * memory_estimate.FLOAT64_CSR_ENTRY_BYTES / memory_estimate.COMPLEX128_CSR_ENTRY_BYTES
    elif complex_values:
        superlu = memory_estimate.host_lu_bytes(n_unknowns, dimensions=dimensions)
    else:
        superlu = memory_estimate.host_real_lu_bytes(n_unknowns, dimensions=dimensions)
    return int(superlu * _host_scale(symmetric, complex_values, mixed))


def solver_host_bytes(n_unknowns, entries, complex_values=True, mixed=False, symmetric=False) -> int:
    """Host bytes a HostDirectFactorization keeps beside the library factor for `entries` stored operator entries: the
    float64 / complex128 E (value + int32 index per entry, indptr; probes, residuals, refinement), its int64 row per
    entry kept per pattern, the row scale, the copy of the matrix the wrapper hands to the library (MUMPS: COO of one
    triangle for symmetric kinds; PARDISO: the upper triangle, or E itself in double, + the int64 take of the values
    written into its handle), mixed: the factor-dtype CSR and the GMRES vectors of one solve."""
    n_unknowns, entries = int(n_unknowns), int(entries)
    value_bytes = 16 if complex_values else 8
    factor_value_bytes = value_bytes // 2 if mixed else value_bytes
    stored = (entries + n_unknowns) // 2 if symmetric else entries
    kept = (value_bytes + 4 + 8) * entries + 4 * (n_unknowns + 1) + 8 * n_unknowns
    library_copy = (factor_value_bytes + 8 + (8 if symmetric else 0)) * stored
    if mixed:
        kept += (factor_value_bytes + 4) * entries + 4 * (n_unknowns + 1)
        kept += (2 * GMRES_RESTART + 10) * value_bytes * n_unknowns
    return int(kept + library_copy)


_OPENBLAS_POOLS = {}   # 'pools': threadpoolctl controller of the loaded OpenBLAS pools, None without any


def _openblas_pools(refresh=False):
    """threadpoolctl controller of the OpenBLAS thread pools loaded in this process (numpy's, scipy's), None when
    there is none (numpy on MKL) or threadpoolctl is not installed; looked up on first use and again with refresh
    (scipy loads its own OpenBLAS lazily; a lookup costs ~0.4 ms)."""
    if refresh or 'pools' not in _OPENBLAS_POOLS:
        pools = None
        try:
            import threadpoolctl
            found = threadpoolctl.ThreadpoolController().select(internal_api='openblas')
            if found.lib_controllers:
                pools = found
        except Exception:   # not installed, or a library it cannot read: no limit
            pools = None
        _OPENBLAS_POOLS['pools'] = pools
    return _OPENBLAS_POOLS['pools']


@contextlib.contextmanager
def quiet_openblas(name, refresh=False):
    """Context holding the OpenBLAS thread pools of numpy / scipy at one thread (restored on exit) around work on a
    PARDISO factor (`name`: a back end, factorization or library name; module header "Threads"); yields the number of
    pools held: 0, and nothing changes, for any other name, without threadpoolctl or without a loaded OpenBLAS. Nests.
    Process-wide while it lasts: only vector-sized BLAS work (norms, Krylov bases) belongs inside."""
    pools = _openblas_pools(refresh) if _is_host_name(name) and library_of(name) == 'pardiso' else None
    if pools is None:
        yield 0
        return
    with pools.limit(limits=1):
        yield len(pools.lib_controllers)


def _openblas_quiet(method):
    """HostDirectFactorization method run inside quiet_openblas of its library."""
    @functools.wraps(method)
    def quiet(self, *args, **kwargs):
        with quiet_openblas(self.back_end):
            return method(self, *args, **kwargs)
    return quiet


def _host_vector(vector):
    """A cupy vector as a numpy copy; numpy arrays, sequences and None as given."""
    if vector is None or isinstance(vector, np.ndarray) or not hasattr(vector, 'get'):
        return vector
    return vector.get()


def _gib(n_bytes) -> str:
    return f'{float(n_bytes) / 2 ** 30:.2f} GiB'


def _figure(value) -> str:
    return 'not computed' if value is None else f'{value:.3g}'


class _FactorRejected(Exception):
    """A factorisation or probe that failed (wraps the error raised when no fallback is left)."""

    def __init__(self, error):
        super().__init__(str(error))
        self.error = error


class HostDirectFactorization:
    """Host MUMPS / PARDISO LU / LDL^T / Cholesky of the equilibrated operator (module header; precision mixed: float32
    / complex64 factor + float64 GMRES refinement) of a square host scipy operator: solve (cupy or numpy in, the same
    out; host_arrays: numpy), solve_host (numpy), transposed solves, solve_preconditioned* (GMRES on this factor of an
    earlier operator), inertia, refactorize (analysis reused on an unchanged pattern), release."""

    host_arrays = False   # numpy in / out of solve: sparse_direct.factorize sets it for a scipy operator
    recyclable = True     # solve_preconditioned(_host): GMRES re-solves on this factor of an earlier operator

    def __init__(self, host_operator, back_end, real=False, symmetric=False, positive_definite=False,
                 symmetrizer=None, row_scale=None, symmetric_known=False):
        if back_end not in HOST_DIRECT_BACK_ENDS:
            raise ValueError(f"HostDirectFactorization back_end must be one of {', '.join(HOST_DIRECT_BACK_ENDS)}, got "
                             f'{back_end!r}')
        if not host_direct_available(back_end):
            raise ValueError(missing_text(back_end))
        if positive_definite and not real:
            raise ValueError('HostDirectFactorization(positive_definite=True) needs real=True (SPD; Hermitian HPD is '
                             'not offered)')
        self.back_end = library_of(back_end)    # the library, both precisions (as CudssFactorization 'cudss')
        self.precision = host_direct_precision(back_end)   # requested; factor_precision is what is factored
        self.label = LIBRARY_LABELS[self.back_end]
        self.real = bool(real)
        self.value_dtype = np.float64 if real else np.complex128   # solve in / out
        self.positive_definite = bool(positive_definite)
        self.symmetric_requested = bool(symmetric) or positive_definite or symmetrizer is not None
        self.has_symmetrizer = symmetrizer is not None
        self.n_unknowns = None
        self._mixed_rejected = False      # a mixed factor failed its probe once: double factors from then on
        self._cholesky_rejected = False   # a Cholesky failed once: LDL^T from then on
        self._double_refined = False      # a double factor that missed its probe: GMRES-refined solves
        self._factor_dtype = self._wanted_factor_dtype()
        self._library = None              # _MumpsFactor / _PardisoFactor of the analysed pattern
        self._operator = self._operator_transpose = self._row_scale = self._symmetrizer = None
        self._pattern = None              # (indptr, indices) copies of the analysed canonical pattern
        self._pattern_source = None       # (indptr, indices) of the refactorize_values caller, matched by identity
        self._pattern_rows = None         # int64 row of every stored entry of that pattern (kept per pattern)
        self._diagonal_entries = None     # positions of its diagonal entries (kept per pattern)
        self._diagonal_rows = None        # PARDISO symmetric kinds: rows of that pattern without a diagonal entry
        self._probe = None
        self._operator_norm = None        # lazy sqrt(||E||_1 ||E||_inf) of the factorised E (operator_norm)
        self._check_pending = False       # defer_check: the next solve judges this factor (_checked_solve)
        self._probe_clean = False         # the last probe / check of a double factor passed unrefined
        self.matrix_kind = None           # 'general' | 'symmetric' | 'spd' of the analysed factor
        self.asymmetry = None             # relative asymmetry of the last E (None: symmetry not requested)
        self.description = f'{self.label} (not factorised)'
        self.solve_backward_error = MIXED_SOLVE_BACKWARD_ERROR   # refined solves: target backward error (per factor)
        self._last_solve_errors = (None, None, None)
        self.host_bytes = 0               # library factor bytes (MUMPS INFOG(19) MB, PARDISO iparm[15] + iparm[16] KiB)
        self.peak_bytes = 0               # the library's estimate after the analysis (the memory guard's number)
        self.lu_entries = 0               # factor entries (MUMPS INFOG(29), PARDISO iparm[17])
        self.perturbed_pivots = 0         # PARDISO iparm[13]; MUMPS INFOG(25) + INFOG(28) (static / null pivots)
        self.threads = None               # library thread count when known
        self.openblas_pools = 0           # OpenBLAS pools held at one thread during this factor's calls
        self.n_analyses = 0               # analyses (ordering + symbolic) and factorisations of this object
        self.n_factorizations = 0
        self.last_analyses = 0            # analyses of the last refactorize (fallback re-analyses included)
        self.analysis_seconds = 0.0       # last analysis (0 when the last refactorize reused it)
        self.total_analysis_seconds = 0.0
        self.factor_seconds = 0.0         # last refactorize: equilibration, (analysis,) factorisation and probe
        self.total_factor_seconds = 0.0
        self.last_plan_reused = False     # the last refactorize kept the analysis (name as CudssFactorization)
        self.n_solves = 0                 # library solves (probes and refinement solves included) and their wall time
        self.solve_seconds = 0.0
        self.n_refinements = 0            # GMRES / plain refinement iterations of all solves, and of the last one
        self.last_refinements = 0
        self.refinement_steps = 0         # refinement iterations of the last probe (0: unrefined double solves)
        self.probe_residual = None
        self.probe_backward_error = None
        self.single_residual = None       # mixed: unrefined probe residual of the low-precision factor
        self.last_double_fallback = False   # the last refactorize replaced a rejected mixed factor by a double one
        self.mixed_rejection = None
        self.refactorize(host_operator, row_scale=row_scale, symmetrizer=symmetrizer, symmetric_known=symmetric_known)

    # --- state ---------------------------------------------------------------------------------------------------
    @property
    def valid(self) -> bool:
        """False after release()."""
        return self._library is not None

    @property
    def symmetric(self) -> bool:
        """True for an LDL^T or Cholesky factor."""
        return self.matrix_kind in ('symmetric', 'spd')

    @property
    def operator_norm(self):
        """sqrt(||E||_1 ||E||_inf) of the factorised E (backward errors), built on first use per factorisation."""
        if self._operator_norm is None and self._operator is not None:
            magnitudes = np.abs(self._operator.data)
            n_rows = self.n_unknowns
            self._operator_norm = float(np.sqrt(
                np.bincount(self._pattern_rows, magnitudes, minlength=n_rows).max(initial=0.0)
                * np.bincount(self._operator.indices, magnitudes, minlength=n_rows).max(initial=0.0)))
        return self._operator_norm

    @property
    def mixed_factor(self) -> bool:
        """True while the factor is low precision (precision 'mixed' and no double fallback yet)."""
        return self._factor_dtype != self.value_dtype

    @property
    def factor_precision(self) -> str:
        """'mixed' or 'double': what is factored (precision is what was requested)."""
        return 'mixed' if self.mixed_factor else 'double'

    def _wanted_factor_dtype(self):
        """Factor dtype of the next factorisation: value_dtype in double or after a rejected mixed factor, else float32
        / complex64."""
        if self.precision == 'double' or self._mixed_rejected:
            return self.value_dtype
        return np.float32 if self.real else np.complex64

    def _requested_kind(self) -> str:
        """Kind of an operator found symmetric under a symmetry request: a Cholesky only on PARDISO, real, double and
        not rejected before (as cuDSS: no low-precision Cholesky), else LDL^T."""
        cholesky = (self.positive_definite and self.back_end == 'pardiso' and not self._cholesky_rejected
                    and self._wanted_factor_dtype() == self.value_dtype)
        return 'spd' if cholesky else 'symmetric'

    def can_refactorize(self, real, precision, symmetric=False, positive_definite=False, symmetrizer=None) -> bool:
        """Whether refactorize serves a request with these options (same rule as CudssFactorization.can_refactorize)."""
        return (self.valid and bool(real) == self.real and precision == self.precision
                and bool(positive_definite) == self.positive_definite
                and (bool(symmetric) or positive_definite or symmetrizer is not None) == self.symmetric_requested
                and (symmetrizer is not None) == self.has_symmetrizer)

    # --- factorisation -------------------------------------------------------------------------------------------
    def refactorize(self, host_operator, row_scale=None, symmetrizer=None, symmetric_known=False) -> bool:
        """Factorise a host scipy operator of the same size (new row_scale / symmetrizer values allowed; a symmetrizer
        exactly when the factor was built with one): numeric factorisation only on the kept analysis when the
        canonical pattern and the kind are unchanged, else a new analysis. symmetric_known: a canonical CSR known to be
        symmetric (no asymmetry check). True when the analysis was reused. Raises HOST_DIRECT_REFUSALS (released)."""
        with quiet_openblas(self.back_end, refresh=True) as self.openblas_pools:
            return self._refactorize(lambda: self._equilibrated(host_operator, row_scale, symmetrizer),
                                     symmetric_known)

    def refactorize_values(self, values, indptr, indices, row_scale=None, symmetric_known=False,
                           defer_check=False) -> bool:
        """Factorise `values` of a canonical CSR pattern (indptr, indices: sorted, no duplicates) of the same size, the
        sparse_direct.HostPencil route (module header "Value refreshes"): the pattern matched by identity first, so
        callers never modify those arrays in place; symmetric_known as refactorize; defer_check: the next solve checks a
        double factor on the kept analysis whose last probe passed unrefined. True when the analysis was reused."""
        if self.has_symmetrizer:
            raise ValueError('HostDirectFactorization.refactorize_values: not for a factor built with a symmetrizer')
        with quiet_openblas(self.back_end, refresh=True) as self.openblas_pools:
            return self._refactorize(lambda: self._equilibrated_values(values, indptr, indices, row_scale),
                                     symmetric_known, defer_check)

    def _refactorize(self, equilibrate, symmetric_known, defer_check=False) -> bool:
        started = time.perf_counter()
        analyses = self.n_analyses
        operator, same_pattern = equilibrate()   # raises before any change
        if not self.symmetric_requested:
            self.asymmetry = None
        else:
            self.asymmetry = 0.0 if symmetric_known else relative_asymmetry(operator)
        symmetric = self.symmetric_requested and self.asymmetry <= SYMMETRY_TOLERANCE
        kind = self._requested_kind() if symmetric else 'general'
        reuse = (self._library is not None and same_pattern and kind == self.matrix_kind
                 and self._factor_dtype == self._wanted_factor_dtype())
        self.last_double_fallback = False
        attempt_kind, retried = kind, False
        try:
            while True:
                try:
                    self._factorize(attempt_kind, analyse=retried or not reuse,
                                    defer=defer_check and reuse and not retried)
                    break
                except _FactorRejected as rejected:
                    attempt_kind = self._next_attempt(rejected, attempt_kind)
                    if attempt_kind is None:
                        raise rejected.error from None
                    retried = True
        except Exception:
            self.release()
            raise
        self.last_plan_reused = reuse and not retried
        if self.last_plan_reused:
            self.analysis_seconds = 0.0
        self.last_analyses = self.n_analyses - analyses
        self._describe()
        self.n_factorizations += 1
        self.factor_seconds = time.perf_counter() - started
        self.total_factor_seconds += self.factor_seconds
        return self.last_plan_reused

    def _equilibrated(self, host_operator, row_scale, symmetrizer):
        """(host CSR E = R S R in value_dtype on this object's own copy of the canonical pattern, same_pattern) of a
        host scipy operator (_scaled_operator); the caller's operator is never modified."""
        n_rows, n_columns = host_operator.shape
        if n_rows != n_columns:
            raise ValueError(f'HostDirectFactorization needs a square operator, got shape {host_operator.shape}')
        if self.n_unknowns is not None and n_rows != self.n_unknowns:
            raise ValueError(f'HostDirectFactorization.refactorize: {n_rows} unknowns, the factor has {self.n_unknowns}')
        if self.real and np.iscomplexobj(host_operator.data):
            raise ValueError('HostDirectFactorization(real=True) needs a real operator')
        if (symmetrizer is not None) != self.has_symmetrizer:
            raise ValueError('HostDirectFactorization.refactorize: give a symmetrizer exactly when the factor was built '
                             'with one')
        symmetrizer, row_scale = _host_vector(symmetrizer), _host_vector(row_scale)   # a cupy vector: one transfer
        csr = canonical_csr(host_operator, self.value_dtype)   # the caller's operator itself when already canonical
        same_pattern = (self._pattern is not None and np.array_equal(self._pattern[0], csr.indptr)
                        and np.array_equal(self._pattern[1], csr.indices))
        # no identity matching afterwards: a refactorize caller may change its arrays in place
        return self._scaled_operator(csr.data, csr.indptr, csr.indices, same_pattern, row_scale, symmetrizer, None)

    def _equilibrated_values(self, values, indptr, indices, row_scale):
        """(E, same_pattern) of values on a canonical pattern (refactorize_values): matched by identity with the last
        refactorize_values pattern, else compared; no canonicalising copy."""
        values = np.asarray(values).reshape(-1)
        n_rows = int(np.asarray(indptr).size) - 1
        if self.n_unknowns is not None and n_rows != self.n_unknowns:
            raise ValueError(f'HostDirectFactorization.refactorize_values: {n_rows} unknowns, the factor has '
                             f'{self.n_unknowns}')
        if values.size != np.asarray(indices).size:
            raise ValueError(f'HostDirectFactorization.refactorize_values: {values.size} values for '
                             f'{np.asarray(indices).size} stored entries')
        if self.real and np.iscomplexobj(values):
            raise ValueError('HostDirectFactorization(real=True) needs real values')
        source = self._pattern_source
        same_pattern = self._pattern is not None and (
            (source is not None and source[0] is indptr and source[1] is indices)
            or (np.array_equal(self._pattern[0], indptr) and np.array_equal(self._pattern[1], indices)))
        return self._scaled_operator(values, indptr, indices, same_pattern, _host_vector(row_scale), None,
                                     (indptr, indices))

    def _scaled_operator(self, values, indptr, indices, same_pattern, row_scale, symmetrizer, source):
        """(E = R S R on this object's own int32 copy of the pattern, same_pattern): sets _operator /
        _operator_transpose / _row_scale / _symmetrizer, the per-pattern rows and diagonal positions and
        _pattern_source (source: the caller's arrays for identity matching, None: none); ||E|| left lazy."""
        import scipy.sparse
        n_rows = int(np.asarray(indptr).size) - 1
        # own int32 copies of a new pattern (E is built on them: a caller changing its arrays cannot alias E)
        pattern = self._pattern if same_pattern else (np.array(indptr, dtype=np.int32), np.array(indices, dtype=np.int32))
        indptr, indices = pattern
        kept = same_pattern and self._pattern_rows is not None
        rows = self._pattern_rows if kept else csr_rows(indptr)
        diagonal_entries = self._diagonal_entries if kept else np.flatnonzero(indices == rows)
        values = np.asarray(values).reshape(-1)   # the caller's values: scaled_values writes a new array
        if symmetrizer is not None:
            symmetrizer = np.asarray(symmetrizer)
            if self.real and np.iscomplexobj(symmetrizer):
                raise ValueError('HostDirectFactorization(real=True) needs a real symmetrizer')
            symmetrizer = symmetrizer.astype(self.value_dtype)
            if symmetrizer.shape != (n_rows,) or not (np.isfinite(symmetrizer).all() and (symmetrizer != 0).all()):
                raise ValueError(f'HostDirectFactorization symmetrizer must be ({n_rows},), finite and nonzero')
        if row_scale is None:   # from the (symmetrized) diagonal
            diagonal_values = values[diagonal_entries].astype(self.value_dtype)
            if symmetrizer is not None:
                diagonal_values = diagonal_values * symmetrizer[rows[diagonal_entries]]
            diagonal = np.zeros(n_rows)
            diagonal[rows[diagonal_entries]] = np.abs(diagonal_values)
            usable = np.isfinite(diagonal) & (diagonal > 0.0)
            row_scale = 1.0 / np.sqrt(np.where(usable, diagonal, 1.0))
        row_scale = np.asarray(row_scale, dtype=np.float64)
        if row_scale.shape != (n_rows,) or not (np.isfinite(row_scale).all() and (row_scale > 0.0).all()):
            raise ValueError(f'HostDirectFactorization row_scale must be ({n_rows},), finite and > 0')
        values = scaled_values(values, indptr, indices, row_scale, symmetrizer, self.value_dtype)
        operator = scipy.sparse.csr_matrix((values, indices, indptr), shape=(n_rows, n_rows))
        operator.has_canonical_format = True
        if not same_pattern:   # every check passed: the state changes from here on
            self._pattern, self._diagonal_rows = pattern, None
        self._pattern_rows, self._diagonal_entries = rows, diagonal_entries
        self._pattern_source = source
        self.n_unknowns = int(n_rows)
        self._operator = operator
        self._operator_transpose = operator.T   # a CSC view sharing the arrays (transposed residuals)
        self._row_scale, self._symmetrizer = row_scale, symmetrizer
        self._operator_norm = None
        return operator, same_pattern

    def _next_attempt(self, rejected, kind):
        """Kind to refactor after a rejected factor of `kind`: a Cholesky -> LDL^T, then a mixed factor -> the double
        one of the same kind (a mixed positive_definite LDL^T -> the PARDISO double Cholesky; both sticky); None when
        nothing is left (as CudssFactorization._next_attempt, without the remembered rejections)."""
        if kind == 'spd':
            self._cholesky_rejected = True
            return 'symmetric'
        if self.mixed_factor:
            self._mixed_rejected = True
            self.last_double_fallback = True
            self.mixed_rejection = str(rejected)
            return kind if kind == 'general' else self._requested_kind()
        return None

    def _factorize(self, kind, analyse, defer=False):
        """Library factorisation of the current E as `kind` (analyse: a new analysis, the memory guard after it; else
        numeric only on the kept analysis), then the probe (defer: a double factor whose last probe passed unrefined
        leaves it to the next solve); _FactorRejected on a zero pivot or a missed probe."""
        deferred = (defer and not analyse and self._probe_clean
                    and self._wanted_factor_dtype() == self.value_dtype)
        self._double_refined = False
        self._check_pending = False
        self._factor_dtype = self._wanted_factor_dtype()
        matrix = self._library_matrix(kind)
        try:
            if analyse:
                self._free_library()
                started = time.perf_counter()
                self._library = _ADAPTERS[self.back_end](matrix, kind, self.real)
                self.matrix_kind = kind
                self.analysis_seconds = time.perf_counter() - started
                self.total_analysis_seconds += self.analysis_seconds
                self.n_analyses += 1
                self.peak_bytes = self._library.estimated_bytes()
                self._check_memory(self.peak_bytes)
                self._library.factor()
            else:
                self._library.refactor(matrix)
        except HostDirectSingularError as error:
            raise _FactorRejected(error) from None
        self.host_bytes = self._library.factor_bytes()
        self.lu_entries = self._library.lu_entries()
        self.perturbed_pivots = self._library.perturbed_pivots()
        self.threads = self._library.threads()
        if deferred:
            self._check_pending = True   # _checked_solve: the next solve judges this factor
            return
        self._probe_factor()

    def _library_matrix(self, kind):
        """The host CSR handed to the library: E, or its factor-dtype copy (mixed); a PARDISO symmetric kind with an
        explicit zero in every diagonal slot the pattern lacks (MKL needs each diagonal entry stored)."""
        matrix = self._operator
        if self._factor_dtype != self.value_dtype:
            matrix = matrix.astype(self._factor_dtype)
        if self.back_end == 'pardiso' and kind != 'general':
            matrix = self._with_diagonal(matrix)
        return matrix

    def _with_diagonal(self, matrix):
        """matrix with explicit zeros added at the diagonal slots of the analysed pattern that are missing (worked out
        once per pattern, so every refactorisation hands the library the same padded pattern)."""
        if self._diagonal_rows is None:
            indptr, indices = self._pattern
            rows = csr_rows(indptr)
            has_diagonal = np.zeros(self.n_unknowns, dtype=bool)
            has_diagonal[rows[indices == rows]] = True
            self._diagonal_rows = np.flatnonzero(~has_diagonal)
        missing = self._diagonal_rows
        if not missing.size:
            return matrix
        import scipy.sparse
        entries = matrix.tocoo()
        padded = scipy.sparse.csr_matrix(
            (np.concatenate([entries.data, np.zeros(missing.size, dtype=matrix.dtype)]),
             (np.concatenate([entries.row, missing]), np.concatenate([entries.col, missing]))), shape=matrix.shape)
        padded.sum_duplicates()   # canonical; explicit zeros kept
        return padded

    def _check_memory(self, estimate) -> None:
        """HostDirectMemoryError when the library's estimate exceeds HOST_FRACTION of the available RAM (no check when
        either is unknown)."""
        from ..core.host_memory import available_bytes
        available = available_bytes()
        if not estimate or available is None:
            return
        if estimate > HOST_FRACTION * available:
            raise HostDirectMemoryError(
                f'the {self.label} factorisation of {self.n_unknowns:,} unknowns needs an estimated {_gib(estimate)} '
                f'of host memory (its own estimate after the analysis), over {HOST_FRACTION:.0%} of the available RAM '
                f'({_gib(available)}): refused before it could swap the machine; coarsen the mesh or shrink the model')

    def _free_library(self) -> None:
        if self._library is not None:
            self._library.release()
        self._library = None

    def _describe(self) -> None:
        kind_text = _KIND_TEXTS[self.matrix_kind]
        refinement = ('float32' if self.real else 'complex64') + ' + float64 GMRES refinement'
        if self.mixed_factor:
            self.description = f'{self.label} {kind_text} {refinement}'
        elif self._mixed_rejected:
            self.description = f'{self.label} {kind_text} (double: the {refinement} was rejected)'
        elif self._double_refined:
            self.description = f'{self.label} {kind_text} + float64 GMRES refinement'
        else:
            self.description = f'{self.label} {kind_text}'

    # --- probe ---------------------------------------------------------------------------------------------------
    def _probe_vector(self) -> np.ndarray:
        """Probe rhs of E (seeded normal, complex unless real), drawn once per size."""
        if self._probe is None or self._probe.size != self.n_unknowns:
            rng = np.random.default_rng(_PROBE_SEED)
            probe = rng.standard_normal(self.n_unknowns)
            if not self.real:
                probe = probe + 1j * rng.standard_normal(self.n_unknowns)
            self._probe = probe.astype(self.value_dtype)
        return self._probe

    def _probe_accepted(self, residual_tolerance) -> bool:
        """The last probe met residual_tolerance or BACKWARD_ERROR_TOLERANCE (False on NaN)."""
        return bool(self.probe_residual <= residual_tolerance
                    or self.probe_backward_error <= BACKWARD_ERROR_TOLERANCE)

    def _probe_errors(self, rhs, refinement_steps=None):
        """(relative residual ||c - E y|| / ||c||, normwise backward error ||c - E y|| / (||E|| ||y|| + ||c||)) of a
        solve E y = c (probe solve: the stall test on)."""
        solution = self._solve_equilibrated(rhs, refinement_steps=refinement_steps, stall_test=True)
        residual_norm = float(np.linalg.norm(rhs - self._operator @ solution))
        rhs_norm = float(np.linalg.norm(rhs))
        return (residual_norm / rhs_norm,
                residual_norm / (self.operator_norm * float(np.linalg.norm(solution)) + rhs_norm))

    def _probe_factor(self) -> None:
        """Probe of a fresh factor (module header); _FactorRejected when it misses."""
        probe = self._probe_vector()
        self._probe_clean = False
        if self.mixed_factor:
            self.single_residual, _ = self._probe_errors(probe, refinement_steps=0)   # the low-precision factor alone
            self.probe_residual, self.probe_backward_error = self._probe_errors(probe)
            self.refinement_steps = self.last_refinements
            if not self._probe_accepted(MIXED_SOLVE_TOLERANCE):
                raise _FactorRejected(self._mixed_accuracy_error())
            return
        self.single_residual = None
        self.refinement_steps = 0
        if self._solve_accepted(probe, self._solve_equilibrated(probe, stall_test=True)):
            self._probe_clean = True   # a later value refresh may defer its probe (refactorize_values)
            return
        self._double_refined = True   # GMRES-IR on this double factor from now on (every solve)
        solution = self._solve_equilibrated(probe, stall_test=True)
        self.refinement_steps = self.last_refinements
        if not self._solve_accepted(probe, solution):
            raise _FactorRejected(self._double_accuracy_error())

    def _solve_accepted(self, rhs, solution, transpose=False):
        """Probe gate of a double solve of E y = rhs (E^T y with transpose), its errors the probe values: True / False,
        None for a zero rhs (nothing judged). A relative residual within ACCEPT_RESIDUAL decides alone
        (probe_backward_error None, the lazy ||E|| not built; as cuDSS R4)."""
        rhs_norm = float(np.linalg.norm(rhs))
        if rhs_norm == 0.0:
            return None
        operator = self._operator_transpose if transpose else self._operator
        residual_norm = float(np.linalg.norm(rhs - operator @ solution))
        self.probe_residual = residual_norm / rhs_norm
        if self.probe_residual <= ACCEPT_RESIDUAL:
            self.probe_backward_error = None
            return True
        self.probe_backward_error = residual_norm / (self.operator_norm * float(np.linalg.norm(solution)) + rhs_norm)
        return self._probe_accepted(ACCEPT_RESIDUAL)

    def _checked_solve(self, rhs, transpose=False) -> np.ndarray:
        """y of E y = rhs judging a deferred check (_factorize defer): a miss switches this factor to float64 GMRES-IR
        and solves again, a second miss releases it and raises HostDirectAccuracyError; a zero rhs leaves the check
        pending."""
        solution = self._library_solve(rhs, transpose)
        accepted = self._solve_accepted(rhs, solution, transpose)
        if accepted is None:
            return solution
        self._check_pending = False
        if accepted:
            return solution
        self._probe_clean = False
        self._double_refined = True   # GMRES-IR on this double factor from now on (every solve)
        solution = self._solve_equilibrated(rhs, transpose, stall_test=True)
        self.refinement_steps = self.last_refinements
        if not self._solve_accepted(rhs, solution, transpose):
            error = self._double_accuracy_error()
            self.release()
            raise error
        self._describe()
        return solution

    def _double_accuracy_error(self) -> HostDirectAccuracyError:
        return HostDirectAccuracyError(
            f'{self.label} {_KIND_TEXTS[self.matrix_kind]} solves miss the relative residual {ACCEPT_RESIDUAL:g} and '
            f'the backward error {BACKWARD_ERROR_TOLERANCE:g} ({_figure(self.probe_residual)} / '
            f'{_figure(self.probe_backward_error)} after {self.refinement_steps} float64 GMRES iterations; '
            f'{self.perturbed_pivots} perturbed pivots): an ill-conditioned or singular operator; {_TO_SUPERLU}')

    def _mixed_accuracy_error(self) -> HostDirectAccuracyError:
        return HostDirectAccuracyError(
            f'float64 GMRES refinement of the {np.dtype(self._factor_dtype).name} {self.label} '
            f'{_KIND_TEXTS[self.matrix_kind]} factor stalled: residual {_figure(self.probe_residual)}, backward error '
            f'{_figure(self.probe_backward_error)} after {self.refinement_steps} iterations (unrefined '
            f'{_figure(self.single_residual)}): an ill-conditioned operator, e.g. a drive near a high-Q resonance')

    # --- solves --------------------------------------------------------------------------------------------------
    def _library_solve(self, rhs, transpose=False) -> np.ndarray:
        """value_dtype solution of E y = rhs (transpose: E^T y = rhs) by one library solve in the factor dtype."""
        started = time.perf_counter()
        solution = self._library.solve(np.ascontiguousarray(rhs, dtype=self._factor_dtype), transpose)
        self.solve_seconds += time.perf_counter() - started
        self.n_solves += 1
        return np.asarray(solution, dtype=self.value_dtype)

    def _unit_solve(self, rhs, transpose=False) -> np.ndarray:
        """Low-precision (or refined double) solve of E y = rhs with the rhs scaled to unit norm (float32 underflow;
        a zero rhs solves to 0)."""
        norm = float(np.linalg.norm(rhs))
        scale = norm if norm > 0.0 else 1.0
        return self._library_solve(rhs / scale, transpose) * scale

    def _solve_equilibrated(self, rhs, transpose=False, refinement_steps=None, stall_test=False, backward_error=None):
        """value_dtype y of E y = rhs (transpose: E^T, general kinds only): one library solve for an unrefined double
        factor (refinement_steps ignored, as cuDSS double); a mixed factor (or a refined double one) refines by
        gmres_refine (refinement_steps None; target backward_error, None = solve_backward_error; stall_test: a probe,
        target at most BACKWARD_ERROR_TOLERANCE), a mixed one with an int takes that many plain refinement steps; a
        pending deferred check is judged by this solve (_checked_solve)."""
        if self._check_pending:
            return self._checked_solve(rhs, transpose)
        if not (self.mixed_factor or self._double_refined):
            return self._library_solve(rhs, transpose)
        operator = self._operator_transpose if transpose else self._operator

        def single_solve(vector):
            return self._unit_solve(vector, transpose)

        def product(vector):
            return operator @ vector

        solution = single_solve(rhs)
        self.last_refinements = 0
        if refinement_steps is None or not self.mixed_factor:
            target = self.solve_backward_error if backward_error is None else float(backward_error)
            if stall_test:   # a looser caller target must not fail the probe gate
                target = min(target, BACKWARD_ERROR_TOLERANCE)
            solution, self.last_refinements, self._last_solve_errors = gmres_refine(
                np, product, single_solve, rhs, solution, lambda: self.operator_norm, stall_test=stall_test,
                backward_error=target)
        else:
            for _ in range(int(refinement_steps)):
                solution += single_solve(rhs - product(solution))
                self.last_refinements += 1
        self.n_refinements += self.last_refinements
        return solution

    @_openblas_quiet
    def solve_host(self, rhs, transpose=False, refinement_steps=None, backward_error=None):
        """numpy solution x (value_dtype) of A x = rhs (transpose: A^T x = rhs, unconjugated) for a numpy (n,) rhs;
        mixed: refinement_steps / backward_error as CudssFactorization.solve. A real factor solves a complex rhs as two
        real solves (complex128 solution), never a cast."""
        if self._library is None:
            raise RuntimeError('HostDirectFactorization.solve after release()')
        rhs = np.asarray(rhs)
        if rhs.shape != (self.n_unknowns,):
            raise ValueError(f'rhs must be a flat ({self.n_unknowns},) array, got shape {rhs.shape}')
        if self.real and rhs.dtype.kind == 'c':   # a cast would drop the imaginary part
            real_part, imaginary_part = (self.solve_host(np.ascontiguousarray(part), transpose=transpose,
                                                         refinement_steps=refinement_steps,
                                                         backward_error=backward_error)
                                         for part in (rhs.real, rhs.imag))
            return real_part + 1j * imaginary_part
        rhs = rhs.astype(self.value_dtype, copy=False)
        options = {'refinement_steps': refinement_steps, 'backward_error': backward_error}
        scale, symmetrizer = self._row_scale, self._symmetrizer
        transposed = transpose and not self.symmetric   # E^T = E for a symmetric kind
        if symmetrizer is None:   # A = R^-1 E R^-1
            return scale * self._solve_equilibrated(scale * rhs, transposed, **options)
        if not transpose:         # S x = D b, S = D A = R^-1 E R^-1
            return scale * self._solve_equilibrated(scale * (symmetrizer * rhs), **options)
        # A^T = S^T D^-1: x = D S^-T b = D R E^-T R b (E^-1 for a symmetric S)
        return symmetrizer * (scale * self._solve_equilibrated(scale * rhs, transposed, **options))

    def solve(self, rhs, transpose=False, refinement_steps=None, backward_error=None):
        """Solution x of A x = rhs (transpose: A^T x = rhs): numpy in / out with host_arrays or a numpy rhs, else a
        cupy rhs moved to the host and the solution back (one transfer each way)."""
        if self.host_arrays or isinstance(rhs, np.ndarray):
            return self.solve_host(np.asarray(rhs), transpose=transpose, refinement_steps=refinement_steps,
                                   backward_error=backward_error)
        import cupy as cp
        return cp.asarray(self.solve_host(cp.asnumpy(rhs), transpose=transpose, refinement_steps=refinement_steps,
                                          backward_error=backward_error))

    def _lu_solve(self, rhs, trans='N'):
        """numpy x of A x = rhs (trans 'T': A^T): the sparse_direct.Factorization hook of DiffusionDirectSolver and
        the recycled GMRES; an (n, 1) column (scipy LinearOperator) comes back in its shape)."""
        rhs = np.asarray(rhs)
        return self.solve_host(rhs.reshape(-1), transpose=trans == 'T').reshape(rhs.shape)

    @_openblas_quiet
    def solve_preconditioned(self, operator, rhs, initial_guess=None, transpose=False):
        """(x, iterations) of operator x = rhs (transpose: operator^T x = rhs; cupyx CSR and cupy rhs, or scipy / numpy
        with host_arrays) by host GMRES preconditioned with this factor of an earlier, nearby operator; x None above
        sparse_direct.RECYCLE_RTOL (as Factorization.solve_preconditioned)."""
        from .sparse_direct import recycled_gmres
        if self.host_arrays:
            return self.solve_preconditioned_host(operator, rhs, initial_guess, transpose=transpose)
        import cupy as cp
        host_rhs = cp.asnumpy(cp.asarray(rhs, dtype=cp.complex128))
        if host_rhs.shape != (self.n_unknowns,):
            raise ValueError(f'rhs must be a flat ({self.n_unknowns},) array, got shape {host_rhs.shape}')
        host_operator = operator.get().tocsr().astype(np.complex128)
        if transpose:
            host_operator = host_operator.T.tocsr()
        initial = None if initial_guess is None else cp.asnumpy(cp.asarray(initial_guess, dtype=cp.complex128)).ravel()
        solution, iterations = recycled_gmres(self._lu_solve, self.n_unknowns, host_operator, host_rhs, initial,
                                              'T' if transpose else 'N')
        return (None if solution is None else cp.asarray(solution)), iterations

    @_openblas_quiet
    def solve_preconditioned_host(self, matrix, rhs, initial_guess=None, transpose=False):
        """(x, iterations) of matrix x = rhs (transpose: matrix^T x = rhs), host numpy / scipy, by GMRES preconditioned
        with this factor of an earlier, nearby matrix; x None above sparse_direct.RECYCLE_RTOL."""
        from .sparse_direct import recycled_gmres
        rhs = np.asarray(rhs, dtype=np.complex128)
        if rhs.shape != (self.n_unknowns,):
            raise ValueError(f'rhs must be a flat ({self.n_unknowns},) array, got shape {rhs.shape}')
        initial = None if initial_guess is None else np.asarray(initial_guess, dtype=np.complex128).ravel()
        if transpose:
            matrix = matrix.T
        return recycled_gmres(self._lu_solve, self.n_unknowns, matrix.tocsr().astype(np.complex128), rhs, initial,
                              'T' if transpose else 'N')

    def inertia(self):
        """(n_positive, n_negative) pivots of a real LDL^T / Cholesky factor = the inertia of A (E = R A R is a
        congruence; MUMPS INFOG(12) / PARDISO iparm[21] / [22]); None for LU, complex or symmetrizer factors and when
        pivots were perturbed or null (unreliable counts); mixed: the low-precision pivots."""
        if self._library is None or not self.real or self.has_symmetrizer or not self.symmetric:
            return None
        if self.perturbed_pivots:
            return None
        if self.matrix_kind == 'spd':
            return self.n_unknowns, 0
        return self._library.inertia(self.n_unknowns)

    def release(self) -> None:
        """Free the library factor and the operator copies (idempotent)."""
        self._free_library()
        self._operator = self._operator_transpose = self._row_scale = self._symmetrizer = None
        self._check_pending = self._probe_clean = False

    def __del__(self):
        try:
            self.release()
        except Exception:   # interpreter shutdown
            pass


def host_direct_factorization(host_operator, back_end, previous=None, real=False, symmetric=False,
                              positive_definite=False, symmetrizer=None, row_scale=None, symmetric_known=False):
    """HostDirectFactorization of a host scipy operator on a host direct back end (HOST_DIRECT_BACK_ENDS); a valid
    `previous` HostDirectFactorization of the same library, size and options is refactorised in place (analysis reused
    for an unchanged pattern), any other `previous` with a release() is released first (it is being replaced).
    ValueError for another back end or a missing wrapper (missing_text)."""
    if back_end not in HOST_DIRECT_BACK_ENDS:
        raise ValueError(f"host_direct_factorization needs back_end one of {', '.join(HOST_DIRECT_BACK_ENDS)}, got "
                         f'{back_end!r}')
    if not host_direct_available(back_end):
        raise ValueError(missing_text(back_end))
    if (isinstance(previous, HostDirectFactorization) and previous.back_end == library_of(back_end)
            and previous.n_unknowns == host_operator.shape[0]
            and previous.can_refactorize(real, host_direct_precision(back_end), symmetric, positive_definite,
                                         symmetrizer)):
        previous.refactorize(host_operator, row_scale=row_scale, symmetrizer=symmetrizer,
                             symmetric_known=symmetric_known)
        return previous
    release = getattr(previous, 'release', None)
    if release is not None:
        release()
    return HostDirectFactorization(host_operator, back_end, real=real, symmetric=symmetric,
                                   positive_definite=positive_definite, symmetrizer=symmetrizer, row_scale=row_scale,
                                   symmetric_known=symmetric_known)


def host_direct_values_factorization(values, indptr, indices, back_end, previous=None, real=False, symmetric=False,
                                     row_scale=None, symmetric_known=False, defer_check=False, positive_definite=False):
    """HostDirectFactorization of values on a canonical CSR pattern (sparse_direct.HostPencil, Newton tangents): a
    valid `previous` of the same library, size and options refactorises them in place (refactorize_values:
    identity-matched pattern, defer_check), any other `previous` with a release() is released first."""
    if back_end not in HOST_DIRECT_BACK_ENDS:
        raise ValueError(f"host_direct_values_factorization needs back_end one of {', '.join(HOST_DIRECT_BACK_ENDS)}, "
                         f'got {back_end!r}')
    n_rows = int(np.asarray(indptr).size) - 1
    if (isinstance(previous, HostDirectFactorization) and previous.back_end == library_of(back_end)
            and previous.n_unknowns == n_rows
            and previous.can_refactorize(real, host_direct_precision(back_end), symmetric, positive_definite)):
        previous.refactorize_values(values, indptr, indices, row_scale=row_scale, symmetric_known=symmetric_known,
                                    defer_check=defer_check)
        return previous
    release = getattr(previous, 'release', None)
    if release is not None:
        release()
    import scipy.sparse
    matrix = scipy.sparse.csr_matrix((np.asarray(values).reshape(-1), indices, indptr), shape=(n_rows, n_rows))
    matrix.has_canonical_format = True   # the caller's canonical pattern: no sort / duplicate pass
    return HostDirectFactorization(matrix, back_end, real=real, symmetric=symmetric, row_scale=row_scale,
                                   symmetric_known=symmetric_known, positive_definite=positive_definite)


# --- library adapters: every MUMPS / PARDISO call ---------------------------------------------------------------------
# MUMPS: checked against the sources of python-mumps 0.0.6 (src/mumps/mumps.py, _mumps.pyx.in), not run here;
# PARDISO: run against pydiso 0.3.1 (pydiso/mkl_solver.py) on oneMKL 2026.2. Each adapter holds one analysed pattern:
# estimated_bytes() after the analysis, factor(), refactor(matrix) (same pattern), solve(rhs, transpose) in the factor
# dtype, factor_bytes(), lu_entries(), perturbed_pivots(), inertia(n), threads(), release(); library errors are mapped
# to HostDirectSingularError / HostDirectMemoryError (else RuntimeError naming the library code).
# MUMPS (python-mumps): Context.set_matrix(a, symmetric) takes scipy.sparse.triu itself for symmetric=True (SYM=2:
# general symmetric, complex symmetric in c/zmumps, not Hermitian) and creates the s/d/c/z instance from the data dtype
# only when the dtype changes (SYM stays that of its first call: one Context per analysis here); analyze() = JOB 1
# (ordering 'auto'), factor(reuse_analysis=True) = JOB 2 (it retries INFO -8 / -9 itself, doubling ICNTL(14)),
# solve(b) = JOB 3 on a copy of b; mumps_instance.icntl / infog are 1-based; ICNTL(9) != 1 solves A^T x = b
# (transposed, believed unconjugated for complex as the MUMPS guide states; checked by selftest
# solver_host_direct_cpu_gpu); INFOG(12) = negative pivots for SYM != 0; MUMPSError(RuntimeError).error = INFOG(1);
# dropping the instance runs JOB -2 (dealloc). The call releases the GIL.
_MUMPS_SINGULAR = (-10, -6)
_MUMPS_MEMORY = (-5, -7, -8, -9, -13, -17, -19, -20)


def _mumps_megabytes(value) -> int:
    return int(max(float(value), 0.0) * MEGABYTE)


def _mumps_count(value) -> int:
    """MUMPS large counts: negative = -value x 10^6."""
    value = int(value)
    return value if value >= 0 else -value * 1_000_000


def _environment_threads(name):
    try:
        return int(os.environ.get(name, '') or 0) or None
    except ValueError:
        return None


class _MumpsFactor:
    """One python-mumps Context: analysed at construction (one triangle for a symmetric kind)."""

    def __init__(self, matrix, kind, real):
        import mumps
        self._error_type = mumps.MUMPSError
        self.symmetric = kind != 'general'
        self.kind = kind
        self._context = mumps.Context(verbose=False)
        self._call(lambda: self._context.set_matrix(matrix, symmetric=self.symmetric), 'analysis')
        self._call(lambda: self._context.analyze(ordering='auto'), 'analysis')

    def _call(self, function, phase):
        try:
            return function()
        except self._error_type as error:
            raise _library_error('MUMPS', self.kind, phase, getattr(error, 'error', None), str(error),
                                 _MUMPS_SINGULAR, _MUMPS_MEMORY) from None

    def _infog(self, index):
        return self._context.mumps_instance.infog[index]

    def estimated_bytes(self) -> int:
        return _mumps_megabytes(self._infog(17))   # INFOG(17): estimated in-core total, MB

    def factor(self) -> None:
        self._call(lambda: self._context.factor(reuse_analysis=True), 'factorisation')

    def refactor(self, matrix) -> None:
        self._call(lambda: self._context.set_matrix(matrix, symmetric=self.symmetric), 'factorisation')
        self.factor()

    def solve(self, rhs, transpose=False):
        instance = self._context.mumps_instance
        if transpose:
            instance.icntl[9] = 2   # A^T x = b
        try:
            return self._call(lambda: self._context.solve(rhs), 'solve')
        finally:
            if transpose:
                instance.icntl[9] = 1

    def factor_bytes(self) -> int:
        return _mumps_megabytes(self._infog(19))   # INFOG(19): allocated by the factorisation, total MB

    def lu_entries(self) -> int:
        return _mumps_count(self._infog(29))       # INFOG(29): entries in the factors

    def perturbed_pivots(self) -> int:
        return int(self._infog(25)) + int(self._infog(28))   # static-pivoting tiny pivots + null pivots

    def inertia(self, n_unknowns):
        negative = int(self._infog(12))            # SYM != 0: negative pivots
        return n_unknowns - negative, negative

    def threads(self):
        return _environment_threads('OMP_NUM_THREADS')

    def release(self) -> None:
        if self._context is not None:
            self._context.mumps_instance = None    # dealloc: JOB -2
        self._context = None


# PARDISO (pydiso): MKLPardisoSolver(A, matrix_type, factor=False) = phase 11 (A validated: a symmetric type keeps
# scipy.sparse.triu(A)), here with the OpenMP nested dissection (iparm[1] = 3, _PARDISO_ORDERING: the fill of the
# default METIS, iparm[1] = 2, at half the analysis time on 2 - 4 threads and the same time on one, every matrix type
# and both precisions; minimum degree, iparm[1] = 0, is faster still on 2D grids but fills 1.7 x more in 3D; the
# two-level factorisation, iparm[23], and iparm[24] gave nothing on 4 threads); refactor(A) = phase 22 on the kept
# analysis (pydiso checks only the nnz count: the caller compares patterns); solve(b, transpose=True) sets
# iparm[11] = 2 (A^T, not conjugated); the handle sets iparm[27] = 1 for float32 / complex64 data (single precision);
# iparm property = a copy; PardisoError carries only a message ('... step error, <text of the code>'): mapped back by
# the module's _err_messages; dealloc runs phase -1. Matrix types: 11 real / 13 complex unsymmetric, -2 real symmetric
# indefinite, 2 real SPD, 6 complex symmetric.
_PARDISO_TYPES = {('general', True): 11, ('general', False): 13, ('symmetric', True): -2, ('symmetric', False): 6,
                  ('spd', True): 2}
_PARDISO_ORDERING = 3   # iparm[1]: OpenMP nested dissection (pydiso FillReducingOrdering.PARALLEL_NESTED_DISSECTION)
# constructor keywords by pydiso generation, newest first (a TypeError: the next; the last = the library defaults)
_PARDISO_OPTIONS = ({'fill_reducing_ordering': _PARDISO_ORDERING, 'report_nnz': True}, {'report_nnz': True}, {})
_PARDISO_SINGULAR = (-4, -7)
_PARDISO_MEMORY = (-2, -9)


class _PardisoFactor:
    """One pydiso MKLPardisoSolver: analysed at construction (phase 11), factorised by factor() (phase 22)."""

    def __init__(self, matrix, kind, real):
        from pydiso import mkl_solver
        self._module = mkl_solver
        self.kind = kind
        self._matrix = matrix
        matrix_type = _PARDISO_TYPES[(kind, bool(real))]

        def create():
            for options in _PARDISO_OPTIONS:
                try:
                    return mkl_solver.MKLPardisoSolver(matrix, matrix_type=matrix_type, factor=False, verbose=False,
                                                       **options)
                except TypeError:   # an older pydiso without these keywords
                    if not options:
                        raise
        self._solver = self._call(create, 'analysis')
        self._direct, self._value_take = self._direct_values(matrix)

    def _direct_values(self, matrix):
        """(direct, take): direct True when pydiso's private value / index arrays hold this analysed pattern as kept
        here (a symmetric type: the upper triangle in CSR order; take = its positions in matrix.data, None = all),
        checked once per analysis; factor() then writes values straight into the handle. (False, None) otherwise."""
        solver = self._solver
        if not all(hasattr(solver, name) for name in ('_data', '_indptr', '_indices', '_data_dtype', '_factor')):
            return False, None
        take, indptr, indices = None, matrix.indptr, matrix.indices
        if self.kind != 'general':   # pydiso keeps sp.triu(A) of a symmetric type
            rows = csr_rows(matrix.indptr)
            take = np.flatnonzero(matrix.indices >= rows)
            indices = matrix.indices[take]
            indptr = np.concatenate([[0], np.cumsum(np.bincount(rows[take], minlength=matrix.shape[0]))])
        try:
            same = (np.asarray(solver._data).size == indices.size and np.array_equal(solver._indptr, indptr)
                    and np.array_equal(solver._indices, indices))
        except Exception:   # an unexpected pydiso layout: the public refactor
            same = False
        return (True, take) if same else (False, None)

    def _call(self, function, phase):
        try:
            return function()
        except self._module.PardisoError as error:
            text = str(error)
            messages = getattr(self._module, '_err_messages', {}) or {}
            matches = [code for code, message in messages.items() if code and message and message in text]
            code = max(matches, key=lambda found: len(messages[found])) if matches else None
            raise _library_error('PARDISO', self.kind, phase, code, text, _PARDISO_SINGULAR,
                                 _PARDISO_MEMORY) from None

    def _iparm(self):
        return self._solver.iparm   # a copy of the 64 parameters (C order: iparm[k] = MKL iparm(k + 1))

    def estimated_bytes(self) -> int:
        iparm = self._iparm()
        return 1024 * int(max(iparm[14], iparm[15] + iparm[16]))   # KiB: peak of analysis vs permanent + factor

    def factor(self) -> None:
        """Phase 22 of the kept matrix: its values written into the handle (direct) or pydiso's refactor."""
        if self._direct:
            data = self._matrix.data if self._value_take is None else self._matrix.data[self._value_take]
            solver = self._solver
            solver._data = np.require(data, solver._data_dtype, requirements='C')
            self._call(solver._factor, 'factorisation')
            return
        self._call(lambda: self._solver.refactor(self._matrix), 'factorisation')

    def refactor(self, matrix) -> None:
        """Phase 22 of a matrix on the analysed (padded) pattern (the caller compared patterns)."""
        self._matrix = matrix
        self.factor()

    def solve(self, rhs, transpose=False):
        return self._call(lambda: self._solver.solve(rhs, transpose=bool(transpose)), 'solve')

    def factor_bytes(self) -> int:
        iparm = self._iparm()
        return 1024 * int(iparm[15] + iparm[16])

    def lu_entries(self) -> int:
        return int(self._iparm()[17])

    def perturbed_pivots(self) -> int:
        return int(self._iparm()[13])

    def inertia(self, n_unknowns):
        iparm = self._iparm()
        return int(iparm[21]), int(iparm[22])

    def threads(self):
        try:
            return int(self._module.get_mkl_pardiso_max_threads())
        except Exception:
            return None

    def release(self) -> None:
        self._solver = self._matrix = None   # dealloc: phase -1


_ADAPTERS = {'mumps': _MumpsFactor, 'pardiso': _PardisoFactor}


def _library_error(label, kind, phase, code, text, singular_codes, memory_codes):
    """The open_sol exception of a library error code (zero pivot: HostDirectSingularError; memory:
    HostDirectMemoryError; else RuntimeError)."""
    what = f'the {label} {_KIND_TEXTS.get(kind, kind)} {phase} failed (error {code}: {text})'
    if code in singular_codes:
        return HostDirectSingularError(f'{what}: a zero pivot of a singular operator')
    if code in memory_codes:
        return HostDirectMemoryError(f'{what}: not enough memory; coarsen the mesh or shrink the model')
    return RuntimeError(what)
