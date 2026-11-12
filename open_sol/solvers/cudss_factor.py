"""cuDSS sparse LU / LDL^T / Cholesky on the device (nvmath-python DirectSolver): the 'cudss' / 'cudss_mixed'
factorizations of arpack_shift_invert and back ends of sparse_direct, a sparse_direct.Factorization stand-in. numpy only
at module level."""
# The factor and cuDSS's workspace live outside the cupy pool (cudaMalloc), like the cusolver_qr workspace: the guard is
# cuDSS's own peak estimate after planning against the free device memory (DEVICE_FRACTION). cuDSS pivots within
# supernodes and perturbs tiny pivots (pivot_eps) instead of SuperLU's threshold partial pivoting, so a probe solve
# checks the residual and turns on cuDSS iterative refinement when it misses ACCEPT_RESIDUAL. A random probe's float64
# residual floors near eps kappa(E) ||y|| / ||c|| (hirsch2025 E4: 100 um x 100 nm string on 25 nm P2 tetrahedra,
# 1.8e-9 after any number of steps), so a solve also passes when its normwise backward error ||c - E y|| / (||E|| ||y||
# + ||c||) reaches BACKWARD_ERROR_TOLERANCE (||E|| = sqrt(||E||_1 ||E||_inf), a 2-norm bound). Reordering: cuDSS's own
# nested dissection (a caller's elimination ordering is not used). cuDSS is not thread-safe: GPU-owning thread only.
# symmetric=True: an LDL^T of the lower triangle (complex symmetric A = A^T, not Hermitian), about half the factor
# entries and factorisation work of the LU, once relative_asymmetry passes SYMMETRY_TOLERANCE; above it the general LU
# (self.symmetric False, self.asymmetry says why). A solve reads L twice, so it costs about what the LU solve does.
# positive_definite=True (real only): a Cholesky (matrix type SPD) of the lower triangle; a failed Cholesky (zero status
# or probe) re-plans once as LDL^T (logged). Double factors only: a mixed SPD request plans the float32 LDL^T (the
# float32 Cholesky numeric factorisation of the hirsch2025 E5 static matrix hung for > 175 s; its float32 LDL^T was
# rejected by the probe cleanly), and its double fallback is the double Cholesky (_requested_kind). symmetrizer=D:
# factors S = D A (D diagonal, D A symmetric), solves A x = b as S x = D b and A^T x = b as x = D S^-1 b; with a plain
# symmetric factor A^T = A. Without a symmetric factor the transposed solve is refused (cuDSS exposes no LU^T solve
# here).
# Both precisions factor E = R S R, R = |diag S|^-1/2 (keeps complex symmetry; x = R y): a diagonal scaling such as the
# FDFD symmetrizer (cell areas ~1e-16 m^2) left the pinned identity rows of mirror / wall cells ~1e-16 x the others,
# below cuDSS's pivot epsilon, and the perturbed pivots gave a probe residual 0.13 (2D nanobeam quarter model).
# precision='mixed': E factorised in complex64 / float32, solves corrected by float64 residuals of E. Without the
# equilibration the CPML stretches spread the Mie eighth sphere's entries over 4e5 and the complex64 factor was useless
# (error 0.38, refinement diverging); with it 9e-5 unrefined and ~3e3 x per refinement step (220,500 unknowns:
# factorise 1.1 s vs 13.9 s, solve 18 ms vs 32 ms, half the memory). Plain refinement contracts by ~eps32 kappa(E) per
# step, and near a high-Q resonance one eigenvalue of E is ~1/Q of the rest (mauranyapin2021 E7, closed membrane, loss
# factor 1/5000: 2.3 x per step on resonance, 1e-8 after 12 steps): a full-accuracy solve (refinement_steps None)
# therefore refines by restarted flexible GMRES on E preconditioned by the low-precision solve (GMRES-IR), which
# removes such an outlier in about one extra iteration; a fixed step count (the eigen polish) keeps plain refinement.
# The solve stops at MIXED_SOLVE_BACKWARD_ERROR (1e-15, about a double solve), not at the probe gate's 1e-12: on a
# resonance ||E|| ||y|| >> ||c||, so a 1e-12 backward error left relative residuals of 1e-7 - 1e-5, and E7's adaptive
# reduced sweep (rom_tolerance 1e-9 on the true residual) took a quarter more factorisations and stopped short. Where
# float64 floors above it, FGMRES's estimate meets the target while the true residual does not: the solve stops there.
# The target is per factor (solve_backward_error) and per call (solve(backward_error=)): a direct sweep may loosen it
# (E7: the 1e-15 target costs 6 iterations, about half the mixed solve, while mixed and double series agree to 1e-9;
# decision D2, default unchanged); probes and deferred checks never refine to less than BACKWARD_ERROR_TOLERANCE. The
# GMRES keeps basis and directions as (cycle, n) device arrays: classical Gram-Schmidt twice by batched products, one
# host read per iteration (coefficients + norm; E7 measured ~9 ms of Python / syncs per 48 ms solve with one .item()
# per coefficient), the single solve's scaling norm stays on the device (B8). The loop is the module function
# gmres_refine (array module xp): cupy here, numpy for the mixed / refined host factors of solvers.host_direct.
# A mixed factor whose probe still misses is replaced by the double factor of the same kind (last_double_fallback,
# mixed_rejection text) and the object factors in double from then on (a sweep would otherwise re-plan per shift).
# Stall test (H4): a probe (and a deferred check) stops its GMRES after MIXED_STALL_ITERATIONS when the rate so far,
# extrapolated to MIXED_MAX_REFINEMENTS, misses the probe gate by over MIXED_STALL_MARGIN (hirsch2025 E5 static, real
# SPD, 1.46e6 unknowns: unrefined 0.19, ~1.02 x per iteration, 16 iterations 0.52 s before). Not a threshold on the
# unrefined residual alone: on a high-Q resonance one eigenvalue leaves it near 0.4 and GMRES still reaches the gate.
# Rejected mixed factors: while remember_cudss_orderings is on (same scope as the orderings), a rejection is kept per
# (kind, real, pattern); a later plan of that key by another factor object goes to double at once (no low-precision
# plan and probe; mixed_rejection says so, not counted as a double fallback). The values may differ (another sweep
# value or Newton step): a mesh whose float32 factor failed once is taken to fail again (sliver / thin-film meshes).
# Deferred check (refactorize_device_values(defer_check=True): sweeps of sparse_direct.factorize_pencil): a mixed
# refactorisation on a plan whose earlier factor passed its probe skips the probe (E7: 63 of 86 ms per frequency, the
# GMRES of a separate solve); the first full-accuracy solve is the check, judged by the same gate on its true
# equilibrated residual (probe_residual / single_residual / refinement_steps then come from it; until then they are
# the previous factor's). A miss re-plans the double factor in place from the device E (no host copy;
# fallback_listener counts it), probes it and solves again, so a CudssAccuracyError may then come from that solve.
# A double refactorisation defers the same way (R3: E7 probe 7.1 ms vs check 1.3 ms of an 88.6 ms refactorisation):
# its first solve (any refinement_steps, transposed too) is judged by ACCEPT_RESIDUAL / BACKWARD_ERROR_TOLERANCE on
# its equilibrated residual (one product); a miss turns on cuDSS refinement (REFINEMENT_STEPS) and solves again, and a
# second miss releases the factor and raises CudssAccuracyError from that solve (no fallback left). ir_num_steps of
# the plan carries over, so a pencil whose earlier factor needed refinement refines from the start. A factor that is
# never solved (inertia counts only) is never judged.
# A caller's row_scale replaces |diag S|^-1/2 where
# that diagonal may nearly cancel (FEM curl-curl minus k^2 mass on coarse degree-2 tetrahedra: physics/em3d/fem3d
# passes (|diag K| + k_t^2 |diag M|)^-1/2).
# Plan reuse (refactorize): a new operator of the same sparsity pattern (after duplicate summing; explicit zeros kept)
# and matrix type is written into the planned device CSR values in place and only factorised again: no host
# reordering / symbolic phase. Never DirectSolver.reset_operands(a=) for this: with device operands nvmath 1.0 drops the
# plan there (solver_planned = False, "plan() not called" at factorize). A changed pattern or
# matrix type re-plans (and re-checks the device memory). Equilibration and the symmetrizer act on the CSR values
# entry by entry (no scipy products, which drop entries that cancel and would change the pattern).
# Ordering reuse (a re-plan of a pattern planned before): after plan() the row permutation and the nested-dissection
# partition tree are kept per (ordering family, n, nnz, pattern digest); a later plan of that key hands both back
# (cudss data_set USER_PERM + USER_ND_PARTITION_TREE) and cuDSS skips its host reordering (romero2019 E2(a), 591k
# unknowns LDL^T: plan
# 0.26 s vs 4.8 s, identical lu_nnz, peak memory and factorize / solve times). The permutation alone (or a pre-permuted
# matrix with ReorderingAlg.NONE) loses the tree: same fill, but a 10x slower factorisation, so the pair is set together
# or not at all (a failed set re-creates the solver). The object reuses its own ordering (re-plans of the same pattern:
# the double fallback, refactorize after a pattern round trip); across objects only while remember_cudss_orderings is
# on (model.study.run_study around every study, the GUI worker for its lifetime; off elsewhere so a selftest case never
# plans with another case's ordering). Families (ORDERING_FAMILIES): Cholesky and LDL^T share one, real and complex
# alike (hirsch2025 E5, 1.46e6 unknowns, every pair of real Cholesky / real LDL^T / complex LDL^T: plan 1.05 s vs 4.5
# s, same lu_nnz, numeric time and residual; .temp_scratchpad/cudss_ordering_kinds_r1a.py), so the E5 eigen pencil
# plans on the static Cholesky's ordering and a mixed -> double or Cholesky -> LDL^T fallback on its own. LU keeps
# its own family: a symmetric ordering served the LU of a symmetric matrix, but an unsymmetric one (drift_diffusion
# Jacobians) is not measured. nvmath exposes no setter, so the raw bindings take DirectSolver.handle / data_ptr: any
# nvmath / cuDSS error plans as before.
# refactorize_device_values (plan G5, H1, R7): values on the device on a canonical host pattern: R from |diag| (or the
# caller's row_scale) and E = R A R by device index maps (rows, columns, indptr, diagonal slots, lower take and the
# lower triangle's indices / indptr: ~20 B per entry, built once per plan), no host CSR. On the planned pattern and kind
# the values are refreshed in place; else (first call, new pattern or kind) the plan is made from those device arrays
# (from_device_values: a new factor). Two entry points: "symmetric known" (the caller vouches for symmetry when it
# asks for it: DevicePencil sweeps, device Newton tangents) and "check first" (check_symmetry: max |E - E^T| / max |E|
# on the device by a transpose map, searchsorted of the mirrored keys, built once per plan). The second serves
# refactorize itself (H1): a host operator on the planned pattern (no symmetrizer) only uploads its values (hirsch2025
# E5: 3.2 s host route vs 0.77 s per static refactorisation, 5.2 vs 1.6 s per eigen one), and
# sparse_direct.factorize_device_csr (DiffusionDirectSolver: no .get() round trip, the pattern matched on the device).
# refactorize_device_lower (R4): a symmetric DevicePencil stores only the lower triangle; its combined lower values go
# through one ElementwiseKernel (_scaled_lower_kernel: E_l = a_l r_i r_j and the factor-dtype copy in one pass) into
# the planned buffers (E7: refresh 4.9 -> 2.8 ms, pencil terms 53 -> 29 MB). The operator norm of a device refresh is
# lazy (operator_norm: one pass over the stored E, a symmetric one from its lower triangle) and a double deferred check
# whose relative residual passes skips the backward error (probe_backward_error None), so it is built only when a
# mixed GMRES, a probe or a missed check reads it.
# Fallback re-plans (_replan_in_place) keep the device storage: only the kind or the factor dtype changes. The host
# route (_equilibrated, relative_asymmetry, _lower_triangle, _plan) remains for first plans and new patterns of host
# operators and for symmetrizer factors.
# Device copies (cupy pool): a symmetric factor keeps only the lower triangle L of E in value_dtype (residuals and mixed
# refinement apply E y = L y + L^T y - diag(L) y; double: it is also the matrix cuDSS reads) plus, mixed, its
# factor_dtype copy; a general factor keeps the full E (+ the factor_dtype copy when mixed). operator_copy_bytes
# estimates them for the memory estimators (+ mixed: the GMRES vectors of one full-accuracy solve).
# Errors: CudssSingularError (zero-pivot status) and CudssAccuracyError (probe missed: an ill-conditioned operator,
# e.g. a drive on a nearly lossless resonance), both RuntimeError, so callers can word the cause.
import collections
import functools
import hashlib
import importlib.util
import logging
import math
import threading
import time

import numpy as np

DEVICE_FRACTION = 0.8          # largest share of the free device memory cuDSS's peak estimate may claim
ACCEPT_RESIDUAL = 1e-10        # probe-solve relative residual ||b - A x|| / ||b|| accepted without refinement
REFINEMENT_STEPS = 2           # cuDSS iterative refinement steps per solve once the probe misses ACCEPT_RESIDUAL
SYMMETRY_TOLERANCE = 1e-12     # max |A - A^T| / max |A| below which symmetric=True factors the lower triangle
PRECISIONS = ('double', 'mixed')
BACKWARD_ERROR_TOLERANCE = 1e-12   # probe gates: normwise backward error accepted in place of ACCEPT / MIXED_SOLVE_TOLERANCE
MIXED_SOLVE_TOLERANCE = 1e-12  # mixed: a solve without a step count refines until ||c - E y|| / ||c|| reaches this
MIXED_SOLVE_BACKWARD_ERROR = 1e-15   # mixed: a full-accuracy solve refines to this backward error (about a double
                                     # cuDSS solve's) unless the relative residual met MIXED_SOLVE_TOLERANCE first
MIXED_MAX_REFINEMENTS = 16     # mixed: GMRES iterations at most per full-accuracy solve (a contraction of ~3e3 needs
                               # 3-5; the 600 nm Si L3 eighth at 1600-1660 nm, plain refinement: unrefined 2-3e-3, ~15 x
                               # per step, 9 steps to 1e-12)
GMRES_RESTART = 8              # mixed: GMRES cycle length (basis 9 + directions 8 solve-size vectors, cupy pool, per solve)
ORDERING_FAMILIES = {'symmetric': 'symmetric', 'spd': 'symmetric', 'general': 'general'}   # matrix kind -> the
                               # family whose kept orderings it shares (module header "Ordering reuse")
KEPT_ORDERINGS = 8             # orderings remember_cudss_orderings keeps (oldest dropped; ~4 B per unknown each)
                               # (also the number of rejected mixed patterns kept)
MIXED_STALL_ITERATIONS = 4     # mixed probe / deferred check: GMRES iterations before the stall test (module header)
MIXED_STALL_MARGIN = 100.0     # stalled when the rate so far, extrapolated to MIXED_MAX_REFINEMENTS, misses the gate
                               # by more than this factor
_PROBE_SEED = 11
_KIND_TEXTS = {'general': 'LU', 'symmetric': 'LDL^T', 'spd': 'Cholesky'}
_LOGGER = logging.getLogger('open_sol.cudss')
_LOGGER.setLevel(logging.ERROR)   # nvmath logs every phase at INFO and the unused threading layer at WARNING
_TO_SUPERLU = ("use host SuperLU instead (eigen solver factorization 'superlu', sparse_direct back_end 'scipy_splu'; "
               "'auto' is cuDSS on the gpu backend for the roles listing auto_back_ends, describe "
               "'auto_back_end_by_grid', and falls back to SuperLU by itself)")


class CudssSingularError(RuntimeError):
    """A cuDSS factorisation that failed with a zero pivot (singular operator)."""


class CudssAccuracyError(RuntimeError):
    """cuDSS solves that miss the probe accuracy (an ill-conditioned operator)."""


class CudssMemoryError(ValueError):
    """A cuDSS plan refused before it could overrun the device (its peak estimate over DEVICE_FRACTION of the free
    memory)."""


CUDSS_REFUSALS = (CudssSingularError, CudssAccuracyError, CudssMemoryError)   # what an 'auto' back end falls back on


@functools.lru_cache(maxsize=1)
def cudss_available() -> bool:
    """True when nvmath-python is installed (find_spec once per process, nothing imported: GUI-thread safe)."""
    return importlib.util.find_spec('nvmath') is not None


_library_thread = None   # warm_cudss_library's host thread; _direct_solver joins it before the first cuDSS call


def warm_cudss_library() -> None:
    """Import nvmath's DirectSolver and load the cuDSS library on a daemon host thread, once per process (speedup B3);
    no CUDA call there: handles and plans stay on the GPU-owning thread, which joins it first."""
    global _library_thread
    if _library_thread is not None or not cudss_available():
        return
    _library_thread = threading.Thread(target=_load_cudss_library, name='open_sol cuDSS load', daemon=True)
    _library_thread.start()


def _load_cudss_library():
    """Host-only part of the first cuDSS use: the nvmath import and the library load (cudssGetProperty)."""
    try:
        import nvmath.sparse.advanced  # noqa: F401
        from nvmath.bindings import cudss
        cudss.get_property(0)   # libraryPropertyType MAJOR_VERSION: loads cuDSS and its dependencies
    except Exception:   # the first plan on the GPU-owning thread reports a broken install
        pass


def _join_library_thread():
    """Wait for warm_cudss_library's thread (cuDSS is not thread-safe: nothing of it runs while the GPU thread plans)."""
    if _library_thread is not None:
        _library_thread.join()


def _thread_size_buffers():
    """nvmath fills its cuDSS getter buffer (`_tls.size_written`) only on the thread that imported it (the load thread):
    give the calling thread its own, else every plan / info read raises AttributeError."""
    from nvmath.sparse._internal import cudss_config_ifc, cudss_data_ifc
    for module in (cudss_config_ifc, cudss_data_ifc):
        if not hasattr(module._tls, 'size_written'):
            module._tls.size_written = np.empty((1,), dtype=np.uint64)


_kept_orderings = None   # ordering key -> (row permutation, partition tree) while remember_cudss_orderings is on
_rejected_mixed = None   # (kind, real, n, nnz, digest) -> rejection text of a mixed factor, same lifetime (H4)


def remember_cudss_orderings(enabled=True) -> None:
    """Keep the orderings of the last KEPT_ORDERINGS planned patterns for re-plans by other factor objects (module
    header "Ordering reuse") and the patterns whose mixed factor was rejected ("Rejected mixed factors"); off forgets
    both."""
    global _kept_orderings, _rejected_mixed
    _kept_orderings = collections.OrderedDict() if enabled else None
    _rejected_mixed = collections.OrderedDict() if enabled else None


def remembers_cudss_orderings() -> bool:
    """True while remember_cudss_orderings is on."""
    return _kept_orderings is not None


def cudss_orderings_kept():
    """Context manager: remember_cudss_orderings inside the block, the previous state restored after (run_study)."""
    return _KeptOrderings()


class _KeptOrderings:
    def __enter__(self):
        self._was_on = remembers_cudss_orderings()
        if not self._was_on:
            remember_cudss_orderings()
        return self

    def __exit__(self, *exc_info):
        if not self._was_on:   # forgets the kept orderings (host memory)
            remember_cudss_orderings(False)
        return False


def _read_ordering(solver):
    """(row permutation, nested-dissection partition tree) of a planned DirectSolver as int32 host copies; None when
    cuDSS / nvmath do not give both."""
    try:
        from nvmath.bindings import cudss
        permutation = np.array(solver.plan_info.row_permutation, dtype=np.int32, copy=True)
        size = np.zeros(1, dtype=np.uint64)   # the tree's byte size first (a wrongly sized buffer: INVALID_VALUE)
        cudss.data_get(solver.handle, solver.data_ptr, cudss.DataParam.ND_PARTITION_TREE, 0, 0, size.ctypes.data)
        tree = np.zeros(int(size[0]) // np.dtype(np.int32).itemsize, dtype=np.int32)
        if not tree.size:
            return None
        cudss.data_get(solver.handle, solver.data_ptr, cudss.DataParam.ND_PARTITION_TREE, tree.ctypes.data,
                       int(tree.nbytes), size.ctypes.data)
        return permutation, tree
    except Exception as error:   # an nvmath / cuDSS without these: no reuse
        _LOGGER.info('cuDSS ordering not kept (%s)', error)
        return None


def _set_ordering(solver, ordering) -> bool:
    """Hand a kept (permutation, tree) to an unplanned DirectSolver (host arrays: the caller keeps them alive until the
    solver is freed); False when cuDSS refused either."""
    try:
        from nvmath.bindings import cudss
        permutation, tree = ordering
        for parameter, array in ((cudss.DataParam.USER_PERM, permutation),
                                 (cudss.DataParam.USER_ND_PARTITION_TREE, tree)):
            cudss.data_set(solver.handle, solver.data_ptr, parameter, array.ctypes.data, int(array.nbytes))
        return True
    except Exception as error:
        _LOGGER.info('cuDSS ordering not reused (%s)', error)
        return False


def _pattern_key(indptr, indices):
    """(n, nnz, digest) of a host CSR pattern (the arrays' own buffers hashed, no copies: ~30 ms per 26 MB)."""
    digest = hashlib.blake2b(digest_size=16)
    for array in (indptr, indices):
        array = np.ascontiguousarray(array)
        digest.update(array.dtype.str.encode())
        digest.update(array)
    return int(indptr.size) - 1, int(indices.size), digest.hexdigest()


_SCALED_LOWER_KERNELS = {}   # single_copy -> ElementwiseKernel (built on first use)


def _scaled_lower_kernel(single_copy):
    """ElementwiseKernel E_l = a_l r[row_l] r[column_l] over the lower-triangle entries (R4), with single_copy also the
    factor-dtype copy written in the same pass (float32 / complex64 of a mixed factor)."""
    if single_copy not in _SCALED_LOWER_KERNELS:
        import cupy as cp
        inputs = 'T value, int32 row, int32 column, raw float64 scale'
        operation = 'scaled = value * (scale[row] * scale[column]);'
        if single_copy:
            _SCALED_LOWER_KERNELS[True] = cp.ElementwiseKernel(inputs, 'T scaled, F single',
                                                               operation + ' single = (F)scaled;',
                                                               'open_sol_cudss_scaled_lower_single')
        else:
            _SCALED_LOWER_KERNELS[False] = cp.ElementwiseKernel(inputs, 'T scaled', operation,
                                                                'open_sol_cudss_scaled_lower')
    return _SCALED_LOWER_KERNELS[single_copy]


_SINGLE_SOLVE_KERNELS = {}   # name -> ElementwiseKernel of _single_solve (built on first use)


def _unit_scaled_kernel():
    """ElementwiseKernel single = (F)(rhs / ||rhs||) from the device ||rhs||^2 (raw (1,) float64; 0 divides by 1)."""
    if 'unit' not in _SINGLE_SOLVE_KERNELS:
        import cupy as cp
        _SINGLE_SOLVE_KERNELS['unit'] = cp.ElementwiseKernel(
            'T rhs, raw float64 norm_squared', 'F single',
            'const double squared = norm_squared[0]; single = (F)(rhs * (squared > 0.0 ? rsqrt(squared) : 1.0));',
            'open_sol_cudss_unit_scaled')
    return _SINGLE_SOLVE_KERNELS['unit']


def _rescaled_kernel():
    """ElementwiseKernel solution = (T)single * ||rhs|| (the inverse of _unit_scaled_kernel's scaling)."""
    if 'rescaled' not in _SINGLE_SOLVE_KERNELS:
        import cupy as cp
        _SINGLE_SOLVE_KERNELS['rescaled'] = cp.ElementwiseKernel(
            'F single, raw float64 norm_squared', 'T solution',
            'const double squared = norm_squared[0]; solution = ((T)single) * (squared > 0.0 ? sqrt(squared) : 1.0);',
            'open_sol_cudss_rescaled')
    return _SINGLE_SOLVE_KERNELS['rescaled']


CUDSS_SOLVE_PHASE = 'cuDSS solve'   # transient phase of one solve or probe: solve_peak_bytes
CUDSS_DOUBLE_SOLVE_VECTORS = 4      # rhs copy, equilibrated rhs, cuDSS's solution, the rescaled one returned
CUDSS_MIXED_SOLVE_VECTORS = 2 * GMRES_RESTART + 10   # + the GMRES basis and directions, residual, product temporaries


def operator_copy_bytes(n_unknowns, entries, value_bytes=16, symmetric=True, mixed=False) -> int:
    """Pool bytes of the CSR copies a CudssFactorization keeps per plan (_plan: value dtype, the lower triangle +
    diagonal when symmetric; mixed: + the half-size factor copy) of `entries` stored entries (int32 indices); the rhs
    buffer and row scale: solve_buffer_bytes, a solve's vectors: solve_peak_bytes."""
    n_unknowns, entries = int(n_unknowns), int(entries)
    stored = (entries + n_unknowns) // 2 if symmetric else entries
    copies = ((value_bytes, value_bytes // 2) if mixed else (value_bytes,))
    diagonal = value_bytes * n_unknowns if symmetric else 0
    return sum((copy_bytes + 4) * stored + 4 * (n_unknowns + 1) for copy_bytes in copies) + diagonal


def solve_buffer_bytes(n_unknowns, value_bytes=16, mixed=False) -> int:
    """Pool bytes a plan keeps beside its CSR copies: the factor-dtype rhs buffer cuDSS reads and the float64 row scale."""
    return ((int(value_bytes) // 2 if mixed else int(value_bytes)) + 8) * int(n_unknowns)


def solve_peak_bytes(n_unknowns, value_bytes=16, mixed=False) -> int:
    """Transient pool bytes of one solve or probe (CUDSS_SOLVE_PHASE): CUDSS_DOUBLE_SOLVE_VECTORS, mixed
    CUDSS_MIXED_SOLVE_VECTORS (GMRES-IR) value-dtype vectors, the returned solution among them."""
    vectors = CUDSS_MIXED_SOLVE_VECTORS if mixed else CUDSS_DOUBLE_SOLVE_VECTORS
    return vectors * int(value_bytes) * int(n_unknowns)


def index_map_bytes(n_unknowns, entries, symmetry_check=True) -> int:
    """Pool bytes of the index maps a device refactorisation keeps per plan (rows, columns, lower take, its rows /
    indices, diagonal slots; symmetry_check: the transpose map): a run item once the first refresh made them."""
    n_unknowns, entries = int(n_unknowns), int(entries)
    lower = (entries + n_unknowns) // 2
    return 8 * entries + 12 * lower + 29 * n_unknowns + (4 * entries if symmetry_check else 0)


def refresh_peak_bytes(entries, value_bytes=16, symmetry_check=True, n_unknowns=0, mixed=False) -> int:
    """Transient pool bytes of one refresh on the planned pattern (uploaded and equilibrated values, the float64
    row-scale gathers; symmetry_check: the int64 keys of the transpose map), freed when it returns; n_unknowns: or the
    uploaded and equilibrated values held through the factorisation's probe solve (solve_peak_bytes), if larger."""
    value_bytes, entries = int(value_bytes), int(entries)
    refresh = (2 * value_bytes + 32 if symmetry_check else max(2 * value_bytes + 8, value_bytes + 24)) * entries
    probe = 2 * value_bytes * entries + solve_peak_bytes(n_unknowns, value_bytes, mixed) if n_unknowns else 0
    return max(refresh, probe)


def refresh_bytes(n_unknowns, entries, value_bytes=16, symmetry_check=True) -> int:
    """index_map_bytes + refresh_peak_bytes: what a refactorize on the planned pattern (H1: refactorize_device_values,
    e.g. the next frequency, order or Newton step) adds to operator_copy_bytes; estimators book the two parts apart."""
    return (index_map_bytes(n_unknowns, entries, symmetry_check)
            + refresh_peak_bytes(entries, value_bytes, symmetry_check))


# cuDSS factor size rule (outside the cupy pool: permanent_device_memory of the plan), fitted on one RTX 3050 to the
# selftest counters cudss_factor_device_bytes / eigen_factor_device_bytes (2026-11): 2D complex symmetric 5-point FDFD
# (900 .. 115,600 unknowns) ~ 54 B x n log2 n, 3D em_fdfd_3d curl-curl (13 per row, 1,440 .. 73,167 unknowns) ~ 95 B x
# n^(4/3), the 2D P2 heat triangles (~19 per row, real) within 10 % with the square-root row scaling; BOR (6.3 / 8.9
# per row) +3 / -23 % (2026-11 profile). Open, to settle from cudss_lu_entries (summary.json solvers counters, the
# vram profiler's plan table): the fit is in bytes, so it books real values at half the complex bytes although the
# index part does not halve, and it lumps cuDSS's per-row part (~300 B / row in the BOR plans) into the fill constant
# (BOR m = 0 predicts 1.7x its lu entries, m >= 1 within 3 %). Known double count: those counters are device_bytes =
# permanent + the in-pool operator copies, rhs, scales and index maps, which the estimators also book as pool items
# (cudss_permanent_bytes / device_bytes 0.77 .. 0.89 on the 2D FDFD runs); the rule is to be rebuilt as factor entries
# x the bytes each entry stores (no rescaling to measured bytes: hides an extra copy instead of showing it). The growth
# law is the other open point: n^(4/3) holds for a mesh that is thick in all three directions; a film of a few swept
# prism layers has line separators (a 2D mesh of node columns) and fills far less (sin_square_membrane_modes_3d, 2
# quadratic layers, 355,555 unknowns, 73 entries per row: 138.5e6 factor entries where the rule counts 369e6;
# float64 LDL^T permanent bytes / factor entry 10.5).
CUDSS_FILL_2D = 3.4                   # factor values per n log2 n at CUDSS_REFERENCE_ROWS[2] stored entries per row
CUDSS_FILL_3D = 6.0                   # factor values per n^(4/3) at CUDSS_REFERENCE_ROWS[3] entries per row
CUDSS_REFERENCE_ROWS = {2: 5.0, 3: 13.0}
CUDSS_FIXED_BYTES = 256 * 1024        # per factor, any size (workspace, plan data)
CUDSS_PEAK_RATIO = 1.0                # peak_device_memory / permanent_device_memory: equal in every 2026-11 selftest run
CUDSS_LU_OVER_LDLT = 2.0              # a general LU stores L and U: twice the entries of one triangle
CUDSS_INDEX_BYTES = 8                 # per factor entry beyond its value (BOR plans: double - mixed = the value bytes alone)
# Counted rule (a FactorFill in place of the growth law): permanent = factor entries x the factor's value bytes + the
# matrix copy cuDSS keeps as given (lower triangle of a symmetric kind; factor value bytes + int32 column per stored
# entry) + CUDSS_ROW_BYTES per row (index lists, permutations, solve workspace). The first two are what the plan
# stores; the row part is what cudss_permanent_bytes leaves beyond them: 458 / 374 B on the 110^2 / 190^2 complex
# 5-point FDFD grids, 560 B on the 355,555-unknown float64 film (hankel cudss - cudss_mixed = 8 B x (factor entries +
# stored lower entries) exactly: values and matrix copy halve, nothing else does).
CUDSS_ROW_BYTES = 450


class FactorFill:
    """Counted fill for factor_device_bytes, given in place of `dimensions`: entries of one triangle of the factor
    (diagonal included) in unknowns, e.g. fem.dissection node pairs x (dofs per node)^2 on a built element mesh."""

    def __init__(self, entries):
        self.entries = float(entries)


def factor_device_bytes(n_unknowns, entries, value_bytes=16, dimensions=2, mixed=False, symmetric=True) -> tuple:
    """(permanent, peak extra) estimated device bytes of a cuDSS factor outside the cupy pool: n_unknowns rows,
    `entries` stored operator entries (full pattern), value_bytes of the operator values (mixed: half-size values, the
    CUDSS_INDEX_BYTES unchanged); dimensions 2 (n log2 n fill: tensor 2D, triangles, BOR) or 3 (n^(4/3): 3D grids,
    tetrahedra), or a FactorFill: its counted entries x the factor value bytes + the kept matrix copy + CUDSS_ROW_BYTES
    per row (no growth law, no fitted fill); symmetric False: a general LU (CUDSS_LU_OVER_LDLT). Fit: module notes."""
    if isinstance(dimensions, FactorFill):
        n_unknowns, entries = max(int(n_unknowns), 2), int(entries)
        factor_value_bytes = int(value_bytes) // 2 if mixed else int(value_bytes)
        stored = (entries + n_unknowns) // 2 if symmetric else entries
        factor_entries = max(dimensions.entries * (1.0 if symmetric else CUDSS_LU_OVER_LDLT), float(stored))
        permanent = int(CUDSS_FIXED_BYTES + factor_value_bytes * factor_entries
                        + (factor_value_bytes + 4) * stored + CUDSS_ROW_BYTES * n_unknowns)
        return permanent, int((CUDSS_PEAK_RATIO - 1.0) * permanent)
    n_unknowns = max(int(n_unknowns), 2)
    dimensions = 3 if dimensions == 3 else 2
    rows = max(float(entries) / n_unknowns, 1.0)
    growth = n_unknowns ** (4.0 / 3.0) if dimensions == 3 else n_unknowns * math.log2(n_unknowns)
    fill = (CUDSS_FILL_3D if dimensions == 3 else CUDSS_FILL_2D) * math.sqrt(rows / CUDSS_REFERENCE_ROWS[dimensions])
    stored_value_bytes = int(value_bytes) / 2 if mixed else int(value_bytes)
    precision = (stored_value_bytes + CUDSS_INDEX_BYTES) / (int(value_bytes) + CUDSS_INDEX_BYTES)
    scale = precision * (1.0 if symmetric else CUDSS_LU_OVER_LDLT)
    permanent = int(CUDSS_FIXED_BYTES + fill * growth * int(value_bytes) * scale)
    return permanent, int((CUDSS_PEAK_RATIO - 1.0) * permanent)


_external = {'live': 0, 'high_water': 0}   # device bytes of the live cuDSS solvers (outside the pool), high-water mark


def external_live_bytes() -> int:
    """permanent_device_memory of every planned, not yet freed cuDSS solver of this process (outside the cupy pool)."""
    return _external['live']


def external_high_water_bytes() -> int:
    """High-water mark of the live cuDSS bytes since reset_external_high_water (a plan counts its peak estimate)."""
    return _external['high_water']


def reset_external_high_water() -> None:
    """Restart the high-water mark at the bytes live now (study.build_session: one build and its run)."""
    _external['high_water'] = _external['live']


def relative_asymmetry(matrix) -> float:
    """max |A - A^T| / max |A| of a host scipy sparse matrix (transpose, not conjugate: 0 = complex symmetric)."""
    csr = matrix.tocsr()
    largest = float(abs(csr).max()) if csr.nnz else 0.0
    if largest == 0.0:
        return 0.0
    difference = (csr - csr.T).tocsr()
    return float(abs(difference).max()) / largest if difference.nnz else 0.0


def _givens(diagonal, below):
    """(cosine, sine) of the rotation [[c, s], [-conj(s), c]] that zeroes `below` (real >= 0) under `diagonal`."""
    scale = math.hypot(abs(diagonal), below)
    if scale == 0.0:
        return 1.0, 0.0
    if diagonal == 0:
        return 0.0, 1.0
    return abs(diagonal) / scale, (diagonal / abs(diagonal)) * (below / scale)


def _figure(value) -> str:
    """value with 3 significant digits for error texts, 'not computed' for None (a backward error skipped by R4)."""
    return 'not computed' if value is None else f'{value:.3g}'


def _to_host(xp):
    """Function giving a host numpy copy of an array of the array module xp (cupy: .get(), numpy: the array)."""
    return np.asarray if xp is np else (lambda array: array.get())


def _orthogonalized(basis, vector, xp):
    """Classical Gram-Schmidt, twice, of a vector of the array module xp (cupy or numpy; in place) against the
    orthonormal rows of `basis`: (host coefficients, host norm of what is left) in one device-to-host read (B8: was one
    .item() per row). Real values skip the conjugations (identity there; each one a kernel launch and a copy)."""
    complex_values = vector.dtype.kind == 'c'

    def inner_products(vector):   # <b_i, v> without a conjugated copy of the basis
        return xp.conj(basis @ xp.conj(vector)) if complex_values else basis @ vector
    coefficients = inner_products(vector)
    vector -= coefficients @ basis
    correction = inner_products(vector)   # the second pass keeps the basis orthogonal to rounding
    vector -= correction @ basis
    coefficients += correction
    norm = xp.asarray(xp.linalg.norm(vector)).reshape(1).astype(coefficients.dtype, copy=False)
    packed = _to_host(xp)(xp.concatenate((coefficients, norm)))
    return packed[:-1], float(packed[-1].real)


def gmres_refine(xp, product, single_solve, rhs, solution, operator_norm, stall_test=False,
                 backward_error=MIXED_SOLVE_BACKWARD_ERROR, used=0):
    """(solution, used, (relative residual, backward error, unrefined)) of restarted flexible GMRES on E y = rhs from
    `solution` (updated in place), preconditioned by the low-precision solve single_solve(v) (GMRES-IR; product(v) =
    E v, new arrays both; operator_norm() = sqrt(||E||_1 ||E||_inf), called when needed), until the relative residual
    MIXED_SOLVE_TOLERANCE, the backward error `backward_error`, the float64 floor (module header) or
    MIXED_MAX_REFINEMENTS iterations (`used` counts them, starting from the given value); stall_test (a probe or deferred
    check): also stop after MIXED_STALL_ITERATIONS when the gate is out of reach (module header). xp is the array module
    of the vectors (cupy: CudssFactorization, numpy: host_direct). Basis and directions are (cycle, n) arrays: classical
    Gram-Schmidt twice by batched products, one host read per iteration (coefficients + norm) and one per restart (two
    norms), one product for the update (B8). The errors are those of the returned solution ((None, None, None) for a
    zero rhs)."""
    to_host = _to_host(xp)
    n_unknowns = int(rhs.shape[0])
    value_dtype = solution.dtype
    rhs_norm = float(xp.linalg.norm(rhs))
    estimate_met = False   # the last cycle's GMRES estimate met the target
    stalled = False        # stall_test: the rate so far cannot reach the probe gate
    unrefined = None
    while True:
        residual = rhs - product(solution)
        residual_norm, solution_norm = (float(norm) for norm in
                                        to_host(xp.stack((xp.linalg.norm(residual), xp.linalg.norm(solution)))))
        target = max(MIXED_SOLVE_TOLERANCE * rhs_norm, backward_error
                     * (operator_norm() * solution_norm + rhs_norm))
        if unrefined is None and rhs_norm > 0.0:
            unrefined = residual_norm / rhs_norm
        cycle = min(GMRES_RESTART, MIXED_MAX_REFINEMENTS - used)
        # estimate met, true residual not: the floor
        if residual_norm <= target or cycle <= 0 or estimate_met or stalled:
            # (relative residual, backward error, unrefined) of E y = rhs: the deferred check's gate values
            errors = ((None, None, None) if rhs_norm == 0.0 else
                      (residual_norm / rhs_norm, residual_norm / (operator_norm() * solution_norm + rhs_norm),
                       unrefined))
            return solution, used, errors
        basis = xp.empty((cycle + 1, n_unknowns), dtype=value_dtype)
        directions = xp.empty((cycle, n_unknowns), dtype=value_dtype)
        basis[0] = residual / residual_norm
        rotations, filled = [], 0
        hessenberg = np.zeros((cycle + 1, cycle), dtype=value_dtype)
        projected = np.zeros(cycle + 1, dtype=value_dtype)   # rotated residual coordinates
        projected[0] = residual_norm
        for column in range(cycle):
            directions[column] = single_solve(basis[column])
            vector = product(directions[column])
            hessenberg[:column + 1, column], next_norm = _orthogonalized(basis[:column + 1], vector, xp)
            filled = column + 1
            for row, (cosine, sine) in enumerate(rotations):
                upper, lower = hessenberg[row, column], hessenberg[row + 1, column]
                hessenberg[row, column] = cosine * upper + sine * lower
                hessenberg[row + 1, column] = -np.conj(sine) * upper + cosine * lower
            cosine, sine = _givens(hessenberg[column, column], next_norm)
            rotations.append((cosine, sine))
            hessenberg[column, column] = cosine * hessenberg[column, column] + sine * next_norm
            previous = projected[column]
            projected[column], projected[column + 1] = cosine * previous, -np.conj(sine) * previous
            used += 1
            estimate = abs(projected[column + 1])
            if estimate <= target or next_norm == 0.0:
                estimate_met = True
                break
            if stall_test and used == MIXED_STALL_ITERATIONS:
                gate = max(MIXED_SOLVE_TOLERANCE * rhs_norm, BACKWARD_ERROR_TOLERANCE
                           * (operator_norm() * solution_norm + rhs_norm))
                rate = (residual_norm / estimate) ** (1.0 / MIXED_STALL_ITERATIONS)
                remaining = MIXED_MAX_REFINEMENTS - MIXED_STALL_ITERATIONS
                if estimate / rate ** remaining > MIXED_STALL_MARGIN * gate:
                    stalled = True
                    break
            xp.divide(vector, next_norm, out=basis[column + 1])   # no temporary + copy
        weights = np.linalg.solve(np.triu(hessenberg[:filled, :filled]), projected[:filled])
        solution += xp.asarray(weights) @ directions[:filled]


def csr_rows(indptr, dtype=np.int64):
    """(nnz,) row of every stored entry of a host CSR pattern (indptr), in `dtype`: the one per-entry row helper of
    the solvers package."""
    indptr = np.asarray(indptr)
    return np.repeat(np.arange(indptr.size - 1, dtype=dtype), np.diff(indptr))


def _row_indices(csr):
    """(nnz,) row of every stored entry of a host CSR."""
    return csr_rows(csr.indptr)


def _int32_csr(data, indices, indptr, shape):
    """Host scipy CSR with int32 indices / indptr (cuDSS / cupyx index type), entries as given (int32 index arrays
    shared, not copied)."""
    import scipy.sparse
    return scipy.sparse.csr_matrix((data, indices.astype(np.int32, copy=False), indptr.astype(np.int32, copy=False)),
                                   shape=shape)


def canonical_csr(host_operator, dtype=None):
    """host_operator itself when it is a canonical scipy CSR (sorted, no duplicates) of `dtype` (None: its own), else
    a canonical CSR copy (explicit zeros kept); the caller's operator is never modified."""
    csr = host_operator.tocsr()
    if dtype is not None:
        csr = csr.astype(dtype, copy=False)
    if csr.has_canonical_format:
        return csr
    if csr is host_operator:
        csr = csr.copy()
    csr.sum_duplicates()
    return csr


def _lower_triangle(csr):
    """Lower triangle (diagonal included) of a canonical host CSR, explicit zeros kept: its pattern follows the
    operator's pattern, never its values."""
    rows = _row_indices(csr)
    keep = csr.indices <= rows
    indptr = np.concatenate([[0], np.cumsum(np.bincount(rows[keep], minlength=csr.shape[0]))])
    return _int32_csr(csr.data[keep], csr.indices[keep], indptr, csr.shape)


def _direct_solver(operator, rhs, kind):
    """Blocking nvmath DirectSolver on a cupyx CSR (kind 'symmetric' / 'spd': its lower triangle) and its cupy rhs
    buffer, single-threaded host planning unless the environment variable CUDSS_THREADING_LIB names a threading layer."""
    # Not the wheel's own layer: nvidia/cu13/bin/cudss_mtlayer_vcomp14064_0.dll solved correctly here, then crashed
    # the process with access violations in its worker threads (Windows, nvmath 1.0 / cuDSS 0.8).
    _join_library_thread()
    from nvmath.sparse.advanced import DirectSolver, DirectSolverMatrixType, DirectSolverMatrixViewType
    _thread_size_buffers()
    options ={'blocking': True, 'logger': _LOGGER}
    if kind != 'general':
        options.update(sparse_system_type=(DirectSolverMatrixType.SPD if kind == 'spd'
                                           else DirectSolverMatrixType.SYMMETRIC),
                       sparse_system_view=DirectSolverMatrixViewType.LOWER)
    return DirectSolver(operator, rhs, options=options)


def _free_device_bytes() -> int:
    """Free device memory after returning the cupy pool's cached blocks (cuDSS allocates outside the pool)."""
    import cupy as cp
    from ..core.gpu_memory import memory_status
    cp.get_default_memory_pool().free_all_blocks()
    return int(memory_status().device_free)


class _FactorRejected(Exception):
    """A factorisation or probe that failed (wraps the error raised when no fallback is left)."""

    def __init__(self, error):
        super().__init__(str(error))
        self.error = error


class ProbedFactor:
    """Request / precision state and probe gate shared by CudssFactorization and host_direct.HostDirectFactorization
    (their __init__ sets real, precision, value_dtype, positive_definite, symmetric_requested, has_symmetrizer,
    matrix_kind, n_unknowns, _factor_dtype, _mixed_rejected, _probe and the probe errors; each defines `valid`)."""

    @property
    def symmetric(self) -> bool:
        """True for an LDL^T or Cholesky factor (cuDSS: also what makes the transposed solve available)."""
        return self.matrix_kind in ('symmetric', 'spd')

    @property
    def mixed_factor(self) -> bool:
        """True while the planned factor is low precision (precision 'mixed' and no double fallback yet)."""
        return self._factor_dtype != self.value_dtype

    @property
    def factor_precision(self) -> str:
        """'mixed' or 'double': what the planned factor is (precision is what was requested)."""
        return 'mixed' if self.mixed_factor else 'double'

    def _wanted_factor_dtype(self):
        """Factor dtype of the next plan: value_dtype in double or after a rejected mixed factor, else float32 /
        complex64."""
        if self.precision == 'double' or self._mixed_rejected:
            return self.value_dtype
        return np.float32 if self.real else np.complex64

    def can_refactorize(self, real, precision, symmetric=False, positive_definite=False, symmetrizer=None) -> bool:
        """Whether refactorize serves a request with these options (same dtype, precision and symmetry request)."""
        return (self.valid and bool(real) == self.real and precision == self.precision
                and bool(positive_definite) == self.positive_definite
                and (bool(symmetric) or positive_definite or symmetrizer is not None) == self.symmetric_requested
                and (symmetrizer is not None) == self.has_symmetrizer)

    def _probe_vector(self) -> np.ndarray:
        """Host probe rhs (seeded normal, complex unless real: value_dtype), drawn once per size (a sweep refactorises
        per frequency)."""
        if self._probe is None or self._probe.size != self.n_unknowns:
            rng = np.random.default_rng(_PROBE_SEED)
            probe = rng.standard_normal(self.n_unknowns)
            if not self.real:
                probe = probe + 1j * rng.standard_normal(self.n_unknowns)
            self._probe = probe
        return self._probe

    def _probe_accepted(self, residual_tolerance) -> bool:
        """The last probe met residual_tolerance or BACKWARD_ERROR_TOLERANCE (False on NaN)."""
        return bool(self.probe_residual <= residual_tolerance
                    or self.probe_backward_error <= BACKWARD_ERROR_TOLERANCE)


class CudssFactorization(ProbedFactor):
    """Device LU / LDL^T / Cholesky (precision mixed: low-precision factor of the equilibrated operator + float64
    refinement) of a square host scipy operator by cuDSS (module header); solve cupy in / out, solve_host numpy in / out
    (transpose: only with a symmetric factor), refactorize(new operator) reuses the plan of an unchanged pattern;
    host_operator None makes an empty factor for from_device_values."""

    back_end = 'cudss'   # the factor kind of both precisions (callers branch on precision)
    host_bytes = 0
    host_arrays = False
    recyclable = False   # no GMRES on an earlier factor: a value-only refactorize is the cheap re-solve path

    def __init__(self, host_operator, real=False, symmetric=False, precision='double', row_scale=None,
                 positive_definite=False, symmetrizer=None, symmetric_known=False):
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {', '.join(PRECISIONS)}, got {precision!r}")
        if positive_definite and not real:
            raise ValueError('CudssFactorization(positive_definite=True) needs real=True (SPD; Hermitian HPD is not '
                             'offered)')
        self.precision = precision
        self.real = bool(real)
        self.value_dtype = np.float64 if real else np.complex128   # solve in / out
        self._mixed_rejected = False      # a mixed factor failed its probe once: double factors from then on
        self.last_double_fallback = False   # the last refactorize replaced a rejected mixed factor by a double one
        self.mixed_rejection = None       # text of that rejection
        self._factor_dtype = self._wanted_factor_dtype()   # dtype of the planned factor
        self.positive_definite = bool(positive_definite)
        self._cholesky_rejected = False   # a Cholesky failed once: LDL^T from then on
        self.symmetric_requested = bool(symmetric) or positive_definite or symmetrizer is not None
        self.has_symmetrizer = symmetrizer is not None
        self.n_unknowns = None
        self._operator_norm = None      # sqrt(||E||_1 ||E||_inf) of the last factorised E (operator_norm: lazy)
        self._solver = None
        self._operator = self._factor_matrix = self._rhs = self._row_scale = self._symmetrizer = None
        self._diagonal = None           # symmetric: diag(E) on the device (_operator then holds only the lower triangle)
        self._operator_transpose = None
        self._pattern = None            # host (indptr, indices) of the planned full operator
        self._pattern_source = None     # caller arrays last found equal to it (refactorize_device_values identity test)
        self._device_maps = None        # device index arrays of that pattern (refactorize_device_values), per plan
        self._probe = None              # host probe rhs of _factorize_and_probe (same seed every factorisation)
        self.probe_residual = None      # last probe (or deferred check): relative residual, backward error
        self.probe_backward_error = None
        self._check_pending = False     # deferred check: the next (mixed: full-accuracy) solve judges this factor
        # mixed: backward error a full-accuracy solve refines to (a caller may loosen it per factor or per solve call:
        # decision D2; checks and probes never go above BACKWARD_ERROR_TOLERANCE)
        self.solve_backward_error = MIXED_SOLVE_BACKWARD_ERROR
        self._last_solve_errors = (None, None, None)   # (relative residual, backward error, unrefined) of the last GMRES
        # callable(plans, plan_seconds, reused_orderings) told of a double re-plan inside a solve (deferred check)
        self.fallback_listener = None
        self.matrix_kind = None         # 'general' | 'symmetric' | 'spd' of the planned solver
        self.permanent_bytes = 0        # cuDSS's permanent / peak device estimates of the last plan (outside the pool)
        self.peak_bytes = 0
        self.lu_entries = 0             # factor entries (cuDSS lu_nnz) of the last factorisation
        self._tracked_bytes = 0        # this object's share of external_live_bytes (its planned solver)
        self.n_solves = 0               # cuDSS solves (probes included) and their wall time (blocking: synchronised)
        self.solve_seconds = 0.0
        self.n_refinements = 0          # mixed: float64 refinement steps of all solves, and of the last one
        self.last_refinements = 0
        self.n_plans = 0                # cuDSS analyses (reordering + symbolic) and factorisations of this object
        self.n_factorizations = 0
        self.total_plan_seconds = 0.0
        self.total_factor_seconds = 0.0
        self.plan_seconds = 0.0         # last plan (0 when the last refactorize reused it)
        self.last_plan_reused = False
        self.last_plans = 0             # plans / reused orderings / plan seconds of the last refactorize (fallback
        self.last_reused_orderings = 0  # re-plans included: SparseDirectSolver counts them)
        self.last_plans_seconds = 0.0
        self.asymmetry = None           # relative asymmetry of the last operator (None: symmetry not requested)
        self._ordering = None           # (key, row permutation, partition tree) of the last plan (ordering reuse)
        self.last_ordering_reused = False   # the last plan took a kept ordering (no host reordering)
        self.n_reused_orderings = 0
        if host_operator is not None:   # None: from_device_values factorises next
            self.refactorize(host_operator, row_scale=row_scale, symmetrizer=symmetrizer,
                             symmetric_known=symmetric_known)

    @classmethod
    def from_device_values(cls, values, indptr, indices, real=False, symmetric=False, precision='double',
                           row_scale=None, positive_definite=False, check_symmetry=False, lower=False):
        """A factor planned straight from device values (cupy (nnz,)) on a canonical host pattern (R7: no host CSR);
        check_symmetry as refactorize_device_values (else the caller vouches for symmetry whenever it asks for it);
        lower: the values are the lower triangle only (refactorize_device_lower; symmetric request)."""
        factor = cls(None, real=real, symmetric=symmetric, precision=precision, positive_definite=positive_definite)
        if lower:
            factor.refactorize_device_lower(values, indptr, indices, row_scale=row_scale)
        else:
            factor.refactorize_device_values(values, indptr, indices, row_scale=row_scale,
                                             check_symmetry=check_symmetry)
        return factor

    def host_pattern_of(self, device_csr):
        """Host (indptr, indices) of a canonical cupyx CSR: the planned pattern itself when the device index arrays
        equal it (compared on the device, no download), else downloaded copies."""
        import cupy as cp
        if (self._solver is not None and device_csr.indptr.size == self._pattern[0].size
                and device_csr.indices.size == self._pattern[1].size):
            maps = self._device_index_maps()
            if bool(cp.array_equal(device_csr.indptr, maps['indptr'])) and bool(
                    cp.array_equal(device_csr.indices, maps['columns'])):
                return self._pattern
        return cp.asnumpy(device_csr.indptr), cp.asnumpy(device_csr.indices)

    @property
    def valid(self) -> bool:
        """False after release()."""
        return self._solver is not None

    def inertia(self):
        """(n_positive, n_negative) pivots of a real LDL^T / Cholesky factor = the inertia of A (E = R A R is a
        congruence); None for a general LU, a complex or a symmetrizer (D A) factor; mixed: the float32 pivots."""
        if self._solver is None or not self.real or self.has_symmetrizer or not self.symmetric:
            return None
        if self.matrix_kind == 'spd':
            return self.n_unknowns, 0
        counts = self._solver.factorization_info.inertia
        return int(counts[0]), int(counts[1])

    def refactorize(self, host_operator, row_scale=None, symmetrizer=None, symmetric_known=False) -> bool:
        """Factorise a new host operator of the same size (new row_scale / symmetrizer values allowed); True when the
        plan was reused (same pattern and matrix type: values in place + factorize), False after a fresh plan.
        symmetric_known: a canonical CSR (sorted, no duplicates) known to be symmetric: no canonicalising copy and no
        asymmetry check (fem.fixed_pattern reduced tangents). On the planned pattern only the values (and a
        symmetrizer) go to the device and refactorize_device_values does the rest there (H1; asymmetry checked on the
        device)."""
        import cupy as cp
        if (self._solver is not None and (symmetrizer is not None) == self.has_symmetrizer
                and host_operator.shape == (self.n_unknowns, self.n_unknowns)):
            csr = canonical_csr(host_operator)
            if np.array_equal(self._pattern[0], csr.indptr) and np.array_equal(self._pattern[1], csr.indices):
                if self.real and np.iscomplexobj(csr.data):
                    raise ValueError('CudssFactorization(real=True) needs a real operator')
                return self.refactorize_device_values(cp.asarray(csr.data, dtype=self.value_dtype), *self._pattern,
                                                      row_scale=row_scale, check_symmetry=not symmetric_known,
                                                      symmetrizer=symmetrizer)
        started = time.perf_counter()
        mark = self._plan_mark()
        host_csr, row_scale, symmetrizer = self._equilibrated(host_operator, row_scale, symmetrizer,
                                                              canonical=symmetric_known)
        if not self.symmetric_requested:
            self.asymmetry = None
        else:
            self.asymmetry = 0.0 if symmetric_known else relative_asymmetry(host_csr)
        symmetric = self.symmetric_requested and self.asymmetry <= SYMMETRY_TOLERANCE
        factor_host = _lower_triangle(host_csr) if symmetric else host_csr
        same_pattern = (self._solver is not None and np.array_equal(self._pattern[0], host_csr.indptr)
                        and np.array_equal(self._pattern[1], host_csr.indices))
        kind, reuse = self._kind_and_reuse(symmetric, same_pattern, host_csr.indptr, host_csr.indices)
        self._row_scale = cp.asarray(row_scale)
        self._symmetrizer = None if symmetrizer is None else cp.asarray(symmetrizer)
        attempt_kind, retried = kind, False
        while True:
            if retried or not reuse:
                self._plan(host_csr, factor_host, attempt_kind)
            else:
                self._upload_values(host_csr, factor_host)
            try:
                self._factorize_and_probe()
                break
            except _FactorRejected as rejected:
                attempt_kind = self._next_attempt(rejected, attempt_kind)
                if attempt_kind is None:
                    self.release()
                    raise rejected.error from None
                retried = True
        return self._finish_refactorize(reuse and not retried, started, mark)

    def _finish_refactorize(self, reused, started, mark) -> bool:
        """refactorize's bookkeeping once a factor was accepted (reused: on the kept plan); returns reused."""
        self.last_plan_reused = reused
        if reused:
            self.plan_seconds = 0.0
        self.last_plans, self.last_reused_orderings, self.last_plans_seconds = self._plans_since(mark)
        self.n_factorizations += 1
        self.factor_seconds = time.perf_counter() - started   # equilibration, (plan,) factorize and probe
        self.total_factor_seconds += self.factor_seconds
        return reused

    def _plan_mark(self):
        """(plans, reused orderings, plan seconds) so far, for _plans_since."""
        return self.n_plans, self.n_reused_orderings, self.total_plan_seconds

    def _plans_since(self, mark):
        """(plans, reused orderings, plan seconds) made since `mark` (fallback re-plans included)."""
        plans, reused, seconds = mark
        return self.n_plans - plans, self.n_reused_orderings - reused, self.total_plan_seconds - seconds

    def _remembered_mixed_rejection(self, indptr, indices, kind) -> bool:
        """True (and double from now on, no low-precision attempt; mixed_rejection says why) when a mixed factor of this
        kind and pattern was rejected earlier while remember_cudss_orderings is on (module header "Rejected mixed
        factors")."""
        if _rejected_mixed is None or self._wanted_factor_dtype() == self.value_dtype:
            return False
        planned = self._pattern is not None and indptr is self._pattern[0] and indices is self._pattern[1]
        key = (kind, self.real, *(self._planned_pattern_key() if planned else _pattern_key(indptr, indices)))
        text = _rejected_mixed.get(key)
        if text is None:
            return False
        _rejected_mixed.move_to_end(key)
        _LOGGER.info('cuDSS mixed %s skipped: rejected before on this pattern (%s)', _KIND_TEXTS[kind], text)
        self._mixed_rejected = True   # not a last_double_fallback: no factor was rejected (counters stay plan-exact)
        self.mixed_rejection = f'{text} (remembered: no new low-precision attempt on this pattern)'
        return True

    def _planned_pattern_key(self):
        """_pattern_key of the planned pattern (an own copy, never changed in place), hashed once per plan."""
        kept = getattr(self, '_planned_digest', None)
        if kept is None or kept[0] is not self._pattern:
            kept = self._planned_digest = (self._pattern, _pattern_key(*self._pattern))
        return kept[1]

    def _kind_and_reuse(self, symmetric, same_pattern, indptr, indices):
        """(kind, reuse) of the next factor: the requested kind of a symmetric operator, else 'general'; reuse = the
        planned solver on the same pattern, kind and factor dtype; not reused, a pattern whose mixed factor was rejected
        before plans double at once (H4). Clears last_double_fallback."""
        kind = self._requested_kind() if symmetric else 'general'
        reuse = (self._solver is not None and same_pattern and kind == self.matrix_kind
                 and self._factor_dtype == self._wanted_factor_dtype())
        self.last_double_fallback = False
        if not reuse and self._remembered_mixed_rejection(indptr, indices, kind):
            kind = self._requested_kind() if symmetric else 'general'
        return kind, reuse

    def _remember_mixed_rejection(self, kind, text) -> None:
        """Keep a mixed rejection of the planned pattern while remember_cudss_orderings is on (oldest dropped)."""
        if _rejected_mixed is None or self._pattern is None:
            return
        key = (kind, self.real, *self._planned_pattern_key())
        _rejected_mixed[key] = text
        _rejected_mixed.move_to_end(key)
        while len(_rejected_mixed) > KEPT_ORDERINGS:
            _rejected_mixed.popitem(last=False)

    def _next_attempt(self, rejected, kind):
        """Matrix kind to re-plan after a rejected factor of `kind`: a Cholesky -> LDL^T, then a mixed factor -> the
        double one (a mixed SPD LDL^T -> the double Cholesky; both sticky for later refactorisations, the mixed
        rejection also remembered for this pattern); None when nothing is left."""
        if kind == 'spd':
            _LOGGER.info('cuDSS Cholesky rejected (%s): re-planned as LDL^T', rejected)
            self._cholesky_rejected = True
            return 'symmetric'
        if self.mixed_factor:
            self._mixed_rejected = True
            self.last_double_fallback = True
            self.mixed_rejection = str(rejected)
            self._remember_mixed_rejection(kind, self.mixed_rejection)
            return kind if kind == 'general' else self._requested_kind()
        return None

    def _requested_kind(self) -> str:
        """Matrix kind refactorize plans for an operator known to be symmetric when symmetry was requested: a Cholesky
        only for a double factor (a float32 Cholesky hung on the hirsch2025 E5 static matrix: LDL^T instead)."""
        if not self.symmetric_requested:
            return 'general'
        cholesky = (self.positive_definite and not self._cholesky_rejected
                    and self._wanted_factor_dtype() == self.value_dtype)
        return 'spd' if cholesky else 'symmetric'

    def refactorize_device_values(self, values, indptr, indices, row_scale=None, defer_check=False,
                                  check_symmetry=False, symmetrizer=None) -> bool:
        """refactorize of a canonical CSR given as device values (cupy (nnz,)) on the host pattern (indptr, indices),
        all on the device (no host CSR; plan G5, H1, R7): rows times the symmetrizer D (exactly when the factor was
        built with one: E = R D A R), equilibration (R = row_scale, else from |diag|), lower-triangle take, the value
        refresh on the planned pattern and kind, else (first call, new pattern or kind) a plan of those device arrays.
        check_symmetry: the asymmetry is checked on the device (an operator from the host route); else the caller
        vouches for symmetry whenever it asked for it (as symmetric_known). defer_check: a factor on the reused plan is
        checked by its first (mixed: full-accuracy) solve (module header). True = plan reused."""
        import cupy as cp
        if (symmetrizer is not None) != self.has_symmetrizer:
            raise ValueError('CudssFactorization.refactorize_device_values: give a symmetrizer exactly when the factor '
                             'was built with one')
        indptr, indices = np.asarray(indptr), np.asarray(indices)
        n_rows = int(indptr.size) - 1
        if self.n_unknowns is not None and n_rows != self.n_unknowns:
            raise ValueError(f'CudssFactorization.refactorize_device_values: {n_rows} unknowns, the factor has '
                             f'{self.n_unknowns}')
        if self.real and cp.iscomplexobj(values):
            raise ValueError('CudssFactorization(real=True) needs a real operator')
        values = cp.asarray(values, dtype=self.value_dtype).reshape(-1)
        if values.size != indices.size:
            raise ValueError(f'CudssFactorization.refactorize_device_values: {values.size} values for a pattern of '
                             f'{indices.size} entries')
        started = time.perf_counter()
        mark = self._plan_mark()
        same_pattern = self._pattern is not None and self._on_planned_pattern(indptr, indices)
        if not same_pattern:
            self._set_pattern(indptr, indices)
        self.n_unknowns = n_rows
        maps = self._device_index_maps()
        if symmetrizer is not None:   # D A row by row (the host route's _equilibrated order: D before R)
            symmetrizer = cp.asarray(symmetrizer, dtype=self.value_dtype).reshape(-1)
            if symmetrizer.shape != (n_rows,) or not bool(cp.all(cp.isfinite(symmetrizer) & (symmetrizer != 0))):
                raise ValueError(f'CudssFactorization symmetrizer must be ({n_rows},), finite and nonzero')
            values = values * symmetrizer[maps['rows']]
        scaled = self._device_equilibrated(values, row_scale, maps)
        self._symmetrizer = symmetrizer
        if not self.symmetric_requested:
            self.asymmetry = None
        else:   # of E, as the host route's relative_asymmetry
            self.asymmetry = self._device_asymmetry(scaled, maps) if check_symmetry else 0.0
        symmetric = self.symmetric_requested and self.asymmetry <= SYMMETRY_TOLERANCE
        kind, reuse = self._kind_and_reuse(symmetric, same_pattern, *self._pattern)
        if symmetric:   # the storage of a symmetric kind: lower triangle + diagonal
            stored = scaled[maps['lower']]
            diagonal = cp.where(maps['has_diagonal'], scaled[maps['diagonal_slots']], 0.0)
        else:
            stored, diagonal = scaled, None
        if reuse:
            self._refresh_device_values(stored, diagonal)
        else:
            self._device_plan(stored, diagonal, kind, started)
        return self._factorize_with_fallbacks(reuse, kind, defer_check, started, mark)

    def refactorize_device_lower(self, lower_values, indptr, indices, row_scale=None, defer_check=False) -> bool:
        """refactorize_device_values of an operator whose symmetry was requested and is vouched for, given by its
        lower-triangle values (cupy, in the CSR order of the pattern's lower take: a lower-stored DevicePencil, R4):
        R from the lower diagonal (or row_scale), E_lower and its factor-dtype copy by one kernel straight into the
        planned buffers (else into a new plan), the operator norm left lazy. True = plan reused."""
        import cupy as cp
        if self.has_symmetrizer or not self.symmetric_requested:
            raise ValueError('CudssFactorization.refactorize_device_lower needs a symmetric request without a '
                             'symmetrizer')
        indptr, indices = np.asarray(indptr), np.asarray(indices)
        n_rows = int(indptr.size) - 1
        if self.n_unknowns is not None and n_rows != self.n_unknowns:
            raise ValueError(f'CudssFactorization.refactorize_device_lower: {n_rows} unknowns, the factor has '
                             f'{self.n_unknowns}')
        if self.real and cp.iscomplexobj(lower_values):
            raise ValueError('CudssFactorization(real=True) needs a real operator')
        lower_values = cp.asarray(lower_values, dtype=self.value_dtype).reshape(-1)
        started = time.perf_counter()
        mark = self._plan_mark()
        same_pattern = self._pattern is not None and self._on_planned_pattern(indptr, indices)
        if not same_pattern:
            self._set_pattern(indptr, indices)
        self.n_unknowns = n_rows
        maps = self._device_index_maps()
        if lower_values.size != maps['lower'].size:
            raise ValueError(f'CudssFactorization.refactorize_device_lower: {lower_values.size} values for a lower '
                             f"triangle of {maps['lower'].size} entries")
        row_scale = self._checked_row_scale(lower_values[maps['lower_diagonal_slots']], row_scale, maps)
        self.asymmetry = 0.0
        kind, reuse = self._kind_and_reuse(True, same_pattern, *self._pattern)
        self._row_scale, self._symmetrizer, self.operator_norm = row_scale, None, None   # norm: lazy
        rows, columns = maps['lower_rows'], maps['lower_indices']
        if reuse:
            if self._factor_matrix is self._operator:
                _scaled_lower_kernel(False)(lower_values, rows, columns, row_scale, self._operator.data)
            else:
                _scaled_lower_kernel(True)(lower_values, rows, columns, row_scale, self._operator.data,
                                           self._factor_matrix.data)
            self._diagonal[...] = cp.where(maps['has_diagonal'], self._operator.data[maps['lower_diagonal_slots']],
                                           0.0)
        else:
            stored = _scaled_lower_kernel(False)(lower_values, rows, columns, row_scale)
            self._device_plan(stored, cp.where(maps['has_diagonal'], stored[maps['lower_diagonal_slots']], 0.0),
                              kind, started)
        return self._factorize_with_fallbacks(reuse, kind, defer_check, started, mark)

    def _factorize_with_fallbacks(self, reuse, kind, defer_check, started, mark) -> bool:
        """Factorise the stored device values (deferred: defer_check on a reused plan whose earlier probe passed), the
        fallback re-plans in place, then refactorize's bookkeeping; True = plan reused."""
        attempt_kind, retried = kind, False
        while True:
            try:   # deferred only on a reused plan whose earlier probe passed (a new plan is always probed)
                self._factorize_and_probe(defer=reuse and not retried and defer_check
                                          and self.probe_residual is not None)
                break
            except _FactorRejected as rejected:
                attempt_kind = self._next_attempt(rejected, attempt_kind)
                if attempt_kind is None:
                    self.release()
                    raise rejected.error from None
                self._replan_in_place(attempt_kind)   # same storage: only the kind or the factor dtype changes
                retried = True
        return self._finish_refactorize(reuse and not retried, started, mark)

    def _set_pattern(self, indptr, indices) -> None:
        """A new planned pattern (int32 host copies; the caller's arrays remembered for the identity test)."""
        self._pattern = (np.array(indptr, dtype=np.int32), np.array(indices, dtype=np.int32))
        self._pattern_source = (indptr, indices)
        self._device_maps = None

    def _checked_row_scale(self, diagonal_values, row_scale, maps):
        """Device R: |diag A|^-1/2 from the pattern's diagonal values ((n,) gathered at the diagonal slots; rows without
        a diagonal entry or a zero one get 1), else the caller's row_scale checked ((n,), finite, > 0)."""
        import cupy as cp
        n_rows = self.n_unknowns
        if row_scale is None:
            diagonal = cp.abs(cp.where(maps['has_diagonal'], diagonal_values, 0.0))
            return 1.0 / cp.sqrt(cp.where(diagonal > 0.0, diagonal, 1.0))
        row_scale = cp.asarray(row_scale, dtype=cp.float64).reshape(-1)
        if row_scale.shape != (n_rows,) or not bool(cp.all(cp.isfinite(row_scale) & (row_scale > 0.0))):
            raise ValueError(f'CudssFactorization row_scale must be ({n_rows},), finite and > 0')
        return row_scale

    def _device_equilibrated(self, values, row_scale, maps):
        """Device E = R A R entry by entry (as _equilibrated; R = row_scale, else |diag A|^-1/2): sets _row_scale; the
        operator norm is left lazy (operator_norm)."""
        row_scale = self._checked_row_scale(values[maps['diagonal_slots']], row_scale, maps)
        scaled = values * (row_scale[maps['rows']] * row_scale[maps['columns']])
        self._row_scale, self._symmetrizer, self.operator_norm = row_scale, None, None
        return scaled

    @property
    def operator_norm(self):
        """sqrt(||E||_1 ||E||_inf) of the factorised E (backward errors); a device refresh leaves it to the first use
        (R4: one pass over the stored E; a symmetric E from its lower triangle, row sums = column sums)."""
        if self._operator_norm is None and self._operator is not None:
            import cupy as cp
            import cupyx
            maps = self._device_index_maps()
            magnitudes = cp.abs(self._operator.data)
            row_sums = cp.zeros(self.n_unknowns)
            if self._diagonal is not None:
                rows, columns = maps['lower_rows'], maps['lower_indices']
                cupyx.scatter_add(row_sums, rows, magnitudes)
                cupyx.scatter_add(row_sums, columns, cp.where(rows != columns, magnitudes, 0.0))
                self._operator_norm = float(row_sums.max())
            else:
                column_sums = cp.zeros(self.n_unknowns)
                cupyx.scatter_add(row_sums, maps['rows'], magnitudes)
                cupyx.scatter_add(column_sums, maps['columns'], magnitudes)
                self._operator_norm = float(cp.sqrt(row_sums.max() * column_sums.max()))
        return self._operator_norm

    @operator_norm.setter
    def operator_norm(self, value):
        self._operator_norm = value

    def _refresh_device_values(self, stored, diagonal) -> None:
        """Equilibrated device values in the planned storage (lower triangle + diagonal of a symmetric kind, else the
        full E) written into the planned buffers in place (no re-plan)."""
        self._operator.data[...] = stored
        if diagonal is not None:
            self._diagonal[...] = diagonal
        if self._factor_matrix is not self._operator:
            self._factor_matrix.data[...] = stored.astype(self._factor_dtype)

    def _device_plan(self, stored, diagonal, kind, started) -> None:
        """Device operator copies of equilibrated device values in the storage of `kind` (stored: lower triangle with
        its diagonal for a symmetric kind, else the full E) on the pattern's device index arrays, and a DirectSolver of
        `kind` planned on them (the previous one freed; R7)."""
        import cupyx.scipy.sparse
        self._free_solver()
        self._factor_matrix = self._operator_transpose = None   # the old factor-dtype copy before the new one
        self._factor_dtype = self._wanted_factor_dtype()
        maps = self._device_index_maps()
        shape = (self.n_unknowns, self.n_unknowns)
        symmetric = kind != 'general'
        if symmetric:
            indices, indptr = maps['lower_indices'], maps['lower_indptr']
        else:
            indices, indptr = maps['columns'], maps['indptr']
        self._operator = cupyx.scipy.sparse.csr_matrix((stored, indices, indptr), shape=shape)
        self._diagonal = diagonal if symmetric else None
        self._operator_transpose = self._operator.T if symmetric else None   # a CSC view sharing the arrays
        self._factor_matrix = (self._operator if self._factor_dtype == self.value_dtype else   # cuDSS keeps pointers
                               cupyx.scipy.sparse.csr_matrix((stored.astype(self._factor_dtype), indices, indptr),
                                                             shape=shape))
        self._plan_solver(kind, started)

    def _replan_in_place(self, kind) -> None:
        """Re-plan the current device operator as `kind` with the factor dtype now wanted (a fallback: storage
        unchanged, a double factor reads _operator itself)."""
        self._free_solver()
        self._factor_dtype = self._wanted_factor_dtype()
        if self._factor_dtype == self.value_dtype:
            self._factor_matrix = self._operator
        self._plan_solver(kind, time.perf_counter())

    def _device_asymmetry(self, values, maps) -> float:
        """max |A - A^T| / max |A| of device values on the planned canonical pattern (an entry without a stored mirror
        counts |a_ij|), by a transpose map built on the device once per plan (H1: no host transpose / argsort)."""
        import cupy as cp
        largest = float(cp.abs(values).max()) if values.size else 0.0
        if largest == 0.0:
            return 0.0
        if 'transpose' not in maps:   # int64 keys built in place (refresh_bytes: their peak)
            n_rows = self.n_unknowns
            keys = maps['rows'].astype(cp.int64)   # ascending: canonical CSR order
            keys *= n_rows
            keys += maps['columns']
            mirrored = maps['columns'].astype(cp.int64)
            mirrored *= n_rows
            mirrored += maps['rows']
            slots = cp.searchsorted(keys, mirrored)
            cp.minimum(slots, keys.size - 1, out=slots)
            found = keys[slots] == mirrored
            del keys, mirrored
            maps['transpose'] = cp.where(found, slots, -1).astype(maps['lower'].dtype)
            del slots, found
        transpose = maps['transpose']
        difference = values[cp.maximum(transpose, 0)]   # the mirror entry (0 without one), minus the entry, in place
        difference[transpose < 0] = 0
        difference -= values
        return float(cp.abs(difference).max()) / largest

    def _on_planned_pattern(self, indptr, indices) -> bool:
        """Whether (indptr, indices) is the planned pattern: the factor's own copies or the very arrays matched last
        (identity, E7-7a: a DevicePencil's fixed arrays, ~0.9 ms of np.array_equal per call on E7) first, else
        np.array_equal (a match remembers those arrays). Callers must not modify the arrays they pass in place."""
        for source in (self._pattern, self._pattern_source):
            if source is not None and indptr is source[0] and indices is source[1]:
                return True
        if np.array_equal(self._pattern[0], indptr) and np.array_equal(self._pattern[1], indices):
            self._pattern_source = (indptr, indices)
            return True
        return False

    def _device_index_maps(self) -> dict:
        """Device index arrays of the planned pattern, built once per plan: rows / columns of every entry, indptr, the
        diagonal slot of every row (has_diagonal False where the pattern has none), the lower-triangle take and the
        lower triangle's own rows / indices / indptr / diagonal slots (+ 'transpose', built by _device_asymmetry when
        first needed)."""
        if self._device_maps is None:
            import cupy as cp
            indptr, indices = self._pattern
            n_rows = indptr.size - 1
            rows = csr_rows(indptr, np.int32)
            on_diagonal = np.flatnonzero(indices == rows)
            diagonal_slots = np.zeros(n_rows, dtype=np.int64)
            has_diagonal = np.zeros(n_rows, dtype=bool)
            diagonal_slots[rows[on_diagonal]] = on_diagonal
            has_diagonal[rows[on_diagonal]] = True
            take_dtype = np.int32 if indices.size < 2 ** 31 else np.int64
            is_lower = indices <= rows
            lower = np.flatnonzero(is_lower)
            lower_indptr = np.concatenate([[0], np.cumsum(np.bincount(rows[lower], minlength=n_rows))])
            lower_rank = np.cumsum(is_lower) - 1   # position of a lower entry in the lower take
            self._device_maps = {'rows': cp.asarray(rows), 'columns': cp.asarray(indices.astype(np.int32, copy=False)),
                                 'indptr': cp.asarray(indptr.astype(np.int32, copy=False)),
                                 'diagonal_slots': cp.asarray(diagonal_slots), 'has_diagonal': cp.asarray(has_diagonal),
                                 'lower': cp.asarray(lower.astype(take_dtype)),
                                 'lower_rows': cp.asarray(rows[lower]),
                                 'lower_indices': cp.asarray(indices[lower].astype(np.int32, copy=False)),
                                 'lower_indptr': cp.asarray(lower_indptr.astype(np.int32)),
                                 'lower_diagonal_slots': cp.asarray(lower_rank[diagonal_slots].astype(np.int64))}
        return self._device_maps

    def _equilibrated(self, host_operator, row_scale, symmetrizer, canonical=False):
        """(canonical int32 host CSR of E = R D A R in value_dtype, R, D or None): a copy of the values, scaled entry by
        entry (canonical: host_operator is a canonical CSR already, its index arrays are shared, never modified)."""
        n_rows, n_columns = host_operator.shape
        if n_rows != n_columns:
            raise ValueError(f'CudssFactorization needs a square operator, got shape {host_operator.shape}')
        if self.n_unknowns is not None and n_rows != self.n_unknowns:
            raise ValueError(f'CudssFactorization.refactorize: {n_rows} unknowns, the factor has {self.n_unknowns}')
        self.n_unknowns = int(n_rows)
        if self.real and np.iscomplexobj(host_operator.data):
            raise ValueError('CudssFactorization(real=True) needs a real operator')
        if (symmetrizer is not None) != self.has_symmetrizer:
            raise ValueError('CudssFactorization.refactorize: give a symmetrizer exactly when the factor was built with '
                             'one')
        if canonical:   # only the values are copied (scaled below)
            host_csr = _int32_csr(np.array(host_operator.data, dtype=self.value_dtype), host_operator.indices,
                                  host_operator.indptr, host_operator.shape)
        else:
            csr = host_operator.tocsr().astype(self.value_dtype)   # a copy: the caller's operator stays as it is
            csr.sum_duplicates()                                  # canonical (sorted, summed); explicit zeros kept
            host_csr = _int32_csr(csr.data, csr.indices, csr.indptr, csr.shape)
        rows = _row_indices(host_csr)
        if symmetrizer is not None:
            symmetrizer = np.asarray(symmetrizer, dtype=self.value_dtype)
            if symmetrizer.shape != (n_rows,) or not (np.isfinite(symmetrizer).all() and (symmetrizer != 0).all()):
                raise ValueError(f'CudssFactorization symmetrizer must be ({n_rows},), finite and nonzero')
            host_csr.data *= symmetrizer[rows]
        if row_scale is None:
            diagonal = np.abs(host_csr.diagonal())
            row_scale = 1.0 / np.sqrt(np.where(diagonal > 0.0, diagonal, 1.0))
        row_scale = np.asarray(row_scale, dtype=np.float64)
        if row_scale.shape != (n_rows,) or not (np.isfinite(row_scale).all() and (row_scale > 0.0).all()):
            raise ValueError(f'CudssFactorization row_scale must be ({n_rows},), finite and > 0')
        host_csr.data *= row_scale[rows] * row_scale[host_csr.indices]
        magnitudes = np.abs(host_csr.data)
        self.operator_norm = float(np.sqrt(np.bincount(rows, magnitudes, minlength=n_rows).max()
                                           * np.bincount(host_csr.indices, magnitudes, minlength=n_rows).max()))
        return host_csr, row_scale, symmetrizer

    def _factor_copy(self, factor_host):
        """Device CSR cuDSS factorises: _operator itself in double (the full E, or the lower triangle of a symmetric
        kind), else a factor_dtype copy of factor_host."""
        import cupyx.scipy.sparse
        if self._factor_dtype == self.value_dtype:
            return self._operator
        return cupyx.scipy.sparse.csr_matrix(factor_host.astype(self._factor_dtype))

    def _plan(self, host_csr, factor_host, kind):
        """Upload, create and plan a DirectSolver of `kind` (freeing the previous one); ValueError when cuDSS's peak
        estimate exceeds DEVICE_FRACTION of the free device memory."""
        import cupy as cp
        import cupyx.scipy.sparse
        self._free_solver()
        self._device_maps = None   # a new plan may have a new pattern
        # the old device copies go before the new ones are uploaded (a new order's pattern: else both sets at once)
        self._operator = self._factor_matrix = self._diagonal = self._operator_transpose = None
        self._factor_dtype = self._wanted_factor_dtype()
        started = time.perf_counter()
        symmetric = kind != 'general'
        # value_dtype, residuals and refinement: the full E, or its lower triangle + diagonal (_product)
        self._operator = cupyx.scipy.sparse.csr_matrix(factor_host if symmetric else host_csr)
        self._diagonal = cp.asarray(factor_host.diagonal()) if symmetric else None
        self._operator_transpose = self._operator.T if symmetric else None   # a CSC view sharing the arrays
        self._factor_matrix = self._factor_copy(factor_host)   # cuDSS keeps its pointers
        self._pattern = (host_csr.indptr.copy(), host_csr.indices.copy())
        self._pattern_source = None   # the caller arrays matched to it (_on_planned_pattern)
        self._plan_solver(kind, started)

    def _plan_solver(self, kind, started):
        """Create and plan the DirectSolver of `kind` on _factor_matrix (a fresh rhs buffer of the factor dtype; the
        previous solver already freed), timed from `started`; ValueError when cuDSS's peak estimate exceeds
        DEVICE_FRACTION of the free device memory."""
        import cupy as cp
        from ..core.gpu_memory import format_bytes
        self._rhs = cp.zeros(self.n_unknowns, dtype=self._factor_dtype)       # cuDSS keeps this buffer's pointer
        self._solver = _direct_solver(self._factor_matrix, self._rhs, kind)
        key = self._ordering_key(kind)
        known = self._known_ordering(key)
        self.last_ordering_reused = known is not None and _set_ordering(self._solver, known)
        if known is not None and not self.last_ordering_reused:   # never the permutation without its tree
            self._solver.free()
            self._solver = _direct_solver(self._factor_matrix, self._rhs, kind)
        self._solver.plan()
        if self.last_ordering_reused:
            self._ordering = (key, *known)   # kept alive while this solver may read them
            self.n_reused_orderings += 1
        elif key is not None:
            ordering = _read_ordering(self._solver)
            if ordering is not None:
                self._remember_ordering(key, ordering)
        self.matrix_kind = kind
        self.plan_seconds = time.perf_counter() - started   # upload + reordering (host) + symbolic factorisation
        self.total_plan_seconds += self.plan_seconds
        self.n_plans += 1
        estimates = self._solver.plan_info.memory_estimates
        self._permanent_bytes = int(estimates['permanent_device_memory'])
        peak_bytes = int(estimates['peak_device_memory'])
        self.permanent_bytes, self.peak_bytes = self._permanent_bytes, peak_bytes
        self._track_external(self._permanent_bytes, peak_bytes)
        free_bytes = _free_device_bytes()
        if peak_bytes > DEVICE_FRACTION * free_bytes:
            self.release()
            raise CudssMemoryError(f'the cuDSS factorisation of {self.n_unknowns:,} unknowns needs an estimated '
                             f'{format_bytes(peak_bytes)} of GPU memory, over {DEVICE_FRACTION:.0%} of the free device '
                             f'memory ({format_bytes(free_bytes)}): refused before it could overrun the device; '
                             f'{_TO_SUPERLU}, or coarsen the mesh')

    def _ordering_key(self, kind):
        """(ordering family of kind, n, nnz, digest of the planned pattern); None before a plan. Real and complex
        operators of one pattern share the key."""
        if self._pattern is None:
            return None
        return (ORDERING_FAMILIES[kind], *self._planned_pattern_key())

    def _known_ordering(self, key):
        """(permutation, tree) kept for this key: this object's last plan, else the remember_cudss_orderings store."""
        if key is None:
            return None
        if self._ordering is not None and self._ordering[0] == key:
            return self._ordering[1:]
        if _kept_orderings is not None and key in _kept_orderings:
            _kept_orderings.move_to_end(key)
            return _kept_orderings[key]
        return None

    def _remember_ordering(self, key, ordering) -> None:
        """Keep a fresh plan's ordering on this object and, while remember_cudss_orderings is on, in the store."""
        self._ordering = (key, *ordering)
        if _kept_orderings is not None:
            _kept_orderings[key] = ordering
            _kept_orderings.move_to_end(key)
            while len(_kept_orderings) > KEPT_ORDERINGS:
                _kept_orderings.popitem(last=False)

    def _upload_values(self, host_csr, factor_host):
        """New values of the planned pattern written into the planned device buffers in place (no re-plan)."""
        import cupy as cp
        symmetric = self._diagonal is not None
        self._operator.data[...] = cp.asarray((factor_host if symmetric else host_csr).data)
        if symmetric:
            self._diagonal[...] = cp.asarray(factor_host.diagonal())
        if self._factor_matrix is not self._operator:
            self._factor_matrix.data[...] = cp.asarray(factor_host.data.astype(self._factor_dtype))

    def _factorize_and_probe(self, defer=False):
        """cuDSS factorize + probe solve (double: cuDSS refinement when it misses ACCEPT_RESIDUAL; mixed: GMRES-refined
        to MIXED_SOLVE_TOLERANCE; defer: the probe left to the first (full-accuracy) solve); _FactorRejected on a
        failed status or probe."""
        self._check_pending = False
        factorize_started = time.perf_counter()
        info = self._solver.factorize()
        self.factorize_seconds = time.perf_counter() - factorize_started
        status = int(info.info)
        kind_text = _KIND_TEXTS[self.matrix_kind]
        if status != 0:
            raise _FactorRejected(CudssSingularError(
                f'the cuDSS {self._factor_dtype.__name__} {kind_text} factorisation failed (status {status}: a zero '
                f'pivot of a singular operator); {_TO_SUPERLU}'))
        self.lu_entries = int(info.lu_nnz)
        arrays = [vector for vector in (self._rhs, self._row_scale, self._symmetrizer, self._diagonal)
                  if vector is not None]
        arrays += [array for matrix in (self._operator, self._factor_matrix)
                   for array in (matrix.data, matrix.indices, matrix.indptr)]
        arrays += list((self._device_maps or {}).values())
        # each device array once (the device plan's matrices share the index maps)
        operator_bytes = sum(int(array.nbytes) for array in {id(array): array for array in arrays}.values())
        self.device_bytes = self._permanent_bytes + operator_bytes
        probe = self._probe_vector()
        refinement = ('float32' if self.real else 'complex64') + ' + float64 GMRES refinement'
        if self.mixed_factor:
            self.description = f'cuDSS {kind_text} {refinement}'
        elif self._mixed_rejected:
            self.description = f'cuDSS {kind_text} (double: the {refinement} was rejected)'
        else:
            self.description = f'cuDSS {kind_text}'
        if self.mixed_factor:
            if defer:   # the first full-accuracy solve is the check (module header "Deferred check")
                self._check_pending = True
                return
            self.single_residual, _ = self._probe_errors(probe, refinement_steps=0)   # the low-precision factor alone
            self.probe_residual, self.probe_backward_error = self._probe_errors(probe)
            self.refinement_steps = self.last_refinements   # what a full-accuracy solve takes
            if not self._probe_accepted(MIXED_SOLVE_TOLERANCE):
                raise _FactorRejected(self._mixed_accuracy_error())
            return
        self.single_residual = None
        self.refinement_steps = int(self._solver.solution_config.ir_num_steps or 0)
        if defer:   # R3: the first solve is the check (module header "Deferred check"; ir_num_steps kept)
            self._check_pending = True
            return
        self.probe_residual, self.probe_backward_error = self._probe_errors(probe)
        if not self._probe_accepted(ACCEPT_RESIDUAL):
            self._solver.solution_config.ir_num_steps = REFINEMENT_STEPS
            self.refinement_steps = REFINEMENT_STEPS
            self.probe_residual, self.probe_backward_error = self._probe_errors(probe)
            if not self._probe_accepted(ACCEPT_RESIDUAL):
                raise _FactorRejected(self._double_accuracy_error())

    def _double_accuracy_error(self) -> CudssAccuracyError:
        """CudssAccuracyError of a double factor whose probe (or deferred check) missed the gate after refinement."""
        return CudssAccuracyError(
            f'cuDSS solves miss the relative residual {ACCEPT_RESIDUAL:g} and the backward error '
            f'{BACKWARD_ERROR_TOLERANCE:g} ({_figure(self.probe_residual)} / {_figure(self.probe_backward_error)} '
            f'after {self.refinement_steps} refinement steps); {_TO_SUPERLU}')

    def _double_solve_accepted(self, rhs, solution):
        """Deferred check of a double factor by a solve of E y = rhs: its errors become the probe values; True / False
        by the probe gate, None for a zero rhs (nothing judged). A relative residual within ACCEPT_RESIDUAL decides
        alone: probe_backward_error None, the lazy operator norm not built (R4)."""
        import cupy as cp
        rhs_norm = float(cp.linalg.norm(rhs))
        if rhs_norm == 0.0:
            return None
        residual_norm = float(cp.linalg.norm(rhs - self._product(solution)))
        self.probe_residual = residual_norm / rhs_norm
        if self.probe_residual <= ACCEPT_RESIDUAL:
            self.probe_backward_error = None
            return True
        self.probe_backward_error = residual_norm / (self.operator_norm * float(cp.linalg.norm(solution)) + rhs_norm)
        return self._probe_accepted(ACCEPT_RESIDUAL)

    def _mixed_accuracy_error(self) -> CudssAccuracyError:
        """CudssAccuracyError of a mixed factor whose probe (or deferred check) missed the gate."""
        contraction = (None if self.single_residual is None or not self.probe_residual else
                       (self.single_residual / self.probe_residual) ** (1.0 / max(self.refinement_steps, 1)))
        return CudssAccuracyError(
            f'float64 GMRES refinement of the {self._factor_dtype.__name__} cuDSS {_KIND_TEXTS[self.matrix_kind]} '
            f'factor stalled: residual {_figure(self.probe_residual)}, backward error '
            f'{_figure(self.probe_backward_error)} after {self.refinement_steps} iterations (unrefined '
            f'{_figure(self.single_residual)}, ~{_figure(contraction)} x per iteration): an ill-conditioned operator, '
            'e.g. a drive near a high-Q resonance')

    def _deferred_check_passed(self) -> bool:
        """Judge a pending check by the full-accuracy solve just made (its true equilibrated residual, the probe gate);
        its errors become the probe values. A zero rhs judges nothing: the check stays pending."""
        relative, backward, unrefined = self._last_solve_errors
        if relative is None:
            return True
        self._check_pending = False
        self.probe_residual, self.probe_backward_error = relative, backward
        self.single_residual, self.refinement_steps = unrefined, self.last_refinements
        return self._probe_accepted(MIXED_SOLVE_TOLERANCE)

    def _replace_by_double(self) -> None:
        """A deferred check missed: re-plan the double factor (_next_attempt's kind) in place on the device E (no host
        copy; sticky, as _next_attempt), probe it and tell fallback_listener(plans, plan seconds, reused orderings);
        the rejection error once nothing is left."""
        started = time.perf_counter()
        mark = self._plan_mark()
        rejected = _FactorRejected(self._mixed_accuracy_error())
        kind = self.matrix_kind
        while True:
            kind = self._next_attempt(rejected, kind)
            if kind is None:
                self.release()
                raise rejected.error from None
            self._replan_in_place(kind)
            try:
                self._factorize_and_probe()
                break
            except _FactorRejected as error:
                rejected = error
        self.n_factorizations += 1
        self.factor_seconds = time.perf_counter() - started
        self.total_factor_seconds += self.factor_seconds
        if self.fallback_listener is not None:
            plans, reused_orderings, plan_seconds = self._plans_since(mark)
            self.fallback_listener(plans, plan_seconds, reused_orderings)

    def _probe_errors(self, host_rhs, refinement_steps=None):
        """(relative residual ||c - E y|| / ||c||, normwise backward error ||c - E y|| / (||E|| ||y|| + ||c||)) of a
        solve in the equilibrated system, c = R b, y = x / R."""
        import cupy as cp
        rhs = cp.asarray(host_rhs, dtype=self.value_dtype)
        solution = self._inner_solve(rhs, refinement_steps, stall_test=True)
        rhs, solution = self._row_scale * rhs, solution / self._row_scale
        residual_norm = float(cp.linalg.norm(rhs - self._product(solution)))
        rhs_norm = float(cp.linalg.norm(rhs))
        return (residual_norm / rhs_norm,
                residual_norm / (self.operator_norm * float(cp.linalg.norm(solution)) + rhs_norm))

    def _product(self, vector):
        """E @ vector on the device: the full E, or from its lower triangle L as L v + L^T v - diag(L) v (unconjugated
        transpose: complex symmetric)."""
        if self._diagonal is None:
            return self._operator @ vector
        return self._operator @ vector + self._operator_transpose @ vector - self._diagonal * vector

    def _cudss_solve(self):
        """cuDSS solve of the rhs buffer, timed and counted."""
        started = time.perf_counter()
        solution = self._solver.solve()
        self.solve_seconds += time.perf_counter() - started
        self.n_solves += 1
        return solution

    def _single_solve(self, rhs):
        """value_dtype solution of E y = rhs by the low-precision factor (rhs scaled to unit norm: no float32 underflow;
        the norm stays on the device, a zero rhs divides by 1 and solves to 0: no host read, B8). S5d: one multi-block
        norm and one fused kernel each way (was ~10 launches of norm / where / divide / astype / multiply)."""
        import cupy as cp
        from .pcg import device_vector_dot
        rhs = cp.ascontiguousarray(rhs)
        flat = rhs.view(np.float64) if rhs.dtype == np.complex128 else rhs   # |z|^2 = re^2 + im^2
        norm_squared = device_vector_dot(flat, flat).reshape(1)
        _unit_scaled_kernel()(rhs, norm_squared, self._rhs)
        solution = cp.empty(self.n_unknowns, dtype=self.value_dtype)
        _rescaled_kernel()(self._cudss_solve(), norm_squared, solution)
        return solution

    def _inner_solve(self, rhs, refinement_steps=None, stall_test=False, backward_error=None):
        """cupy x of S x = rhs (S = D A, or A): equilibrated factor solve; a mixed factor GMRES-refines to full accuracy
        (refinement_steps None; target backward_error, None = solve_backward_error; stall_test: a probe, as is a
        pending deferred check, whose target is at most the gate's BACKWARD_ERROR_TOLERANCE) or takes that many plain
        refinement steps; a double factor's pending check is judged by this solve (any refinement_steps)."""
        equilibrated = self._row_scale * rhs
        if not self.mixed_factor:
            self._rhs[...] = equilibrated
            solution = self._cudss_solve()
            if self._check_pending:
                accepted = self._double_solve_accepted(equilibrated, solution)
                if accepted is not None:
                    self._check_pending = False
                    if not accepted and self.refinement_steps < REFINEMENT_STEPS:   # as the probe: refine, again
                        self._solver.solution_config.ir_num_steps = REFINEMENT_STEPS
                        self.refinement_steps = REFINEMENT_STEPS
                        self._rhs[...] = equilibrated
                        solution = self._cudss_solve()
                        accepted = self._double_solve_accepted(equilibrated, solution)
                    if not accepted:   # nothing left to fall back to (as refactorize: released, then raised)
                        error = self._double_accuracy_error()
                        self.release()
                        raise error
            return self._row_scale * solution
        solution = self._single_solve(equilibrated)
        self.last_refinements = 0
        if refinement_steps is None:
            checking = stall_test or self._check_pending
            target = self.solve_backward_error if backward_error is None else float(backward_error)
            if checking:   # a looser caller target must not fail the probe gate
                target = min(target, BACKWARD_ERROR_TOLERANCE)
            solution = self._gmres_refine(equilibrated, solution, stall_test=checking, backward_error=target)
            if self._check_pending and not self._deferred_check_passed():
                self.n_refinements += self.last_refinements
                self._replace_by_double()
                return self._inner_solve(rhs)
        else:
            for _ in range(int(refinement_steps)):
                solution += self._single_solve(equilibrated - self._product(solution))
                self.last_refinements += 1
        self.n_refinements += self.last_refinements
        return self._row_scale * solution

    def _gmres_refine(self, rhs, solution, stall_test=False, backward_error=MIXED_SOLVE_BACKWARD_ERROR):
        """Module gmres_refine on the device (E = _product, preconditioned by _single_solve), its iterations counted
        on in last_refinements, its errors kept as _last_solve_errors (the deferred check's gate values)."""
        import cupy as cp
        solution, self.last_refinements, self._last_solve_errors = gmres_refine(
            cp, self._product, self._single_solve, rhs, solution, lambda: self.operator_norm, stall_test=stall_test,
            backward_error=backward_error, used=self.last_refinements)
        return solution

    def solve(self, rhs, transpose=False, refinement_steps=None, backward_error=None):
        """cupy solution x (value_dtype) of A x = rhs (transpose: A^T x = rhs, unconjugated, symmetric factors only) for
        a cupy or numpy (n,) rhs; mixed: refinement_steps float64 steps, None = GMRES until MIXED_SOLVE_TOLERANCE or
        the backward error `backward_error` (None = solve_backward_error). A real factor solves a complex rhs as two
        real solves (complex128 solution)."""
        if transpose and not self.symmetric:
            raise NotImplementedError(f'the cuDSS {_KIND_TEXTS.get(self.matrix_kind, "LU")} factorisation has no '
                                      'transposed solve (only a symmetric factor of A or of a symmetrizer D A has one); '
                                      f'{_TO_SUPERLU}')
        if self._solver is None:
            raise RuntimeError('CudssFactorization.solve after release()')
        import cupy as cp
        rhs = cp.asarray(rhs)
        if self.real and rhs.dtype.kind == 'c':   # a cast would drop the imaginary part
            real_part, imaginary_part = (self.solve(part, transpose=transpose, refinement_steps=refinement_steps,
                                                    backward_error=backward_error) for part in (rhs.real, rhs.imag))
            return real_part + 1j * imaginary_part
        rhs = rhs.astype(self.value_dtype, copy=False)
        if rhs.shape != (self.n_unknowns,):
            raise ValueError(f'rhs must be a flat ({self.n_unknowns},) array, got shape {rhs.shape}')
        if self._symmetrizer is None:
            return self._inner_solve(rhs, refinement_steps, backward_error=backward_error)
        if transpose:   # A^T = S D^-1 for symmetric S = D A: x = D S^-1 b
            return self._symmetrizer * self._inner_solve(rhs, refinement_steps, backward_error=backward_error)
        return self._inner_solve(self._symmetrizer * rhs, refinement_steps, backward_error=backward_error)

    def solve_host(self, rhs, refinement_steps=None, transpose=False, backward_error=None):
        """numpy solution x (value_dtype) of A x = rhs (transpose as solve) for a numpy (n,) rhs: one transfer each way
        (mixed: refinement_steps / backward_error as solve; a real factor keeps a complex rhs complex)."""
        import cupy as cp
        rhs = np.asarray(rhs)
        if not (self.real and rhs.dtype.kind == 'c'):
            rhs = rhs.astype(self.value_dtype, copy=False)
        if rhs.shape != (self.n_unknowns,):
            raise ValueError(f'rhs must be a flat ({self.n_unknowns},) array, got shape {rhs.shape}')
        return cp.asnumpy(self.solve(rhs, transpose=transpose, refinement_steps=refinement_steps,
                                     backward_error=backward_error))

    def _track_external(self, permanent_bytes, peak_bytes) -> None:
        """Book a fresh plan in the process-wide outside-pool counters (external_live_bytes, its high-water mark)."""
        _external['high_water'] = max(_external['high_water'], _external['live'] + max(permanent_bytes, peak_bytes))
        _external['live'] += permanent_bytes
        self._tracked_bytes = permanent_bytes

    def _free_solver(self) -> None:
        """Free the planned DirectSolver (if any) and take its bytes off external_live_bytes."""
        if self._solver is not None:
            self._solver.free()
        self._solver = None
        _external['live'] = max(_external['live'] - self._tracked_bytes, 0)
        self._tracked_bytes = 0

    def release(self) -> None:
        """Free the cuDSS factor and the device operator copies."""
        self._free_solver()
        self._operator = self._factor_matrix = self._rhs = self._row_scale = self._symmetrizer = None
        self._diagonal = self._operator_transpose = self._device_maps = None

    def __del__(self):
        try:
            self.release()
        except Exception:   # interpreter shutdown: the CUDA context may already be gone
            pass
