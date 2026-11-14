"""'arpack_shift_invert' solver type runtime (role 'eigen'): SuperLU (host) or cuDSS (device) LU of the shifted operator
and scipy ARPACK (eigs, 'LM') on its inverse, symmetric shift-invert Lanczos of a host pencil (solid_mechanics), or
factor-free eigsh on a caller's inverse (matrix-free solid_mechanics_3d); counted. numpy only at module level."""
import math
import time
from typing import NamedTuple

import numpy as np

from .cudss_factor import CUDSS_REFUSALS, CudssAccuracyError, canonical_csr, csr_rows
from .host_direct import (HOST_DIRECT_FACTORIZATIONS, HOST_DIRECT_MIXED_BACK_ENDS, double_of,
                          host_direct_compressed, host_direct_factorization, is_host_direct_factor, quiet_openblas)
from .sparse_direct import LOWER_STORED_FRACTION, SparseDirectSolver, SuperLUOrdering, factorize_host

# Shift-invert across several shifts (B1): the shifted operator K - sigma M is built on one fixed pattern (ShiftPencil
# below; pattern_combination keeps explicit zeros), so its sparsity pattern never depends on sigma and a cuDSS factor passed back as `previous` is refactorised in place (plan
# reused); SuperLU refactors from scratch. Inertia (Sylvester): n_negative(K - sigma M) = eigenvalues below sigma of a
# real symmetric / Hermitian pencil with M positive definite; read from the cuDSS LDL^T pivots (real only, the
# equilibration is a congruence; mixed: float32 pivots, exact=False) or from a SuperLU factor whose pivots stayed on the
# diagonal (else one counting SuperLU with diag_pivot_thresh 0, counted as an eigen factorisation). Pencils and the
# shifted operator stay host CSR: SuperLU converts to CSC itself (_ordered_superlu). MUMPS / PARDISO factorizations
# (host_direct) count from their real LDL^T / Cholesky pivots (MUMPS INFOG(12), PARDISO iparm[21] / [22]); an LU,
# perturbed / null pivots or a Hermitian pencil get the counting SuperLU; previous= refactorises on the kept analysis;
# a block low-rank LDL^T (mumps_blr) counts with exact=False, as a float32 factor.
# ShiftPencil (H2): the terms of K - sigma M put on one canonical pattern once; per shift SuperLU gets a one-pass host
# combination of the values, a cuDSS factorization one DevicePencil (uploaded and symmetry-checked once, through a
# private sparse_direct runtime) refreshed by factorize_pencil on the device (host CSR, asymmetry check and lower take
# skipped; a refreshed factor, double or mixed, is checked by its first solve: a miss raises CudssAccuracyError from
# the ARPACK call, reworded there as the shift's factorisation failure); there the ARPACK M products run on the device
# too (the DevicePencil's M term; with a host complex M x, E5 200 nm took ~70 ms per application). The terms are K's
# own pattern (index arrays copied: cuDSS matches them by identity) when M (and B) lie inside it (FEM), else the
# merged one. Callers with several shifts keep the pencil next to the factor (modal.ModalPencil); without one, each
# call builds its own.
# Device Krylov-Schur (eigs_device, S5b; Stewart 2001): for a matvec that runs on the GPU (solve_resonances with its own
# cuDSS factor) the Arnoldi basis stays on the device as (m + 1, n) rows: classical Gram-Schmidt twice (CGS2: both passes
# always, cuBLAS gemv on the rows seen as F-order (n, k) columns, no conjugated copies; a second pass on cancellation
# only cost a host read each time, and Arnoldi on a shift-invert map cancels on almost every step: em_modes 4857 reads
# for 2429 applications), the next unit row normalised on the device, then ONE packed read per application into pinned
# memory: coefficients, |w|^2 and, around a host matvec, the next row itself (the vector the host solve needs: no
# asnumpy of its own). Per restart the host
# takes the complex Schur form of the small H[:m, :m], moves the `keep` largest |Ritz values| to the front (LAPACK trsen
# by position), tests ARPACK's criterion on the Ritz estimates |b Q y| and keeps V Q[:, :keep] (device products by
# column chunks of KRYLOV_CHUNK_BYTES, in place: no (keep, n) temporary; likewise the Ritz vectors).
# Host ARPACK streamed its n x m complex128 basis several times per step (Mie sphere, 6.5e4 unknowns, m 80 / 160: ~15
# ms of ~45 ms per application) and moved one vector each way. A widening (solve_resonances: too few non-PML
# resonances) resumes from the earlier decomposition instead of a new start vector (S5a), in place on rows the first
# pass reserved (reserve_basis; without them a second basis is copied next to the first). `eigs` itself runs it too
# (device_arnoldi_fits): on a caller's device_matvec (a cuDSS solve: device_factor_matvec), else on a gpu runtime
# (device_arnoldi, model runtimes) around the host matvec (SuperLU, line-block), one vector transfer each way, only
# from HOST_MATVEC_DEVICE_MIN_ENTRIES basis entries: host ARPACK costs 2-5 ns per basis entry and application, the device
# path a fixed cost per application that grew with the host solve between reads (the GPU idles there: the read after
# a 0.5 ms solve took ~0.15 ms, after a 30 ms one ~6.6 ms; 15:51 profiles, three reads per application then).
# Host Krylov-Schur (eigs_host, _HostKrylovSchur): the same algorithm on numpy rows for solve_resonances around a host-only
# factor, so its widening resumes too (host ARPACK restarted from a new start: the PARDISO Mie run's pass 2 at basis 160
# was the larger part of ~620 applications of 80 ms); CGS2 costs ~4 basis passes per application on the host BLAS.
# a count shift on an eigenvalue (singular factor) is retried once at sigma (1 + this)
COUNT_SHIFT_NUDGE = 1e-9
# ShiftPencils with fewer stored entries keep the host route on cuDSS too (factorize_host: host equilibration,
# asymmetry check, plan reuse by previous=; host M products): a DevicePencil's upload + device symmetry check (several
# syncs) cost tens of ms per pencil (eigen_inertia_count, 600 unknowns: 0.65 -> 2.1 s with one per call), the host
# route ~40 ns per entry per shift (E7). A guess between the two; ArpackShiftInvertSolver.device_pencil_min_entries
# overrides it per runtime (selftests: 0)
DEVICE_PENCIL_MIN_ENTRIES = 200_000


class ShiftInertia(NamedTuple):
    """Eigenvalues below sigma of a symmetric pencil (n_negative of K - sigma M); exact False for a float32 factor."""
    sigma: float
    n_below: int
    exact: bool


def pattern_combination(matrices, coefficients):
    """Host CSR sum_i c_i A_i keeping every stored entry (explicit zeros too): its pattern never depends on the
    coefficients (cuDSS plan reuse across shifts; the solvers twin of fem.assembly.pattern_sum)."""
    import scipy.sparse
    parts = [matrix.tocoo() for matrix in matrices]
    dtype = np.result_type(*(part.dtype for part in parts), *(np.asarray(value).dtype for value in coefficients))
    data = np.concatenate([part.data.astype(dtype) * coefficient for part, coefficient in zip(parts, coefficients)])
    rows = np.concatenate([part.row for part in parts])
    columns = np.concatenate([part.col for part in parts])
    combined = scipy.sparse.coo_matrix((data, (rows, columns)), shape=parts[0].shape).tocsr()
    combined.sum_duplicates()
    return combined


def _shift_factor_error(error, sigma) -> CudssAccuracyError:
    """The CudssAccuracyError of a shift factor that missed its accuracy check in an ARPACK solve (factorize_pencil
    defers a refreshed factor's check to its first solve), worded as the factorisation failure it is."""
    return CudssAccuracyError(f'the cuDSS factorisation of K - sigma M at sigma = {sigma:.6g} failed its accuracy check '
                              f'(made at its first solve, inside ARPACK): {error}')


MERGE_COLD_ENTRIES = 2_000_000   # pattern entries from which the numba row merge runs although numba is not loaded yet


def _merge_positions_host(base_indptr, base_indices, indptr, indices, positions) -> None:
    """numpy form of the row merge (its arguments): positions[e] = slot of entry e of (indptr, indices) in the base
    pattern, -1 outside it (one searchsorted of row * n + column keys; both canonical, same row count)."""
    n_columns = max(int(base_indices.max()) if base_indices.size else 0, int(indices.max()) if indices.size else 0) + 1
    base_keys = csr_rows(base_indptr) * n_columns + base_indices
    keys = csr_rows(indptr) * n_columns + indices
    if not base_keys.size:
        positions[...] = -1
        return
    found = np.minimum(np.searchsorted(base_keys, keys), base_keys.size - 1)
    positions[...] = np.where(base_keys[found] == keys, found, -1)


def _merge_positions_numba(numba):
    """Row-parallel merge of each row's sorted columns into the base row's (cost ~ both entry counts)."""
    @numba.njit(parallel=True, cache=True)
    def merge(base_indptr, base_indices, indptr, indices, positions):
        for prange_row in numba.prange(indptr.size - 1):
            row = np.int64(prange_row)
            slot = np.int64(base_indptr[row])
            end = np.int64(base_indptr[row + 1])
            for entry in range(indptr[row], indptr[row + 1]):
                column = indices[entry]
                while slot < end and base_indices[slot] < column:
                    slot += 1
                if slot < end and base_indices[slot] == column:
                    positions[entry] = slot
                else:
                    positions[entry] = -1
    return merge


def merge_positions(base_indptr, base_indices, indptr, indices, numba=None) -> np.ndarray:
    """Int64 slot in the base pattern of every stored entry of a canonical CSR pattern (same rows), -1 outside it; numba
    None: the numba twin once numba is loaded or from MERGE_COLD_ENTRIES base entries, True / False force one."""
    import sys
    from ..core.kernels import host_twin
    twin = host_twin('solvers.pattern_merge', _merge_positions_host, _merge_positions_numba)
    if numba is None:
        numba = 'numba' in sys.modules or base_indices.size >= MERGE_COLD_ENTRIES
    positions = np.empty(len(indices), dtype=np.int64)
    (twin if numba else twin.cpu)(np.asarray(base_indptr), np.asarray(base_indices), np.asarray(indptr),
                                  np.asarray(indices), positions)
    return positions


# stored mass entries from which a host ShiftPencil hands ARPACK the numba row-parallel M product instead of the scipy
# CSR (one thread; ~16 ms per product at 8e6 entries); below, scipy's call overhead is the smaller cost
MASS_PRODUCT_MIN_ENTRIES = 1_000_000


def _csr_product_host(indptr, indices, data, vector, out) -> None:
    """numpy form of the row-parallel CSR product (its arguments): out = A x by scipy."""
    import scipy.sparse
    n_rows = int(indptr.size) - 1
    out[...] = scipy.sparse.csr_matrix((data, indices, indptr), shape=(n_rows, vector.size)) @ vector


def _csr_product_numba(numba):
    """out[i] += sum_k data[k] x[indices[k]] over row i (prange over rows; out zeros on entry; real or complex)."""
    @numba.njit(parallel=True, cache=True)
    def product(indptr, indices, data, vector, out):
        for prange_row in numba.prange(indptr.size - 1):
            row = np.int64(prange_row)
            total = out[row]   # zeros from the caller: the output dtype
            for entry in range(indptr[row], indptr[row + 1]):
                total += data[entry] * vector[indices[entry]]
            out[row] = total
    return product


def _values_on_pattern(base, matrix):
    """Values of a canonical CSR on the canonical pattern of `base` (zeros elsewhere), None when one of its stored
    entries lies outside that pattern (FEM: the mass pattern lies inside the stiffness one)."""
    positions = merge_positions(base.indptr, base.indices, matrix.indptr, matrix.indices)
    if positions.size and int(positions.min()) < 0:
        return None
    values = np.zeros(base.indices.size, dtype=matrix.dtype)
    values[positions] = matrix.data
    return values


class ShiftPencil:
    """Host terms (K_1, ..., K_m, M) of K - sigma M = sum_i a_i K_i - sigma M on one canonical CSR pattern, their
    diagonals, M itself (dtype, for products) and, on a cuDSS factorization with at least device_min_entries stored
    entries, their DevicePencil (else device None: the host route); symmetric_known: the caller vouches that every term
    is (complex) symmetric, so no symmetry check is made (device or host route)."""

    def __init__(self, stiffness_terms, mass, dtype, symmetric, device_runtime=None, device_min_entries=0,
                 symmetric_known=False):
        self.dtype = np.dtype(dtype)
        self.symmetric = bool(symmetric)   # the symmetric (LDL^T) request of every shifted operator
        self.symmetric_known = self.symmetric and bool(symmetric_known)
        self.n_stiffness_terms = len(stiffness_terms)
        self.mass = mass.tocsr().astype(self.dtype, copy=False)
        self.n_rows = int(self.mass.shape[0])
        matrices = [canonical_csr(term, self.dtype) for term in stiffness_terms] + [canonical_csr(self.mass,
                                                                                                    self.dtype)]
        base = matrices[0]
        term_values = [base.data] + [_values_on_pattern(base, matrix) for matrix in matrices[1:]]
        if any(values is None for values in term_values):   # an entry outside the first term's pattern: merge them all
            terms = [pattern_combination(matrices, [float(index == position) for index in range(len(matrices))])
                     for position in range(len(matrices))]   # the same COO entries: one pattern, explicit zeros kept
            base, term_values = terms[0], [term.data for term in terms]
        # own index arrays: cuDSS matches a refreshed pattern by their identity (never modified in place), so a caller
        # changing its K afterwards cannot alias them; base.data is read, never modified
        self.indptr, self.indices = base.indptr.copy(), base.indices.copy()
        self.term_values = term_values
        rows = np.arange(self.n_rows, dtype=np.int64)
        positions = merge_positions(base.indptr, base.indices, np.arange(self.n_rows + 1, dtype=np.int64), rows)
        stored = positions >= 0
        safe = np.where(stored, positions, 0)
        self.diagonals = [np.where(stored, values[safe], 0) if values.size else np.zeros(self.n_rows, self.dtype)
                          for values in term_values]
        self.device = (None if device_runtime is None or self.indices.size < device_min_entries else
                       device_runtime.device_pencil(self.indptr, self.indices, self.term_values, symmetric=symmetric,
                                                    symmetric_known=self.symmetric_known))
        self._device_mass = None

    def drop_device(self) -> None:
        """Forget the DevicePencil (a runtime fell back to SuperLU): host route and host M products from now on."""
        self.device = self._device_mass = None

    def mass_product(self, vector, numba=None) -> np.ndarray:
        """Host M x of a flat vector by the row-parallel CSR product (host twin 'solvers.csr_product'; scipy's is one
        thread: ~16 ms per product of a 3D P2 vector mass, 116 per eigsh); numba False: the numpy twin (scipy)."""
        from ..core.kernels import host_twin
        twin = host_twin('solvers.csr_product', _csr_product_host, _csr_product_numba)
        vector = np.ascontiguousarray(np.asarray(vector).reshape(-1))
        out = np.zeros(self.n_rows, dtype=np.result_type(self.mass.dtype, vector.dtype))
        mass = self.mass
        (twin.cpu if numba is False else twin)(mass.indptr, mass.indices, mass.data, vector, out)
        return out

    def device_mass(self):
        """cupyx CSR of M on its own pattern (the host M uploaded once): a vector FEM mass stores a third of the pencil
        pattern's entries, and mirroring the lower-stored M term to the full pencil pattern (DevicePencil.full_term)
        costs six int64 arrays of that pattern while it is built."""
        if self._device_mass is None:
            import cupyx.scipy.sparse
            self._device_mass = cupyx.scipy.sparse.csr_matrix(self.mass)
        return self._device_mass

    def mass_operator(self):
        """M for scipy: the host CSR (from MASS_PRODUCT_MIN_ENTRIES stored entries with numba a LinearOperator of the
        row-parallel product, mass_product), or on a DevicePencil a LinearOperator whose product runs on the device."""
        if self.device is None:
            from ..core.kernels import numba_available
            if self.mass.nnz < MASS_PRODUCT_MIN_ENTRIES or not numba_available():
                return self.mass
            import scipy.sparse.linalg
            return scipy.sparse.linalg.LinearOperator((self.n_rows, self.n_rows), dtype=self.dtype,
                                                      matvec=self.mass_product)
        import cupy as cp
        import scipy.sparse.linalg
        device_mass = self.device_mass()
        return scipy.sparse.linalg.LinearOperator((self.n_rows, self.n_rows), dtype=self.dtype,
                                                  matvec=lambda vector: cp.asnumpy(device_mass @ cp.asarray(vector)))

    def coefficients(self, stiffness_coefficients, sigma) -> tuple:
        """(a_1, ..., a_m, -sigma); stiffness_coefficients None = every a_i 1."""
        if stiffness_coefficients is None:
            stiffness_coefficients = (1.0,) * self.n_stiffness_terms
        if len(stiffness_coefficients) != self.n_stiffness_terms:
            raise ValueError(f'ShiftPencil: {len(stiffness_coefficients)} stiffness coefficients for '
                             f'{self.n_stiffness_terms} terms')
        return (*stiffness_coefficients, -float(sigma))

    def operator(self, coefficients):
        """Host CSR of sum_k c_k X_k on the shared pattern (its index arrays shared, never modified)."""
        import scipy.sparse
        dtype = np.result_type(self.dtype, *(np.asarray(coefficient).dtype for coefficient in coefficients))
        values = np.zeros(self.indices.size, dtype=dtype)
        for coefficient, term in zip(coefficients, self.term_values):
            if coefficient != 0:
                values += coefficient * term
        operator = scipy.sparse.csr_matrix((values, self.indices, self.indptr), shape=(self.n_rows, self.n_rows))
        operator.has_canonical_format = True
        return operator

    def row_scale(self, coefficients):
        """cuDSS equilibration (|diag sum a_i K_i| + |sigma| |diag M|)^-1/2 (cannot cancel, unlike |diag(K - sigma M)|)."""
        stiffness_diagonal = sum(coefficient * diagonal
                                 for coefficient, diagonal in zip(coefficients[:-1], self.diagonals[:-1]))
        pencil_diagonal = np.abs(stiffness_diagonal) + abs(coefficients[-1]) * np.abs(self.diagonals[-1])
        return 1.0 / np.sqrt(np.where(pencil_diagonal > 0.0, pencil_diagonal, 1.0))

# factorize_host: host SuperLU, cuDSS on the GPU (solvers.cudss_factor) in double precision, or its complex64 factor with
# float64 refinement (only roles whose setting_choices accept it: callers that ask for loose solves, em_fdfd_3d).
# 'auto' (B1a): the role's auto_back_ends[grid] on the gpu with nvmath-python (schema.auto_back_end_values, marked by
# sparse_direct.AUTO_BACK_END_KEY: the runtime then falls back to SuperLU on a cuDSS refusal, B1b), else SuperLU
# Host direct factorizations (solvers.host_direct: MUMPS / MKL PARDISO, + _mixed): opt-in like cuDSS, never 'auto', on
# both backends; factorize_host returns a HostDirectFactorization (host solves, LDL^T of a symmetric request, inertia
# from the library's pivots); a refusal raises (no fallback); *_mixed below mixed_min_unknowns factors in double (S5c).
FACTORIZATIONS = ('auto', 'superlu', 'cudss', 'cudss_mixed') + HOST_DIRECT_FACTORIZATIONS
CUDSS_FACTORIZATIONS = ('cudss', 'cudss_mixed')
LDLT_FACTORIZATIONS = CUDSS_FACTORIZATIONS + HOST_DIRECT_FACTORIZATIONS   # an own factor honouring symmetric /
                                                                         # symmetrizer / row_scale (SuperLU: LU)
MIXED_FACTORIZATIONS = ('cudss_mixed',) + HOST_DIRECT_MIXED_BACK_ENDS   # a float32 / complex64 factor + refinement
SCIPY_DEFAULT_BASIS = 20   # scipy eigs ncv default: min(n, max(2 k + 1, 20))
# Arnoldi / Lanczos restarts of eigs / eigsh_shifted: converging solves need tens; scipy's own default 10 n let a solve
# whose rounding noise exceeds the tolerance (a shift on a resonance) run uninterruptibly for minutes
MAX_RESTARTS = 300
# eigsh_shifted factors and runs eigsh at sigma = shift (1 + this): a target exactly on a resonance would give a
# (near-)singular factor; 1e-4 bounds the rounding amplification along it (as physics/em/conical). A zero shift stays 0:
# no offset scale is safe there (slender solids have lambda_1 / lambda_max ~ 1e-10), and the rigid checks keep K SPD.
EIGSH_SHIFT_RELATIVE_OFFSET = 1e-4
# run-time estimate guess: operator applications per eigen solve = this x the Arnoldi basis (shift-invert near the
# wanted eigenvalues converges in a few restarts; calibrate from summary operator_applications / eigen_solves)
APPLICATIONS_PER_BASIS_GUESS = 2.5
# a real symmetric pencil (eigsh, Lanczos) near the shift converges on its first pass: ~ basis + 1 applications
# (om_bor_sphere_expansion solid: 21 at scipy's default basis 20)
LANCZOS_APPLICATIONS_PER_BASIS = 1.1
# cuDSS run-time recipe (3D curl-curl only), fitted on an RTX 3050 8 GiB (fused 206 GB/s, gemm 116 GFLOP/s) to
# cudss_mixed LDL^T runs of the eighth nanobeam (1.4e5-5.2e5 unknowns, graded) and the Mie sphere (2.2e5, uniform):
# factor + operator device bytes ~ CUDSS_BYTES_PER_UNKNOWN n (n / CUDSS_REFERENCE_UNKNOWNS)^CUDSS_BYTES_GROWTH (within
# 5 %, the Mie +25 %); one complex64 solve streams CUDSS_SOLVE_READS x those bytes (L twice + the operator: 8-45 ms,
# within 8 %); factorise ~ gemm-equivalent CUDSS_FACTOR_FLOPS (n / reference)^CUDSS_FACTOR_GROWTH (0.45-3.1 s, within
# 6 %); the host plan (METIS + symbolic) ~ CUDSS_PLAN_HOST_BYTES host-stream bytes per unknown (2.2-6.8 s, +-30 %).
# Double precision (Mie, LDL^T): 1.84 x the bytes, 12.3 x the factorise time.
CUDSS_REFERENCE_UNKNOWNS = 2.64e5
CUDSS_BYTES_PER_UNKNOWN = 6089.0
CUDSS_BYTES_GROWTH = 0.24
CUDSS_SOLVE_READS = 2.5
CUDSS_FACTOR_FLOPS = 1.41e11
CUDSS_FACTOR_GROWTH = 1.45
CUDSS_PLAN_HOST_BYTES = 2.0e5
CUDSS_DOUBLE_BYTES = 1.84
CUDSS_DOUBLE_FACTOR_TIME = 12.3
# a real factor (solid_mechanics_3d LDL^T) over the complex fit: half the value bytes, a quarter of the multiply-adds
# (uncalibrated; first piezo LN run, 5.9e5 unknowns: factorise 17.2 s)
CUDSS_REAL_BYTES = 0.5
CUDSS_REAL_FACTOR_TIME = 0.25
# complex64 solves per solve of cudss_mixed in the recipe: the eigenfrequency polish path (1 refinement step: 2)
CUDSS_MIXED_SOLVES_PER_SOLVE = 2.0
# S5c: a cudss_mixed request of a model runtime (definition._create_arpack) factors operators below this many unknowns
# in double. A mixed solve reads the bytes of a double one (complex64 solve + refinement solve) plus ~1-6 ms of
# refinement launches / syncs (GMRES-IR), so mixed pays only through its cheaper factorisation: em3d_fdfd_bragg_cell
# (468 unknowns, LU) took 14 ms per application, its cuDSS solves 0.25 ms. From the CUDSS_* fit, at 2e4 3D unknowns
# the double factorisation costs ~0.3 s more and ~600 solves save ~0.9 s; the Mie sphere (6.5e4) still gains from
# mixed (~3 s less factorisation). A guess between the two (uncalibrated); script runtimes keep 0 (mixed as asked).
MIXED_EIGEN_MIN_UNKNOWNS = 20_000
# S5b: device Krylov-Schur (eigs_device): the smallest unknowns / basis ratio it runs at (else host ARPACK)
DEVICE_ARNOLDI_MIN_RATIO = 4
# eigs around a host matvec: the device Krylov-Schur from this many basis entries n (m + 1). 15:51 profiles (three reads
# per application, host SuperLU), device vs host ARPACK ms per application: em_fdtd_3d ports 2.2e5 1.8 vs 1.05,
# em_eme_3d 2.4e5 2.2 vs 0.48, em_modes 5.9e5 5.2 vs 3.0, em_modes triangles 1.45e6 27 vs 7.4 (30 ms solves),
# heat_noise 1.6e6 3.6 vs 5.9: only the last gained. Refit after a run of the one-read path (it may move far down)
HOST_MATVEC_DEVICE_MIN_ENTRIES = 1_500_000
_BREAKDOWN_SEED = 17   # a random direction after an invariant Krylov subspace (deterministic)
KRYLOV_CHUNK_BYTES = 8 * 2 ** 20   # device scratch of one column chunk of the restart / Ritz products (_column_chunks)
# E6: BLAS threads of a host Arnoldi run (host ARPACK, host Krylov-Schur) on at most HOST_ARPACK_BLAS_MAX_UNKNOWNS
# unknowns; 0 = the libraries' default (off). Threaded BLAS on ~1e4-long vectors may spin more than it computes
# (em_modes ArpackShiftInvertSolver.eigs exclusive 0.98 s at cpu / wall 1.71 on 9.4e3 unknowns). Unmeasured: set 1 only
# if the MKL_NUM_THREADS=1 example_profiler runs of em_modes and em3d_fdtd (--no-cprofile) beat the default runs
HOST_ARPACK_BLAS_THREADS = 0
HOST_ARPACK_BLAS_MAX_UNKNOWNS = 200_000
# E5: share (by norm) of a seeded random vector added to a warm ARPACK start (earlier modes as v0), so an eigenvector
# the warm start lacks still has a component (missed-mode guard)
WARM_START_RANDOM_SHARE = 0.1


def host_blas_limit(n_unknowns):
    """Context of a host Arnoldi run on n_unknowns (E6): every BLAS pool at HOST_ARPACK_BLAS_THREADS while that is > 0
    and n_unknowns <= HOST_ARPACK_BLAS_MAX_UNKNOWNS (host_direct.limited_blas_threads), else nothing changes."""
    from .host_direct import limited_blas_threads
    threads = HOST_ARPACK_BLAS_THREADS if int(n_unknowns) <= HOST_ARPACK_BLAS_MAX_UNKNOWNS else 0
    return limited_blas_threads(threads)


def guarded_start_vector(warm, seed, real=False):
    """ARPACK v0 of a warm start (E5): warm + WARM_START_RANDOM_SHARE ||warm|| r / ||r|| (r seeded normal, float64 when
    real, else complex128); None for a None or vanishing warm vector."""
    if warm is None:
        return None
    warm = np.asarray(warm)
    norm = float(np.linalg.norm(warm))
    if not norm > 0.0:
        return None
    random = np.random.default_rng(seed)
    noise = (random.standard_normal(warm.size) if real
             else random.standard_normal(warm.size) + 1j * random.standard_normal(warm.size))
    return warm.ravel() + (WARM_START_RANDOM_SHARE * norm / float(np.linalg.norm(noise))) * noise


def device_arnoldi_fits(n_unknowns, basis_size, host_matvec=False) -> bool:
    """True when eigs runs the device Krylov-Schur for a basis of basis_size (CPU, estimators too): n_unknowns >=
    DEVICE_ARNOLDI_MIN_RATIO x (basis + 1), around a host matvec also n (basis + 1) >= HOST_MATVEC_DEVICE_MIN_ENTRIES."""
    rows = int(basis_size) + 1
    if int(n_unknowns) < DEVICE_ARNOLDI_MIN_RATIO * rows:
        return False
    return not host_matvec or float(n_unknowns) * rows >= HOST_MATVEC_DEVICE_MIN_ENTRIES


def arnoldi_basis_bytes(n_unknowns, basis_size, backend, device_matvec=False, real=False):
    """(on_device, bytes) of an eigs caller's Arnoldi basis (memory estimators): the device Krylov-Schur's (basis + 1)
    complex128 rows + its product and packed-read vectors on the gpu backend when device_arnoldi_fits (device_matvec: a
    cuDSS solve), else ARPACK's host basis (real: float64, eigs(real=True))."""
    on_device = backend == 'gpu' and device_arnoldi_fits(n_unknowns, basis_size, host_matvec=not device_matvec)
    rows = int(basis_size) + (3 if on_device else 0)
    return on_device, rows * (8 if real and not on_device else 16) * int(n_unknowns)


def device_factor_matvec(factorization, mass, **solve_options):
    """cupy matvec x -> factorization.solve(B x, **solve_options) of a cuDSS factor (B host scipy, uploaded as complex128
    on first use) for eigs(device_matvec=), None for any other factor (SuperLU, line-block: host solves only)."""
    if getattr(factorization, 'back_end', None) != 'cudss':
        return None
    device_mass = []

    def apply(vector):
        if not device_mass:
            import cupyx.scipy.sparse
            device_mass.append(cupyx.scipy.sparse.csr_matrix(mass.tocsr().astype(np.complex128)))
        return factorization.solve(device_mass[0] @ vector, **solve_options)
    return apply


def arnoldi_basis_size(n_wanted, n_unknowns, subspace_size=0, default_basis=0) -> int:
    """Arnoldi vectors ARPACK keeps for n_wanted eigenpairs (CPU, memory estimators): the subspace_size setting raised
    to n_wanted + 2, else the caller's default_basis (likewise), else scipy's default; at most n_unknowns."""
    if subspace_size or default_basis:
        return int(min(n_unknowns, max(int(subspace_size or default_basis), int(n_wanted) + 2)))
    return int(min(n_unknowns, max(2 * int(n_wanted) + 1, SCIPY_DEFAULT_BASIS)))


def backend_misfit(values, backend) -> str:
    """SolverType.backend_misfit of arpack_shift_invert: '' unless a cuDSS factorization is asked of the cpu backend."""
    factorization = values.get('factorization')
    if backend == 'cpu' and factorization in CUDSS_FACTORIZATIONS:
        return f"factorization '{factorization}' runs on the GPU only (use 'superlu' on the CPU backend)"
    return ''


class ArpackShiftInvertSolver:
    """EigenSolver of 'arpack_shift_invert': factorize_host(A - sigma B) and eigs(matvec of (A - sigma B)^-1 B) with the
    node's tolerance (0 = machine precision), subspace size (0 = scipy's default) and factorization (FACTORIZATIONS; an
    unresolved 'auto' = superlu); keeps no factor itself. fallback_factorization ('superlu' for a cuDSS value resolved
    from 'auto'): a cuDSS refusal switches the runtime to it for good (B1b, _fall_back). device_arnoldi (a gpu model
    runtime): eigs runs the device Krylov-Schur around host matvecs too (device_arnoldi_fits)."""

    def __init__(self, tolerance=0.0, subspace_size=0, label=None, factorization='superlu',
                 fallback_factorization=None, mixed_min_unknowns=0, device_arnoldi=False):
        if not (float(tolerance) >= 0.0 and int(subspace_size) >= 0):
            raise ValueError(f'need tolerance >= 0 and subspace_size >= 0, got {tolerance!r}, {subspace_size!r}')
        if factorization not in FACTORIZATIONS:
            raise ValueError(f"factorization must be one of {', '.join(FACTORIZATIONS)}, got {factorization!r}")
        self.tolerance = float(tolerance)
        self.subspace_size = int(subspace_size)
        self.factorization = 'superlu' if factorization == 'auto' else factorization
        self.fallback_factorization = fallback_factorization
        self.n_auto_fallbacks = 0   # B1b: a cuDSS factorization resolved from 'auto' switched to SuperLU (0 / 1)
        self.auto_fallback = ''     # its warning text
        # A4: None, or a caller's hook n_unknowns -> '' (fall back) or the reason the SuperLU fallback is refused (e.g.
        # em_fdfd_3d tetrahedra: the counted Nedelec fill); asked only when a fallback would happen
        self.fallback_refusal = None
        self.label = label if label is not None else (lambda: 'arpack_shift_invert')
        self.n_factorizations = 0
        self.n_eigen_solves = 0
        self.n_unconverged = 0
        self.operator_applications = 0
        self.total_eigen_seconds = 0.0
        self.total_factor_seconds = 0.0
        self.host_bytes = 0     # L + U of the last SuperLU factorisation (the consumer keeps the factor)
        self.device_bytes = 0   # cuDSS factor + operator copy of the last cudss factorisation
        self.permanent_bytes = 0   # cuDSS permanent device memory alone (outside the pool) of the last cudss factor and
        self.lu_entries = 0        # its factor entries: the data of cudss_factor.factor_device_bytes' fit
        self.refinement_steps = 0   # cuDSS iterative refinement steps per solve of the last cudss factorisation
        self.symmetric_factor = False   # the last cudss factorisation is an LDL^T
        self.double_fallbacks = 0   # cudss_mixed factors rejected by their probe and replaced by double ones
        self.n_cudss_plans = 0      # cuDSS analyses of factorize_host (the pencil runtime counts its own)
        self.n_cudss_reused_orderings = 0   # of those, plans on a kept ordering (cudss_factor "Ordering reuse")
        self.total_cudss_plan_seconds = 0.0
        self.n_host_direct_analyses = 0   # MUMPS / PARDISO analyses (fallback re-analyses included)
        self.compression_fallbacks = 0    # mumps_blr factors refactored full rank after a miss
        self.host_direct_threads = None   # the last host direct factor's library thread count
        self.orderings = SuperLUOrdering()   # SuperLU elimination order of the last pattern (shifts of one pencil)
        self._pencil_runtime = None   # private SparseDirectSolver of the cuDSS ShiftPencils (factorize_pencil)
        self.device_pencil_min_entries = DEVICE_PENCIL_MIN_ENTRIES
        self.mixed_min_unknowns = int(mixed_min_unknowns)   # S5c: cudss_mixed below this factors in double
        self.device_arnoldi = bool(device_arnoldi)
        self.last_arnoldi = ''   # 'ARPACK' ('ARPACK (real)'), 'device Krylov-Schur' or 'host Krylov-Schur' (log texts)

    def factorization_for(self, n_unknowns) -> str:
        """The factorization used for an operator of n_unknowns: cudss for a cudss_mixed request below
        mixed_min_unknowns (S5c; a host *_mixed request likewise its double name), else the runtime's own."""
        if self.factorization in MIXED_FACTORIZATIONS and int(n_unknowns) < self.mixed_min_unknowns:
            return 'cudss' if self.factorization == 'cudss_mixed' else double_of(self.factorization)
        return self.factorization

    def shift_pencil(self, stiffness_terms, mass, dtype, symmetric, symmetric_known=False) -> ShiftPencil:
        """ShiftPencil of K = sum_i a_i K_i (stiffness_terms) and M in dtype; symmetric: LDL^T requested (cuDSS: a
        DevicePencil on the private runtime, symmetry checked once on the device, or never with symmetric_known: the
        caller vouches that every term is symmetric by construction). A pencil the S5c rule factors in double takes
        the host route (factorize_host does the switch)."""
        if self.factorization in CUDSS_FACTORIZATIONS and self._pencil_runtime is None:
            self._pencil_runtime = SparseDirectSolver(back_end=self.factorization, label=self.label, backend='gpu')
        device = (self.factorization in CUDSS_FACTORIZATIONS
                  and self.factorization_for(mass.shape[0]) == self.factorization)
        runtime = self._pencil_runtime if device else None
        return ShiftPencil(stiffness_terms, mass, dtype, symmetric, device_runtime=runtime,
                           device_min_entries=self.device_pencil_min_entries, symmetric_known=symmetric_known)

    def _fall_back(self, error, n_unknowns, pencil=None) -> bool:
        """B1b: switch a runtime resolved from 'auto' (fallback_factorization set) from cuDSS to host SuperLU for good
        after a cuDSS refusal (cudss_factor.CUDSS_REFUSALS: zero pivot, accuracy, device memory) of an operator within
        SPLU_LIMIT_UNKNOWNS; warned, counted, the pencil's DevicePencil dropped. True when it switched (the caller redoes
        its factorisation / shift on SuperLU), else False (re-raise). RuntimeError (the remedy, from the refusal) when
        the fallback_refusal hook refuses SuperLU for this operator (A4)."""
        import warnings
        from .sparse_direct import SPLU_LIMIT_UNKNOWNS
        if (self.fallback_factorization is None or self.factorization not in CUDSS_FACTORIZATIONS
                or not isinstance(error, CUDSS_REFUSALS) or int(n_unknowns) > SPLU_LIMIT_UNKNOWNS):
            return False
        refusal = '' if self.fallback_refusal is None else self.fallback_refusal(int(n_unknowns))
        if refusal:
            raise RuntimeError(f"{self.label()}: factorization 'auto' resolved to '{self.factorization}', which refused "
                               f'the shifted operator ({error}); no host SuperLU fallback: {refusal}') from error
        self.auto_fallback =(f"{self.label()}: factorization 'auto' resolved to '{self.factorization}', which refused "
                              f"the shifted operator ({error}); host SuperLU ('{self.fallback_factorization}') from "
                              'now on')
        warnings.warn(self.auto_fallback, RuntimeWarning, stacklevel=2)
        self.factorization = self.fallback_factorization
        self.n_auto_fallbacks += 1
        if pencil is not None:
            pencil.drop_device()
        return True

    def _checked_pencil(self, pencil, stiffness_terms, mass, dtype, symmetric) -> ShiftPencil:
        """The caller's pencil (ValueError when its size, dtype or symmetry request differ), else a new one."""
        if pencil is None:
            return self.shift_pencil(stiffness_terms, mass, dtype, symmetric)
        if pencil.n_rows != int(mass.shape[0]) or pencil.dtype != np.dtype(dtype) or pencil.symmetric != bool(symmetric):
            raise ValueError(f'ShiftPencil of {pencil.n_rows} unknowns, {pencil.dtype}, symmetric {pencil.symmetric} does '
                             f'not fit a {mass.shape[0]}-unknown {np.dtype(dtype)} pencil, symmetric {bool(symmetric)}')
        return pencil

    def _factorize_shift(self, pencil, coefficients, row_scale=None, previous=None, defer_check=True):
        """(factor, host operator or None) of sum_k c_k X_k of a ShiftPencil, counted: factorize_pencil on its
        DevicePencil (row_scale None = pencil.row_scale; previous refreshed on the device, its probe deferred to the
        first solve unless defer_check is False), else host SuperLU of pencil.operator."""
        start = time.perf_counter()   # eigen_factor_seconds include the combination and the row scale
        real = pencil.dtype.kind != 'c' and all(not np.iscomplexobj(coefficient) for coefficient in coefficients)
        if row_scale is None and self.factorization in LDLT_FACTORIZATIONS:
            row_scale = pencil.row_scale(coefficients)
        if pencil.device is None:
            operator = pencil.operator(coefficients)   # canonical CSR: cuDSS may skip its asymmetry check when vouched
            self.total_factor_seconds += time.perf_counter() - start
            return self.factorize_host(operator, real=real, symmetric=pencil.symmetric, row_scale=row_scale,
                                       previous=previous, symmetric_known=pencil.symmetric_known), operator
        try:
            factor = self._pencil_runtime.factorize_pencil(pencil.device, coefficients, row_scale=row_scale,
                                                           previous=previous, defer_check=defer_check)
        except CUDSS_REFUSALS as error:
            if not self._fall_back(error, pencil.n_rows, pencil):
                raise
            self.total_factor_seconds += time.perf_counter() - start
            return self._factorize_shift(pencil, coefficients)   # the host route (SuperLU) now
        self.device_bytes = factor.device_bytes
        self.permanent_bytes, self.lu_entries = factor.permanent_bytes, factor.lu_entries
        self.refinement_steps = factor.refinement_steps
        self.symmetric_factor = factor.symmetric
        self.total_factor_seconds += time.perf_counter() - start
        self.n_factorizations += 1
        self.host_bytes = 0
        return factor, None

    def factorize_host(self, matrix, real=False, ordering=None, symmetric=False, row_scale=None, previous=None,
                       symmetric_known=False):
        """Factorization (solve_host; float64 with real) of the shifted host scipy operator, counted: host SuperLU with
        the caller's elimination ordering (None = minimum degree), or cuDSS on the GPU (its own ordering; symmetric:
        LDL^T of a (complex) symmetric matrix once its symmetry check passes; row_scale: the cuDSS equilibration, None =
        |diag|^-1/2; previous: a cuDSS factor refactorised in place, plan reused for an unchanged pattern;
        symmetric_known: a canonical CSR vouched symmetric, no canonicalising copy or asymmetry check; SuperLU ignores
        all four). A runtime resolved from 'auto' falls back to SuperLU on a cuDSS refusal (_fall_back)."""
        start = time.perf_counter()
        factor = None
        if self.factorization in HOST_DIRECT_FACTORIZATIONS:   # MUMPS / PARDISO: refusals raise (no fallback)
            factor = host_direct_factorization(matrix, self.factorization_for(matrix.shape[0]), previous=previous,
                                               real=real, symmetric=symmetric, row_scale=row_scale,
                                               symmetric_known=symmetric_known and symmetric)
            factor.host_arrays = True
            self.symmetric_factor = factor.symmetric
            self.refinement_steps = factor.refinement_steps
            self.lu_entries = factor.lu_entries
            self.double_fallbacks += int(factor.last_double_fallback)
            self.compression_fallbacks += int(factor.last_compression_fallback)
            self.n_host_direct_analyses += factor.last_analyses
            self.host_direct_threads = factor.threads
        elif self.factorization in CUDSS_FACTORIZATIONS:
            from .sparse_direct import cudss_factorization
            try:
                factor = cudss_factorization(matrix, self.factorization_for(matrix.shape[0]), previous=previous,
                                             real=real, symmetric=symmetric, row_scale=row_scale,
                                             symmetric_known=symmetric_known and symmetric)
            except CUDSS_REFUSALS as error:
                if not self._fall_back(error, matrix.shape[0]):
                    raise
            else:
                self.device_bytes = factor.device_bytes
                self.permanent_bytes, self.lu_entries = factor.permanent_bytes, factor.lu_entries
                self.refinement_steps = factor.refinement_steps
                self.symmetric_factor = factor.symmetric
                self.double_fallbacks += int(factor.last_double_fallback)
                self.n_cudss_plans += int(not factor.last_plan_reused)
                self.n_cudss_reused_orderings += factor.last_reused_orderings
                self.total_cudss_plan_seconds += factor.last_plans_seconds
        if factor is None:
            factor = factorize_host(matrix, real=real, ordering=ordering, orderings=self.orderings)
        self.total_factor_seconds += time.perf_counter() - start
        self.n_factorizations += 1
        self.host_bytes = factor.host_bytes or 0
        return factor

    def _shift_inertia(self, factor, operator, sigma):
        """ShiftInertia of a factor of the symmetric / Hermitian K - sigma M (module header), None when uncountable
        (cuDSS general LU or complex LDL^T); a SuperLU with off-diagonal pivots and a MUMPS / PARDISO factor without a
        count (LU, perturbed or null pivots) get one counting SuperLU factorisation."""
        counts = factor.inertia()
        # what is factored, not the request; a block low-rank LDL^T (mumps_blr) perturbs its pivots like float32 does
        exact = (getattr(factor, 'factor_precision', 'double') == 'double'
                 and not getattr(factor, 'compressed_factor', False))
        if counts is None and (getattr(factor, 'back_end', '') == 'scipy_splu' or is_host_direct_factor(factor)):
            if is_host_direct_factor(factor):
                exact = True   # the counting SuperLU is double
            from .sparse_direct import superlu_counting_inertia
            start = time.perf_counter()
            counts = superlu_counting_inertia(operator)
            self.total_factor_seconds += time.perf_counter() - start
            self.n_factorizations += 1
        if counts is None:
            return None
        return ShiftInertia(sigma=float(sigma), n_below=int(counts[1]), exact=exact)

    def count_below(self, stiffness, mass, sigma, previous=None, pencil=None):
        """(ShiftInertia or None, factor) of a real symmetric / complex Hermitian pencil (M positive definite) at sigma
        itself (no offset; a singular factor retried once at sigma (1 + COUNT_SHIFT_NUDGE)); previous as factorize_host;
        pencil: this pencil's ShiftPencil (None: one is built). A Hermitian pencil on cuDSS (general LU, no count):
        None, nothing factorised (factor = previous); a runtime resolved from 'auto' counts it by one counting host
        SuperLU instead (factor = previous), so band searches stay verified; on a host direct factorization (MUMPS /
        PARDISO) a Hermitian pencil is always counted by that counting SuperLU (factor = previous)."""
        hermitian = (pencil.dtype.kind == 'c' if pencil is not None else
                     np.iscomplexobj(stiffness.data) or np.iscomplexobj(mass.data))
        if hermitian and self.factorization in CUDSS_FACTORIZATIONS and self.fallback_factorization is None:
            return None, previous
        pencil = self._checked_pencil(pencil, (stiffness,), mass, np.complex128 if hermitian else np.float64,
                                      symmetric=not hermitian)
        sigma = float(sigma)
        if hermitian and self.factorization in CUDSS_FACTORIZATIONS + HOST_DIRECT_FACTORIZATIONS:
            return self._superlu_count(pencil, sigma), previous   # a host direct LU has no count: SuperLU counts
        for attempt in range(2):
            try:
                # probed now: a count is never solved with, so a deferred check would never judge the factor whose
                # (perturbed) pivots it reads
                factor, operator = self._factorize_shift(pencil, pencil.coefficients(None, sigma), previous=previous,
                                                         defer_check=False)
            except RuntimeError:
                if attempt:
                    raise
                previous = None
                sigma = sigma * (1.0 + COUNT_SHIFT_NUDGE) if sigma else COUNT_SHIFT_NUDGE
                continue
            return self._shift_inertia(factor, operator, sigma), factor
        raise AssertionError('unreachable')

    def _superlu_count(self, pencil, sigma):
        """ShiftInertia (or None: a zero diagonal forced a pivot swap) of a Hermitian ShiftPencil at sigma by one counting
        host SuperLU (counted as an eigen factorisation; a singular one retried once at sigma (1 + COUNT_SHIFT_NUDGE))."""
        from .sparse_direct import superlu_counting_inertia
        for attempt in range(2):
            start = time.perf_counter()
            try:
                counts = superlu_counting_inertia(pencil.operator(pencil.coefficients(None, sigma)))
            except RuntimeError:
                if attempt:
                    raise
                sigma = sigma * (1.0 + COUNT_SHIFT_NUDGE) if sigma else COUNT_SHIFT_NUDGE
                continue
            finally:
                self.total_factor_seconds += time.perf_counter() - start
            self.n_factorizations += 1
            return None if counts is None else ShiftInertia(sigma=float(sigma), n_below=int(counts[1]), exact=True)
        raise AssertionError('unreachable')

    def eigsh_shifted(self, stiffness, mass, shift, n_wanted, start_vector=None, row_scale=None, previous=None,
                      inertia=False, pencil=None):
        """(lambda, vectors, applications, converged, factor) of the n_wanted eigenpairs K x = lambda M x nearest the real
        shift of a real symmetric (or complex Hermitian) host scipy pencil, M positive definite: counted factor of K -
        sigma M (its ShiftPencil `pencil`, None: one is built; float64 for a real pencil, sigma = shift (1 +
        EIGSH_SHIFT_RELATIVE_OFFSET); cuDSS equilibration row_scale, None = ShiftPencil.row_scale; previous: a factor to
        refactorise, plan reused) and scipy eigsh shift-invert about the same sigma; RuntimeError when that factor is
        singular. inertia=True appends a sixth item: ShiftInertia at that sigma, or None (uncountable)."""
        import scipy.sparse.linalg
        hermitian = np.issubdtype(stiffness.dtype, np.complexfloating) or np.issubdtype(mass.dtype, np.complexfloating)
        dtype = np.complex128 if hermitian else np.float64
        stiffness = stiffness.tocsr().astype(dtype, copy=False)
        pencil = self._checked_pencil(pencil, (stiffness,), mass, dtype, symmetric=not hermitian)
        mass = pencil.mass_operator()   # cuDSS: products on the device
        sigma = float(shift) * (1.0 + EIGSH_SHIFT_RELATIVE_OFFSET)
        factor, operator = self._factorize_shift(pencil, pencil.coefficients(None, sigma), row_scale, previous)
        shift_inertia = self._shift_inertia(factor, operator, sigma) if inertia else None
        applications = [0]

        def shifted_inverse(vector):
            applications[0] += 1
            return factor.solve_host(vector)

        n_unknowns = int(stiffness.shape[0])
        inverse = scipy.sparse.linalg.LinearOperator((n_unknowns, n_unknowns), matvec=shifted_inverse, dtype=dtype)
        ncv = arnoldi_basis_size(n_wanted, n_unknowns, self.subspace_size) if self.subspace_size else None
        start_vector = None if start_vector is None else np.asarray(start_vector, dtype=dtype)
        start = time.perf_counter()
        converged = True
        try:   # a complex Hermitian pencil goes through scipy eigs (mode 3, M inner product) inside eigsh
            with quiet_openblas(self.factorization):   # ARPACK's BLAS between the PARDISO solves (host_direct header)
                eigenvalues, vectors = scipy.sparse.linalg.eigsh(stiffness, k=n_wanted, M=mass, sigma=sigma,
                                                                 which='LM', OPinv=inverse, v0=start_vector, ncv=ncv,
                                                                 tol=self.tolerance, maxiter=MAX_RESTARTS)
        except scipy.sparse.linalg.ArpackNoConvergence as exc:
            eigenvalues, vectors = exc.eigenvalues, exc.eigenvectors
            converged = False
            self.n_unconverged += 1
        except CudssAccuracyError as exc:   # a refreshed factor's deferred check missed in its first solve
            if self._fall_back(exc, n_unknowns, pencil):   # 'auto': this shift again, on SuperLU
                return self.eigsh_shifted(stiffness, pencil.mass, shift, n_wanted, start_vector=start_vector,
                                          inertia=inertia, pencil=pencil)
            raise _shift_factor_error(exc, sigma) from None
        self.total_eigen_seconds += time.perf_counter() - start
        self.n_eigen_solves += 1
        self.operator_applications += applications[0]
        result = (np.real(eigenvalues), vectors, applications[0], converged, factor)
        return result + (shift_inertia,) if inertia else result

    def eigs_shifted(self, stiffness, mass, shift, n_wanted, start_vector=None, row_scale=None, previous=None,
                     pencil=None, stiffness_coefficients=None):
        """(lambda complex, vectors, applications, converged, factor) of the n_wanted eigenpairs K x = lambda M x nearest
        the real shift of a complex symmetric (not Hermitian) host pencil, M symmetric (real semi-definite, or complex:
        an elastic PML):
        counted complex symmetric factor of K - sigma M (sigma, pencil and previous as eigsh_shifted; with a pencil of
        several stiffness terms K = sum a_i K_i, a_i = stiffness_coefficients, and stiffness unused; cuDSS LDL^T with
        row_scale, None = ShiftPencil.row_scale) and Arnoldi (eigs) on (K - sigma M)^-1 M; massless rows give theta = 0
        (lambda infinite: NaN); RuntimeError when that factor is singular."""
        pencil = self._checked_pencil(pencil, (stiffness,), mass, np.complex128, symmetric=True)
        sigma = float(shift) * (1.0 + EIGSH_SHIFT_RELATIVE_OFFSET)
        factor, _ = self._factorize_shift(pencil, pencil.coefficients(stiffness_coefficients, sigma), row_scale,
                                          previous)
        n_unknowns = pencil.n_rows
        start_vector = None if start_vector is None else np.asarray(start_vector, dtype=np.complex128)
        if pencil.device is None:
            device_apply = None

            def apply(vector):
                return factor.solve_host(pencil.mass @ vector)
        else:   # M x and the solve on the device (the device Krylov-Schur: no transfers; host ARPACK: one each way)
            import cupy as cp
            device_mass = pencil.device_mass()

            def device_apply(vector):
                return factor.solve(device_mass @ vector)

            def apply(vector):
                return cp.asnumpy(device_apply(cp.asarray(vector)))
        try:
            theta, vectors, applications, converged = self.eigs(apply, n_unknowns, n_wanted, start_vector=start_vector,
                                                                device_matvec=device_apply)
        except CudssAccuracyError as exc:   # a refreshed factor's deferred check missed in its first solve
            if self._fall_back(exc, n_unknowns, pencil):   # 'auto': this shift again, on SuperLU
                return self.eigs_shifted(None, pencil.mass, shift, n_wanted, start_vector=start_vector, pencil=pencil,
                                         stiffness_coefficients=stiffness_coefficients)
            raise _shift_factor_error(exc, sigma) from None
        theta = np.asarray(theta, dtype=np.complex128)
        nonzero = np.abs(theta) > 0.0
        eigenvalues = np.full(theta.shape, np.nan, dtype=np.complex128)
        eigenvalues[nonzero] = sigma + 1.0 / theta[nonzero]
        return eigenvalues, vectors, applications, converged, factor

    def eigs(self, matvec, n_unknowns, n_wanted, start_vector=None, tolerance_floor=0.0, max_restarts=MAX_RESTARTS,
             default_basis=0, device_matvec=None, real=False):
        """(theta, host vectors, operator applications, converged) of the n_wanted largest-|theta| eigenpairs of the
        linear map `matvec` (numpy complex128, n_unknowns), tolerance = max(node, tolerance_floor), at most max_restarts
        restarts, basis = the node's subspace_size, else default_basis, else scipy's; without convergence the converged
        part. The device Krylov-Schur (eigs_device) instead of host ARPACK when device_arnoldi_fits: on device_matvec
        (cupy -> cupy, e.g. device_factor_matvec), else on a device_arnoldi runtime around matvec. real (E1): a real map
        (matvec real -> real, start_vector real): host ARPACK runs the real dnaupd (one real solve per application
        instead of two); the Krylov-Schur paths stay complex (matvec then gets complex vectors)."""
        basis_size = arnoldi_basis_size(n_wanted, n_unknowns, self.subspace_size, default_basis)
        if ((device_matvec is not None or self.device_arnoldi)
                and device_arnoldi_fits(n_unknowns, basis_size, host_matvec=device_matvec is None)):
            theta, vectors, applications, converged, _ = self.eigs_device(
                matvec if device_matvec is None else device_matvec, n_unknowns, n_wanted, start_vector,
                tolerance_floor=tolerance_floor, max_restarts=max_restarts, default_basis=default_basis,
                matvec_on_host=device_matvec is None)
            return theta, vectors, applications, converged
        return self._host_eigs(matvec, n_unknowns, n_wanted, start_vector, tolerance_floor, max_restarts,
                               default_basis, real=real)

    def _host_eigs(self, matvec, n_unknowns, n_wanted, start_vector, tolerance_floor, max_restarts, default_basis,
                   real=False):
        """eigs by scipy ARPACK (host basis; real: float64 operator and v0, dnaupd), counted."""
        import scipy.sparse.linalg
        self.last_arnoldi = 'ARPACK (real)' if real else 'ARPACK'
        applications = [0]

        def counted(vector):
            applications[0] += 1
            return matvec(vector)

        dtype = np.float64 if real else np.complex128
        operator = scipy.sparse.linalg.LinearOperator((n_unknowns, n_unknowns), matvec=counted, dtype=dtype)
        if real and start_vector is not None:
            start_vector = np.ascontiguousarray(np.real(start_vector), dtype=np.float64)
        # dnaupd needs ncv >= k + 2 (arnoldi_basis_size keeps it; scipy's own default too)
        ncv = (arnoldi_basis_size(n_wanted, n_unknowns, self.subspace_size, default_basis)
               if self.subspace_size or default_basis else None)
        start = time.perf_counter()
        converged = True
        try:
            # ARPACK's BLAS between the PARDISO solves (host_direct header); E6 thread limit on small runs
            with quiet_openblas(self.factorization), host_blas_limit(n_unknowns):
                theta, vectors = scipy.sparse.linalg.eigs(operator, k=n_wanted, which='LM', v0=start_vector,
                                                          tol=max(self.tolerance, float(tolerance_floor)), ncv=ncv,
                                                          maxiter=max_restarts)
        except scipy.sparse.linalg.ArpackNoConvergence as exc:
            theta, vectors = exc.eigenvalues, exc.eigenvectors
            converged = False
            self.n_unconverged += 1
        self.total_eigen_seconds += time.perf_counter() - start
        self.n_eigen_solves += 1
        self.operator_applications += applications[0]
        return theta, vectors, applications[0], converged

    def eigs_device(self, matvec, n_unknowns, n_wanted, start_vector=None, tolerance_floor=0.0,
                    max_restarts=MAX_RESTARTS, default_basis=0, resume=None, matvec_on_host=False, reserve_basis=0):
        """eigs on the GPU (S5b): (theta, host vectors, applications, converged, state) of the n_wanted largest-|theta|
        eigenpairs of `matvec` (cupy complex128 (n,) -> cupy array; matvec_on_host: numpy -> numpy) by Krylov-Schur with
        the basis on the device (module header "Device Krylov-Schur"); tolerance, basis and restarts as eigs; resume =
        the state of an earlier call on the same matvec (a widening: its Krylov decomposition kept, expanded to the
        larger basis, in place when its storage holds it); reserve_basis: allocate rows for a later basis that large.
        Too small for a device basis (device_arnoldi_fits without the host rule): host eigs, state None."""
        import cupy as cp
        basis_size = arnoldi_basis_size(n_wanted, n_unknowns, self.subspace_size, default_basis)
        if not device_arnoldi_fits(n_unknowns, basis_size):
            host_matvec = matvec if matvec_on_host else (lambda vector: cp.asnumpy(matvec(cp.asarray(vector))))
            theta, vectors, applications, converged = self._host_eigs(
                host_matvec, n_unknowns, n_wanted, start_vector, tolerance_floor, max_restarts, default_basis)
            return theta, vectors, applications, converged, None
        self.last_arnoldi = 'device Krylov-Schur'
        start = time.perf_counter()
        tolerance = max(self.tolerance, float(tolerance_floor)) or float(np.finfo(np.float64).eps)
        search = _KrylovSchur(matvec, int(n_unknowns), int(n_wanted), basis_size, tolerance, start_vector, resume,
                              matvec_on_host=matvec_on_host,
                              capacity=min(int(reserve_basis), int(n_unknowns) // DEVICE_ARNOLDI_MIN_RATIO - 1))
        try:
            theta, vectors, converged = search.run(int(max_restarts))
        finally:
            self.total_eigen_seconds += time.perf_counter() - start
            self.n_eigen_solves += 1
            self.operator_applications += search.applications
        if not converged:
            self.n_unconverged += 1
        return theta, vectors, search.applications, converged, search.state()

    def eigs_host(self, matvec, n_unknowns, n_wanted, start_vector=None, tolerance_floor=0.0,
                  max_restarts=MAX_RESTARTS, default_basis=0, resume=None, reserve_basis=0):
        """eigs_device on the host (_HostKrylovSchur, numpy matvec): the same 5-tuple, so a widening resumes the last
        decomposition instead of a host ARPACK restart; below the DEVICE_ARNOLDI_MIN_RATIO size host ARPACK, state None."""
        basis_size = arnoldi_basis_size(n_wanted, n_unknowns, self.subspace_size, default_basis)
        if not device_arnoldi_fits(n_unknowns, basis_size):
            theta, vectors, applications, converged = self._host_eigs(
                matvec, n_unknowns, n_wanted, start_vector, tolerance_floor, max_restarts, default_basis)
            return theta, vectors, applications, converged, None
        self.last_arnoldi = 'host Krylov-Schur'
        start = time.perf_counter()
        tolerance = max(self.tolerance, float(tolerance_floor)) or float(np.finfo(np.float64).eps)
        search = _HostKrylovSchur(matvec, int(n_unknowns), int(n_wanted), basis_size, tolerance, start_vector, resume,
                                  capacity=min(int(reserve_basis), int(n_unknowns) // DEVICE_ARNOLDI_MIN_RATIO - 1))
        try:
            # its gemv between the PARDISO solves (host_direct header); E6 thread limit on small runs
            with quiet_openblas(self.factorization), host_blas_limit(n_unknowns):
                theta, vectors, converged = search.run(int(max_restarts))
        finally:
            self.total_eigen_seconds += time.perf_counter() - start
            self.n_eigen_solves += 1
            self.operator_applications += search.applications
        if not converged:
            self.n_unconverged += 1
        return theta, vectors, search.applications, converged, search.state()

    def eigsh_inverse(self, inverse_matvec, mass_matvec, n_unknowns, n_wanted, start_vector=None,
                      tolerance_floor=0.0, max_restarts=MAX_RESTARTS):
        """(lambda, vectors, applications, converged, tolerance) of the n_wanted eigenpairs K x = lambda M x nearest 0 of
        an SPD pencil known only by inverse_matvec(b) ~ K^-1 b (an inexact inner solve, no factor) and mass_matvec(x) =
        M x (float64 host vectors): scipy eigsh shift-invert at sigma = 0, tolerance = max(node, tolerance_floor), at
        most max_restarts Lanczos restarts."""
        import scipy.sparse.linalg
        applications = [0]

        def counted(vector):
            applications[0] += 1
            return inverse_matvec(vector)

        def unused(vector):
            raise RuntimeError('eigsh_inverse: shift-invert mode never applies the stiffness itself')

        shape = (n_unknowns, n_unknowns)
        stiffness = scipy.sparse.linalg.LinearOperator(shape, matvec=unused, dtype=np.float64)
        mass = scipy.sparse.linalg.LinearOperator(shape, matvec=mass_matvec, dtype=np.float64)
        inverse = scipy.sparse.linalg.LinearOperator(shape, matvec=counted, dtype=np.float64)
        tolerance = max(self.tolerance, float(tolerance_floor))
        ncv = arnoldi_basis_size(n_wanted, n_unknowns, self.subspace_size) if self.subspace_size else None
        start = time.perf_counter()
        converged = True
        try:
            eigenvalues, vectors = scipy.sparse.linalg.eigsh(stiffness, k=n_wanted, M=mass, sigma=0.0, which='LM',
                                                             OPinv=inverse, v0=start_vector, ncv=ncv,
                                                             maxiter=max_restarts, tol=tolerance)
        except scipy.sparse.linalg.ArpackNoConvergence as exc:
            eigenvalues, vectors = exc.eigenvalues, exc.eigenvectors
            converged = False
            self.n_unconverged += 1
        self.total_eigen_seconds += time.perf_counter() - start
        self.n_eigen_solves += 1
        self.operator_applications += applications[0]
        return eigenvalues, vectors, applications[0], converged, tolerance

    def work_counters(self) -> dict:
        """Work so far (summary 'solvers' record of its (physics, role)); eigen_seconds / eigen_factor_seconds are host
        wall times of the Arnoldi runs / the factorisations."""
        counters = {'eigen_solves': self.n_eigen_solves, 'eigen_factorizations': self.n_factorizations,
                    'operator_applications': self.operator_applications, 'eigen_unconverged': self.n_unconverged,
                    'eigen_seconds': self.total_eigen_seconds, 'eigen_factor_seconds': self.total_factor_seconds,
                    'eigen_reused_orderings': self.orderings.n_reused}
        if self.fallback_factorization is not None or self.n_auto_fallbacks:
            counters['eigen_auto_fallbacks'] = self.n_auto_fallbacks
        if self.factorization in CUDSS_FACTORIZATIONS or self.n_auto_fallbacks:
            runtime = self._pencil_runtime   # its plans (deferred fallbacks included); factorisations counted here
            plans, fallbacks, reused, plan_seconds = (
                (0, 0, 0, 0.0) if runtime is None else
                (runtime.n_cudss_plans, runtime.n_cudss_double_fallbacks, runtime.n_cudss_reused_orderings,
                 runtime.total_cudss_plan_seconds))
            # eigen_reused_orderings above is SuperLU's; a cuDSS plan on a kept ordering (e.g. the static K_ff's) shows
            # here, plan seconds = upload, index maps, reordering (unless reused) and symbolic factorisation
            counters.update(eigen_factor_device_bytes=self.device_bytes, eigen_permanent_bytes=self.permanent_bytes,
                            eigen_lu_entries=self.lu_entries, eigen_refinement_steps=self.refinement_steps,
                            eigen_factor_symmetric=int(self.symmetric_factor),
                            eigen_double_fallbacks=self.double_fallbacks + fallbacks,
                            eigen_cudss_plans=self.n_cudss_plans + plans,
                            eigen_cudss_reused_orderings=self.n_cudss_reused_orderings + reused,
                            eigen_cudss_plan_seconds=self.total_cudss_plan_seconds + plan_seconds)
        if self.factorization in HOST_DIRECT_FACTORIZATIONS:
            counters.update(eigen_host_direct_analyses=self.n_host_direct_analyses, eigen_lu_entries=self.lu_entries,
                            eigen_factor_symmetric=int(self.symmetric_factor),
                            eigen_double_fallbacks=self.double_fallbacks,
                            eigen_refinement_steps=self.refinement_steps,
                            eigen_host_direct_threads=self.host_direct_threads)
            if host_direct_compressed(self.factorization):   # mumps_blr: full-rank refactors after a miss
                counters['eigen_compression_fallbacks'] = self.compression_fallbacks
        return counters


_ONE, _ZERO, _MINUS_ONE = np.complex128(1.0), np.complex128(0.0), np.complex128(-1.0)
_UNIT_ROW_FLOOR = 1e-300   # |w|^2 floor of the device normalisation (a breakdown is caught on the host afterwards)
_UNIT_ROW_KERNEL = []


def _unit_row_kernel():
    """ElementwiseKernel row = w / sqrt(|w|^2), |w|^2 read on the device from slot `slot` of a float64 array (no sync)."""
    if not _UNIT_ROW_KERNEL:
        import cupy as cp
        _UNIT_ROW_KERNEL.append(cp.ElementwiseKernel(
            'complex128 w, raw float64 squared, int64 slot', 'complex128 row',
            f'row = w * (1.0 / sqrt(fmax(squared[slot], {_UNIT_ROW_FLOOR!r})))', 'open_sol_krylov_unit_row'))
    return _UNIT_ROW_KERNEL[0]


def _gram_schmidt(basis, vector, coefficients, correction):
    """CGS2 of a cupy complex128 vector (in place) against the orthonormal device rows of basis (k, n): both classical
    passes always, cuBLAS gemv on the rows as F-order (n, k) columns; coefficients (k,) = the summed projections. No
    host read (the caller packs it)."""
    from cupy import cublas
    columns = basis.T   # F-contiguous view: V^H w by op 'H', w - V c by op 'N'
    cublas.gemv('H', _ONE, columns, vector, _ZERO, coefficients)
    cublas.gemv('N', _MINUS_ONE, columns, coefficients, _ONE, vector)
    cublas.gemv('H', _ONE, columns, vector, _ZERO, correction)
    cublas.gemv('N', _MINUS_ONE, columns, correction, _ONE, vector)
    coefficients += correction


class _KrylovSchur:
    """One eigs_device run: the Krylov decomposition A V_p = V_{p+1} H (rows of V on the device, H (p+1, p) on the host)
    expanded to the basis size, Schur-ordered, checked and truncated per restart (module header); matvec_on_host: a
    numpy matvec, fed from the packed read of the previous step."""

    def __init__(self, matvec, n_unknowns, n_wanted, basis_size, tolerance, start_vector, resume,
                 matvec_on_host=False, capacity=0):
        import cupy as cp
        import cupyx
        self.matvec = matvec
        self.matvec_on_host = bool(matvec_on_host)
        # per application: the product w, the second-pass projections, one packed device buffer [coefficients (m + 1
        # slots), |w|^2 (real part of slot m + 1), host matvec: the next unit row] and its pinned host twin
        self._product = cp.empty(n_unknowns, dtype=np.complex128)
        self._correction = cp.empty(basis_size + 1, dtype=np.complex128)
        self._norm_slot = basis_size + 1
        packed_size = basis_size + 2 + (n_unknowns if self.matvec_on_host else 0)
        self._packed = cp.zeros(packed_size, dtype=np.complex128)
        self._packed_real = self._packed.view(np.float64)
        self._host_packed = cupyx.empty_pinned((packed_size,), dtype=np.complex128)
        self._host_row = None   # matvec_on_host: (basis row, its host copy in the pinned buffer) of the next application
        # E4, device matvec: the H columns of an expansion and their |w|^2 stay on the device (row j = column j:
        # coefficients in slots 0..j, |w|^2 in the real part of slot m + 1), one read per expansion into a pinned twin
        self._columns = self._columns_real = self._host_columns = None
        if not self.matvec_on_host:
            self._columns = cp.zeros((basis_size, basis_size + 2), dtype=np.complex128)
            self._columns_real = self._columns.reshape(-1).view(np.float64)
            self._host_columns = cupyx.empty_pinned((basis_size, basis_size + 2), dtype=np.complex128)
        self.n_unknowns = n_unknowns
        self.n_wanted = n_wanted
        self.basis_size = basis_size   # m: Arnoldi vectors before each restart (V holds m + 1 rows)
        self.keep = min(basis_size - 1, max(n_wanted + (basis_size - n_wanted) // 2, n_wanted + 1))
        self.tolerance = tolerance
        self.applications = 0
        self.hessenberg = np.zeros((basis_size + 1, basis_size), dtype=np.complex128)
        self.size = 0   # p: filled columns of H
        self._random = np.random.default_rng(_BREAKDOWN_SEED)
        self._state = None
        if resume is not None and resume[1].shape[1] < basis_size:   # a widening: the earlier decomposition kept
            kept_basis, kept_hessenberg = resume
            self.size = kept_hessenberg.shape[1]
            if kept_basis.shape[0] >= basis_size + 1:   # its storage reserved the rows: expanded in place, no copy
                self.basis = kept_basis
            else:   # a second basis next to the first while copying (the caller drops the first afterwards)
                self.basis = cp.empty((max(basis_size, capacity) + 1, n_unknowns), dtype=np.complex128)
                self.basis[:self.size + 1] = kept_basis[:self.size + 1]
            self.hessenberg[:self.size + 1, :self.size] = kept_hessenberg
            return
        self.basis = cp.empty((max(basis_size, capacity) + 1, n_unknowns), dtype=np.complex128)   # V rows (+ reserve)
        if start_vector is None:
            start_vector = self._random.standard_normal(n_unknowns) + 1j * self._random.standard_normal(n_unknowns)
        vector = cp.asarray(start_vector, dtype=np.complex128)
        norm = float(cp.linalg.norm(vector))
        if not norm > 0.0:
            raise ValueError('eigs_device needs a nonzero start vector')
        self.basis[0] = vector / norm

    def state(self):
        """(device basis storage (>= m + 1 rows, n; the first m + 1 hold V), host H (m + 1, m)) of the last check, before
        any truncation (resume=)."""
        return self._state

    def run(self, max_restarts):
        """(theta, host vectors (n, k), converged): restarts until the n_wanted largest |theta| meet the tolerance
        (ARPACK's test: Ritz estimate <= tol max(eps^(2/3), |theta|)), else the converged part after max_restarts."""
        floor = float(np.finfo(np.float64).eps) ** (2.0 / 3.0)
        restarts = 0
        while True:
            self._expand()
            schur, unitary, residual_row = self._sorted_schur()
            values, ritz = np.linalg.eig(schur[:self.keep, :self.keep])   # unit columns
            estimates = np.abs(residual_row[:self.keep] @ ritz)
            wanted = np.argsort(-np.abs(values), kind='stable')[:self.n_wanted]
            met = estimates[wanted] <= self.tolerance * np.maximum(np.abs(values[wanted]), floor)
            converged = bool(np.all(met))
            if converged or restarts >= max_restarts:
                self._state = (self.basis, self.hessenberg.copy())
                if not converged:
                    wanted = wanted[met]
                return values[wanted], self._vectors(unitary[:, :self.keep] @ ritz[:, wanted]), converged
            self._truncate(schur, unitary, residual_row)
            restarts += 1

    def _expand(self):
        """Arnoldi steps from column p to m (one operator application and one packed host read each; a device matvec:
        _expand_device)."""
        if not self.matvec_on_host:
            self._expand_device()
            return
        for column in range(self.size, self.basis_size):
            # the next row came with the previous read (else one transfer: restarts, resume)
            product = self.matvec(self._host_vector(column))
            self._product.set(np.ascontiguousarray(product, dtype=np.complex128))
            self.applications += 1
            coefficients, norm = self._orthogonalized(column + 1)
            self.hessenberg[:column + 1, column] = coefficients
            if norm <= 1e-12 * math.sqrt(float(np.sum(np.abs(coefficients) ** 2)) + norm ** 2):
                self.basis[column + 1] = self._fresh_direction(column + 1)   # invariant subspace: H[column + 1] = 0
                self._host_row = None
            else:
                self.hessenberg[column + 1, column] = norm
        self.size = self.basis_size

    def _expand_device(self):
        """Arnoldi steps from column p to m on a device matvec (E4): CGS2, |w|^2 and the next unit row on the device per
        step, the H columns read once afterwards; a breakdown found in that read gets a fresh direction and the steps
        after it are redone (they were built on a round-off row)."""
        from .pcg import device_vector_dot
        row_slots = self.basis_size + 2
        while self.size < self.basis_size:
            first = self.size
            for column in range(first, self.basis_size):
                self._product[...] = self.matvec(self.basis[column])
                self.applications += 1
                rows = column + 1
                _gram_schmidt(self.basis[:rows], self._product, self._columns[column, :rows],
                              self._correction[:rows])
                flat = self._product.view(np.float64)
                squared_slot = 2 * (column * row_slots + self._norm_slot)   # real part, float64 view of the rows
                device_vector_dot(flat, flat, out=self._columns_real[squared_slot])
                _unit_row_kernel()(self._product, self._columns_real, np.int64(squared_slot), self.basis[rows])
            host = self._host_columns[first:self.basis_size]
            self._columns[first:self.basis_size].get(out=host)
            self.size = self.basis_size
            for column in range(first, self.basis_size):
                line = host[column - first]
                coefficients = line[:column + 1]
                norm = math.sqrt(max(float(line[self._norm_slot].real), 0.0))
                self.hessenberg[:column + 1, column] = coefficients
                if norm <= 1e-12 * math.sqrt(float(np.sum(np.abs(coefficients) ** 2)) + norm ** 2):
                    self.hessenberg[column + 1, column] = 0.0   # invariant subspace
                    self.basis[column + 1] = self._fresh_direction(column + 1)
                    self.size = column + 1
                    break
                self.hessenberg[column + 1, column] = norm

    def _orthogonalized(self, rows):
        """CGS2 of the product against the first `rows` rows, basis[rows] = its unit vector (normalised on the device),
        then one read of [coefficients, |w|^2(, the new row)]: (host coefficients, norm)."""
        from .pcg import device_vector_dot
        coefficients = self._packed[:rows]
        _gram_schmidt(self.basis[:rows], self._product, coefficients, self._correction[:rows])
        flat = self._product.view(np.float64)
        squared_slot = 2 * self._norm_slot   # real part of the norm slot in the float64 view
        device_vector_dot(flat, flat, out=self._packed_real[squared_slot])
        _unit_row_kernel()(self._product, self._packed_real, np.int64(squared_slot), self.basis[rows])
        if self.matvec_on_host:
            self._packed[self._norm_slot + 1:] = self.basis[rows]
        size = self._packed.size if self.matvec_on_host else self._norm_slot + 1
        host = self._host_packed[:size]
        self._packed[:size].get(out=host)
        if self.matvec_on_host:
            self._host_row = (rows, host[self._norm_slot + 1:])
        return host[:rows].copy(), math.sqrt(max(float(host[self._norm_slot].real), 0.0))

    def _host_vector(self, row):
        """Host copy of basis[row] for a host matvec: the one the last read brought, else one transfer."""
        import cupy as cp
        if self._host_row is not None and self._host_row[0] == row:
            return self._host_row[1]
        return cp.asnumpy(self.basis[row])

    def _fresh_direction(self, rows):
        """A unit random device vector orthogonal to the first `rows` basis rows (after an invariant subspace)."""
        import cupy as cp
        vector = cp.asarray(self._random.standard_normal(self.n_unknowns)
                            + 1j * self._random.standard_normal(self.n_unknowns))
        scratch = cp.empty((2, rows), dtype=np.complex128)
        _gram_schmidt(self.basis[:rows], vector, scratch[0], scratch[1])
        return vector / cp.linalg.norm(vector)

    def _sorted_schur(self):
        """(T, Q, b Q): the complex Schur form S = Q T Q^H of H[:m, :m] with the `keep` largest |eigenvalues| leading
        (LAPACK trsen by position: a reordering never fails in complex arithmetic) and the residual row b = H[m, :m]."""
        import scipy.linalg
        size = self.basis_size
        schur, unitary = scipy.linalg.schur(self.hessenberg[:size, :size], output='complex')
        select = np.zeros(size, dtype=np.int32)
        select[np.argsort(-np.abs(np.diag(schur)), kind='stable')[:self.keep]] = 1
        trsen = scipy.linalg.get_lapack_funcs('trsen', (schur,))
        result = trsen(select, schur, unitary, lwork=max(1, size * size))
        if result[-1] != 0:
            raise RuntimeError(f'eigs_device: LAPACK trsen failed (info {result[-1]})')
        schur, unitary = np.triu(result[0]), result[1]
        return schur, unitary, self.hessenberg[size, :size] @ unitary

    def _truncate(self, schur, unitary, residual_row):
        """Keep the leading `keep` Schur vectors: V_keep = V_m Q[:, :keep], H = [T_keep; (b Q)_keep], p = keep."""
        import cupy as cp
        keep, size = self.keep, self.basis_size
        coefficient_rows = cp.asarray(np.ascontiguousarray(unitary[:, :keep].T))
        for start, stop in _column_chunks(self.n_unknowns, size):   # in place: each chunk read before it is written
            self.basis[:keep, start:stop] = coefficient_rows @ self.basis[:size, start:stop]
        self.basis[keep] = self.basis[size]
        if self._host_row is not None and self._host_row[0] == size:   # the residual row moved, its host copy holds
            self._host_row = (keep, self._host_row[1])
        self.hessenberg[:] = 0.0
        self.hessenberg[:keep, :keep] = schur[:keep, :keep]
        self.hessenberg[keep, :keep] = residual_row[:keep]
        self.size = keep

    def _vectors(self, coefficients):
        """Host (n, k) Ritz vectors V_m C of an (m, k) coefficient matrix (device products by column chunks)."""
        import cupy as cp
        coefficient_rows = cp.asarray(np.ascontiguousarray(coefficients.T))
        vectors = np.empty((self.n_unknowns, coefficient_rows.shape[0]), dtype=np.complex128)
        for start, stop in _column_chunks(self.n_unknowns, self.basis_size):
            vectors[start:stop] = cp.asnumpy(coefficient_rows @ self.basis[:self.basis_size, start:stop]).T
        return vectors


class _HostKrylovSchur(_KrylovSchur):
    """The numpy twin of _KrylovSchur (eigs_host): basis rows in host memory, a numpy matvec, CGS2 by BLAS gemv on the
    rows (no conjugated basis copy); run / Schur ordering / restart test shared, so a widening resumes as on the device."""

    def __init__(self, matvec, n_unknowns, n_wanted, basis_size, tolerance, start_vector, resume, capacity=0):
        self.matvec = matvec
        self.matvec_on_host = True
        self._product = np.empty(n_unknowns, dtype=np.complex128)
        self._host_row = None
        self.n_unknowns = n_unknowns
        self.n_wanted = n_wanted
        self.basis_size = basis_size
        self.keep = min(basis_size - 1, max(n_wanted + (basis_size - n_wanted) // 2, n_wanted + 1))
        self.tolerance = tolerance
        self.applications = 0
        self.hessenberg = np.zeros((basis_size + 1, basis_size), dtype=np.complex128)
        self.size = 0
        self._random = np.random.default_rng(_BREAKDOWN_SEED)
        self._state = None
        if resume is not None and resume[1].shape[1] < basis_size:   # a widening: the earlier decomposition kept
            kept_basis, kept_hessenberg = resume
            self.size = kept_hessenberg.shape[1]
            if kept_basis.shape[0] >= basis_size + 1:   # reserved rows (np.empty: untouched pages until used)
                self.basis = kept_basis
            else:
                self.basis = np.empty((max(basis_size, capacity) + 1, n_unknowns), dtype=np.complex128)
                self.basis[:self.size + 1] = kept_basis[:self.size + 1]
            self.hessenberg[:self.size + 1, :self.size] = kept_hessenberg
            return
        self.basis = np.empty((max(basis_size, capacity) + 1, n_unknowns), dtype=np.complex128)
        if start_vector is None:
            start_vector = self._random.standard_normal(n_unknowns) + 1j * self._random.standard_normal(n_unknowns)
        vector = np.asarray(start_vector, dtype=np.complex128)
        norm = float(np.linalg.norm(vector))
        if not norm > 0.0:
            raise ValueError('eigs_host needs a nonzero start vector')
        self.basis[0] = vector / norm

    def _expand(self):
        """Arnoldi steps from column p to m (one operator application each)."""
        for column in range(self.size, self.basis_size):
            self._product[...] = self.matvec(self.basis[column])
            self.applications += 1
            coefficients, norm = self._orthogonalized(column + 1)
            self.hessenberg[:column + 1, column] = coefficients
            if norm <= 1e-12 * math.sqrt(float(np.sum(np.abs(coefficients) ** 2)) + norm ** 2):
                self.basis[column + 1] = self._fresh_direction(column + 1)   # invariant subspace: H[column + 1] = 0
            else:
                self.hessenberg[column + 1, column] = norm
        self.size = self.basis_size

    @staticmethod
    def _host_gram_schmidt(rows, vector):
        """CGS2 of `vector` (in place) against the orthonormal rows (k, n): V^H w = conj(V conj(w)), w -= c V (gemv
        on the C-contiguous rows both ways); the summed projections."""
        coefficients = np.conj(rows @ np.conj(vector))
        vector -= coefficients @ rows
        correction = np.conj(rows @ np.conj(vector))
        vector -= correction @ rows
        return coefficients + correction

    def _orthogonalized(self, rows):
        coefficients = self._host_gram_schmidt(self.basis[:rows], self._product)
        norm = float(np.linalg.norm(self._product))
        np.multiply(self._product, 1.0 / max(norm, _UNIT_ROW_FLOOR), out=self.basis[rows])
        return coefficients, norm

    def _fresh_direction(self, rows):
        vector = self._random.standard_normal(self.n_unknowns) + 1j * self._random.standard_normal(self.n_unknowns)
        self._host_gram_schmidt(self.basis[:rows], vector)
        return vector / np.linalg.norm(vector)

    def _truncate(self, schur, unitary, residual_row):
        keep, size = self.keep, self.basis_size
        coefficient_rows = np.ascontiguousarray(unitary[:, :keep].T)
        for start, stop in _column_chunks(self.n_unknowns, size):   # in place: each chunk read before it is written
            self.basis[:keep, start:stop] = coefficient_rows @ self.basis[:size, start:stop]
        self.basis[keep] = self.basis[size]
        self.hessenberg[:] = 0.0
        self.hessenberg[:keep, :keep] = schur[:keep, :keep]
        self.hessenberg[keep, :keep] = residual_row[:keep]
        self.size = keep

    def _vectors(self, coefficients):
        coefficient_rows = np.ascontiguousarray(coefficients.T)
        vectors = np.empty((self.n_unknowns, coefficient_rows.shape[0]), dtype=np.complex128)
        for start, stop in _column_chunks(self.n_unknowns, self.basis_size):
            vectors[start:stop] = (coefficient_rows @ self.basis[:self.basis_size, start:stop]).T
        return vectors


def _column_chunks(n_unknowns, rows):
    """(start, stop) unknown ranges of about KRYLOV_CHUNK_BYTES of `rows` complex128 basis rows each (the restart and
    Ritz products: a (keep, n) temporary was 48 / 96 rows of the Mie example's 331 MiB run high-water)."""
    width = max(1024, KRYLOV_CHUNK_BYTES // (16 * max(int(rows), 1)))
    return [(start, min(start + width, n_unknowns)) for start in range(0, n_unknowns, width)]


def time_estimate(values, work):
    """Run-time recipe of an arpack_shift_invert role (core.time_estimate.SolverWork, solves = eigen solves of
    work.eigenpairs each): shift factorisations (host SuperLU) and operator applications (a triangular solve, or the
    caller's product_ops for a factor-free inverse, + the Arnoldi orthogonalisation); cuDSS factorizations: _cudss_ops
    (2D the profile's cudss curves, 3D the fitted recipe above), other dimensions not estimated. work.plan_reused
    marks a ShiftPencil caller (its set-up once, host M products, per-shift combination; work.nnz = the pencil's
    entries); the device Krylov-Schur wherever eigs runs it (gpu backend, device_arnoldi_fits; work.device_arnoldi
    forces it: solve_resonances' own factor); cudss_mixed below MIXED_EIGEN_MIN_UNKNOWNS priced as cudss (S5c); the
    host direct factorizations on the SuperLU items x sparse_direct.HOST_DIRECT_* (uncalibrated)."""
    from dataclasses import replace
    from ..core import time_estimate as te
    factorization = values.get('factorization')
    if factorization == 'cudss_mixed' and float(work.unknowns) < MIXED_EIGEN_MIN_UNKNOWNS:
        factorization = 'cudss'   # S5c: a model runtime factors small operators in double
    on_device = factorization in CUDSS_FACTORIZATIONS and not work.product_ops
    if on_device and work.dimensions not in (2, 3):
        return te.TimeEstimate(not_estimated=(f"{work.role} solver (arpack_shift_invert factorization "
                                              f"'{factorization}', {work.dimensions}D)",))
    n_unknowns = float(work.unknowns)
    basis = arnoldi_basis_size(max(work.eigenpairs, 1), max(int(n_unknowns), 1), int(values.get('subspace_size', 0)),
                               work.default_basis)
    complex_factor = 4.0 if work.complex else 1.0
    # plan_reused: a ShiftPencil caller (modal.ModalPencil): one M product per application (scipy mode 3 / eigs on
    # (K - sigma M)^-1 M); per shift a one-pass host combination of K and M (+ SuperLU's CSC copy), or with a
    # DevicePencil (cuDSS, >= DEVICE_PENCIL_MIN_ENTRIES entries) the device refresh (2D device curve) and device M products
    pencil_caller = work.plan_reused and not work.product_ops
    value_bytes, entries = float(work.value_bytes), float(work.nnz)
    terms = float(work.pencil_terms)   # K_1 .. K_m and M (loaded: K, B, M)
    device_pencil = pencil_caller and on_device and entries >= DEVICE_PENCIL_MIN_ENTRIES
    # the device Krylov-Schur (eigs: device_arnoldi_fits) on a gpu run (work.run_backend: `backend` is 'cpu' for host
    # SuperLU curves): on a device matvec (a cuDSS factor outside a pencil, a DevicePencil), else around the host one;
    # a real work is eigsh (eigsh_shifted / eigsh_inverse: host Lanczos), a complex one eigs (a complex Hermitian
    # eigsh_shifted pencil is priced as eigs: unchecked)
    lanczos = not work.complex
    device_matvec = on_device and (not pencil_caller or device_pencil)
    device_arnoldi = (not lanczos and (work.device_arnoldi or work.run_backend == 'gpu')
                      and device_arnoldi_fits(n_unknowns, basis, host_matvec=not device_matvec))
    plan = ()
    if on_device:
        cudss_work = replace(work, device_values=True) if device_pencil else work
        apply, plan, factorisation = _cudss_ops(factorization, replace(cudss_work, device_arnoldi=True)
                                                if device_arnoldi and device_matvec else cudss_work)
        if not work.plan_reused:   # a fresh plan with every factorisation
            factorisation, plan = plan + factorisation, ()
    else:   # host SuperLU, or MUMPS / PARDISO priced as SuperLU x sparse_direct.HOST_DIRECT_* (uncalibrated; a mixed
        # factor's solves x CUDSS_MIXED_SOLVES_PER_SOLVE as cudss_mixed: the eigen polish)
        from .sparse_direct import (HOST_DIRECT_COMPRESSED_FACTOR_TIME, HOST_DIRECT_COMPRESSED_FULL_SOLVES,
                                    HOST_DIRECT_FACTOR_TIME, HOST_DIRECT_SOLVE_TIME)
        host_direct = factorization in HOST_DIRECT_FACTORIZATIONS
        compressed = host_direct_compressed(factorization)   # mumps_blr: GMRES-refined solves on a cheaper factor
        factor_scale = work.factor_scale * ((HOST_DIRECT_FACTOR_TIME
                                             * (HOST_DIRECT_COMPRESSED_FACTOR_TIME if compressed else 1.0))
                                            if host_direct else 1.0)
        solve_scale = work.solve_scale * ((HOST_DIRECT_SOLVE_TIME * (CUDSS_MIXED_SOLVES_PER_SOLVE
                                                                     if factorization in MIXED_FACTORIZATIONS
                                                                     and float(work.unknowns)
                                                                     >= MIXED_EIGEN_MIN_UNKNOWNS else
                                                                     HOST_DIRECT_COMPRESSED_FULL_SOLVES if compressed
                                                                     else 1.0))
                                          if host_direct else 1.0)
        apply = work.product_ops or (te.HostSolve(n_unknowns, work.dimensions, work.complex, rhs=work.rhs,
                                                  scale=solve_scale),)
        factorisation = (te.HostLU(n_unknowns, work.dimensions, work.complex, scale=factor_scale),)
        if pencil_caller:   # SuperLU's CSC copy of the shifted operator
            factorisation += (te.HostStream(2.0 * (value_bytes + te.INDEX_BYTES) * entries),)
    if pencil_caller and not device_pencil:   # the one-pass host combination (every term read, 1 written)
        factorisation += (te.HostStream((terms + 1.0) * value_bytes * entries),)
    if device_pencil and terms > 2.0:   # DevicePencil.combined: terms beyond the device curve's two (lower-stored)
        factorisation += (te.Elementwise(2.0 * (terms - 2.0) * value_bytes * LOWER_STORED_FRACTION * entries,
                                         (terms - 2.0) * value_bytes * LOWER_STORED_FRACTION * entries,
                                         launches=int(terms - 2.0)),)
    if pencil_caller:   # DevicePencil: its M term on the device (+ a vector each way for host ARPACK)
        apply += (((te.Spmv(n_unknowns, entries, int(value_bytes)),)
                   + (() if device_arnoldi else (te.Transfer(2.0 * value_bytes * n_unknowns),)))
                  if device_pencil else
                  (te.HostStream((value_bytes + te.INDEX_BYTES) * entries + 3.0 * value_bytes * n_unknowns),))
    if device_arnoldi:
        # eigs_device (S5b): CGS2 reads ~3 x the complex128 basis per application (4 gemv over rows 0..j, j ~ 0.75 m
        # between restarts), the restart product V Q amortised over the applications, ~9 launches, one packed host
        # read (E4, a device matvec: one read per expansion of m - keep steps); around a host SuperLU solve a
        # complex128 vector each way (a cuDSS solve_host curve holds its own)
        wanted = max(int(work.eigenpairs), 1)
        keep = min(basis - 1, max(wanted + (basis - wanted) // 2, wanted + 1))
        reads = 1.0 / max(basis - keep, 1) if device_matvec else 1.0
        arnoldi = (te.Elementwise(3.0 * 16.0 * n_unknowns * basis, 3.0 * 16.0 * n_unknowns, launches=9),
                   te.DenseFlops(4.0 * complex_factor * n_unknowns * basis), te.Sync(reads))
        if not on_device:
            arnoldi += ((2.0, te.Transfer(16.0 * n_unknowns)),)
    else:   # host ARPACK: the n x basis vectors streamed ~twice per application (Gram-Schmidt and update; memory-bound,
        # S9: the Mie sphere's ~15 ms per application were priced as flops alone) + its restart flops
        arnoldi = (te.HostStream(2.0 * value_bytes * n_unknowns * basis),
                   te.DenseFlops(4.0 * complex_factor * n_unknowns * basis, host=True))
    application = apply + arnoldi
    solves = float(work.solves)
    per_basis = work.applications_per_basis or APPLICATIONS_PER_BASIS_GUESS
    items = [te.TimeItem('Arnoldi operator applications', 'application', solves * per_basis * basis,
                         'guess', application, limit=solves * basis * MAX_RESTARTS, per_step=work.per_step,
                         counter=work.counter('operator_applications'), seconds_counter=work.counter('eigen_seconds'))]
    if pencil_caller and work.factorizations:   # ShiftPencil: the terms placed on K's pattern by a key search (FEM)
        key_search = 8.0 * math.log2(max(entries, 2.0))   # int64 keys of every term, searchsorted + scatter per non-K term
        setup = (te.HostStream((2.0 * 8.0 * terms + (terms - 1.0) * key_search + terms * value_bytes) * entries),
                 te.PythonCalls(10 * terms))
        if device_pencil:   # every term uploaded and lower-taken (ModalPencil vouches: no symmetry sort); device_mass
            # mirrors M's lower triangle once (int64 keys, a device searchsorted, the gather)
            setup += (te.Transfer(terms * value_bytes * entries),
                      te.Elementwise(terms * value_bytes * entries, terms * value_bytes * LOWER_STORED_FRACTION * entries,
                                     launches=2 * int(terms)),
                      te.Elementwise((6.0 * 8.0 + key_search) * entries, (value_bytes + te.INDEX_BYTES) * entries,
                                     launches=10))
        items.append(te.TimeItem('shift pencil set-up (terms on one pattern)', 'assembly', 1.0, 'exact', setup))
    if not work.product_ops and work.factorizations:
        if plan:
            items.append(te.TimeItem('cuDSS plan', 'plan', 1.0, 'exact', plan))
        items.append(te.TimeItem('shift-invert factorisations', 'factorisation', float(work.factorizations),
                                 work.factorizations_kind, factorisation,
                                 per_step=work.factor_per_step, counter=work.counter('eigen_factorizations'),
                                 seconds_counter=work.counter('eigen_factor_seconds')))
    return te.TimeEstimate(tuple(items))


def _cudss_ops(factorization, work):
    """(ops per operator application, plan ops, factorisation ops) of a cuDSS factor. 2D: work.rhs solve_host calls,
    plan and factorisation on the profile curves (sparse_direct.cudss_2d_ops). 3D curl-curl (the
    fitted recipe): a vector each way, work.rhs solves each streaming the factor (cudss_mixed: complex64 solves + a
    complex128 residual product per solve), and the host plan + device factorisation; a real operator (value_bytes 8)
    scaled by CUDSS_REAL_BYTES / CUDSS_REAL_FACTOR_TIME."""
    from ..core import time_estimate as te
    n_unknowns = float(work.unknowns)
    mixed = factorization == 'cudss_mixed'
    if work.dimensions == 2:
        from .sparse_direct import cudss_2d_ops
        plan, factor, solve = cudss_2d_ops(mixed, work)
        return ((float(work.rhs), solve),), (plan,), (factor,)
    value_bytes = float(work.value_bytes)
    real_bytes, real_time = (1.0, 1.0) if work.complex else (CUDSS_REAL_BYTES, CUDSS_REAL_FACTOR_TIME)
    device_bytes = (CUDSS_BYTES_PER_UNKNOWN * n_unknowns * (n_unknowns / CUDSS_REFERENCE_UNKNOWNS) ** CUDSS_BYTES_GROWTH
                    * (1.0 if mixed else CUDSS_DOUBLE_BYTES) * real_bytes)
    solve = te.Elementwise(CUDSS_SOLVE_READS * device_bytes, value_bytes * n_unknowns)
    solves = float(work.rhs)
    transfers = () if work.device_arnoldi else ((2.0, te.Transfer(value_bytes * n_unknowns)),)   # eigs_device: none
    apply = transfers + (
        ((solves * CUDSS_MIXED_SOLVES_PER_SOLVE, solve), (solves, te.Spmv(n_unknowns, work.nnz, int(work.value_bytes))))
        if mixed else ((solves, solve),))
    flops = (CUDSS_FACTOR_FLOPS * (n_unknowns / CUDSS_REFERENCE_UNKNOWNS) ** CUDSS_FACTOR_GROWTH
             * (1.0 if mixed else CUDSS_DOUBLE_FACTOR_TIME) * real_time)
    return apply, (te.HostStream(CUDSS_PLAN_HOST_BYTES * n_unknowns),), (te.DenseFlops(flops),)
