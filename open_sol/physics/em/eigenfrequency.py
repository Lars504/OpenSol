"""A6 eigenfrequency solve of an assembled FDFD system (host numpy / scipy; with an own cuDSS factor the operator
applications and the Krylov-Schur basis run on the GPU, cupy inside the solver): shift-invert Arnoldi about the target
omega; dispersion and the PML stretch are frozen at the target, so resonances are accurate only near it."""
# Eigenproblem, dimension-free (2D FDFD cells, 3D em_fdfd_3d E faces): the operator assembled at the target omega_t
# is A = L + k_t^2 M on the caller's flat unknowns, k = omega / C0: L = the negated curl-curl part (face weights, image
# folds, PML stretch, all at omega_t), M = the complex mass diagonal at omega_t (eps, or mu for hz; sigma_e, poles),
# 0 on pinned identity rows (2D PEC / outermost wall cells), or a sparse mass matrix (tetrahedral Nedelec, em3d fem3d:
# _PencilMass; then the PML share splits Re(conj(u) W u) by rows). A resonance solves L u = -k^2 M u. With B = -M,
# (A^-1 B) u = theta u, theta = 1 / (k^2 - k_t^2) ('shift_invert'): ARPACK's largest |theta| are the k^2 nearest k_t^2;
# pinned unknowns give theta = 0 and are dropped. A curl-curl operator without identity rows (3D) has a gradient null
# space at k = 0 (theta = -1 / k_t^2, one per cell), which plain shift-invert ranks ahead of every k > sqrt(2) k_t:
# 'squared_cayley' (the BOR map, squared_cayley_roots) sends it to 0. omega = C0 sqrt(k^2), principal branch. In
# exp(+j omega t) a decaying (lossy or radiating) resonance has Im omega > 0 and Q = Re omega / (2 Im omega). Because M
# and the stretch belong to omega_t, the problem is linear in k^2 only by freezing them: exact for non-dispersive,
# PML-free models, otherwise accurate to first order in |omega / omega_t - 1|. An own factor shifts the spectral
# parameter only, k_t^2 -> sigma = k_t^2 (1 + SHIFT_RELATIVE_OFFSET) with A + (sigma - k_t^2) M (M, L stay frozen at
# omega_t: the same resonances), so a target on a lossless resonance is not singular; a caller's factorization (line
# block of A) keeps sigma = k_t^2. The size rule of the host factor is the caller's (its warnings join the solve's).
# Region energy fraction (new_modules_plan P2, as fem.harmonic_resonances): in_region flags the unknowns of a region
# (a resonance_table selection: cells, or the free dofs of its elements, element_region_rows); region_fraction = the
# in_region share of the per-row energies (the PML-share split), NaN without a region; below min_region_fraction the
# resonance is dropped like a PML mode (widenings count it), so table rows and snapshots stay numbered alike.
import functools
import math
import os
import time
from dataclasses import dataclass

import numpy as np

from ...core.constants import C0
from ...solvers.arpack_shift_invert import (LDLT_FACTORIZATIONS, ArpackShiftInvertSolver, arnoldi_basis_size,
                                            device_arnoldi_fits)
from ...solvers.host_direct import is_host_direct_factor, library_solve_text

PML_RESONANCE_FRACTION = 0.5    # resonances with a larger share of the |u|^2 energy weights in PML cells are dropped
RESONANCE_TABLE_COLUMNS = ('resonance_number', 'wavelength', 'frequency', 'frequency_imag', 'q_factor', 'pml_fraction')
RESONANCE_TABLE_UNITS = ('1', 'm', 'Hz', 'Hz', '1', '1')
REGION_COLUMN = ('region_energy_fraction', '1')   # (name, unit) appended to the table when the solve had a region
SPECTRAL_MAPS = ('shift_invert', 'squared_cayley')
STATIC_RELATIVE_K2 = 1e-6       # squared_cayley: |k^2| below this times k_t^2 is a discrete gradient (static), dropped
# own factor: sigma = k_t^2 (1 + this) and A + (sigma - k_t^2) M (as conical.py): a lossless target exactly on a
# resonance made A singular (the 2D PEC box example targets its discrete (1, 1) mode: cuDSS probe residual 1.25e-7)
SHIFT_RELATIVE_OFFSET = 1e-4
_THETA_FLOOR = 1e-12            # |theta| below this times the largest: pinned / PEC unknowns (infinite k^2)
_LOSSLESS_RELATIVE_IMAG = 1e-12   # |Im omega| below this times Re omega is round-off: Q = inf
_START_VECTOR_SEED = 6          # deterministic ARPACK start vector (no symmetry, so odd modes are reached too)
# mixed LDL^T of D A (cudss_mixed on the device, mumps_mixed / pardiso_mixed on the host): one float64 refinement step
# per solve (~3e-8 on the Mie eighth sphere, cuDSS) and ARPACK stopped above that noise; the Rayleigh quotients of the
# vectors then restore the eigenvalues to ~(1e-6)^2
# A double LDL^T of D A takes the same ARPACK floor and polish when the caller floored ARPACK already (an open model)
# and A is complex (PML / loss: Q finite, so the ~1e-12 imaginary round-off of a quotient is no lossless Q = inf): host
# ARPACK to 1e-10 dominated the PARDISO Mie run (39 ms solves, two per application, ~660 applications)
# E2: a caller that measured D A complex symmetric passes rayleigh_symmetrizer (2D FDFD, never with a Bloch phase): any
# factor of a complex A (SuperLU, line-block too) then takes the same floor and polish (DBR: ~124 applications at eps)
MIXED_EIGS_REFINEMENTS = 1
MIXED_ARPACK_TOLERANCE = 1e-6
# Tried and removed (2026-11): eigs on unrefined complex64 solves (half the solves: Mie eigs 7.0 -> 4.8 s), accepted
# when the ARPACK -> Rayleigh move d of k^2 had d^2 below the lambda / 1/Q bound, else refined Rayleigh-Ritz sweeps or
# a refined rerun. d^2 underestimates the quotient error (~ d^2 / gap): em_bor_cudss came out 1.3e-7 off against 1e-9;
# the sweeps lost a resonance (em_bor_fem_cudss), and a rerun from other start vectors missed em_conical_fem_cudss 1e-8
# fewer than n_resonances left after the PML filter: eigs again with 2x the eigenpairs on the same factor, at most this
# often (4x; PML modes crowd the target of an open model), and not after a doubling that found none. Each host ARPACK
# pass costs ~ its eigenpairs squared on the host (basis 5 x wanted): nanobeam eighth, 4.2e5 unknowns, 30 -> 120
# eigenpairs took eigs from ~25 s to 315 s and found 3 more, low-Q resonances. S5a: with an own cuDSS factor the next
# pass resumes the device Krylov-Schur decomposition of the last one (its m vectors kept, expanded to the new basis);
# around a host-only factor the host Krylov-Schur twin (eigs_host) resumes alike (rows reserved as on the device); host
# ARPACK (operators too small for it) restarts from the random start plus the Ritz vectors found (_widened_start).
# Mie sphere example (6.5e4 unknowns): pass 2 from scratch found nothing new; before S5a/S5b the two passes took 663 applications / 30 s.
# The first device pass reserves basis rows for one widening (WIDENING_BASIS_RESERVE x its basis): the widening then
# resumes in place (no second basis next to the first: Mie run pool 331 MiB, 81 + 161 rows + temporaries)
MAX_WIDENINGS = 2
WIDENING_BASIS_RESERVE = 2
# Resonance filter (_ResonanceFilter): Ritz vectors per batched sparse product (A V, B V, W V of a column block); its
# scratch is ~4 blocks of n x this complex128. The per-column host csr_matvec loops it replaced took 2.4 s of an 8 s
# tetrahedral Mie run (44k Nedelec unknowns, 48 eigenpairs over two passes) and 1.0 s of the conical pipe (194k)
FILTER_BLOCK_COLUMNS = 16
# host B v of a sparse B (_PencilMass.times: two per squared-Cayley application) cut into row blocks on this many threads
# from this many stored entries: scipy's CSR kernel is serial but releases the GIL (tetrahedral Mie PARDISO profile,
# 44k Nedelec unknowns: 609 serial products took 2.3 s, ~10 % of the run)
HOST_MASS_THREADS = 8
HOST_MASS_THREAD_MIN_ENTRIES = 500_000
_MASS_POOL = []   # the ThreadPoolExecutor of _mass_pool, made on first use


def _mass_pool():
    """Process-wide thread pool of the threaded host B products (HOST_MASS_THREADS workers)."""
    if not _MASS_POOL:
        from concurrent.futures import ThreadPoolExecutor
        _MASS_POOL.append(ThreadPoolExecutor(max_workers=HOST_MASS_THREADS, thread_name_prefix='pencil_mass'))
    return _MASS_POOL[0]


@dataclass(frozen=True, eq=False)
class Resonance:
    """One resonance: complex omega [rad/s] (Im > 0 decays), vacuum wavelength 2 pi C0 / Re omega [m], Q, PML share of
    the |u|^2 energy weights, the field (the caller's field_of of the unknowns, 1 real at its peak |u|) and the region
    share of the same weights (NaN without a region)."""
    omega: complex
    wavelength: float
    q_factor: float
    pml_fraction: float
    field: np.ndarray
    region_fraction: float = math.nan

    def text(self):
        """'lambda = 1536.54 nm, Q = 1023' (Q = inf when lossless)."""
        return f'lambda = {self.wavelength * 1e9:.6g} nm, Q = {self.q_factor:.4g}'


@dataclass(frozen=True, eq=False)
class EigenSolve:
    """Resonances of one solve, nearest the target first, and the solve info (timings, counts, warnings)."""
    resonances: list
    info: dict

    def summary(self):
        """One line for the study log: resonances found, unknowns, wall times, factor size, ARPACK counts."""
        info = self.info
        ordering = ' (nested dissection)' if info.get('nested_dissection') else ''
        factor = info.get('factor_text') or (f"SuperLU factorisation{ordering} {info['factor_seconds']:.3g} s, L + U "
                                             f"factors {info['factor_host_bytes'] / 2 ** 20:.4g} MiB of host memory")
        rayleigh = (f", Rayleigh quotients moved k^2 by up to {info['rayleigh_change']:.2g}"
                    if 'rayleigh_change' in info else '')
        widened = (f" in {info['widenings'] + 1} passes (too few non-PML resonances: eigenpairs doubled"
                   + (', basis kept)' if info.get('basis_kept') else ')') if info.get('widenings') else '')
        arnoldi = f" ({info['arnoldi']})" if info.get('arnoldi') else ''
        region = (f", {info.get('region_resonances_dropped', 0)} below the region energy fraction "
                  f"{info.get('min_region_fraction', 0.0):g} dropped" if info.get('region') else '')
        return (f"{len(self.resonances)} of {info['n_requested']} resonances near lambda = "
                f"{info['target_wavelength'] * 1e9:.6g} nm: {info['n_unknowns']:,} unknowns ({factor}; "
                f"eigs{arnoldi} {info['eigen_seconds']:.3g} s{widened}, {info['operator_applications']} operator "
                f"applications, "
                f"{info['eigenvalues_converged']} of {info['eigenvalues_requested']} eigenvalues, "
                f"{info['pml_resonances_dropped']} PML resonances dropped{region}{rayleigh})")


@dataclass(frozen=True, eq=False)
class _PencilMass:
    """B = -M of the pencil on flat unknowns: an (n,) complex128 diagonal (FDFD faces / cells) or a complex128 scipy
    CSR (FEM edge mass); `free` = rows of B not all zero (pinned rows have none)."""
    diagonal: object   # (n,) or None
    matrix: object     # scipy CSR or None

    @classmethod
    def of_mass(cls, mass):
        """B = -mass for an (n,) M diagonal or a square scipy sparse M."""
        import scipy.sparse
        if scipy.sparse.issparse(mass):
            if mass.shape[0] != mass.shape[1]:
                raise ValueError(f'solve_resonances needs a square sparse mass matrix, got shape {mass.shape}')
            return cls(diagonal=None, matrix=(-mass).tocsr().astype(np.complex128))
        return cls(diagonal=-np.asarray(mass, dtype=np.complex128).ravel(), matrix=None)   # a copy (unary minus)

    @property
    def size(self) -> int:
        return int(self.diagonal.size if self.matrix is None else self.matrix.shape[0])

    @property
    def free(self) -> np.ndarray:
        if self.matrix is None:
            return self.diagonal != 0
        rows = abs(self.matrix)
        return np.asarray(rows.sum(axis=1)).ravel() > 0.0

    def sparse(self):
        """Scipy sparse B."""
        import scipy.sparse
        return scipy.sparse.diags(self.diagonal) if self.matrix is None else self.matrix

    def scaled(self, row_scale):
        """D B for the (n,) diagonal D."""
        import scipy.sparse
        if self.matrix is None:
            return _PencilMass(diagonal=row_scale * self.diagonal, matrix=None)
        return _PencilMass(diagonal=None, matrix=(scipy.sparse.diags(row_scale) @ self.matrix).tocsr())

    @functools.cached_property
    def _row_blocks(self):
        """((start, stop, CSR rows sharing B's arrays), ...) of a sparse B cut at equal stored entries for the threaded
        host product; None for a diagonal B, below HOST_MASS_THREAD_MIN_ENTRIES or on one core."""
        import scipy.sparse
        threads = min(HOST_MASS_THREADS, os.cpu_count() or 1)
        if self.matrix is None or self.matrix.nnz < HOST_MASS_THREAD_MIN_ENTRIES or threads < 2:
            return None
        matrix = self.matrix
        n_rows, indptr = matrix.shape[0], matrix.indptr
        bounds = np.searchsorted(indptr, np.linspace(0, indptr[-1], threads + 1)).clip(0, n_rows)
        bounds[0], bounds[-1] = 0, n_rows
        blocks = []
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if stop > start:
                first, last = int(indptr[start]), int(indptr[stop])
                rows = scipy.sparse.csr_matrix((matrix.data[first:last], matrix.indices[first:last],
                                                indptr[start:stop + 1] - first), shape=(int(stop - start), n_rows))
                blocks.append((int(start), int(stop), rows))
        return tuple(blocks) if len(blocks) > 1 else None

    def times(self, vectors):
        """B @ vectors on the host, vectors (n,) or (n, k); a large sparse B by row blocks on _mass_pool threads."""
        if self.matrix is not None:
            blocks = self._row_blocks
            if blocks is None:
                return self.matrix @ vectors
            vectors = np.asarray(vectors)
            out = np.empty((self.matrix.shape[0],) + vectors.shape[1:], dtype=np.result_type(self.matrix.dtype, vectors))

            def product(block):
                start, stop, rows = block
                out[start:stop] = rows @ vectors
            list(_mass_pool().map(product, blocks))   # list: a worker's exception raises here
            return out
        return self.diagonal.reshape((-1,) + (1,) * (np.ndim(vectors) - 1)) * vectors

    def device_times(self):
        """Callable v -> B v on cupy vectors (B uploaded once)."""
        import cupy as cp
        if self.matrix is None:
            diagonal = cp.asarray(self.diagonal)
            return lambda vector: diagonal * vector
        import cupyx.scipy.sparse
        matrix = cupyx.scipy.sparse.csr_matrix(self.matrix)
        return lambda vector: matrix @ vector


def solve_resonances(host_operator, mass, omega_target, energy_weights, in_window, n_resonances, field_of=None,
                     spectral_map='shift_invert', warnings=(), eigen_solver=None, factorization=None, ordering=None,
                     tolerance_floor=0.0, basis_per_wanted=0, symmetrizer=None, symmetric=False, equilibration=None,
                     in_region=None, min_region_fraction=0.0, rayleigh_symmetrizer=None):
    """EigenSolve of up to n_resonances resonances nearest omega_target [rad/s] of the host operator A assembled there
    (module header) on flat unknowns: mass = complex M diagonal (0 = pinned row) or scipy sparse M (FEM; a pinned row
    is all zero), energy_weights = (n,) |u|^2 weights or a sparse real energy matrix W (per-row share Re(conj(u) W u))
    and in_window the non-PML rows of the PML share; field_of(vector) -> Resonance.field (None: the vector itself);
    spectral_map of SPECTRAL_MAPS; warnings = the caller's size-rule notes; eigen_solver (solvers.arpack_shift_invert,
    None = default settings) on `factorization` of A (solve_host, host_bytes, description; None = the eigen solver's
    factorize_host: host SuperLU with the elimination `ordering` of the unknowns, None = SuperLU's minimum degree, or
    cuDSS / MUMPS / PARDISO with their own ordering); ARPACK tolerance at least
    tolerance_floor, basis basis_per_wanted x the wanted eigenpairs unless the node sets one (0 = scipy's default;
    MAX_WIDENINGS doublings of the wanted eigenpairs while the PML filter leaves fewer than n_resonances);
    symmetrizer (n,) diagonal D with D A complex symmetric: an own cuDSS / MUMPS / PARDISO factor is the LDL^T of D A
    (D B the mass); symmetric: A itself is complex symmetric (FEM, no symmetrizer: the same LDL^T); equilibration (n,)
    > 0: the row scale of such a factor (None = its |diag|^-1/2); in_region (n,) bool rows of a region (None: none)
    and min_region_fraction in [0, 1] (module header); rayleigh_symmetrizer (n,) D with D A complex symmetric, checked
    by the caller: a complex A is polished (module header) on any factor, SuperLU and a given one too."""
    started = time.perf_counter()
    eigen_solver = ArpackShiftInvertSolver() if eigen_solver is None else eigen_solver
    omega_target = float(omega_target)
    if not (math.isfinite(omega_target) and omega_target > 0.0):
        raise ValueError(f'solve_resonances needs a finite target angular frequency > 0 [rad/s], got {omega_target!r}')
    if isinstance(n_resonances, bool) or int(n_resonances) != n_resonances or n_resonances < 1:
        raise ValueError(f'n_resonances must be an integer >= 1, got {n_resonances!r}')
    if spectral_map not in SPECTRAL_MAPS:
        raise ValueError(f"spectral_map must be one of {', '.join(SPECTRAL_MAPS)}, got {spectral_map!r}")
    n_resonances = int(n_resonances)
    b_mass = _PencilMass.of_mass(mass)
    n_unknowns = b_mass.size
    energy_weights = _energy_weights(energy_weights, n_unknowns)
    in_window = np.asarray(in_window, dtype=bool).ravel()
    if in_window.size != n_unknowns:
        raise ValueError(f'solve_resonances needs one window flag per unknown ({n_unknowns}), got {in_window.size}')
    in_region, min_region_fraction = region_filter(in_region, min_region_fraction, n_unknowns)
    warnings = list(warnings)
    free = b_mass.free
    n_free = int(np.count_nonzero(free))
    if n_free < 4:
        raise ValueError(f'solve_resonances: only {n_free} free unknowns remain outside the PEC / wall cells')
    shift = (omega_target / C0) ** 2
    factor_text = None if factorization is None else f"{factorization.description} factor of the linear solver"
    if factorization is None:   # own factor of A + (sigma - k_t^2) M, sigma = shift: never exactly on a resonance
        import scipy.sparse
        target_squared, shift = shift, shift * (1.0 + SHIFT_RELATIVE_OFFSET)
        host_operator = (host_operator - (shift - target_squared) * b_mass.sparse()).tocsr()
    on_device = False   # an own cuDSS factor: each application runs on the GPU, one vector transfer each way
    polish = False      # LDL^T of D A: ARPACK to MIXED_ARPACK_TOLERANCE, then the float64 Rayleigh quotient per vector
    refine = False      # a mixed polished factor: MIXED_EIGS_REFINEMENTS float64 steps per solve
    scaled_b = b_mass   # (D A)^-1 (D B) = A^-1 B: same theta and vectors
    factor_operator = host_operator
    row_scale = None    # D of an own symmetrized factor (the Rayleigh quotients use D A and D B)
    if factorization is None:
        # cuDSS / MUMPS / PARDISO factor the LDL^T of D A (SuperLU: A itself, the symmetrizer unused)
        symmetrized = symmetrizer is not None and eigen_solver.factorization in LDLT_FACTORIZATIONS
        if symmetrized:
            symmetrizer = row_scale = np.asarray(symmetrizer, dtype=np.complex128).ravel()
            scaled_b = b_mass.scaled(symmetrizer)
            factor_operator = (scipy.sparse.diags(symmetrizer) @ host_operator).tocsr()
        factorization = eigen_solver.factorize_host(factor_operator, ordering=ordering,
                                                    symmetric=symmetrized or bool(symmetric), row_scale=equilibration)
        on_device = factorization.back_end == 'cudss'
        # a mixed LDL^T: cuDSS on the device, MUMPS / PARDISO on the host (factor_precision 'double' after a fallback);
        # a double one of a complex A whose caller floored ARPACK (module constants)
        mixed = getattr(factorization, 'factor_precision', 'double') == 'mixed'
        polish = bool(getattr(factorization, 'symmetric', False)) and (
            mixed or (float(tolerance_floor) > 0.0 and _is_complex(host_operator)))
        refine = polish and mixed
        if polish:
            tolerance_floor = max(float(tolerance_floor), MIXED_ARPACK_TOLERANCE)
    if rayleigh_symmetrizer is not None and not polish and _is_complex(host_operator):
        # E2: D A complex symmetric (the caller measured it): any factor of A (SuperLU, line-block, double LDL^T) takes
        # ARPACK to MIXED_ARPACK_TOLERANCE and the float64 Rayleigh quotients of D A, D B; a real A keeps machine precision
        polish = True
        row_scale = np.asarray(rayleigh_symmetrizer, dtype=np.complex128).ravel()
        tolerance_floor = max(float(tolerance_floor), MIXED_ARPACK_TOLERANCE)
    factorized = time.perf_counter()
    n_wanted = min(2 * n_resonances + 10, n_free - 2)
    random = np.random.default_rng(_START_VECTOR_SEED)
    start_vector = (random.standard_normal(n_unknowns) + 1j * random.standard_normal(n_unknowns))
    start_vector[~free] = 0.0
    target_wavelength = 2.0 * math.pi * C0 / omega_target
    if on_device:
        b_times = scaled_b.device_times()
        solves_before = (factorization.n_solves, factorization.solve_seconds)

        def solve(vector):   # cudss_mixed: MIXED_EIGS_REFINEMENTS steps when polished, else to full accuracy
            return factorization.solve(vector, refinement_steps=MIXED_EIGS_REFINEMENTS if refine else None)
    else:
        b_times = scaled_b.times
        if refine:   # host mixed LDL^T (mumps_mixed / pardiso_mixed): MIXED_EIGS_REFINEMENTS steps per solve
            def solve(vector):
                return factorization.solve_host(vector, refinement_steps=MIXED_EIGS_REFINEMENTS)
        else:
            solve = factorization.solve_host
    if spectral_map == 'shift_invert':
        def apply_on_backend(vector):
            return solve(b_times(vector))
    else:
        def apply_on_backend(vector):   # squared Cayley map (squared_cayley_roots): two solves per application
            scaled = shift * solve(b_times(vector))   # k_t^2 v / (k^2 - k_t^2)
            return scaled + shift * solve(b_times(scaled))
    resonance_filter = _ResonanceFilter(host_operator, b_mass, energy_weights, in_window, in_region,
                                        row_scale if polish else None, on_gpu=on_device or eigen_solver.device_arnoldi,
                                        quotients=spectral_map != 'shift_invert' or polish)

    def eigen_passes(n_wanted, start_vector):
        """_EigenPasses of eigs on the current solves from n_wanted eigenpairs, doubled while the PML filter leaves
        fewer than n_resonances (MAX_WIDENINGS; a device / host Krylov-Schur widening resumes its decomposition)."""
        applications, widenings, previous_count, basis_kept = 0, 0, -1, False
        kept_decomposition = None   # device Krylov-Schur: the last pass's decomposition, resumed by a widening (S5a)
        while True:   # PML modes crowding the target: eigs again with twice the eigenpairs on the same factor
            default_basis = int(basis_per_wanted) * n_wanted
            basis = arnoldi_basis_size(n_wanted, n_unknowns, eigen_solver.subspace_size, default_basis)
            # S5b: the device Krylov-Schur on an own cuDSS factor (no transfers), and around a host factor on a gpu
            # runtime where eigs would run it too (device_arnoldi_fits): then a widening resumes it there as well
            host_device = (not on_device and eigen_solver.device_arnoldi
                           and device_arnoldi_fits(n_unknowns, basis, host_matvec=True))
            reserve_basis = WIDENING_BASIS_RESERVE * basis if MAX_WIDENINGS and widenings == 0 else 0
            eigs_host = getattr(eigen_solver, 'eigs_host', None)   # host Krylov-Schur: a host widening resumes too
            if on_device or host_device:
                basis_kept = basis_kept or kept_decomposition is not None
                theta, vectors, pass_applications, converged, kept_decomposition = eigen_solver.eigs_device(
                    apply_on_backend, n_unknowns, n_wanted, start_vector, tolerance_floor=tolerance_floor,
                    default_basis=default_basis, resume=kept_decomposition, matvec_on_host=host_device,
                    reserve_basis=reserve_basis)
            elif eigs_host is not None:
                basis_kept = basis_kept or kept_decomposition is not None
                theta, vectors, pass_applications, converged, kept_decomposition = eigs_host(
                    apply_on_backend, n_unknowns, n_wanted, start_vector, tolerance_floor=tolerance_floor,
                    default_basis=default_basis, resume=kept_decomposition, reserve_basis=reserve_basis)
            else:
                theta, vectors, pass_applications, converged = eigen_solver.eigs(
                    apply_on_backend, n_unknowns, n_wanted, start_vector, tolerance_floor=tolerance_floor,
                    default_basis=default_basis)
                kept_decomposition = None
            applications += pass_applications
            theta = np.asarray(theta, dtype=np.complex128)
            candidates, dropped, rayleigh_change = _filtered_candidates(
                theta, vectors, spectral_map, shift, resonance_filter, omega_target, polish, min_region_fraction)
            # stop when enough, at the limit, or when the last doubling found no new resonance (only PML modes near)
            if (len(candidates) >= n_resonances or len(candidates) == previous_count or widenings == MAX_WIDENINGS
                    or n_wanted >= n_free - 2):
                break
            previous_count = len(candidates)
            n_wanted = min(2 * n_wanted, n_free - 2)
            widenings += 1
            if kept_decomposition is None:   # host ARPACK takes one start vector: the random one + the Ritz vectors
                start_vector = _widened_start(start_vector, vectors, free)
        return _EigenPasses(theta=theta, vectors=np.asarray(vectors), candidates=candidates, dropped=dropped,
                            rayleigh_change=rayleigh_change, applications=applications, widenings=widenings,
                            n_wanted=n_wanted, converged=converged, basis_kept=basis_kept)

    passes = eigen_passes(n_wanted, start_vector)   # the decomposition is freed on return
    resonance_filter = None   # frees the device copies before the fields are built
    theta, candidates, dropped, n_wanted = passes.theta, passes.candidates, passes.dropped, passes.n_wanted
    converged, rayleigh_change, basis_kept = passes.converged, passes.rayleigh_change, passes.basis_kept
    applications, widenings = passes.applications, passes.widenings
    arnoldi = eigen_solver.last_arnoldi
    if not converged:
        warnings.append(f'ARPACK converged {np.size(theta)} of {n_wanted} eigenvalues near lambda = '
                        f'{target_wavelength * 1e9:.6g} nm; resonances may be missing')
    eigen_done = time.perf_counter()
    if on_device:
        factor_text = _cudss_factor_text(factorization, factorization.n_solves - solves_before[0],
                                         factorization.solve_seconds - solves_before[1], polish)
    elif factor_text is None and is_host_direct_factor(factorization):   # an own MUMPS / PARDISO factor
        factor_text = _host_direct_factor_text(factorization, factorized - started, polish)
    resonances = [_resonance(omega, pml_fraction, vector if field_of is None else field_of(vector), region_fraction)
                  for _, omega, pml_fraction, region_fraction, vector in candidates[:n_resonances]]
    if len(resonances) < n_resonances:
        region_text = (f" and {dropped['region']} below the region energy fraction {min_region_fraction:g}"
                       if in_region is not None else '')
        warnings.append(f'{len(resonances)} of {n_resonances} resonances found near lambda = '
                        f"{target_wavelength * 1e9:.6g} nm ({dropped['pml']} PML resonances{region_text} dropped among "
                        f'the {n_wanted} eigenvalues nearest it): set the wavelength nearer the resonance (a coarse mesh '
                        'shifts it), raise n_resonances, or resolve the cladding and PML better (PML modes crowd a '
                        'coarse one)' + (', or lower min_region_fraction' if dropped['region'] else ''))
    info = {'n_unknowns': n_unknowns, 'n_requested': n_resonances, 'target_wavelength': target_wavelength,
            'factor_host_bytes': factorization.host_bytes, 'factor_seconds': factorized - started,
            'eigen_seconds': eigen_done - factorized, 'seconds': time.perf_counter() - started,
            'operator_applications': applications, 'eigenvalues_requested': n_wanted, 'widenings': widenings,
            'eigenvalues_converged': int(theta.size), 'pml_resonances_dropped': dropped['pml'], 'warnings': warnings,
            'nested_dissection': factor_text is None and ordering is not None, 'region': in_region is not None,
            'region_resonances_dropped': dropped['region'], 'min_region_fraction': min_region_fraction}
    if factor_text is not None:
        info['factor_text'] = factor_text
    if arnoldi in ('device Krylov-Schur', 'host Krylov-Schur'):
        info['arnoldi'] = arnoldi
    info['basis_kept'] = basis_kept
    if rayleigh_change is not None:
        info['rayleigh_change'] = rayleigh_change
    return EigenSolve(resonances=resonances, info=info)


def _is_complex(host_operator) -> bool:
    """Whether a scipy sparse operator has a non-zero imaginary entry (PML, loss)."""
    return bool(np.any(host_operator.data.imag)) if np.iscomplexobj(host_operator.data) else False


@dataclass(frozen=True, eq=False)
class _EigenPasses:
    """The eigs passes of solve_resonances: the last pass's theta, host vectors, candidates (_filtered_candidates),
    dropped counts and Rayleigh change; applications / widenings of all passes, final eigenpairs wanted, converged."""
    theta: np.ndarray
    vectors: np.ndarray
    candidates: list
    dropped: dict
    rayleigh_change: object   # float or None
    applications: int
    widenings: int
    n_wanted: int
    converged: bool
    basis_kept: bool


def _widened_start(start_vector, vectors, free):
    """Start vector of a host ARPACK widening (S5a): the earlier random start plus the earlier pass's unit Ritz vectors,
    each part of equal norm (the found directions enter the new Krylov space at once, the random part keeps every
    other direction reachable), pinned rows 0."""
    vectors = np.asarray(vectors)
    if not vectors.size:
        return start_vector
    found = np.sum(vectors / np.maximum(np.linalg.norm(vectors, axis=0), 1e-300), axis=1)
    found_norm = float(np.linalg.norm(found))
    if not found_norm > 0.0:
        return start_vector
    widened = start_vector / np.linalg.norm(start_vector) + found / found_norm
    widened[~free] = 0.0
    return widened


def _energy_weights(energy_weights, n_unknowns):
    """(n,) float64 weights or a real scipy CSR energy matrix of n_unknowns rows (ValueError otherwise)."""
    import scipy.sparse
    if scipy.sparse.issparse(energy_weights):
        if energy_weights.shape != (n_unknowns, n_unknowns):
            raise ValueError(f'solve_resonances needs an ({n_unknowns}, {n_unknowns}) energy matrix, got shape '
                             f'{energy_weights.shape}')
        return energy_weights.tocsr()
    energy_weights = np.asarray(energy_weights, dtype=np.float64).ravel()
    if energy_weights.size != n_unknowns:
        raise ValueError(f'solve_resonances needs one energy weight per unknown ({n_unknowns}), got '
                         f'{energy_weights.size}')
    return energy_weights


class _ResonanceFilter:
    """A, B = -M, the energy weights W, the window / region rows and the polish row scale D of solve_resonances, on the
    device (on_gpu) or the host: per-column measures of a pass's Ritz vectors by one sparse product per operator;
    quotients False (plain shift-invert, no polish): A and B are neither stored nor applied, their measures are 0."""
    _MEASURES = ('mass', 'operator', 'scaled_operator', 'scaled_mass', 'energy', 'window', 'region')

    def __init__(self, host_operator, b_mass, energy_weights, in_window, in_region, row_scale, on_gpu,
                 quotients=True):
        if on_gpu:
            import cupy as xp
            import cupyx.scipy.sparse

            def stored(matrix):   # complex128 CSR (a complex cuSPARSE product of the complex vectors)
                return cupyx.scipy.sparse.csr_matrix(matrix.tocsr().astype(np.complex128))
        else:
            xp = np

            def stored(matrix):
                return matrix.tocsr()
        self.xp = xp
        self.quotients = bool(quotients)
        self.operator = stored(host_operator) if self.quotients else None
        self.mass_diagonal = b_mass.matrix is None
        self.mass = (None if not self.quotients else
                     xp.asarray(b_mass.diagonal)[:, None] if self.mass_diagonal else stored(b_mass.matrix))
        self.weights_diagonal = isinstance(energy_weights, np.ndarray)
        self.weights = (xp.asarray(energy_weights)[:, None] if self.weights_diagonal else stored(energy_weights))
        self.window = xp.asarray(in_window, dtype=np.float64)[:, None]
        self.region = None if in_region is None else xp.asarray(in_region, dtype=np.float64)[:, None]
        self.row_scale = None if row_scale is None else xp.asarray(row_scale)[:, None]

    def measures(self, vectors) -> dict:
        """Host (k,) arrays per column v of host vectors (n, k): mass v^H B v, operator v^H A v, scaled_operator
        v^T D A v, scaled_mass v^T D B v (D = 1 without a row scale), energy / window / region sums of the row energies
        |v|^2 w or Re(conj(v) (W v)) (all rows / in_window / in_region; region 0 without one)."""
        xp = self.xp
        n_columns = vectors.shape[1]
        result = np.zeros((len(self._MEASURES), n_columns), dtype=np.complex128)
        for start in range(0, n_columns, FILTER_BLOCK_COLUMNS):
            block = slice(start, min(start + FILTER_BLOCK_COLUMNS, n_columns))
            columns = xp.asarray(np.ascontiguousarray(vectors[:, block]), dtype=np.complex128)
            conjugate = xp.conj(columns)
            energies = (xp.abs(columns) ** 2 * self.weights if self.weights_diagonal
                        else xp.real(conjugate * (self.weights @ columns)))
            region = (xp.zeros(columns.shape[1], dtype=np.float64) if self.region is None
                      else xp.sum(energies * self.region, axis=0))
            if self.quotients:
                applied = self.operator @ columns
                massed = self.mass * columns if self.mass_diagonal else self.mass @ columns
                scaled = columns if self.row_scale is None else self.row_scale * columns   # D v (v^T D = (D v)^T)
                quotient_parts = (xp.sum(conjugate * massed, axis=0), xp.sum(conjugate * applied, axis=0),
                                  xp.sum(scaled * applied, axis=0), xp.sum(scaled * massed, axis=0))
            else:
                quotient_parts = (xp.zeros(columns.shape[1], dtype=np.complex128),) * 4
            parts = xp.stack((*quotient_parts, xp.sum(energies, axis=0).astype(np.complex128),
                              xp.sum(energies * self.window, axis=0).astype(np.complex128),
                              region.astype(np.complex128)))
            result[:, block] = parts.get() if xp is not np else parts   # one read per block
        return dict(zip(self._MEASURES, result))


def region_filter(in_region, min_region_fraction, n_unknowns):
    """(bool (n,) region rows or None, min_region_fraction as float) checked: ValueError for a wrong size, a fraction
    outside [0, 1] or a fraction > 0 without a region."""
    min_region_fraction = float(min_region_fraction)
    if not 0.0 <= min_region_fraction <= 1.0:
        raise ValueError(f'min_region_fraction must lie in [0, 1], got {min_region_fraction!r}')
    if in_region is None:
        if min_region_fraction > 0.0:
            raise ValueError(f'min_region_fraction {min_region_fraction:g} needs a region (in_region)')
        return None, min_region_fraction
    in_region = np.asarray(in_region, dtype=bool).ravel()
    if in_region.size != n_unknowns:
        raise ValueError(f'solve_resonances needs one region flag per unknown ({n_unknowns}), got {in_region.size}')
    return in_region, min_region_fraction


def element_region_rows(dofs, free, element_mask) -> np.ndarray:
    """Bool per free unknown (dofs indices `free`) of a fem DofMap (element_position, element_dofs, n_dofs): the dofs of
    the elements in element_mask (bool per mesh element; elements outside the map are skipped)."""
    positions = dofs.element_position[np.asarray(element_mask, dtype=bool)]
    rows = np.zeros(int(dofs.n_dofs), dtype=bool)
    rows[dofs.element_dofs[positions[positions >= 0]].ravel()] = True
    return rows[np.asarray(free)]


def resonance_filter_time_item(backend, unknowns, filter_entries, eigenpairs):
    """Run-time TimeItem of the resonance filter per eigenpair of a pass (_ResonanceFilter; filter_entries = stored
    entries of A and a sparse B / W): on the gpu one upload, the sparse products and ~14 vector passes; on the cpu the
    batched host product (matrix read once per FILTER_BLOCK_COLUMNS). Count: the first pass, a guess up to the widenings."""
    from ...core import time_estimate as te
    from ...core.memory_estimate import COMPLEX128_BYTES, COMPLEX128_CSR_ENTRY_BYTES
    vector_bytes = COMPLEX128_BYTES * float(unknowns)
    if backend == 'gpu':
        ops = (te.Transfer(vector_bytes), te.Spmv(float(unknowns), float(filter_entries), COMPLEX128_BYTES),
               te.Elementwise(12.0 * vector_bytes, 2.0 * vector_bytes, launches=2), te.PythonCalls(2))
    else:
        ops = (te.HostStream(COMPLEX128_CSR_ENTRY_BYTES * float(filter_entries) / FILTER_BLOCK_COLUMNS
                             + COMPLEX128_BYTES * float(filter_entries) + 14.0 * vector_bytes), te.PythonCalls(2))
    passes = 2 ** (MAX_WIDENINGS + 1) - 1   # eigenpairs doubled per widening
    return te.TimeItem('resonance filter (Rayleigh quotients, PML shares)', 'eigenpair', float(eigenpairs), 'guess',
                       ops, limit=float(eigenpairs) * passes)


def _filtered_candidates(theta, vectors, spectral_map, shift, resonance_filter, omega_target, polish,
                         min_region_fraction=0.0):
    """(candidates (|omega - omega_t|, omega, pml_fraction, region_fraction, vector) nearest first, dropped counts
    {'pml', 'region'}, largest relative k^2 move of the polish or None) of one ARPACK pass, measures from
    resonance_filter (a _ResonanceFilter); polish: float64 Rayleigh quotients of D A, D B; min_region_fraction: region."""
    usable = np.isfinite(theta) & (np.abs(theta) > _THETA_FLOOR * float(np.max(np.abs(theta), initial=0.0)))
    vectors = np.asarray(vectors)[:, usable]
    measures = resonance_filter.measures(vectors)
    if spectral_map == 'shift_invert':
        k_squared = shift + 1.0 / theta[usable]
    else:
        k_squared = shift * squared_cayley_roots(theta[usable], measures['operator'], measures['mass'], shift)
    rayleigh_change = None
    if polish:   # D A complex symmetric: u^T D A u / u^T D M u is stationary, error ~ (vector error)^2
        numerators = measures['scaled_operator']
        denominators = -measures['scaled_mass']   # D M = -D B
        polished = shift - numerators / denominators   # A u = (k_t^2 - k^2) M u
        rayleigh_change = float(np.max(np.abs(polished - k_squared) / np.abs(k_squared), initial=0.0))
        k_squared = polished
    static = (np.zeros(k_squared.shape, dtype=bool) if spectral_map == 'shift_invert'
              else np.abs(k_squared) <= STATIC_RELATIVE_K2 * shift)   # discrete gradient: no resonance
    omegas = C0 * np.sqrt(k_squared)   # principal branch: Re >= 0
    candidates = []
    dropped = {'pml': 0, 'region': 0}
    totals, windows, regions = measures['energy'].real, measures['window'].real, measures['region'].real
    for column, omega in enumerate(omegas):
        if static[column] or not omega.real > 1e-6 * omega_target:   # static (hz Neumann constant) or spurious
            continue
        total = float(totals[column])
        pml_fraction = 1.0 - float(windows[column]) / total if total > 0.0 else 1.0
        if pml_fraction > PML_RESONANCE_FRACTION:
            dropped['pml'] += 1
            continue
        region_fraction = math.nan
        if resonance_filter.region is not None:
            region_fraction = float(regions[column]) / total if total > 0.0 else 0.0
            if region_fraction < min_region_fraction:
                dropped['region'] += 1
                continue
        candidates.append((abs(omega - omega_target), complex(omega), pml_fraction, region_fraction,
                           vectors[:, column]))
    candidates.sort(key=lambda candidate: candidate[0])
    return candidates, dropped, rayleigh_change


def _cudss_factor_text(factor, n_solves, solve_seconds, polish):
    """Study-log text of an own cuDSS factor: plan / factorise times, device memory, refinement, the device solves
    inside eigs (the rest of eigs: the device Krylov-Schur)."""
    if factor.factor_precision == 'mixed':
        steps = (f'{MIXED_EIGS_REFINEMENTS} float64 refinement step(s) per solve, ARPACK to '
                 f'{MIXED_ARPACK_TOLERANCE:g}, Rayleigh quotients' if polish
                 else f'float64 refinement to {factor.refinement_steps} step(s) per solve')
        refinement = f', unrefined residual {factor.single_residual:.2g}, {steps}'
    else:
        refinement = f', {factor.refinement_steps} refinement steps per solve' if factor.refinement_steps else ''
        if polish:
            refinement += f', ARPACK to {MIXED_ARPACK_TOLERANCE:g}, Rayleigh quotients'
    kind = factor.description if factor.asymmetry is None else (
        f'{factor.description} (scaled operator asymmetry {factor.asymmetry:.2g})')
    return (f'{kind} on the GPU: plan {factor.plan_seconds:.3g} s + factorise {factor.factorize_seconds:.3g} s, '
            f'factor + operator {factor.device_bytes / 2 ** 20:.4g} MiB of device memory{refinement}; {n_solves} '
            f'device solves {solve_seconds:.3g} s of the eigs time')


def _host_direct_factor_text(factor, factor_seconds, polish):
    """Study-log text of an own MUMPS / PARDISO factor (solvers.host_direct): kind, factorise time (analysis
    included), the library's factor bytes of host memory, refinement."""
    if factor.factor_precision == 'mixed':
        steps = (f'{MIXED_EIGS_REFINEMENTS} float64 refinement step(s) per solve, ARPACK to '
                 f'{MIXED_ARPACK_TOLERANCE:g}, Rayleigh quotients' if polish else 'float64 refinement per solve')
        single = getattr(factor, 'single_residual', None)
        refinement = (f', unrefined residual {single:.2g}' if single is not None else '') + f', {steps}'
    else:   # a double factor whose probe missed: float64 GMRES on its own factor (refinement_steps: the probe's)
        steps = getattr(factor, 'refinement_steps', 0)
        refinement = f', refined solves (probe: {steps} float64 GMRES steps)' if steps else ''
        if polish:
            refinement += f', ARPACK to {MIXED_ARPACK_TOLERANCE:g}, Rayleigh quotients'
    asymmetry = getattr(factor, 'asymmetry', None)
    kind = factor.description if asymmetry is None else (
        f'{factor.description} (scaled operator asymmetry {asymmetry:.2g})')
    return (f'{kind} on the host: factorise {factor_seconds:.3g} s, factor {(factor.host_bytes or 0) / 2 ** 20:.4g} '
            f'MiB of host memory{refinement}{library_solve_text(factor)}')


def _resonance(omega, pml_fraction, field, region_fraction=math.nan):
    """Resonance of a complex omega with its field scaled to 1 (real) at the first cell of largest |u|."""
    peak_index = int(np.argmax(np.abs(field)))
    field = field / field.flat[peak_index]
    return Resonance(omega=omega, wavelength=2.0 * math.pi * C0 / omega.real, q_factor=q_factor_of(omega),
                     pml_fraction=pml_fraction, field=field, region_fraction=float(region_fraction))


def q_factor_of(omega):
    """Q = Re omega / (2 Im omega) of a complex resonance omega [rad/s]; inf when |Im| is round-off (lossless)."""
    omega = complex(omega)
    if abs(omega.imag) <= _LOSSLESS_RELATIVE_IMAG * omega.real:
        return math.inf
    return omega.real / (2.0 * omega.imag)


def squared_cayley_roots(theta, operator_products, mass_products, target_squared):
    """k^2 / k_t^2 of each squared-Cayley theta (theta x^2 - (2 theta + 1) x + theta = 0: roots x and 1 / x), the root
    nearer the Rayleigh quotient 1 + v^H A v / (k_t^2 v^H B v) of its eigenvector v (operator_products = v^H A v,
    mass_products = v^H B v per theta); 0 if B-null."""
    # The map A^-1 M k_t^2 (I + A^-1 M k_t^2) with A = K - k_t^2 M (K = the curl-curl part): theta = k_t^2 k^2 / (k^2 -
    # k_t^2)^2 sends the gradient null space (k = 0) and massless rows (k = inf) to 0, and |theta| is equal at k and
    # k_t^2 / k, so the 'LM' slots go to the resonances nearest the target in |ln(k / k_t)| on both sides. The quotient
    # is invariant under negating A and M together (the FDFD A = L + k_t^2 M, B = -M of solve_resonances).
    theta = np.asarray(theta, dtype=np.complex128)
    mass_products = np.asarray(mass_products, dtype=np.complex128)
    roots = np.zeros(theta.shape, dtype=np.complex128)
    massive = mass_products != 0.0
    value = theta[massive]
    root_term = np.sqrt(4.0 * value + 1.0)
    plus, minus = 2.0 * value + 1.0 + root_term, 2.0 * value + 1.0 - root_term
    large_root = np.where(np.abs(plus) >= np.abs(minus), plus, minus) / (2.0 * value)   # |x| >= 1 (no cancellation)
    quotient = 1.0 + np.asarray(operator_products, dtype=np.complex128)[massive] / (target_squared
                                                                                    * mass_products[massive])
    roots[massive] = np.where(np.abs(large_root - quotient) <= np.abs(1.0 / large_root - quotient), large_root,
                              1.0 / large_root)
    return roots


# ---- resonance_table feature (reads the solver's last EigenSolve) ----
@dataclass(frozen=True, eq=False)
class ResonanceRegion:
    """Region of a resonance_table feature: host bool per full-grid cell (2D / 3D tensor) or per mesh element
    (triangles, tetrahedra) and the min_region_fraction below which a resonance is dropped."""
    cells: np.ndarray
    min_fraction: float = 0.0


class ResonanceTable:
    """Marker object of a resonance_table feature (its ResonanceRegion or None): its result reads
    FDFDPhysics.eigen_solve."""

    def __init__(self, region=None):
        self.region = region


def resonance_table_result(table, physics) -> dict:
    """'table' of the last eigenfrequency solve of `physics` (RESONANCE_TABLE_COLUMNS, + REGION_COLUMN when the solve
    had a region; nearest the target first); extras target_wavelength [m], selected_resonance; ValueError before such
    a solve."""
    solve = physics.eigen_solve
    if solve is None:
        raise ValueError('not solved yet: the resonance table needs an eigenfrequency study')
    with_region = bool(solve.info.get('region'))
    rows = []
    for number, resonance in enumerate(solve.resonances, start=1):
        row = [number, resonance.wavelength, resonance.omega.real / (2.0 * math.pi),
               resonance.omega.imag / (2.0 * math.pi), resonance.q_factor, resonance.pml_fraction]
        if with_region:
            row.append(float(getattr(resonance, 'region_fraction', math.nan)))
        rows.append(row)
    columns = RESONANCE_TABLE_COLUMNS + ((REGION_COLUMN[0],) if with_region else ())
    units = RESONANCE_TABLE_UNITS + ((REGION_COLUMN[1],) if with_region else ())
    return {'kind': 'table', 'columns': list(columns), 'units': list(units), 'rows': rows,
            'target_wavelength': float(solve.info['target_wavelength']),
            'selected_resonance': physics.selected_resonance}
