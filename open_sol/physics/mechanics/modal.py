"""Vibration modes (eigenfrequencies) K u = omega^2 M u on the free dofs, host numpy / scipy: shift-invert Lanczos of an
assembled host pencil about the target (the 'eigen' role: SuperLU, or cuDSS LDL^T on the GPU), or matrix-free at sigma =
0 (3D tensor meshes); undamped. Several targets or a frequency band: ModalPencil, solve_modes_at_shifts, solve_band_modes."""
# K = the assembled stiffness, M = the consistent mass (rho N_k N_l of the solid elements), both restricted to the free
# dofs (held and unconnected dofs carry u = 0: supports are homogeneous in a modal solve, prescribed displacements, loads
# and thermal strain play no part). ArpackShiftInvertSolver.eigsh_shifted (scipy eigsh in shift-invert mode about sigma =
# omega_t^2 (1 + 1e-4), off a resonance on the target, on a real SuperLU factor of K - sigma M) returns the k eigenvalues
# omega^2 nearest sigma; they are ranked by |omega - omega_t| and the first n kept. The rigid-mode checks keep K positive
# definite, so sigma = 0 (the lowest modes, no offset) is never singular; a free body (tetrahedra, target > 0: the rigid
# check counts its free motions instead) keeps its rigid modes at omega^2 ~ 0, asked for on top (rigid_motions) and
# dropped below RIGID_EIGENVALUE_FRACTION x the shift. The same driver serves the 2.5D guided modes
# (physics/mechanics/guided_modes.py). A resistive electrode load (solve_loaded_modes) adds j B / omega on its master
# potential: T(omega) is frozen at the target (complex symmetric, eigs_shifted) and each mode's omega refined by Newton on
# x^T T(omega) x = 0 (x fixed). An elastic PML shell (pml3d) and a loss factor K_el (1 + j eta) (new_modules_plan K3)
# make K (and M) complex but keep the pencil linear in omega^2 (solve_damped_modes: eigs_shifted, no refinement); PML
# modes mostly in the shell are dropped. A prestressed pencil (K2) only adds the real K_sigma to K. Bloch cells
# (bloch3d): a prolongation P replaces the free selection, the pencil
# P^H K P, P^H M P is complex Hermitian (scipy eigs inside eigsh on a complex factor) and the modes complex (peak
# component made real positive); a free cell keeps rigid modes, solved only about a target > 0, whose omega^2 below
# RIGID_EIGENVALUE_FRACTION x the shift are dropped.
# Several shifts (B1): ModalPencil reduces K, M to the free dofs once (kinds 'symmetric' / 'damped' / 'loaded') and keeps
# one factor across shifts (the shifted operator's pattern does not depend on sigma: a cuDSS plan is reused). The single-
# target functions above are ModalPencil.solve_at + release. solve_modes_at_shifts = the union of 'nearest' solves at
# several targets. solve_band_modes = every mode with f_lo <= Re f < f_hi: shifts march upward, a solve's trusted radius
# r (its farthest eigenvalue from sigma) covers [sigma - r, sigma + r]; uncovered gaps get a shift at their middle; for a
# 'symmetric' pencil Sylvester inertia counts at the band edges and at every shift give the mode count per interval, and
# an interval short of its count gets a repair solve (doubled basis, other start vector). Damped / PML / loaded pencils
# (and Hermitian ones on cuDSS) march with 50 % overlap and no count (warning). Merge: by Re lambda; a cluster within
# MERGE_TOLERANCE relative keeps only its independent vectors (M-orthonormalised for 'symmetric'), nearest-shift first,
# so degenerate pairs survive and duplicates of two shifts go. Results of several shifts are in frequency order.
import math
import time
from dataclasses import dataclass, replace

import numpy as np

from ...solvers.arpack_shift_invert import CUDSS_FACTORIZATIONS, ArpackShiftInvertSolver

MODAL_TABLE_COLUMNS = ('resonance_number', 'frequency', 'angular_frequency')
MODAL_TABLE_UNITS = ('1', 'Hz', 'rad/s')
# appended when a resistive electrode load, an elastic PML or a loss factor damps the modes: Im f (> 0 decays), Re omega / (2 Im
# omega), 2 Im omega
DAMPED_TABLE_COLUMNS = ('frequency_imag', 'q_factor', 'energy_decay_rate')
DAMPED_TABLE_UNITS = ('Hz', '1', '1/s')
# loaded modes: Newton steps on the Rayleigh functional of each frozen-pencil mode, stop at |d omega| <= this x |omega|
NEWTON_STEPS = 30
NEWTON_TOLERANCE = 1e-15
SPLU_WARN_DOFS = 1_000_000     # free dofs: the host SuperLU factor of K - sigma M grows faster than the dofs
SPLU_LIMIT_DOFS = 2_000_000    # L + U of order 10 GB of host memory
EXTRA_EIGENVALUES = 4          # eigenvalues asked beyond n_resonances (ranking by |omega - omega_t|, ARPACK margin)
RIGID_EIGENVALUE_FRACTION = 1e-8   # |omega^2| at or below this x omega_t^2: a rigid-body mode (f < 1e-4 f_t), dropped
RIGID_TEXT_HEAD = 'rigid-body mode(s)'   # warning text after the count of dropped rigid-body modes
_START_VECTOR_SEED = 6         # deterministic Lanczos start vector
# Matrix-free (3D): an inner CG to relative residual tol leaves a Ritz-residual floor of order tol, so the Lanczos
# tolerance is at least this factor x the CG tolerance (else ARPACK restarts until max_restarts); restarts are capped
# since each operator application is a full PCG solve.
LANCZOS_TOLERANCE_PER_CG = 1e3
MAX_LANCZOS_RESTARTS = 100
# Several shifts (module header)
MERGE_TOLERANCE = 1e-8         # |lambda_i - lambda_j| <= this x |lambda|: one cluster of the merge
LOADED_MERGE_TOLERANCE = 1e-5  # loaded modes: Newton on frozen pencils of two shifts agrees to second order only
DUPLICATE_RESIDUAL = 1e-3      # a cluster vector whose part outside the kept ones is below this (relative): a duplicate
BAND_STEP_FRACTION = 0.5       # counted band: next shift = covered edge + this x the last trusted radius
BAND_FIRST_SHIFT_FRACTION = 1e-2   # band from 0 on a pencil needing a shift > 0: first shift at this x omega_hi
BAND_SHIFTS_PER_SLICE = 4      # marching budget: this x ceil(max_modes / n_per_slice) shifts (+ MAX_GAP_SHIFTS)
MAX_GAP_SHIFTS = 8             # shifts filling uncovered gaps, per band solve
MAX_COUNT_REPAIRS = 4          # repair solves of intervals short of their inertia count, per band solve
_UNCOUNTED_BAND = ('band coverage not verified (no inertia count for a damped, PML, loaded or complex Bloch pencil on '
                   'this factorization): shifts overlap by half their trusted radius, modes may be missing')
_TINY = 1e-300


@dataclass(frozen=True, eq=False)
class MechanicalResonance:
    """One vibration mode: omega = Re omega [rad/s], frequency [Hz], the nodal displacement (host float64 over every dof,
    0 on held ones; complex128 for a Bloch or a loaded mode) scaled to a peak node |u| of 1 m, its largest component
    there real positive; frequency_imag = Im f [Hz] (> 0 decays, exp(+j omega t); 0 undamped)."""
    omega: float
    frequency: float
    displacement: np.ndarray
    frequency_imag: float = 0.0

    @property
    def q_factor(self) -> float:
        """Re f / (2 Im f); inf undamped."""
        return self.frequency / (2.0 * self.frequency_imag) if self.frequency_imag else math.inf

    def text(self):
        """'f = 312456 Hz' (the snapshot label after 'resonance k: '), with ', Q = ...' when damped."""
        if self.frequency_imag:
            return f'f = {self.frequency:.6g} Hz, Q = {self.q_factor:.4g}'
        return f'f = {self.frequency:.6g} Hz'


@dataclass(frozen=True, eq=False)
class ModalSolve:
    """Resonances of one modal solve (one target: nearest first; several targets or a band: frequency order) and the
    solve info (timings, counts, warnings)."""
    resonances: list
    info: dict

    def summary(self):
        """One line for the study log: resonances found, dofs, wall times, factor size, Lanczos counts."""
        info = self.info
        if info.get('method') == 'matrix_free':
            return (f"{len(self.resonances)} of {info['n_requested']} lowest vibration modes: {info['n_free']:,} free "
                    f"dofs (matrix-free shift-invert at 0: {info['operator_applications']} stiffness PCG solves, "
                    f"{info.get('cg_iterations', 0):,} CG iterations, {info['eigen_seconds']:.3g} s; Lanczos tolerance "
                    f"{info['lanczos_tolerance']:.1e}, {info['eigenvalues_converged']} of "
                    f"{info['eigenvalues_requested']} eigenvalues)")
        guided = info.get('axial_wavenumber')
        kind = '' if guided is None else f'guided (k_z = {guided:.6g} rad/m) '
        if info.get('bloch'):
            kind = f"Bloch ({info['bloch']}) "
        loads = info.get('electrode_loads')
        if loads:
            kind = 'loaded '
        if info.get('loss'):
            kind = f"{kind}damped ({info['loss']}) "
        if info.get('pml'):
            kind = f"PML ({info['pml']}) " + ('damped ' if info.get('loss') else '')
        driver = 'eigs' if loads or info.get('damped') else 'eigsh'
        device_bytes = info.get('factor_device_bytes', 0)
        memory = (f"{device_bytes / 2 ** 20:.4g} MiB of device memory" if device_bytes
                  else f"L + U factors {info['factor_host_bytes'] / 2 ** 20:.4g} MiB of host memory")
        split = (f" (cuDSS plan {info['factor_plan_seconds']:.3g} s, numeric {info['factor_numeric_seconds']:.3g} s)"
                 if 'factor_plan_seconds' in info else '')
        refinement = (f", {info['refinements_per_solve']:.2g} float64 refinement steps per solve"
                      if 'refinements_per_solve' in info else '')
        load_text = ''
        if loads:
            load_text = (f"; resistive load on {', '.join(f'{name} ({ohm:.4g} Ohm)' for name, ohm in loads)}: pencil "
                         f"frozen at the target, Newton per mode, largest residual {info['load_residual']:.1e}")
        if info.get('band') is not None:
            low, high = info['band']
            expected = info.get('expected_count')
            count = f'{expected} by inertia count' if expected is not None else 'count not verified'
            head = (f"{len(self.resonances)} {kind}vibration modes in f = {low:.6g} - {high:.6g} Hz ({count}, "
                    f"{info['shifts']} shifts)")
        elif info.get('targets'):
            head = (f"{len(self.resonances)} {kind}vibration modes near f = "
                    f"{', '.join(f'{target:.6g}' for target in info['targets'])} Hz ({info['shifts']} shifts)")
        else:
            head = (f"{len(self.resonances)} of {info['n_requested']} {kind}vibration modes near f = "
                    f"{info['target_frequency']:.6g} Hz")
        setup = ''
        if 'pencil_seconds' in info:   # the rest: candidates, merge, mode spreading, inertia bookkeeping
            rest = info['seconds'] - info['factor_seconds'] - info['eigen_seconds'] - info['pencil_seconds']
            setup = (f"; free-dof restriction {info['reduction_seconds']:.3g} s, shift pencil "
                     f"{info['pencil_seconds']:.3g} s, rest {rest:.3g} s")
        return (f"{head}: {info['n_free']:,} free dofs ({info.get('factor', 'SuperLU')} "
                f"factorisation {info['factor_seconds']:.3g} s{split}, {memory}; {driver} "
                f"{info['eigen_seconds']:.3g} s, {info['operator_applications']} operator applications{refinement}, "
                f"{info['eigenvalues_converged']} of {info['eigenvalues_requested']} eigenvalues{load_text}{setup})")


@dataclass(frozen=True, eq=False)
class _Candidate:
    """One usable eigenpair of a shift solve: lambda = omega^2, omega (complex; Newton-refined when loaded), the reduced
    eigenvector, |Re omega - omega_shift| (rank), the loaded Newton residual and the vector's norm (_inner, once)."""
    eigenvalue: complex
    omega: complex
    vector: np.ndarray
    distance: float
    residual: float = 0.0
    norm: float = 0.0


class ModalPencil:
    """K x = omega^2 M x reduced once to the free dofs (or to x = P x_r of a Bloch prolongation) for shift-invert solves
    at any number of shifts with one kept factor (plan reuse); kind 'symmetric' (eigsh), 'damped' (eigs: loss / PML
    shell) or 'loaded' (frozen j B / omega_t + Newton). release() frees the factor."""

    def __init__(self, stiffness, mass, free, components_per_node, eigen_solver=None, prolongation=None,
                 displacement_dofs=None, rigid_motions=0, damped=False, shell_dofs=None, energy_share=0.5, loads=None,
                 axial_wavenumber=None, bloch_text='', pml_text='', loss_text=''):
        import scipy.sparse
        self.eigen_solver = ArpackShiftInvertSolver() if eigen_solver is None else eigen_solver
        self.kind = 'loaded' if loads else ('damped' if damped or shell_dofs is not None else 'symmetric')
        if self.kind == 'loaded' and prolongation is not None:
            raise ValueError('ModalPencil: a resistive electrode load takes no Bloch prolongation')
        self.components_per_node = int(components_per_node)
        self.displacement_dofs = displacement_dofs
        self.rigid_motions = int(rigid_motions)
        self.energy_share = float(energy_share)
        self.axial_wavenumber = None if axial_wavenumber is None else float(axial_wavenumber)
        self.bloch_text, self.pml_text, self.loss_text = bloch_text, pml_text, loss_text
        self.loads = list(loads or ())
        self.n_dofs = int(stiffness.shape[0])
        self.factor = None
        self.shift_pencil = None   # the eigen solver's ShiftPencil (cuDSS: a DevicePencil), kept with the factor
        self.prolongation = None if prolongation is None else prolongation.tocsr()
        if self.prolongation is None:
            self.free_dofs = np.flatnonzero(np.asarray(free, dtype=bool))
            self.n_free = int(self.free_dofs.size)
        else:
            self.free_dofs = None
            self.n_free = int(self.prolongation.shape[1])
        self.size_warnings = []
        self.on_device = _check_size(self.n_free, self.eigen_solver, self.size_warnings)
        started = time.perf_counter()
        self.stiffness, self.mass = self._reduced(stiffness), self._reduced(mass)
        self.reduction_seconds = time.perf_counter() - started   # H6 timers: restriction and shift pencil set-up
        self.pencil_seconds = 0.0
        self.in_shell = None
        if shell_dofs is not None:
            shell = np.asarray(shell_dofs, dtype=bool)
            self.in_shell = (shell[self.free_dofs] if self.prolongation is None
                             else np.asarray(np.abs(self.prolongation).T @ shell) > 0)
        self.load_matrix = self.load_positions = self.load_values = None
        if self.kind == 'loaded':
            position = np.full(self.n_dofs, -1, dtype=np.int64)
            position[self.free_dofs] = np.arange(self.n_free)
            self.load_positions = position[np.array([dof for _, dof, _, _ in self.loads], dtype=np.int64)]
            if np.any(self.load_positions < 0):
                raise ValueError('a resistively loaded electrode has no free potential (every node held or inactive)')
            self.load_values = np.array([value for _, _, value, _ in self.loads], dtype=np.float64)
            self.load_matrix = scipy.sparse.csr_matrix(
                (self.load_values, (self.load_positions, self.load_positions)), shape=(self.n_free, self.n_free))

    def _reduced(self, matrix):
        """Host CSR of the free-dof block (or P^H A P); float64 except for a damped (complex) pencil."""
        matrix = matrix.tocsr() if self.kind == 'damped' else matrix.tocsr().astype(np.float64)
        if self.prolongation is None:
            return matrix[self.free_dofs][:, self.free_dofs].tocsr()
        return (self.prolongation.conj().T.tocsr() @ matrix @ self.prolongation).tocsr()

    @property
    def needs_positive_target(self) -> bool:
        """True when a shift at 0 is refused or singular (loaded, PML shell, free rigid motions)."""
        return self.kind == 'loaded' or self.in_shell is not None or self.rigid_motions > 0

    def spread(self, vector):
        """Reduced eigenvector -> every dof (0 on held ones; complex128 unless 'symmetric')."""
        if self.prolongation is not None:
            full = np.asarray(self.prolongation @ vector)
            return full if self.kind == 'symmetric' else full.astype(np.complex128)
        full = np.zeros(self.n_dofs, dtype=vector.dtype if self.kind == 'symmetric' else np.complex128)
        full[self.free_dofs] = vector
        return full

    def release(self) -> None:
        """Free the kept factor (a cuDSS factor lives outside the cupy pool) and drop the shift pencil."""
        factor, self.factor = self.factor, None
        self.shift_pencil = None
        if factor is not None and hasattr(factor, 'release'):
            factor.release()

    def _pencil(self):
        """The eigen solver's ShiftPencil of this pencil, built on first use: (K, M) real symmetric or Hermitian
        ('symmetric'), complex symmetric (K, M) ('damped') or (K, B, M) with a = (1, j / omega_t) ('loaded')."""
        if self.shift_pencil is None:
            started = time.perf_counter()
            # vouched symmetric: FEM K (+ K_sigma, loss, PML stretch: complex symmetric), M and the diagonal B are
            # symmetric by construction, kept by a free-dof selection or a real P^T A P; a complex (Bloch) P^H A P of a
            # complex A is not, so the device symmetry check stays there
            vouched = self.prolongation is None or not np.iscomplexobj(self.prolongation.data)
            if self.kind == 'symmetric':
                hermitian = np.iscomplexobj(self.stiffness.data) or np.iscomplexobj(self.mass.data)
                self.shift_pencil = self.eigen_solver.shift_pencil(
                    (self.stiffness,), self.mass, np.complex128 if hermitian else np.float64, symmetric=not hermitian,
                    symmetric_known=vouched)
            else:
                terms = (self.stiffness,) if self.kind == 'damped' else (self.stiffness, self.load_matrix)
                self.shift_pencil = self.eigen_solver.shift_pencil(terms, self.mass, np.complex128, symmetric=True,
                                                                   symmetric_known=vouched)
            self.pencil_seconds += time.perf_counter() - started
        return self.shift_pencil

    def solve_at(self, omega_target, n_resonances):
        """ModalSolve of up to n_resonances modes nearest omega_target [rad/s], nearest first (the factor is kept: the
        caller releases)."""
        started = time.perf_counter()
        omega_target = self._checked_target(omega_target)
        n_resonances = _checked_count(n_resonances)
        before = _solver_tally(self.eigen_solver)
        warnings = list(self.size_warnings)
        n_wanted = self._n_wanted(n_resonances)
        eigenvalues, vectors, _ = self._shift_solve(omega_target, n_wanted, warnings)
        kept = sorted(self._candidates(eigenvalues, vectors, omega_target, warnings,
                                       RIGID_EIGENVALUE_FRACTION * omega_target ** 2),
                      key=lambda candidate: candidate.distance)[:n_resonances]
        resonances = [self._resonance(candidate) for candidate in kept]
        self._growing_warning(resonances, warnings)
        if len(resonances) < n_resonances:
            warnings.append(_too_few_text(len(resonances), n_resonances, omega_target))
        info = self._info(started, before, n_resonances, omega_target / (2.0 * math.pi), n_wanted,
                          int(np.size(eigenvalues)), warnings, kept)
        return ModalSolve(resonances=resonances, info=info)

    def count_below(self, eigenvalue):
        """ShiftInertia (eigenvalues omega^2 below `eigenvalue`) of a 'symmetric' pencil, None when uncountable (other
        kinds, a Hermitian pencil on cuDSS); the factor is kept for plan reuse."""
        if self.kind != 'symmetric':
            return None
        try:
            shift_inertia, self.factor = self.eigen_solver.count_below(self.stiffness, self.mass, eigenvalue,
                                                                       previous=self.factor, pencil=self._pencil())
        except RuntimeError as exc:
            self.factor = None
            raise ValueError(f'eigenfrequency band: the inertia count factorisation of K - omega^2 M at f = '
                             f'{math.sqrt(max(eigenvalue, 0.0)) / (2.0 * math.pi):.6g} Hz failed ({exc})') from None
        return shift_inertia

    def _checked_target(self, omega_target) -> float:
        """omega_target as float; ValueError when not finite, negative, or <= 0 where the kind needs a shift > 0."""
        omega = float(omega_target)
        if self.kind == 'loaded':
            if not (math.isfinite(omega) and omega > 0.0):
                raise ValueError('a resistive electrode load needs an eigenfrequency target > 0 (its 1 / omega term is '
                                 f'evaluated there), got omega = {omega_target!r} rad/s')
        elif self.kind == 'damped':
            if not (math.isfinite(omega) and omega >= 0.0) or (self.in_shell is not None and omega <= 0.0):
                what = ('an elastic PML needs an eigenfrequency target > 0' if self.in_shell is not None else
                        'solve_damped_modes needs a finite target angular frequency >= 0')
                raise ValueError(f'{what}, got omega = {omega_target!r} rad/s')
        elif not (math.isfinite(omega) and omega >= 0.0):
            raise ValueError(f'solve_modes needs a finite target angular frequency >= 0 [rad/s], got {omega_target!r}')
        return omega

    def _n_wanted(self, n_resonances) -> int:
        """Eigenvalues asked per shift for n_resonances modes (+ EXTRA_EIGENVALUES, rigid motions; a PML shell twice)."""
        extra = n_resonances if self.in_shell is not None else 0
        limit = self.n_free - 1 if self.kind == 'symmetric' else self.n_free - 2   # eigsh k < n, eigs k < n - 1
        return min(n_resonances + extra + EXTRA_EIGENVALUES + self.rigid_motions, limit)

    def _shift_solve(self, omega_target, n_wanted, warnings, inertia=False, seed=_START_VECTOR_SEED):
        """(eigenvalues lambda, reduced vectors, ShiftInertia or None) of n_wanted eigenpairs nearest omega_target^2 on
        the kept factor (refactorised; 'symmetric' with inertia: the shift's count); ValueError on a failed factor."""
        solver = self.eigen_solver
        start_vector = np.random.default_rng(seed).standard_normal(self.n_free)
        try:
            if self.kind == 'symmetric':
                result = solver.eigsh_shifted(self.stiffness, self.mass, omega_target ** 2, n_wanted,
                                              start_vector=start_vector, previous=self.factor, inertia=inertia,
                                              pencil=self._pencil())
            else:
                coefficients = None if self.kind == 'damped' else (1.0, 1j / omega_target)
                result = solver.eigs_shifted(None, self.mass, omega_target ** 2, n_wanted, start_vector=start_vector,
                                             previous=self.factor, pencil=self._pencil(),
                                             stiffness_coefficients=coefficients)
        except RuntimeError as exc:
            self.factor = None
            raise ValueError(self._failure_text(exc, omega_target)) from None
        eigenvalues, vectors, _applications, converged, self.factor = result[:5]
        if not converged:
            warnings.append(f'ARPACK converged {np.size(eigenvalues)} of {n_wanted} eigenvalues near f = '
                            f'{omega_target / (2.0 * math.pi):.6g} Hz; modes may be missing')
        shift_inertia = result[5] if self.kind == 'symmetric' and inertia else None
        return eigenvalues, np.asarray(vectors), shift_inertia

    def _failure_text(self, exc, omega_target) -> str:
        """Study message of a failed shift factor (cuDSS failure, or a singular SuperLU factor)."""
        frequency = omega_target / (2.0 * math.pi)
        if self.kind == 'symmetric':
            prefix = ''
        elif self.kind == 'loaded':
            prefix = 'the loaded '
        else:
            prefix = 'the PML pencil ' if self.in_shell is not None else 'the damped pencil '
        if self.on_device:   # cuDSS: memory guard, residual or refinement failure
            return (f'eigenfrequency solve: the cuDSS factorisation of {prefix}K - (2 pi f_target)^2 M failed ({exc}); '
                    "use eigen factorization 'cudss' (double) or 'superlu'")
        if self.kind == 'symmetric':   # SuperLU 'Factor is exactly singular' (eigsh_shifted offsets a nonzero target)
            return (f'eigenfrequency solve: K - (2 pi f_target)^2 M is singular at the target f = {frequency:.6g} Hz '
                    '(at 0 Hz a part that moves without strain, else a resonance on the shift): check the supports or '
                    f'move target_frequency slightly ({exc})')
        return (f'eigenfrequency solve: {prefix}K - (2 pi f_target)^2 M is singular at f = {frequency:.6g} Hz: move '
                f'target_frequency slightly ({exc})')

    def _candidates(self, eigenvalues, vectors, omega_target, warnings, rigid_floor):
        """[_Candidate] of the usable eigenpairs of one shift: rigid-body (|lambda| <= rigid_floor), non-positive /
        infinite and PML modes dropped with warnings; loaded: omega refined by Newton."""
        if self.kind == 'symmetric':
            values = np.asarray(eigenvalues, dtype=np.float64)
            usable = _usable_eigenvalues(values, rigid_floor, warnings)
            return [_Candidate(complex(value), complex(math.sqrt(value)), vectors[:, column],
                               abs(math.sqrt(value) - omega_target), norm=self._norm(vectors[:, column]))
                    for column, value in enumerate(values) if usable[column]]
        candidates, n_pml, n_invalid, n_rigid = [], 0, 0, 0
        for column, value in enumerate(np.asarray(eigenvalues)):
            if np.isfinite(value) and rigid_floor > 0.0 and abs(value) <= rigid_floor:
                n_rigid += 1
                continue
            if not (np.isfinite(value) and value.real > 0.0):
                n_invalid += 1
                continue
            if self.in_shell is not None:
                energy = np.abs(vectors[:, column]) ** 2
                if energy[self.in_shell].sum() > self.energy_share * energy.sum():
                    n_pml += 1
                    continue
            residual = 0.0
            if self.kind == 'loaded':
                omega, residual = _rayleigh_frequency(self.stiffness, self.mass, self.load_positions, self.load_values,
                                                      vectors[:, column], complex(np.sqrt(value)))
            else:
                omega = complex(np.sqrt(complex(value)))
            candidates.append(_Candidate(complex(value), omega, vectors[:, column], abs(omega.real - omega_target),
                                         residual, self._norm(vectors[:, column])))
        if n_rigid:
            warnings.append(_rigid_warning(n_rigid))
        if n_pml:
            warnings.append(f'{n_pml} PML mode(s) (more than {self.energy_share:g} of |u|^2 in the shell) dropped')
        if n_invalid:
            warnings.append(f'{n_invalid} eigenvalue(s) with Re omega^2 <= 0 or infinite dropped')
        return candidates

    def _resonance(self, candidate):
        """MechanicalResonance of a candidate (peak-normalised over every dof; Im f only off the 'symmetric' kind)."""
        omega = candidate.omega
        imaginary = 0.0 if self.kind == 'symmetric' else omega.imag
        return _resonance(omega.real, self.spread(candidate.vector), self.components_per_node, self.displacement_dofs,
                          imaginary)

    def _growing_warning(self, resonances, warnings):
        """Damped kind: warn about kept modes with Im f < 0."""
        if self.kind != 'damped':
            return
        growing = sum(resonance.frequency_imag < 0.0 for resonance in resonances)
        if growing:
            warnings.append(f'{growing} kept mode(s) with Im f < 0 (growing: round-off of a lossless mode, or a too thin '
                            '/ coarse PML)')

    def _inner(self, left, right):
        """x^H M y for the 'symmetric' kind (mass inner product), else x^H y."""
        return np.vdot(left, self.mass @ right) if self.kind == 'symmetric' else np.vdot(left, right)

    def _norm(self, vector) -> float:
        """sqrt(_inner(vector, vector)) (0 for round-off below zero)."""
        return math.sqrt(max(float(np.real(self._inner(vector, vector))), 0.0))

    def _merged(self, candidates):
        """Candidates in Re omega order with each cluster (|d lambda| <= the merge tolerance x |lambda|) reduced to its
        independent vectors, nearest-shift first ('symmetric': M-orthonormalised; a singleton cluster reuses the
        candidate's norm, no product)."""
        tolerance = LOADED_MERGE_TOLERANCE if self.kind == 'loaded' else MERGE_TOLERANCE
        clusters = []
        for candidate in sorted(candidates, key=lambda item: item.eigenvalue.real):
            previous = clusters[-1][-1].eigenvalue if clusters else None
            if previous is not None and (abs(candidate.eigenvalue - previous)
                                         <= tolerance * max(abs(candidate.eigenvalue), _TINY)):
                clusters[-1].append(candidate)
            else:
                clusters.append([candidate])
        kept = []
        for cluster in clusters:
            basis = []
            for candidate in sorted(cluster, key=lambda item: item.distance):
                residual, original = candidate.vector, candidate.norm
                for direction in basis:
                    residual = residual - direction * self._inner(direction, residual)
                size = self._norm(residual) if basis else original
                if not size > DUPLICATE_RESIDUAL * original:
                    continue
                basis.append(residual / size)
                kept.append(replace(candidate, vector=residual / size, norm=1.0) if self.kind == 'symmetric'
                            else candidate)
        kept.sort(key=lambda item: item.omega.real)
        return kept

    def _info(self, started, before, n_requested, target_frequency, eigenvalues_requested, eigenvalues_converged,
              warnings, kept) -> dict:
        """ModalSolve.info of the solves since `before` (_solver_tally): timings from the eigen runtime's counters,
        the last factor's size / kind, the kind's labels."""
        solver, factor = self.eigen_solver, self.factor
        applications = solver.operator_applications - before[2]
        info = {'n_dofs': self.n_dofs, 'n_free': self.n_free, 'n_requested': n_requested,
                'target_frequency': target_frequency,
                'factor_host_bytes': int(getattr(factor, 'host_bytes', 0) or 0),
                'factor': getattr(factor, 'description', 'SuperLU'),
                'factor_device_bytes': int(getattr(factor, 'device_bytes', 0) or 0),
                'factor_seconds': solver.total_factor_seconds - before[0],
                'eigen_seconds': solver.total_eigen_seconds - before[1],
                'reduction_seconds': self.reduction_seconds, 'pencil_seconds': self.pencil_seconds,
                'seconds': time.perf_counter() - started, 'operator_applications': applications,
                'eigenvalues_requested': int(eigenvalues_requested), 'eigenvalues_converged': int(eigenvalues_converged),
                'axial_wavenumber': self.axial_wavenumber, 'bloch': self.bloch_text, 'warnings': warnings,
                **_cudss_timing(factor, applications)}
        if self.kind == 'damped':
            info.update(pml=self.pml_text, loss=self.loss_text, damped=True)
        if self.kind == 'loaded':
            info.update(electrode_loads=[(name, float(resistance)) for name, _, _, resistance in self.loads],
                        load_residual=max((candidate.residual for candidate in kept), default=0.0))
        return info


def _solver_tally(eigen_solver):
    """(factor seconds, eigen seconds, operator applications) so far of an eigen runtime."""
    return (eigen_solver.total_factor_seconds, eigen_solver.total_eigen_seconds, eigen_solver.operator_applications)


def _checked_count(value, name='n_resonances') -> int:
    """value as int >= 1, else ValueError."""
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f'{name} must be an integer >= 1, got {value!r}')
    return int(value)


def _too_few_text(found, wanted, omega_target) -> str:
    return f'{found} of {wanted} vibration modes found near f = {omega_target / (2.0 * math.pi):.6g} Hz'


def solve_modes(stiffness, mass, free, omega_target, n_resonances, components_per_node, eigen_solver=None,
                axial_wavenumber=None, displacement_dofs=None, prolongation=None, bloch_text='', rigid_motions=0):
    """ModalSolve of up to n_resonances modes nearest omega_target [rad/s, >= 0] of host scipy sparse stiffness / mass
    (all dofs, dof = components_per_node node + component) on the dofs where host bool `free` is True, or on x = P x_r
    of a host CSR prolongation (n_dofs, n_r) (Bloch: free unused); eigen_solver = the 'eigen' role runtime (None: a
    default one); axial_wavenumber [rad/m] / bloch_text only label a 2.5D / Bloch solve; displacement_dofs (None = every
    dof) = host int indices, node-major with components_per_node per node, of the dofs the peak normalisation reads
    (piezo: not phi); rigid_motions = free rigid motions of the pencil (extra eigenvalues asked, their modes dropped)."""
    pencil = ModalPencil(stiffness, mass, free, components_per_node, eigen_solver, prolongation=prolongation,
                         displacement_dofs=displacement_dofs, rigid_motions=rigid_motions,
                         axial_wavenumber=axial_wavenumber, bloch_text=bloch_text)
    try:
        return pencil.solve_at(omega_target, n_resonances)
    finally:
        pencil.release()


def solve_modes_at_shifts(pencil, omega_targets, n_per_shift):
    """ModalSolve of the union of the n_per_shift modes nearest each target [rad/s] (each as pencil.solve_at) on one
    kept factor (plan reused across shifts), merged (duplicates of two shifts dropped, degenerate pairs kept) in
    frequency order; info 'targets' [Hz], 'shifts', target_frequency NaN. Releases the pencil's factor."""
    started = time.perf_counter()
    try:
        targets = sorted({pencil._checked_target(target) for target in omega_targets})
        if not targets:
            raise ValueError('solve_modes_at_shifts needs at least one target angular frequency')
        n_per_shift = _checked_count(n_per_shift)
        before = _solver_tally(pencil.eigen_solver)
        warnings = list(pencil.size_warnings)
        n_wanted = pencil._n_wanted(n_per_shift)
        candidates, n_converged = [], 0
        for omega_target in targets:
            eigenvalues, vectors, _ = pencil._shift_solve(omega_target, n_wanted, warnings)
            n_converged += int(np.size(eigenvalues))
            found = sorted(pencil._candidates(eigenvalues, vectors, omega_target, warnings,
                                              RIGID_EIGENVALUE_FRACTION * omega_target ** 2),
                           key=lambda candidate: candidate.distance)[:n_per_shift]
            if len(found) < n_per_shift:
                warnings.append(_too_few_text(len(found), n_per_shift, omega_target))
            candidates.extend(found)
        kept = pencil._merged(candidates)
        resonances = [pencil._resonance(candidate) for candidate in kept]
        pencil._growing_warning(resonances, warnings)
        info = pencil._info(started, before, n_per_shift * len(targets), math.nan, n_wanted * len(targets),
                            n_converged, warnings, kept)
        info.update(targets=[target / (2.0 * math.pi) for target in targets], shifts=len(targets))
        return ModalSolve(resonances=resonances, info=info)
    finally:
        pencil.release()


def solve_band_modes(pencil, f_lo, f_hi, n_per_slice, max_modes):
    """ModalSolve of the modes with f_lo <= Re f < f_hi [Hz] (at most max_modes, the lowest; frequency order) by shifts
    of n_per_slice modes marching upward (module header); a 'symmetric' pencil verifies the count per interval by
    inertia (info 'expected_count'), others overlap 50 % uncounted (warning). info 'band' (f_lo, f_hi), 'shifts',
    'shift_frequencies' [Hz], target_frequency NaN. Releases the pencil's factor."""
    started = time.perf_counter()
    try:
        f_lo, f_hi = float(f_lo), float(f_hi)
        if not (math.isfinite(f_lo) and math.isfinite(f_hi) and 0.0 <= f_lo < f_hi):
            raise ValueError(f'solve_band_modes needs finite 0 <= f_lo < f_hi [Hz], got {f_lo!r}, {f_hi!r}')
        n_per_slice = _checked_count(n_per_slice, 'n_per_slice')
        max_modes = _checked_count(max_modes, 'max_modes')
        search = _BandSearch(pencil, f_lo, f_hi, n_per_slice, max_modes)
        before = _solver_tally(pencil.eigen_solver)
        search.run()
        kept = search.kept
        resonances = [pencil._resonance(candidate) for candidate in kept]
        pencil._growing_warning(resonances, search.warnings)
        info = pencil._info(started, before, max_modes, math.nan, search.n_requested, search.n_converged,
                            search.warnings, kept)
        info.update(band=(f_lo, f_hi), expected_count=search.expected, shifts=len(search.shift_frequencies),
                    shift_frequencies=list(search.shift_frequencies))
        return ModalSolve(resonances=resonances, info=info)
    finally:
        pencil.release()


class _BandSearch:
    """State of one solve_band_modes run: shifts, candidates, inertia points, gaps and warnings."""

    def __init__(self, pencil, f_lo, f_hi, n_per_slice, max_modes):
        self.pencil = pencil
        self.max_modes = max_modes
        self.n_per_slice = n_per_slice
        self.omega_lo, self.omega_hi = 2.0 * math.pi * f_lo, 2.0 * math.pi * f_hi
        self.lambda_hi = self.omega_hi ** 2
        self.rigid_floor = RIGID_EIGENVALUE_FRACTION * self.lambda_hi if pencil.rigid_motions else 0.0
        self.lambda_start = max(self.omega_lo ** 2, self.rigid_floor)
        self.n_wanted = pencil._n_wanted(n_per_slice)
        self.n_limit = pencil._n_wanted(pencil.n_free)
        self.counted = pencil.kind == 'symmetric'
        self.warnings = list(pencil.size_warnings)
        self.candidates, self.inertias, self.shift_frequencies = [], [], []
        self.gaps = []   # (lambda_low, lambda_high) not covered by any trusted window yet
        self.n_requested = self.n_converged = 0
        self.expected = None
        self.kept = []

    def run(self):
        """Counts at the edges, the upward march, gap shifts, count repairs; leaves self.kept."""
        pencil = self.pencil
        lower = upper = None
        if self.counted:
            lower = pencil.count_below(self.lambda_start)
            upper = pencil.count_below(self.lambda_hi)
            self.counted = lower is not None and upper is not None
        if self.counted:
            self.expected = upper.n_below - lower.n_below
            if self.expected > self.max_modes:
                self.warnings.append(f'the band holds {self.expected} modes by the inertia count: only the lowest '
                                     f'{self.max_modes} (max_modes) are solved')
        else:
            self.warnings.append(_UNCOUNTED_BAND)
        covered, capped = self._march()
        self._fill_gaps()
        if self.counted:   # verified up to the upper edge, or up to the last trusted window when stopped early
            limit = covered if capped else self.lambda_hi
            points = [lower] + [point for point in self.inertias if lower.sigma < point.sigma < limit]
            self._repair_counts(points if capped else points + [upper])
        kept = self._in_band(pencil._merged(self.candidates))
        if len(kept) > self.max_modes:
            self.warnings.append(f'{len(kept)} modes found in the band: the lowest {self.max_modes} kept (max_modes)')
            kept = kept[:self.max_modes]
        self.kept = kept

    def _in_band(self, candidates):
        return [candidate for candidate in candidates if self.omega_lo <= candidate.omega.real < self.omega_hi]

    def _shift(self, omega_shift, n_wanted, seed=_START_VECTOR_SEED) -> float:
        """One shift solve (candidates, inertia point); returns its trusted radius max |lambda - sigma| (0: none)."""
        eigenvalues, vectors, shift_inertia = self.pencil._shift_solve(omega_shift, n_wanted, self.warnings,
                                                                       inertia=self.counted, seed=seed)
        self.shift_frequencies.append(omega_shift / (2.0 * math.pi))
        self.n_requested += int(n_wanted)
        self.n_converged += int(np.size(eigenvalues))
        self.candidates.extend(self.pencil._candidates(eigenvalues, vectors, omega_shift, self.warnings,
                                                       self.rigid_floor))
        if shift_inertia is not None:
            self.inertias.append(shift_inertia)
        values = np.asarray(eigenvalues)
        values = values[np.isfinite(values)]
        return float(np.max(np.abs(values - omega_shift ** 2))) if values.size else 0.0

    def _march(self):
        """Shifts upward from the lower edge until the trusted windows pass the upper edge (or the budget / max_modes);
        returns (covered lambda, capped)."""
        omega_shift = math.sqrt(self.lambda_start)
        if self.pencil.needs_positive_target:
            omega_shift = max(omega_shift, BAND_FIRST_SHIFT_FRACTION * self.omega_hi)
        covered = self.lambda_start
        budget = BAND_SHIFTS_PER_SLICE * math.ceil(self.max_modes / self.n_per_slice) + MAX_GAP_SHIFTS
        while covered < self.lambda_hi:
            if len(self.shift_frequencies) >= budget:
                self.warnings.append(f'band search stopped after {budget} shifts at f = {_hertz(covered):.6g} Hz: '
                                     'raise the modes per shift')
                return covered, True
            if (len(self.candidates) >= self.max_modes
                    and len(self._in_band(self.pencil._merged(self.candidates))) >= self.max_modes):
                return covered, True
            square = omega_shift ** 2
            radius = self._shift(omega_shift, self.n_wanted)
            if not radius > 0.0:
                self.warnings.append(f'band search: the shift at f = {omega_shift / (2.0 * math.pi):.6g} Hz returned no '
                                     'eigenvalue: stopped there')
                return covered, True
            if square - radius > covered:
                self.gaps.append((covered, square - radius))
            covered = max(covered, square + radius)
            omega_shift = math.sqrt(covered + (BAND_STEP_FRACTION * radius if self.counted else 0.0))
        return covered, False

    def _fill_gaps(self):
        """A shift at the middle of every uncovered gap (sub-gaps queued) within MAX_GAP_SHIFTS; the rest warned."""
        fills = 0
        while self.gaps and fills < MAX_GAP_SHIFTS:
            low, high = self.gaps.pop(0)
            middle = 0.5 * (low + high)
            radius = self._shift(math.sqrt(middle), self.n_wanted)
            fills += 1
            if middle - radius > low:
                self.gaps.append((low, middle - radius))
            if middle + radius < high:
                self.gaps.append((middle + radius, high))
        for low, high in self.gaps:
            self.warnings.append(f'band search: f = {_hertz(low):.6g} - {_hertz(high):.6g} Hz not reached by any shift '
                                 '(gap budget spent): modes there may be missing')

    def _repair_counts(self, points):
        """Compare found modes with the inertia count per interval of sorted points; short intervals get a repair solve
        at their middle (doubled basis, new start vector) within MAX_COUNT_REPAIRS; remaining mismatches warned."""
        repairs = 0
        while True:
            points = sorted(points, key=lambda point: point.sigma)
            merged = self._in_band(self.pencil._merged(self.candidates))
            mismatches = []
            for low, high in zip(points, points[1:]):
                if not high.sigma > low.sigma:
                    continue
                expected = high.n_below - low.n_below
                found = sum(low.sigma <= candidate.eigenvalue.real < high.sigma for candidate in merged)
                if found != expected:
                    mismatches.append((low, high, expected, found))
            missing = [item for item in mismatches if item[3] < item[2]]
            if not missing or repairs >= MAX_COUNT_REPAIRS:
                break
            for low, high, expected, _found in missing[:MAX_COUNT_REPAIRS - repairs]:
                repairs += 1
                n_inertias = len(self.inertias)
                wanted = min(max(2 * self.n_wanted, expected + EXTRA_EIGENVALUES), self.n_limit)
                self._shift(math.sqrt(0.5 * (low.sigma + high.sigma)), wanted, seed=_START_VECTOR_SEED + repairs)
                points = points + self.inertias[n_inertias:]
        for low, high, expected, found in mismatches:
            what = 'modes may be missing' if found < expected else 'a duplicate kept or a mode on an interval edge'
            self.warnings.append(f'between f = {_hertz(low.sigma):.6g} and {_hertz(high.sigma):.6g} Hz the inertia '
                                 f'count gives {expected} mode(s), the shifts found {found}: {what}')
        if not all(point.exact for point in points):
            self.warnings.append('inertia counts from a float32 factor (cudss_mixed): a pivot near a shift may be '
                                 'miscounted')


def _hertz(eigenvalue) -> float:
    """f [Hz] of lambda = omega^2 (0 below 0)."""
    return math.sqrt(max(float(eigenvalue), 0.0)) / (2.0 * math.pi)


def _cudss_timing(factorization, applications) -> dict:
    """Log items of a cuDSS factor: plan (upload, host reordering, symbolic) and numeric factorisation seconds, mixed:
    float64 refinement steps per eigen solve (the probe's excluded); {} for SuperLU."""
    if not hasattr(factorization, 'plan_seconds'):
        return {}
    timing = {'factor_plan_seconds': float(factorization.plan_seconds),
              'factor_numeric_seconds': float(factorization.factorize_seconds)}
    if factorization.factor_precision == 'mixed' and applications:
        timing['refinements_per_solve'] = (factorization.n_refinements - factorization.refinement_steps) / applications
    return timing


def _check_size(n_free, eigen_solver, warnings):
    """True for a cuDSS eigen factor (it guards its own memory); ValueError above SPLU_LIMIT_DOFS on the host SuperLU or
    below 3 free dofs, a warning above SPLU_WARN_DOFS."""
    on_device = getattr(eigen_solver, 'factorization', 'superlu') in CUDSS_FACTORIZATIONS
    if n_free > SPLU_LIMIT_DOFS and not on_device:
        raise ValueError(f'eigenfrequency solve: {n_free:,} free dofs exceed the {SPLU_LIMIT_DOFS:,} limit of the host '
                         'SuperLU factor (its L + U would fill the host memory): use a coarser mesh')
    if n_free > SPLU_WARN_DOFS and not on_device:
        warnings.append(f'{n_free:,} free dofs exceed ~{SPLU_WARN_DOFS:,}, the practical size of the host SuperLU '
                        'factor: expect a long solve and much host memory')
    if n_free < 3:
        raise ValueError(f'solve_modes: only {n_free} free dofs remain outside the supports')
    return on_device


def solve_loaded_modes(stiffness, mass, free, omega_target, n_resonances, components_per_node, eigen_solver, loads,
                       displacement_dofs=None, rigid_motions=0):
    """ModalSolve of the damped modes T(omega) x = (K + j B / omega - omega^2 M) x = 0 nearest omega_target [rad/s, >
    0]: B = diagonal loads [(name, dof, b, resistance [Ohm])] (a resistive electrode load on its master potential, b =
    s^2 / R); T frozen at omega_target (eigs_shifted), then per mode Newton on x^T T(omega) x = 0 with x fixed (exact for
    a mode on the target, second order in the frozen-pencil error otherwise); complex modes, frequency_imag > 0 decays.
    Arguments otherwise as solve_modes (no prolongation; rigid-body eigenvalues dropped as there)."""
    pencil = ModalPencil(stiffness, mass, free, components_per_node, eigen_solver, displacement_dofs=displacement_dofs,
                         rigid_motions=rigid_motions, loads=loads)
    try:
        return pencil.solve_at(omega_target, n_resonances)
    finally:
        pencil.release()


def solve_pml_modes(stiffness, mass, free, omega_target, n_resonances, components_per_node, eigen_solver, shell_dofs,
                    pml_text='', energy_share=0.5):
    """solve_damped_modes of an elastic PML shell (shell_dofs: the PML modes' filter; omega_target > 0)."""
    return solve_damped_modes(stiffness, mass, free, omega_target, n_resonances, components_per_node, eigen_solver,
                              shell_dofs=shell_dofs, pml_text=pml_text or 'on', energy_share=energy_share)


def solve_damped_modes(stiffness, mass, free, omega_target, n_resonances, components_per_node, eigen_solver,
                       shell_dofs=None, pml_text='', energy_share=0.5, prolongation=None, bloch_text='',
                       rigid_motions=0, loss_text=''):
    """ModalSolve of the damped modes K x = omega^2 M x of a complex pencil linear in omega^2 (no frozen term): an
    elastic PML shell (complex symmetric K(s), M(s)) and / or a loss K_el (1 + j eta) (K complex), nearest omega_target
    [rad/s; > 0 with a PML] by eigs_shifted on the free dofs or on x = P x_r of a host CSR prolongation (Bloch: P^H K P,
    not Hermitian with a loss); omega = sqrt(lambda) with Re > 0, frequency_imag > 0 decays (Q); with shell_dofs (host
    bool over every dof) modes holding more than energy_share of their |u|^2 there are PML modes, dropped; rigid_motions
    asked on top, |lambda| <= RIGID_EIGENVALUE_FRACTION omega_t^2 dropped. Arguments otherwise as solve_modes."""
    pencil = ModalPencil(stiffness, mass, free, components_per_node, eigen_solver, prolongation=prolongation,
                         rigid_motions=rigid_motions, damped=True, shell_dofs=shell_dofs, energy_share=energy_share,
                         bloch_text=bloch_text, pml_text=pml_text, loss_text=loss_text)
    try:
        return pencil.solve_at(omega_target, n_resonances)
    finally:
        pencil.release()


def _rayleigh_frequency(stiffness, mass, load_positions, load_values, vector, omega):
    """(omega, relative residual |T(omega) x| / |K x|): the root near the start omega of the Rayleigh functional x^T (K +
    j B / omega - omega^2 M) x = 0 (transpose, not conjugate: T is complex symmetric) for the fixed vector x, by Newton."""
    stiffness_vector, mass_vector = stiffness @ vector, mass @ vector
    modal_stiffness, modal_mass = vector @ stiffness_vector, vector @ mass_vector
    modal_load = np.sum(load_values * vector[load_positions] ** 2)
    for _ in range(NEWTON_STEPS):
        value = modal_stiffness + 1j * modal_load / omega - omega * omega * modal_mass
        slope = -1j * modal_load / (omega * omega) - 2.0 * omega * modal_mass
        step = value / slope
        omega -= step
        if abs(step) <= NEWTON_TOLERANCE * abs(omega):
            break
    residual = stiffness_vector - omega * omega * mass_vector
    residual[load_positions] += 1j * load_values * vector[load_positions] / omega
    return complex(omega), float(np.linalg.norm(residual) / max(np.linalg.norm(stiffness_vector), 1e-300))


def solve_lowest_modes_matrix_free(inverse_stiffness, mass, free_dofs, n_dofs, n_resonances, components_per_node,
                                   eigen_solver, tolerance_floor):
    """ModalSolve of the n_resonances lowest modes from host callables on the free dofs (host int indices free_dofs):
    inverse_stiffness(b) ~ K_ff^-1 b (one inner PCG solve), mass(v) = M_ff v; eigen_solver.eigsh_inverse at sigma = 0."""
    started = time.perf_counter()
    if isinstance(n_resonances, bool) or int(n_resonances) != n_resonances or n_resonances < 1:
        raise ValueError(f'n_resonances must be an integer >= 1, got {n_resonances!r}')
    n_resonances = int(n_resonances)
    free_dofs = np.asarray(free_dofs, dtype=np.int64)
    n_free = int(free_dofs.size)
    if n_free < 3:
        raise ValueError(f'solve_lowest_modes_matrix_free: only {n_free} free dofs remain outside the supports')
    n_wanted = min(n_resonances + EXTRA_EIGENVALUES, n_free - 1)
    start_vector = np.random.default_rng(_START_VECTOR_SEED).standard_normal(n_free)
    eigenvalues, vectors, applications, converged, tolerance = eigen_solver.eigsh_inverse(
        inverse_stiffness, mass, n_free, n_wanted, start_vector=start_vector, tolerance_floor=tolerance_floor,
        max_restarts=MAX_LANCZOS_RESTARTS)
    warnings = []
    if not converged:
        warnings.append(f'Lanczos converged {np.size(eigenvalues)} of {n_wanted} of the lowest eigenvalues within '
                        f'{MAX_LANCZOS_RESTARTS} restarts (tolerance {tolerance:.1e}): modes may be missing; loosen the '
                        "eigen solver tolerance or tighten the linear solver's")
    eigen_done = time.perf_counter()

    def spread(vector):
        full = np.zeros(int(n_dofs), dtype=vector.dtype)
        full[free_dofs] = vector
        return full

    resonances = _ranked_resonances(eigenvalues, vectors, 0.0, n_resonances, spread, components_per_node, warnings)
    info = {'method': 'matrix_free', 'n_dofs': int(n_dofs), 'n_free': n_free, 'n_requested': n_resonances,
            'target_frequency': 0.0, 'factor_host_bytes': 0, 'factor_seconds': 0.0,
            'eigen_seconds': eigen_done - started, 'seconds': time.perf_counter() - started,
            'operator_applications': applications, 'lanczos_tolerance': tolerance,
            'eigenvalues_requested': n_wanted, 'eigenvalues_converged': int(np.size(eigenvalues)),
            'warnings': warnings}
    return ModalSolve(resonances=resonances, info=info)


def _ranked_resonances(eigenvalues, vectors, omega_target, n_resonances, spread, components_per_node, warnings,
                       displacement_dofs=None):
    """Up to n_resonances MechanicalResonance of the eigenpairs (omega^2, reduced columns; spread(column) -> every dof)
    nearest omega_target; appends to warnings for dropped rigid-body modes (|omega^2| <= RIGID_EIGENVALUE_FRACTION x
    omega_target^2), dropped omega^2 <= 0 and too few modes."""
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    vectors = np.asarray(vectors)
    usable = _usable_eigenvalues(eigenvalues, RIGID_EIGENVALUE_FRACTION * omega_target ** 2, warnings)
    candidates = sorted(((abs(math.sqrt(value) - omega_target), math.sqrt(value), column)
                         for column, value in enumerate(eigenvalues) if usable[column]), key=lambda item: item[0])
    resonances = [_resonance(omega, spread(vectors[:, column]), components_per_node, displacement_dofs)
                  for _, omega, column in candidates[:n_resonances]]
    if len(resonances) < n_resonances:
        warnings.append(_too_few_text(len(resonances), n_resonances, omega_target))
    return resonances


def _usable_eigenvalues(values, floor, warnings) -> np.ndarray:
    """Bool mask of the real eigenvalues omega^2 > floor (finite); warnings for the dropped rigid-body (|omega^2| <=
    floor, when floor > 0) and the non-positive ones (symmetric pencils, ModalPencil and the matrix-free ranking)."""
    finite = np.isfinite(values)
    rigid = finite & (np.abs(values) <= floor) if floor > 0.0 else np.zeros(values.shape, dtype=bool)
    usable = finite & (values > floor)
    if rigid.any():
        warnings.append(_rigid_warning(int(np.count_nonzero(rigid))))
    if not (usable | rigid).all():
        warnings.append(f'{int(np.count_nonzero(~(usable | rigid)))} eigenvalue(s) omega^2 <= 0 dropped (round-off of '
                        'a nearly free solid: check the supports)')
    return usable


def _rigid_warning(count) -> str:
    """Log line for `count` dropped rigid-body modes ('<count> ' + RIGID_TEXT_HEAD + ...: self-tests read the count)."""
    return (f'{count} {RIGID_TEXT_HEAD} (f below {math.sqrt(RIGID_EIGENVALUE_FRACTION):g} x the target) dropped: the '
            'solid can move freely (a free body without supports, or a Bloch cell with every periodic phase 1)')


def _resonance(omega, displacement, components_per_node, displacement_dofs=None, omega_imag=0.0):
    """MechanicalResonance of omega (+ j omega_imag) with the eigenvector over every dof (real or complex) scaled to a
    peak node |u| = 1 m, its largest component there real positive (u = the displacement_dofs, None = every dof; the
    rest, e.g. a piezo phi, is scaled alike)."""
    nodal = displacement if displacement_dofs is None else displacement[displacement_dofs]
    nodes = nodal.reshape(-1, components_per_node)
    magnitude = np.sum(np.abs(nodes) ** 2, axis=1)
    peak_node = int(np.argmax(magnitude))
    peak_value = nodes[peak_node, int(np.argmax(np.abs(nodes[peak_node])))]
    scale = math.sqrt(float(magnitude[peak_node])) * (peak_value / abs(peak_value))   # real: the sign
    return MechanicalResonance(omega=omega, frequency=omega / (2.0 * math.pi), displacement=displacement / scale,
                               frequency_imag=float(omega_imag) / (2.0 * math.pi))


# ---- eigenfrequency search 'targets' / 'band' (model.solver_hooks.MultiShiftEigenSolver) ----
class MultiShiftModes:
    """Mixin of the solid solvers: solve_eigenfrequencies_targets / _band on the pencil of the solver's
    _modal_pencil(lowest_target) -> (ModalPencil, finish(ModalSolve) -> ModalSolve), kept as eigen_solve."""

    def solve_eigenfrequencies_targets(self, n_resonances, target_frequencies):
        """Union of the n_resonances modes nearest each target [Hz] (one kept factor across targets), frequency
        order -> ModalSolve, kept as eigen_solve."""
        self.eigen_solve = None
        targets = [float(target) for target in target_frequencies]
        if not targets:
            raise ValueError(f'{self._search_label()}: the eigenfrequency search needs at least one target frequency')
        pencil, finish = self._modal_pencil(min(targets))
        self.eigen_solve = finish(solve_modes_at_shifts(pencil, [2.0 * math.pi * target for target in targets],
                                                        n_resonances))
        return self.eigen_solve

    def solve_eigenfrequencies_band(self, f_lo, f_hi, n_per_slice, max_resonances):
        """Every mode with f_lo <= Re f < f_hi [Hz] (at most max_resonances, the lowest; inertia-counted for a real
        symmetric pencil), frequency order -> ModalSolve, kept as eigen_solve; a piezoelectric solid refuses it."""
        self.eigen_solve = None
        if getattr(self, 'piezoelectric', False):
            raise ValueError(f"{self._search_label()}: the eigenfrequency search 'band' is not available for a "
                             "piezoelectric solid (its coupled pencil is indefinite: no inertia count); use search "
                             "'targets'")
        pencil, finish = self._modal_pencil(None)
        self.eigen_solve = finish(solve_band_modes(pencil, f_lo, f_hi, n_per_slice, max_resonances))
        return self.eigen_solve

    def _search_label(self) -> str:
        """The solver's error label (_label() of the FEM solvers, system.label() of the tensor ones)."""
        label = getattr(self, '_label', None)
        return label() if label is not None else self.system.label()


# ---- resonance_table feature (reads the solver's last ModalSolve) ----
class ModalTable:
    """Marker object of a solid resonance_table feature: its result reads SolidMechanicsPhysics.eigen_solve."""


def modal_table_result(table, physics) -> dict:
    """'table' of the last eigenfrequency solve of `physics` (MODAL_TABLE_COLUMNS, + DAMPED_TABLE_COLUMNS with a
    resistive electrode load, an elastic PML or a loss factor; one target: nearest first, several / a band: frequency
    order); extras target_frequency [Hz] (NaN for several targets or a band), targets [Hz] / band (f_lo, f_hi) /
    expected_count (None for one target), selected_resonance, axial_wavenumber [rad/m] (None = plane strain); ValueError
    before such a solve."""
    solve = physics.eigen_solve
    if solve is None:
        raise ValueError('not solved yet: the resonance table needs an eigenfrequency study')
    damped = bool(solve.info.get('electrode_loads') or solve.info.get('damped'))
    rows = []
    for number, resonance in enumerate(solve.resonances, start=1):
        row = [number, resonance.frequency, resonance.omega]
        if damped:
            row += [resonance.frequency_imag, resonance.q_factor, 4.0 * math.pi * resonance.frequency_imag]
        rows.append(row)
    columns = MODAL_TABLE_COLUMNS + (DAMPED_TABLE_COLUMNS if damped else ())
    units = MODAL_TABLE_UNITS + (DAMPED_TABLE_UNITS if damped else ())
    band = solve.info.get('band')
    return {'kind': 'table', 'columns': list(columns), 'units': list(units), 'rows': rows,
            'target_frequency': float(solve.info['target_frequency']), 'selected_resonance': physics.selected_resonance,
            'axial_wavenumber': solve.info.get('axial_wavenumber'), 'targets': solve.info.get('targets'),
            'band': None if band is None else list(band), 'expected_count': solve.info.get('expected_count')}
