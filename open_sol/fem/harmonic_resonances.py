"""Physics-neutral complex resonances of a harmonic FEM pencil K u = omega^2 M u (new_modules_plan S8): shift-invert
ARPACK about a target frequency [Hz] (solvers.arpack_shift_invert), PML modes dropped by their energy share outside
the window, an optional region energy fraction; HarmonicResonance / HarmonicResonanceSolve with the duck-typed text,
summary and info['warnings'] of the eigenfrequency study, and the resonance table rows. Host numpy, scipy inside
functions."""
# exp(+j omega t): lambda = omega^2, omega = sqrt(lambda) on the principal branch (Re omega >= 0); a decaying resonance
# (PML radiation, a loss factor a (1 + j eta)) has Im omega > 0 and Q = Re omega / (2 Im omega) (inf when |Im| is
# round-off). Pencils: 'real_symmetric' (K, M real, M positive definite: eigsh_shifted, the solver's own shift offset),
# 'complex_symmetric' (a PML or complex coefficients: an own LDL^T-able factor of K - sigma M, sigma = omega_t^2 (1 +
# SHIFT_RELATIVE_OFFSET) so a target on a lossless resonance is not singular, and eigs of (K - sigma M)^-1 M),
# 'general' (likewise without the symmetric factor request); 'auto' = real_symmetric when both are real, else
# complex_symmetric. Resonances are ranked by |omega - omega_t|. Energy share per row = |u|^2 x energy_weights (default
# 1; HarmonicEigenPencil.energy_weights: lumped |M|) or Re(conj(u) W u) of a sparse W; pml_fraction = 1 - the in_window
# share (> pml_fraction_limit: a PML mode, dropped; the complex paths then widen ARPACK on the same factor, at most
# MAX_WIDENINGS doublings); region_fraction = the in_region share (NaN without a region; below min_region_fraction:
# dropped). |lambda| <= STATIC_RELATIVE_EIGENVALUE x sigma (a constant mode of a free edge / Neumann region) and Re omega
# <= 0 are no resonances.
import math
import time
from dataclasses import dataclass

import numpy as np

PENCIL_KINDS = ('auto', 'real_symmetric', 'complex_symmetric', 'general')
PML_RESONANCE_FRACTION = 0.5          # resonances with a larger energy share outside the window are PML modes
SHIFT_RELATIVE_OFFSET = 1e-4          # sigma = omega_t^2 (1 + this) of the own factor (complex paths; as eigsh_shifted)
STATIC_RELATIVE_EIGENVALUE = 1e-8     # |lambda| below this x sigma: a static (constant) mode, dropped
EXTRA_EIGENVALUES = 4                 # eigenvalues asked beyond the wanted resonances (ranking margin)
MAX_WIDENINGS = 2                     # doublings of the eigenpairs while the filters leave too few resonances
_THETA_FLOOR = 1e-12                  # |theta| below this x the largest: massless rows (lambda infinite)
_LOSSLESS_RELATIVE_IMAG = 1e-12       # |Im omega| below this x Re omega is round-off: Q = inf
_START_VECTOR_SEED = 6                # deterministic ARPACK start vector
RESONANCE_COLUMNS = ('resonance_number', 'frequency', 'frequency_imag', 'q_factor', 'pml_fraction')
RESONANCE_UNITS = ('1', 'Hz', 'Hz', '1', '1')
REGION_COLUMN = ('region_energy_fraction', '1')   # appended when the solve had a region


@dataclass(frozen=True, eq=False)
class HarmonicResonance:
    """One resonance: complex omega [rad/s] (Im > 0 decays), the energy share outside the window (pml_fraction) and in
    the region (NaN without one), the field (field_of of the eigenvector; peak |u| 1, real positive there)."""
    omega: complex
    pml_fraction: float
    region_fraction: float
    field: np.ndarray

    @property
    def frequency(self) -> float:
        """Re f [Hz]."""
        return self.omega.real / (2.0 * math.pi)

    @property
    def frequency_imag(self) -> float:
        """Im f [Hz] (> 0 decays)."""
        return self.omega.imag / (2.0 * math.pi)

    @property
    def q_factor(self) -> float:
        return q_factor_of(self.omega)

    def text(self) -> str:
        """'f = 312456 Hz' or 'f = 312456 Hz, Q = 1023' (the snapshot label after 'resonance k: ')."""
        q_factor = self.q_factor
        return f'f = {self.frequency:.6g} Hz' + ('' if math.isinf(q_factor) else f', Q = {q_factor:.4g}')


@dataclass(frozen=True, eq=False)
class HarmonicResonanceSolve:
    """Resonances nearest the target first and the solve info (counts, timings, warnings)."""
    resonances: list
    info: dict

    def summary(self) -> str:
        """One line for the study log."""
        info = self.info
        widened = (f" in {info['widenings'] + 1} ARPACK passes (too few resonances: eigenpairs doubled)"
                   if info.get('widenings') else '')
        label = f"{info['label']} " if info.get('label') else ''
        region = (f", {info['region_resonances_dropped']} below the region energy fraction "
                  f"{info.get('min_region_fraction', 0.0):g}" if info.get('region') else '')
        if info.get('band') is not None:   # fem.resonance_search: several shifts
            low, high = info['band']
            expected = info.get('expected_count')
            count = f'{expected} by inertia count' if expected is not None else 'count not verified'
            head = (f"{len(self.resonances)} {label}resonances in f = {low:.6g} - {high:.6g} Hz ({count}, "
                    f"{info['shifts']} shifts)")
        elif info.get('targets'):
            head = (f"{len(self.resonances)} {label}resonances near f = "
                    f"{', '.join(f'{target:.6g}' for target in info['targets'])} Hz ({info['shifts']} shifts)")
        else:
            head = (f"{len(self.resonances)} of {info['n_requested']} {label}resonances near f = "
                    f"{info['target_frequency']:.6g} Hz")
        return (f"{head}: {info['n_free']:,} free dofs ({info['pencil']} pencil, "
                f"{info['factor']} factorisation {info['factor_seconds']:.3g} s, "
                f"{info['factor_host_bytes'] / 2 ** 20:.4g} MiB; eigs {info['eigen_seconds']:.3g} s{widened}, "
                f"{info['operator_applications']} operator applications, {info['eigenvalues_converged']} of "
                f"{info['eigenvalues_requested']} eigenvalues; dropped: {info['pml_resonances_dropped']} PML"
                f"{region})")


def q_factor_of(omega) -> float:
    """Q = Re omega / (2 Im omega) of a complex resonance omega [rad/s]; inf when |Im| is round-off (lossless)."""
    omega = complex(omega)
    if abs(omega.imag) <= _LOSSLESS_RELATIVE_IMAG * abs(omega.real):
        return math.inf
    return omega.real / (2.0 * omega.imag)


def _energy_weights(energy_weights, n_unknowns):
    """(n,) float64 weights (None = 1) or a scipy CSR energy matrix of n rows."""
    import scipy.sparse
    if energy_weights is None:
        return np.ones(n_unknowns)
    if scipy.sparse.issparse(energy_weights):
        if energy_weights.shape != (n_unknowns, n_unknowns):
            raise ValueError(f'solve_harmonic_resonances needs an ({n_unknowns}, {n_unknowns}) energy matrix, got '
                             f'{energy_weights.shape}')
        return energy_weights.tocsr()
    energy_weights = np.asarray(energy_weights, dtype=np.float64).ravel()
    if energy_weights.size != n_unknowns:
        raise ValueError(f'solve_harmonic_resonances needs one energy weight per unknown ({n_unknowns}), got '
                         f'{energy_weights.size}')
    return energy_weights


def _min_region_fraction(min_region_fraction, in_region) -> float:
    """min_region_fraction as float; ValueError outside [0, 1] or > 0 without a region (as em eigenfrequency)."""
    min_region_fraction = float(min_region_fraction)
    if not 0.0 <= min_region_fraction <= 1.0:
        raise ValueError(f'min_region_fraction must lie in [0, 1], got {min_region_fraction!r}')
    if in_region is None and min_region_fraction > 0.0:
        raise ValueError(f'min_region_fraction {min_region_fraction:g} needs a region (in_region / region_elements)')
    return min_region_fraction


def _row_mask(mask, n_unknowns, what):
    if mask is None:
        return None
    mask = np.asarray(mask, dtype=bool).ravel()
    if mask.size != n_unknowns:
        raise ValueError(f'solve_harmonic_resonances needs one {what} flag per unknown ({n_unknowns}), got {mask.size}')
    return mask


def _row_energies(vector, energy_weights):
    if isinstance(energy_weights, np.ndarray):
        return np.abs(vector) ** 2 * energy_weights
    return np.real(np.conj(vector) * (energy_weights @ vector))


def solve_harmonic_resonances(stiffness, mass, target_frequency, n_resonances, *, eigen_solver=None, pencil='auto',
                              in_window=None, energy_weights=None, in_region=None, min_region_fraction=0.0,
                              field_of=None, pml_fraction_limit=PML_RESONANCE_FRACTION, label=''):
    """HarmonicResonanceSolve of up to n_resonances resonances nearest target_frequency [Hz, >= 0] of host scipy K, M
    (square, the free unknowns only): eigen_solver = the 'eigen' role runtime (ArpackShiftInvertSolver; None = default
    settings), pencil of PENCIL_KINDS, in_window / in_region bool per row (None: no PML filter / no region),
    energy_weights (module header), field_of(vector) -> HarmonicResonance.field (None: the vector), label names the
    resonances in the summary ('membrane')."""
    from ..solvers.arpack_shift_invert import EIGSH_SHIFT_RELATIVE_OFFSET, ArpackShiftInvertSolver, device_factor_matvec
    started = time.perf_counter()
    eigen_solver = ArpackShiftInvertSolver() if eigen_solver is None else eigen_solver
    target_frequency = float(target_frequency)
    if not (math.isfinite(target_frequency) and target_frequency >= 0.0):
        raise ValueError(f'resonance solve needs a finite target frequency >= 0 [Hz], got {target_frequency!r}')
    if isinstance(n_resonances, bool) or int(n_resonances) != n_resonances or n_resonances < 1:
        raise ValueError(f'n_resonances must be an integer >= 1, got {n_resonances!r}')
    if pencil not in PENCIL_KINDS:
        raise ValueError(f"pencil must be one of {', '.join(PENCIL_KINDS)}, got {pencil!r}")
    n_resonances = int(n_resonances)
    stiffness, mass = stiffness.tocsr(), mass.tocsr()
    n_free = int(stiffness.shape[0])
    if stiffness.shape != (n_free, n_free) or mass.shape != stiffness.shape:
        raise ValueError(f'resonance solve needs square K and M of one shape, got {stiffness.shape} and {mass.shape}')
    if n_free < 4:
        raise ValueError(f'resonance solve: only {n_free} free unknowns')
    real = not (np.iscomplexobj(stiffness.data) or np.iscomplexobj(mass.data))
    if pencil == 'auto':
        pencil = 'real_symmetric' if real else 'complex_symmetric'
    elif pencil == 'real_symmetric' and not real:
        raise ValueError("pencil 'real_symmetric' needs real K and M (a PML or complex coefficients: "
                         "'complex_symmetric')")
    weights = _energy_weights(energy_weights, n_free)
    in_window = _row_mask(in_window, n_free, 'window')
    in_region = _row_mask(in_region, n_free, 'region')
    min_region_fraction = _min_region_fraction(min_region_fraction, in_region)
    omega_target = 2.0 * math.pi * target_frequency
    warnings = []
    applications_before = eigen_solver.operator_applications
    eigen_seconds_before = eigen_solver.total_eigen_seconds   # ARPACK alone (the factor precedes it)
    if pencil == 'real_symmetric':
        n_wanted = min(n_resonances + EXTRA_EIGENVALUES, n_free - 1)
        start_vector = np.random.default_rng(_START_VECTOR_SEED).standard_normal(n_free)
        try:
            eigenvalues, vectors, _, converged, factorization = eigen_solver.eigsh_shifted(
                stiffness.astype(np.float64), mass.astype(np.float64), omega_target ** 2, n_wanted,
                start_vector=start_vector)
        except RuntimeError as exc:
            raise ValueError(_singular_text(target_frequency, exc)) from None
        sigma = omega_target ** 2 * (1.0 + EIGSH_SHIFT_RELATIVE_OFFSET)
        candidates, counts = _candidates(np.asarray(eigenvalues, dtype=np.complex128), vectors, sigma, omega_target,
                                         weights, in_window, in_region, min_region_fraction, pml_fraction_limit)
        widenings = 0
    else:
        sigma = omega_target ** 2 * (1.0 + SHIFT_RELATIVE_OFFSET)
        try:
            factorization = eigen_solver.factorize_host((stiffness - sigma * mass).tocsc().astype(np.complex128),
                                                        symmetric=pencil == 'complex_symmetric')
        except RuntimeError as exc:
            raise ValueError(_singular_text(target_frequency, exc)) from None
        mass_complex = mass.astype(np.complex128)
        random = np.random.default_rng(_START_VECTOR_SEED)
        start_vector = random.standard_normal(n_free) + 1j * random.standard_normal(n_free)
        n_wanted = min(2 * n_resonances + EXTRA_EIGENVALUES, n_free - 2)
        widenings, previous_count = 0, -1
        device_apply = device_factor_matvec(factorization, mass_complex)   # cuDSS: the device Krylov-Schur
        while True:
            theta, vectors, _, converged = eigen_solver.eigs(
                lambda vector: factorization.solve_host(mass_complex @ vector), n_free, n_wanted, start_vector,
                device_matvec=device_apply)
            theta = np.asarray(theta, dtype=np.complex128)
            usable = np.isfinite(theta) & (np.abs(theta) > _THETA_FLOOR * float(np.max(np.abs(theta), initial=0.0)))
            eigenvalues = np.full(theta.shape, np.nan, dtype=np.complex128)
            eigenvalues[usable] = sigma + 1.0 / theta[usable]
            candidates, counts = _candidates(eigenvalues, vectors, sigma, omega_target, weights, in_window, in_region,
                                             min_region_fraction, pml_fraction_limit)
            if (len(candidates) >= n_resonances or len(candidates) == previous_count or widenings == MAX_WIDENINGS
                    or n_wanted >= n_free - 2):
                break
            previous_count = len(candidates)
            n_wanted = min(2 * n_wanted, n_free - 2)
            widenings += 1
    eigen_seconds = eigen_solver.total_eigen_seconds - eigen_seconds_before
    eigen_done = time.perf_counter()
    if not converged:
        warnings.append(f'ARPACK converged only part of {n_wanted} eigenvalues near f = {target_frequency:.6g} Hz; '
                        'resonances may be missing')
    for key, text in (('static', 'static (constant) mode(s) dropped'),
                      ('invalid', 'eigenvalue(s) with Re omega <= 0 or infinite dropped'),
                      ('pml', f'PML mode(s) (more than {pml_fraction_limit:g} of the energy outside the window) '
                              'dropped'),
                      ('region', f'resonance(s) below the region energy fraction {min_region_fraction:g} dropped')):
        if counts[key]:
            warnings.append(f'{counts[key]} {text}')
    resonances = [_resonance(omega, pml_fraction, region_fraction, vector if field_of is None else field_of(vector))
                  for _, omega, pml_fraction, region_fraction, vector in candidates[:n_resonances]]
    growing = sum(resonance.omega.imag < -_LOSSLESS_RELATIVE_IMAG * resonance.omega.real for resonance in resonances)
    if growing:
        warnings.append(f'{growing} kept resonance(s) with Im f < 0 (growing: a too thin or coarse PML, or gain)')
    if len(resonances) < n_resonances:
        warnings.append(f'{len(resonances)} of {n_resonances} resonances found near f = {target_frequency:.6g} Hz: '
                        'move the target nearer, raise the number of resonances, or resolve the PML better'
                        + (', or lower min_region_fraction' if counts['region'] else ''))
    info = {'n_free': n_free, 'n_requested': n_resonances, 'target_frequency': target_frequency, 'pencil': pencil,
            'label': label, 'factor': getattr(factorization, 'description', 'SuperLU'),
            'factor_host_bytes': int(getattr(factorization, 'host_bytes', 0) or 0),
            'factor_device_bytes': int(getattr(factorization, 'device_bytes', 0) or 0),
            'factor_seconds': max(eigen_done - started - eigen_seconds, 0.0), 'eigen_seconds': eigen_seconds,
            'seconds': time.perf_counter() - started,
            'operator_applications': eigen_solver.operator_applications - applications_before,
            'eigenvalues_requested': n_wanted, 'eigenvalues_converged': int(np.size(eigenvalues)),
            'widenings': widenings, 'pml_resonances_dropped': counts['pml'],
            'region_resonances_dropped': counts['region'], 'region': in_region is not None,
            'min_region_fraction': min_region_fraction, 'warnings': warnings}
    # the fields are host copies by now: free a cuDSS shift factor (outside the cupy pool) instead of leaving it to gc
    getattr(factorization, 'release', lambda: None)()
    return HarmonicResonanceSolve(resonances=resonances, info=info)


def _singular_text(target_frequency, exc) -> str:
    return (f'resonance solve: K - (2 pi f_target)^2 M is singular at f = {target_frequency:.6g} Hz ({exc}): at 0 Hz a '
            'region that moves without strain (free edges only), else a resonance on the shift: hold a boundary or '
            'move the target frequency slightly')


def _candidates(eigenvalues, vectors, sigma, omega_target, weights, in_window, in_region, min_region_fraction,
                pml_fraction_limit) -> tuple:
    """(candidates (|omega - omega_t|, omega, pml_fraction, region_fraction, vector) nearest first, dropped counts)."""
    vectors = np.asarray(vectors)
    counts = {'static': 0, 'invalid': 0, 'pml': 0, 'region': 0}
    candidates = []
    static_floor = STATIC_RELATIVE_EIGENVALUE * abs(sigma)
    for column, value in enumerate(np.asarray(eigenvalues, dtype=np.complex128)):
        if not np.isfinite(value):
            counts['invalid'] += 1
            continue
        if static_floor > 0.0 and abs(value) <= static_floor:
            counts['static'] += 1
            continue
        omega = complex(np.sqrt(value))
        if not omega.real > 0.0:
            counts['invalid'] += 1
            continue
        vector = vectors[:, column]
        energy = _row_energies(vector, weights)
        total = float(np.sum(energy))
        pml_fraction = 0.0
        if in_window is not None:
            pml_fraction = 1.0 - float(np.sum(energy[in_window])) / total if total > 0.0 else 1.0
        if pml_fraction > pml_fraction_limit:
            counts['pml'] += 1
            continue
        region_fraction = math.nan
        if in_region is not None:
            region_fraction = float(np.sum(energy[in_region])) / total if total > 0.0 else 0.0
            if region_fraction < min_region_fraction:
                counts['region'] += 1
                continue
        candidates.append((abs(omega - omega_target), omega, pml_fraction, region_fraction, vector))
    candidates.sort(key=lambda candidate: candidate[0])
    return candidates, counts


def _resonance(omega, pml_fraction, region_fraction, field) -> HarmonicResonance:
    """HarmonicResonance with its field scaled to 1 (real) at the first entry of largest |u|."""
    field = np.asarray(field)
    peak = int(np.argmax(np.abs(field)))
    if field.flat[peak] != 0:
        field = field / field.flat[peak]
    return HarmonicResonance(omega=complex(omega), pml_fraction=float(pml_fraction),
                             region_fraction=float(region_fraction), field=field)


def solve_system_resonances(system, target_frequency, n_resonances, *, eigen_solver=None, region_elements=None,
                            min_region_fraction=0.0, pencil='auto', label=''):
    """solve_harmonic_resonances of a fem.harmonic / harmonic3d system's eigen pencil (Dirichlet held at 0, its window
    rows and lumped |M| energy weights; region_elements: bool per mesh element, None = no region, column or filter);
    fields in the system's full-mesh numbering (complex128, like system.values)."""
    eigen = system.eigen_pencil(region_elements)
    return solve_harmonic_resonances(eigen.stiffness, eigen.mass, target_frequency, n_resonances,
                                     eigen_solver=eigen_solver, pencil=pencil, in_window=eigen.in_window,
                                     energy_weights=eigen.energy_weights(), in_region=eigen.in_region,
                                     min_region_fraction=min_region_fraction, field_of=eigen.spread, label=label)


def resonance_table(solve, selected_resonance=None) -> dict:
    """'table' result of a HarmonicResonanceSolve (RESONANCE_COLUMNS nearest the target first, several targets / a
    band: frequency order; + REGION_COLUMN when the solve had a region); extras target_frequency [Hz] (NaN for several
    shifts), selected_resonance, targets [Hz] / band (f_lo, f_hi) / expected_count (None for one target)."""
    with_region = bool(solve.info.get('region'))
    rows = []
    for number, resonance in enumerate(solve.resonances, start=1):
        row = [number, resonance.frequency, resonance.frequency_imag, resonance.q_factor, resonance.pml_fraction]
        if with_region:
            row.append(resonance.region_fraction)
        rows.append(row)
    columns = RESONANCE_COLUMNS + ((REGION_COLUMN[0],) if with_region else ())
    units = RESONANCE_UNITS + ((REGION_COLUMN[1],) if with_region else ())
    band = solve.info.get('band')
    return {'kind': 'table', 'columns': list(columns), 'units': list(units), 'rows': rows,
            'target_frequency': float(solve.info['target_frequency']), 'selected_resonance': selected_resonance,
            'targets': solve.info.get('targets'), 'band': None if band is None else list(band),
            'expected_count': solve.info.get('expected_count')}
