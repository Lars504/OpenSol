"""Several shifts of a complex resonance solve (eigenfrequency search 'targets' / 'band'): the union of the resonances
nearest several targets, and every resonance of a frequency band, from a per-shift solve callable; merged, frequency
order. Host numpy."""
# solve_at(target_frequency [Hz], n) -> HarmonicResonanceSolve (resonances nearest the target first, info as
# harmonic_resonances); count_below(lambda) -> eigenvalues below lambda = omega^2 (Sylvester inertia of a real symmetric
# pencil) or None. Merge: by Re omega; a cluster within MERGE_TOLERANCE x |omega^2| keeps only the fields independent of
# those kept (residual above DUPLICATE_RESIDUAL relative, Hermitian inner products: two shifts give the same mode up to
# a phase), nearest-shift first, so degenerate pairs survive and duplicates go. Band (f_lo <= Re f < f_hi, half-open as
# the inertia count): shifts march upward from f_lo; a shift trusts |Re omega - omega_s| out to its farthest kept
# resonance and the next shift sits at the covered edge (windows overlap by about half); a window starting above the
# covered edge leaves a gap, filled by a shift at its middle (at most MAX_GAP_SHIFTS); with count_below the merged
# count is compared with the inertia count of the band (warning on a mismatch, no repair solve), without it a coverage
# warning. No factor is kept across shifts (each solve_at factorises afresh: no cuDSS plan reuse, unlike
# physics/mechanics/modal.ModalPencil).
import math
import time

import numpy as np

from .harmonic_resonances import HarmonicResonanceSolve

MERGE_TOLERANCE = 1e-6         # |omega_i^2 - omega_j^2| <= this x |omega^2|: one cluster of the merge
DUPLICATE_RESIDUAL = 1e-3      # a cluster field whose part outside the kept ones is below this (relative): a duplicate
BAND_SHIFTS_PER_SLICE = 4      # marching budget: this x ceil(max_resonances / n_per_slice) shifts (+ MAX_GAP_SHIFTS)
MAX_GAP_SHIFTS = 8             # shifts filling uncovered gaps, per band solve
BAND_FIRST_SHIFT_FRACTION = 1e-2   # a band from 0 whose shift at 0 is singular (free edges): first shift at this x f_hi
_SUMMED_INFO = ('factor_seconds', 'eigen_seconds', 'operator_applications', 'eigenvalues_requested',
                'eigenvalues_converged', 'pml_resonances_dropped', 'region_resonances_dropped', 'widenings',
                'n_requested')
_UNCOUNTED_BAND = ('band coverage not verified (no inertia count for a complex pencil: loss factor, PML or Bloch '
                   'phase): shifts overlap by about half their trusted radius, resonances may be missing')
_TINY = 1e-300


def solve_resonances_at_targets(solve_at, target_frequencies, n_per_target):
    """HarmonicResonanceSolve of the union of the n_per_target resonances nearest each target [Hz] (one solve_at per
    target), merged, frequency order; info targets [Hz], shifts, target_frequency NaN."""
    started = time.perf_counter()
    targets = sorted({float(target) for target in target_frequencies})
    if not targets:
        raise ValueError('the resonance search needs at least one target frequency')
    solves = [solve_at(target, n_per_target) for target in targets]
    candidates = [(resonance, abs(resonance.omega.real - 2.0 * math.pi * target))
                  for target, solve in zip(targets, solves) for resonance in solve.resonances]
    info = _combined_info(solves, started, [])
    info.update(targets=targets, shifts=len(targets))
    return HarmonicResonanceSolve(resonances=_merged(candidates), info=info)


def solve_resonances_in_band(solve_at, f_lo, f_hi, n_per_slice, max_resonances, count_below=None):
    """HarmonicResonanceSolve of the resonances with f_lo <= Re f < f_hi [Hz] (at most max_resonances, the lowest;
    frequency order) by shifts of n_per_slice resonances marching upward (module header); count_below (real symmetric
    pencils) gives info 'expected_count'. info 'band' (f_lo, f_hi), 'shifts', 'shift_frequencies' [Hz]."""
    started = time.perf_counter()
    f_lo, f_hi = float(f_lo), float(f_hi)
    if not (math.isfinite(f_lo) and math.isfinite(f_hi) and 0.0 <= f_lo < f_hi):
        raise ValueError(f'the band search needs finite 0 <= f_lo < f_hi [Hz], got {f_lo!r}, {f_hi!r}')
    for value, name in ((n_per_slice, 'n_per_slice'), (max_resonances, 'max_resonances')):
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f'{name} must be an integer >= 1, got {value!r}')
    n_per_slice, max_resonances = int(n_per_slice), int(max_resonances)
    omega_lo, omega_hi = 2.0 * math.pi * f_lo, 2.0 * math.pi * f_hi
    solves, candidates, warnings, shift_frequencies = [], [], [], []

    def shift(omega_shift) -> float:
        """One solve at omega_shift; its trusted radius in Re omega (0: no resonance)."""
        solve = solve_at(omega_shift / (2.0 * math.pi), n_per_slice)
        solves.append(solve)
        shift_frequencies.append(omega_shift / (2.0 * math.pi))
        distances = [abs(resonance.omega.real - omega_shift) for resonance in solve.resonances]
        candidates.extend(zip(solve.resonances, distances))
        return max(distances) if distances else 0.0

    def in_band(resonances):
        return [resonance for resonance in resonances if omega_lo <= resonance.omega.real < omega_hi]   # as _band_count

    budget = BAND_SHIFTS_PER_SLICE * math.ceil(max_resonances / n_per_slice) + MAX_GAP_SHIFTS
    covered = omega_shift = omega_lo
    gaps, capped = [], False
    while covered < omega_hi:
        if len(shift_frequencies) >= budget:
            warnings.append(f'band search stopped after {budget} shifts at f = {covered / (2.0 * math.pi):.6g} Hz: '
                            'raise the resonances per shift')
            capped = True
            break
        if len(candidates) >= max_resonances and len(in_band(_merged(candidates))) >= max_resonances:
            capped = True
            break
        try:
            radius = shift(omega_shift)
        except ValueError:
            if shift_frequencies or omega_shift > 0.0:
                raise
            omega_shift = BAND_FIRST_SHIFT_FRACTION * omega_hi   # singular at 0 Hz (free edges): start above it
            continue
        if not radius > 0.0:
            warnings.append(f'band search: the shift at f = {omega_shift / (2.0 * math.pi):.6g} Hz returned no '
                            'resonance: stopped there')
            capped = True
            break
        if omega_shift - radius > covered:
            gaps.append((covered, omega_shift - radius))
        covered = max(covered, omega_shift + radius)
        omega_shift = covered
    fills = 0
    while gaps and fills < MAX_GAP_SHIFTS:
        low, high = gaps.pop(0)
        middle = 0.5 * (low + high)
        radius = shift(middle)
        fills += 1
        if middle - radius > low:
            gaps.append((low, middle - radius))
        if middle + radius < high:
            gaps.append((middle + radius, high))
    for low, high in gaps:
        warnings.append(f'band search: f = {low / (2.0 * math.pi):.6g} - {high / (2.0 * math.pi):.6g} Hz not reached '
                        'by any shift (gap budget spent): resonances there may be missing')
    kept = in_band(_merged(candidates))
    if len(kept) > max_resonances:
        warnings.append(f'{len(kept)} resonances found in the band: the lowest {max_resonances} kept (max_resonances)')
        kept = kept[:max_resonances]
    expected = _band_count(count_below, omega_lo, omega_hi)
    if expected is None:
        warnings.append(_UNCOUNTED_BAND)
    elif expected > max_resonances:
        warnings.append(f'the band holds {expected} resonances by the inertia count: only the lowest {max_resonances} '
                        '(max_resonances) are solved')
    elif len(kept) != expected and not capped:
        what = 'resonances may be missing' if len(kept) < expected else 'a duplicate kept or a resonance on a band edge'
        warnings.append(f'the inertia count gives {expected} resonance(s) in the band, the shifts found {len(kept)}: '
                        f'{what}')
    info = _combined_info(solves, started, warnings)
    info.update(band=(f_lo, f_hi), expected_count=expected, shifts=len(shift_frequencies),
                shift_frequencies=list(shift_frequencies))
    return HarmonicResonanceSolve(resonances=kept, info=info)


def _band_count(count_below, omega_lo, omega_hi):
    """Eigenvalues omega^2 in [omega_lo^2, omega_hi^2) by two inertia counts; None without count_below or a count."""
    if count_below is None:
        return None
    lower, upper = count_below(omega_lo ** 2), count_below(omega_hi ** 2)
    if lower is None or upper is None:
        return None
    return int(upper) - int(lower)


def _merged(candidates):
    """[(HarmonicResonance, distance to its shift)] -> resonances in Re omega order, each cluster reduced to its
    independent fields (nearest-shift first)."""
    clusters = []
    for resonance, distance in sorted(candidates, key=lambda item: item[0].omega.real):
        if clusters:
            previous = clusters[-1][-1][0].omega
            if abs(resonance.omega ** 2 - previous ** 2) <= MERGE_TOLERANCE * max(abs(resonance.omega) ** 2, _TINY):
                clusters[-1].append((resonance, distance))
                continue
        clusters.append([(resonance, distance)])
    kept = []
    for cluster in clusters:
        basis = []
        for resonance, _ in sorted(cluster, key=lambda item: item[1]):
            field = np.asarray(resonance.field, dtype=np.complex128).ravel()
            residual = field.copy()
            for direction in basis:
                residual -= direction * np.vdot(direction, residual)
            size, original = float(np.linalg.norm(residual)), float(np.linalg.norm(field))
            if not size > DUPLICATE_RESIDUAL * original:
                continue
            basis.append(residual / size)
            kept.append(resonance)
    kept.sort(key=lambda resonance: resonance.omega.real)
    return kept


def _combined_info(solves, started, warnings) -> dict:
    """HarmonicResonanceSolve.info of several shift solves: the first's labels, summed counts / timings, the largest
    factor, every distinct warning, target_frequency NaN."""
    info = dict(solves[0].info) if solves else {'n_free': 0, 'pencil': '', 'label': '', 'factor': '',
                                                'factor_host_bytes': 0, 'factor_device_bytes': 0,
                                                'region': False, 'min_region_fraction': 0.0}
    for key in _SUMMED_INFO:
        info[key] = sum(solve.info.get(key, 0) for solve in solves)
    for key in ('factor_host_bytes', 'factor_device_bytes'):
        info[key] = max((int(solve.info.get(key, 0)) for solve in solves), default=0)
    combined = []
    for text in [text for solve in solves for text in solve.info['warnings']] + list(warnings):
        if text not in combined:
            combined.append(text)
    info.update(target_frequency=math.nan, seconds=time.perf_counter() - started, warnings=combined)
    return info
