"""Harmonic channels and the mixing table of the NLFD-chi solver (numpy only, GUI-thread safe): channel resolution
(input wavelengths, integer combinations, ranks), MixingTerm enumeration with degeneracy factors, describe lines."""
# A channel is one (frequency, polarisation): 'ez' carries E_z (model component 2), 'hz' carries E_x, E_y (0, 1).
# Frequencies of generated channels are integer combinations of the input channels, so a process frequency matches a
# channel exactly by its coefficient vector; channels given by separate wavelengths are compared with 1e-12 relative.
# Response of order n at channel k (plan 2.1): P_k = eps0 2^(1-n) sum over ordered signed source tuples chi(n) : E..E,
# E_{-m} = conj(E_m). The tensors are symmetric in their source indices (tensors.symmetrize_sources), so one term per
# multiset of signed sources carries factor 2^(1-n) x (number of orderings).
import itertools
import math
import re
from dataclasses import dataclass, replace

import numpy as np

from ...core.constants import C0
from .tensors import symmetrize_sources

POLARIZATIONS = ('ez', 'hz', 'both')
POLARIZATION_COMPONENTS = {'ez': (2,), 'hz': (0, 1)}
_AXIS_LETTERS = 'xyz'
FREQUENCY_MATCH_RTOL = 1e-12   # separately given wavelengths: a process lands on a channel within this
NEAR_MISS_RTOL = 1e-3          # ... and is reported as a near miss below this
COMPONENT_RTOL = 1e-14         # tensor entries below this x max |T| are rotation round-off (dropped)
_NAME = re.compile(r'[A-Za-z_][A-Za-z0-9_]*\Z')
_TERM = re.compile(r'\s*([+-])?\s*(\d+)?\s*\*?\s*([A-Za-z_][A-Za-z0-9_]*)\s*')


@dataclass(frozen=True)
class ChannelSpec:
    """One harmonic_channel feature: an input (wavelength [m], Bloch k) or a generated channel (combination text)."""
    name: str
    polarization: str = 'ez'          # 'ez' | 'hz' | 'both' (one channel per polarisation)
    wavelength: float = None          # input channel [m]
    combination: str = None           # generated channel, e.g. '2 p', 'p + s', '2 p - s'
    bloch_k: tuple = (0.0, 0.0)       # input channel (k_x, k_y) [rad/m] on periodic axes; generated: the combination's


@dataclass(frozen=True)
class Channel:
    """Resolved channel: name unique per (frequency, polarisation); coefficients over the input channels."""
    name: str                 # spec name, or '<name>_ez' / '<name>_hz' for a 'both' spec
    frequency_name: str       # the spec name
    polarization: str         # 'ez' | 'hz'
    coefficients: tuple       # ints, one per input channel (input order)
    omega: float              # rad/s
    rank: int                 # inputs 0, a combination 1 + max rank of the channels it names
    bloch_k: tuple = (0.0, 0.0)   # (k_x, k_y) [rad/m] = sum of coefficients x input k (exact phase matching)

    @property
    def components(self):
        return POLARIZATION_COMPONENTS[self.polarization]

    @property
    def parity_exponent(self):
        """0 or 1: the channel's E mirror parity is s^e on a face whose input parity is s (s^(sum of coefficients):
        an even combination, e.g. a second harmonic, is mirror-even on every plane)."""
        return int(sum(self.coefficients)) % 2


@dataclass(frozen=True, eq=False)
class MediumTensor:
    """chi(n) of one medium in model axes [(m/V)^(n-1)], source indices symmetrised; process_scales = ((process,
    real factor), ...) multiplying the factor of that process's terms (per-process chi3, e.g. THG)."""
    name: str
    order: int
    tensor: np.ndarray
    process_scales: tuple = ()

    @classmethod
    def from_tensor(cls, name, order, tensor, process_scales=()):
        tensor = np.asarray(tensor)
        if tensor.shape != (3,) * (order + 1):
            raise ValueError(f"medium {name!r}: a chi({order}) tensor has shape {(3,) * (order + 1)}, got "
                             f"{tensor.shape}")
        return cls(name=name, order=int(order), tensor=symmetrize_sources(tensor),
                   process_scales=tuple((str(process), float(scale)) for process, scale in process_scales))


@dataclass(frozen=True)
class MixingTerm:
    """One process landing on channel `target`: P += eps0 factor sum chi[i, j..] E_s1[j] .. on medium `medium`."""
    process: str              # 'SHG', 'SFG', 'DFG', 'THG', 'TFG', 'SPM', 'XPM', 'FWM'
    order: int
    target: int               # channel index
    sources: tuple            # sorted ((channel index, conjugate), ...)
    factor: float             # 2^(1-n) x orderings
    medium: str
    kerr: bool                # chi(3) |E_m|^2 E_target (operator term in the pump-depleted mode)
    components: tuple         # ((i, (j, ..), value), ...) non-zero entries the target / source channels carry
    exact: bool               # frequency matched by the coefficient vectors (else within FREQUENCY_MATCH_RTOL)


@dataclass(frozen=True)
class MixingTable:
    """Every process of the enabled media landing on a declared channel, plus validation notes."""
    channels: tuple
    terms: tuple
    dropped: tuple = ()        # processes on a channel whose contraction vanishes for the polarisations
    near_misses: tuple = ()    # products within NEAR_MISS_RTOL of a channel but not on it
    off_set: tuple = ()        # examples of products off the channel set (lowest order first)
    off_set_count: int = 0

    def terms_for(self, target):
        return tuple(term for term in self.terms if term.target == target)

    def undepleted_terms(self):
        """Rank-ordered cascade (decision 6): terms whose source channels all rank below the target."""
        ranks = [channel.rank for channel in self.channels]
        return tuple(term for term in self.terms
                     if all(ranks[index] < ranks[term.target] for index, _ in term.sources))


def parse_combination(text):
    """{channel name: integer coefficient} of an integer combination such as '2 p - s', 'p + s', '2*p' (no eval);
    ValueError for malformed text or an all-zero combination."""
    text = str(text)
    coefficients = {}
    position = 0
    while position < len(text.rstrip()) or not coefficients:
        match = _TERM.match(text, position)
        if match is None or match.end() == position:
            raise ValueError(f"combination {text!r}: expected '[+|-] [integer] name' at character {position + 1} "
                             f"(e.g. '2 p - s')")
        sign, count, name = match.groups()
        if sign is None and coefficients:
            raise ValueError(f"combination {text!r}: missing + or - before {name!r}")
        value = (int(count) if count is not None else 1) * (-1 if sign == '-' else 1)
        coefficients[name] = coefficients.get(name, 0) + value
        position = match.end()
    coefficients = {name: value for name, value in coefficients.items() if value != 0}
    if not coefficients:
        raise ValueError(f"combination {text!r} cancels to zero")
    return coefficients


def resolve_channels(specs):
    """(channels tuple, input names) of ChannelSpecs. ValueError for duplicate or invalid names, unknown / cyclic
    references, a frequency <= 0 or two channels at one frequency and polarisation."""
    specs = tuple(specs)
    names = [spec.name for spec in specs]
    for spec in specs:
        if not _NAME.match(str(spec.name)):
            raise ValueError(f"channel name {spec.name!r} must be an identifier (letters, digits, _)")
        if spec.polarization not in POLARIZATIONS:
            raise ValueError(f"channel {spec.name}: polarization must be one of {', '.join(POLARIZATIONS)}, got "
                             f"{spec.polarization!r}")
        if (spec.wavelength is None) == (spec.combination is None):
            raise ValueError(f"channel {spec.name}: give either a wavelength (input) or a combination (generated)")
        bloch_k = tuple(float(value) for value in spec.bloch_k)
        if len(bloch_k) != 2 or not all(math.isfinite(value) for value in bloch_k):
            raise ValueError(f"channel {spec.name}: bloch_k must be two finite (k_x, k_y) [rad/m], got {spec.bloch_k!r}")
        if spec.combination is not None and any(bloch_k):
            raise ValueError(f"channel {spec.name}: a generated channel takes the Bloch k of its combination; give "
                             "bloch_k on the input channels only")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"channel names must be unique, repeated: {', '.join(duplicates)}")
    by_name = {spec.name: spec for spec in specs}
    input_names = tuple(spec.name for spec in specs if spec.wavelength is not None)
    if not input_names:
        raise ValueError("at least one channel needs a wavelength (an input channel); combinations alone have no "
                         "frequency")
    input_omegas = []
    for name in input_names:
        wavelength = float(by_name[name].wavelength)
        if not (math.isfinite(wavelength) and wavelength > 0.0):
            raise ValueError(f"channel {name}: wavelength must be finite and > 0 [m], got {wavelength!r}")
        input_omegas.append(2.0 * math.pi * C0 / wavelength)
    input_bloch = np.array([[float(value) for value in by_name[name].bloch_k] for name in input_names],
                           dtype=np.float64)   # (n_inputs, 2)

    resolved = {}   # name -> (coefficients np int array, rank)

    def resolve(name, path):
        if name in resolved:
            return resolved[name]
        if name in path:
            raise ValueError(f"channel combinations form a cycle: {' -> '.join((*path, name))}")
        spec = by_name[name]
        if spec.wavelength is not None:
            vector = np.zeros(len(input_names), dtype=np.int64)
            vector[input_names.index(name)] = 1
            resolved[name] = (vector, 0)
            return resolved[name]
        try:
            terms = parse_combination(spec.combination)
        except ValueError as exc:
            raise ValueError(f"channel {name}: {exc}") from None
        unknown = sorted(set(terms) - set(by_name))
        if unknown:
            raise ValueError(f"channel {name}: combination {spec.combination!r} names unknown channels "
                             f"{', '.join(unknown)} (known: {', '.join(names)})")
        vector = np.zeros(len(input_names), dtype=np.int64)
        rank = 0
        for other, count in terms.items():
            other_vector, other_rank = resolve(other, (*path, name))
            vector = vector + count * other_vector
            rank = max(rank, other_rank + 1)
        resolved[name] = (vector, rank)
        return resolved[name]

    channels = []
    for spec in specs:
        vector, rank = resolve(spec.name, ())
        omega = float(np.dot(vector, input_omegas))
        scale = float(np.dot(np.abs(vector), input_omegas))
        if not omega > FREQUENCY_MATCH_RTOL * scale:
            raise ValueError(f"channel {spec.name}: combination {spec.combination!r} gives frequency "
                             f"{omega / (2.0 * math.pi):.6g} Hz; generated channels need a frequency > 0")
        bloch_k = tuple(float(value) for value in vector @ input_bloch)
        polarizations = ('ez', 'hz') if spec.polarization == 'both' else (spec.polarization,)
        for polarization in polarizations:
            name = f'{spec.name}_{polarization}' if spec.polarization == 'both' else spec.name
            channels.append(Channel(name=name, frequency_name=spec.name, polarization=polarization,
                                    coefficients=tuple(int(value) for value in vector), omega=omega, rank=rank,
                                    bloch_k=bloch_k))
    expanded = [channel.name for channel in channels]
    clashes = sorted({name for name in expanded if expanded.count(name) > 1})
    if clashes:
        raise ValueError(f"channel names clash after splitting 'both' channels: {', '.join(clashes)}")
    for first, second in itertools.combinations(channels, 2):
        if (first.polarization == second.polarization
                and abs(first.omega - second.omega) <= FREQUENCY_MATCH_RTOL * max(first.omega, second.omega)):
            raise ValueError(f"channels {first.frequency_name} and {second.frequency_name} have the same frequency "
                             f"and polarisation ({first.polarization}); declare it once")
    return tuple(channels), input_names


def solve_order(channels):
    """Channel indices in solve order: rank, then ascending omega, then name."""
    return tuple(sorted(range(len(channels)),
                        key=lambda index: (channels[index].rank, channels[index].omega, channels[index].name)))


def _process_name(order, target, sources):
    """Process label and the Kerr flag (chi(3) multiset {target, m, conj m}) of a source multiset."""
    plain = sorted(index for index, conjugate in sources if not conjugate)
    conjugated = [index for index, conjugate in sources if conjugate]
    if order == 2:
        if conjugated:
            return 'DFG', False
        return ('SHG' if plain[0] == plain[1] else 'SFG'), False
    if order == 3:
        if len(conjugated) == 1 and target in plain:
            remaining = list(plain)
            remaining.remove(target)
            if remaining == conjugated:
                return ('SPM' if conjugated[0] == target else 'XPM'), True
        if not conjugated:
            return ('THG' if len(set(plain)) == 1 else 'TFG'), False
        return 'FWM', False
    return f'chi{order}', False


def _active_components(tensor, target_components, source_components):
    """((i, (j, ..), value), ...) entries above COMPONENT_RTOL x max |T| with i / j.. in the given component sets."""
    scale = float(np.max(np.abs(tensor)))
    if scale == 0.0:
        return ()
    entries = []
    for index in itertools.product(target_components, *source_components):
        value = tensor[index]
        if abs(value) > COMPONENT_RTOL * scale:
            value = complex(value) if np.iscomplexobj(tensor) else float(value)
            entries.append((index[0], tuple(int(j) for j in index[1:]), value))
    return tuple(entries)


def source_text(channels, sources):
    """'p + p', 'p - s', 'p + s - i' of a source multiset (conjugated sources subtracted)."""
    plain = [channels[index].name for index, conjugate in sources if not conjugate]
    conjugated = [channels[index].name for index, conjugate in sources if conjugate]
    return ' + '.join(plain) + ''.join(f' - {name}' for name in conjugated)


def build_mixing_table(channels, media):
    """MixingTable of resolved channels and MediumTensors: every signed source multiset of each medium's order whose
    frequency lands on a channel and whose contraction is non-zero for the channel polarisations."""
    channels = tuple(channels)
    coefficient_vectors = [np.asarray(channel.coefficients, dtype=np.int64) for channel in channels]
    signed = [(index, False) for index in range(len(channels))] + [(index, True) for index in range(len(channels))]
    max_omega = max(channel.omega for channel in channels)
    terms, dropped, near_misses, off_set = [], [], [], []
    off_set_count = 0
    for medium in sorted(media, key=lambda medium: medium.order):
        order = medium.order
        process_scales = dict(medium.process_scales)
        for sources in itertools.combinations_with_replacement(signed, order):
            omega = sum(-channels[index].omega if conjugate else channels[index].omega for index, conjugate in sources)
            if not omega > FREQUENCY_MATCH_RTOL * max_omega:
                continue
            vector = sum((-1 if conjugate else 1) * coefficient_vectors[index] for index, conjugate in sources)
            counts = {}
            for source in sources:
                counts[source] = counts.get(source, 0) + 1
            orderings = math.factorial(order) // math.prod(math.factorial(count) for count in counts.values())
            source_components = [channels[index].components for index, _ in sources]
            landed = False
            for target, channel in enumerate(channels):
                exact = bool(np.array_equal(vector, coefficient_vectors[target]))
                mismatch = abs(omega - channel.omega) / channel.omega
                if not (exact or mismatch <= FREQUENCY_MATCH_RTOL):
                    if mismatch < NEAR_MISS_RTOL:
                        near_misses.append(f'{source_text(channels, sources)} is {mismatch:.2e} (relative) off '
                                           f'channel {channel.name}: not coupled')
                    continue
                landed = True
                process, kerr = _process_name(order, target, sources)
                components = _active_components(medium.tensor, channel.components, source_components)
                label = f'{process} {source_text(channels, sources)} -> {channel.name} (medium {medium.name})'
                if not components:
                    dropped.append(f'{label}: no tensor component couples these polarisations')
                    continue
                factor = 2.0 ** (1 - order) * orderings * process_scales.get(process, 1.0)
                if factor == 0.0:
                    dropped.append(f'{label}: process scale 0')
                    continue
                terms.append(MixingTerm(process=process, order=order, target=target, sources=tuple(sources),
                                        factor=factor, medium=medium.name, kerr=kerr, components=components,
                                        exact=exact))
            if not landed and _active_components(medium.tensor, (0, 1, 2), source_components):
                off_set_count += 1
                if len(off_set) < 5:
                    wavelength_nm = 2.0 * math.pi * C0 / omega * 1e9
                    off_set.append(f'chi{order} {source_text(channels, sources)} at {wavelength_nm:.4g} nm '
                                   f'(medium {medium.name})')
    return MixingTable(channels=channels, terms=tuple(terms), dropped=tuple(dropped),
                       near_misses=tuple(dict.fromkeys(near_misses)), off_set=tuple(off_set),
                       off_set_count=off_set_count)


# Bloch k and mirror parity of a term: P of signed sources carries sum (+-k_source) and the product of the source
# parities, which an exact term (coefficient vectors) always matches on its target; a term landing by frequency only
# (separately given wavelengths) must match them too, else its polarisation is no Bloch wave / no definite-parity field
# of the target's boundary conditions.
BLOCH_PHASE_RTOL = 1e-6   # (k_target - sum +-k_sources) L off a multiple of 2 pi by more than this x max(1 rad, sum
#                           |k| L) is a mismatch (the float32 grid widths of the run vs the mesh edges of validation)


def inexact_term_issues(table, periods, odd_mirror):
    """Problem texts of the inexact terms (frequency match only) whose sources' Bloch phase (k L per periodic axis,
    periods = (L_x or None, L_y or None) [m]) misses the target's modulo 2 pi, or (odd_mirror: the model has an
    antisymmetric mirror face) whose sources' parity exponents differ from the target's."""
    problems = []
    channels = table.channels
    for term in table.terms:
        if term.exact:
            continue
        target = channels[term.target]
        label = f'{term.process} {source_text(channels, term.sources)} -> {target.name}'
        for axis, period in enumerate(periods):
            if period is None:
                continue
            carried = sum((-1.0 if conjugate else 1.0) * channels[index].bloch_k[axis]
                          for index, conjugate in term.sources)
            phase = (target.bloch_k[axis] - carried) * float(period)
            scale = (abs(target.bloch_k[axis]) + sum(abs(channels[index].bloch_k[axis]) for index, _ in term.sources)
                     ) * float(period)
            if abs(phase - 2.0 * math.pi * round(phase / (2.0 * math.pi))) > BLOCH_PHASE_RTOL * max(1.0, scale):
                problems.append(f"{label}: its polarisation carries Bloch k{'xy'[axis]} = {carried:.6g} rad/m, but "
                                f"{target.name} has {target.bloch_k[axis]:.6g} rad/m (not equal modulo 2 pi / "
                                f'period); set the input Bloch k so that they match')
        exponent = sum(channels[index].parity_exponent for index, _ in term.sources) % 2
        if odd_mirror and exponent != target.parity_exponent:
            problems.append(f'{label}: with an antisymmetric mirror face its polarisation has the other mirror parity '
                            f'than {target.name} (an input channel follows the face setting); declare {target.name} '
                            'as a combination of the inputs instead')
    return problems


# Miller's rule: chi(n)(w_target; w_1..w_n) ~ chi1(w_target) prod chi1(w_i), normalised to the process the
# coefficients are given for: chi2 SHG at the reference w_r (target 2 w_r, sources w_r, w_r), chi3 SPM at w_r (n2).
MILLER_REFERENCE_HARMONICS = {2: (2, (1, 1)), 3: (1, (1, 1, 1))}   # order -> (target, sources) in units of w_r


def miller_scales(table, chi1_by_medium):
    """{term index: real Miller factor} of the terms whose medium is in chi1_by_medium = {medium: (chi1 per channel
    index, reference product)} (Re chi1 = Re eps - 1 of the medium at each channel frequency)."""
    scales = {}
    for index, term in enumerate(table.terms):
        entry = chi1_by_medium.get(term.medium)
        if entry is None:
            continue
        chi1, reference = entry
        scales[index] = chi1[term.target] * math.prod(chi1[source] for source, _ in term.sources) / reference
    return scales


def scaled_table(table, scales):
    """MixingTable with term factors multiplied by scales {term index: factor} (Miller's rule at build)."""
    terms = tuple(replace(term, factor=term.factor * scales[index]) if index in scales else term
                  for index, term in enumerate(table.terms))
    return replace(table, terms=terms)


def describe_mixing(table, terms=None, max_components=3):
    """Lines 'SHG p + p -> sh, chi2 medium ppln, factor 0.5, chi_zzz' for validation info and the run log."""
    lines = []
    for term in (table.terms if terms is None else terms):
        strongest = sorted(term.components, key=lambda entry: -abs(entry[2]))[:max_components]
        names = ', '.join('chi_' + _AXIS_LETTERS[i] + ''.join(_AXIS_LETTERS[j] for j in js) for i, js, _ in strongest)
        more = f' (+{len(term.components) - len(strongest)} more)' if len(term.components) > len(strongest) else ''
        lines.append(f'{term.process} {source_text(table.channels, term.sources)} -> '
                     f'{table.channels[term.target].name}, chi{term.order} medium {term.medium}, factor '
                     f'{term.factor:g}, {names}{more}')
    return lines
