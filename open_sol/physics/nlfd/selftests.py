"""Self-tests of NLFD-chi (nonlinear_optics_plan section 9): CPU cases 1-3 (mixing table, tensors, C / C* power
identity) + the Kerr linearisation and the Newton-Krylov tangents, GPU model cases 4, 12 (undepleted SHG vs the
3-point lattice Green's function), 8 (SPM), 6 (pump-depleted tanh^2) with picard and newton runs, and the templates;
QPM, THG and OPA gain are checked by the examples/em analytic_nlfd_* models."""
import cmath
import math
from dataclasses import replace

import numpy as np

from ...core.constants import C0, EPS0, MU0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import (PERIODIC_Y, analytic_slab_index, normalized_result, probe_phasor, relative_error,
                                       worst_result)
from ...model.spec import (AnalyticCurveSpec, DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec,
                           MeshSpec, ModelSpec, ParameterSpec, PhysicsSpec, StudySpec, SweepSpec)

_PM_PER_V = 1e-12
_LN_D = {'d22': 2.1 * _PM_PER_V, 'd31': -4.6 * _PM_PER_V, 'd33': -25.2 * _PM_PER_V}   # 3m, LN-like [m/V]
_GAAS_D14 = 119.0 * _PM_PER_V
_SILICON_N2 = 4.5e-18     # [m^2/W]
_SILICON_BETA = 8.0e-12   # [m/W]
_SILICON_N0 = 3.48
_IDENTITY_TOLERANCE = 1e-12
_ROUND_OFF_TOLERANCE = 1e-13


def _result(violations, details, measured=None, tolerance=0.0):
    """Self-test result dict: error = measured (or 0) + number of violations."""
    measured = 0.0 if measured is None else float(measured)
    details['violations'] = violations
    return {'measured': measured, 'expected': 0.0, 'error': measured + len(violations), 'tolerance': tolerance,
            'details': details}


def _rows(table):
    """{(target name, sorted ((source name, conjugate), ..)): (factor, process, kerr)} of a MixingTable."""
    names = [channel.name for channel in table.channels]
    return {(names[term.target], tuple(sorted((names[index], conjugate) for index, conjugate in term.sources))):
            (term.factor, term.process, term.kerr) for term in table.terms}


def _key(target, *sources):
    """Row key of _rows: sources as 'p' or '-p' (conjugate)."""
    return target, tuple(sorted((source.lstrip('-'), source.startswith('-')) for source in sources))


def _check_mixing_degeneracy(results):
    from .mixing import ChannelSpec, MediumTensor, build_mixing_table, parse_combination, resolve_channels
    from .tensors import chi2_from_voigt, chi3_isotropic, voigt_matrix
    violations = []

    # 1. combination parser and channel resolution
    parsed = {'2 p - s': {'p': 2, 's': -1}, 'p + s': {'p': 1, 's': 1}, '2p': {'p': 2}, '2*p + 0 s': {'p': 2},
              ' - s + 3 p': {'s': -1, 'p': 3}}
    for text, expected in parsed.items():
        try:
            got = parse_combination(text)
        except ValueError as exc:
            violations.append(f'parse {text!r} raised: {exc}')
            continue
        if got != expected:
            violations.append(f'parse {text!r} gave {got}, expected {expected}')
    for text in ('p s', 'p - p', '', '2 + p', 'p +'):
        try:
            parse_combination(text)
            violations.append(f'parse {text!r} did not raise')
        except ValueError:
            pass
    bad_sets = {
        'cycle': (ChannelSpec('p', wavelength=1.55e-6), ChannelSpec('a', combination='b + p'),
                  ChannelSpec('b', combination='a - p')),
        'unknown name': (ChannelSpec('p', wavelength=1.55e-6), ChannelSpec('sh', combination='2 q')),
        'negative frequency': (ChannelSpec('p', wavelength=1.55e-6), ChannelSpec('n', combination='p - 2 p')),
        'duplicate frequency': (ChannelSpec('p', wavelength=1.55e-6), ChannelSpec('q', combination='3 p - 2 p')),
        'no input': (ChannelSpec('sh', combination='2 sh'),),
    }
    for label, specs in bad_sets.items():
        try:
            resolve_channels(specs)
            violations.append(f'resolve_channels with {label} did not raise')
        except ValueError:
            pass
    channels, _ = resolve_channels((ChannelSpec('p', 'both', wavelength=1.55e-6),
                                    ChannelSpec('sh', combination='2 p')))
    names_ranks = [(channel.name, channel.rank) for channel in channels]
    if names_ranks != [('p_ez', 0), ('p_hz', 0), ('sh', 1)]:
        violations.append(f"'both' expansion / ranks gave {names_ranks}")
    if abs(channels[2].omega - 2.0 * channels[0].omega) > 1e-15 * channels[2].omega:
        violations.append('generated channel frequency is not exactly 2 omega_p')

    z_cut_d33 = MediumTensor.from_tensor('d33', 2, chi2_from_voigt(voigt_matrix('general', {'d33': 1e-11})))

    # 2. chi(2): SHG, SFG, DFG and back-action rows with their factors; the undepleted rank filter
    channels, _ = resolve_channels((ChannelSpec('p', wavelength=1.55e-6), ChannelSpec('s', wavelength=2.0e-6),
                                    ChannelSpec('sh', combination='2 p'), ChannelSpec('sf', combination='p + s'),
                                    ChannelSpec('i', combination='p - s')))
    table = build_mixing_table(channels, [z_cut_d33])
    rows = _rows(table)
    expected_rows = {_key('sh', 'p', 'p'): (0.5, 'SHG', False), _key('sf', 'p', 's'): (1.0, 'SFG', False),
                     _key('i', 'p', '-s'): (1.0, 'DFG', False), _key('p', 'sh', '-p'): (1.0, 'DFG', False),
                     _key('s', 'p', '-i'): (1.0, 'DFG', False), _key('sh', 'sf', 'i'): (1.0, 'SFG', False)}
    for key, expected in expected_rows.items():
        if rows.get(key) != expected:
            violations.append(f'chi2 row {key}: got {rows.get(key)}, expected {expected}')
    undepleted = set(_rows(type(table)(channels=table.channels, terms=table.undepleted_terms())))
    expected_undepleted = {_key('sh', 'p', 'p'), _key('sf', 'p', 's'), _key('i', 'p', '-s')}
    if undepleted != expected_undepleted:
        violations.append(f'undepleted terms {sorted(undepleted)}, expected {sorted(expected_undepleted)}')

    # 3. chi(3): THG, SPM, XPM, FWM; Kerr flags
    chi3 = MediumTensor.from_tensor('kerr', 3, chi3_isotropic(1e-19))
    channels, _ = resolve_channels((ChannelSpec('p', wavelength=1.55e-6), ChannelSpec('s', wavelength=1.6e-6),
                                    ChannelSpec('th', combination='3 p'), ChannelSpec('i3', combination='2 p - s')))
    rows = _rows(build_mixing_table(channels, [chi3]))
    expected_rows = {_key('th', 'p', 'p', 'p'): (0.25, 'THG', False), _key('p', 'p', 'p', '-p'): (0.75, 'SPM', True),
                     _key('p', 'p', 's', '-s'): (1.5, 'XPM', True), _key('s', 's', 'p', '-p'): (1.5, 'XPM', True),
                     _key('i3', 'p', 'p', '-s'): (0.75, 'FWM', False)}
    for key, expected in expected_rows.items():
        if rows.get(key) != expected:
            violations.append(f'chi3 row {key}: got {rows.get(key)}, expected {expected}')

    # 4. zero contractions dropped: d33-only z-cut with hz channels couples nothing
    channels, _ = resolve_channels((ChannelSpec('p', 'hz', wavelength=1.55e-6),
                                    ChannelSpec('sh', 'hz', combination='2 p')))
    table = build_mixing_table(channels, [z_cut_d33])
    if table.terms or not table.dropped:
        violations.append(f'hz channels in a d33-only z-cut medium gave {len(table.terms)} terms, '
                          f'{len(table.dropped)} dropped notes (expected 0 terms, dropped notes)')

    # 5. separately given commensurate wavelengths match within 1e-12 (inexact) and are coupled
    channels, _ = resolve_channels((ChannelSpec('p', wavelength=1.55e-6), ChannelSpec('h', wavelength=0.775e-6)))
    shg = [term for term in build_mixing_table(channels, [z_cut_d33]).terms if term.target == 1]
    if not (len(shg) == 1 and shg[0].process == 'SHG' and not shg[0].exact):
        violations.append(f'commensurate wavelengths: SHG terms {shg}')

    # 6. mixed polarisation: -43m z = [001], hz pump -> SH only in the ez channel, P_z = 2 eps0 d14 E_x E_y
    gaas = MediumTensor.from_tensor('gaas', 2, chi2_from_voigt(voigt_matrix('-43m', {'d14': _GAAS_D14})))
    channels, _ = resolve_channels((ChannelSpec('p', 'hz', wavelength=1.55e-6),
                                    ChannelSpec('sh', 'both', combination='2 p')))
    table = build_mixing_table(channels, [gaas])
    shg = [term for term in table.terms if channels[term.target].frequency_name == 'sh']
    if len(shg) != 1 or channels[shg[0].target].polarization != 'ez':
        violations.append(f'-43m hz pump: SH terms {shg} (expected one, in the ez channel)')
    else:
        # sum over the contracted entries for E_x = E_y = 1: factor x (chi_zxy + chi_zyx) = 0.5 x 4 d14
        effective = shg[0].factor * sum(value for component, _, value in shg[0].components if component == 2)
        if abs(effective - 2.0 * _GAAS_D14) > 1e-12 * _GAAS_D14:
            violations.append(f'-43m P_z / (eps0 E_x E_y) = {effective:.6g}, expected 2 d14 = {2 * _GAAS_D14:.6g}')
    return _result(violations, {'checked_tables': 6})


def _signed_permutation_rotation(tensor, rotation):
    """T'_{ij..} = s_i s_j .. T_{a(i) b(j) ..} of a signed-permutation rotation (R e_a = s e_pi(a)), by index mapping."""
    rotation = np.round(rotation)
    crystal_index = [int(np.flatnonzero(rotation[i])[0]) for i in range(3)]
    signs = [rotation[i, crystal_index[i]] for i in range(3)]
    rotated = np.zeros_like(tensor)
    for index in np.ndindex(tensor.shape):
        rotated[index] = math.prod(signs[i] for i in index) * tensor[tuple(crystal_index[i] for i in index)]
    return rotated


def _check_tensor_rotation(results):
    from .tensors import (CRYSTAL_CLASSES, CUT_ROTATIONS, chi2_from_voigt, chi3_cubic, chi3_from_n2_beta,
                          chi3_isotropic, euler_rotation, kleinman_asymmetry, n2_beta_from_chi3, rotate_tensor,
                          voigt_matrix)
    violations = []
    rng = np.random.default_rng(7)

    # 1. 3m Voigt -> rank 3 (chi = 2 d, symmetric in the source indices)
    ln = chi2_from_voigt(voigt_matrix('3m', _LN_D))
    d22, d31, d33 = _LN_D['d22'], _LN_D['d31'], _LN_D['d33']
    for index, expected in {(2, 2, 2): 2 * d33, (2, 0, 0): 2 * d31, (2, 1, 1): 2 * d31, (0, 0, 2): 2 * d31,
                            (0, 2, 0): 2 * d31, (1, 1, 2): 2 * d31, (1, 1, 1): 2 * d22, (1, 0, 0): -2 * d22,
                            (0, 0, 1): -2 * d22, (0, 1, 0): -2 * d22}.items():
        if abs(ln[index] - expected) > 1e-15 * abs(expected):
            violations.append(f'3m chi{index} = {ln[index]:.6g}, expected {expected:.6g}')

    # 2. Kleinman symmetry of the class forms with Kleinman-consistent values; 'general' random is not symmetric
    kleinman_values = {'3m': _LN_D, '-43m': {'d14': _GAAS_D14}, '6mm': {'d15': -2e-12, 'd31': -2e-12, 'd33': 5e-12},
                       'mm2': {'d15': 2e-12, 'd24': 3.6e-12, 'd31': 2e-12, 'd32': 3.6e-12, 'd33': 13.7e-12},
                       '32': {'d11': 0.3e-12, 'd14': 0.0}}
    asymmetries = {}
    for crystal_class, values in kleinman_values.items():
        asymmetries[crystal_class] = kleinman_asymmetry(chi2_from_voigt(voigt_matrix(crystal_class, values)))
        if asymmetries[crystal_class] > 1e-15:
            violations.append(f'{crystal_class}: Kleinman asymmetry {asymmetries[crystal_class]:.3g}')
    general = {name: float(value) for name, value in zip(CRYSTAL_CLASSES['general'], rng.normal(size=18) * 1e-12)}
    asymmetries['general'] = kleinman_asymmetry(chi2_from_voigt(voigt_matrix('general', general)))
    if asymmetries['general'] < 1e-3:
        violations.append(f"random 'general' d: Kleinman asymmetry {asymmetries['general']:.3g} (expected > 1e-3)")

    # 3. Euler rotations orthonormal; 90 degree rotations and the cut presets = signed index permutations
    rotation_errors = []
    for angles in rng.uniform(-180.0, 180.0, size=(4, 3)):
        rotation = euler_rotation(*angles)
        rotation_errors.append(float(np.max(np.abs(rotation @ rotation.T - np.eye(3)))))
        rotation_errors.append(abs(float(np.linalg.det(rotation)) - 1.0))
        back = rotate_tensor(rotate_tensor(ln, rotation), rotation.T)
        rotation_errors.append(float(np.max(np.abs(back - ln))) / float(np.max(np.abs(ln))))
    permutation_rotations = {**CUT_ROTATIONS, 'euler(90, 0, 0)': euler_rotation(90.0, 0.0, 0.0),
                             'euler(0, 90, 0)': euler_rotation(0.0, 90.0, 0.0),
                             'euler(90, 90, 90)': euler_rotation(90.0, 90.0, 90.0)}
    for label, rotation in permutation_rotations.items():
        mapped = _signed_permutation_rotation(ln, rotation)
        error = float(np.max(np.abs(rotate_tensor(ln, rotation) - mapped))) / float(np.max(np.abs(ln)))
        rotation_errors.append(error)
        if error > _ROUND_OFF_TOLERANCE:
            violations.append(f'{label}: rotated tensor differs from the index permutation by {error:.3g}')
    if CUT_ROTATIONS['x_cut'] @ np.array([0.0, 0.0, 1.0]) @ np.array([1.0, 0.0, 0.0]) != 1.0:
        violations.append("'x_cut' does not put the crystal c axis along model x")
    if max(rotation_errors) > _ROUND_OFF_TOLERANCE:
        violations.append(f'rotation round-off {max(rotation_errors):.3g} above {_ROUND_OFF_TOLERANCE}')

    # 4. chi(3): isotropic invariant under any rotation; cubic (rho 1.27) under 90 degrees about z only
    isotropic = chi3_isotropic(1e-19)
    rotation = euler_rotation(*rng.uniform(-180.0, 180.0, size=3))
    isotropic_error = float(np.max(np.abs(rotate_tensor(isotropic, rotation) - isotropic))) / 1e-19
    if isotropic_error > _ROUND_OFF_TOLERANCE:
        violations.append(f'isotropic chi3 changed by {isotropic_error:.3g} under a rotation')
    cubic = chi3_cubic(1e-19, 1.27)
    quarter_turn = float(np.max(np.abs(rotate_tensor(cubic, euler_rotation(90.0, 0.0, 0.0)) - cubic))) / 1e-19
    eighth_turn = float(np.max(np.abs(rotate_tensor(cubic, euler_rotation(45.0, 0.0, 0.0)) - cubic))) / 1e-19
    if quarter_turn > _ROUND_OFF_TOLERANCE or eighth_turn < 1e-3:
        violations.append(f'cubic chi3: 90 deg change {quarter_turn:.3g} (expected 0), 45 deg {eighth_turn:.3g} '
                          f'(expected > 1e-3)')
    if kleinman_asymmetry(cubic) > 1e-15:
        violations.append('cubic chi3 is not fully permutation symmetric')

    # 5. n2 / beta -> chi3 -> n2 / beta (exp(+j omega t): TPA gives Im chi3 < 0)
    omega = 2.0 * math.pi * C0 / 1.55e-6
    chi_xxxx = chi3_from_n2_beta(_SILICON_N2, _SILICON_BETA, _SILICON_N0, omega)
    n2, beta = n2_beta_from_chi3(chi_xxxx, _SILICON_N0, omega)
    round_trip = max(abs(n2 - _SILICON_N2) / _SILICON_N2, abs(beta - _SILICON_BETA) / _SILICON_BETA)
    if round_trip > _ROUND_OFF_TOLERANCE or not chi_xxxx.imag < 0.0:
        violations.append(f'n2 / beta round trip {round_trip:.3g}, Im chi3 = {chi_xxxx.imag:.3g} (must be < 0)')
    return _result(violations, {'kleinman_asymmetry': asymmetries, 'max_rotation_round_off': max(rotation_errors),
                                'chi3_xxxx_silicon': [chi_xxxx.real, chi_xxxx.imag], 'n2_beta_round_trip': round_trip})


def _random_complex(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _check_collocation_transpose(results):
    from ..em.harmonic import face_mean_at_centres
    from .media import bulk_medium
    from .mixing import ChannelSpec, build_mixing_table, resolve_channels
    from .sources import centres_to_faces, delivered_power, polarization_density
    from .tensors import chi2_from_voigt, chi3_from_n2_beta, chi3_isotropic, euler_rotation, rotate_tensor, voigt_matrix
    violations = []
    rng = np.random.default_rng(11)
    nx, ny = 7, 6
    widths = (rng.uniform(0.5, 1.5, size=nx) * 2e-8, rng.uniform(0.5, 1.5, size=ny) * 2e-8)   # graded
    cell_area = np.outer(*widths)

    # 1. <C u, v> = <u, C* v> in the area-weighted inner products, both axes, periodic or not
    transpose_errors = {}
    for axis in (0, 1):
        dual = 0.5 * (widths[axis] + np.roll(widths[axis], -1))
        face_area = np.outer(dual, widths[1]) if axis == 0 else np.outer(widths[0], dual)
        for periodic in (False, True):
            faces, centres = _random_complex(rng, (nx, ny)), _random_complex(rng, (nx, ny))
            lhs = np.sum(cell_area * face_mean_at_centres(faces, axis, periodic, xp=np) * centres)
            rhs = np.sum(face_area * faces * centres_to_faces(centres, axis, periodic, widths[axis], xp=np))
            scale = np.sum(np.abs(cell_area * face_mean_at_centres(faces, axis, periodic, xp=np) * centres))
            transpose_errors[f'axis {axis}, periodic {periodic}'] = float(abs(lhs - rhs) / scale)
    worst_transpose = max(transpose_errors.values())
    if worst_transpose > _ROUND_OFF_TOLERANCE:
        violations.append(f'C* is not the adjoint of C: {transpose_errors}')

    # 2. power exchange sum W_k = 0 for every table term on random fields (real Kleinman tensors, poled chi2 boxes)
    mask = np.zeros((nx, ny), dtype=bool)
    mask[1:6, 1:5] = True
    poling = rng.normal(size=(nx, ny))
    ln = rotate_tensor(chi2_from_voigt(voigt_matrix('3m', _LN_D)), euler_rotation(30.0, 50.0, 70.0))
    gaas = rotate_tensor(chi2_from_voigt(voigt_matrix('-43m', {'d14': _GAAS_D14})), euler_rotation(10.0, 40.0, 0.0))
    kerr = chi3_isotropic(chi3_from_n2_beta(_SILICON_N2, 0.0, _SILICON_N0, 1e15).real)
    cases = {
        'SHG': ((ChannelSpec('p', 'both', wavelength=1.55e-6), ChannelSpec('sh', 'both', combination='2 p')),
                [bulk_medium('ln', 2, ln, mask, poling, xp=np)]),
        'SFG + DFG': ((ChannelSpec('p', 'both', wavelength=1.55e-6), ChannelSpec('s', 'ez', wavelength=2.0e-6),
                       ChannelSpec('sf', 'both', combination='p + s'), ChannelSpec('i', 'hz', combination='p - s')),
                      [bulk_medium('gaas', 2, gaas, mask, poling, xp=np), bulk_medium('ln', 2, ln, mask, xp=np)]),
        'THG + SPM': ((ChannelSpec('p', 'both', wavelength=1.55e-6), ChannelSpec('th', 'both', combination='3 p')),
                      [bulk_medium('kerr', 3, kerr, mask, xp=np)]),
    }
    imbalances, face_errors = {}, {}
    for label, (specs, media) in cases.items():
        channels, _ = resolve_channels(specs)
        table = build_mixing_table(channels, [medium.mixing_tensor() for medium in media])
        fields = {}
        face_fields = {}
        for index, channel in enumerate(channels):
            if channel.polarization == 'ez':
                fields[index] = (None, None, _random_complex(rng, (nx, ny)))
            else:
                ex_faces, ey_faces = _random_complex(rng, (nx, ny)), _random_complex(rng, (nx, ny))
                face_fields[index] = (ex_faces, ey_faces)
                fields[index] = (face_mean_at_centres(ex_faces, 1, False, xp=np),
                                 face_mean_at_centres(ey_faces, 0, False, xp=np), None)
        by_name = {medium.name: medium for medium in media}
        exchange = []
        for target, channel in enumerate(channels):
            terms = table.terms_for(target)
            if not terms:
                continue
            density = polarization_density(terms, fields, by_name, channel.components, (nx, ny), xp=np)
            power = delivered_power(channel.omega, fields[target], density, cell_area, xp=np)
            exchange.append(power)
            if channel.polarization == 'hz':   # centre work = face work of the C*-distributed polarisation
                ex_faces, ey_faces = face_fields[target]
                dual_y = 0.5 * (widths[1] + np.roll(widths[1], -1))
                dual_x = 0.5 * (widths[0] + np.roll(widths[0], -1))
                face_power = 0.5 * channel.omega * float(np.sum(
                    np.outer(widths[0], dual_y)
                    * (np.conj(ex_faces) * centres_to_faces(density[0], 1, False, widths[1], xp=np)).imag
                    + np.outer(dual_x, widths[1])
                    * (np.conj(ey_faces) * centres_to_faces(density[1], 0, False, widths[0], xp=np)).imag))
                face_errors[f'{label}: {channel.name}'] = abs(face_power - power) / max(abs(power), 1e-300)
        total = sum(abs(power) for power in exchange)
        imbalances[label] = abs(sum(exchange)) / total if total > 0.0 else float('inf')
    worst_imbalance = max(imbalances.values())
    if worst_imbalance > _IDENTITY_TOLERANCE:
        violations.append(f'power exchange imbalance {imbalances}')
    worst_face = max(face_errors.values()) if face_errors else float('inf')
    if worst_face > _ROUND_OFF_TOLERANCE:
        violations.append(f'centre vs face work {face_errors}')
    return _result(violations, {'transpose_errors': transpose_errors, 'imbalances': imbalances,
                                'face_work_errors': face_errors}, measured=worst_imbalance,
                   tolerance=_IDENTITY_TOLERANCE)


def _check_kerr_linearization(results):
    from .media import bulk_medium
    from .mixing import ChannelSpec, build_mixing_table, resolve_channels
    from .sources import kerr_diagonal, polarization_density
    from .tensors import chi3_cubic, euler_rotation, rotate_tensor
    violations = []
    rng = np.random.default_rng(13)
    nx, ny = 6, 5
    mask = np.zeros((nx, ny), dtype=bool)
    mask[1:5, 1:4] = True
    tensor = rotate_tensor(chi3_cubic(1e-19 - 2e-20j, 1.27), euler_rotation(20.0, 30.0, 40.0))
    medium = bulk_medium('kerr', 3, tensor, mask, cell_scale=rng.uniform(1.0, 12.0, size=(nx, ny)), xp=np)
    channels, _ = resolve_channels((ChannelSpec('p', 'both', wavelength=1.55e-6), ChannelSpec('s', wavelength=1.3e-6)))
    table = build_mixing_table(channels, [medium.mixing_tensor()])
    fields = {}
    for index, channel in enumerate(channels):   # hz: E_y = 0, so P_x = M_xx E_x exactly (M_xy E_y drops)
        fields[index] = ((None, None, _random_complex(rng, (nx, ny))) if channel.polarization == 'ez'
                         else (_random_complex(rng, (nx, ny)), np.zeros((nx, ny), dtype=complex), None))
    errors = {}
    for target, channel in enumerate(channels):
        kerr = [term for term in table.terms_for(target) if term.kerr]
        if not kerr:
            violations.append(f'channel {channel.name}: no Kerr (SPM / XPM) term')
            continue
        density = polarization_density(kerr, fields, {'kerr': medium}, channel.components, (nx, ny), xp=np)
        delta = kerr_diagonal(kerr, target, fields, {'kerr': medium}, channel.components, medium.box, xp=np)
        component = channel.components[0]
        expected = EPS0 * delta[component] * fields[target][component][medium.box]
        measured = density[component][medium.box]
        errors[channel.name] = float(np.max(np.abs(measured - expected)) / np.max(np.abs(measured)))
    worst = max(errors.values()) if errors else math.inf
    if worst > _ROUND_OFF_TOLERANCE:
        violations.append(f'P_kerr != eps0 M_aa E_a: {errors}')
    return _result(violations, {'relative_errors': errors}, measured=worst, tolerance=_ROUND_OFF_TOLERANCE)


_TANGENT_TOLERANCE = 1e-8   # relative: round-off of the 4-point formula (exact for degree <= 4; carriers degree 5)


class _TangentStore:
    """Minimal ChannelFieldStore stand-in for FreeCarriers on the CPU (channels, versions, key)."""

    def __init__(self, channels):
        self.channels = tuple(channels)
        self.versions = [1] * len(self.channels)

    def key(self, indices):
        return tuple(self.versions[index] for index in indices)


def _centre_fields(rng, channels, shape):
    """Random complex centre vectors (E_x, E_y, E_z) per channel index (None for absent components)."""
    return {index: ((None, None, _random_complex(rng, shape)) if channel.polarization == 'ez'
                    else (_random_complex(rng, shape), _random_complex(rng, shape), None))
            for index, channel in enumerate(channels)}


def _shifted(fields, tangents, step):
    return {index: tuple(None if field is None else field + step * tangent
                         for field, tangent in zip(fields[index], tangents[index])) for index in fields}


def _finite_derivative(evaluate, fields, tangents, step):
    """{component: d/dt evaluate(E + t V) at 0} by the 4-point central formula (exact for polynomials of degree 4)."""
    values = {offset: evaluate(_shifted(fields, tangents, offset * step)) for offset in (-2, -1, 1, 2)}
    return {component: (8.0 * (values[1][component] - values[-1][component])
                        - (values[2][component] - values[-2][component])) / (12.0 * step)
            for component in values[1]}


def _worst_relative(measured, expected):
    scale = max(float(np.max(np.abs(values))) for values in expected.values())
    return max(float(np.max(np.abs(measured[component] - expected[component]))) for component in expected) / scale


def _check_newton_tangent(results):
    from .adjoint import carrier_cotangents
    from .media import CarrierModel, bulk_medium
    from .mixing import ChannelSpec, build_mixing_table, resolve_channels
    from .sources import FreeCarriers, polarization_density, polarization_tangent
    from .tensors import chi2_from_voigt, chi3_cubic, chi3_isotropic, euler_rotation, rotate_tensor, voigt_matrix
    violations, errors = [], {}
    rng = np.random.default_rng(29)
    shape = (6, 5)
    mask = np.zeros(shape, dtype=bool)
    mask[1:5, 1:4] = True
    # mixing terms: rotated poled -43m chi2 + rotated complex cubic chi3, mixed polarisations
    chi2 = rotate_tensor(chi2_from_voigt(voigt_matrix('-43m', {'d14': _GAAS_D14})), euler_rotation(30.0, 20.0, 10.0))
    chi3 = rotate_tensor(chi3_cubic(1e-19 - 3e-20j, 1.27), euler_rotation(20.0, 30.0, 40.0))
    media = {'chi2': bulk_medium('chi2', 2, chi2, mask, poling=rng.uniform(-1.0, 1.0, shape), xp=np),
             'chi3': bulk_medium('chi3', 3, chi3, mask, cell_scale=rng.uniform(1.0, 12.0, shape), xp=np)}
    channels, _ = resolve_channels((ChannelSpec('p', 'both', wavelength=1.55e-6), ChannelSpec('s', wavelength=2.0e-6),
                                    ChannelSpec('sh', combination='2 p'), ChannelSpec('d', 'hz', combination='p - s')))
    table = build_mixing_table(channels, [medium.mixing_tensor() for medium in media.values()])
    fields, tangents = _centre_fields(rng, channels, shape), _centre_fields(rng, channels, shape)
    for target, channel in enumerate(channels):
        terms = table.terms_for(target)
        if not terms:
            violations.append(f'channel {channel.name}: no mixing term')
            continue
        exact = polarization_tangent(terms, fields, tangents, media, channel.components, shape, xp=np)
        numeric = _finite_derivative(lambda shifted: polarization_density(terms, shifted, media, channel.components,
                                                                           shape, xp=np), fields, tangents, 1.0)
        errors[f'mixing {channel.name}'] = _worst_relative(exact, numeric)
    # free carriers of an isotropic TPA medium (G > 0 in every medium cell: no kink of max(G, 0))
    carrier_model = CarrierModel(lifetime=1e-9, fcd_coefficient=1.35e-27, fca_cross_section=1.45e-21,
                                 reference_omega=2.0 * math.pi * C0 / 1.55e-6)
    tpa = bulk_medium('tpa', 3, chi3_isotropic(2e-19 - 1e-19j), mask, cell_scale=rng.uniform(4.0, 12.0, shape),
                      xp=np, carriers=carrier_model, carrier_index_squared=rng.uniform(4.0, 12.0, shape))
    kerr_channels, _ = resolve_channels((ChannelSpec('p', wavelength=1.55e-6), ChannelSpec('s', 'hz',
                                                                                          wavelength=1.3e-6)))
    kerr_terms = tuple(term for term in build_mixing_table(kerr_channels, [tpa.mixing_tensor()]).terms if term.kerr)
    store = _TangentStore(kerr_channels)
    fields, tangents = _centre_fields(rng, kerr_channels, shape), _centre_fields(rng, kerr_channels, shape)
    carriers = FreeCarriers(kerr_terms, store, [tpa], xp=np)
    carrier_tangent = carriers.carrier_tangent(fields, tangents)
    for target, channel in enumerate(kerr_channels):
        def carrier_density(shifted, target=target, channel=channel):
            density = {component: np.zeros(shape, dtype=complex) for component in channel.components}
            FreeCarriers(kerr_terms, store, [tpa], xp=np).add_density(target, shifted, density)
            return density
        exact = {component: np.zeros(shape, dtype=complex) for component in channel.components}
        carriers.add_tangent(target, fields, tangents, carrier_tangent, exact)
        numeric = _finite_derivative(carrier_density, fields, tangents, 1e-3)
        errors[f'carriers {channel.name}'] = _worst_relative(exact, numeric)
        # (c) the adjoint transpose: 2 Re(c . dP[V]) = 2 Re(sum_m g_m . V_m), g = carrier_cotangents(c)
        cotangent = {component: _random_complex(rng, shape) for component in channel.components}
        transposed = carrier_cotangents(carriers, target, channel.omega, fields, cotangent, {'tpa': tpa}, shape, np,
                                        {})
        products = [cotangent[component] * exact[component] for component in channel.components]
        forward = sum(np.sum(product) for product in products).real
        backward = sum(np.sum(parts[component] * tangents[index][component]) for index, parts in transposed.items()
                       for component in range(3) if parts[component] is not None).real
        scale = sum(float(np.sum(np.abs(product))) for product in products)
        errors[f'carrier transpose {channel.name}'] = abs(forward - backward) / scale
    peak = float(np.max(carriers._state['tpa'][0]))
    if not peak > 0.0:
        violations.append(f'no carriers generated (peak N {peak:g})')
    worst = max(errors.values()) if errors else math.inf
    if worst > _TANGENT_TOLERANCE:
        violations.append(f'tangent != finite difference: {errors}')
    return _result(violations, {'relative_errors': errors, 'peak_carrier_density': peak}, measured=worst,
                   tolerance=_TANGENT_TOLERANCE)


def _check_bloch_parity(results):
    from ...core.mirror import field_plane_signs, plane_signs_text
    from ..em.boundaries import BCType, BoundaryConfig
    from ..em.harmonic import face_mean_at_centres
    from .adjoint import collocation_transpose, distribution_transpose
    from .mixing import ChannelSpec, MediumTensor, build_mixing_table, inexact_term_issues, resolve_channels
    from .solver import channel_boundary_config, channel_plane_signs
    from .sources import centres_to_faces
    from .tensors import chi2_from_voigt, voigt_matrix
    from types import SimpleNamespace
    violations, errors = [], {}
    rng = np.random.default_rng(31)
    nx, ny = 7, 6
    widths = (rng.uniform(0.5, 1.5, size=nx) * 2e-8, rng.uniform(0.5, 1.5, size=ny) * 2e-8)   # graded
    cell_area = np.outer(*widths)

    # 1. Bloch C / C*: <C_w f, p>_A = <f, C*_w p>_F (sesquilinear), and the plain transposes of the adjoint
    for axis in (0, 1):
        wrap = complex(np.exp(1j * rng.uniform(0.3, 2.8)))
        dual = 0.5 * (widths[axis] + np.roll(widths[axis], -1))
        face_area = np.outer(dual, widths[1]) if axis == 0 else np.outer(widths[0], dual)
        faces, centres = _random_complex(rng, (nx, ny)), _random_complex(rng, (nx, ny))
        collocated = face_mean_at_centres(faces, axis, True, xp=np, wrap_factor=wrap)
        distributed = centres_to_faces(centres, axis, True, widths[axis], xp=np, wrap_factor=wrap)
        scale = float(np.sum(np.abs(cell_area * collocated * centres)))
        errors[f'adjoint axis {axis}'] = abs(np.sum(cell_area * np.conj(collocated) * centres)
                                             - np.sum(face_area * np.conj(faces) * distributed)) / scale
        # plain (unweighted) sums: their own scales, not the area-weighted one (~1e-16 m^2 per cell)
        errors[f'C^T axis {axis}'] = abs(np.sum(collocated * centres) - np.sum(
            faces * collocation_transpose(centres, axis, True, wrap))) / float(np.sum(np.abs(collocated * centres)))
        errors[f'C*^T axis {axis}'] = abs(np.sum(distributed * faces) - np.sum(
            centres * distribution_transpose(faces, axis, True, widths[axis], wrap))) / float(
            np.sum(np.abs(distributed * faces)))
    # sign convention: hi-face samples of F = g(x) e^{-j k x} (g periodic) collocate at cell 0 to the mean of F at its
    # hi face and at x = 0 (the wrapped last face, one period down)
    edges = np.concatenate(([0.0], np.cumsum(widths[0])))
    period, wavenumber = float(edges[-1]), 0.7 * 2.0 * math.pi / float(edges[-1])

    def bloch_samples(x):
        return (1.0 + 0.3 * np.cos(2.0 * math.pi * x / period)) * np.exp(-1j * wavenumber * x)

    face_values = np.repeat(bloch_samples(edges[1:])[:, None], ny, axis=1)
    centre = face_mean_at_centres(face_values, 0, True, xp=np, wrap_factor=complex(np.exp(1j * wavenumber * period)))
    expected = 0.5 * (bloch_samples(edges[1]) + bloch_samples(0.0))
    errors['wrap sign'] = float(np.max(np.abs(centre[0] - expected)))
    worst = max(errors.values())
    if worst > _ROUND_OFF_TOLERANCE:
        violations.append(f'Bloch collocation: {errors}')

    # 2. channel Bloch k = the combination of the inputs; parity exponent = sum of coefficients mod 2
    k_p, k_s = (1.1e6, 0.4e6), (0.5e6, -0.2e6)
    channels, _ = resolve_channels((ChannelSpec('p', wavelength=1.55e-6, bloch_k=k_p),
                                    ChannelSpec('s', 'hz', wavelength=2.0e-6, bloch_k=k_s),
                                    ChannelSpec('sh', combination='2 p'), ChannelSpec('i', combination='p - s'),
                                    ChannelSpec('th', combination='3 p')))
    by_name = {channel.name: channel for channel in channels}
    expected_k = {'p': k_p, 's': k_s, 'sh': (2 * k_p[0], 2 * k_p[1]), 'i': (k_p[0] - k_s[0], k_p[1] - k_s[1]),
                  'th': (3 * k_p[0], 3 * k_p[1])}
    expected_parity = {'p': 1, 's': 1, 'sh': 0, 'i': 0, 'th': 1}
    for name, channel in by_name.items():
        if max(abs(a - b) for a, b in zip(channel.bloch_k, expected_k[name])) > 1e-6 \
                or channel.parity_exponent != expected_parity[name]:
            violations.append(f'channel {name}: bloch_k {channel.bloch_k}, parity {channel.parity_exponent}; expected '
                              f'{expected_k[name]}, {expected_parity[name]}')
    try:
        resolve_channels((ChannelSpec('p', wavelength=1.55e-6), ChannelSpec('sh', combination='2 p',
                                                                            bloch_k=(1.0, 0.0))))
        violations.append('a generated channel with its own bloch_k did not raise')
    except ValueError:
        pass

    # 3. separately given inputs: the SH input must carry 2 k_p (modulo 2 pi / L) and, under an antisymmetric face, the
    # even parity of the process (an input is odd there: always a mismatch); two inexact terms each, SHG p + p -> h and
    # the back-action DFG h - p -> p
    d33 = MediumTensor.from_tensor('d33', 2, chi2_from_voigt(voigt_matrix('general', {'d33': 1e-11})))
    length = 1.0e-6
    counts = {}
    for label, k_h, odd in (('matched', 2 * k_p[0], False), ('lattice shift', 2 * k_p[0] + 2 * math.pi / length,
                                                              False),
                            ('mismatched', k_p[0], False), ('odd mirror', 2 * k_p[0], True)):
        pair, _ = resolve_channels((ChannelSpec('p', wavelength=1.55e-6, bloch_k=(k_p[0], 0.0)),
                                    ChannelSpec('h', wavelength=0.775e-6, bloch_k=(k_h, 0.0))))
        counts[label] = len(inexact_term_issues(build_mixing_table(pair, [d33]), (length, None), odd))
    expected_counts = {'matched': 0, 'lattice shift': 0, 'mismatched': 2, 'odd mirror': 2}
    if counts != expected_counts:
        violations.append(f'inexact term issues {counts}, expected {expected_counts}')

    # 4. channel boundary configs: x Bloch per channel, an even channel turns the antisymmetric y_lo symmetric; the
    # FieldInfo plane_signs replace the physics face signs when unfolding
    base = BoundaryConfig('periodic', 'periodic', 'antisymmetric', 'cpml')
    configs = {name: channel_boundary_config(base, by_name[name]) for name in ('p', 'sh')}
    checks = {
        'p bloch x': configs['p'].bloch_x and configs['p'].bloch_kx == k_p[0] and not configs['p'].bloch_y,
        'p y_lo antisymmetric': configs['p'].y_lo == BCType.ANTISYMMETRIC,
        'sh bloch x 2 k': configs['sh'].bloch_x and abs(configs['sh'].bloch_kx - 2 * k_p[0]) < 1e-6,
        'sh y_lo symmetric': configs['sh'].y_lo == BCType.SYMMETRIC,
        'padding unchanged': configs['sh'].npml_sides == base.npml_sides,
        'base untouched': base.x_lo == BCType.PERIODIC and base.y_lo == BCType.ANTISYMMETRIC,
        'plane signs': (channel_plane_signs(base, by_name['p']) == {'y_lo': -1}
                        and channel_plane_signs(base, by_name['sh']) == {'y_lo': 1}),
        'unfold signs': field_plane_signs({'y_lo': -1, 'x_hi': 1}, SimpleNamespace(
            plane_signs=plane_signs_text({'y_lo': 1}))) == {'y_lo': 1, 'x_hi': 1},
    }
    violations.extend(f'channel config check failed: {name}' for name, passed in checks.items() if not passed)
    return _result(violations, {'collocation_errors': errors, 'inexact_counts': counts,
                                'config_checks': {name: bool(passed) for name, passed in checks.items()}},
                   measured=worst, tolerance=_ROUND_OFF_TOLERANCE)


NLFD_CPU_SELFTESTS = (
    SelfTestCase(
        name='nlfd_mixing_degeneracy',
        label='Nonlinear FDFD: mixing table and degeneracy factors',
        description='CPU check of physics/nlfd/mixing.py (no model run): the combination parser and channel '
                    'resolution (cycles, unknown names, frequency <= 0, duplicate frequency refused; both -> two '
                    'channels; ranks), the chi2 rows SHG 1/2, SFG 1, DFG 1 and the SHG / DFG back-action terms, the '
                    'undepleted rank filter (only rank-0 sources feed rank-1 targets), the chi3 rows THG 1/4, SPM 3/4, '
                    'XPM 6/4 (Kerr-tagged) and degenerate FWM 3/4, dropped zero contractions (hz channels in a '
                    'd33-only z-cut medium), separately given commensurate wavelengths, and the mixed-polarisation '
                    '-43m case (hz pump, SH only in the ez channel with P_z = 2 eps0 d14 E_x E_y). error = failed '
                    'checks; tolerance 0.',
        specs=lambda: {},
        check=_check_mixing_degeneracy,
    ),
    SelfTestCase(
        name='nlfd_tensor_rotation',
        label='Nonlinear FDFD: crystal tensors and rotations',
        description='CPU check of physics/nlfd/tensors.py: 3m Voigt d -> chi2 = 2 d entries, Kleinman symmetry of '
                    'the class forms (3m, -43m, 6mm, mm2, 32 with Kleinman-consistent values; a random general d is '
                    'detected as asymmetric), z-x-z Euler rotations orthonormal with a round trip, 90 degree '
                    'rotations and the x / y / z cut presets equal to signed index permutations, isotropic chi3 '
                    'invariant under a random rotation, cubic chi3 (rho 1.27) invariant under 90 but not 45 degrees '
                    'about z, and the n2 / beta_TPA -> chi3 -> n2 / beta round trip with Im chi3 < 0. error = failed '
                    'checks; tolerance 0.',
        specs=lambda: {},
        check=_check_tensor_rotation,
    ),
    SelfTestCase(
        name='nlfd_collocation_transpose',
        label='Nonlinear FDFD: collocation adjoint and power exchange',
        description='CPU check of physics/nlfd/sources.py on a graded 7 x 6 grid: C* (centres_to_faces) is the adjoint '
                    'of the face-to-centre mean C (harmonic.face_mean_at_centres) in the area-weighted inner products '
                    '(both axes, periodic or not), and on random complex fields of every channel the power '
                    'delivered by the nonlinear polarisation of every mixing-table term sums to zero (|sum W_k| / '
                    'sum |W_k|) for SHG (rotated poled 3m), SFG + DFG (rotated -43m + 3m, mixed polarisations) and '
                    'THG + SPM (isotropic real chi3), with the centre work equal to the face work of the '
                    'C*-distributed polarisation on hz channels. measured = worst imbalance; error = measured + failed '
                    f'checks; tolerance {_IDENTITY_TOLERANCE}.',
        specs=lambda: {},
        check=_check_collocation_transpose,
    ),
    SelfTestCase(
        name='nlfd_kerr_linearization',
        label='Nonlinear FDFD: Kerr operator linearisation',
        description='CPU check of physics/nlfd/sources.kerr_diagonal (the pump-depleted Kerr operator term): for the '
                    'SPM / XPM terms of every channel (p both polarisations, s E_z) of a rotated complex cubic chi3 '
                    '(rho 1.27) with a random per-cell n0^2 factor, on random fields (hz channels with E_y = 0, so the '
                    'off-diagonal M_xy drops), the rhs-form polarisation equals eps0 M_aa E_a of the diagonal '
                    'permittivity change cell by cell. measured = worst relative deviation; error = measured + failed '
                    f'checks; tolerance {_ROUND_OFF_TOLERANCE}.',
        specs=lambda: {},
        check=_check_kerr_linearization,
    ),
    SelfTestCase(
        name='nlfd_newton_tangent',
        label='Nonlinear FDFD: Newton-Krylov Jacobian products',
        description='CPU check of the exact Jacobian-vector products of physics/nlfd/sources.py (the pump-depleted '
                    'Newton-Krylov solve): (a) polarization_tangent of every mixing-table term of a rotated poled -43m '
                    'chi2 and a rotated complex cubic chi3 medium on channels p (both polarisations), s (E_z, 2 um), '
                    'sh = 2 p and d = p - s (H_z), and (b) the free-carrier part (FreeCarriers.carrier_tangent + '
                    'add_tangent) of an isotropic two-photon-absorbing medium with a random per-cell n0 on an E_z and '
                    'an H_z channel, each against the 4-point central difference of the polarisation along random '
                    'complex tangent fields (conjugated slots take conj V: a real-direction derivative), and (c) the '
                    'adjoint transpose adjoint.carrier_cotangents: Re(c . dP[V]) = Re(sum_m g_m . V_m) for a random '
                    'cotangent c. measured = worst relative deviation; error = measured + failed checks; tolerance '
                    f'{_TANGENT_TOLERANCE:g}.',
        specs=lambda: {},
        check=_check_newton_tangent,
    ),
    SelfTestCase(
        name='nlfd_bloch_parity',
        label='Nonlinear FDFD: Bloch wrap factors and channel mirror parity',
        description='CPU check (symmetry 2.2 / 2.5) on a graded 7 x 6 grid: (1) with a Bloch wrap factor w = e^{j k L} '
                    'on a periodic axis, C* (centres_to_faces, conj(w) on the last face) stays the adjoint of C '
                    '(face_mean_at_centres, w on the wrapped lo face of cell 0) in the area-weighted inner products, '
                    'adjoint.collocation_transpose / distribution_transpose are the plain transposes of C / C*, and C '
                    'of the face samples of a Bloch function g(x) e^{-j k x} at cell 0 is the mean of its value on '
                    'the hi face and at x = 0 (the sign convention); (2) resolve_channels gives each generated '
                    "channel the combination of its inputs' Bloch k (2 p, p - s, 3 p) and the parity exponent sum of "
                    'coefficients mod 2, and refuses a bloch_k on a generated channel; (3) mixing.inexact_term_issues '
                    'flags a separately given SH input whose k misses 2 k_p modulo 2 pi / L, or (antisymmetric face) '
                    'whose parity misses the process, for both inexact terms (SHG and its DFG back-action), and not a '
                    'k shifted by a reciprocal lattice vector; (4) solver.channel_boundary_config: bloch faces with '
                    'the channel k, an even channel (2 p) solves an antisymmetric face as symmetric, padding and the '
                    'shared config unchanged, channel_plane_signs and the FieldInfo plane_signs override of '
                    'core.mirror. measured = the worst collocation round-off; error = measured + failed checks; '
                    f'tolerance {_ROUND_OFF_TOLERANCE}.',
        specs=lambda: {},
        check=_check_bloch_parity,
    ),
)


# ---- GPU model cases 4, 12: undepleted SHG in a y-periodic strip ----
# Strip of _STRIP_ROWS rows, periodic in y, CPML along x, uniform 10 nm mesh, window cell k spanning [k h, (k + 1) h],
# one medium in every cell (PML included): every field is y-uniform and the solve is the 1D 3-point problem
# (E_{i+1} - 2 E_i + E_{i-1}) / h^2 + k^2 E_i = s_i (harmonic.py header; ez rhs s = -omega^2 mu0 P_z, P per volume).
# Its lattice Green's function: G_n = j h^2 / (2 sin(K h)) e^{-j K h |n|}, sin(K h / 2) = k h / 2 (the outgoing
# discrete wave). Downstream of the pump line source (cell 100) the pump is the discrete plane wave a e^{-j K_w h (m -
# m_a)} (hz: E_x / E_y collocated at the centres by C, the same mean the probe and the nonlinear source take), with a
# = the probe in the first slab cell m_a = 200 (the last slab cell checks it). The chi2 slab fills cells [200, 200 + N)
# over the full height; SH probes in cell 50 (backward, beyond the source) and 200 + N + 100.
_PHYSICS = 'nl'
_STRIP_CELL = 10e-9                 # m
_STRIP_ROWS = 4
_PUMP_WAVELENGTH = 1.55e-6          # m
_PUMP_OMEGA = 2.0 * math.pi * C0 / _PUMP_WAVELENGTH
_PUMP_AMPLITUDE = 1e7               # V/m
_SOURCE_CELL = 100
_BACKWARD_CELL = 50
_SLAB_START = 200
_FORWARD_GAP = 100                  # forward probe cells after the slab; as many again to the x_hi window edge
_PROBE_ROW = 1
_UNIFORM_EPS_R = 2.25               # n = 1.5, non-dispersive
_SHG_SLAB_CELLS = 65                # (K_2w + 2 K_w) h 65 / 2 ~ 7.9 rad: backward SH near a maximum, ~13 % of forward
_LN_D33 = 25e-12                    # m/V
_SH_TOLERANCE = 2e-3                # relative complex phasor (PML floor not measured yet)
_MIXED_ROTATION = 45.0              # deg, -43m crystal axes about z: P_z = eps0 d14 (E_y^2 - E_x^2)


def _strip_centre(cell):
    return (cell + 0.5) * _STRIP_CELL


def _strip_point(name, cell):
    return GeometrySpec(name, 'point', {'x': _strip_centre(cell), 'y': _strip_centre(_PROBE_ROW)})


def _channel_probe(name, channel, component, point):
    return FeatureSpec(name, 'field_probe', [point], {'channel': channel, 'component': component})


def _uniform_medium():
    return MaterialSpec('medium', properties={'eps_r': _UNIFORM_EPS_R}, selection=['all'])


def _strip_spec(title, medium, slab_cells, chi2_values, pump_polarization='ez', sh_polarization='ez'):
    """y-periodic strip: pump channel p (line source in cell 100, probes in the first / last slab cell), SH channel
    sh = 2 p, the chi2 slab on cells [200, 200 + slab_cells), SH probes backward (cell 50) and forward."""
    slab_end = _SLAB_START + slab_cells
    size_y = _STRIP_ROWS * _STRIP_CELL
    pump_component = 'ez' if pump_polarization == 'ez' else 'ey'
    sh_ez_channel = 'sh' if sh_polarization == 'ez' else 'sh_ez'
    source_x = _strip_centre(_SOURCE_CELL)
    geometry = [GeometrySpec('source_line', 'line', {'x0': source_x, 'y0': 0.0, 'x1': source_x, 'y1': size_y}),
                GeometrySpec('chi2_slab', 'rectangle', {'x0': _SLAB_START * _STRIP_CELL, 'y0': 0.0,
                                                         'x1': slab_end * _STRIP_CELL, 'y1': size_y}),
                _strip_point('pump_start_point', _SLAB_START), _strip_point('pump_end_point', slab_end - 1),
                _strip_point('forward_point', slab_end + _FORWARD_GAP), _strip_point('backward_point', _BACKWARD_CELL)]
    features = [
        FeatureSpec('p', 'harmonic_channel', [], {'definition': 'wavelength', 'wavelength': _PUMP_WAVELENGTH,
                                                  'polarization': pump_polarization}),
        FeatureSpec('sh', 'harmonic_channel', [], {'definition': 'combination', 'combination': '2 p',
                                                   'polarization': sh_polarization}),
        FeatureSpec('chi2', 'chi2_medium', ['chi2_slab'], dict(chi2_values)),
        FeatureSpec('pump', 'line_source', ['source_line'], {'channel': 'p', 'amplitude': _PUMP_AMPLITUDE,
                                                             'phase': 0.0}),
        _channel_probe('pump_start', 'p', pump_component, 'pump_start_point'),
        _channel_probe('pump_end', 'p', pump_component, 'pump_end_point'),
        _channel_probe('sh_forward', sh_ez_channel, 'ez', 'forward_point'),
        _channel_probe('sh_backward', sh_ez_channel, 'ez', 'backward_point')]
    if pump_polarization == 'hz':
        features.append(_channel_probe('pump_ex', 'p', 'ex', 'pump_start_point'))
    if sh_polarization == 'both':
        features.append(_channel_probe('sh_hz_forward', 'sh_hz', 'ey', 'forward_point'))
    return ModelSpec(
        title=f'Self-test: em_fdfd_nonlinear {title}',
        domain=DomainSpec(size_x=(slab_end + 2 * _FORWARD_GAP) * _STRIP_CELL, size_y=size_y),
        geometry=geometry, materials=[medium],
        physics=[PhysicsSpec(_PHYSICS, 'em_fdfd_nonlinear', settings=dict(PERIODIC_Y), features=features)],
        mesh=MeshSpec(mode='uniform', dx=_STRIP_CELL),
        study=StudySpec(type='nonlinear_frequency_domain'))


def _cell_wavenumber(omega, eps):
    """K h of the lossless 3-point lattice wave: sin(K h / 2) = k h / 2, k = omega sqrt(eps) / C0."""
    return 2.0 * math.asin(0.5 * omega * math.sqrt(eps) * _STRIP_CELL / C0)


def _shg_uniform_specs():
    return {'run': _strip_spec('undepleted SHG in a uniform medium', _uniform_medium(), _SHG_SLAB_CELLS,
                               {'crystal_class': '3m', 'd33': _LN_D33, 'orientation': 'z_cut'})}


def _mixed_polarization_specs():
    return {'run': _strip_spec('mixed-polarisation SHG (-43m, hz pump)', _uniform_medium(), _SHG_SLAB_CELLS,
                               {'crystal_class': '-43m', 'd14': _GAAS_D14, 'orientation': 'euler',
                                'euler_alpha': _MIXED_ROTATION}, pump_polarization='hz', sh_polarization='both')}


def _pair(value):
    return [float(value.real), float(value.imag)]


def _lattice_sh(probe_cell, signs, start_product, pump_k, sh_k, harmonic=2):
    """E_nw [V/m] in probe_cell = sum_m G(probe - m) s_m, s_m = -(n w)^2 mu0 eps0 sign_m start_product e^{-n j K_w h
    (m - m_a)}; start_product = P_z / eps0 [V/m] in the first slab cell, n = harmonic."""
    cells = _SLAB_START + np.arange(signs.size)
    source = (-(harmonic * _PUMP_OMEGA) ** 2 * MU0 * EPS0 * start_product) * signs * np.exp(
        -1j * harmonic * pump_k * (cells - _SLAB_START))
    green = (1j * _STRIP_CELL ** 2 / (2.0 * math.sin(sh_k))) * np.exp(-1j * sh_k * np.abs(probe_cell - cells))
    return complex(np.sum(green * source))


def _strip_errors(results, signs, eps_pump, eps_sh, start_product, label, backward=True, harmonic=2):
    """({item: relative error}, details, {probe: (measured, expected)}) of one strip run: the pump plane wave across the
    slab and the harmonic (sh_*) probes vs the lattice sum."""
    pump_k, sh_k = _cell_wavenumber(_PUMP_OMEGA, eps_pump), _cell_wavenumber(harmonic * _PUMP_OMEGA, eps_sh)
    pump_start, pump_end = probe_phasor(results, _PHYSICS, 'pump_start'), probe_phasor(results, _PHYSICS, 'pump_end')
    predicted_end = pump_start * cmath.exp(-1j * pump_k * (signs.size - 1))
    errors = {f'{label} pump plane wave': abs(pump_end - predicted_end) / abs(pump_start) if pump_start else math.inf}
    details = {f'{label}_pump_start': _pair(pump_start), f'{label}_pump_end': _pair(pump_end),
               f'{label}_pump_end_predicted': _pair(predicted_end), f'{label}_pump_Kh': pump_k, f'{label}_sh_Kh': sh_k}
    probe_cells = {'sh_forward': _SLAB_START + signs.size + _FORWARD_GAP}
    if backward:
        probe_cells['sh_backward'] = _BACKWARD_CELL
    fields = {}
    for name, cell in probe_cells.items():
        measured = probe_phasor(results, _PHYSICS, name)
        expected = _lattice_sh(cell, signs, start_product, pump_k, sh_k, harmonic)
        fields[name] = (measured, expected)
        errors[f'{label} {name}'] = relative_error(measured, expected)
        details.update({f'{label}_{name}_measured': _pair(measured), f'{label}_{name}_expected': _pair(expected)})
    return errors, details, fields


def _normalized_result(errors, tolerances, details):
    """Self-test dict: measured = error = the worst error / tolerance (NaN counts as the worst), tolerance 1."""
    normalized = {item: error / tolerances.get(item, _SH_TOLERANCE) for item, error in errors.items()}
    return normalized_result(normalized, details, relative_errors=errors, normalized_errors=normalized)


def _check_shg_uniform(results_by_name):
    results = results_by_name['run']
    start_product = _LN_D33 * probe_phasor(results, _PHYSICS, 'pump_start') ** 2   # P_z / eps0 = d33 E_z^2
    errors, details, _ = _strip_errors(results, np.ones(_SHG_SLAB_CELLS), _UNIFORM_EPS_R, _UNIFORM_EPS_R,
                                       start_product, 'uniform')
    return _normalized_result(errors, {}, details)


def _check_mixed_polarization(results_by_name):
    from .tensors import chi2_from_voigt, euler_rotation, rotate_tensor, voigt_matrix
    results = results_by_name['run']
    tensor = rotate_tensor(chi2_from_voigt(voigt_matrix('-43m', {'d14': _GAAS_D14})),
                           euler_rotation(_MIXED_ROTATION, 0.0, 0.0))
    pump_field = (probe_phasor(results, _PHYSICS, 'pump_ex'),   # (E_x, E_y) centres
                  probe_phasor(results, _PHYSICS, 'pump_start'))
    start_product = 0.5 * sum(tensor[2, j, k] * pump_field[j] * pump_field[k] for j in (0, 1) for k in (0, 1))
    errors, details, fields = _strip_errors(results, np.ones(_SHG_SLAB_CELLS), _UNIFORM_EPS_R, _UNIFORM_EPS_R,
                                            start_product, 'mixed')
    errors['pump E_x / E_y'] = abs(pump_field[0]) / abs(pump_field[1]) if pump_field[1] else math.inf
    sh_hz = probe_phasor(results, _PHYSICS, 'sh_hz_forward')
    errors['sh_hz / sh_ez'] = abs(sh_hz) / abs(fields['sh_forward'][1])
    details.update(pump_ex=_pair(pump_field[0]), sh_hz_forward=_pair(sh_hz),
                   d_eff_over_d14=float(0.5 * tensor[2, 1, 1] / _GAAS_D14))
    return _normalized_result(errors, {}, details)


# ---- symmetry 2.2: oblique pump on the strip (Bloch k_y, cpu backend) ----
# The case-4 strip with the pump channel at bloch_ky = 0.3 k0: the line source along the whole y period carries
# e^{-j k_y y}, every field is F(x) e^{-j k y} on the uniform rows (pump k_y, SH 2 k_y by its combination), and the
# y difference of e^{-j k y} is exactly -kappa^2, kappa = (2 / h) sin(k h / 2), so the x problem is the 3-point lattice
# with k^2 - kappa^2: _lattice_sh with eps_eff = eps - (kappa C0 / omega)^2 per channel. Probes share one row, so the
# row phase of the pump squared is the SH row phase.
_BLOCH_KY_FRACTION = 0.3   # bloch_ky = 0.3 k0 of the pump


def _shg_bloch_specs():
    spec = _strip_spec('undepleted SHG of an oblique pump (Bloch k_y)', _uniform_medium(), _SHG_SLAB_CELLS,
                       {'crystal_class': '3m', 'd33': _LN_D33, 'orientation': 'z_cut'})
    spec.physics[0].features[0].params['bloch_ky'] = _BLOCH_KY_FRACTION * _PUMP_OMEGA / C0   # channel p
    spec.solver_defaults.backend = 'cpu'
    return {'run': spec}


def _oblique_eps(omega, bloch_ky):
    """eps_eff of the x lattice: eps - (kappa C0 / omega)^2, kappa = (2 / h) sin(k_y h / 2)."""
    kappa = 2.0 / _STRIP_CELL * math.sin(0.5 * bloch_ky * _STRIP_CELL)
    return _UNIFORM_EPS_R - (kappa * C0 / omega) ** 2


def _check_shg_bloch(results_by_name):
    results = results_by_name['run']
    bloch_ky = _BLOCH_KY_FRACTION * _PUMP_OMEGA / C0
    start_product = _LN_D33 * probe_phasor(results, _PHYSICS, 'pump_start') ** 2
    errors, details, _ = _strip_errors(results, np.ones(_SHG_SLAB_CELLS), _oblique_eps(_PUMP_OMEGA, bloch_ky),
                                       _oblique_eps(2.0 * _PUMP_OMEGA, 2.0 * bloch_ky), start_product, 'oblique')
    return _normalized_result(errors, {}, details)


# ---- symmetry 2.5: channel mirror parity (cpu backend) ----
# The strip (y periodic, 10 nm, uniform eps 2.25) symmetric about the cell edge x = 1 um: full model 2 um wide (CPML
# both x ends) with an odd pump (line sources at 1 -+ 0.455 um, phases 180 / 0 deg) and a z-cut d33 slab |x - 1 um| <
# 0.3 um, against its half 1 um wide with an antisymmetric x_lo on the plane (one line source at 0.455 um, slab [0,
# 0.3) um). E_z of the pump is odd, its SH P_z = eps0 d33 E_z^2 even: the SH channel (2 p) must solve the plane as
# symmetric. The half is the exact restriction of the full discrete problem, so the probes agree to round-off; an SH
# channel kept antisymmetric would be wrong by O(1).
_MIRROR_PLANE_X = 1.0e-6        # m, full-model plane
_MIRROR_SOURCE_X = 0.455e-6     # m from the plane (cell 45 centre)
_MIRROR_SLAB = 0.3e-6           # m, slab half width
_MIRROR_PROBES = {'pump_probe': ('p', 0.205e-6), 'sh_inside': ('sh', 0.155e-6), 'sh_outside': ('sh', 0.705e-6)}
_MIRROR_TOLERANCE = 1e-5        # relative complex phasor (exact restriction)


def _mirror_strip_spec(half):
    """Full (plane at 1 um) or half (antisymmetric x_lo) strip of the channel-parity case on the cpu backend."""
    offset = 0.0 if half else _MIRROR_PLANE_X
    size_x = _MIRROR_PLANE_X if half else 2.0 * _MIRROR_PLANE_X
    size_y = _STRIP_ROWS * _STRIP_CELL
    probe_y = _strip_centre(_PROBE_ROW)
    geometry = [GeometrySpec('chi2_slab', 'rectangle', {'x0': 0.0 if half else offset - _MIRROR_SLAB, 'y0': 0.0,
                                                         'x1': offset + _MIRROR_SLAB, 'y1': size_y})]
    features = [
        FeatureSpec('p', 'harmonic_channel', [], {'definition': 'wavelength', 'wavelength': _PUMP_WAVELENGTH}),
        FeatureSpec('sh', 'harmonic_channel', [], {'definition': 'combination', 'combination': '2 p'}),
        FeatureSpec('chi2', 'chi2_medium', ['chi2_slab'], {'crystal_class': '3m', 'd33': _LN_D33,
                                                           'orientation': 'z_cut'})]
    sources = [('pump_plus', offset + _MIRROR_SOURCE_X, 0.0)]
    if not half:   # the odd image source
        sources.append(('pump_minus', offset - _MIRROR_SOURCE_X, 180.0))
    for name, x, phase in sources:
        geometry.append(GeometrySpec(f'{name}_line', 'line', {'x0': x, 'y0': 0.0, 'x1': x, 'y1': size_y}))
        features.append(FeatureSpec(name, 'line_source', [f'{name}_line'], {'channel': 'p',
                                                                            'amplitude': _PUMP_AMPLITUDE,
                                                                            'phase': phase}))
    for name, (channel, x) in _MIRROR_PROBES.items():
        geometry.append(GeometrySpec(f'{name}_point', 'point', {'x': offset + x, 'y': probe_y}))
        features.append(_channel_probe(name, channel, 'ez', f'{name}_point'))
    settings = dict(PERIODIC_Y)
    if half:
        settings['bc_x_lo'] = 'antisymmetric'
    spec = ModelSpec(
        title=f"Self-test: em_fdfd_nonlinear channel mirror parity ({'half, antisymmetric x_lo' if half else 'full'})",
        domain=DomainSpec(size_x=size_x, size_y=size_y), geometry=geometry, materials=[_uniform_medium()],
        physics=[PhysicsSpec(_PHYSICS, 'em_fdfd_nonlinear', settings=settings, features=features)],
        mesh=MeshSpec(mode='uniform', dx=_STRIP_CELL), study=StudySpec(type='nonlinear_frequency_domain'))
    spec.solver_defaults.backend = 'cpu'
    return spec


def _check_mirror_parity(results_by_name):
    """Half vs full probe phasors (pump odd, SH even); error = the largest relative deviation."""
    full, half = results_by_name['full'], results_by_name['half']
    errors, details = {}, {}
    for name in _MIRROR_PROBES:
        full_value, half_value = probe_phasor(full, _PHYSICS, name), probe_phasor(half, _PHYSICS, name)
        errors[name] = relative_error(half_value, full_value)
        details[name] = {'full': _pair(full_value), 'half': _pair(half_value)}
    return worst_result(errors, _MIRROR_TOLERANCE, **details)


# ---- GPU model cases 6, 8 (stages 2 and 3): depleted SHG, SPM on the same strip ----
# Coupled-mode references on the 3-point lattice: a channel field A e^{-j K h m} driven by a phase-matched P obeys
# A' = -j (omega^2 mu0 / (2 k_eff)) P_envelope, k_eff = sin(K h) / h (the lattice Green's function), and an hz channel,
# probed at the centres (A_c = cos(K h / 2) A_face) and fed through C*, picks up cos^2(K h / 2). Phase matching is exact
# on the lattice by birefringence: E_z channels see eps_zz (the eps_r array), H_z channels (E_y along x) eps_r -> eps_yy.
_D_BIREFRINGENT = 25e-12            # m/V: 'general' d23 = d34 (P_y = eps0 chi_yzz E_z E_z, P_z = eps0 chi_zyz E_y E_z)
_DEPLETED_SLAB_CELLS = 2000         # 20 um: Gamma / (2 K) ~ 5e-3 at 70 % conversion
_DEPLETED_TARGETS = {'high': 1.18568, 'low': 0.32745}   # Gamma L: tanh^2 = 0.70, 0.10
_DEPLETED_RUNS = {'high': ('high', 'picard'), 'low': ('low', 'picard'),   # run -> (conversion, iteration_method)
                  'high_newton': ('high', 'newton'), 'low_newton': ('low', 'newton')}
_ITERATION_METHODS = ('picard', 'newton')
_NEWTON_STEPS = 30                  # max_iterations of the Newton runs
_DEPLETED_TOLERANCE = 1e-2          # relative, coupled-mode accuracy (Gamma / K, backward waves); not yet measured
_SPM_CHI3 = 1e-19                   # m^2/V^2
_SPM_DELTA_EPS = 2e-3               # (3/4) chi3 |E|^2 in the slab (ripple ~1e-3: do not raise further at 2e-3 tol)
_SPM_SLAB_CELLS = 10000             # 100 um: nonlinear phase ~0.41 rad
_SPM_TOLERANCE = 2e-3               # relative nonlinear phase (Fabry-Perot ripple ~delta / 2)
_EZ_EPS = 2.25


def _channel_strip_spec(title, medium, slab_cells, nonlinear, channels, sources, probes, settings=None):
    """y-periodic strip as _strip_spec, general: nonlinear = (feature type, values) on cells [200, 200 + slab_cells);
    channels [(name, polarization, definition params)], sources [(name, channel, amplitude)] on the line in cell 100,
    probes [(name, channel, component, cell)]."""
    slab_end = _SLAB_START + slab_cells
    size_y = _STRIP_ROWS * _STRIP_CELL
    source_x = _strip_centre(_SOURCE_CELL)
    cells = sorted({cell for _, _, _, cell in probes})
    geometry = [GeometrySpec('source_line', 'line', {'x0': source_x, 'y0': 0.0, 'x1': source_x, 'y1': size_y}),
                GeometrySpec('nl_slab', 'rectangle', {'x0': _SLAB_START * _STRIP_CELL, 'y0': 0.0,
                                                       'x1': slab_end * _STRIP_CELL, 'y1': size_y}),
                *(_strip_point(f'point_{cell}', cell) for cell in cells)]
    features = [FeatureSpec(name, 'harmonic_channel', [], {'polarization': polarization, **definition})
                for name, polarization, definition in channels]
    features.append(FeatureSpec('nonlinear', nonlinear[0], ['nl_slab'], dict(nonlinear[1])))
    features += [FeatureSpec(name, 'line_source', ['source_line'], {'channel': channel, 'amplitude': amplitude,
                                                                    'phase': 0.0})
                 for name, channel, amplitude in sources]
    features += [_channel_probe(name, channel, component, f'point_{cell}') for name, channel, component, cell in probes]
    return ModelSpec(
        title=f'Self-test: em_fdfd_nonlinear {title}',
        domain=DomainSpec(size_x=(slab_end + 2 * _FORWARD_GAP) * _STRIP_CELL, size_y=size_y),
        geometry=geometry, materials=[medium],
        physics=[PhysicsSpec(_PHYSICS, 'em_fdfd_nonlinear', settings={**PERIODIC_Y, **(settings or {})},
                             features=features)],
        mesh=MeshSpec(mode='uniform', dx=_STRIP_CELL),
        study=StudySpec(type='nonlinear_frequency_domain'))


def _pump_probes(channel, component, slab_cells):
    """pump_start / pump_end (first / last slab cell) and pump_forward probes of channel `channel`."""
    slab_end = _SLAB_START + slab_cells
    return [('pump_start', channel, component, _SLAB_START), ('pump_end', channel, component, slab_end - 1),
            ('pump_forward', channel, component, slab_end + _FORWARD_GAP)]


def _harmonic_probes(channel, component, slab_cells, prefix='sh'):
    slab_end = _SLAB_START + slab_cells
    return [(f'{prefix}_start', channel, component, _SLAB_START),
            (f'{prefix}_forward', channel, component, slab_end + _FORWARD_GAP),
            (f'{prefix}_backward', channel, component, _BACKWARD_CELL)]


def _birefringent_eps(omega, wavenumber):
    """Relative eps whose 3-point lattice wave at omega has K h = wavenumber (sin(K h / 2) = k h / 2)."""
    return (2.0 * C0 * math.sin(0.5 * wavenumber) / (omega * _STRIP_CELL)) ** 2


def _k_eff(wavenumber):
    """sin(K h) / h [1/m] of a lattice wave with K h = wavenumber."""
    return math.sin(wavenumber) / _STRIP_CELL


def _general_d(d):
    return {'crystal_class': 'general', 'd23': d, 'd34': d, 'orientation': 'z_cut'}


def _depleted_design():
    """(eps_zz of the ez pump, eps_r = eps_yy of the hz SH, K_w h, K_2w h, kappa_pump, kappa_sh [1/(V)]) of the
    birefringently phase-matched SHG strip (K_2w = 2 K_w)."""
    pump_k = _cell_wavenumber(_PUMP_OMEGA, _EZ_EPS)
    sh_k = 2.0 * pump_k
    eps_sh = _birefringent_eps(2.0 * _PUMP_OMEGA, sh_k)
    kappa_pump = _PUMP_OMEGA ** 2 * _D_BIREFRINGENT / (C0 ** 2 * _k_eff(pump_k))   # P = eps0 2 d E_2w conj(E_w)
    kappa_sh = (math.cos(0.5 * sh_k) ** 2 * (2.0 * _PUMP_OMEGA) ** 2 * _D_BIREFRINGENT
                / (2.0 * C0 ** 2 * _k_eff(sh_k)))                                     # P = eps0 d E_w^2
    return eps_sh, pump_k, sh_k, kappa_pump, kappa_sh


def _depleted_specs():
    eps_sh, _, _, kappa_pump, kappa_sh = _depleted_design()
    length = _DEPLETED_SLAB_CELLS * _STRIP_CELL
    medium = MaterialSpec('medium', properties={'eps_r': eps_sh, 'eps_zz': _EZ_EPS}, selection=['all'])
    channels = [('p', 'ez', {'definition': 'wavelength', 'wavelength': _PUMP_WAVELENGTH}),
                ('sh', 'hz', {'definition': 'combination', 'combination': '2 p'})]
    probes = _pump_probes('p', 'ez', _DEPLETED_SLAB_CELLS) + _harmonic_probes('sh', 'ey', _DEPLETED_SLAB_CELLS)
    specs = {}
    for label, (level, method) in _DEPLETED_RUNS.items():
        amplitude = _DEPLETED_TARGETS[level] / (length * math.sqrt(kappa_pump * kappa_sh))
        specs[label] = _channel_strip_spec(
            f'pump-depleted SHG ({level} conversion, {method})', medium, _DEPLETED_SLAB_CELLS,
            ('chi2_medium', _general_d(_D_BIREFRINGENT)), channels, [('pump', 'p', amplitude)], probes,
            settings=_pump_depleted_settings(method, anderson_depth=5 if level == 'high' else 0, max_iterations=100))
    return specs


def _pump_depleted_settings(method, **picard):
    """pump_depleted settings of iteration_method `method` (picard: the given extra settings; newton: 30 steps)."""
    if method == 'newton':
        return {'coupling_mode': 'pump_depleted', 'iteration_method': 'newton', 'max_iterations': _NEWTON_STEPS}
    return {'coupling_mode': 'pump_depleted', 'iteration_method': 'picard', **picard}


def _solve_record(results):
    """summary nonlinear_solve record of the strip physics ({} when missing)."""
    return results.summary().get('nonlinear_solve', {}).get(_PHYSICS, {})


def _check_depleted(results_by_name):
    _, pump_k, sh_k, kappa_pump, kappa_sh = _depleted_design()
    length = _DEPLETED_SLAB_CELLS * _STRIP_CELL
    errors, details = {}, {'pump_Kh': pump_k, 'sh_Kh': sh_k, 'kappa_pump': kappa_pump, 'kappa_sh': kappa_sh}
    for label in _DEPLETED_RUNS:
        results = results_by_name[label]
        a0 = abs(probe_phasor(results, _PHYSICS, 'pump_start'))
        gamma_length = a0 * math.sqrt(kappa_pump * kappa_sh) * length
        pump_out = abs(probe_phasor(results, _PHYSICS, 'pump_forward')) / a0
        sh_out = abs(probe_phasor(results, _PHYSICS, 'sh_forward')) / (a0 * math.sqrt(kappa_sh / kappa_pump))
        errors[f'{label} pump sech'] = relative_error(pump_out, 1.0 / math.cosh(gamma_length))
        errors[f'{label} SH tanh'] = relative_error(sh_out, math.tanh(gamma_length))
        record = _solve_record(results)
        errors[f'{label} not converged'] = 0.0 if record.get('converged') else math.inf
        details.update({f'{label}_gamma_L': gamma_length, f'{label}_pump_transmission': pump_out ** 2,
                        f'{label}_sh_fraction': sh_out ** 2, f'{label}_tanh2': math.tanh(gamma_length) ** 2,
                        f'{label}_sweeps': record.get('sweeps'), f'{label}_residual': record.get('residual'),
                        f'{label}_gmres_iterations': record.get('gmres_iterations')})
    tolerances = {item: _DEPLETED_TOLERANCE for item in errors}
    return _normalized_result(errors, tolerances, details)


def _spm_amplitude():
    return math.sqrt(_SPM_DELTA_EPS / (0.75 * _SPM_CHI3))


def _spm_specs():
    medium = MaterialSpec('medium', properties={'eps_r': 1.0}, selection=['all'])
    channels = [('p', 'ez', {'definition': 'wavelength', 'wavelength': _PUMP_WAVELENGTH})]
    return {method: _channel_strip_spec(
        f'Kerr self-phase modulation (pump-depleted, {method})', medium, _SPM_SLAB_CELLS,
        ('chi3_medium', {'tensor': 'isotropic', 'strength': 'chi3', 'chi3_real': _SPM_CHI3}), channels,
        [('pump', 'p', _spm_amplitude())], _pump_probes('p', 'ez', _SPM_SLAB_CELLS),
        settings=_pump_depleted_settings(method)) for method in _ITERATION_METHODS}


def _check_spm(results_by_name):
    errors, details, tolerances = {}, {}, {}
    cells = _SPM_SLAB_CELLS - 1
    for method in _ITERATION_METHODS:
        results = results_by_name[method]
        start, end = probe_phasor(results, _PHYSICS, 'pump_start'), probe_phasor(results, _PHYSICS, 'pump_end')
        delta = 0.75 * _SPM_CHI3 * abs(start) ** 2
        linear_k, kerr_k = _cell_wavenumber(_PUMP_OMEGA, 1.0), _cell_wavenumber(_PUMP_OMEGA, 1.0 + delta)
        nonlinear_phase = cells * (kerr_k - linear_k)
        measured = end / start
        phase_error = abs(cmath.phase(measured * cmath.exp(1j * cells * kerr_k)))
        record = _solve_record(results)
        errors.update({f'{method} nonlinear phase': phase_error / nonlinear_phase,
                       f'{method} transmission': abs(abs(measured) - 1.0),
                       f'{method} not converged': 0.0 if record.get('converged') else math.inf})
        tolerances.update({f'{method} nonlinear phase': _SPM_TOLERANCE, f'{method} transmission': _SPM_TOLERANCE,
                           f'{method} not converged': 1.0})
        details.update({f'{method}_delta_eps': delta, f'{method}_nonlinear_phase_expected': nonlinear_phase,
                        f'{method}_phase_error_rad': phase_error, f'{method}_end_over_start': _pair(measured),
                        f'{method}_sweeps': record.get('sweeps'), f'{method}_residual': record.get('residual'),
                        f'{method}_gmres_iterations': record.get('gmres_iterations')})
    return _normalized_result(errors, tolerances, details)


_STRIP_TEXT = ('y-periodic strip (4 rows, CPML along x, uniform 10 nm mesh, one medium in every cell incl. the PML, '
               'so the 2D solve is the 1D 3-point problem exactly), pump channel p at 1.55 um driven by a line source '
               f'({_PUMP_AMPLITUDE:g} V/m) in cell 100, SH channel sh = 2 p, a chi2 slab from cell 200 over the full '
               'height (undepleted cascade, nonlinear_frequency_domain study). Reference: the pump phasor a probed in '
               'the first slab cell, continued as the discrete plane wave a e^{-j K_w h (m - 200)} (sin(K h / 2) = k h '
               '/ 2; the probe in the last slab cell checks it), gives P_z(m) per slab cell, and the SH at a probe is '
               'the lattice Green function sum E_2w(i) = sum_m j h^2 / (2 sin(K_2w h)) e^{-j K_2w h |i - m|} (-(2 w)^2 '
               'mu0 P_z(m)) (exact for the discrete equations, so discrete phase mismatch and dispersion are '
               'included).')
_TOLERANCE_TEXT = (f'measured = error = the worst relative complex deviation / {_SH_TOLERANCE:g}, tolerance 1 (budget: '
                   'CPML reflections of pump and SH and solver round-off; the tolerance is to be set from the measured '
                   'PML floor, ~1e-4 expected).')

NLFD_GPU_SELFTESTS = (
    SelfTestCase(
        name='nlfd_shg_uniform_undepleted',
        label='Nonlinear FDFD: undepleted SHG vs the lattice Green function',
        description=f'A {_STRIP_TEXT} Uniform non-dispersive medium eps_r {_UNIFORM_EPS_R:g} (only lattice dispersion '
                    f'mismatches 2w and w), E_z channels, 3m z-cut d33 = {_LN_D33:g} m/V on {_SHG_SLAB_CELLS} cells '
                    '(P_z = eps0 d33 E_z^2): the forward (100 cells after the slab) and backward (cell 50) SH phasors '
                    'must equal the lattice sum (the backward wave, ~13 % of the forward one, pins the phase of the '
                    'source and of G), and the pump the discrete plane wave across the slab. '
                    f'{_TOLERANCE_TEXT} Pins the ez polarisation source -omega^2 mu0 P, the SHG factor 1/2 and the '
                    'channel cascade.',
        specs=_shg_uniform_specs,
        check=_check_shg_uniform),
    SelfTestCase(
        name='nlfd_mixed_polarization',
        label='Nonlinear FDFD: mixed-polarisation SHG (-43m, H_z pump)',
        description=f'The {_STRIP_TEXT} Uniform eps_r {_UNIFORM_EPS_R:g}; the pump channel p is H_z polarised (its '
                    'in-plane E_y collocated at the centres by C), the SH channel sh is polarization both, and the '
                    f'chi2 slab ({_SHG_SLAB_CELLS} cells) is -43m d14 = {_GAAS_D14:g} m/V with the crystal axes '
                    f'rotated {_MIXED_ROTATION:g} deg about z (euler_alpha), so P_z = eps0 d14 (E_y^2 - E_x^2) lands '
                    'in sh_ez only (no tensor entry couples in-plane E pairs to in-plane P: sh_hz has no source). '
                    'Checks: the sh_ez forward / backward phasors vs the lattice sum of P_z from the rotated tensor '
                    'and the probed pump (E_x, E_y), the pump plane wave, |E_x| / |E_y| of the pump and |sh_hz| / '
                    f'|sh_ez| (both ~0). {_TOLERANCE_TEXT} Pins the hz collocation C of the source fields and the '
                    'channel polarisation mixing.',
        specs=_mixed_polarization_specs,
        check=_check_mixed_polarization),
    SelfTestCase(
        name='nlfd_shg_depleted_tanh',
        label='Nonlinear FDFD: pump-depleted SHG (tanh^2)',
        description='y-periodic strip (as nlfd_shg_uniform_undepleted) with an E_z pump p (eps_zz 2.25) and an H_z '
                    'second harmonic sh = 2 p whose eps_yy (the eps_r property) makes the lattice wavenumbers exactly '
                    f'phase matched (K_2w = 2 K_w), general d23 = d34 = {_D_BIREFRINGENT:g} m/V over '
                    f'{_DEPLETED_SLAB_CELLS} cells, coupling_mode pump_depleted. Four runs, Gamma L = 1.186 (70 % '
                    'conversion; picard with Anderson depth 5, and newton) and 0.327 (10 %; plain Picard, and '
                    'newton): the forward pump |E| / a0 vs '
                    'sech(Gamma L) and the forward SH vs a0 sqrt(kappa_sh / kappa_p) tanh(Gamma L), Gamma = a0 '
                    'sqrt(kappa_p kappa_sh) from the lattice coupled-mode coefficients (k_eff = sin(K h) / h, the hz '
                    'centre factor cos^2(K h / 2)), a0 = the pump probed in the first slab cell; and convergence. '
                    f'measured = worst relative error / {_DEPLETED_TOLERANCE:g} (coupled-mode accuracy, not yet '
                    'measured). Pins the back-action term, the Gauss-Seidel / Anderson loop, the Newton-Krylov solve '
                    'and energy exchange.',
        specs=_depleted_specs,
        check=_check_depleted),
    SelfTestCase(
        name='nlfd_kerr_spm',
        label='Nonlinear FDFD: Kerr self-phase modulation',
        description=f'y-periodic vacuum strip, one E_z channel, isotropic chi3 {_SPM_CHI3:g} m^2/V^2 on '
                    f'{_SPM_SLAB_CELLS} cells with the amplitude giving delta eps = (3/4) chi3 |E|^2 = '
                    f'{_SPM_DELTA_EPS:g}, coupling_mode pump_depleted, two runs (picard: SPM through the lagged '
                    'operator term; newton: SPM in the rhs with the exact Jacobian): the '
                    'phase of E(last slab cell) / E(first) vs the lattice wave of eps 1 + delta (from the probed |E|), '
                    f'as a fraction of the nonlinear phase (~0.4 rad), tolerance {_SPM_TOLERANCE:g} (budget: the '
                    'slab-end Fabry-Perot ripple ~delta / 2); |ratio| = 1 within the same tolerance; convergence. '
                    'Slab 10000 cells at delta 2e-3 (was 20000 at 1e-3): not yet confirmed by a run.',
        specs=_spm_specs,
        check=_check_spm),
    SelfTestCase(   # cpu backend (the spec), listed with the strip cases
        name='nlfd_shg_bloch',
        label='Nonlinear FDFD: undepleted SHG of an oblique pump (Bloch k)',
        description=f'The {_STRIP_TEXT} on the cpu backend, as nlfd_shg_uniform_undepleted but the pump input '
                    f'channel at bloch_ky = {_BLOCH_KY_FRACTION:g} k0 (oblique incidence; the line source across the '
                    'period carries exp(-j k_y y)); the SH channel 2 p takes 2 k_y from its combination. Every field '
                    'is F(x) exp(-j k y) on the uniform rows, so the x problem is the 3-point lattice with k^2 - '
                    'kappa^2, kappa = (2 / h) sin(k_y h / 2) per channel: forward / backward SH phasors vs that '
                    f'lattice sum and the pump plane wave across the slab. {_TOLERANCE_TEXT} Pins the per-channel '
                    'Bloch k (a wrong SH k breaks the periodic source; the wrap phase of C / C* enters only hz '
                    'channels: nlfd_bloch_parity).',
        specs=_shg_bloch_specs,
        check=_check_shg_bloch),
    SelfTestCase(   # cpu backend (the specs)
        name='nlfd_mirror_parity',
        label='Nonlinear FDFD: channel mirror parity (half vs full)',
        description='The 10 nm y-periodic strip (uniform eps 2.25, E_z channels, z-cut d33 slab |x - 1 um| < 0.3 '
                    'um) on the cpu backend: full model 2 um wide with an odd pump (line sources at 1 -+ 0.455 um, '
                    'phases 180 / 0 deg) against the half 1 um wide with an antisymmetric x_lo mirror on the plane '
                    '(one source at 0.455 um). The pump is odd, its SH P_z = eps0 d33 E_z^2 even, so the SH channel '
                    '(2 p) must solve the plane as symmetric (channel_boundary_config). Expected: the half model is '
                    'the exact restriction of the full one, so the pump probe (0.205 um) and the SH probes in the '
                    'slab (0.155 um) and beyond the source (0.705 um, from the plane) agree. error = the largest '
                    f'relative complex deviation, tolerance {_MIRROR_TOLERANCE:g} (round-off; an SH channel left '
                    'antisymmetric is off by O(1)).',
        specs=lambda: {'full': _mirror_strip_spec(False), 'half': _mirror_strip_spec(True)},
        check=_check_mirror_parity),
)


# ---- templates ----
# ppln_slab_shg: a 0.6 um LN slab (E_z = extraordinary, z-cut) in silica along the whole 24 x 1.8 um box (CPML all
# round, uniform 15 nm mesh: 40 cells across the slab, ~245k unknowns per channel), mode source (1 kW/m) at x = 1 um,
# 20 um poled from x = 2 um, mode monitors 1 um after it. LN extraordinary as one lossless Lorentz pole at 200 nm
# fitted to the Zelmon 1997 n_e at 1550 nm (2.1376) and 775 nm (2.1784) exactly (eps_inf 1.2774, delta_eps 3.2370);
# silica non-dispersive (2.085). The CMT reference (template parameters dbeta, C_cmt): continuum symmetric-slab TE
# modes (selftest_helpers.analytic_slab_index), the along-x lattice correction K = (2 / h) asin(beta h / 2) on dbeta =
# K_2w - 2 K_w (the 15 nm mesh shifts the QPM period by ~3 %), first-order QPM (2 / pi), the overlap integral over the
# core, P_2w = C_cmt d33^2 P_w^2 L^2 sinc^2((dbeta - 2 pi / Lambda) L / 2).
PPLN_LABEL = 'PPLN slab waveguide SHG (QPM tuning curve)'
PPLN_DESCRIPTION = ('Nonlinear FDFD (undepleted): a 1550 nm mode pumps a periodically poled z-cut lithium niobate '
                    'slab in silica; sweep of the poling period over the QPM peak, forward second-harmonic mode power '
                    'vs the coupled-mode sinc^2 curve.')
_PPLN_CELL = 15e-9                  # m
_PPLN_HEIGHT = 1.8e-6               # m
_PPLN_CORE = (0.6e-6, 1.2e-6)       # m, y range of the LN slab
_PPLN_START = 2e-6                  # m, poled region start (x_pp)
_PPLN_LENGTH = 20e-6                # m, poled length (L_pp)
_PPLN_LN_EPS_INF, _PPLN_LN_DELTA_EPS, _PPLN_LN_RESONANCE = 1.277394, 3.23696, 0.2e-6   # Lorentz fit of n_e
_PPLN_SILICA_EPS = 2.085
_PPLN_D33 = 25e-12                  # m/V
_PPLN_POWER = 1e3                   # W/m
_PPLN_SWEEP_FRACTIONS = tuple(-0.3 + 0.075 * index for index in range(9))   # Lambda / Lambda_0 - 1


def _lorentz_eps(eps_inf, delta_eps, resonance, wavelength):
    """eps of a lossless Lorentz pole at vacuum wavelength [m]: eps_inf + delta_eps / (1 - (resonance / wavelength)^2)."""
    return eps_inf + delta_eps / (1.0 - (resonance / wavelength) ** 2)


def _slab_mode_profile(y, k0, n_eff, core_eps, cladding_eps, half_thickness):
    """Unnormalised even TE mode of the symmetric slab: cos(kappa y) inside, cos(kappa a) e^{-gamma (|y| - a)} out."""
    kappa = k0 * math.sqrt(core_eps - n_eff ** 2)
    gamma = k0 * math.sqrt(n_eff ** 2 - cladding_eps)
    outside = math.cos(kappa * half_thickness) * np.exp(-gamma * (np.abs(y) - half_thickness))
    return np.where(np.abs(y) <= half_thickness, np.cos(kappa * y), outside), gamma


def _ppln_design():
    """(dbeta [rad/m], C_cmt [W/m per (m/V)^2 (W/m)^2 m^2], Lambda_0 [m] rounded to 1 nm) of the default slab."""
    thickness = _PPLN_CORE[1] - _PPLN_CORE[0]
    half_thickness = 0.5 * thickness
    k0 = 2.0 * math.pi / _PUMP_WAVELENGTH
    modes = []   # (omega, beta, core eps, n_eff) of the pump and the SH
    for harmonic in (1, 2):
        core_eps = _lorentz_eps(_PPLN_LN_EPS_INF, _PPLN_LN_DELTA_EPS, _PPLN_LN_RESONANCE, _PUMP_WAVELENGTH / harmonic)
        n_eff = analytic_slab_index(harmonic * k0, thickness, core_eps, _PPLN_SILICA_EPS, 'ez')
        modes.append((harmonic * _PUMP_OMEGA, harmonic * k0 * n_eff, core_eps, n_eff))
    _, gamma_pump = _slab_mode_profile(np.zeros(1), k0, modes[0][3], modes[0][2], _PPLN_SILICA_EPS, half_thickness)
    y = np.linspace(-half_thickness - 20.0 / gamma_pump, half_thickness + 20.0 / gamma_pump, 40001)
    dy = y[1] - y[0]
    profiles = []
    for harmonic, (_, _, core_eps, n_eff) in zip((1, 2), modes):
        profile, _ = _slab_mode_profile(y, harmonic * k0, n_eff, core_eps, _PPLN_SILICA_EPS, half_thickness)
        profiles.append(profile / math.sqrt(float(np.sum(profile ** 2)) * dy))   # integral e^2 dy = 1
    overlap = float(np.sum(np.where(np.abs(y) <= half_thickness, profiles[1] * profiles[0] ** 2, 0.0))) * dy
    (omega_1, beta_1, _, _), (omega_2, beta_2, _, _) = modes
    lattice = [2.0 / _PPLN_CELL * math.asin(0.5 * beta * _PPLN_CELL) for beta in (beta_1, beta_2)]
    dbeta = lattice[1] - 2.0 * lattice[0]
    # A_2 = (w2^2 mu0 eps0 d / (2 beta2)) A_1^2 I (2 / pi) L sinc, |A|^2 = 2 w mu0 P / beta, P_2 = beta2 |A_2|^2 / (2 w2 mu0)
    c_cmt = (beta_2 / (2.0 * omega_2 * MU0) * (omega_2 ** 2 * MU0 * EPS0 / (2.0 * beta_2)) ** 2
             * (2.0 * omega_1 * MU0 / beta_1) ** 2 * overlap ** 2 * (2.0 / math.pi) ** 2)
    period = round(2.0 * math.pi / dbeta * 1e9) * 1e-9
    return dbeta, c_cmt, period


def ppln_slab_shg_spec():
    """Template 'ppln_slab_shg': undepleted SHG in a PPLN slab waveguide, poling period swept over the QPM peak."""
    dbeta, c_cmt, period = _ppln_design()
    y0, y1 = _PPLN_CORE
    monitor_x = 'x_pp + L_pp + 1*um'
    ln_resonance_omega = 2.0 * math.pi * C0 / _PPLN_LN_RESONANCE
    return ModelSpec(
        title=PPLN_LABEL,
        description=('Second-harmonic generation in a 0.6 um z-cut lithium niobate slab in silica (2D, E_z = the '
                     'extraordinary index, d33): a 1550 nm fundamental-mode source pumps a 20 um periodically poled '
                     'section; the nonlinear_frequency_domain study sweeps the poling period Lambda over the '
                     'quasi-phase-matching peak, and the derived value P_sh (forward SH mode power, W/m) is drawn '
                     'against the coupled-mode curve cmt_sinc2 = C_cmt d33^2 P_pump^2 L_pp^2 sinc^2(u_qpm). dbeta '
                     '(the lattice-corrected mismatch of the continuum slab modes on the 15 nm mesh) and C_cmt (the '
                     'mode overlap) hold for this slab, wavelength and mesh only: recompute them when changing those. '
                     'LN extraordinary is a one-pole Lorentz fit to n_e at 1550 and 775 nm; silica is '
                     'non-dispersive. Fields and powers are per unit length in z (W/m).'),
        parameters=[ParameterSpec('lambda_p', '1550*nm', 'pump vacuum wavelength'),
                    ParameterSpec('P_pump', _PPLN_POWER, 'pump mode power [W/m]'),
                    ParameterSpec('d33', _PPLN_D33, 'LN d33 [m/V]'),
                    ParameterSpec('x_pp', _PPLN_START, 'start of the poled section [m]'),
                    ParameterSpec('L_pp', _PPLN_LENGTH, 'poled length [m]'),
                    ParameterSpec('Lambda', period, 'poling period [m] (swept)'),
                    ParameterSpec('dbeta', dbeta, 'beta_2w - 2 beta_w of the slab modes on the mesh [rad/m]'),
                    ParameterSpec('C_cmt', c_cmt, 'coupled-mode prefactor of P_2w / (d33^2 P_w^2 L^2)'),
                    ParameterSpec('u_qpm', '(dbeta - 2*pi/Lambda)*L_pp/2', 'QPM detuning phase [rad]')],
        domain=DomainSpec(size_x='x_pp + L_pp + 2*um', size_y=_PPLN_HEIGHT),
        geometry=[GeometrySpec('ln_slab', 'rectangle', {'x0': 0.0, 'y0': y0, 'x1': 'x_pp + L_pp + 2*um', 'y1': y1}),
                  GeometrySpec('poled', 'rectangle', {'x0': 'x_pp', 'y0': y0, 'x1': 'x_pp + L_pp', 'y1': y1}),
                  GeometrySpec('source_line', 'line', {'x0': 1e-6, 'y0': 0.0, 'x1': 1e-6, 'y1': _PPLN_HEIGHT}),
                  GeometrySpec('monitor_line', 'line', {'x0': monitor_x, 'y0': 0.0, 'x1': monitor_x,
                                                        'y1': _PPLN_HEIGHT})],
        materials=[MaterialSpec('silica', library='silica', selection=['all']),
                   MaterialSpec('ln_extraordinary', properties={'eps_r': _PPLN_LN_EPS_INF}, selection=['ln_slab'],
                                dispersion=[{'model': 'lorentz', 'delta_eps': _PPLN_LN_DELTA_EPS,
                                             'omega_0': ln_resonance_omega, 'gamma': 0.0}])],
        physics=[PhysicsSpec('nl', 'em_fdfd_nonlinear', features=[
            FeatureSpec('p', 'harmonic_channel', [], {'definition': 'wavelength', 'wavelength': 'lambda_p',
                                                      'polarization': 'ez'}),
            FeatureSpec('sh', 'harmonic_channel', [], {'definition': 'combination', 'combination': '2 p',
                                                       'polarization': 'ez'}),
            FeatureSpec('ppln', 'chi2_medium', ['poled'], {'crystal_class': '3m', 'd33': 'd33', 'orientation': 'z_cut',
                                                           'poling': 'sign(sin(2*pi*(x - x_pp)/Lambda))'}),
            FeatureSpec('pump', 'mode_source', ['source_line'], {'channel': 'p', 'mode_number': 1,
                                                                 'direction': 'positive', 'power': 'P_pump',
                                                                 'phase': 0.0}),
            FeatureSpec('pump_out', 'mode_monitor', ['monitor_line'], {'channel': 'p', 'mode_number': 1}),
            FeatureSpec('sh_out', 'mode_monitor', ['monitor_line'], {'channel': 'sh', 'mode_number': 1})])],
        mesh=MeshSpec(mode='uniform', dx=_PPLN_CELL),
        study=StudySpec(type='nonlinear_frequency_domain', sweep=SweepSpec(
            'Lambda', [round(period * (1.0 + fraction) * 1e9) * 1e-9 for fraction in _PPLN_SWEEP_FRACTIONS])),
        derived_values=[DerivedValueSpec('P_sh', 'power_positive', [], 'max', source='nl.sh_out'),
                        DerivedValueSpec('P_generated', 'nl_power_exchange_sh', [], 'mean')],
        analytic_curves=[AnalyticCurveSpec('cmt_sinc2', 'sweep',
                                           'C_cmt*d33**2*P_pump**2*L_pp**2*(sin(u_qpm)/u_qpm)**2', series='P_sh')])


# gaas_disk_shg: a GaAs disk (r 0.6 um, [001] out of plane, -43m d14) in vacuum, 3 x 3 um all-CPML box, uniform 10 nm
# mesh (~110k unknowns per channel); an H_z plane wave (in-plane E_y, 1e7 V/m) at 1550 nm in a TFSF box drives the E_z
# second harmonic through P_z = 2 eps0 d14 E_x' E_y' (crystal axes), so the SH has the other polarisation than the
# pump. The crystal angle theta_c (euler_alpha) is swept 0-90 deg: P_z = eps0 d14 ((E_y^2 - E_x^2) sin 2 theta + 2 E_x
# E_y cos 2 theta), whose two parts radiate with opposite y-mirror parity, so the SH power repeats every 90 deg (the 4-fold
# axis of the crystal) and the far-field pattern changes with the angle. GaAs as a lossless Lorentz pole at 400 nm
# fitted to n = 3.374 (1550 nm) and 3.68 (775 nm; the real material absorbs there, k ~ 0.09, not modelled).
GAAS_DISK_LABEL = 'GaAs disk SHG (mixed polarisation)'
GAAS_DISK_DESCRIPTION = ('Nonlinear FDFD (undepleted): an H_z plane wave at 1550 nm on a GaAs disk generates an E_z '
                         'second harmonic through d14; sweep of the crystal angle (90 degree period), SH far field.')
_GAAS_EPS_INF, _GAAS_DELTA_EPS, _GAAS_RESONANCE = 3.4586, 7.3976, 0.4e-6
_GAAS_SIZE = 3e-6                   # m, square box
_GAAS_ANGLES = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)   # deg


def gaas_disk_shg_spec():
    """Template 'gaas_disk_shg': H_z-pumped GaAs disk, E_z SH channel, crystal angle sweep, SH far field."""
    centre = 0.5 * _GAAS_SIZE
    return ModelSpec(
        title=GAAS_DISK_LABEL,
        description=('Second-harmonic generation in a GaAs disk with the [001] axis out of the plane (2D, -43m d14): '
                     'an in-plane polarised (H_z) plane wave at 1550 nm drives P_z = 2 eps0 d14 E_x E_y in crystal '
                     'axes, so the second harmonic is E_z polarised: the pump channel p is hz and the SH channel sh '
                     'ez. The study sweeps the crystal angle theta_c (degrees, the rotation of the crystal axes about '
                     'z): the generated power P_sh (power_exchange_sh, W/m) repeats every 90 degrees, and the far '
                     'field of the SH changes with the angle. GaAs is a lossless one-pole Lorentz fit to n at 1550 '
                     'and 775 nm (its absorption at 775 nm is not modelled).'),
        parameters=[ParameterSpec('lambda_p', '1550*nm', 'pump vacuum wavelength'),
                    ParameterSpec('E0', 1e7, 'incident in-plane field [V/m]'),
                    ParameterSpec('r_disk', '600*nm', 'disk radius'),
                    ParameterSpec('d14', _GAAS_D14, 'GaAs d14 [m/V]'),
                    ParameterSpec('theta_c', 0.0, 'crystal angle about z [deg] (swept)')],
        domain=DomainSpec(size_x=_GAAS_SIZE, size_y=_GAAS_SIZE),
        geometry=[GeometrySpec('disk', 'circle', {'cx': centre, 'cy': centre, 'r': 'r_disk'}),
                  GeometrySpec('tfsf_box', 'rectangle', {'x0': 0.7e-6, 'y0': 0.7e-6, 'x1': 2.3e-6, 'y1': 2.3e-6}),
                  GeometrySpec('far_field_box', 'rectangle', {'x0': 0.4e-6, 'y0': 0.4e-6, 'x1': 2.6e-6,
                                                              'y1': 2.6e-6})],
        materials=[MaterialSpec('gaas', properties={'eps_r': _GAAS_EPS_INF}, selection=['disk'], dispersion=[
            {'model': 'lorentz', 'delta_eps': _GAAS_DELTA_EPS, 'omega_0': 2.0 * math.pi * C0 / _GAAS_RESONANCE,
             'gamma': 0.0}])],
        physics=[PhysicsSpec('nl', 'em_fdfd_nonlinear', features=[
            FeatureSpec('p', 'harmonic_channel', [], {'definition': 'wavelength', 'wavelength': 'lambda_p',
                                                      'polarization': 'hz'}),
            FeatureSpec('sh', 'harmonic_channel', [], {'definition': 'combination', 'combination': '2 p',
                                                       'polarization': 'ez'}),
            FeatureSpec('gaas_chi2', 'chi2_medium', ['disk'], {'crystal_class': '-43m', 'd14': 'd14',
                                                               'orientation': 'euler', 'euler_alpha': 'theta_c'}),
            FeatureSpec('pump', 'plane_wave_tfsf', ['tfsf_box'], {'channel': 'p', 'amplitude': 'E0', 'phase': 0.0}),
            FeatureSpec('sh_far_field', 'far_field', ['far_field_box'], {'channel': 'sh', 'n_angles': 360})])],
        mesh=MeshSpec(mode='uniform', dx=10e-9),
        study=StudySpec(type='nonlinear_frequency_domain', sweep=SweepSpec('theta_c', list(_GAAS_ANGLES))),
        derived_values=[DerivedValueSpec('P_sh', 'nl_power_exchange_sh', [], 'mean')])


# silicon_thg_waveguide: a 220 nm silicon slab in silica (2D, E_z), 8 nm mesh (~12 cells per TH wavelength in silicon,
# 1000 x 250 cells per channel), a 1 MW/m pump mode at x = 1 um, a 4 um chi3 section (cubic, rho 1.27, n2 4.5e-18 at
# 1550 nm with the per-cell n0) from x = 2 um, the TH = 3 p channel undepleted. Silicon is the library entry with an
# nk_fit over 440-1450 nm (3 Lorentz poles): the TH at 450-480 nm is absorbed within ~1 um and strongly phase
# mismatched, so the forward TH comes from the last absorption length of the section and the backward TH from its
# first; the sweep of the pump wavelength (1.36-1.44 um, inside the fit range) shows both.
SI_THG_LABEL = 'Silicon slab waveguide THG (absorbing silicon)'
SI_THG_DESCRIPTION = ('Nonlinear FDFD (undepleted chi3): a guided pump mode in a silicon slab waveguide generates the '
                      'third harmonic in dispersive, absorbing silicon; sweep of the pump wavelength, forward and '
                      'backward third-harmonic power.')
_SI_THG_CELL = 8e-9                 # m
_SI_HEIGHT = 2.0e-6                 # m
_SI_CORE = (0.89e-6, 1.11e-6)       # m, y range of the 220 nm slab
_SI_THG_WAVELENGTHS = (1.36e-6, 1.38e-6, 1.40e-6, 1.42e-6, 1.44e-6)   # m, inside the nk_fit range
_SI_N2 = 4.5e-18                    # m^2/W at 1550 nm
_SI_TPA = 8e-12                     # m/W at 1550 nm


def _silicon_slab_geometry(size_x, x_nl, length, monitor_x, back_x=None):
    """Silicon slab along the whole box, its chi3 section, the source line at x = 1 um and the monitor line(s)."""
    y0, y1 = _SI_CORE
    geometry = [GeometrySpec('si_slab', 'rectangle', {'x0': 0.0, 'y0': y0, 'x1': size_x, 'y1': y1}),
                GeometrySpec('nl_section', 'rectangle', {'x0': x_nl, 'y0': y0, 'x1': f'{x_nl} + {length}', 'y1': y1}),
                GeometrySpec('source_line', 'line', {'x0': 1e-6, 'y0': 0.0, 'x1': 1e-6, 'y1': _SI_HEIGHT}),
                GeometrySpec('out_line', 'line', {'x0': monitor_x, 'y0': 0.0, 'x1': monitor_x, 'y1': _SI_HEIGHT})]
    if back_x is not None:
        geometry.append(GeometrySpec('back_line', 'line', {'x0': back_x, 'y0': 0.0, 'x1': back_x, 'y1': _SI_HEIGHT}))
    return geometry


def silicon_thg_waveguide_spec():
    """Template 'silicon_thg_waveguide': undepleted THG of a silicon slab mode, pump wavelength sweep."""
    return ModelSpec(
        title=SI_THG_LABEL,
        description=('Third-harmonic generation in a 220 nm silicon slab waveguide in silica (2D, E_z): a guided pump '
                     'mode drives a 4 um chi3 section (cubic silicon tensor, n2 at 1550 nm with the index of each '
                     'cell); the third harmonic is a generated channel th = 3 p in the undepleted mode. Silicon is the '
                     'library entry with a three-pole fit of its tabulated n, k (440-1450 nm), so the harmonic sees '
                     'the strong dispersion and absorption of silicon below 500 nm: it is absorbed within about a '
                     'micrometre and far from phase matched, and leaves mostly through the ends of the section. The '
                     'study sweeps the pump wavelength; P_th_forward / P_th_backward are the TH fluxes beyond the '
                     'section, P_th_generated the power the polarisation delivers (the rest is absorbed). One chi3 '
                     'serves THG and SPM (thg_scale = 1); set thg_scale for a measured THG susceptibility. Fields '
                     'and powers are per unit length in z (W/m).'),
        parameters=[ParameterSpec('lambda_p', '1400*nm', 'pump vacuum wavelength (swept)'),
                    ParameterSpec('P_pump', 1e6, 'pump mode power [W/m]'),
                    ParameterSpec('n2_si', _SI_N2, 'silicon Kerr index at 1550 nm [m^2/W]'),
                    ParameterSpec('x_nl', 2e-6, 'start of the chi3 section [m]'),
                    ParameterSpec('L_nl', 4e-6, 'chi3 section length [m]')],
        domain=DomainSpec(size_x='x_nl + L_nl + 2*um', size_y=_SI_HEIGHT),
        geometry=_silicon_slab_geometry('x_nl + L_nl + 2*um', 'x_nl', 'L_nl', 'x_nl + L_nl + 1*um', back_x=1.5e-6),
        materials=[MaterialSpec('silica', library='silica', selection=['all']),
                   MaterialSpec('silicon', library='silicon', selection=['si_slab'],
                                dispersion=[{'model': 'nk_fit', 'n_lorentz': 3, 'wavelength_start': 440e-9,
                                             'wavelength_stop': 1450e-9}])],
        physics=[PhysicsSpec('nl', 'em_fdfd_nonlinear', features=[
            FeatureSpec('p', 'harmonic_channel', [], {'definition': 'wavelength', 'wavelength': 'lambda_p',
                                                      'polarization': 'ez'}),
            FeatureSpec('th', 'harmonic_channel', [], {'definition': 'combination', 'combination': '3 p',
                                                       'polarization': 'ez'}),
            FeatureSpec('si_chi3', 'chi3_medium', ['nl_section'], {
                'tensor': 'cubic', 'anisotropy': 1.27, 'orientation': 'z_cut', 'strength': 'n2_beta', 'n2': 'n2_si',
                'reference_wavelength': 1.55e-6}),
            FeatureSpec('pump', 'mode_source', ['source_line'], {'channel': 'p', 'mode_number': 1,
                                                                 'direction': 'positive', 'power': 'P_pump',
                                                                 'phase': 0.0}),
            FeatureSpec('pump_out', 'mode_monitor', ['out_line'], {'channel': 'p', 'mode_number': 1}),
            FeatureSpec('th_forward', 'flux_monitor', ['out_line'], {'channel': 'th'}),
            FeatureSpec('th_backward', 'flux_monitor', ['back_line'], {'channel': 'th'})])],
        mesh=MeshSpec(mode='uniform', dx=_SI_THG_CELL),
        study=StudySpec(type='nonlinear_frequency_domain', sweep=SweepSpec('lambda_p', list(_SI_THG_WAVELENGTHS))),
        derived_values=[DerivedValueSpec('P_th_forward', 'power', [], 'mean', source='nl.th_forward'),
                        DerivedValueSpec('P_th_backward', '-power', [], 'mean', source='nl.th_backward'),
                        DerivedValueSpec('P_th_generated', 'nl_power_exchange_th', [], 'mean')])


# shg_saturation_tanh: the nlfd_shg_depleted_tanh strip (birefringent exact lattice phase matching: E_z pump p, H_z
# second harmonic sh, general d23 = d34) with the pump amplitude E0 = gamma_L / G_coef swept over Gamma L = 0.2-2,
# pump_depleted Newton-Krylov; the probed SH amplitude against the lattice coupled-mode E0 kappa_ratio tanh(Gamma L)
# and the transmitted pump against E0 / cosh(Gamma L) (the line source launches exactly E0 into the uniform medium).
SHG_SATURATION_LABEL = 'Pump-depleted SHG saturation (tanh^2)'
SHG_SATURATION_DESCRIPTION = ('Nonlinear FDFD (pump depleted, Newton-Krylov): phase-matched second-harmonic '
                              'generation in a periodic strip with the pump amplitude swept into saturation; SH and '
                              'depleted pump amplitudes against the coupled-mode tanh and sech curves.')
_SHG_SATURATION_GAMMA_L = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0)


def shg_saturation_tanh_spec():
    """Template 'shg_saturation_tanh': pump-depleted SHG of a phase-matched strip, sweep of Gamma L to saturation."""
    eps_sh, _, _, kappa_pump, kappa_sh = _depleted_design()
    medium = MaterialSpec('medium', properties={'eps_r': eps_sh, 'eps_zz': _EZ_EPS}, selection=['all'])
    channels = [('p', 'ez', {'definition': 'wavelength', 'wavelength': _PUMP_WAVELENGTH}),
                ('sh', 'hz', {'definition': 'combination', 'combination': '2 p'})]
    probes = _pump_probes('p', 'ez', _DEPLETED_SLAB_CELLS) + _harmonic_probes('sh', 'ey', _DEPLETED_SLAB_CELLS)
    spec = _channel_strip_spec('pump-depleted SHG', medium, _DEPLETED_SLAB_CELLS,
                               ('chi2_medium', _general_d(_D_BIREFRINGENT)), channels, [('pump', 'p', 'E0')], probes,
                               settings=_pump_depleted_settings('newton'))
    gamma_per_field = _DEPLETED_SLAB_CELLS * _STRIP_CELL * math.sqrt(kappa_pump * kappa_sh)
    return replace(
        spec, title=SHG_SATURATION_LABEL,
        description=('Second-harmonic generation into saturation (2D, coupling_mode pump_depleted, Newton-Krylov): a '
                     'y-periodic strip (4 rows of 10 nm, CPML along x) filled with a birefringent medium whose E_z '
                     '(pump, 1550 nm, eps_zz 2.25) and H_z (second harmonic, eps_yy set by the template) lattice '
                     'waves are exactly phase matched, and a 20 um chi2 slab (general d23 = d34 = 25 pm/V). The '
                     'study sweeps Gamma L, the pump amplitude being E0 = gamma_L / G_coef; A_sh (forward SH, |E_y| at '
                     'the cell centre) and A_pump (transmitted pump |E_z|) are drawn against the lattice coupled-mode '
                     'curves E0 kappa_ratio tanh(Gamma L) and E0 / cosh(Gamma L) (accuracy ~1 %: coupled-mode '
                     'theory, backward waves). G_coef and kappa_ratio hold for this strip only. Fields are per unit '
                     'length in z.'),
        parameters=[ParameterSpec('gamma_L', 1.0, 'conversion length product Gamma L (swept)'),
                    ParameterSpec('G_coef', gamma_per_field, 'Gamma L per unit pump amplitude [m/V]'),
                    ParameterSpec('kappa_ratio', math.sqrt(kappa_sh / kappa_pump),
                                  'SH / pump amplitude ratio at full conversion'),
                    ParameterSpec('E0', 'gamma_L/G_coef', 'pump amplitude [V/m]')],
        study=StudySpec(type='nonlinear_frequency_domain', sweep=SweepSpec('gamma_L', list(_SHG_SATURATION_GAMMA_L))),
        derived_values=[DerivedValueSpec('A_sh', 'abs', [], 'mean', source='nl.sh_forward'),
                        DerivedValueSpec('A_pump', 'abs', [], 'mean', source='nl.pump_forward')],
        analytic_curves=[AnalyticCurveSpec('cmt_tanh', 'sweep', 'E0*kappa_ratio*tanh(gamma_L)', series='A_sh'),
                         AnalyticCurveSpec('cmt_sech', 'sweep', 'E0/cosh(gamma_L)', series='A_pump')])


# silicon_fwm_waveguide: a 220 nm silicon slab (library silicon, n 3.476) in silica, 20 nm mesh (2200 x 100 cells per
# channel), pump p 1550 nm and signal s 1560 nm mode sources at x = 1 um, idler i = 2 p - s (1540 nm), a 40 um chi3
# section (cubic, n2 4.5e-18, tpa_beta 8e-12, free carriers tau 1 ns), pump_depleted Newton-Krylov, sweep of the pump
# power. In 2D the intensity is ~P / 0.3 um: 4 MW/m is ~13 GW/cm^2 (a 0.5 W pump in a 3D wire would be ~0.5 GW/cm^2),
# so at the top of the sweep the free carriers absorb ~30 % and shift the phase by ~1 rad, and the idler conversion
# saturates below the low-power (gamma P L)^2 law.
SI_FWM_LABEL = 'Silicon waveguide FWM with TPA and free carriers'
SI_FWM_DESCRIPTION = ('Nonlinear FDFD (pump depleted, Newton-Krylov): degenerate four-wave mixing in a silicon slab '
                      'waveguide with two-photon and free-carrier absorption; sweep of the pump power, idler '
                      'conversion and the nonlinear losses.')
_SI_FWM_CELL = 20e-9                # m
_SI_FWM_POWERS = (2.5e5, 5e5, 1e6, 2e6, 4e6)   # W/m


def silicon_fwm_waveguide_spec():
    """Template 'silicon_fwm_waveguide': pump-depleted FWM in a silicon slab with TPA and free carriers."""
    return ModelSpec(
        title=SI_FWM_LABEL,
        description=('Degenerate four-wave mixing in a 220 nm silicon slab waveguide in silica (2D, E_z, coupling_mode '
                     'pump_depleted with the Newton-Krylov iteration): pump p (1550 nm) and signal s (1560 nm) mode '
                     'sources, the idler i = 2 p - s (1540 nm) generated in a 40 um chi3 section (cubic silicon '
                     'tensor, n2, two-photon absorption and steady-state free carriers with a 1 ns lifetime). The '
                     'study sweeps the pump power: eta_idler = P_idler / P_signal grows as P_pump^2 at low power and '
                     'saturates when two-photon and free-carrier absorption (TPA_loss, FCA_loss, peak carrier '
                     'density N_peak) and the carrier index change set in. All mixing products off the three '
                     'channels are dropped (validation info). Silicon is non-dispersive here (n 3.476): phase '
                     'matching comes from the slab dispersion only. Fields and powers are per unit length in z (W/m): '
                     'the 2D intensity is about the power over 0.3 um.'),
        parameters=[ParameterSpec('P_pump', 1e6, 'pump mode power [W/m] (swept)'),
                    ParameterSpec('P_signal', 1e4, 'signal mode power [W/m]'),
                    ParameterSpec('n2_si', _SI_N2, 'silicon Kerr index at 1550 nm [m^2/W]'),
                    ParameterSpec('beta_si', _SI_TPA, 'silicon two-photon absorption at 1550 nm [m/W]'),
                    ParameterSpec('tau_c', 1e-9, 'free-carrier lifetime [s]'),
                    ParameterSpec('x_nl', 2e-6, 'start of the chi3 section [m]'),
                    ParameterSpec('L_nl', 40e-6, 'chi3 section length [m]')],
        domain=DomainSpec(size_x='x_nl + L_nl + 2*um', size_y=_SI_HEIGHT),
        geometry=_silicon_slab_geometry('x_nl + L_nl + 2*um', 'x_nl', 'L_nl', 'x_nl + L_nl + 1*um'),
        materials=[MaterialSpec('silica', library='silica', selection=['all']),
                   MaterialSpec('silicon', library='silicon', selection=['si_slab'])],
        physics=[PhysicsSpec('nl', 'em_fdfd_nonlinear', settings={
            'coupling_mode': 'pump_depleted', 'iteration_method': 'newton', 'max_iterations': _NEWTON_STEPS},
            features=[
                FeatureSpec('p', 'harmonic_channel', [], {'definition': 'wavelength', 'wavelength': 1.55e-6,
                                                          'polarization': 'ez'}),
                FeatureSpec('s', 'harmonic_channel', [], {'definition': 'wavelength', 'wavelength': 1.56e-6,
                                                          'polarization': 'ez'}),
                FeatureSpec('i', 'harmonic_channel', [], {'definition': 'combination', 'combination': '2 p - s',
                                                          'polarization': 'ez'}),
                FeatureSpec('si_chi3', 'chi3_medium', ['nl_section'], {
                    'tensor': 'cubic', 'anisotropy': 1.27, 'orientation': 'z_cut', 'strength': 'n2_beta',
                    'n2': 'n2_si', 'tpa_beta': 'beta_si', 'reference_wavelength': 1.55e-6, 'free_carriers': True,
                    'carrier_lifetime': 'tau_c'}),
                FeatureSpec('pump', 'mode_source', ['source_line'], {'channel': 'p', 'mode_number': 1,
                                                                     'direction': 'positive', 'power': 'P_pump',
                                                                     'phase': 0.0}),
                FeatureSpec('signal', 'mode_source', ['source_line'], {'channel': 's', 'mode_number': 1,
                                                                       'direction': 'positive', 'power': 'P_signal',
                                                                       'phase': 0.0}),
                *(FeatureSpec(f'{channel}_out', 'mode_monitor', ['out_line'], {'channel': channel, 'mode_number': 1})
                  for channel in ('p', 's', 'i'))])],
        mesh=MeshSpec(mode='uniform', dx=_SI_FWM_CELL),
        study=StudySpec(type='nonlinear_frequency_domain', sweep=SweepSpec('P_pump', list(_SI_FWM_POWERS))),
        derived_values=[DerivedValueSpec('eta_idler', 'power_positive/P_signal', [], 'max', source='nl.i_out'),
                        DerivedValueSpec('T_pump', 'power_positive/P_pump', [], 'max', source='nl.p_out'),
                        DerivedValueSpec('G_signal', 'power_positive/P_signal', [], 'max', source='nl.s_out'),
                        DerivedValueSpec('TPA_loss', 'nl_nonlinear_absorption', [], 'mean'),
                        DerivedValueSpec('FCA_loss', 'nl_free_carrier_absorption', [], 'mean'),
                        DerivedValueSpec('N_peak', 'nl_peak_carrier_density', [], 'mean')])


NLFD_TEMPLATES = (   # (key, label, description, factory, summary) for registry.register_template
    ('ppln_slab_shg', PPLN_LABEL, PPLN_DESCRIPTION, ppln_slab_shg_spec,
     'Second harmonic of a 1550 nm guided mode in a periodically poled lithium niobate slab: second-harmonic power '
     'against the poling period, compared with the coupled-mode curve.'),
    ('gaas_disk_shg', GAAS_DISK_LABEL, GAAS_DISK_DESCRIPTION, gaas_disk_shg_spec,
     'Second harmonic of a gallium arsenide disk lit by light polarised in the plane: the harmonic comes out with '
     'the electric field out of the plane, and its power repeats as the crystal is turned by 90 degrees.'),
    ('silicon_thg_waveguide', SI_THG_LABEL, SI_THG_DESCRIPTION, silicon_thg_waveguide_spec,
     'Third harmonic of a guided mode in a silicon waveguide, where silicon absorbs the harmonic strongly: '
     'harmonic power leaving forward and backward against the pump wavelength.'),
    ('shg_saturation_tanh', SHG_SATURATION_LABEL, SHG_SATURATION_DESCRIPTION, shg_saturation_tanh_spec,
     'Second-harmonic generation driven into saturation: the harmonic grows and the pump is used up as the pump '
     'gets stronger, compared with the textbook tanh and sech curves.'),
    ('silicon_fwm_waveguide', SI_FWM_LABEL, SI_FWM_DESCRIPTION, silicon_fwm_waveguide_spec,
     'Four-wave mixing in a silicon waveguide: a new wavelength generated from a pump and a signal, limited at high '
     'power by two-photon and free-carrier absorption.'),
)
