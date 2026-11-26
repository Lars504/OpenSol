"""Self-tests of the spontaneous Brillouin math (physics/mechanics/brillouin.py; CPU, specs() empty, check({})): field
products and optical stress vs a 3D four-index contraction, the azimuthal m = 0 / 2 split vs a phi quadrature of the 3D
work, the FDT readout vs a closed-form oscillator and the sum rule, the damped-oscillator fit, the homogeneous and local
reference spectra with their zero padding, and the bilinear / phase-corrected / phase-gradient sampling. error =
number of failed sub-checks (tolerance 0); physics imports inside the checks."""
import math

import numpy as np

from ...core.constants import EPS0, HBAR, K_B
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import SubChecks, max_relative_error, norm_relative_error, no_runs

_POINTS = 9                 # random sample points of the algebraic checks
_ALGEBRA_RTOL = 1e-13       # identities of sums of a few products (round-off)
_PHI_POINTS = 64            # phi quadrature (exact for the trigonometric degree 4 of sigma . S)
_TEMPERATURE = 293.15       # K
_VOIGT = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (1, 2): 3, (2, 1): 3, (0, 2): 4, (2, 0): 4, (0, 1): 5, (1, 0): 5}
_CUBIC_P = (0.12, 0.27, -0.06)   # p11, p12, p44 of a cubic crystal (silicon-like) for the rotated tensor
_EULER = (0.4, 1.1, -0.7)        # rad: z-x-z rotation of the cubic tensor
_ISOTROPIC_P = (0.31, 0.27, 0.05)   # p11, p12 and an explicit p44 != (p11 - p12) / 2
# FDT oscillator (3 dofs, dimensionless SI numbers): K, M, loss factor, viscous beta of the sum rule
_FDT_RTOL = 1e-12
_LOSS_FACTOR = 1e-3
_VISCOUS_BETA = 2e-3        # C = beta K: damping ratios 1e-3 .. 3e-3 (a constant loss factor has a log tail at f -> 0)
_SUM_RULE_STEP = 2e-4       # rad/s grid of the sum rule (5 points per half width of the sharpest line)
_SUM_RULE_TOP = 20.0        # rad/s: the tail above carries ~1e-7
_SUM_RULE_RTOL = 1e-4       # first estimate (quadrature and tails ~1e-6)
# DHO fit round trip (water_532nm numbers of the plan)
_DHO = (7.439e9, 0.65e9, 1e-12)   # f0 [Hz], gamma [Hz], peak psd [1/Hz]
_DHO_BAND = (6.5e9, 8.3e9, 61)
_DHO_RTOL = 1e-8
# homogeneous reference (dimensionless medium: rho 1, M 1, v 1, a phonon wavelength 1 m, Gamma = 0.2 Omega_0)
_DENSITY = 1.0
_MODULUS = 1.0
_DAMPING = 0.2              # Gamma / Omega_0
_SHEAR_VISCOSITY = 0.01     # Pa s: a liquid G* = j Omega eta (no transverse drive here)
_SINGLE_GRID = (64, 8, 1.0 / 16.0, 0.5)   # n_x, n_y, dx, dy: q0 = 2 pi exactly on the FFT bin 4 of the 4 m box
_SINGLE_RTOL = 1e-10
_ENVELOPE = (80.0, 20.0, 0.125, 2.0, 4.0)   # w_x, w_y [m], dx, dy, half extent in widths (padding 2)
_ENVELOPE_RTOL = 2e-3       # L2 over the band: q spread (1 / (w_x Gamma / 2))^2 ~ 4e-4 + transverse shift ~3e-4
_STATIC_RTOL = 1e-10
_BAND = (0.6, 1.4, 21)      # Hz around f0 = 1 Hz
# a short drive (2 x 2 m, 2 phonon wavelengths along x): the automatic padding against twice as much per axis; the
# two nearest images per axis hold together below brillouin.IMAGE_TOLERANCE = 1e-3 of the readout at the automatic
# padding (first estimate)
_SHORT_GRID = (32, 4, 1.0 / 16.0, 0.5)
_SHORT_PADDING_RTOL = 3e-3
# ceil(1 + ln(2 / 1e-3) / (delta L)) per axis (brillouin.reference_padding), L_x = 32 / 16 = L_y = 4 x 0.5 = 2 m,
# delta = |Im k| at the band's lowest 0.6 Hz: eta_L = 0.2 / (2 pi), Omega eta_L = 0.12, M* = 1 + 0.12j, |Im k| =
# Omega |Im (M*)^(-1/2)| = 3.7699 x 1.0144^(-1/4) sin(atan(0.12) / 2) = 3.7699 x 0.99643 x 0.059679 = 0.22418 1/m (the
# shear law of viscosity 0.01, |Im k| ~ 13.7 1/m, attenuates far faster); ln(2000) = 7.6009, 7.6009 / (0.22418 x 2)
# = 16.953, so P = ceil(17.953) = 18 on both axes (ln(1000) alone would give ceil(16.407) = 17)
_SHORT_PADDING = (18, 18)
_PHASE_RTOL = 1e-12         # phase-corrected sampling of a plane wave (round-off)
_LAW_RTOL = 1e-14


def _complex(rng, shape) -> np.ndarray:
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _four_index(voigt) -> np.ndarray:
    """(3, 3, 3, 3) p_ijkl = p_IJ of a Voigt (6, 6) p (engineering strains: Delta(1/eps)_ij = p_ijkl S_kl)."""
    tensor = np.empty((3, 3, 3, 3))
    for (i, j), first in _VOIGT.items():
        for (k, l), second in _VOIGT.items():
            tensor[i, j, k, l] = voigt[first, second]
    return tensor


def _voigt_matrix(tensor) -> np.ndarray:
    pairs = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    return np.array([[tensor[i, j, k, l] for k, l in pairs] for i, j in pairs])


def _isotropic_voigt(p11, p12, p44) -> np.ndarray:
    voigt = np.full((3, 3), p12) + (p11 - p12) * np.eye(3)
    return np.block([[voigt, np.zeros((3, 3))], [np.zeros((3, 3)), p44 * np.eye(3)]])


def _rotated(voigt, angles) -> np.ndarray:
    """Voigt p of the four-index tensor rotated by z-x-z Euler angles."""
    def axis_rotation(angle, axis):
        c, s = math.cos(angle), math.sin(angle)
        rows = np.eye(3)
        a, b = [index for index in range(3) if index != axis]
        rows[a, a], rows[a, b], rows[b, a], rows[b, b] = c, -s, s, c
        return rows
    rotation = axis_rotation(angles[0], 2) @ axis_rotation(angles[1], 0) @ axis_rotation(angles[2], 2)
    tensor = np.einsum('ia,jb,kc,ld,abcd->ijkl', rotation, rotation, rotation, rotation, _four_index(voigt))
    return _voigt_matrix(tensor)


def _brute_amplitude(tensor, strain, eps, collection, incident) -> np.ndarray:
    """(n,) sum_ij Delta eps_ij E_c,i E_i,j with Delta eps_ij = -eps_i eps_j p_ijkl S_kl (S (3, 3, n) tensor strain)."""
    inverse = np.einsum('ijkl,kln->ijn', tensor, strain)
    change = -eps[:, None, :] * eps[None, :, :] * inverse
    return np.einsum('ijn,in,jn->n', change, collection, incident)


def _engineering(strain) -> np.ndarray:
    """(6, n) Voigt engineering strains of a tensor strain (3, 3, n)."""
    return np.stack([strain[0, 0], strain[1, 1], strain[2, 2], 2.0 * strain[1, 2], 2.0 * strain[0, 2],
                     2.0 * strain[0, 1]])


def _check_field_products(_):
    from .brillouin import (PLANAR_COMPONENTS, field_products, interface_traction, moving_boundary_traction,
                            optical_stress)
    from .optomechanics_fem import moving_boundary_weights

    checks = SubChecks()
    rng = np.random.default_rng(61)
    incident, collection = _complex(rng, (3, _POINTS)), _complex(rng, (3, _POINTS))
    eps = 1.0 + 2.0 * rng.random((3, _POINTS))
    strain = rng.standard_normal((3, 3, _POINTS))
    strain = 0.5 * (strain + strain.transpose(1, 0, 2))
    engineering = _engineering(strain)
    names = PLANAR_COMPONENTS
    products = field_products(dict(zip(names, incident)), dict(zip(names, collection)),
                              {name + name: eps[index] for index, name in enumerate(names)})
    p11, p12, p44 = _ISOTROPIC_P
    tensors = {'isotropic': _isotropic_voigt(p11, p12, p44),
               'rotated_cubic': _rotated(_isotropic_voigt(*_CUBIC_P), _EULER),
               'general': rng.standard_normal((6, 6))}   # not symmetric in (I, J): pins the transposed contraction
    everywhere = np.ones(_POINTS, dtype=bool)
    for name, voigt in tensors.items():
        expected = _brute_amplitude(_four_index(voigt), strain, eps, collection, incident)
        checks.add_bound(f'products_{name}', max_relative_error(np.einsum('in,ij,jn->n', products, voigt,
                                                                          engineering), expected), _ALGEBRA_RTOL)
        regions = () if name == 'isotropic' else ((everywhere, voigt),)
        stress = optical_stress(products, p11, p12, p44, crystal_regions=regions)
        checks.add_bound(f'stress_{name}', max_relative_error(np.einsum('in,in->n', stress, engineering), expected),
                         _ALGEBRA_RTOL)
    default = optical_stress(products, p11, p12)   # p44 unset: (p11 - p12) / 2
    expected = _brute_amplitude(_four_index(_isotropic_voigt(p11, p12, 0.5 * (p11 - p12))), strain, eps, collection,
                                incident)
    checks.add_bound('stress_default_p44', max_relative_error(np.einsum('in,in->n', default, engineering), expected),
                     _ALGEBRA_RTOL)
    # em_fdfd_ez: E_z only, a pure dilation drive sigma_xx = sigma_yy = p12 G_zz, sigma_zz = p11 G_zz, no shear
    dilation = optical_stress(field_products({'z': incident[2]}, {'z': collection[2]}, eps[2]), p11, p12, p44)
    g_zz = -eps[2] ** 2 * collection[2] * incident[2]
    reference = np.stack([p12 * g_zz, p12 * g_zz, p11 * g_zz] + [np.zeros(_POINTS)] * 3)
    checks.add_bound('ez_dilation', max_relative_error(dilation, reference), _ALGEBRA_RTOL)
    checks.add_raises('unknown_component_raises', lambda: field_products({'ex': incident[0]}, {'x': incident[0]}, 1.0))
    # moving boundary with E_c = E_i*: the optomechanics_fem bracket (cylindrical (r, phi, z) = planar (x, z, y))
    normals = rng.standard_normal((_POINTS, 2))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    sides = [_complex(rng, (3, _POINTS)) for _ in range(2)]
    eps_sides = (1.0 + rng.random(_POINTS), 2.0 + rng.random(_POINTS))
    value = moving_boundary_traction(normals, [dict(zip(names, side)) for side in sides],
                                     [dict(zip(names, np.conj(side))) for side in sides], eps_sides)
    cylindrical = [np.stack([side[0], side[2], side[1]], axis=-1) for side in sides]   # (r, phi, z) = (x, z, y)
    expected = moving_boundary_weights(np.full(_POINTS, 0.5 / math.pi), np.ones(_POINTS), normals, cylindrical[0],
                                       cylindrical[1], eps_sides[0], eps_sides[1]) / EPS0
    checks.add_bound('moving_boundary_johnson', max_relative_error(value, expected), _ALGEBRA_RTOL)
    # the one-point form from the continuous E_par and D_n equals the two-sided form of sides that share them (E_s =
    # E_par + (D_n / eps_s) n per side), whatever tangential D the point carries
    tangent = np.stack([-normals[:, 1], normals[:, 0]], axis=1)
    fields = {}
    for name in ('incident', 'collection'):
        parallel, out_of_plane, normal_d = (_complex(rng, _POINTS) for _ in range(3))
        sides_of = [{'x': parallel * tangent[:, 0] + normal_d / eps * normals[:, 0],
                     'y': parallel * tangent[:, 1] + normal_d / eps * normals[:, 1], 'z': out_of_plane}
                    for eps in eps_sides]
        stray = _complex(rng, _POINTS)   # tangential D: not read
        point_d = {'x': normal_d * normals[:, 0] + stray * tangent[:, 0],
                   'y': normal_d * normals[:, 1] + stray * tangent[:, 1]}
        fields[name] = (sides_of, sides_of[1], point_d)
    two_sided = moving_boundary_traction(normals, fields['incident'][0], fields['collection'][0], eps_sides)
    one_point = interface_traction(normals, fields['incident'][1], fields['collection'][1], fields['incident'][2],
                                   fields['collection'][2], eps_sides)
    checks.add_bound('interface_one_point_vs_two_sided', max_relative_error(one_point, two_sided), _ALGEBRA_RTOL)
    return checks.result()


def _standing_waves(amplitudes, cross, phi) -> dict:
    """{'r', 'phi', 'z'} fields at phi of an m = 1 standing wave (header): (A_r cos, A_phi sin, A_z cos), rotated by
    +90 deg when cross: (A_r sin, -A_phi cos, A_z sin)."""
    c, s = math.cos(phi), math.sin(phi)
    if cross:
        return {'r': amplitudes['r'] * s, 'phi': -amplitudes['phi'] * c, 'z': amplitudes['z'] * s}
    return {'r': amplitudes['r'] * c, 'phi': amplitudes['phi'] * s, 'z': amplitudes['z'] * c}


def _strain_fields(normal, shear, phi, kind) -> np.ndarray:
    """(6, n) optical-order cylindrical strain (rr, zz, phiphi, z-phi, r-phi, rz) at phi of a solid field: 'm0' normal
    const, 'torsion' shear const, 'even' normal cos 2 phi / shear sin 2 phi, 'odd' normal sin 2 phi / shear -cos."""
    c, s = math.cos(2.0 * phi), math.sin(2.0 * phi)
    factors = {'m0': (1.0, 0.0), 'torsion': (0.0, 1.0), 'even': (c, s), 'odd': (s, -c)}[kind]
    strain = np.zeros((6,) + normal.shape[1:], dtype=np.complex128)
    strain[[0, 1, 2, 5]] = factors[0] * normal
    strain[[3, 4]] = factors[1] * shear
    return strain


def _check_azimuthal_split(_):
    from .brillouin import (AZIMUTHAL_PARTS, CYLINDRICAL_COMPONENTS, DETECTIONS, azimuthal_interface_traction,
                            azimuthal_parts, field_products, interface_traction, optical_stress)

    checks = SubChecks()
    rng = np.random.default_rng(67)
    names = CYLINDRICAL_COMPONENTS
    incident = {name: _complex(rng, _POINTS) for name in names}
    collection = {name: _complex(rng, _POINTS) for name in names}
    eps = {'rr': 1.5 + rng.random(_POINTS), 'zz': 1.5 + rng.random(_POINTS), 'phiphi': 1.5 + rng.random(_POINTS)}
    p11, p12, p44 = _ISOTROPIC_P
    weights = rng.random(_POINTS)
    normal_strain, shear_strain = rng.standard_normal((4, _POINTS)), rng.standard_normal((2, _POINTS))
    solid_strain = np.concatenate([normal_strain, shear_strain])   # solid rows rr, zz, tt, g_rz, g_zt, g_rt
    normals = rng.standard_normal((_POINTS, 2))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    # interface amplitudes: E and D (the D phi entry is not read) of the incident and collection fields
    interface = {key: {name: _complex(rng, _POINTS) for name in names}
                 for key in ('incident', 'collection', 'incident_d', 'collection_d')}
    eps_sides = (1.0 + rng.random(_POINTS), 2.0 + rng.random(_POINTS))
    step = 2.0 * math.pi / _PHI_POINTS
    for detection in DETECTIONS:
        cross = detection == 'cross_polarized'
        parts, torsion_fraction = azimuthal_parts(incident, collection, eps, detection, p11, p12, p44, weights)
        works = {kind: np.zeros(_POINTS, dtype=np.complex128) for kind in ('m0', 'torsion', 'even', 'odd')}
        torsion_rows = np.zeros((2, _POINTS), dtype=np.complex128)
        boundary = {'constant': 0.0, 'cos2': 0.0, 'sin2': 0.0}
        for index in range(_PHI_POINTS):
            phi = index * step
            stress = optical_stress(field_products(_standing_waves(incident, False, phi),
                                                   _standing_waves(collection, cross, phi), eps,
                                                   components=names), p11, p12, p44)
            for kind in works:
                works[kind] += step * np.sum(stress * _strain_fields(normal_strain, shear_strain, phi, kind), axis=0)
            torsion_rows += step / (2.0 * math.pi) * stress[[3, 4]]
            value = interface_traction(
                normals, _standing_waves(interface['incident'], False, phi),
                _standing_waves(interface['collection'], cross, phi),
                _standing_waves(interface['incident_d'], False, phi),
                _standing_waves(interface['collection_d'], cross, phi), eps_sides, components=names)
            boundary['constant'] = boundary['constant'] + step / (2.0 * math.pi) * value
            boundary['cos2'] = boundary['cos2'] + step / math.pi * math.cos(2.0 * phi) * value
            boundary['sin2'] = boundary['sin2'] + step / math.pi * math.sin(2.0 * phi) * value
        expected_keys = AZIMUTHAL_PARTS[detection]
        checks.add(f'{detection}_part_keys', set(parts) == set(expected_keys), value=[list(key) for key in parts])
        kinds = {(0, 'even'): ('m0', 2.0 * math.pi, normal_strain), (2, 'even'): ('even', math.pi, solid_strain),
                 (2, 'odd'): ('odd', math.pi, solid_strain)}
        scale = max(float(np.max(np.abs(works[kinds[key][0]]))) for key in expected_keys)
        for key, (kind, ring, amplitudes) in kinds.items():
            label = f'{detection}_m{key[0]}_{key[1]}'
            if key in parts:
                checks.add_bound(label, max_relative_error(ring * np.sum(parts[key] * amplitudes, axis=0),
                                                           works[kind]), _ALGEBRA_RTOL)
            else:
                checks.add_bound(f'{label}_zero', float(np.max(np.abs(works[kind]))) / scale, _ALGEBRA_RTOL)
        dropped = 2.0 * math.pi * float(np.sum(weights * np.sum(np.abs(torsion_rows) ** 2, axis=0)))
        kept = sum((2.0 * math.pi if order == 0 else math.pi) * float(np.sum(weights * np.sum(np.abs(rows) ** 2,
                                                                                               axis=0)))
                   for (order, _), rows in parts.items())
        checks.add_bound(f'{detection}_torsion_fraction', abs(torsion_fraction - dropped / (dropped + kept)),
                         _ALGEBRA_RTOL)
        if not cross:
            checks.add_bound('co_polarized_torsion_zero', torsion_fraction, _ALGEBRA_RTOL)
        boundary_parts = azimuthal_interface_traction(
            normals, interface['incident'], interface['collection'], interface['incident_d'],
            interface['collection_d'], eps_sides, detection)
        harmonic_of = {(0, 'even'): 'constant', (2, 'even'): 'cos2', (2, 'odd'): 'sin2'}
        boundary_scale = max(float(np.max(np.abs(boundary[harmonic_of[key]]))) for key in expected_keys)
        for key, harmonic in harmonic_of.items():
            label = f'{detection}_boundary_m{key[0]}_{key[1]}'
            if key in boundary_parts:
                checks.add_bound(label, max_relative_error(boundary_parts[key], boundary[harmonic]), _ALGEBRA_RTOL)
            else:
                checks.add_bound(f'{label}_zero', float(np.max(np.abs(boundary[harmonic]))) / boundary_scale,
                                 _ALGEBRA_RTOL)
    # low NA: an x-polarised uniform field (A_r = E, A_phi = -E, A_z = 0) has E . E phi independent, so an isotropic
    # dilation drive (p11 = p12, p44 = 0) has no m = 2 part
    field = _complex(rng, _POINTS)
    uniform = {'r': field, 'phi': -field, 'z': np.zeros(_POINTS)}
    parts, _ = azimuthal_parts(uniform, uniform, 1.78, 'co_polarized', 0.31, 0.31, 0.0)
    checks.add_bound('low_na_no_m2', float(np.max(np.abs(parts[(2, 'even')])))
                     / float(np.max(np.abs(parts[(0, 'even')]))), _ALGEBRA_RTOL)
    return checks.result()


def _check_fdt_oscillator(_):
    from scipy.linalg import eigh

    from .brillouin import band_integral, fdt_psd, sideband_factors

    checks = SubChecks()
    rng = np.random.default_rng(71)
    stiffness = np.array([[2.0, -0.6, 0.1], [-0.6, 3.5, -0.9], [0.1, -0.9, 5.0]])
    mass = np.diag([1.2, 1.0, 0.8])
    drive = _complex(rng, 3)
    eigenvalues, modes = eigh(stiffness, mass)   # M-orthonormal: chi = sum phi phi^T / (lambda (..) - Omega^2)
    overlaps = np.abs(modes.T @ drive) ** 2
    scale = 2.0                                  # a mirror factor
    omega = np.linspace(0.5, 3.0, 41)
    readouts = np.array([np.conj(drive) @ np.linalg.solve(stiffness * (1.0 + 1j * _LOSS_FACTOR) - value ** 2 * mass,
                                                          drive) for value in omega])
    closed = (2.0 * K_B * _TEMPERATURE / omega * scale
              * np.sum(overlaps * eigenvalues * _LOSS_FACTOR / np.abs(eigenvalues * (1.0 + 1j * _LOSS_FACTOR)
                                                                     - omega[:, None] ** 2) ** 2, axis=1))
    checks.add_bound('oscillator_closed_form', max_relative_error(fdt_psd(omega, readouts, _TEMPERATURE, scale),
                                                                  closed), _FDT_RTOL)
    # sum rule 2 int_0^inf psd df = k_B T w^H K^-1 w (Kelvin-Voigt C = beta K: a loss factor's 1 / Omega tail
    # diverges logarithmically at f -> 0)
    grid = np.arange(_SUM_RULE_STEP, _SUM_RULE_TOP, _SUM_RULE_STEP)
    viscous = np.sum(overlaps / (eigenvalues * (1.0 + 1j * grid[:, None] * _VISCOUS_BETA) - grid[:, None] ** 2), axis=1)
    psd = fdt_psd(grid, viscous, _TEMPERATURE, 1.0)
    static = K_B * _TEMPERATURE * float(np.real(np.conj(drive) @ np.linalg.solve(stiffness, drive)))
    checks.add_relative('sum_rule', 2.0 * float(band_integral(grid / (2.0 * math.pi), psd)), static, _SUM_RULE_RTOL)
    active = np.array([np.conj(drive) @ np.linalg.solve(stiffness * (1.0 - 1j * _LOSS_FACTOR) - 1.0 * mass, drive)])
    checks.add_raises('active_operator_raises', lambda: fdt_psd(np.array([1.0]), active, _TEMPERATURE, 1.0))
    frequencies = np.array([1e9, 7.5e9, 3e11])
    stokes, anti_stokes = sideband_factors(frequencies, _TEMPERATURE)
    x = HBAR * 2.0 * math.pi * frequencies / (K_B * _TEMPERATURE)
    checks.add_bound('sideband_difference', max_relative_error(stokes - anti_stokes, x), _FDT_RTOL)
    checks.add_bound('sideband_ratio', max_relative_error(stokes / anti_stokes, np.exp(x)), _FDT_RTOL)
    return checks.result()


def _dho(frequencies, shift, gamma, amplitude):
    return amplitude / ((shift ** 2 - frequencies ** 2) ** 2 + gamma ** 2 * frequencies ** 2)


def _check_dho_fit(_):
    from .brillouin import band_integral, dho_fit

    checks = SubChecks()
    shift, gamma, peak = _DHO
    minimum = gamma ** 2 * shift ** 2 - gamma ** 4 / 4.0   # D at f^2 = f0^2 - gamma^2 / 2
    amplitude = peak * minimum
    frequencies = np.linspace(*_DHO_BAND)
    fit = dho_fit(frequencies, _dho(frequencies, shift, gamma, amplitude))
    peak_x = shift ** 2 - gamma ** 2 / 2.0
    fwhm = math.sqrt(peak_x + math.sqrt(minimum)) - math.sqrt(peak_x - math.sqrt(minimum))
    for name, expected in (('shift', shift), ('gamma', gamma), ('peak', peak), ('fwhm', fwhm)):
        checks.add_relative(f'fit_{name}', fit[name], expected, _DHO_RTOL)
    checks.add_bound('fit_r_squared', 1.0 - fit['r_squared'], 1e-12)
    checks.add('fit_no_warning', fit['fit_warning'] == '', value=fit['fit_warning'])
    failed = dho_fit(frequencies, np.zeros_like(frequencies))
    checks.add('zero_spectrum_warns', bool(failed['fit_warning']) and math.isnan(failed['shift']),
               value=failed['fit_warning'])
    linear = 3.0 + 2.0 * frequencies / frequencies[-1]
    exact = (3.0 * (frequencies[-1] - frequencies[0])
             + (frequencies[-1] ** 2 - frequencies[0] ** 2) / frequencies[-1])
    checks.add_relative('band_integral_linear', float(band_integral(frequencies, linear)), exact, _ALGEBRA_RTOL)
    return checks.result()


def _viscous_laws():
    """(longitudinal M* = M + j Omega eta_L, shear G* = j Omega eta_s, eta_L) of the dimensionless medium with Gamma =
    _DAMPING Omega_0 at q0 = 2 pi (Gamma = eta_L q0^2 / rho)."""
    q0 = 2.0 * math.pi
    omega0 = q0 * math.sqrt(_MODULUS / _DENSITY)
    viscosity = _DAMPING * omega0 * _DENSITY / q0 ** 2

    def longitudinal(omega):
        return _MODULUS + 1j * np.asarray(omega) * viscosity

    def shear(omega):
        return 1j * np.asarray(omega) * _SHEAR_VISCOSITY
    return longitudinal, shear, viscosity


def _dho_line(frequencies, weight_sum, viscosity):
    """(k_B T / M) sum |c|^2 dA 2 Gamma Omega_0^2 / ((Omega_0^2 - Omega^2)^2 + Gamma^2 Omega^2) at q0 = 2 pi."""
    q0 = 2.0 * math.pi
    omega = 2.0 * math.pi * np.asarray(frequencies)
    omega0_squared = _MODULUS * q0 ** 2 / _DENSITY
    width = viscosity * q0 ** 2 / _DENSITY
    return (K_B * _TEMPERATURE / _MODULUS * weight_sum * 2.0 * width * omega0_squared
            / ((omega0_squared - omega ** 2) ** 2 + width ** 2 * omega ** 2))


def _check_homogeneous_reference(_):
    from .brillouin import (homogeneous_padding, homogeneous_reference, homogeneous_static, local_reference,
                            local_wavenumbers, longitudinal_function, modulus_values)
    from .viscoelastic import ModulusLaw

    checks = SubChecks()
    rng = np.random.default_rng(73)
    longitudinal, shear, viscosity = _viscous_laws()
    frequencies = np.linspace(*_BAND)
    # (a) a plane wave exactly on one FFT bin (padding 1): the single-q DHO line, and the local reference of it
    n_x, n_y, dx, dy = _SINGLE_GRID
    x = (np.arange(n_x) + 0.5) * dx
    y = (np.arange(n_y) + 0.5) * dy
    amplitude = 0.7 - 0.4j
    dilation = amplitude * np.exp(2j * math.pi * x)[:, None] * np.ones(n_y)[None, :]
    grid = np.stack([dilation, dilation, np.zeros_like(dilation)])
    weight_sum = float(np.sum(np.abs(dilation) ** 2)) * dx * dy
    expected = _dho_line(frequencies, weight_sum, viscosity)
    single = homogeneous_reference(frequencies, grid, dx, dy, longitudinal, shear, _DENSITY, _TEMPERATURE, 1.0,
                                   padding=1)
    checks.add_bound('single_bin_dho', max_relative_error(single, expected), _SINGLE_RTOL)
    q_x, q_y = local_wavenumbers(dilation, x, y)
    q = np.hypot(q_x, q_y).ravel()
    checks.add_bound('plane_wave_wavenumber', max_relative_error(q, np.full(q.shape, 2.0 * math.pi)), 1e-12)
    local = local_reference(frequencies, (np.abs(dilation) ** 2 * dx * dy).ravel(), q, longitudinal, _DENSITY,
                            _TEMPERATURE, 1.0)
    checks.add_bound('local_single_q', max_relative_error(local, expected), _SINGLE_RTOL)
    # (b) a wide Gaussian envelope, zero-padded 2x: the same line within the envelope's q spread
    width_x, width_y, step_x, step_y, extent = _ENVELOPE
    x = np.arange(-extent * width_x, extent * width_x, step_x) + 0.5 * step_x
    y = np.arange(-extent * width_y, extent * width_y, step_y) + 0.5 * step_y
    envelope = (np.exp(2j * math.pi * x - (x / width_x) ** 2)[:, None] * np.exp(-(y / width_y) ** 2)[None, :])
    grid = np.stack([envelope, envelope, np.zeros_like(envelope)])
    weight_sum = float(np.sum(np.abs(envelope) ** 2)) * step_x * step_y
    padded = homogeneous_reference(frequencies, grid, step_x, step_y, longitudinal, shear, _DENSITY, _TEMPERATURE,
                                   1.0)
    checks.add_bound('envelope_dho', norm_relative_error(padded, _dho_line(frequencies, weight_sum, viscosity)),
                     _ENVELOPE_RTOL)
    # (b') a short drive: the automatic padding from the attenuation (its expected factors) holds the two nearest
    # periodic images per axis below IMAGE_TOLERANCE, so twice the padding per axis changes the spectrum by about
    # that; a nearly lossless medium would need a raster above MAX_REFERENCE_CELLS and raises
    n_x, n_y, dx, dy = _SHORT_GRID
    x = (np.arange(n_x) + 0.5) * dx
    short = (0.4 + 0.3j) * np.exp(2j * math.pi * x)[:, None] * np.ones(n_y)[None, :]
    grid = np.stack([short, short, np.zeros_like(short)])
    automatic = homogeneous_padding(frequencies, (n_x, n_y), dx, dy, longitudinal, shear, _DENSITY)
    checks.add('short_drive_padding', tuple(automatic) == _SHORT_PADDING, value=list(automatic))
    reference = homogeneous_reference(frequencies, grid, dx, dy, longitudinal, shear, _DENSITY, _TEMPERATURE, 1.0)
    finer = homogeneous_reference(frequencies, grid, dx, dy, longitudinal, shear, _DENSITY, _TEMPERATURE, 1.0,
                                  padding=tuple(2 * value for value in automatic))
    checks.add_bound('short_drive_images', norm_relative_error(reference, finer), _SHORT_PADDING_RTOL)

    def nearly_lossless(omega):
        return _MODULUS + 1j * np.asarray(omega) * viscosity * 1e-6
    checks.add_raises('padding_above_limit_raises', lambda: homogeneous_padding(
        frequencies, (n_x, n_y), dx, dy, nearly_lossless, shear, _DENSITY))
    # (c) static Parseval: k_B T / M int |c|^2 dA of a random dilation (liquid, no shear modulus); a shear drive
    # without shear modulus raises
    random = _complex(rng, (12, 10))
    grid = np.stack([random, random, np.zeros_like(random)])
    static = homogeneous_static(grid, 0.3, 0.2, 2.2e9, 0.0, _TEMPERATURE, 3.0)
    reference = K_B * _TEMPERATURE / 2.2e9 * float(np.sum(np.abs(random) ** 2)) * 0.3 * 0.2 * 3.0
    checks.add_relative('static_parseval', static, reference, _STATIC_RTOL)
    sheared = np.stack([random, -random, random])
    checks.add_raises('static_shear_without_modulus_raises',
                      lambda: homogeneous_static(sheared, 0.3, 0.2, 2.2e9, 0.0, _TEMPERATURE, 1.0))
    # (d) a law evaluated per distinct omega (a repeat included) is K* + 4/3 G* of viscoelastic.ModulusLaw.at per
    # omega; an incompressible law (ModulusLaw.at = inf) raises
    law = ModulusLaw(relaxed=2.0e9, loss_factor=0.01, viscosity=3e-3, branches=((5e8, 1e-10), (2e8, 3e-9)))
    shear_law = ModulusLaw(relaxed=1.0e9, loss_factor=0.02, viscosity=1e-3, branches=((1e8, 2e-10),))
    omegas = np.array([0.0, 1e6, 4.7e10, 3e12, 1e6])
    combined = modulus_values(longitudinal_function(law.at, shear_law.at), omegas)
    expected = np.array([law.at(w) + 4.0 / 3.0 * shear_law.at(w) for w in omegas])
    checks.add_bound('longitudinal_function', max_relative_error(combined, expected), _LAW_RTOL)
    incompressible = ModulusLaw(relaxed=math.inf, loss_factor=0.0, viscosity=0.0, branches=())
    checks.add_raises('incompressible_law_raises', lambda: modulus_values(incompressible.at, omegas))
    return checks.result()


def _check_bilinear(_):
    from .brillouin import (bilinear_sample, bilinear_weights, edges_to_centres, local_wavenumbers,
                            phase_corrected_sample)

    checks = SubChecks()
    rng = np.random.default_rng(79)
    x = np.cumsum(0.5 + rng.random(7))
    y = np.cumsum(0.5 + rng.random(5))
    a, b, c, d = 0.3 + 0.1j, -1.2 + 0.4j, 0.7 - 0.9j, 0.25 + 0.5j
    values = a + b * x[:, None] + c * y[None, :] + d * x[:, None] * y[None, :]   # bilinear: interpolated exactly
    inside = np.column_stack([x[0] + (x[-1] - x[0]) * rng.random(40), y[0] + (y[-1] - y[0]) * rng.random(40)])
    corners = np.array([[x[0], y[0]], [x[-1], y[-1]], [x[0], y[-1]], [x[-1], y[0]], [x[3], y[2]]])
    points = np.concatenate([inside, corners])
    index, weight = bilinear_weights(x, y, points)
    exact = a + b * points[:, 0] + c * points[:, 1] + d * points[:, 0] * points[:, 1]
    checks.add_bound('bilinear_exact', max_relative_error(bilinear_sample(values, index, weight), exact),
                     _ALGEBRA_RTOL)
    checks.add_bound('weights_sum_one', float(np.max(np.abs(weight.sum(axis=1) - 1.0))), _ALGEBRA_RTOL)
    checks.add_raises('outside_raises', lambda: bilinear_weights(x, y, np.array([[x[-1] + 1e-3, y[0]]])))
    q = (0.8, -0.6)   # 1/m: |q| (x_(i+1) - x_(i-1)) < pi on the 0.5 .. 1.5 spacing (no phase wrap)
    plane = (0.8 + 0.3j) * np.exp(1j * (q[0] * x[:, None] + q[1] * y[None, :]))
    q_x, q_y = local_wavenumbers(plane, x, y)
    checks.add_bound('phase_gradient_exact', max(max_relative_error(q_x, np.full(q_x.shape, q[0])),
                                                 max_relative_error(q_y, np.full(q_y.shape, q[1]))), 1e-12)
    # phase-corrected sampling: exact for a plane wave of <= pi / 4 per cell (here |k| 0.5 on spacings <= 1.5), where
    # plain bilinear loses ~sinc^2(k h / 2); linear in the arrays with a shared step; a real field (steps 0 or pi) is
    # sampled as plain bilinear; edges_to_centres exact for a plane wave along its axis
    k = (0.4, -0.3)
    plane = (0.8 + 0.3j) * np.exp(1j * (k[0] * x[:, None] + k[1] * y[None, :]))
    exact = (0.8 + 0.3j) * np.exp(1j * (k[0] * points[:, 0] + k[1] * points[:, 1]))
    corrected, = phase_corrected_sample([plane], index, weight)
    checks.add_bound('phase_corrected_plane_wave', max_relative_error(corrected, exact), _PHASE_RTOL,
                     plain_bilinear_error=max_relative_error(bilinear_sample(plane, index, weight), exact))
    other = (0.2 - 0.5j) * plane * (1.0 + 0.1 * x[:, None])
    shared = phase_corrected_sample([plane, other, plane + 2.0 * other], index, weight, phase_count=1)
    checks.add_bound('phase_corrected_linear', max_relative_error(shared[2], shared[0] + 2.0 * shared[1]),
                     _ALGEBRA_RTOL)
    real = np.cos(1.3 * x)[:, None] * np.cos(0.7 * y)[None, :]
    checks.add_bound('phase_corrected_real_is_bilinear',
                     max_relative_error(phase_corrected_sample([real], index, weight)[0],
                                        bilinear_sample(real, index, weight)), _ALGEBRA_RTOL)
    edges = np.concatenate([[x[0] - 0.7], 0.5 * (x[:-1] + x[1:]), [x[-1] + 0.6]])
    centres = 0.5 * (edges[:-1] + edges[1:])
    on_edges = np.exp(1j * k[0] * edges)[:, None] * np.exp(1j * k[1] * y)[None, :]
    checks.add_bound('edges_to_centres_plane_wave',
                     max_relative_error(edges_to_centres(on_edges, 0),
                                        np.exp(1j * k[0] * centres)[:, None] * np.exp(1j * k[1] * y)[None, :]),
                     _PHASE_RTOL)
    return checks.result()


def _case(name, label, description, check, wall=60.0):
    return SelfTestCase(name=name, label=label, description=description + ' error = number of failed sub-checks, '
                        'tolerance 0 (first estimates, not yet run).', specs=no_runs, check=check, max_wall_time=wall)


BRILLOUIN_DRIVE_CASES = (
    _case('brillouin_field_products', 'Brillouin: field products and optical stress',
          'CPU check of brillouin.field_products / optical_stress against the 3D four-index contraction sum_ij '
          'Delta eps_ij E_c,i E_i,j, Delta eps_ij = -eps_i eps_j p_ijkl S_kl, for random complex fields, diagonal eps '
          'and a random symmetric strain: isotropic p (explicit p44 and the default (p11 - p12) / 2), a rotated cubic '
          'p and a general (I, J)-unsymmetric p (pins sigma~ = p^T G) as crystal regions (1e-13); the E_z-only '
          'dilation rows; an unknown component raises; the non-conjugated Johnson value with E_c = E_i* equals the '
          'optomechanics_fem moving-boundary bracket (1e-13).', _check_field_products),
    _case('brillouin_azimuthal_split', 'Brillouin: azimuthal m = 0 / 2 split of two m = 1 fields',
          'CPU check of brillouin.azimuthal_parts / azimuthal_moving_boundary: for random m = 1 standing waves (co- '
          'and cross-polarised collection), diagonal cylindrical eps and isotropic p, ring_angle(m) x part . S equals '
          'the 64-point phi quadrature of the 3D work sigma~(phi) . S(phi) for m = 0, m = 2 even (normal cos, shear '
          'sin) and m = 2 odd (normal sin, shear -cos: the rotated even solve), the parts not returned integrate to 0, '
          'the torsion fraction matches the quadrature (0 co-polarised), the moving-boundary harmonics equal the phi '
          'projections (1e-13); a low-NA x-polarised field with p11 = p12, p44 = 0 has no m = 2 part.',
          _check_azimuthal_split),
    _case('brillouin_fdt_oscillator', 'Brillouin: FDT readout of a 3-dof oscillator',
          'CPU check of brillouin.fdt_psd: K (1 + j eta) - Omega^2 M (eta 1e-3) solved densely per frequency vs the '
          'modal closed form (2 k_B T / Omega) sum |phi^T w|^2 lambda eta / |lambda (1 + j eta) - Omega^2|^2 x scale '
          '(1e-12); the sum rule 2 int_0^inf psd df = k_B T w^H K^-1 w with Kelvin-Voigt damping C = 2e-3 K on a '
          'fine grid (1e-4; a constant loss factor has a log-divergent 1 / f tail, so it is not used here); an active '
          'operator (1 - j eta) raises; quantum sideband factors x (n + 1) - x n = x and their ratio e^x (1e-12).',
          _check_fdt_oscillator),
    _case('brillouin_dho_fit', 'Brillouin: damped-oscillator fit round trip',
          'CPU check of brillouin.dho_fit on an exact DHO spectrum (f0 7.439 GHz, gamma 0.65 GHz, 61 points over '
          '6.5-8.3 GHz): shift, gamma, peak and the exact half-maximum width within 1e-8, r_squared 1 - 1e-12, no '
          'warning; an all-zero spectrum warns and does not raise; band_integral exact for a linear function.',
          _check_dho_fit),
    _case('brillouin_homogeneous_reference', 'Brillouin: homogeneous and local reference spectra',
          'CPU check of brillouin.homogeneous_reference / local_reference in a dimensionless viscous medium (rho 1, '
          'M 1, Gamma = 0.2 Omega_0 at q0 = 2 pi): a dilation plane wave on one FFT bin (padding 1) gives the single-q '
          'DHO line (k_B T / M) int |c|^2 dA 2 Gamma Omega_0^2 / ((Omega_0^2 - Omega^2)^2 + Gamma^2 Omega^2) and so '
          'does the local reference of its phase-gradient q (1e-10); a wide Gaussian envelope (80 x 20, automatic '
          'padding 2) the same line within 2e-3 (L2; the envelope q spread); a short 2 x 2 m drive takes the automatic '
          'padding 18 x 18 (ceil(1 + ln(2 / 1e-3) / (delta L)), delta = |Im k| at the lowest band frequency: the two '
          'nearest images per axis) and differs from twice that padding by < 3e-3 (L2: the periodic images below '
          '1e-3), a nearly lossless medium raises '
          '(padded raster above the limit); the static '
          'sum equals k_B T / M int |c|^2 dA '
          '(Parseval, 1e-10), a shear drive without shear modulus raises; a law evaluated per distinct frequency '
          'gives K* + 4/3 G* of viscoelastic.ModulusLaw.at (1e-14), an incompressible law raises.',
          _check_homogeneous_reference),
    _case('brillouin_bilinear', 'Brillouin: bilinear and phase-corrected sampling, local wavenumbers',
          'CPU check of brillouin.bilinear_weights / bilinear_sample on non-uniform centres: exact for a bilinear '
          'complex field at random and corner points (1e-13), weights sum to 1, a point outside the centres raises; '
          'local_wavenumbers exact for a plane wave on the non-uniform grid (1e-12); phase_corrected_sample exact for '
          'a plane wave of <= pi / 4 per cell (1e-12; plain bilinear loses ~sinc^2(k h / 2), recorded), linear in its '
          'arrays with a shared phase step and equal to plain bilinear for a real field (1e-13); edges_to_centres '
          'exact for a plane wave (1e-12).', _check_bilinear),
)
