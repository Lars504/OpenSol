"""Spontaneous Brillouin scattering by the fluctuation-dissipation theorem (host numpy, scipy inside functions): field
products and the optical stress of non-conjugated incident x collection fields, the azimuthal m = 0 / 2 split of two
m = 1 fields, the scattering prefactor, the FDT readout, quantum sideband factors, the damped-oscillator fit, the
local and homogeneous reference spectra and the result dict."""
# Conventions (help texts of brillouin_scattering cite this header). exp(+j omega t) peak phasors; omega = the optical,
# Omega = 2 pi f the phonon angular frequency (f = the frequency_response frequency = the Brillouin shift).
# Amplitude (Lorentz reciprocity): a = kappa int sum_ij Delta eps_ij E_c,i E_i,j dV, kappa = omega eps0 / (4 sqrt(P_i
# P_c)) (amplitude_prefactor), NON-conjugated: E_c is the reciprocal field launched from the detector, so for beams
# ~ exp(-j n k0 z) E_c E_i ~ exp(-2 j n k0 z) phase-matches the backscattering phonon q = 2 n k0; |a|^2 = the power
# fraction scattered into the collected channel (planar 2D: P in W/m, per unit length).
# Photoelastic: Delta eps_ij = -eps_ii eps_jj Delta(1/eps)_ij (diagonal eps: the staircase cell eps of the optics), Nye
# Delta(1/eps)_I = p_IJ S_J, Voigt I = (xx, yy, zz, yz, xz, xy), engineering shear S_4..6 = 2 S_ij (optomechanics.py).
# Then sum_ij Delta eps_ij E_c,i E_i,j = G^T p S with G_I = -eps_ii eps_jj (E_c,i E_i,j + [i != j] E_c,j E_i,i)
# (field_products), so a = kappa int sigma~^T S dV with the optical stress sigma~ = p^T G (optical_stress: isotropic
# rows p11 / p12 / p44, unset p44 = (p11 - p12) / 2; a crystal region applies its (6, 6) p TRANSPOSED: a Voigt p need
# not be symmetric in (I, J), e.g. p14 != p41 in trigonal crystals). sigma~_I are tensor components (the engineering
# factor sits on S). FEM: w = kappa int B^T sigma~ dV (fem.elastic_drives), a = w^T u for every u; plane strain pairs the rows
# (xx, yy, xy) only (planar_rows: S_zz = S_xz = S_yz = 0). em_fdfd_ez: G_zz = -eps^2 E_cz E_iz, a pure dilation drive.
# Moving boundary (Johnson, non-conjugated): per interface point, unit normal n out of side 0, Delta = side 0 - side 1,
# f = Delta eps (E_c,par . E_i,par) - Delta(1/eps) (eps E_c,n)(eps E_i,n), each product the mean of the two sides'
# products (optomechanics_fem convention, |E|^2 -> E_c E_i); a_mb = kappa int (u . n) f ds, i.e. the normal traction
# t = kappa n f (moving_boundary_traction). Total w = w_pe + w_mb.
# FDT (Callen-Welton / Levin, classical): the solid's K(Omega) (K (1 + j eta) + viscoelastic terms - Omega^2 M) is
# complex symmetric with Im >= 0 (passive for exp(+j Omega t)); u = K(Omega)^-1 w (the drive the only load, held
# values 0). The two-sided matrix FDT S_uu = (2 k_B T / Omega)(-Im chi), chi = K^-1 (Re / Im chi real symmetric) gives
# S_aa = w^T S_uu w* = (2 k_B T / Omega)(-Im(w^H u)) (fdt_psd) x mirror_factor / L_axial [1/Hz]: the classical power
# fraction per Hz scattered into the collected channel in ONE sideband (Stokes omega - Omega, anti-Stokes omega +
# Omega, classically equal); the one-sided Levin spectrum of both sidebands is 2 x psd. A single oscillator k (1 + j
# eta): -Im chi = k eta / |K|^2. Readouts with -Im(w^H u) < -ACTIVE_TOLERANCE |w^H u| raise (an active operator).
# Sum rule: 2 int_0^inf psd df = k_B T w^H K0^-1 w (static compliance, loss independent; a constant loss factor has a
# log-divergent 1 / Omega tail at low f, so finite bands carry a small eta ln share); an isotropic liquid with a
# dilation drive c: (k_B T / K_S) int |c|^2 dV (Landau-Placzek). Quantum sidebands (sideband_factors): x = hbar Omega
# / (k_B T), n = 1 / (e^x - 1), psd_stokes = psd x (n + 1), psd_anti_stokes = psd x n.
# Axisymmetric (em_fdfd_bor m = 1 standing waves; azimuthal_parts): E = (A_r cos phi, A_phi sin phi, A_z cos phi), A_phi
# = u = -j E_phi of the solve (x-polarised); the cross-polarised collection is the same field rotated by +90 deg,
# (A_r sin phi, -A_phi cos phi, A_z sin phi). Each product of two components (a cos + b sin)(a' cos + b' sin) is
# (a a' + b b') / 2 + (a a' - b b') / 2 cos 2 phi + (a b' + b a') / 2 sin 2 phi; the optics' Voigt axes map x -> r,
# y -> z, z -> phi (CYLINDRICAL_COMPONENTS; eps diagonal (rr, zz, phiphi), isotropic p: rotationally invariant, no
# crystals). Solid rows (fem.elasticity): m = 0 (rr, zz, tt, g_rz) [torsion free], m >= 1 (rr, zz, tt, g_rz) ~ cos(m
# phi), (g_zt, g_rt) ~ sin(m phi). Parts: m = 0 even = the constant normal rows (the constant shear rows are the m = 0
# torsion, not modelled: dropped, share torsion_drive_fraction); m = 2 even = (cos 2 phi normal rows, sin 2 phi shear
# rows); m = 2 odd (normal ~ sin, shear ~ cos) = the even solve rotated by pi / (2 m): rows (sin normal, -cos shear) in
# the even convention. Co-polarised: (0, even) + (2, even); cross-polarised: (2, odd) (+ the dropped torsion). Weights
# ring_angle(m) int B_m^T sigma~ r dA (2 pi at m = 0, pi at m >= 1: fem.elastic_drives); psd = the sum of every part's
# FDT (even and odd decouple). Low NA: E . E is phi independent, so an isotropic dilation drive has no m = 2 part.
# Normalisation: planar 2D fields of P' = 1 W/m give S_3D = S_2D / L for a sheet beam of length L >> the acoustic
# attenuation length (q_z = 0 phonons); every model mirror plane doubles the drawn half (scale = mirror_factor / L or
# mirror_factor).
# References: local (local_reference): per point the longitudinal projection sigma_L = q^T sigma~ q / |q|^2 of the
# local wavevector q = Im(P* grad P) / |P|^2, P = E_c . E_i (local_wavenumbers), and the DHO line psd_ref = sum (k_B T
# / M_L) |kappa sigma_L|^2 dV 2 Gamma Omega_0^2 / ((Omega_0^2 - Omega^2)^2 + Gamma^2 Omega^2), M_L = Re M*(Omega_0),
# M* = K* + 4/3 G*, Omega_0 = |q| sqrt(M_L / rho), Gamma = q^2 Im M*(Omega_0) / (rho Omega_0) (normalised: 2 int_0^inf
# = (k_B T / M_L) sum |kappa sigma_L|^2 dV). Homogeneous (homogeneous_reference, planar, one uniform isotropic medium):
# the exact plane-wave sum of an unbounded medium, S = (2 k_B T / Omega) int d^2q / (2 pi)^2 (-Im[q^2 |s_L|^2 / (M* q^2
# - rho Omega^2) + q^2 |s_T|^2 / (G* q^2 - rho Omega^2)]), s_L = q^T sigma^(q) q / q^2, s_T = t^T sigma^(q) q / q, with
# sigma^ = dx dy FFT of the zero-padded raster (complex: a full fft2 per row) and d^2q / (2 pi)^2 = 1 / (P_x P_y dx dy).
# Fit (dho_fit): S = A / ((f0^2 - f^2)^2 + gamma^2 f^2) by scipy least_squares from a peak / half-maximum start; shift =
# f0, fwhm = the exact half-maximum width of the fitted curve; failures set fit_warning, never raise.
import math
from dataclasses import dataclass, field

import numpy as np

from ...core.constants import EPS0, HBAR, K_B
from ...core.metric import ring_angle
from .optomechanics import inverse_permittivity_changes, shear_photoelastic_coefficient

DETECTIONS = ('co_polarized', 'cross_polarized')   # brillouin_scattering setting detection (cross: em_fdfd_bor only)
REFERENCES = ('local', 'homogeneous', 'none')       # brillouin_scattering setting reference
PARITIES = ('even', 'odd')
AZIMUTHAL_PARTS = {'co_polarized': ((0, 'even'), (2, 'even')), 'cross_polarized': ((2, 'odd'),)}   # (m, parity)
PLANAR_COMPONENTS = ('x', 'y', 'z')                # field component names of the planar optics (in-plane x, y)
CYLINDRICAL_COMPONENTS = ('r', 'z', 'phi')         # BOR amplitudes on the optical Voigt axes x -> r, y -> z, z -> phi
VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))   # Voigt I = xx, yy, zz, yz, xz, xy
# optical Voigt rows of the solid's strain rows (fem.elasticity): planar (xx, yy, g); axisymmetric m = 0 (rr, zz, tt,
# g_rz); m >= 1 (rr, zz, tt, g_rz, g_zt, g_rt) under x -> r, y -> z, z -> phi
SOLID_VOIGT_ROWS = {'planar': (0, 1, 5), 'axisymmetric': (0, 1, 2, 5), 'azimuthal': (0, 1, 2, 5, 3, 4)}
_NORMAL_ROWS = SOLID_VOIGT_ROWS['axisymmetric']    # rr, zz, phiphi, rz: cos(m phi) parts of the even solid
_SHEAR_ROWS = (3, 4)                               # z-phi, r-phi: sin(m phi) parts (constant: the m = 0 torsion)
ACTIVE_TOLERANCE = 1e-9    # fdt_psd: -Im(w^H u) below -this x |w^H u| raises (non-passive); above, negatives read 0
# result names (brillouin_result): series per position (suffix _<k> with several positions), fit extras
PSD_SERIES = 'psd'
STOKES_SERIES = 'psd_stokes'
ANTI_STOKES_SERIES = 'psd_anti_stokes'
REFERENCE_SERIES = 'psd_reference'
DB_SERIES = 'psd_db'
SERIES_UNITS = {PSD_SERIES: '1/Hz', STOKES_SERIES: '1/Hz', ANTI_STOKES_SERIES: '1/Hz', REFERENCE_SERIES: '1/Hz',
                DB_SERIES: 'dB/Hz'}
FIT_EXTRAS = ('shift', 'fwhm', 'gamma', 'peak', 'r_squared')   # dho_fit keys (+ 'fit_warning')
SCAN_COLUMNS = ('position', 'offset_x', 'offset_y', 'shift', 'fwhm', 'peak', 'scattered_fraction')
SCAN_UNITS = ('1', 'm', 'm', 'Hz', 'Hz', '1/Hz', '1')
X_LABEL, Y_LABEL = 'Frequency', 'Scattered spectral density'
_RESONANCE_ITERATIONS = 100   # local_reference: fixed point Omega_0 = |q| sqrt(Re M*(Omega_0) / rho)
_RESONANCE_RTOL = 1e-13
_CHUNK_VALUES = 1 << 21       # (frequency x point / wavevector) values per vectorised block of the references
_HALF_MAXIMUM_MIN_POINTS = 5  # dho_fit: fewer frequencies are not fitted
# m = 1 standing-wave patterns (cos phi, sin phi) of the amplitudes A_r, A_phi, A_z (header)
_INCIDENT_PATTERN = {'r': (1.0, 0.0), 'phi': (0.0, 1.0), 'z': (1.0, 0.0)}
_COLLECTION_PATTERNS = {'co_polarized': _INCIDENT_PATTERN,
                        'cross_polarized': {'r': (0.0, 1.0), 'phi': (-1.0, 0.0), 'z': (0.0, 1.0)}}
_HARMONICS = ('constant', 'cos2', 'sin2')


# ---- scan positions, prefactor and field products ----
def scan_positions(offsets_x=(), offsets_y=()) -> np.ndarray:
    """(n, 2) focus offsets [m] of a scan: the Cartesian product of the x and y offset lists (an empty list = [0]), x
    outer, y inner, so position 1 = (first x, first y) and position k is row k - 1 (the _<k> result suffix)."""
    xs = [float(value) for value in offsets_x] or [0.0]
    ys = [float(value) for value in offsets_y] or [0.0]
    positions = np.array([(x, y) for x in xs for y in ys], dtype=np.float64)
    if not np.isfinite(positions).all():
        raise ValueError('scan_positions: offsets must be finite numbers [m]')
    return positions


def amplitude_prefactor(omega, incident_power, collection_power) -> float:
    """kappa = omega eps0 / (4 sqrt(P_i P_c)) [header units] of the optical angular frequency [rad/s] and the incident
    / collection powers [W, planar W/m] (> 0: the focused beams' incident_power)."""
    omega, incident_power, collection_power = float(omega), float(incident_power), float(collection_power)
    if not (omega > 0.0 and math.isfinite(omega)):
        raise ValueError(f'amplitude_prefactor: the optical angular frequency must be > 0, got {omega!r}')
    if not (incident_power > 0.0 and collection_power > 0.0
            and math.isfinite(incident_power) and math.isfinite(collection_power)):
        raise ValueError(f'amplitude_prefactor: incident and collection powers must be finite and > 0, got '
                         f'{incident_power!r} and {collection_power!r}')
    return omega * EPS0 / (4.0 * math.sqrt(incident_power * collection_power))


def _field_components(fields, names, what) -> list:
    """[array or None per name] of a {component: values} dict (missing = 0); ValueError for an unknown component."""
    unknown = [str(name) for name in fields if name not in names]
    if unknown:
        raise ValueError(f"{what}: unknown field components {', '.join(unknown)} (components are "
                         f"{', '.join(names)})")
    return [None if fields.get(name) is None else np.asarray(fields[name]) for name in names]


def _diagonal_eps(eps, name):
    """eps_{name name} of a {'xx' / 'rr' / 'phiphi' ...: values} dict, or the value itself (isotropic)."""
    if not isinstance(eps, dict):
        return np.asarray(eps)
    key = name + name
    if key not in eps:
        raise ValueError(f"field products: eps has no component {key!r} (given: {', '.join(map(str, eps))})")
    return np.asarray(eps[key])


def _voigt_rows(product, eps, names, shape) -> np.ndarray:
    """Complex (6, *shape) G_I = -eps_ii eps_jj (P_ij + [i != j] P_ji) of products product(i, j) (array or None = 0)."""
    rows = np.zeros((6,) + tuple(shape), dtype=np.complex128)
    for row, (i, j) in enumerate(VOIGT_PAIRS):
        parts = [part for part in (product(i, j), product(j, i) if i != j else None) if part is not None]
        if parts:
            rows[row] = -_diagonal_eps(eps, names[i]) * _diagonal_eps(eps, names[j]) * sum(parts)
    return rows


def _shape(*field_lists) -> tuple:
    present = [value for values in field_lists for value in values if value is not None]
    if not present:
        raise ValueError('field products: the incident and collection fields hold no component')
    return np.broadcast_shapes(*(value.shape for value in present))


def field_products(incident, collection, eps, components=PLANAR_COMPONENTS) -> np.ndarray:
    """Complex (6, n) Voigt field products G (header) of {component: (n,)} incident and collection fields (missing
    components 0; PLANAR_COMPONENTS, or CYLINDRICAL_COMPONENTS for (r, z, phi) fields) and the diagonal eps (a dict
    {'xx': ...} / {'rr': ...}, or one isotropic value)."""
    names = tuple(components)
    incident_values = _field_components(incident, names, 'field_products incident')
    collection_values = _field_components(collection, names, 'field_products collection')
    shape = _shape(incident_values, collection_values)

    def product(i, j):
        first, second = collection_values[i], incident_values[j]
        return None if first is None or second is None else first * second
    return _voigt_rows(product, eps, names, shape)


def optical_stress(products, p11, p12, p44=None, crystal_regions=()) -> np.ndarray:
    """Complex (6, n) optical stress sigma~ = p^T G of Voigt field products G (6, n): isotropic p11 / p12 / p44 per
    point or scalars (p44 None / NaN = (p11 - p12) / 2: optomechanics.shear_photoelastic_coefficient), and crystal
    regions ((bool mask (n,), (6, 6) Voigt p), ...) whose transposed p replaces the isotropic rows there."""
    products = np.asarray(products)
    if products.shape[:1] != (6,):
        raise ValueError(f'optical_stress: products must have 6 Voigt rows, got shape {products.shape}')
    transposed = tuple((np.asarray(mask, dtype=bool), np.asarray(tensor, dtype=np.float64).T)
                       for mask, tensor in crystal_regions)
    for mask, tensor in transposed:
        if mask.shape != products.shape[1:] or tensor.shape != (6, 6):
            raise ValueError(f'optical_stress: a crystal region needs a mask of shape {products.shape[1:]} and a '
                             f'(6, 6) p, got {mask.shape} and {tensor.shape}')
    return inverse_permittivity_changes(tuple(products), p11, p12, shear_photoelastic_coefficient(p11, p12, p44),
                                        transposed)


def solid_rows(stress, *, axisymmetric=False, azimuthal_order=0) -> np.ndarray:
    """The rows (k, n) of an optical stress (6, n) that the solid's strain rows pair with (SOLID_VOIGT_ROWS: planar 3,
    axisymmetric m = 0 4, m >= 1 6; an azimuthal stress of azimuthal_parts is already in the solid rows)."""
    if azimuthal_order:
        if not axisymmetric:
            raise ValueError('solid_rows: an azimuthal order needs the axisymmetric form')
        key = 'azimuthal'
    else:
        key = 'axisymmetric' if axisymmetric else 'planar'
    return np.asarray(stress)[list(SOLID_VOIGT_ROWS[key])]


def planar_rows(stress) -> np.ndarray:
    """(3, n) plane-strain rows (xx, yy, xy) of an optical stress (6, n) (S_zz = S_xz = S_yz = 0)."""
    return solid_rows(stress)


# ---- axisymmetric split of two m = 1 standing waves ----
def _harmonic_coefficient(first, second, harmonic) -> float:
    """Coefficient of 1, cos 2 phi or sin 2 phi in (a cos + b sin)(a' cos + b' sin) (header)."""
    (a, b), (c, d) = first, second
    if harmonic == 'constant':
        return 0.5 * (a * c + b * d)
    if harmonic == 'cos2':
        return 0.5 * (a * c - b * d)
    return 0.5 * (a * d + b * c)


def _checked_detection(detection) -> str:
    if detection not in DETECTIONS:
        raise ValueError(f"detection must be one of {', '.join(DETECTIONS)}, got {detection!r}")
    return detection


def azimuthal_products(incident, collection, eps, detection) -> dict:
    """{'constant' / 'cos2' / 'sin2': complex (6, n) Voigt field products} of the phi harmonics 1, cos 2 phi, sin 2 phi
    of two m = 1 standing waves {'r', 'phi', 'z': amplitudes (n,)} (header: A_phi = -j E_phi; the collection rotated
    by +90 deg when cross_polarized), eps {'rr', 'zz', 'phiphi'} or isotropic, on the Voigt axes x -> r, y -> z,
    z -> phi."""
    names = CYLINDRICAL_COMPONENTS
    collection_pattern = _COLLECTION_PATTERNS[_checked_detection(detection)]
    incident_values = _field_components(incident, names, 'azimuthal_products incident')
    collection_values = _field_components(collection, names, 'azimuthal_products collection')
    shape = _shape(incident_values, collection_values)
    harmonics = {}
    for harmonic in _HARMONICS:
        def product(i, j, harmonic=harmonic):
            first, second = collection_values[i], incident_values[j]
            if first is None or second is None:
                return None
            coefficient = _harmonic_coefficient(collection_pattern[names[i]], _INCIDENT_PATTERN[names[j]], harmonic)
            return None if coefficient == 0.0 else coefficient * first * second
        harmonics[harmonic] = _voigt_rows(product, eps, names, shape)
    return harmonics


def azimuthal_parts(incident, collection, eps, detection, p11, p12, p44=None, weights=None) -> tuple:
    """({(m, parity): complex optical stress in the solid rows of order m (4 rows at m = 0, 6 at m = 2; odd parts in
    the even convention)}, torsion_drive_fraction) of two m = 1 standing waves (azimuthal_products arguments; isotropic
    p11 / p12 / p44 per point or scalars). Parts per detection: AZIMUTHAL_PARTS. torsion_drive_fraction = the dropped m
    = 0 torsion share of sum ring_angle(m) w |sigma|^2 over the parts and the torsion (weights (n,) >= 0, None = 1)."""
    harmonics = {name: optical_stress(products, p11, p12, p44)
                 for name, products in azimuthal_products(incident, collection, eps, detection).items()}
    normal, shear = list(_NORMAL_ROWS), list(_SHEAR_ROWS)
    available = {(0, 'even'): harmonics['constant'][normal],
                 (2, 'even'): np.concatenate([harmonics['cos2'][normal], harmonics['sin2'][shear]]),
                 (2, 'odd'): np.concatenate([harmonics['sin2'][normal], -harmonics['cos2'][shear]])}
    parts = {key: available[key] for key in AZIMUTHAL_PARTS[detection]}
    torsion = harmonics['constant'][shear]
    weights = np.ones(torsion.shape[1:]) if weights is None else np.asarray(weights, dtype=np.float64)
    if weights.shape != torsion.shape[1:] or (weights < 0.0).any():
        raise ValueError(f'azimuthal_parts: weights must be >= 0 with shape {torsion.shape[1:]}, got {weights.shape}')

    def measure(rows, order):
        return ring_angle(order) * float(np.sum(weights * np.sum(np.abs(rows) ** 2, axis=0)))
    dropped = measure(torsion, 0)
    total = dropped + sum(measure(rows, order) for (order, _), rows in parts.items())
    return parts, (dropped / total if total > 0.0 else 0.0)


# ---- moving boundary ----
def _side_forms(normals, collection, incident, names, coefficient) -> tuple:
    """(E_c,par . E_i,par, E_c,n E_i,n) of one side: normals (..., 2) on the in-plane axes names[:2], names[2] the
    out-of-plane component; coefficient(i, j) scales the product of collection component i and incident component j."""
    normal_a, normal_b = normals[..., 0], normals[..., 1]
    collection_values = _field_components(collection, names, 'moving boundary collection')
    incident_values = _field_components(incident, names, 'moving boundary incident')

    def product(i, j):
        first, second = collection_values[i], incident_values[j]
        if first is None or second is None:
            return 0.0
        return coefficient(i, j) * first * second
    mixed = product(0, 1) + product(1, 0)
    normal = normal_a * normal_a * product(0, 0) + normal_a * normal_b * mixed + normal_b * normal_b * product(1, 1)
    tangential = normal_b * normal_b * product(0, 0) - normal_a * normal_b * mixed + normal_a * normal_a * product(1, 1)
    return tangential + product(2, 2), normal


def _johnson_value(normals, incident_sides, collection_sides, eps_sides, names, coefficient) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float64)
    if normals.shape[-1:] != (2,):
        raise ValueError(f'moving boundary: normals must end in 2 in-plane components, got shape {normals.shape}')
    eps_0, eps_1 = (np.asarray(value, dtype=np.complex128) for value in eps_sides)
    parallel_0, normal_0 = _side_forms(normals, collection_sides[0], incident_sides[0], names, coefficient)
    parallel_1, normal_1 = _side_forms(normals, collection_sides[1], incident_sides[1], names, coefficient)
    parallel = 0.5 * (parallel_0 + parallel_1)
    displacement = 0.5 * (eps_0 * eps_0 * normal_0 + eps_1 * eps_1 * normal_1)
    value = (eps_0 - eps_1) * parallel - (1.0 / eps_0 - 1.0 / eps_1) * displacement
    return np.array(np.broadcast_to(value, np.broadcast_shapes(np.shape(value), normals.shape[:-1])),
                    dtype=np.complex128)


def moving_boundary_traction(normals, incident_sides, collection_sides, eps_sides,
                             components=PLANAR_COMPONENTS) -> np.ndarray:
    """Complex (...) Johnson value f = Delta eps (E_c,par . E_i,par) - Delta(1/eps)(eps E_c,n)(eps E_i,n) per
    interface point (header; the drive's normal traction is kappa n f): unit normals (..., 2) out of side 0 on the
    in-plane axes, incident / collection sides = (side 0 fields, side 1 fields) as {component: (...)} dicts,
    eps_sides = (isotropic eps of side 0, of side 1)."""
    return _johnson_value(normals, incident_sides, collection_sides, eps_sides, tuple(components), lambda i, j: 1.0)


def azimuthal_moving_boundary(normals, incident_sides, collection_sides, eps_sides, detection) -> dict:
    """{(m, parity): complex (...) Johnson value of that phi harmonic} (parts as azimuthal_parts: m = 0 even = the
    constant, m = 2 even = cos 2 phi, m = 2 odd = sin 2 phi; a normal traction has no torsion part) of two m = 1
    standing waves sampled on both sides ({'r', 'phi', 'z'} dicts), normals (n_r, n_z) out of side 0."""
    names = CYLINDRICAL_COMPONENTS
    collection_pattern = _COLLECTION_PATTERNS[_checked_detection(detection)]
    harmonic_of = {(0, 'even'): 'constant', (2, 'even'): 'cos2', (2, 'odd'): 'sin2'}
    parts = {}
    for key in AZIMUTHAL_PARTS[detection]:
        harmonic = harmonic_of[key]

        def coefficient(i, j, harmonic=harmonic):
            return _harmonic_coefficient(collection_pattern[names[i]], _INCIDENT_PATTERN[names[j]], harmonic)
        parts[key] = _johnson_value(normals, incident_sides, collection_sides, eps_sides, names, coefficient)
    return parts


# ---- sampling the optical grid ----
def _centres(values, what) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not (np.diff(values) > 0.0).all():
        raise ValueError(f'{what} must be at least 2 strictly increasing cell centres [m]')
    return values


def bilinear_weights(x_centres, y_centres, points) -> tuple:
    """(flat indices (n, 4) into a C-ordered (n_x, n_y) cell array, weights (n, 4)) of the bilinear interpolation
    between cell centres at points (..., 2) [m] (the window frame: lower corner 0 on both meshes); ValueError for a
    point outside the centres' hull (fail loudly: the drive selection must stay inside the optical window)."""
    x = _centres(x_centres, 'bilinear_weights x centres')
    y = _centres(y_centres, 'bilinear_weights y centres')
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    px, py = points[:, 0], points[:, 1]
    slack_x, slack_y = 1e-12 * (x[-1] - x[0]), 1e-12 * (y[-1] - y[0])
    outside = (px < x[0] - slack_x) | (px > x[-1] + slack_x) | (py < y[0] - slack_y) | (py > y[-1] + slack_y)
    if outside.any():
        first = int(np.argmax(outside))
        raise ValueError(f'{int(outside.sum())} drive points lie outside the optical sample window [{x[0]:.6g}, '
                         f'{x[-1]:.6g}] x [{y[0]:.6g}, {y[-1]:.6g}] m (first at ({px[first]:.6g}, {py[first]:.6g}) '
                         'm): keep the drive selection inside the optical window, away from its PML')
    i = np.clip(np.searchsorted(x, px, side='right') - 1, 0, len(x) - 2)
    j = np.clip(np.searchsorted(y, py, side='right') - 1, 0, len(y) - 2)
    tx = np.clip((px - x[i]) / (x[i + 1] - x[i]), 0.0, 1.0)
    ty = np.clip((py - y[j]) / (y[j + 1] - y[j]), 0.0, 1.0)
    n_y = len(y)
    index = np.stack([i * n_y + j, (i + 1) * n_y + j, i * n_y + j + 1, (i + 1) * n_y + j + 1], axis=1)
    weight = np.stack([(1.0 - tx) * (1.0 - ty), tx * (1.0 - ty), (1.0 - tx) * ty, tx * ty], axis=1)
    return index.astype(np.int64), weight


def bilinear_sample(values, index, weight) -> np.ndarray:
    """(n,) samples of a cell array (n_x, n_y) (real or complex) by bilinear_weights' (index, weight)."""
    flat = np.asarray(values).reshape(-1)
    return np.einsum('nk,nk->n', flat[index], weight)


def _phase_gradient(product, centres, axis) -> np.ndarray:
    """d arg(P) / d x along one axis: central phase differences arg(P_(i+1) P*_(i-1)) / (x_(i+1) - x_(i-1)), one-sided
    at the ends (exact for a plane wave with |q dx| < pi / 2; 0 next to P = 0)."""
    moved = np.moveaxis(product, axis, 0)
    gradient = np.empty(moved.shape)
    gradient[1:-1] = np.angle(moved[2:] * np.conj(moved[:-2])) / (centres[2:] - centres[:-2]).reshape(
        (-1,) + (1,) * (moved.ndim - 1))
    gradient[0] = np.angle(moved[1] * np.conj(moved[0])) / (centres[1] - centres[0])
    gradient[-1] = np.angle(moved[-1] * np.conj(moved[-2])) / (centres[-1] - centres[-2])
    return np.moveaxis(gradient, 0, axis)


def local_wavenumbers(product, x_centres, y_centres) -> tuple:
    """(q_x, q_y) (n_x, n_y) [1/m] local wavevector = grad arg(P) (= Im(P* grad P) / |P|^2) of a complex field product
    P = E_c . E_i on the cell centres, by central phase differences (exact for a plane wave; the derivative form
    Im(P* grad P) / |P|^2 of central differences reads sin(q dx) / dx, 11 % low at 7.5 cells per period of P); 0 where
    P = 0 (no drive there)."""
    product = np.asarray(product, dtype=np.complex128)
    x = _centres(x_centres, 'local_wavenumbers x centres')
    y = _centres(y_centres, 'local_wavenumbers y centres')
    if product.shape != (len(x), len(y)):
        raise ValueError(f'local_wavenumbers: product has shape {product.shape}, expected ({len(x)}, {len(y)})')
    present = product != 0.0
    return (np.where(present, _phase_gradient(product, x, 0), 0.0),
            np.where(present, _phase_gradient(product, y, 1), 0.0))


# ---- FDT readout and sidebands ----
def _temperature(temperature) -> float:
    temperature = float(temperature)
    if not (temperature > 0.0 and math.isfinite(temperature)):
        raise ValueError(f'temperature must be finite and > 0 K, got {temperature!r}')
    return temperature


def fdt_psd(omega, readouts, temperature, scale) -> np.ndarray:
    """psd = (2 k_B T / Omega)(-Im readouts) x scale [1/Hz per sideband] of readouts w^H u (complex, leading axis
    along omega [rad/s] > 0; header); ValueError on an active readout (-Im < -ACTIVE_TOLERANCE |w^H u|), values just
    below 0 within it read 0."""
    temperature = _temperature(temperature)
    readouts = np.asarray(readouts, dtype=np.complex128)
    omega = np.asarray(omega, dtype=np.float64)
    if not (omega > 0.0).all() or not np.isfinite(omega).all():
        raise ValueError('fdt_psd: every phonon angular frequency must be finite and > 0')
    omega = omega.reshape(omega.shape + (1,) * max(readouts.ndim - omega.ndim, 0))
    dissipation = -readouts.imag
    active = dissipation < -ACTIVE_TOLERANCE * np.abs(readouts)
    if active.any():
        first = np.unravel_index(int(np.argmax(active)), active.shape)
        raise ValueError(f'fdt_psd: -Im(w^H u) = {float(dissipation[first]):.6g} < 0 at Omega = '
                         f'{float(np.broadcast_to(omega, readouts.shape)[first]):.6g} rad/s (readout '
                         f'{complex(readouts[first])}): the solid operator is active (Im K < 0 for exp(+j Omega t)); '
                         'check the loss factor / viscosity signs')
    return 2.0 * K_B * temperature / omega * np.maximum(dissipation, 0.0) * scale


def sideband_factors(frequencies, temperature) -> tuple:
    """(Stokes x (n + 1), anti-Stokes x n) per frequency [Hz] > 0: the quantum weights of the classical per-sideband
    psd, x = hbar 2 pi f / (k_B T), n = 1 / (e^x - 1) (both -> 1 classically)."""
    temperature = _temperature(temperature)
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if not (frequencies > 0.0).all():
        raise ValueError('sideband_factors: frequencies must be > 0 Hz')
    x = HBAR * 2.0 * math.pi * frequencies / (K_B * temperature)
    return x / -np.expm1(-x), x / np.expm1(x)


# ---- moduli laws ----
def modulus_function(law):
    """Vectorised omega (array) -> complex modulus of a viscoelastic.ModulusLaw-like law (relaxed, loss_factor,
    viscosity, branches: the formula of ModulusLaw.at, pinned by brillouin_homogeneous_reference); ValueError for an
    incompressible (infinite) law."""
    relaxed, loss_factor, viscosity = float(law.relaxed), float(law.loss_factor), float(law.viscosity)
    branches = tuple((float(modulus), float(time)) for modulus, time in law.branches)
    if not math.isfinite(relaxed):
        raise ValueError('modulus_function: an incompressible law has no finite modulus (use the bulk modulus)')

    def value(omega):
        jw = 1j * np.asarray(omega, dtype=np.float64)
        result = relaxed * (1.0 + 1j * loss_factor) + jw * viscosity
        for modulus, time in branches:
            result = result + modulus * jw * time / (1.0 + jw * time)
        return result
    return value


def longitudinal_function(bulk_law, shear_law):
    """omega -> M*(omega) = K*(omega) + 4/3 G*(omega) of two modulus callables (modulus_function or scalar laws)."""
    def value(omega):
        return _law_values(bulk_law, omega) + 4.0 / 3.0 * _law_values(shear_law, omega)
    return value


def _law_values(law, omega) -> np.ndarray:
    """Complex law(omega) with omega's shape: a vectorised callable, a constant one, or (TypeError on arrays) a scalar
    one evaluated per distinct omega."""
    omega = np.asarray(omega, dtype=np.float64)
    try:
        values = np.asarray(law(omega), dtype=np.complex128)
    except TypeError:
        unique, inverse = np.unique(omega, return_inverse=True)
        values = np.array([complex(law(float(item))) for item in unique], dtype=np.complex128)[inverse]
        return values.reshape(omega.shape)
    if values.ndim == 0:
        return np.full(omega.shape, complex(values))
    if values.shape != omega.shape:
        raise ValueError(f'modulus law returned shape {values.shape} for omega of shape {omega.shape}')
    return values


# ---- reference spectra ----
def _frequencies(frequencies) -> np.ndarray:
    frequencies = np.asarray(frequencies, dtype=np.float64).reshape(-1)
    if not (frequencies > 0.0).all() or not np.isfinite(frequencies).all():
        raise ValueError('reference spectra: frequencies must be finite and > 0 Hz')
    return frequencies


def _density(density) -> float:
    density = float(density)
    if not (density > 0.0 and math.isfinite(density)):
        raise ValueError(f'reference spectra: the density must be > 0 kg/m^3, got {density!r}')
    return density


def _resonance(q, law, density) -> np.ndarray:
    """Omega_0 (n,) with Omega_0 = |q| sqrt(Re M*(Omega_0) / rho) by fixed point (from Re M*(0))."""
    omega0 = np.zeros_like(q)
    for _ in range(_RESONANCE_ITERATIONS):
        stiffness = _law_values(law, omega0).real
        if not (stiffness > 0.0).all():
            raise ValueError('local reference: the longitudinal storage modulus Re M* is <= 0 (no sound wave)')
        updated = q * np.sqrt(stiffness / density)
        if np.max(np.abs(updated - omega0) / updated) <= _RESONANCE_RTOL:
            return updated
        omega0 = updated
    raise ValueError(f'local reference: Omega_0 = |q| sqrt(Re M*(Omega_0) / rho) did not converge in '
                     f'{_RESONANCE_ITERATIONS} iterations (a strongly dispersive modulus)')


def local_reference(frequencies, sigma_l_weights, q, longitudinal_law, density, temperature, scale) -> np.ndarray:
    """(n_f,) local reference psd [1/Hz] (header) of points with weights |kappa sigma_L|^2 dV (n,) [ring weights
    included] and local wavenumbers |q| (n,) [1/m] in ONE medium (longitudinal_law omega -> M* = K* + 4/3 G*, density
    [kg/m^3]; call once per material of the drive selection and add); points with q = 0 or weight 0 carry no line."""
    frequencies = _frequencies(frequencies)
    temperature, density = _temperature(temperature), _density(density)
    weights = np.asarray(sigma_l_weights, dtype=np.float64).reshape(-1)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if weights.shape != q.shape or not (np.isfinite(weights).all() and np.isfinite(q).all()):
        raise ValueError(f'local_reference: weights {weights.shape} and q {q.shape} must be finite and of one shape')
    if (weights < 0.0).any() or (q < 0.0).any():
        raise ValueError('local_reference: weights and |q| must be >= 0')
    keep = (weights > 0.0) & (q > 0.0)
    result = np.zeros(len(frequencies))
    if not keep.any():
        return result
    weights, q = weights[keep], q[keep]
    omega0 = _resonance(q, longitudinal_law, density)
    modulus = _law_values(longitudinal_law, omega0)
    gamma = q ** 2 * modulus.imag / (density * omega0)
    if (gamma < 0.0).any():
        raise ValueError('local reference: Im M* < 0 (an active modulus)')
    amplitude = K_B * temperature * weights / modulus.real
    omega_squared = (2.0 * math.pi * frequencies)[:, None] ** 2
    step = max(1, _CHUNK_VALUES // len(frequencies))
    for start in range(0, len(q), step):
        chunk = slice(start, start + step)
        resonance_squared, width = omega0[chunk] ** 2, gamma[chunk]
        line = 2.0 * width * resonance_squared / ((resonance_squared - omega_squared) ** 2
                                                  + width ** 2 * omega_squared)
        result += line @ amplitude[chunk]
    return result * scale


def _stress_spectrum(stress_grid, dx, dy, padding) -> tuple:
    """(sigma^ (3, P_x, P_y) = dx dy FFT of the zero-padded (xx, yy, xy) raster, q_x (P_x, 1), q_y (1, P_y), weight
    d^2q / (2 pi)^2 = 1 / (P_x P_y dx dy))."""
    stress_grid = np.asarray(stress_grid, dtype=np.complex128)
    if stress_grid.ndim != 3 or stress_grid.shape[0] != 3:
        raise ValueError(f'homogeneous reference: stress_grid must be (3, n_x, n_y) rows (xx, yy, xy), got '
                         f'{stress_grid.shape}')
    dx, dy, padding = float(dx), float(dy), int(padding)
    if not (dx > 0.0 and dy > 0.0 and padding >= 1):
        raise ValueError(f'homogeneous reference: dx, dy must be > 0 and padding >= 1, got {dx}, {dy}, {padding}')
    size_x, size_y = padding * stress_grid.shape[1], padding * stress_grid.shape[2]
    spectrum = dx * dy * np.fft.fft2(stress_grid, s=(size_x, size_y), axes=(1, 2))
    q_x = (2.0 * math.pi * np.fft.fftfreq(size_x, dx))[:, None]
    q_y = (2.0 * math.pi * np.fft.fftfreq(size_y, dy))[None, :]
    return spectrum, q_x, q_y, 1.0 / (size_x * size_y * dx * dy)


def _projections(spectrum, q_x, q_y) -> tuple:
    """(q^2 |s_L|^2, q^2 |s_T|^2, q^2) per wavevector (q = 0: 0, 0, 0) and the q = 0 direction averages (|s_L|^2,
    |s_T|^2) of a (xx, yy, xy) spectrum."""
    xx, yy, xy = spectrum
    q_squared = q_x ** 2 + q_y ** 2
    safe = np.where(q_squared > 0.0, q_squared, 1.0)
    longitudinal = np.abs(q_x ** 2 * xx + 2.0 * q_x * q_y * xy + q_y ** 2 * yy) ** 2 / safe
    transverse = np.abs(q_x * q_y * (yy - xx) + (q_x ** 2 - q_y ** 2) * xy) ** 2 / safe
    longitudinal[0, 0] = transverse[0, 0] = 0.0
    mean, half_difference = 0.5 * (xx[0, 0] + yy[0, 0]), 0.5 * (xx[0, 0] - yy[0, 0])
    origin = (abs(mean) ** 2 + 0.5 * abs(half_difference) ** 2 + 0.5 * abs(xy[0, 0]) ** 2,
              0.5 * abs(half_difference) ** 2 + 0.5 * abs(xy[0, 0]) ** 2)
    return longitudinal, transverse, np.broadcast_to(q_squared, longitudinal.shape), origin


def homogeneous_reference(frequencies, stress_grid, dx, dy, longitudinal_law, shear_law, density, temperature, scale,
                          padding=2) -> np.ndarray:
    """(n_f,) exact unbounded-medium psd [1/Hz] (header) of the drive stress kappa sigma~ (3, n_x, n_y) in the plane
    rows (xx, yy, xy) on uniform optical cells dx, dy [m] (zero outside the drive region; zero-padded to padding x),
    one isotropic medium: longitudinal_law omega -> M* = K* + 4/3 G*, shear_law omega -> G* (lossy: Im M* > 0)."""
    frequencies = _frequencies(frequencies)
    temperature, density = _temperature(temperature), _density(density)
    spectrum, q_x, q_y, weight = _stress_spectrum(stress_grid, dx, dy, padding)
    longitudinal, transverse, q_squared, _ = _projections(spectrum, q_x, q_y)
    longitudinal, transverse, q_squared = longitudinal.ravel(), transverse.ravel(), q_squared.ravel()
    omega = 2.0 * math.pi * frequencies
    bulk, shear = _law_values(longitudinal_law, omega), _law_values(shear_law, omega)
    if (bulk.imag <= 0.0).any() or (shear.imag < 0.0).any():
        raise ValueError('homogeneous reference: needs Im M* > 0 and Im G* >= 0 (a lossy, passive medium; a lossless '
                         'longitudinal modulus gives a delta line)')
    result = np.empty(len(frequencies))
    step = max(1, _CHUNK_VALUES // max(len(q_squared), 1))
    for start in range(0, len(frequencies), step):
        chunk = slice(start, start + step)
        inertia = density * omega[chunk, None] ** 2
        total = np.zeros((len(omega[chunk]), len(q_squared)))
        for numerator, modulus in ((longitudinal, bulk[chunk]), (transverse, shear[chunk])):
            real = modulus.real[:, None] * q_squared - inertia
            imaginary = modulus.imag[:, None] * q_squared
            denominator = real ** 2 + imaginary ** 2
            total += numerator * imaginary / np.where(denominator > 0.0, denominator, 1.0)
        result[chunk] = total.sum(axis=1)
    return 2.0 * K_B * temperature / omega * weight * result * scale


def homogeneous_static(stress_grid, dx, dy, longitudinal_modulus, shear_modulus, temperature, scale,
                       padding=2) -> float:
    """Sum rule of homogeneous_reference, 2 int_0^inf psd df = k_B T int d^2q / (2 pi)^2 (|s_L|^2 / M + |s_T|^2 / G) x
    scale, of real static moduli M (> 0) and G (>= 0; 0 only when the transverse part vanishes, a liquid's dilation
    drive: else ValueError); q = 0 takes the direction averages; a dilation drive c gives (k_B T / M) int |c|^2 dA."""
    temperature = _temperature(temperature)
    longitudinal_modulus, shear_modulus = float(longitudinal_modulus), float(shear_modulus)
    if not (longitudinal_modulus > 0.0 and shear_modulus >= 0.0):
        raise ValueError(f'homogeneous_static: needs M > 0 and G >= 0, got {longitudinal_modulus}, {shear_modulus}')
    spectrum, q_x, q_y, weight = _stress_spectrum(stress_grid, dx, dy, padding)
    longitudinal, transverse, q_squared, origin = _projections(spectrum, q_x, q_y)
    safe = np.where(q_squared > 0.0, q_squared, 1.0)
    longitudinal_sum = float(np.sum(longitudinal / safe)) + origin[0]
    transverse_sum = float(np.sum(transverse / safe)) + origin[1]
    if shear_modulus == 0.0:
        if transverse_sum > 1e-12 * longitudinal_sum:
            raise ValueError('homogeneous_static: a transverse drive in a medium without static shear modulus has no '
                             'finite static compliance')
        transverse_term = 0.0
    else:
        transverse_term = transverse_sum / shear_modulus
    return K_B * temperature * weight * (longitudinal_sum / longitudinal_modulus + transverse_term) * scale


# ---- fit and band integral ----
def band_integral(frequencies, psd) -> np.ndarray:
    """Trapezoid int psd df over strictly ascending frequencies [Hz] (psd (..., n_f); 0 for one frequency)."""
    frequencies = np.asarray(frequencies, dtype=np.float64).reshape(-1)
    psd = np.asarray(psd, dtype=np.float64)
    if psd.shape[-1:] != frequencies.shape or not (np.diff(frequencies) > 0.0).all():
        raise ValueError(f'band_integral: psd {psd.shape} must end in the {len(frequencies)} strictly ascending '
                         'frequencies')
    return 0.5 * np.sum((psd[..., 1:] + psd[..., :-1]) * np.diff(frequencies), axis=-1)


def _half_maximum_width(frequencies, psd, peak_index) -> float:
    """Width between the linearly interpolated half-maximum crossings around the peak (one side found: twice its
    distance; none: the band)."""
    half = 0.5 * psd[peak_index]

    def crossing(i):   # psd crosses half between i and i + 1
        return frequencies[i] + (half - psd[i]) / (psd[i + 1] - psd[i]) * (frequencies[i + 1] - frequencies[i])
    crossings = []
    below = np.flatnonzero(psd[:peak_index] < half)
    if below.size:
        crossings.append(crossing(int(below[-1])))
    above = np.flatnonzero(psd[peak_index + 1:] < half)
    if above.size:
        crossings.append(crossing(peak_index + int(above[0])))
    if len(crossings) == 2:
        return float(crossings[1] - crossings[0])
    if len(crossings) == 1:
        return float(2.0 * abs(crossings[0] - frequencies[peak_index]))
    return float(frequencies[-1] - frequencies[0])


def _dho_width(shift, gamma) -> tuple:
    """(peak denominator D_min, fwhm) of A / ((f0^2 - f^2)^2 + gamma^2 f^2): the maximum at f^2 = max(f0^2 -
    gamma^2 / 2, 0), the half-maximum roots of D(x) = 2 D_min in x = f^2 (the lower clipped at 0)."""
    f0_squared, gamma_squared = shift * shift, gamma * gamma
    peak_x = max(f0_squared - 0.5 * gamma_squared, 0.0)
    minimum = (f0_squared - peak_x) ** 2 + gamma_squared * peak_x
    linear = 2.0 * f0_squared - gamma_squared
    root = math.sqrt(max(linear * linear - 4.0 * (f0_squared * f0_squared - 2.0 * minimum), 0.0))
    upper, lower = 0.5 * (linear + root), 0.5 * (linear - root)
    return minimum, math.sqrt(max(upper, 0.0)) - math.sqrt(max(lower, 0.0))


def dho_fit(frequencies, psd) -> dict:
    """{shift f0 [Hz], fwhm [Hz] (exact half-maximum width of the fit), gamma [Hz], peak (fit maximum, psd units),
    r_squared, fit_warning} of S = A / ((f0^2 - f^2)^2 + gamma^2 f^2) fitted (scipy least_squares, normalised units)
    from the peak / half-maximum start; never raises on a failed fit (NaN values and fit_warning)."""
    frequencies = np.asarray(frequencies, dtype=np.float64).reshape(-1)
    psd = np.asarray(psd, dtype=np.float64).reshape(-1)
    if psd.shape != frequencies.shape:
        raise ValueError(f'dho_fit: psd {psd.shape} and frequencies {frequencies.shape} differ')
    result = {name: math.nan for name in FIT_EXTRAS}
    result['fit_warning'] = ''
    if len(frequencies) < _HALF_MAXIMUM_MIN_POINTS or not (np.diff(frequencies) > 0.0).all():
        result['fit_warning'] = (f'no fit: needs >= {_HALF_MAXIMUM_MIN_POINTS} strictly ascending frequencies, got '
                                 f'{len(frequencies)}')
        return result
    if not np.isfinite(psd).all() or not psd.max() > 0.0:
        result['fit_warning'] = 'no fit: the spectrum has no positive finite peak'
        return result
    from scipy.optimize import least_squares
    peak_index = int(np.argmax(psd))
    peak = float(psd[peak_index])
    frequency_scale = float(frequencies[peak_index])
    scaled_f, scaled_psd = frequencies / frequency_scale, psd / peak
    gamma_start = max(_half_maximum_width(frequencies, psd, peak_index) / frequency_scale, 1e-6)
    start = np.array([1.0, gamma_start, gamma_start ** 2])   # a = peak x gamma^2 f0^2 in normalised units

    def residual(parameters):
        shift, gamma, amplitude = parameters
        return amplitude / ((shift ** 2 - scaled_f ** 2) ** 2 + gamma ** 2 * scaled_f ** 2) - scaled_psd
    warnings = []
    if peak_index in (0, len(frequencies) - 1):
        warnings.append('the peak lies at the band edge: widen the frequency range')
    try:
        fitted = least_squares(residual, start, bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf]), x_scale='jac',
                               xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=5000)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        result['fit_warning'] = f'fit failed: {exc}'
        return result
    shift, gamma, amplitude = (float(value) for value in fitted.x)
    if not fitted.success or not (shift > 0.0 and gamma > 0.0 and amplitude > 0.0):
        warnings.append(f'fit did not converge ({fitted.message})')
    shift, gamma = shift * frequency_scale, gamma * frequency_scale
    amplitude = amplitude * peak * frequency_scale ** 4
    minimum, fwhm = _dho_width(shift, gamma)
    spread = float(np.sum((scaled_psd - scaled_psd.mean()) ** 2))
    result.update(shift=shift, fwhm=fwhm, gamma=gamma, peak=amplitude / minimum if minimum > 0.0 else math.inf,
                  r_squared=1.0 - float(np.sum(fitted.fun ** 2)) / spread if spread > 0.0 else math.nan,
                  fit_warning='; '.join(warnings))
    return result


# ---- result ----
@dataclass(frozen=True, eq=False)
class BrillouinSpectra:
    """What brillouin_result reports (built by the runtime BrillouinDrive.result, so this module needs no solver):
    per scan position the per-sideband psd [1/Hz] (scale applied) on ascending frequencies [Hz], the reference (None:
    reference 'none'), the focus offsets and the scalar extras (incident_power, collection_power, wavelength,
    mirror_factor, azimuthal_order, axial_length, torsion_drive_fraction, drive_norms {...}, q_mean[_<k>] ...)."""
    frequencies: np.ndarray        # (n_f,) ascending [Hz]
    psd: np.ndarray                # (n_positions, n_f) [1/Hz]
    reference: object              # (n_positions, n_f) [1/Hz] or None
    offsets: np.ndarray            # (n_positions, 2) focus offsets (x, y) [m] (BOR: (0, dz))
    temperature: float             # [K]
    fit_peak: bool
    extras: dict = field(default_factory=dict)


def _suffixed(name, position, several) -> str:
    return f'{name}_{position}' if several else name


def brillouin_result(spectra) -> dict:
    """'xy' result of a BrillouinSpectra vs frequency [Hz]: per position k (suffix _<k> with several) the series psd,
    psd_stokes, psd_anti_stokes, psd_reference (with a reference), psd_db (10 log10 psd); extras scattered_fraction (2 x
    the band integral), the fit (shift, fwhm, gamma, peak, r_squared, fit_warning; reference_shift of the reference's
    fit) with fit_peak, temperature, the spectra's extras and the scan table; ValueError before any frequency."""
    frequencies = np.asarray(spectra.frequencies, dtype=np.float64).reshape(-1)
    if not len(frequencies):
        raise ValueError('not solved yet: the Brillouin spectrum needs a frequency_response study')
    psd = np.asarray(spectra.psd, dtype=np.float64)
    offsets = np.asarray(spectra.offsets, dtype=np.float64)
    if psd.ndim != 2 or psd.shape[1] != len(frequencies) or offsets.shape != (len(psd), 2):
        raise ValueError(f'brillouin_result: psd {psd.shape} must be (positions, {len(frequencies)}) with offsets '
                         f'(positions, 2), got offsets {offsets.shape}')
    reference = None if spectra.reference is None else np.asarray(spectra.reference, dtype=np.float64)
    if reference is not None and reference.shape != psd.shape:
        raise ValueError(f'brillouin_result: reference {reference.shape} differs from psd {psd.shape}')
    stokes, anti_stokes = sideband_factors(frequencies, spectra.temperature)
    several = len(psd) > 1
    series, units, extras, rows = {}, {}, {}, []
    for index, values in enumerate(psd):
        position = index + 1

        def put(name, data, position=position):
            key = _suffixed(name, position, several)
            series[key], units[key] = data, SERIES_UNITS[name]
        put(PSD_SERIES, values.copy())
        put(STOKES_SERIES, values * stokes)
        put(ANTI_STOKES_SERIES, values * anti_stokes)
        if reference is not None:
            put(REFERENCE_SERIES, reference[index].copy())
        with np.errstate(divide='ignore'):
            put(DB_SERIES, 10.0 * np.log10(values))
        fraction = 2.0 * float(band_integral(frequencies, values)) if len(frequencies) > 1 else 0.0
        extras[_suffixed('scattered_fraction', position, several)] = fraction
        fit = {name: math.nan for name in FIT_EXTRAS}
        if spectra.fit_peak:
            fit = dho_fit(frequencies, values)
            for name in FIT_EXTRAS + ('fit_warning',):
                extras[_suffixed(name, position, several)] = fit[name]
            if reference is not None:
                reference_fit = dho_fit(frequencies, reference[index])
                extras[_suffixed('reference_shift', position, several)] = reference_fit['shift']
        rows.append([position, float(offsets[index, 0]), float(offsets[index, 1]), fit['shift'], fit['fwhm'],
                     fit['peak'], fraction])
    extras.update(spectra.extras)
    extras['temperature'] = float(spectra.temperature)
    extras['scan'] = {'columns': list(SCAN_COLUMNS), 'units': list(SCAN_UNITS), 'rows': rows}
    return {'kind': 'xy', 'x': frequencies.copy(), 'x_label': X_LABEL, 'x_unit': 'Hz', 'y_label': Y_LABEL,
            'series': series, 'series_units': units, **extras}
