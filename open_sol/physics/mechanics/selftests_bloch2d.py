"""Bloch / Floquet unit cells of the 2D solid_mechanics (periodic_x / periodic_y, bloch_kx / bloch_ky; both mesh kinds):
bulk bands of an all-periodic homogeneous cell against the folded sound lines (plane strain, a k = 0 cell with its
rigid modes dropped, guided 2.5D with axial_wavenumber, k + 2 pi / a), the Rayleigh wave of a periodic half-space slab
(fixed or elastic-PML bottom), Lamb waves of a free plate cell against the Rayleigh-Lamb equations, the YZ lithium
niobate SAW (free and metallised surface), the validation rules and the 'solid_bloch_band_diagram' template;
definition registers them."""
import cmath
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import cpu_defaults, lithium_niobate_entry, scaled_worst_result
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, ParameterSpec,
                           PhysicsSpec, StudySpec, SweepSpec)
from .selftests import _elastic, _lame

# definition imports this module for its template: the sibling selftests modules load inside the checks only
UM = 1e-6
_SOLID = 'solid'
_SOLID_TYPE = 'solid_mechanics'   # = selftests.SOLID_TYPE_NAME
_TABLE = 'bands'
_FREQUENCY_COLUMN = 1
_ISOTROPIC = (100e9, 0.25, 2000.0)   # E [Pa], nu, rho [kg/m^3]


def _speeds():
    """(c_L, c_T) [m/s] of _ISOTROPIC in plane strain."""
    youngs_modulus, poisson_ratio, density = _ISOTROPIC
    lame_lambda, mu = _lame(youngs_modulus, poisson_ratio)
    return math.sqrt((lame_lambda + 2.0 * mu) / density), math.sqrt(mu / density)


def _medium(selection=('all',)):
    youngs_modulus, poisson_ratio, density = _ISOTROPIC
    return MaterialSpec('medium', properties=_elastic(youngs_modulus, poisson_ratio, rho=density),
                        selection=list(selection))


def _mesh(kind, step):
    if kind == 'triangles':
        return MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=step)
    return MeshSpec(mode='uniform', dx=step)


def _cell_spec(title, size, kind, step, settings, features=(), count=4, materials=None, geometry=()):
    return ModelSpec(title=title, domain=DomainSpec(size_x=size[0], size_y=size[1]), geometry=list(geometry),
                     materials=list(materials or [_medium()]),
                     physics=[PhysicsSpec(_SOLID, _SOLID_TYPE, settings=dict(settings),
                                          features=[*features, FeatureSpec(_TABLE, 'resonance_table')])],
                     meshes=[_mesh(kind, step)], solver_defaults=cpu_defaults(),   # host modal solves either way
                     study=StudySpec(type='eigenfrequency', settings={'n_resonances': count}))


def _table_frequencies(results):
    return sorted(float(row[_FREQUENCY_COLUMN]) for row in results.result(_SOLID, _TABLE)['rows'])


def _band_error(measured, expected):
    if len(measured) < len(expected):
        return math.inf
    return float(np.max(np.abs(np.asarray(measured[:len(expected)]) / np.asarray(expected) - 1.0)))


# ---- solid_bloch_bulk_bands_2d: an all-periodic homogeneous square cell ----
# a = 1 um, periodic x and y, k = (0.3, 0.17) pi / a: the bands are the folded sound lines c_L |k + G| and c_T |k + G|
# (G on the 2 pi / a lattice; plane strain c_L = sqrt((lambda + 2 mu) / rho), c_T = sqrt(mu / rho)); the 6 lowest.
# Runs: tensor (Q1, a / 32), triangles (P2, h a / 8), shifted (tensor at k + (2 pi / a, 0): the same discrete bands),
# guided (tensor, axial_wavenumber k_z = 0.4 pi / a: c |(k + G, k_z)| with the transverse line twice), gamma_tensor /
# gamma_triangles (k = 0, target 0.5 c_T / a: the two free translations counted and dropped, the 4-fold c_T 2 pi / a
# band, 6 modes asked: with 4 ARPACK may return only 3 copies of the exactly degenerate Q1 band). Tolerances first estimates from the measured errors (Q1 ~2.5e-3 at the 6th band, P2 ~1e-4).
_BULK_SIZE = 1.0 * UM
_BULK_K = (0.3 * math.pi / _BULK_SIZE, 0.17 * math.pi / _BULK_SIZE)
_BULK_KZ = 0.4 * math.pi / _BULK_SIZE
_BULK_BANDS = 6
_GAMMA_BANDS = 4
_BULK_RUNS = {   # run: (mesh kind, step, tolerance)
    'tensor': ('tensor', _BULK_SIZE / 32.0, 4e-3), 'triangles': ('triangles', _BULK_SIZE / 8.0, 3e-4),
    'shifted': ('tensor', _BULK_SIZE / 32.0, 1e-8), 'guided': ('tensor', _BULK_SIZE / 32.0, 4e-3),
    'gamma_tensor': ('tensor', _BULK_SIZE / 32.0, 4e-3), 'gamma_triangles': ('triangles', _BULK_SIZE / 8.0, 3e-4)}


def _bulk_spec(run):
    kind, step, _ = _BULK_RUNS[run]
    k_x, k_y = _BULK_K
    if run == 'shifted':
        k_x += 2.0 * math.pi / _BULK_SIZE
    settings = {'periodic_x': True, 'periodic_y': True, 'bloch_kx': k_x, 'bloch_ky': k_y}
    count = _BULK_BANDS
    if run.startswith('gamma'):
        settings.update(bloch_kx=0.0, bloch_ky=0.0, target_frequency=0.5 * _speeds()[1] / _BULK_SIZE)
        count = _GAMMA_BANDS + 2   # Lanczos may miss a copy of an exactly 4-fold band (Q1 square cells) at n = 4
    if run == 'guided':
        settings['axial_wavenumber'] = _BULK_KZ
    return _cell_spec(f'Self-test: 2D Bloch bulk bands ({run})', (_BULK_SIZE, _BULK_SIZE), kind, step, settings,
                      count=count)


def _sound_lines(k_x, k_y, k_z=None, count=_BULK_BANDS):
    """The `count` lowest folded sound-line frequencies [Hz] (k_z: guided, transverse twice)."""
    c_l, c_t = _speeds()
    speeds = (c_l, c_t) if k_z is None else (c_l, c_t, c_t)
    step = 2.0 * math.pi / _BULK_SIZE
    values = [speed * math.sqrt((k_x + step * i) ** 2 + (k_y + step * j) ** 2 + (k_z or 0.0) ** 2) / (2.0 * math.pi)
              for i in range(-3, 4) for j in range(-3, 4) for speed in speeds]
    return sorted(values)[:count]


def _check_bulk(results_by_name):
    errors, tolerances, details = {}, {}, {}
    measured = {run: _table_frequencies(results) for run, results in results_by_name.items()}
    for run in results_by_name:
        if run == 'shifted':
            expected = measured.get('tensor', [])[:_BULK_BANDS]
        elif run.startswith('gamma'):
            expected = [_speeds()[1] / _BULK_SIZE] * _GAMMA_BANDS
        else:
            expected = _sound_lines(*_BULK_K, k_z=_BULK_KZ if run == 'guided' else None)
        errors[run] = _band_error(measured[run], expected) if expected else math.inf
        tolerances[run] = _BULK_RUNS[run][2]
        details[run] = {'measured_Hz': measured[run], 'expected_Hz': list(expected)}
    return scaled_worst_result(errors, tolerances, **details)


# ---- solid_bloch_rayleigh_2d: the Rayleigh wave of a periodic half-space slab ----
# Period L = 1 um along x at k = pi / (2 L) (the folded branch |k - 2 pi / L| lies 3 x higher), free top, 8 um deep with
# a Fixed bottom (tensor Q1 L / 16, triangles P2 h L / 4), or 3 um above an 8-layer elastic PML on y_lo (triangles,
# target at the Rayleigh frequency). The lowest mode is the Rayleigh wave, c = 2 pi f / k against the root c_R of (2 -
# xi^2)^2 = 4 sqrt(1 - xi^2) sqrt(1 - xi^2 c_T^2 / c_L^2), xi = c / c_T (nu 0.25: c_R = 0.9194 c_T). Tolerance 1e-2
# (measured: tensor 8e-4, fixed triangles 1e-4, PML 2e-3).
_SLAB_PERIOD = 1.0 * UM
_SLAB_K = 0.5 * math.pi / _SLAB_PERIOD
_SLAB_RUNS = {'tensor': ('tensor', _SLAB_PERIOD / 16.0, 8.0 * UM, False),
              'triangles': ('triangles', _SLAB_PERIOD / 4.0, 8.0 * UM, False),
              'pml': ('triangles', _SLAB_PERIOD / 4.0, 3.0 * UM, True)}
_RAYLEIGH_TOLERANCE = 1e-2


def rayleigh_speed(c_l, c_t) -> float:
    """[m/s] root c_R of the Rayleigh equation (2 - xi^2)^2 = 4 sqrt(1 - xi^2) sqrt(1 - xi^2 c_T^2 / c_L^2)."""
    from scipy.optimize import brentq
    ratio = (c_t / c_l) ** 2
    root = brentq(lambda xi: (2.0 - xi * xi) ** 2 - 4.0 * math.sqrt(1.0 - xi * xi) * math.sqrt(1.0 - xi * xi * ratio),
                  0.5, 0.999999)
    return root * c_t


def _slab_spec(run):
    kind, step, depth, pml = _SLAB_RUNS[run]
    settings = {'periodic_x': True, 'bloch_kx': _SLAB_K}
    features = []
    if pml:
        settings.update(pml_y_lo=True, pml_cells=8,
                        target_frequency=rayleigh_speed(*_speeds()) * _SLAB_K / (2.0 * math.pi))
    else:
        features.append(FeatureSpec('bottom', 'fixed', ['y_lo']))
    return _cell_spec(f'Self-test: 2D Bloch Rayleigh slab ({run})', (_SLAB_PERIOD, depth), kind, step, settings,
                      features, count=2)


def _check_rayleigh(results_by_name):
    expected = rayleigh_speed(*_speeds())
    errors, tolerances, details = {}, {}, {'c_R_expected': expected}
    for run, results in results_by_name.items():
        frequencies = _table_frequencies(results)
        speed = 2.0 * math.pi * frequencies[0] / _SLAB_K if frequencies else math.nan
        errors[run] = abs(speed / expected - 1.0) if frequencies else math.inf
        tolerances[run] = _RAYLEIGH_TOLERANCE
        details[run] = {'c_measured': speed, 'frequencies_Hz': frequencies}
    return scaled_worst_result(errors, tolerances, **details)


# ---- solid_bloch_lamb_2d: Lamb waves of a free plate cell ----
# Plate 0.5 um thick, period L = 1 um along x, no supports, k = pi / (2 L): the 3 lowest modes against the roots of the
# Rayleigh-Lamb equations (symmetric (q^2 - k^2)^2 cos(p h/2) sin(q h/2) / q + 4 k^2 p sin(p h/2) cos(q h/2) = 0,
# antisymmetric (q^2 - k^2)^2 sin(p h/2) / p cos(q h/2) + 4 k^2 q sin(q h/2) cos(p h/2) = 0, p^2 = omega^2 / c_L^2 - k^2,
# q^2 = omega^2 / c_T^2 - k^2, real for imaginary p, q) at k and k +- 2 pi / L: A0, S0 and the folded A0. Tolerances
# (first estimates): triangles P2 h / 4 3e-4 (measured 1.8e-4), tensor Q1 h / 16 5e-3 (3e-3).
_PLATE_THICKNESS = 0.5 * UM
_PLATE_RUNS = {'triangles': ('triangles', _PLATE_THICKNESS / 4.0, 3e-4),
               'tensor': ('tensor', _PLATE_THICKNESS / 16.0, 5e-3)}
_PLATE_MODES = 3
_LAMB_SCAN = 4000


def _lamb_residual(omega, k, symmetric):
    c_l, c_t = _speeds()
    half = 0.5 * _PLATE_THICKNESS
    p = cmath.sqrt(omega * omega / c_l ** 2 - k * k)
    q = cmath.sqrt(omega * omega / c_t ** 2 - k * k)

    def sine_over(z):   # sin(z h / 2) / z, real for real or imaginary z
        return half if abs(z) == 0.0 else cmath.sin(z * half) / z
    if symmetric:
        value = (q * q - k * k) ** 2 * cmath.cos(p * half) * sine_over(q) + 4.0 * k * k * p * cmath.sin(p * half) * \
            cmath.cos(q * half)
    else:
        value = (q * q - k * k) ** 2 * sine_over(p) * cmath.cos(q * half) + 4.0 * k * k * q * cmath.sin(q * half) * \
            cmath.cos(p * half)
    return value.real


def lamb_frequencies(k_values, f_max, count) -> list:
    """The `count` lowest Rayleigh-Lamb frequencies [Hz] below f_max over the wavenumbers k_values [rad/m] (sign
    changes on a uniform omega scan, brentq)."""
    from scipy.optimize import brentq
    omegas = np.linspace(1e-3, 1.0, _LAMB_SCAN) * 2.0 * math.pi * f_max   # omega -> 0 is a trivial root
    roots = []
    for k in k_values:
        for symmetric in (True, False):
            values = np.array([_lamb_residual(omega, k, symmetric) for omega in omegas])
            for index in np.flatnonzero(values[:-1] * values[1:] < 0.0):
                roots.append(brentq(_lamb_residual, omegas[index], omegas[index + 1], args=(k, symmetric)))
    return sorted(root / (2.0 * math.pi) for root in roots)[:count]


def _plate_spec(run):
    kind, step, _ = _PLATE_RUNS[run]
    return _cell_spec(f'Self-test: 2D Bloch Lamb plate ({run})', (_SLAB_PERIOD, _PLATE_THICKNESS), kind, step,
                      {'periodic_x': True, 'bloch_kx': _SLAB_K}, count=_PLATE_MODES)


def _check_lamb(results_by_name):
    period = 2.0 * math.pi / _SLAB_PERIOD
    expected = lamb_frequencies((_SLAB_K, _SLAB_K - period, _SLAB_K + period), 4e9, _PLATE_MODES)
    errors, tolerances, details = {}, {}, {'expected_Hz': expected}
    for run, results in results_by_name.items():
        measured = _table_frequencies(results)
        errors[run] = _band_error(measured, expected) if len(expected) == _PLATE_MODES else math.inf
        tolerances[run] = _PLATE_RUNS[run][2]
        details[run] = {'measured_Hz': measured}
    return scaled_worst_result(errors, tolerances, **details)


# ---- solid_bloch_ln_saw_2d: the YZ lithium niobate SAW on a periodic piezoelectric slab ----
# Crystal entry euler (0, 90, 90), plane sagittal (model x = crystal Z, the propagation; y = -crystal Y, the surface
# normal): period L = 1 um at k = pi / (2 L), substrate 8 um deep with a Fixed bottom (D_n = 0 there), Q1 cells of L /
# 16, piezoelectric, no electrode (k != 0 excludes a constant potential). Runs free (3 um of void dielectric eps0 above
# the surface: the open surface) and metallised (an electrode at 0 V on the surface y_hi). Reference: the literature
# YZ LN SAW speeds v_free = 3488 m/s, v_metal = 3404 m/s (the crystal data are transcribed: a sanity check, tolerance
# 2e-2; measured +0.5 % / +0.3 %); details also K^2 = 2 (v_free - v_metal) / v_free (literature ~4.8 %). Run grating:
# one IDT-like period p = L of a shorted strip (electrode_domain at 0 V on the top cell row, metallisation 0.5, no mass),
# 6 um deep, at the zone edge k = pi / p (real phase -1): the two lowest modes are the stopband edges v = 2 p f; no
# closed form here, so the check is a band: both edges within [0.98 v_metal, 1.02 v_free] of the references and a
# relative gap in (0.1 %, 5 %) (measured 3404.5 / 3441.8 m/s, gap 1.1 %).
_SAW_DEPTH = 8.0 * UM
_SAW_VACUUM = 3.0 * UM
_SAW_STEP = _SLAB_PERIOD / 16.0
_SAW_REFERENCE = {'free': 3488.0, 'metallised': 3404.0}
_SAW_TOLERANCE = 2e-2
_SAW_RUNS = ('free', 'metallised', 'grating')
_GRATING_DEPTH = 6.0 * UM
_GRATING_GAP = (1e-3, 5e-2)
YZ_EULER = [0.0, 90.0, 90.0]


def _saw_spec(run):
    features = [FeatureSpec('bottom', 'fixed', ['y_lo'])]
    depth = _GRATING_DEPTH if run == 'grating' else _SAW_DEPTH
    height = depth + (_SAW_VACUUM if run == 'free' else 0.0)
    geometry = [GeometrySpec('substrate', 'rectangle', {'x0': 0.0, 'y0': 0.0, 'x1': _SLAB_PERIOD, 'y1': depth})]
    if run == 'metallised':
        features.append(FeatureSpec('metal', 'electrode', ['y_hi'], {'connection': 'voltage', 'voltage': 0.0}))
    if run == 'grating':
        geometry.append(GeometrySpec('strip', 'rectangle', {'x0': 0.25 * _SLAB_PERIOD, 'y0': depth - _SAW_STEP,
                                                            'x1': 0.75 * _SLAB_PERIOD, 'y1': depth}))
        features.append(FeatureSpec('finger', 'electrode_domain', ['strip'], {'connection': 'voltage',
                                                                               'voltage': 0.0}))
    material = MaterialSpec('yz_ln', dispersion=[lithium_niobate_entry(YZ_EULER, plane='sagittal')],
                            selection=['substrate'])
    wavenumber = math.pi / _SLAB_PERIOD if run == 'grating' else _SLAB_K
    return _cell_spec(f'Self-test: 2D Bloch YZ lithium niobate SAW ({run})', (_SLAB_PERIOD, height), 'tensor',
                      _SAW_STEP, {'periodic_x': True, 'bloch_kx': wavenumber, 'piezoelectric': True}, features,
                      count=2, materials=[material], geometry=geometry)


def _check_saw(results_by_name):
    errors, tolerances, details = {}, {}, {}
    speeds = {}
    for run, results in results_by_name.items():
        frequencies = _table_frequencies(results)
        if run == 'grating':   # stopband edges at the zone edge: a band check (module comment)
            edges = [2.0 * _SLAB_PERIOD * value for value in frequencies[:2]]
            gap = edges[1] / edges[0] - 1.0 if len(edges) == 2 else math.nan
            inside = (len(edges) == 2 and 0.98 * _SAW_REFERENCE['metallised'] <= edges[0]
                      and edges[1] <= 1.02 * _SAW_REFERENCE['free'] and _GRATING_GAP[0] < gap < _GRATING_GAP[1])
            errors[run], tolerances[run] = (0.0 if inside else math.inf), 1.0
            details[run] = {'edge_speeds': edges, 'relative_gap': gap}
            continue
        speeds[run] = 2.0 * math.pi * frequencies[0] / _SLAB_K if frequencies else math.nan
        errors[run] = abs(speeds[run] / _SAW_REFERENCE[run] - 1.0) if frequencies else math.inf
        tolerances[run] = _SAW_TOLERANCE
        details[run] = {'v_measured': speeds[run], 'v_reference': _SAW_REFERENCE[run]}
    if 'free' in speeds and 'metallised' in speeds:
        details['K2'] = 2.0 * (speeds['free'] - speeds['metallised']) / speeds['free']
    return scaled_worst_result(errors, tolerances, **details)


# ---- solid_bloch_rules_2d: the validation of periodic settings (no run) ----
# A valid cell (triangles, periodic x, k != 0, fixed bottom) has no error; errors for: the stationary study, a support
# on a periodic face, a PML on a periodic face, geometry axisymmetric, stress_stiffening, a free trivial-phase cell at
# target 0; a warning for bloch_ky with periodic_y off.
def _rules_base(kind='triangles', **settings):
    values = {'periodic_x': True, 'bloch_kx': _SLAB_K, **settings}
    return _cell_spec('Self-test: 2D Bloch validation', (_SLAB_PERIOD, 2.0 * UM), kind, _SLAB_PERIOD / 4.0, values,
                      [FeatureSpec('bottom', 'fixed', ['y_lo'])])


def _check_rules(_):
    from ...model.validate import validate
    cases = {}
    cases['valid'] = (_rules_base(), None, None)
    stationary = _rules_base()
    stationary.study = StudySpec(type='stationary')
    cases['stationary'] = (stationary, 'error', 'eigenfrequency study only')
    on_face = _rules_base()
    on_face.physics[0].features.append(FeatureSpec('side', 'roller', ['x_lo']))
    cases['support_on_periodic_face'] = (on_face, 'error', 'periodic face')
    cases['pml_on_periodic_face'] = (_rules_base(pml_x_hi=True, target_frequency=1e9), 'error', 'pml on the periodic')
    cases['axisymmetric'] = (_rules_base(geometry='axisymmetric'), 'error', 'geometry axisymmetric')
    cases['stress_stiffening'] = (_rules_base(stress_stiffening=True), 'error', 'stress_stiffening')
    free = _rules_base('tensor', bloch_kx=0.0)
    free.physics[0].features = [feature for feature in free.physics[0].features if feature.type != 'fixed']
    cases['free_trivial_phase'] = (free, 'error', 'can slide')
    cases['ignored_bloch_ky'] = (_rules_base(bloch_ky=1e6), 'warning', 'bloch_ky')
    errors, details = {}, {}
    for name, (spec, level, text) in cases.items():
        issues = validate(spec)
        found = [issue.message for issue in issues if issue.level == 'error']
        if level is None:
            passed = not found
        else:
            found = [issue.message for issue in issues if issue.level == level and text in issue.message]
            passed = bool(found)
        errors[name] = 0.0 if passed else math.inf
        details[name] = found[:2]
    return scaled_worst_result(errors, {name: 1.0 for name in errors}, **details)


# ---- template 'solid_bloch_band_diagram': a square phononic crystal of holes, bands over k_x ----
BAND_DIAGRAM_LABEL = 'Phononic crystal band diagram (2D)'
BAND_DIAGRAM_DESCRIPTION = (
    'Solid mechanics (2D, plane strain) on a quadratic triangle mesh: one unit cell (a = 1 um) of a square lattice of '
    'circular holes (radius 0.35 a) in silicon, periodic along x and y with bloch_kx = k_fraction pi / a swept over '
    '0-1 (Gamma-X) and bloch_ky 0; an eigenfrequency study per k lists the 8 bands nearest 1 GHz (resonance_table; '
    'the target > 0 drops the two rigid translations at Gamma). The bands per k form the in-plane band diagram; gaps '
    'between them are the phononic band gaps of the lattice.')
_TEMPLATE_PERIOD = 1e-6
_TEMPLATE_RADIUS = 0.35e-6
_TEMPLATE_K_FRACTIONS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
_TEMPLATE_TARGET = 1e9
_TEMPLATE_BANDS = 8


def solid_bloch_band_diagram_spec():
    """Template 'solid_bloch_band_diagram': square lattice of holes in silicon, k sweep along x, eigenfrequency study."""
    centre = 0.5 * _TEMPLATE_PERIOD
    settings = {'discretization': 'quadratic', 'periodic_x': True, 'periodic_y': True,
                'bloch_kx': 'k_fraction * pi / a', 'target_frequency': _TEMPLATE_TARGET}
    return ModelSpec(
        title=BAND_DIAGRAM_LABEL,
        parameters=[ParameterSpec('a', _TEMPLATE_PERIOD, 'lattice constant [m]'),
                    ParameterSpec('k_fraction', _TEMPLATE_K_FRACTIONS[-1],
                                  'Bloch kx in units of pi / a (Gamma-X: 0-1)')],
        domain=DomainSpec(size_x=_TEMPLATE_PERIOD, size_y=_TEMPLATE_PERIOD),
        geometry=[GeometrySpec('hole', 'circle', {'cx': centre, 'cy': centre, 'r': _TEMPLATE_RADIUS})],
        materials=[MaterialSpec('silicon', properties=_elastic(169e9, 0.28, rho=2329.0), selection=['all']),
                   MaterialSpec('hole_void', properties={'youngs_modulus': 0.0}, selection=['hole'])],
        physics=[PhysicsSpec(_SOLID, _SOLID_TYPE, settings=settings,
                             features=[FeatureSpec(_TABLE, 'resonance_table')])],
        meshes=[MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=0.1e-6)],
        study=StudySpec(type='eigenfrequency', settings={'n_resonances': _TEMPLATE_BANDS},
                        sweep=SweepSpec('k_fraction', list(_TEMPLATE_K_FRACTIONS))))


BLOCH_2D_CASES = (
    SelfTestCase(
        name='solid_bloch_bulk_bands_2d', label='Solid mechanics 2D: Bloch bulk bands (folded sound lines)',
        description=('eigenfrequency study of solid_mechanics on a 1 um square cell, periodic along x and y, bloch_k = '
                     '(0.3, 0.17) pi / a, isotropic (E 100 GPa, nu 0.25, rho 2000), no supports: the 6 lowest bands '
                     'against the folded plane-strain sound lines c_L |k + G| and c_T |k + G|. Runs tensor (Q1, a / '
                     f"32; {_BULK_RUNS['tensor'][2]:g}), triangles (P2, h a / 8; {_BULK_RUNS['triangles'][2]:g}), "
                     'shifted (tensor at bloch_kx + 2 pi / a: the same bands within 1e-8), guided (tensor, '
                     'axial_wavenumber 0.4 pi / a: c |(k + G, k_z)|, the transverse line twice), gamma_tensor / '
                     'gamma_triangles (k = 0, target 0.5 c_T / a: the two free translations counted and dropped, the '
                     '4-fold c_T 2 pi / a band). Pins both periodic identifications (sides, corners), the complex '
                     'Hermitian pencil, the periodic rigid-mode counts and the guided Bloch pencil. measured = error = '
                     'the worst deviation in tolerance units, tolerance 1 (first estimates).'),
        specs=lambda: {run: _bulk_spec(run) for run in _BULK_RUNS}, check=_check_bulk),
    SelfTestCase(
        name='solid_bloch_rayleigh_2d', label='Solid mechanics 2D: Rayleigh wave of a periodic slab',
        description=('eigenfrequency study of a periodic half-space slab (period 1 um along x, k = pi / (2 L), free '
                     'top; isotropic E 100 GPa, nu 0.25): tensor (Q1 L / 16, 8 um deep, Fixed bottom), triangles (P2 h '
                     'L / 4, same) and pml (triangles, 3 um above an 8-layer elastic PML on y_lo, target at the '
                     'Rayleigh frequency: periodic sides beside a PML ring, the general LU of the complex Bloch '
                     'reduction of the stretched pencil). The lowest mode speed 2 pi f / k against the Rayleigh root '
                     f'c_R = 0.9194 c_T within {_RAYLEIGH_TOLERANCE:g}. measured = error = the worst deviation in '
                     'tolerance units, tolerance 1.'),
        specs=lambda: {run: _slab_spec(run) for run in _SLAB_RUNS}, check=_check_rayleigh),
    SelfTestCase(
        name='solid_bloch_lamb_2d', label='Solid mechanics 2D: Lamb waves of a free plate cell',
        description=('eigenfrequency study of a free 0.5 um plate (isotropic E 100 GPa, nu 0.25), periodic along x '
                     '(period 1 um, k = pi / (2 L)), no supports: the 3 lowest modes (A0, S0, the folded A0) against '
                     'the Rayleigh-Lamb roots at k and k +- 2 pi / L (brentq on an omega scan). Runs triangles (P2 h / '
                     f"4; {_PLATE_RUNS['triangles'][2]:g}) and tensor (Q1 h / 16; {_PLATE_RUNS['tensor'][2]:g}). "
                     'measured = error = the worst deviation in tolerance units, tolerance 1 (first estimates).'),
        specs=lambda: {run: _plate_spec(run) for run in _PLATE_RUNS}, check=_check_lamb),
    SelfTestCase(
        name='solid_bloch_ln_saw_2d', label='Solid mechanics 2D: YZ lithium niobate SAW on a periodic cell',
        description=('eigenfrequency study, piezoelectric, rectangular cells (Q1, L / 16): YZ lithium niobate (crystal '
                     'entry euler 0, 90, 90, plane sagittal), period L = 1 um along x at k = pi / (2 L), 8 um deep '
                     'with a Fixed bottom, no ground electrode (k != 0). free: 3 um of void (vacuum dielectric) above '
                     'the surface; metallised: an Electrode at 0 V on y_hi. The lowest mode speed against the '
                     'literature YZ LN SAW speeds 3488 / 3404 m/s within '
                     f'{_SAW_TOLERANCE:g} (sanity check of the transcribed crystal data and the coupled Bloch pencil: '
                     'u and phi share the phase; K^2 in the details); grating: a shorted strip electrode_domain '
                     '(metallisation 0.5) at the zone edge k = pi / p, its two stopband edges inside [0.98 v_metal, '
                     '1.02 v_free] with a relative gap in (0.1 %, 5 %). measured = error = the worst deviation in '
                     'tolerance units, tolerance 1.'),
        specs=lambda: {run: _saw_spec(run) for run in _SAW_RUNS}, check=_check_saw),
    SelfTestCase(
        name='solid_bloch_rules_2d', label='Solid mechanics 2D: validation of periodic cells',
        description=('model.validate only (no run): a valid periodic cell (triangles, periodic x, k != 0, Fixed bottom) '
                     'has no error; errors for the stationary study, a support on a periodic face, a PML on a periodic '
                     'face, geometry axisymmetric, stress_stiffening and a free cell at bloch_k 0 with target 0; a '
                     'warning for bloch_ky with periodic_y off. error = failed items, tolerance 1.'),
        specs=dict, check=_check_rules),
)
