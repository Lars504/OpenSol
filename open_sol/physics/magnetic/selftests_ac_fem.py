"""Self-tests of magnetic_fields_ac on triangle meshes (MagneticACFEMPhysics): the planar skin-depth slab and the
axisymmetric skin annulus against their continuous closed forms (sinh / modified Bessel), the current-driven solid strip
against a fine finite-volume reference plus the circuit-row and Poynting identities. cpu backend, Automatic host
SuperLU; tolerances are first estimates (not run yet). Registered by definition_ac."""
import math

import numpy as np

from ...core.constants import MU0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import scaled_worst_result
from ...model.spec import (DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec,
                           PhysicsSpec, SolverDefaultsSpec, StudySpec)
from .selftests import _BAND, _LENGTH, _MAGNETIC, _RUN, _WIDTH, _band_rectangle, _derived_value, _mirror_y
from .selftests_ac import (_COPPER, _DRIVE_POTENTIAL, _SKIN_FREQUENCY, _STRIP_CURRENT, _STRIP_FREQUENCY,
                           _STRIP_PHASE, MAGNETIC_AC_TYPE_NAME, _skin_depth, _strip_solution)

_ELEMENT = 5e-6                    # m, uniform triangle size (skin depth 100 um / 50 um: 20 / 10 elements per depth)
_DRIVE_STRIP = 1e-5                # m, width of the held strip of the planar slab
_ANNULUS = (0.2e-3, 0.21e-3)       # m, held annulus of the axisymmetric case (copper from its outer radius to x_hi)
_PSI = 2e-10                       # T m^2, held psi = r A_phi of the annulus (A_phi = 1e-6 T m at r = 0.2 mm)
_PROBE_X = {'slab': (0.1e-3, 0.2e-3, 0.3e-3), 'annulus': (0.25e-3, 0.3e-3, 0.4e-3)}   # m, probes at mid height
_FIELD_TOLERANCE = 1e-3            # relative |A| at the probes, loss and energy against the continuous solution
_REFERENCE_TOLERANCE = 1e-3        # relative R / L / loss of the strip against the 1 um finite-volume reference
_IDENTITY_TOLERANCE = 1e-9         # circuit row and Poynting identity (host float64, exact quadrature)
_REFERENCE_CELLS = 1000            # cells of the finite-volume strip reference (error ~(h / delta)^2 / 12 ~ 3e-5)
_GAUSS_INTERVALS = 2000            # composite Gauss-Legendre intervals of the annulus integrals


# ---- specs ----
def _ac_fem_model(title, features, materials, geometry, frequency, derived_values=(), axisymmetric=False):
    """Stationary magnetic_fields_ac-only P2 model on a uniform 'triangles' mesh, cpu backend."""
    settings = {'frequency': frequency, 'discretization': 'quadratic'}
    if axisymmetric:
        settings['geometry'] = 'axisymmetric'
    return ModelSpec(
        title=title, domain=DomainSpec(size_x=_LENGTH, size_y=_WIDTH), geometry=list(geometry),
        materials=list(materials),
        physics=[PhysicsSpec(_MAGNETIC, MAGNETIC_AC_TYPE_NAME, settings=settings, features=list(features))],
        meshes=[MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=_ELEMENT)],
        solver_defaults=SolverDefaultsSpec(backend='cpu'), study=StudySpec(type='stationary'),
        derived_values=list(derived_values))


def _probes(kind):
    """(probe points, probe features) at _PROBE_X[kind], mid height."""
    points = [GeometrySpec(f'p{index}', 'point', {'x': x, 'y': 0.5 * _WIDTH})
              for index, x in enumerate(_PROBE_X[kind])]
    return points, [FeatureSpec(f'probe{index}', 'probe', [point.name]) for index, point in enumerate(points)]


def _scalar_values():
    return [DerivedValueSpec('P', f'{_MAGNETIC}_eddy_loss', [], 'max'),
            DerivedValueSpec('W', f'{_MAGNETIC}_energy', [], 'max')]


def _skin_slab_specs():
    """'run': copper everywhere, A = 1e-6 T m held on the first 10 um strip, insulation x_hi, symmetry y faces."""
    points, probes = _probes('slab')
    return {_RUN: _ac_fem_model(
        'Self-test: skin-depth slab on triangles (magnetic_fields_ac)',
        [FeatureSpec('drive', 'fixed_potential', ['drive_strip'], {'potential': _DRIVE_POTENTIAL}),
         FeatureSpec('wall', 'magnetic_insulation', ['x_hi']), _mirror_y(), *probes],
        [MaterialSpec('copper', properties={'sigma_dc': _COPPER}, selection=['slab'])],
        [_band_rectangle('slab', 0.0, _LENGTH), _band_rectangle('drive_strip', 0.0, _DRIVE_STRIP), *points],
        _SKIN_FREQUENCY, _scalar_values())}


def _skin_annulus_specs():
    """'run': axisymmetric, psi held on the annulus, copper outside it to x_hi (insulation), air inside, symmetry y."""
    points, probes = _probes('annulus')
    return {_RUN: _ac_fem_model(
        'Self-test: skin-depth annulus on triangles (axisymmetric magnetic_fields_ac)',
        [FeatureSpec('drive', 'fixed_potential', ['ring'], {'potential': _PSI}),
         FeatureSpec('wall', 'magnetic_insulation', ['x_hi']), _mirror_y(), *probes],
        [MaterialSpec('copper', properties={'sigma_dc': _COPPER}, selection=['shell'])],
        [_band_rectangle('ring', *_ANNULUS), _band_rectangle('shell', _ANNULUS[1], _LENGTH), *points],
        _SKIN_FREQUENCY, _scalar_values(), axisymmetric=True)}


def _solid_strip_specs():
    """'run': solid copper strip on the band, current 1 A at 30 deg, insulation x faces (default), symmetry y."""
    return {_RUN: _ac_fem_model(
        'Self-test: current-driven solid strip on triangles (magnetic_fields_ac)',
        [_mirror_y(),
         FeatureSpec('bar', 'coil', ['band'], {'conductor': 'solid', 'drive': 'current', 'current': _STRIP_CURRENT,
                                               'phase': _STRIP_PHASE})],
        [MaterialSpec('copper', properties={'sigma_dc': _COPPER}, selection=['band'])],
        [_band_rectangle('band', *_BAND)], _STRIP_FREQUENCY, [DerivedValueSpec('P', f'{_MAGNETIC}_eddy_loss', [],
                                                                               'max')])}


# ---- continuous solutions ----
def _slab_solution(positions):
    """|A| at positions, loss and energy per m of depth of the held-driven slab: A = A0 sinh(k (L - x)) / sinh(k S),
    k = (1 + j) / delta, S = L - h."""
    delta = _skin_depth(_SKIN_FREQUENCY)
    omega = 2.0 * math.pi * _SKIN_FREQUENCY
    span = _LENGTH - _DRIVE_STRIP
    k = (1.0 + 1.0j) / delta
    potential = _DRIVE_POTENTIAL * np.sinh(k * (_LENGTH - np.asarray(positions))) / np.sinh(k * span)
    ratio = 2.0 * span / delta
    sinh_squared = 0.5 * (math.cosh(ratio) - math.cos(ratio))   # |sinh(k S)|^2
    loss = (_WIDTH * 0.5 * _COPPER * omega ** 2 * _DRIVE_POTENTIAL ** 2 / sinh_squared
            * 0.25 * delta * (math.sinh(ratio) - math.sin(ratio)))
    energy = (_WIDTH / (4.0 * MU0) * _DRIVE_POTENTIAL ** 2 * (2.0 / delta ** 2) / sinh_squared
              * 0.25 * delta * (math.sinh(ratio) + math.sin(ratio)))
    return np.abs(potential), loss, energy


def _annulus_solution(positions):
    """|A_phi| at positions, loss and energy (whole rings) of the axisymmetric case: copper A = c1 I1(k r) + c2 K1(k r)
    with A(r_h) = psi / r_h, A(R) = 0; the air core B_z = 2 psi / r_a^2 uniform."""
    from scipy.special import iv, kv
    inner, outer = _ANNULUS
    omega = 2.0 * math.pi * _SKIN_FREQUENCY
    k = (1.0 + 1.0j) / _skin_depth(_SKIN_FREQUENCY)
    matrix = np.array([[iv(1, k * outer), kv(1, k * outer)], [iv(1, k * _LENGTH), kv(1, k * _LENGTH)]])
    first, second = np.linalg.solve(matrix, np.array([_PSI / outer, 0.0], dtype=np.complex128))

    def potential(r):
        return first * iv(1, k * r) + second * kv(1, k * r)

    def flux_z(r):   # (1 / r) d(r A) / dr
        return k * (first * iv(0, k * r) - second * kv(0, k * r))

    nodes, weights = np.polynomial.legendre.leggauss(5)
    edges = np.linspace(outer, _LENGTH, _GAUSS_INTERVALS + 1)
    half = 0.5 * np.diff(edges)
    radii = (edges[:-1, None] + half[:, None] * (nodes[None, :] + 1.0)).ravel()
    ring_weights = (half[:, None] * weights[None, :]).ravel() * 2.0 * math.pi * radii * _WIDTH
    loss = float(np.sum(ring_weights * 0.5 * _COPPER * omega ** 2 * np.abs(potential(radii)) ** 2))
    core_flux = 2.0 * _PSI / inner ** 2
    energy = (float(np.sum(ring_weights * np.abs(flux_z(radii)) ** 2)) / (4.0 * MU0)
              + core_flux ** 2 / (4.0 * MU0) * math.pi * inner ** 2 * _WIDTH)
    return np.abs(potential(np.asarray(positions))), loss, energy


# ---- checks ----
def _probe_values(results, count):
    values = []
    for index in range(count):
        samples = np.asarray(results.result(_MAGNETIC, f'probe{index}')['values'], dtype=np.float64)
        values.append(float(samples[-1]) if samples.size else math.nan)
    return np.array(values)


def _make_field_check(kind, solution):
    """check of a held-driven skin case: |A| at the probes (relative to the largest), eddy_loss and energy."""
    def check(results_by_name):
        results = results_by_name[_RUN]
        expected_potential, expected_loss, expected_energy = solution(_PROBE_X[kind])
        measured = _probe_values(results, len(_PROBE_X[kind]))
        loss, energy = _derived_value(results, 'P'), _derived_value(results, 'W')
        errors = {'A_probes': float(np.max(np.abs(measured - expected_potential)) / np.max(expected_potential)),
                  'eddy_loss': abs(loss / expected_loss - 1.0), 'energy': abs(energy / expected_energy - 1.0)}
        return scaled_worst_result(errors, dict.fromkeys(errors, _FIELD_TOLERANCE),
                                   A_measured=measured.tolist(), A_expected=expected_potential.tolist(),
                                   loss=loss, loss_expected=expected_loss, energy=energy,
                                   energy_expected=expected_energy)
    return check


def _check_solid_strip(results_by_name):
    """R, L and the loss against the fine finite-volume strip; the circuit row (current = I) and the Poynting identity
    (eddy_loss = Re(V I*) / 2) to _IDENTITY_TOLERANCE."""
    results = results_by_name[_RUN]
    reference = _strip_solution(np.linspace(0.0, _LENGTH, _REFERENCE_CELLS + 1))
    prescribed = _STRIP_CURRENT * np.exp(1j * math.radians(_STRIP_PHASE))
    impedance = reference['v'] / prescribed
    omega = 2.0 * math.pi * _STRIP_FREQUENCY
    expected = {'resistance': impedance.real, 'inductance': impedance.imag / omega, 'loss': reference['loss']}
    row = results.result(_MAGNETIC, 'bar')
    current = complex(row['current_re'], row['current_im'])
    loss = _derived_value(results, 'P')
    measured = {'resistance': float(row['resistance']), 'inductance': float(row['inductance']), 'loss': loss}
    errors = {name: abs(measured[name] / expected[name] - 1.0) for name in expected}
    errors['current_row'] = abs(current - prescribed) / abs(prescribed)
    errors['poynting'] = abs(loss / float(row['power']) - 1.0)
    tolerances = {name: _IDENTITY_TOLERANCE if name in ('current_row', 'poynting') else _REFERENCE_TOLERANCE
                  for name in errors}
    return scaled_worst_result(errors, tolerances, measured=measured, expected=expected,
                               dc_resistance_ohm_per_m=1.0 / (_COPPER * (_BAND[1] - _BAND[0])
                                                              * _WIDTH))


# ---- cases ----
_LAYOUT_TEXT = (f'P2 on uniform {_ELEMENT * 1e6:g} um triangles of a {_LENGTH * 1e3:g} x {_WIDTH * 1e3:g} mm window '
                '(symmetry y faces: 1D in x / r), copper sigma_dc 5.8e7 S/m, cpu backend, one host SuperLU solve')

AC_FEM_SELFTEST_CASES = (
    SelfTestCase(
        name='magnet_ac_fem_skin_slab', label='Magnetic fields AC on triangles: skin depth in a copper slab',
        description=(f'magnetic_fields_ac {_LAYOUT_TEXT}: copper fills the window, fixed_potential A = '
                     f'{_DRIVE_POTENTIAL:g} T m on the first {_DRIVE_STRIP * 1e6:g} um strip, insulation x_hi, f = '
                     f'{_SKIN_FREQUENCY:g} Hz (skin depth {_skin_depth(_SKIN_FREQUENCY) * 1e6:.4g} um). Continuous: A '
                     '= A0 sinh(k (L - x)) / sinh(k (L - h)), k = (1 + j) / delta; |A| at three probe nodes, the '
                     'eddy_loss and energy scalars (closed-form integrals), relative '
                     f'{_FIELD_TOLERANCE:g} each (first estimate: P2 discretisation error). error = the largest error '
                     'over its tolerance; tolerance 1. Not run yet.'),
        specs=_skin_slab_specs, check=_make_field_check('slab', _slab_solution)),
    SelfTestCase(
        name='magnet_ac_fem_skin_annulus', label='Magnetic fields AC on triangles: axisymmetric skin depth',
        description=(f'Axisymmetric magnetic_fields_ac {_LAYOUT_TEXT}: psi = {_PSI:g} T m^2 held on the annulus r in '
                     f'[{_ANNULUS[0] * 1e3:g}, {_ANNULUS[1] * 1e3:g}] mm, copper from its outer radius to x_hi '
                     '(insulation), air core inside (uniform B_z = 2 psi / r_a^2). Continuous: A_phi = c1 I1(k r) + '
                     'c2 K1(k r) in the copper (scipy.special, complex argument); |A_phi| at three probe nodes, '
                     'eddy_loss and energy (whole rings, composite Gauss-Legendre), relative '
                     f'{_FIELD_TOLERANCE:g} each (first estimate). Pins the r weights of the mass, the A_phi form and '
                     'the held psi / r. error = the largest error over its tolerance; tolerance 1. Not run yet.'),
        specs=_skin_annulus_specs, check=_make_field_check('annulus', _annulus_solution)),
    SelfTestCase(
        name='magnet_ac_fem_solid_strip', label='Magnetic fields AC on triangles: current-driven solid strip',
        description=(f'magnetic_fields_ac {_LAYOUT_TEXT}: a solid copper conductor (current {_STRIP_CURRENT:g} A at '
                     f'{_STRIP_PHASE:g} deg) on the band x in [{_BAND[0] * 1e3:g}, {_BAND[1] * 1e3:g}] mm, insulation x '
                     f'faces, f = {_STRIP_FREQUENCY:g} Hz (skin depth {_skin_depth(_STRIP_FREQUENCY) * 1e6:.3g} um). '
                     'Resistance, inductance and eddy_loss against the finite-volume bordered system on '
                     f'{_REFERENCE_CELLS} cells (selftests_ac._strip_solution, error ~3e-5), relative '
                     f'{_REFERENCE_TOLERANCE:g}; the circuit row (result current = I) and the Poynting identity '
                     f'(eddy_loss = Re(V I*) / 2, exact for the shared quadrature) to {_IDENTITY_TOLERANCE:g}. error = '
                     'the largest error over its tolerance; tolerance 1. Not run yet.'),
        specs=_solid_strip_specs, check=_check_solid_strip),
)
