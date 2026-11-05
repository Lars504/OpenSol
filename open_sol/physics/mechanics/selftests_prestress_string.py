"""Self-tests of the prestressed Si3N4 nanostring of the Hirsch tutorial (hirsch2025 plan E1, new_modules_plan K2) on
triangles in plane stress: a prescribed tension (variant A, f1 .. f5) and a released equibiaxial film stress (variant B,
the mid-span stress and f1 .. f3) against the exact clamped-clamped tensioned-beam roots; definition registers them."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import float32_value, last_value, scaled_worst_result
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           SolverDefaultsSpec, StudySpec)

_SOLID = 'solid'
_SOLID_TYPE = 'solid_mechanics'   # = selftests.SOLID_TYPE_NAME
_TABLE = 'modes'

# ---- solid_fem_prestressed_string (hirsch E1): a 100 um x 100 nm strip in the drawing plane (plane stress: the 100 nm
# depth is free, thickness setting 100 nm), Fixed x_lo / x_hi, Si3N4 (E 250 GPa, nu 0.23, rho 3200), P2 triangles of
# 50 nm (2 across, <= 80 nm everywhere; bending boundary layer delta = h sqrt(E / (12 sigma)) ~ 0.46 um = 9 elements).
# The in-plane flexural modes are those of the narrow Euler-Bernoulli beam E I w'''' - T w'' = rho A omega^2 w (I = d
# h^3 / 12, T = sigma d h, A = d h: d cancels) clamped at both ends: k1^2 = s+, k2^2 = -s- of E I s^2 - T s - rho A
# omega^2 = 0, symmetric modes k2 sin(k2 L / 2) + k1 cos(k2 L / 2) tanh(k1 L / 2) = 0, antisymmetric k1 sin(k2 L / 2) -
# k2 cos(k2 L / 2) tanh(k1 L / 2) = 0 (the plan's reference algorithm). A: initial_stress_xx = 1 GPa: the clamped strip
# keeps u = 0 (the initial-stress force acts on the held end faces only), so sigma = 1 GPa (f1 .. f5 = 2.82113 ..
# 14.14048 MHz). B: initial_stress_xx = _yy = 1 GPa released by the static solve of stress_stiffening: sigma_yy -> 0
# and sigma_xx -> sigma0 (1 - nu) = 0.770 GPa away from the clamps (f1 .. f3 = 2.47879 .. 7.44434 MHz); a stationary run
# of B probes the mid-span stress. Tolerances: the plan's 1e-3 (f), 5e-3 (mid-span stress); the computed roots must
# match the plan's tabulated ones to 1e-5 (pins the reference itself).
_LENGTH = 100e-6
_HEIGHT = 100e-9
_MATERIAL = (250e9, 0.23, 3200.0)    # E [Pa], nu, rho [kg/m^3]
_FILM_STRESS = 1e9                   # Pa
_ELEMENT = 50e-9
_MODES_A = 5
_MODES_B = 3
_PLAN_A = (2.82113e6, 5.64400e6, 8.47036e6, 11.30194e6, 14.14048e6)   # Hz, hirsch plan section 6
_PLAN_B = (2.47879e6, 4.95958e6, 7.44434e6)
_PLAN_RELEASED_STRESS = 0.770e9
_FREQUENCY_TOLERANCE = 1e-3
_STRESS_TOLERANCE = 5e-3
_REFERENCE_TOLERANCE = 1e-5          # computed roots vs the plan's 6-digit values


def _material(released):
    youngs_modulus, poisson_ratio, density = _MATERIAL
    material = {'youngs_modulus': youngs_modulus, 'poisson_ratio': poisson_ratio, 'rho': density,
                'initial_stress_xx': _FILM_STRESS}
    if released:
        material['initial_stress_yy'] = _FILM_STRESS
    return material


def _string_spec(title, released, study, features, geometry=()):
    """Plane-stress strip on a uniform triangle mesh (cpu backend), clamped at both ends."""
    settings = {'out_of_plane': 'plane_stress', 'thickness': _HEIGHT, 'stress_stiffening': True,
                'discretization': 'quadratic'}
    return ModelSpec(
        title=title, domain=DomainSpec(size_x=_LENGTH, size_y=_HEIGHT), geometry=list(geometry),
        materials=[MaterialSpec('nitride', properties=_material(released), selection=['all'])],
        physics=[PhysicsSpec(_SOLID, _SOLID_TYPE, settings=settings,
                             features=[FeatureSpec('clamps', 'fixed', ['x_lo', 'x_hi']), *features])],
        meshes=[MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=_ELEMENT)],
        solver_defaults=SolverDefaultsSpec(backend='cpu'), study=study)


def _specs():
    table = [FeatureSpec(_TABLE, 'resonance_table')]
    probes = [FeatureSpec(f'{name}_mid', 'solid_probe', ['mid'], {'component': name})
              for name in ('stress_xx', 'stress_yy')]
    mid = GeometrySpec('mid', 'point', {'x': 0.5 * _LENGTH, 'y': 0.5 * _HEIGHT})
    return {
        'prescribed': _string_spec('Self-test: nanostring under a prescribed tension (hirsch E1 A)', False,
                                   StudySpec(type='eigenfrequency', settings={'n_resonances': _MODES_A}), table),
        'released': _string_spec('Self-test: nanostring released from an equibiaxial film stress (hirsch E1 B)', True,
                                 StudySpec(type='eigenfrequency', settings={'n_resonances': _MODES_B}), table),
        'released_static': _string_spec('Self-test: released nanostring, static stress (hirsch E1 B)', True,
                                        StudySpec(type='stationary'), probes, geometry=[mid])}


def _beam_frequencies(stress, n_modes):
    """[f_n] [Hz] of the clamped-clamped tensioned narrow beam (module comment) for the stored (float32) constants."""
    from scipy.optimize import brentq

    youngs_modulus, _, density = (float32_value(value) for value in _MATERIAL)
    bending = youngs_modulus * _HEIGHT ** 2 / 12.0   # E I / A [Pa m^2]
    half = 0.5 * _LENGTH

    def residual(frequency, symmetric):
        omega = 2.0 * math.pi * frequency
        root = math.sqrt(stress * stress + 4.0 * bending * density * omega * omega)
        k1 = math.sqrt((stress + root) / (2.0 * bending))
        k2 = math.sqrt((root - stress) / (2.0 * bending))
        damping = math.tanh(k1 * half)
        if symmetric:
            return k2 * math.sin(k2 * half) + k1 * math.cos(k2 * half) * damping
        return k1 * math.sin(k2 * half) - k2 * math.cos(k2 * half) * damping

    ratio = _HEIGHT / _LENGTH * math.sqrt(youngs_modulus / (12.0 * stress))
    frequencies = []
    for n in range(1, n_modes + 1):
        estimate = (n / (2.0 * _LENGTH) * math.sqrt(stress / density)
                    * (1.0 + 2.0 * ratio + (1.0 + 0.5 * n * n * math.pi ** 2) * ratio ** 2))
        grid = np.linspace(0.95 * estimate, 1.05 * estimate, 401)
        values = [residual(value, n % 2 == 1) for value in grid]
        brackets = [(grid[i], grid[i + 1]) for i in range(len(grid) - 1) if values[i] * values[i + 1] <= 0.0]
        if not brackets:
            frequencies.append(math.nan)
            continue
        low, high = min(brackets, key=lambda pair: abs(0.5 * (pair[0] + pair[1]) - estimate))
        frequencies.append(brentq(residual, low, high, args=(n % 2 == 1,), xtol=1e-9 * estimate))
    return frequencies


def _table_frequencies(results):
    table = results.result(_SOLID, _TABLE)
    columns = table['columns']
    return sorted(float(row[columns.index('frequency')]) for row in table['rows'])


def _check(results_by_name):
    """Table frequencies of A and B against the beam roots, the roots against the plan, the released mid-span stress."""
    _, poisson_ratio, _ = (float32_value(value) for value in _MATERIAL)
    released_stress = _FILM_STRESS * (1.0 - poisson_ratio)
    errors, tolerances, details = {}, {}, {}
    for run, stress, n_modes, plan in (('prescribed', _FILM_STRESS, _MODES_A, _PLAN_A),
                                       ('released', released_stress, _MODES_B, _PLAN_B)):
        expected = _beam_frequencies(stress, n_modes)
        measured = _table_frequencies(results_by_name[run])
        errors[f'{run} reference'] = max(abs(value / reference - 1.0) for value, reference in zip(expected, plan))
        tolerances[f'{run} reference'] = _REFERENCE_TOLERANCE
        errors[f'{run} frequency'] = (max(abs(value / reference - 1.0) for value, reference in zip(measured, expected))
                                      if len(measured) >= n_modes else math.inf)
        tolerances[f'{run} frequency'] = _FREQUENCY_TOLERANCE
        details[run] = {'measured': measured, 'expected': expected, 'plan': list(plan), 'stress': stress}
    static = results_by_name['released_static']
    stress_xx, stress_yy = last_value(static, _SOLID, 'stress_xx_mid'), last_value(static, _SOLID, 'stress_yy_mid')
    errors['released mid-span stress_xx'] = abs(stress_xx / released_stress - 1.0)
    errors['released mid-span stress_yy'] = abs(stress_yy) / released_stress
    errors['released stress vs plan'] = abs(released_stress / _PLAN_RELEASED_STRESS - 1.0)
    tolerances.update({'released mid-span stress_xx': _STRESS_TOLERANCE, 'released mid-span stress_yy': _STRESS_TOLERANCE,
                       'released stress vs plan': 1e-3})
    details['released_static'] = {'stress_xx': stress_xx, 'stress_yy': stress_yy, 'expected': released_stress}
    return scaled_worst_result(errors, tolerances, **details)


PRESTRESS_STRING_CASES = (
    SelfTestCase(
        name='solid_fem_prestressed_string',
        label='Solid mechanics on triangles: prestressed nanostring (prescribed and released stress, hirsch E1)',
        description=('A 100 um x 100 nm Si3N4 strip (E 250 GPa, nu 0.23, rho 3200) in plane stress on 50 nm P2 '
                     'triangles, Fixed at both ends, stress_stiffening: initial_stress_xx = 1 GPa (A) gives the in-plane '
                     f'flexural f1 .. f{_MODES_A} of the clamped tensioned beam (exact roots of E I w\'\'\'\' - T w\'\' = '
                     'rho A omega^2 w), the equibiaxial 1 GPa film (B) released by the static state gives sigma_xx = '
                     f'sigma0 (1 - nu) = 0.770 GPa at mid-span (a stationary run, {_STRESS_TOLERANCE:g}) and f1 .. '
                     f'f{_MODES_B}; frequencies within {_FREQUENCY_TOLERANCE:g}, the computed roots against the plan '
                     f'table within {_REFERENCE_TOLERANCE:g}. measured = error in tolerance units, tolerance 1 (first '
                     'estimates, not yet run).'),
        specs=_specs, check=_check),
)
