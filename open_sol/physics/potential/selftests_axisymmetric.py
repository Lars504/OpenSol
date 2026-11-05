"""Analytic self-test of the azimuthal order m = 1 (symmetry 2.14) of axisymmetric (r, z) electric_currents /
electrostatics: a uniform transverse field in a layered cylinder, both kinds as one twin model; registered by
definition."""
import math

import numpy as np

from ...core.constants import EPS0
from ...model.schema import SelfTestCase, SelfTestVariant
from ...model.selftest_helpers import with_exact_cg
from ...model.spec import (DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec,
                           PhysicsSpec, StudySpec)
from .selftests import TWIN_TEXT, merged_kind_check

_RUN = 'run'
_TOLERANCE = 1e-6                  # CG 1e-12 x condition ~1e5, float32 V / J / D storage ~1e-7

# Azimuthal order m = 1 (symmetry 2.14): the uniform transverse field V = -E0 x = -E0 r cos(phi) in a cylinder R x H of
# two z layers (coefficients c_lo below H / 2, c_hi above; tangential E is continuous across them), V_1 = -E0 R held on
# x_hi, insulated z faces, no other anchor. V_1 = -E0 r_c is exact on any r widths: each face carries c A E0 (A = pi r_f
# dz at m != 0), so a cell's net radial outflow c E0 pi dr dz equals its c m^2 / r_c^2 term times its volume pi r_c dr dz
# times r_c. The held face then takes -pi R E0 (c_lo + c_hi) H / 2 (the cos(phi) moment), the pi-ring integral of V is
# -E0 pi H dr sum r_c^2, and of q_joule (currents) V_1(R) x that current.
_TRANSVERSE_CELL = 2.5e-5             # m: 40 x 16 cells
_TRANSVERSE_RADIUS = 1e-3             # R
_TRANSVERSE_HEIGHT = 4e-4             # H, layer interface at H / 2 (a cell edge)
_TRANSVERSE_FIELD = 1e3               # E0 [V/m]: V_1(R) = -1 V
_TRANSVERSE_SETTINGS = {'geometry': 'axisymmetric', 'azimuthal_order': 1}
# kind -> (physics name, type, material property, lower / upper layer values, operator coefficient scale, held
# quantity, flux field)
_TRANSVERSE_KINDS = {'currents': ('currents', 'electric_currents', 'sigma_dc', (3.0, 7.0), 1.0, 'current', 'J_x'),
                     'electrostatics': ('electrostatics', 'electrostatics', 'eps_dc', (2.0, 5.0), EPS0, 'charge',
                                        'D_x')}


def _transverse_specs():
    """'run': the m = 1 layered cylinder of both kinds uncoupled in one model (voltage shield -E0 R on x_hi)."""
    half = 0.5 * _TRANSVERSE_HEIGHT
    physics, layer_lo, layer_hi, derived = [], {}, {}, []
    for kind, entry in _TRANSVERSE_KINDS.items():
        physics_name, physics_type, property_name, (value_lo, value_hi), _, quantity, _ = entry
        layer_lo[property_name], layer_hi[property_name] = value_lo, value_hi
        derived += [DerivedValueSpec(f'F_shield_{kind}', f'{physics_name}_{quantity}_shield', [], 'mean'),
                    DerivedValueSpec(f'V_integral_{kind}', f'{physics_name}_V', [], 'integral')]
        physics.append(PhysicsSpec(physics_name, physics_type, settings=dict(_TRANSVERSE_SETTINGS), features=[
            FeatureSpec('shield', 'voltage', ['x_hi'], {'voltage': -_TRANSVERSE_FIELD * _TRANSVERSE_RADIUS})]))
    derived.append(DerivedValueSpec('P_joule', f"{_TRANSVERSE_KINDS['currents'][0]}_q_joule", [], 'integral'))
    return {_RUN: with_exact_cg(ModelSpec(
        title='Self-test: azimuthal order 1 transverse field (electric currents + electrostatics)',
        domain=DomainSpec(size_x=_TRANSVERSE_RADIUS, size_y=_TRANSVERSE_HEIGHT),
        geometry=[GeometrySpec('lower', 'rectangle', {'x0': 0.0, 'y0': 0.0, 'x1': _TRANSVERSE_RADIUS, 'y1': half}),
                  GeometrySpec('upper', 'rectangle', {'x0': 0.0, 'y0': half, 'x1': _TRANSVERSE_RADIUS,
                                                      'y1': _TRANSVERSE_HEIGHT})],
        materials=[MaterialSpec('layer_lo', properties=layer_lo, selection=['lower']),
                   MaterialSpec('layer_hi', properties=layer_hi, selection=['upper'])],
        physics=physics,
        mesh=MeshSpec(mode='uniform', dx=_TRANSVERSE_CELL),
        study=StudySpec(type='stationary'),
        derived_values=derived))}


def _check_transverse(results_by_name, kind):
    """V_1 = -E0 r_c everywhere, E_x = E0 off the axis cell, E_y = 0, the held cos(phi) moment, the scalar output, the
    pi-ring integral of V and (currents) P_joule = V_1(R) I."""
    physics_name, _, _, (value_lo, value_hi), scale, quantity, flux_field = _TRANSVERSE_KINDS[kind]
    results = results_by_name[_RUN]
    n_radial = int(round(_TRANSVERSE_RADIUS / _TRANSVERSE_CELL))
    n_axial = int(round(_TRANSVERSE_HEIGHT / _TRANSVERSE_CELL))
    potential = np.asarray(results.field(f'{physics_name}.V').values, dtype=np.float64)
    if potential.shape != (n_radial, n_axial):
        raise ValueError(f'expected the window shape {(n_radial, n_axial)}, got {potential.shape}')
    field_x = np.asarray(results.field(f'{physics_name}.E_x').values, dtype=np.float64)
    field_y = np.asarray(results.field(f'{physics_name}.E_y').values, dtype=np.float64)
    density = np.asarray(results.field(f'{physics_name}.{flux_field}').values, dtype=np.float64)
    r_centres = (np.arange(n_radial) + 0.5) * _TRANSVERSE_CELL
    coefficients = scale * np.where(np.arange(n_axial) < n_axial // 2, value_lo, value_hi)
    voltage_scale = _TRANSVERSE_FIELD * _TRANSVERSE_RADIUS
    shield_voltage = -voltage_scale
    expected_flux = -math.pi * _TRANSVERSE_RADIUS * _TRANSVERSE_FIELD * float(np.sum(coefficients)) * _TRANSVERSE_CELL
    expected_integral = (-_TRANSVERSE_FIELD * math.pi * _TRANSVERSE_HEIGHT * _TRANSVERSE_CELL
                         * float(np.sum(r_centres ** 2)))
    shield_flux = float(results.result(physics_name, 'shield')[quantity])
    derived = results.summary()['derived']
    errors = {'profile': float(np.max(np.abs(potential + _TRANSVERSE_FIELD * r_centres[:, None]))) / voltage_scale,
              'field_x': float(np.max(np.abs(field_x[1:] - _TRANSVERSE_FIELD))) / _TRANSVERSE_FIELD,
              'field_y': float(np.max(np.abs(field_y))) / _TRANSVERSE_FIELD,
              'flux_density': float(np.max(np.abs(density[1:] - _TRANSVERSE_FIELD * coefficients[None, :])
                                           / (_TRANSVERSE_FIELD * coefficients[None, :]))),
              'shield': abs(shield_flux / expected_flux - 1.0),
              'scalar_output': abs(float(derived[f'F_shield_{kind}']) / expected_flux - 1.0),
              'V_integral': abs(float(derived[f'V_integral_{kind}']) / expected_integral - 1.0)}
    details = {'errors': errors, f'{quantity}_shield': shield_flux, f'{quantity}_expected': expected_flux,
               'V_integral': float(derived[f'V_integral_{kind}']), 'V_integral_expected': expected_integral,
               'axis_cell_E_x': float(field_x[0, 0]), 'n_cells': list(potential.shape)}
    if kind == 'currents':
        power, expected_power = float(derived['P_joule']), shield_voltage * expected_flux
        errors['power'] = abs(power / expected_power - 1.0)
        details.update(P_joule_W=power, P_expected_W=expected_power)
    error = float(np.max(np.asarray(list(errors.values()), dtype=np.float64)))   # NaN propagates
    return {'measured': shield_flux, 'expected': expected_flux, 'error': error, 'tolerance': _TOLERANCE,
            'details': details}


def _check_transverse_fields(results_by_name):
    """The transverse-field check of both kinds on their one twin run."""
    return merged_kind_check(results_by_name, {kind: lambda results, kind=kind: _check_transverse(results, kind)
                                               for kind in _TRANSVERSE_KINDS})


_TRANSVERSE_TEXT = (
    f'azimuthal_order 1 (symmetry 2.14) on an axisymmetric cylinder R x H = {_TRANSVERSE_RADIUS * 1e3:g} x '
    f'{_TRANSVERSE_HEIGHT * 1e3:g} mm ({_TRANSVERSE_CELL * 1e6:g} um cells) of two z layers (rectangles below / above '
    f'H / 2), insulated z faces, voltage shield V_1 = -E0 R = {-_TRANSVERSE_FIELD * _TRANSVERSE_RADIUS:g} V on x_hi and '
    'no other anchor (the m^2 / r^2 term anchors): the cos(phi) part of the uniform transverse field E0 = '
    f'{_TRANSVERSE_FIELD:g} V/m, V = -E0 r cos(phi). Exact on any r widths: V_1 = -E0 r_c (each face carries c A E0 '
    'with A = pi r_f dz, so the net radial outflow c E0 pi dr dz of a cell equals c m^2 / r_c^2 x pi r_c dr dz x r_c), '
    'E_x = E0 off the axis cell (whose axis face carries nothing: E0 / 2), E_y = 0, the held cos(phi) moment -pi R E0 '
    'sum_layers c H_layer, its scalar output, the pi-ring integral of V -E0 pi H dr sum r_c^2{power}. error = the '
    f'largest relative deviation, tolerance {_TOLERANCE:g}. This pins the c m^2 / r_c^2 diagonal, the azimuthal anchor, '
    'the pi ring weights of faces, volumes, held totals and Session derived values{pins}. Variant cpu: the same run on '
    'the cpu backend (numpy, the exact pcg node), same tolerance.')
_CPU_VARIANTS = (SelfTestVariant('cpu', tolerance=_TOLERANCE, backend='cpu'),)

AZIMUTHAL_SELFTEST_CASES = (
    SelfTestCase(
        name='potential_azimuthal_transverse_field',
        label='Electric currents + electrostatics: azimuthal order 1 transverse field in a layered cylinder',
        description=(f'{TWIN_TEXT} Electric currents with sigma_dc {_TRANSVERSE_KINDS["currents"][3][0]:g} / '
                     f'{_TRANSVERSE_KINDS["currents"][3][1]:g} S/m and electrostatics with eps_dc '
                     f'{_TRANSVERSE_KINDS["electrostatics"][3][0]:g} / {_TRANSVERSE_KINDS["electrostatics"][3][1]:g} '
                     'per layer, each with '
                     + _TRANSVERSE_TEXT.format(
                         power=', and (currents) P_joule = the pi-ring integral of q_joule = V_1(R) I',
                         pins=' and (currents) the azimuthal part c m^2 / r^2 V^2 of q_joule')
                     + ' Not yet run as one twin model.'),
        specs=_transverse_specs, check=_check_transverse_fields, variants=_CPU_VARIANTS),
)
