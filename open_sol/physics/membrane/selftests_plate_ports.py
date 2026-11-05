"""Self-tests of the plate ports (experiment plan F2): a simply supported plate strip between two ports (transmission 1,
power balance, beta), a clamped plate step against the same step ending in a plate PML (F1), and the film limit D -> 0
of the port tunnelling model."""
import math
from dataclasses import replace

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import float32_value, scaled_worst_result
from ...model.spec import DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, ModelSpec, PhysicsSpec, StudySpec
from .selftests import (_AREAL_DENSITY, _FILM, _GUIDE_WIDTH, _MEMBRANE_TYPE, _PHYSICS, _TENSION, _TUNNEL_CASES,
                        _plate_strip_frequency, _solver_defaults, _triangles, _tunnel_spec)
from .selftests_plate_pml import _PLATE_GUIDE_RIGIDITY, _plate_guide_beta, _plate_guide_frequency

_PORT_AMPLITUDE = 1e-9            # m
_PORT_MODES = 4
_PLATE_POISSON = float32_value(0.25)


def _port_value(results, port, name):
    return float(np.asarray(results.result(_PHYSICS, port)['series'][name], dtype=np.float64)[0])


def _port_fractions(results, port, count):
    """Sum of power_fraction_1 .. count of a port at the (single) frequency."""
    return sum(_port_value(results, port, f'power_fraction_{number}') for number in range(1, count + 1))


def _port_features(in_face, out_face, count):
    features = [FeatureSpec('in_port', 'port', [in_face], {'mode_count': count, 'incident_mode': 1,
                                                          'amplitude': _PORT_AMPLITUDE})]
    if out_face is not None:
        features.append(FeatureSpec('out_port', 'port', [out_face], {'mode_count': count}))
    return features


# ---- membrane_plate_port_supported: the plate strip of membrane_plate_guide_pml (W = 0.2 mm, y walls simply supported,
# D = T (W / pi)^2, Q+ = (1.5 pi / W)^2: mode 1 propagates, its bending partner and mode 2 / 3 evanescent) between a
# port on x_lo (incident mode 1, A) and one on x_hi, L = W, h = W / 32. The guide is separable (sin(n pi y / W)), so mode
# 1 crosses unchanged: reflected power (every in_port fraction) ~ 0, the outgoing powers sum to the incident one (exact
# discrete balance) and the outgoing mode 1 at x_hi = A exp(-j beta L) with the analytic plate root (FE dispersion of
# the C0 interior penalty ~ 1e-3 in beta at k h = 0.15; first-estimate tolerances).
_SUPPORTED_PORT_H = _GUIDE_WIDTH / 32.0
_SUPPORTED_PORT_LENGTH = _GUIDE_WIDTH
_SUPPORTED_REFLECTION_LIMIT = 1e-5
_PORT_BALANCE_TOLERANCE = 1e-6
_SUPPORTED_PHASE_TOLERANCE = 1e-2


def _plate_material(rigidity):
    return {**_FILM, 'bending_stiffness': rigidity, 'plate_poisson_ratio': _PLATE_POISSON}


def _supported_port_specs():
    features = [FeatureSpec('walls', 'simply_supported', ['y_lo', 'y_hi'])] + _port_features('x_lo', 'x_hi',
                                                                                           _PORT_MODES)
    return {'strip': ModelSpec(
        title='Self-test: simply supported plate strip between two plate ports',
        domain=DomainSpec(size_x=_SUPPORTED_PORT_LENGTH, size_y=_GUIDE_WIDTH),
        materials=[MaterialSpec('plate', properties=_plate_material(_PLATE_GUIDE_RIGIDITY), selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS, _MEMBRANE_TYPE, features=features)],
        meshes=[_triangles(_SUPPORTED_PORT_H)], solver_defaults=_solver_defaults('cpu'),
        study=StudySpec(type='frequency_response', settings={'frequencies': [_plate_guide_frequency()]}))}


def _check_supported_port(results_by_name):
    results = results_by_name['strip']
    reflection = _port_fractions(results, 'in_port', _PORT_MODES)
    transmission = _port_fractions(results, 'out_port', _PORT_MODES)
    outgoing = complex(_port_value(results, 'out_port', 'mode_1_re'), _port_value(results, 'out_port', 'mode_1_im'))
    beta = _plate_guide_beta(_plate_guide_frequency())
    expected = _PORT_AMPLITUDE * np.exp(-1j * beta * _SUPPORTED_PORT_LENGTH)
    errors = {'reflection': reflection, 'balance': abs(transmission + reflection - 1.0),
              'transmitted_phase': abs(outgoing - expected) / _PORT_AMPLITUDE}
    return scaled_worst_result(errors, {'reflection': _SUPPORTED_REFLECTION_LIMIT, 'balance': _PORT_BALANCE_TOLERANCE,
                                        'transmitted_phase': _SUPPORTED_PHASE_TOLERANCE},
                               transmission=transmission, reflection_fraction=reflection,
                               outgoing_mode_1=[outgoing.real, outgoing.imag], expected_mode_1=[expected.real, expected.imag],
                               beta_expected=beta, plate_walls=results.result(_PHYSICS, 'in_port').get('plate_walls'))


# ---- membrane_plate_port_step: clamped plate guide W (y walls clamped) narrowing to W_n = 0.6 W for x < W (a clamped
# hole above y = W_n), D = T (W / pi)^2 / 4 (boundary layer sqrt(D / T) = 5 h at h = W / 32). The frequency lies between
# the first cut-off of the narrow guide and the second of the wide one (exact clamped plate strip roots), so one mode
# propagates on each side. 'ports': in_port on x_lo (incident mode 1, walls the y_lo face and the hole outline), out_port
# on x_hi at 4 W; 'pml': the same with a plate PML (24 h, 24 element layers: 8 gave R = 6.8) on x_hi instead of out_port
# (mode 2 of the wide guide decays by
# 3e-4 before it). Checks: the ports run's power balance (exact) and its transmission against 1 - the reflection of the
# PML run (the PML's own reflection ~1e-2 in amplitude: first-estimate tolerance).
_STEP_NARROW = 0.6
_STEP_RIGIDITY = float32_value(0.25 * _TENSION * (_GUIDE_WIDTH / math.pi) ** 2)
_STEP_H = _GUIDE_WIDTH / 32.0
_STEP_LENGTHS = (1.0, 3.0)        # narrow, wide section / W
_STEP_PML_CELLS = 24
_STEP_PORT_MODES = 6
_STEP_TRANSMISSION_TOLERANCE = 1e-2


def _step_frequency():
    """Geometric mean of the narrow guide's first and the wide guide's second clamped cut-off [Hz]."""
    narrow = _plate_strip_frequency(1, 0.0, _STEP_RIGIDITY, _TENSION, _AREAL_DENSITY, _STEP_NARROW * _GUIDE_WIDTH)
    wide = _plate_strip_frequency(2, 0.0, _STEP_RIGIDITY, _TENSION, _AREAL_DENSITY, _GUIDE_WIDTH)
    if not narrow < wide:
        raise ValueError(f'plate step self-test: no single-mode window (narrow f_1 {narrow:.6g} Hz >= wide f_2 '
                         f'{wide:.6g} Hz)')
    return math.sqrt(narrow * wide)


def _step_spec(name):
    narrow_length, wide_length = (length * _GUIDE_WIDTH for length in _STEP_LENGTHS)
    width, notch_y = _GUIDE_WIDTH, _STEP_NARROW * _GUIDE_WIDTH
    notch = {'x0': 0.0, 'y0': notch_y, 'x1': narrow_length, 'y1': width}
    geometry = [GeometrySpec('box', 'rectangle', {'x0': 0.0, 'y0': 0.0, 'x1': narrow_length + wide_length, 'y1': width}),
                GeometrySpec('notch', 'rectangle', dict(notch)), GeometrySpec('notch_cut', 'rectangle', dict(notch)),
                GeometrySpec('guide', 'difference', {'base': 'box', 'subtract': ['notch_cut']})]
    pml = name == 'pml'
    features = ([FeatureSpec('walls', 'clamped', ['y_lo', 'y_hi', 'notch'])]
                + _port_features('x_lo', None if pml else 'x_hi', _STEP_PORT_MODES))
    settings = {'pml_x_hi': True, 'pml_cells': _STEP_PML_CELLS} if pml else {}
    return ModelSpec(
        title=f'Self-test: clamped plate step ({name})',
        domain=DomainSpec(size_x=narrow_length + wide_length, size_y=width), geometry=geometry,
        materials=[MaterialSpec('plate', properties=_plate_material(_STEP_RIGIDITY), selection=['guide'])],
        physics=[PhysicsSpec(_PHYSICS, _MEMBRANE_TYPE, settings=settings, features=features)],
        meshes=[replace(_triangles(_STEP_H), pml_layers=_STEP_PML_CELLS)], solver_defaults=_solver_defaults('cpu'),
        study=StudySpec(type='frequency_response', settings={'frequencies': [_step_frequency()]}))


def _check_step(results_by_name):
    ports, pml = results_by_name['ports'], results_by_name['pml']
    reflection = _port_fractions(ports, 'in_port', _STEP_PORT_MODES)
    transmission = _port_fractions(ports, 'out_port', _STEP_PORT_MODES)
    pml_reflection = _port_fractions(pml, 'in_port', _STEP_PORT_MODES)
    errors = {'balance': abs(transmission + reflection - 1.0),
              'transmission_vs_pml': abs(transmission - (1.0 - pml_reflection))}
    return scaled_worst_result(errors, {'balance': _PORT_BALANCE_TOLERANCE, 'transmission_vs_pml': _STEP_TRANSMISSION_TOLERANCE},
                               transmission=transmission, reflection=reflection, pml_reflection=pml_reflection,
                               frequency_hz=_step_frequency(),
                               in_port_walls=ports.result(_PHYSICS, 'in_port').get('plate_walls'))


# ---- membrane_plate_port_film_limit: the membrane_port_tunnelling t11 model (mode projections dropped) with a tiny
# bending stiffness D = 1e-16 N m: the plate ports reproduce the film run's T11 with an exact power balance. D k^2 / T is
# not what limits D: a clamped wall's boundary layer delta = sqrt(D / T) moves the outer (film) solution's wall inward by
# delta (also unresolved: the Nitsche clamps capture it), and the barrier near cut-off (kappa^2 = (pi / W2)^2 - k^2 ~ 7 %
# of (pi / W2)^2) turns that into d ln T11 = -4 delta L_b (pi / W2)^2 / (W2 kappa): -18 % at D = 1e-12 (run: -16.6 %),
# -0.2 % at 1e-16.
_FILM_LIMIT_RIGIDITY = float32_value(1e-16)
_FILM_LIMIT_MODES = 10
_FILM_LIMIT_TOLERANCE = 1e-2


def _film_limit_specs():
    spec = _tunnel_spec('t11')
    physics = spec.physics[0]
    plain = replace(physics, features=[feature for feature in physics.features if feature.type != 'mode_projection'])
    material = spec.materials[0]
    plate = replace(material, properties={**material.properties, 'bending_stiffness': _FILM_LIMIT_RIGIDITY})
    return {'film': replace(spec, title='Self-test: port tunnelling, film', physics=[plain]),
            'plate': replace(spec, title='Self-test: port tunnelling, plate D -> 0', physics=[plain],
                             materials=[plate])}


def _check_film_limit(results_by_name):
    film, plate = results_by_name['film'], results_by_name['plate']
    film_t11 = _port_value(film, 'out_port', 'power_fraction_1')
    plate_t11 = _port_value(plate, 'out_port', 'power_fraction_1')
    balance = (_port_fractions(plate, 'in_port', _FILM_LIMIT_MODES)
               + _port_fractions(plate, 'out_port', _FILM_LIMIT_MODES))
    errors = {'t11': abs(plate_t11 / film_t11 - 1.0), 'balance': abs(balance - 1.0)}
    return scaled_worst_result(errors, {'t11': _FILM_LIMIT_TOLERANCE, 'balance': _PORT_BALANCE_TOLERANCE},
                               film_t11=film_t11, plate_t11=plate_t11, reference_t11=_TUNNEL_CASES['t11'][2],
                               plate_walls=plate.result(_PHYSICS, 'in_port').get('plate_walls'))


MEMBRANE_PLATE_PORT_CASES = (
    SelfTestCase(
        'membrane_plate_port_supported', 'Membrane: plate strip between two plate ports (numeric strip modes)',
        'The simply supported plate strip of membrane_plate_guide_pml (W = 0.2 mm, D = T (W / pi)^2, plate_poisson_ratio '
        '0.25) of length W between a port on x_lo (incident mode 1) and one on x_hi, quadratic triangles h = W / 32, at '
        f'Q+ = (1.5 pi / W)^2: the reflected power fraction below {_SUPPORTED_REFLECTION_LIMIT:g}, transmission + '
        f'reflection = 1 to {_PORT_BALANCE_TOLERANCE:g} and the outgoing mode 1 at x_hi against A exp(-j beta W) of the '
        f'analytic plate root ({_SUPPORTED_PHASE_TOLERANCE:g} of A; first estimates).',
        _supported_port_specs, _check_supported_port),
    SelfTestCase(
        'membrane_plate_port_step', 'Membrane: clamped plate step, plate ports against a plate PML',
        f'A clamped plate guide W narrowing to {_STEP_NARROW:g} W over its first W (a clamped hole), D = T (W / pi)^2 / 4, '
        'h = W / 32, one propagating mode on each side: with ports on both ends the power balance to '
        f'{_PORT_BALANCE_TOLERANCE:g}, and the transmission against 1 - the reflection of the same step ending in a '
        f'plate PML ({_STEP_PML_CELLS} h) to {_STEP_TRANSMISSION_TOLERANCE:g} (first estimates).',
        lambda: {name: _step_spec(name) for name in ('ports', 'pml')}, _check_step),
    SelfTestCase(
        'membrane_plate_port_film_limit', 'Membrane: plate ports in the film limit (port tunnelling, D -> 0)',
        'The membrane_port_tunnelling t11 model (5.5 MHz, incident mode 1, 10 modes per port; mode projections dropped) '
        f'with bending_stiffness {_FILM_LIMIT_RIGIDITY:.0e} N m (clamped boundary layer sqrt(D / T) ~ 1 nm, so the '
        'walls stay put): T11 of the plate ports against the same model without bending to '
        f'{_FILM_LIMIT_TOLERANCE:g} relative, and the plate run\'s power balance to {_PORT_BALANCE_TOLERANCE:g} (first '
        'estimates).',
        _film_limit_specs, _check_film_limit),
)
