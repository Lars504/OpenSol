"""Self-tests of the plate PML (experiment plan F1): a simply supported plate strip guide ending in a PML (two-wave fit,
analytic plate dispersion) and the fem.pml stretch derivative against central differences."""
import math
from dataclasses import replace

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import float32_value, scaled_worst_result
from ...model.spec import DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, ModelSpec, PhysicsSpec, StudySpec
from .selftests import (_AREAL_DENSITY, _FILM, _GUIDE_FORCE, _GUIDE_WIDTH, _MEMBRANE_TYPE, _PHYSICS, _TENSION,
                        _plate_frequency, _probe_phasors, _solver_defaults, _triangles)

# ---- membrane_plate_guide_pml: a strip guide of width W, y walls simply supported (w = 0, M_nn = 0: sin(n pi y / W)
# separates), D = T (W / pi)^2 (bending = tension at the first cut-off), window 4 W, PML ring of 24 h on x_hi (0.75 W,
# 24 element layers: the default 8 layers of 3 h reflected |B / A| = 0.79, a plate ring needs one layer per cell),
# line force on x_lo. Plate dispersion rho_s omega^2 = T Q + D Q^2, Q = beta^2 + (n pi / W)^2: the drive sets Q+ = (1.5
# pi / W)^2, so mode 1 propagates with beta = (pi / W) sqrt(1.25), its bending partner Q- = -3.25 (pi / W)^2 and mode 3
# are evanescent (< 1e-4 of mode 1 at the first probe). Fit w = A exp(-j beta x) + B exp(+j beta x) on the centre line
# with beta free: |B / A| < 1e-2 and beta vs the analytic root 1e-3 (C0 interior penalty: O((k h)^2) dispersion error,
# k h = 0.15 at h = W / 32; first-estimate tolerances).
_PLATE_GUIDE_RIGIDITY = float32_value(_TENSION * (_GUIDE_WIDTH / math.pi) ** 2)   # N m
_PLATE_GUIDE_H = _GUIDE_WIDTH / 32.0
_PLATE_GUIDE_PML_CELLS = 24
_PLATE_GUIDE_FACTOR = 1.5          # sqrt(Q+) / (pi / W) at the drive
_PLATE_GUIDE_PROBES = {f'probe_{index}': 1.5 + 0.25 * index for index in range(7)}   # x / W
_PLATE_GUIDE_REFLECTION_LIMIT = 1e-2
_PLATE_GUIDE_BETA_TOLERANCE = 1e-3
_PLATE_GUIDE_RESIDUAL_LIMIT = 1e-2
_STRETCH_TOLERANCE = 1e-6          # central difference of a quadratic s (exact up to rounding ~1e-9 here)


def _plate_guide_frequency():
    """Drive frequency [Hz]: Q+ = (1.5 pi / W)^2 on the float32 material values."""
    squared = (_PLATE_GUIDE_FACTOR * math.pi / _GUIDE_WIDTH) ** 2
    return _plate_frequency(squared, _PLATE_GUIDE_RIGIDITY, _TENSION, _AREAL_DENSITY)


def _plate_guide_beta(frequency):
    """Analytic beta [rad/m] of mode 1: Q+ = 2 rho_s omega^2 / (T + sqrt(T^2 + 4 D rho_s omega^2)) - (pi / W)^2."""
    omega = 2.0 * math.pi * frequency
    squared = (2.0 * _AREAL_DENSITY * omega ** 2
               / (_TENSION + math.sqrt(_TENSION ** 2 + 4.0 * _PLATE_GUIDE_RIGIDITY * _AREAL_DENSITY * omega ** 2)))
    return math.sqrt(squared - (math.pi / _GUIDE_WIDTH) ** 2)


def _plate_guide_specs():
    points = [GeometrySpec(name, 'point', {'x': fraction * _GUIDE_WIDTH, 'y': 0.5 * _GUIDE_WIDTH})
              for name, fraction in _PLATE_GUIDE_PROBES.items()]
    features = [FeatureSpec('walls', 'simply_supported', ['y_lo', 'y_hi']),
                FeatureSpec('drive', 'line_force', ['x_lo'], {'force': _GUIDE_FORCE})]
    features += [FeatureSpec(f'{name}_w', 'field_probe', [name]) for name in _PLATE_GUIDE_PROBES]
    material = {**_FILM, 'bending_stiffness': _PLATE_GUIDE_RIGIDITY, 'plate_poisson_ratio': float32_value(0.25)}
    return {'guide': ModelSpec(
        title='Self-test: PML-terminated plate strip guide', domain=DomainSpec(size_x=4.0 * _GUIDE_WIDTH,
                                                                               size_y=_GUIDE_WIDTH),
        geometry=points, materials=[MaterialSpec('plate', properties=material, selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS, _MEMBRANE_TYPE, settings={'pml_x_hi': True,
                                                                 'pml_cells': _PLATE_GUIDE_PML_CELLS},
                             features=features)],
        meshes=[replace(_triangles(_PLATE_GUIDE_H), pml_layers=_PLATE_GUIDE_PML_CELLS)],
        solver_defaults=_solver_defaults('cpu'),
        study=StudySpec(type='frequency_response', settings={'frequencies': [_plate_guide_frequency()]}))}


def _two_wave_fit(positions, values, beta):
    """((A, B), relative residual) of the least-squares w = A exp(-j beta x) + B exp(+j beta x)."""
    basis = np.stack([np.exp(-1j * beta * positions), np.exp(1j * beta * positions)], axis=1)
    amplitudes, *_ = np.linalg.lstsq(basis, values, rcond=None)
    return amplitudes, float(np.linalg.norm(basis @ amplitudes - values) / np.linalg.norm(values))


def _check_plate_guide(results_by_name):
    import scipy.optimize
    results = results_by_name['guide']
    positions = np.array([fraction * _GUIDE_WIDTH for fraction in _PLATE_GUIDE_PROBES.values()])
    values = np.array([_probe_phasors(results, f'{name}_w')[1][0] for name in _PLATE_GUIDE_PROBES])
    expected = _plate_guide_beta(_plate_guide_frequency())
    found = scipy.optimize.minimize_scalar(lambda beta: _two_wave_fit(positions, values, beta)[1],
                                           bounds=(0.9 * expected, 1.1 * expected), method='bounded',
                                           options={'xatol': 1e-9 * expected})
    beta = float(found.x)
    (forward, backward), residual = _two_wave_fit(positions, values, beta)
    reflection = abs(backward) / abs(forward)
    errors = {'reflection': reflection, 'beta': abs(beta / expected - 1.0), 'two_wave_residual': residual}
    return scaled_worst_result(errors, {'reflection': _PLATE_GUIDE_REFLECTION_LIMIT, 'beta': _PLATE_GUIDE_BETA_TOLERANCE,
                                        'two_wave_residual': _PLATE_GUIDE_RESIDUAL_LIMIT},
                               beta_fitted=beta, beta_expected=expected, forward_abs=abs(forward),
                               backward_abs=abs(backward), reflection_dB=20.0 * math.log10(max(reflection, 1e-300)),
                               frequency_hz=_plate_guide_frequency())


def _check_stretch_derivatives(_results_by_name):
    """fem.pml PMLStretch.stretch_derivatives against central differences of stretches (layers on three faces)."""
    from ...fem.pml import pml_stretch
    stretch = pml_stretch((0.0, 0.0, 1.0, 1.0), {'x_lo': 0.2, 'x_hi': 0.3, 'y_hi': 0.25}, 1.0, 2.0 * math.pi)
    x, y = np.meshgrid(np.linspace(-0.19, 1.29, 37), np.linspace(0.05, 1.24, 31), indexing='ij')
    points = np.stack([x.ravel(), y.ravel()], axis=-1)   # none within 1e-3 of a window edge (s'' jumps there)
    step = 1e-6
    analytic = stretch.stretch_derivatives(points)
    numeric = np.empty_like(analytic)
    for axis in range(2):
        shift = np.zeros(2)
        shift[axis] = step
        numeric[:, axis] = (stretch.stretches(points + shift)[:, axis]
                            - stretch.stretches(points - shift)[:, axis]) / (2.0 * step)
    scale = float(np.abs(analytic).max())
    error = float(np.abs(numeric - analytic).max()) / scale
    return scaled_worst_result({'derivative': error}, {'derivative': _STRETCH_TOLERANCE},
                               largest_derivative=scale, strength=dict(stretch.strength))


MEMBRANE_PLATE_PML_CASES = (
    SelfTestCase(
        'membrane_plate_guide_pml', 'Membrane: PML-terminated plate strip guide (stretched interior penalty)',
        'A plate strip guide W = 0.2 mm (T = 50 N/m, rho_s = 1.55e-4 kg/m^2, D = T (W / pi)^2), y walls simply '
        f'supported, window 4 W with a PML ring of {_PLATE_GUIDE_PML_CELLS} h ({_PLATE_GUIDE_PML_CELLS} element layers) '
        'on x_hi, line force on x_lo at Q+ = (1.5 '
        'pi / W)^2 (mode 1 propagates, its bending partner and mode 3 evanescent), quadratic triangles h = W / 32: a '
        'two-wave least-squares fit of seven centre-line probes with beta free, |B / A| below '
        f'{_PLATE_GUIDE_REFLECTION_LIMIT:g}, beta against the analytic plate root to {_PLATE_GUIDE_BETA_TOLERANCE:g} '
        f'and the fit residual below {_PLATE_GUIDE_RESIDUAL_LIMIT:g} (first estimates).',
        _plate_guide_specs, _check_plate_guide),
    SelfTestCase(
        'membrane_pml_stretch_derivatives', 'Membrane: PML stretch derivative ds / dx (fem.pml)',
        'PMLStretch.stretch_derivatives of layers on x_lo, x_hi and y_hi against central differences (step 1e-6 m) '
        f'of PMLStretch.stretches on a grid through window and layers, to {_STRETCH_TOLERANCE:g} of the largest '
        'derivative (s is quadratic in depth: exact up to rounding). No model run.',
        lambda: {}, _check_stretch_derivatives),
)
