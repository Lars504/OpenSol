"""Shared drift_diffusion self-test fixtures (silicon-like material, n-type bar doping, the abrupt pn junction and its
exact Poisson-Boltzmann check, ring-resistor settings) and the 'silicon_pn_photodiode' template; specs built fresh."""
import math

import numpy as np

from ...core.constants import ELEMENTARY_CHARGE, EPS0, K_B
from ...model.selftest_helpers import worst_error
from ...model.spec import (DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec,
                           ParameterSpec, PhysicsSpec, StudySpec)

DRIFT_DIFFUSION_TYPE_NAME = 'drift_diffusion'   # = definition.DRIFT_DIFFUSION_TYPE_NAME (definition imports this module)
_PHYSICS = 'semiconductor'                      # the type's default instance
_RUN = 'run'
_TEMPERATURE = 300.0                            # K, passed explicitly (independent of the setting's default)
_VT = K_B * _TEMPERATURE / ELEMENTARY_CHARGE   # 25.85 mV
_N_INTRINSIC = 1e16                             # 1/m^3
_MOBILITY_N, _MOBILITY_P = 0.135, 0.048         # m^2/(V s)
_EPS_DC = 11.7
_SETTINGS = {'max_iterations': 100, 'temperature': _TEMPERATURE}


def _silicon(lifetime=1e-6):
    """Silicon-like semiconductor material on 'all' (explicit numbers: the tests do not depend on library data)."""
    return MaterialSpec('silicon', properties={'n_intrinsic': _N_INTRINSIC, 'mobility_n': _MOBILITY_N,
                                               'mobility_p': _MOBILITY_P, 'eps_dc': _EPS_DC, 'lifetime': lifetime},
                        selection=['all'])


def _rectangle(name, x0, x1, height):
    return GeometrySpec(name, 'rectangle', {'x0': x0, 'y0': 0.0, 'x1': x1, 'y1': height})


def _strip_model(title, size, cell, geometry, features, lifetime=1e-6, derived_values=(), extra_settings=None):
    """Stationary drift_diffusion-only strip on a uniform mesh (no EM physics, so no PML)."""
    return ModelSpec(
        title=title,
        domain=DomainSpec(size_x=size[0], size_y=size[1]),
        geometry=list(geometry),
        materials=[_silicon(lifetime)],
        physics=[PhysicsSpec(_PHYSICS, DRIFT_DIFFUSION_TYPE_NAME, settings={**_SETTINGS, **(extra_settings or {})},
                             features=list(features))],
        mesh=MeshSpec(mode='uniform', dx=cell),
        study=StudySpec(type='stationary'),
        derived_values=list(derived_values))


def _neutral_densities(net_doping):
    """(n, p) of charge-neutral equilibrium (minority as n_i^2 / majority)."""
    half = 0.5 * net_doping
    root = math.sqrt(half * half + _N_INTRINSIC ** 2)
    majority = abs(half) + root
    minority = _N_INTRINSIC ** 2 / majority
    return (majority, minority) if net_doping >= 0.0 else (minority, majority)


def _window(results, field_name):
    """(float64 window values, float64 x cell centres) of an output field."""
    field_data = results.field(f'{_PHYSICS}.{field_name}')
    return np.asarray(field_data.values, dtype=np.float64), np.asarray(field_data.x_centers, dtype=np.float64)


def _table(results, feature):
    return results.result(_PHYSICS, feature)


# ---- n-type bar material (the triangle ohmic bar / axial resistor of selftests_fem) ----
# N_D = 1e22 1/m^3: uniform n0, p0 with psi linear between the held contacts solve the discrete equations exactly
# (zero Poisson charge, the Scharfetter-Gummel flux of a uniform density is pure drift, n0 p0 = n_i^2 so R = 0).
_BAR_DONORS = 1e22
_BAR_VOLTAGE = 0.3
_BAR_N0, _BAR_P0 = _neutral_densities(_BAR_DONORS)
# Axisymmetric resistors take the whole 0.3 V in one bias step (max_bias_step 0.3).
_RING_VOLTAGE = 0.3
_RING_CONDUCTIVITY = ELEMENTARY_CHARGE * (_MOBILITY_N * _BAR_N0 + _MOBILITY_P * _BAR_P0)   # sigma' [S/m]
_RING_SETTINGS = {'geometry': 'axisymmetric', 'max_bias_step': _RING_VOLTAGE}


# ---- abrupt pn junction in equilibrium (semiconductor_cpu_gpu of selftests_backend) ----
# Abrupt junction at x = 0.5 um on a 1 um x 2 nm strip of 1 nm cells (1000 x 2): N_A = 1e23 1/m^3 for x < 0.5 um,
# N_D = 2e23 beyond (Debye lengths 13 / 9 nm, depletion ~0.13 um), ohmic contacts on the outer 5 nm, both at 0 V.
# Junction-field error 6e-6 at 0.5 nm cells, second order in dx: ~2.4e-5 expected at 1 nm.
_JUNCTION_SIZE = (1e-6, 2e-9)
_JUNCTION_CELL = 1e-9
_JUNCTION_X = 0.5e-6
_JUNCTION_CONTACT = 5e-9
_ACCEPTORS, _DONORS = 1e23, 2e23
_BULK_X = (0.25e-6, 0.75e-6)               # neutral bulk sample points, >= 0.12 um (> 9 Debye lengths) from depletion
_JUNCTION_TOLERANCE = 2e-3


def _junction_features():
    return [FeatureSpec('p_doping', 'doping', ['p_side'], {'acceptors': _ACCEPTORS}),
            FeatureSpec('n_doping', 'doping', ['n_side'], {'donors': _DONORS}),
            FeatureSpec('anode', 'ohmic_contact', ['anode'], {'voltage': 0.0}),
            FeatureSpec('cathode', 'ohmic_contact', ['cathode'], {'voltage': 0.0})]


def _junction_geometry():
    size_x, size_y = _JUNCTION_SIZE
    return [_rectangle('p_side', 0.0, _JUNCTION_X, size_y), _rectangle('n_side', _JUNCTION_X, size_x, size_y),
            _rectangle('anode', 0.0, _JUNCTION_CONTACT, size_y),
            _rectangle('cathode', size_x - _JUNCTION_CONTACT, size_x, size_y)]


def _pn_junction_specs():
    return {_RUN: _strip_model('Self-test: abrupt pn junction in equilibrium (drift-diffusion)', _JUNCTION_SIZE,
                               _JUNCTION_CELL, _junction_geometry(), _junction_features())}


def _carrier_densities(psi):
    """Boltzmann (n, p) at potential psi [V] with the Fermi level at 0."""
    return _N_INTRINSIC * math.exp(psi / _VT), _N_INTRINSIC * math.exp(-psi / _VT)


def _junction_reference():
    """(psi_p, psi_n, psi_0, |E_0|) of the abrupt junction from the exact first integral of the 1D Poisson-Boltzmann
    equation: (eps / 2) E^2 = -integral of rho dpsi from each neutral bulk, equal at the junction (linear in psi_0)."""
    psi_p = -_VT * math.asinh(_ACCEPTORS / (2.0 * _N_INTRINSIC))
    psi_n = _VT * math.asinh(_DONORS / (2.0 * _N_INTRINSIC))
    n_p, p_p = _carrier_densities(psi_p)
    n_n, p_n = _carrier_densities(psi_n)
    psi_0 = (_ACCEPTORS * psi_p + _DONORS * psi_n + _VT * (p_p - p_n + n_p - n_n)) / (_ACCEPTORS + _DONORS)
    n_0, p_0 = _carrier_densities(psi_0)
    energy = _VT * (p_0 - p_p) + _VT * (n_0 - n_p) + _ACCEPTORS * (psi_0 - psi_p)   # [V / m^3]
    field = math.sqrt(2.0 * ELEMENTARY_CHARGE * energy / (EPS0 * _EPS_DC))
    return psi_p, psi_n, psi_0, field


def _check_pn_junction(results_by_name):
    """Built-in potential between the neutral bulks, junction potential and face field, flat quasi-Fermi levels."""
    results = results_by_name[_RUN]
    potential, x_centres = _window(results, 'V')
    phi_n, _ = _window(results, 'phi_n')
    phi_p, _ = _window(results, 'phi_p')
    psi_p, psi_n, psi_0, field_0 = _junction_reference()
    built_in = psi_n - psi_p
    profile = potential.mean(axis=1)
    bulk = [int(np.argmin(np.abs(x_centres - x))) for x in _BULK_X]
    upper = int(np.searchsorted(x_centres, _JUNCTION_X))          # first cell above the junction face
    face_field = -(profile[upper] - profile[upper - 1]) / (x_centres[upper] - x_centres[upper - 1])
    face_potential = 0.5 * (profile[upper] + profile[upper - 1])
    anode, cathode = _table(results, 'anode'), _table(results, 'cathode')
    errors = {'built_in': abs((profile[bulk[1]] - profile[bulk[0]]) - built_in) / built_in,
              'junction_potential': abs(face_potential - psi_0) / built_in,
              'junction_field': abs(abs(face_field) - field_0) / field_0,
              'quasi_fermi': float(max(np.max(np.abs(phi_n)), np.max(np.abs(phi_p)))) / built_in}
    return {'measured': abs(face_field), 'expected': field_0, 'error': worst_error(errors),
            'tolerance': _JUNCTION_TOLERANCE,
            'details': {'errors': errors, 'V_bi': built_in, 'psi_p': psi_p, 'psi_n': psi_n, 'psi_0': psi_0,
                        'V_bulk': [float(profile[index]) for index in bulk], 'V_junction_face': face_potential,
                        'E_face_V_per_m': face_field, 'E0_exact_V_per_m': field_0,
                        'I_anode_A_per_m': anode['current'], 'I_cathode_A_per_m': cathode['current'],
                        'n_cells': list(potential.shape)}}


# ---- template ----
# silicon_pn_photodiode: lateral p+ / i / n+ silicon diode, 4 um x 0.2 um of 10 nm cells (400 x 20), p+ 1e24 on x < 1
# um, n+ 1e24 on x > 3 um, background donors 1e21 (depletes ~1.6 um at 1 V reverse), contacts on the outer 0.1 um,
# uniform generation G_light in the intrinsic absorber 1..3 um: photocurrent ~ q G A = 6.4e-5 A/m.
PN_DIODE_LABEL = 'Silicon pin photodiode'
PN_DIODE_DESCRIPTION = (
    'drift_diffusion, stationary: a lateral p+ / i / n+ silicon diode (4 x 0.2 um, 10 nm cells) reverse-biased by '
    'V_reverse = 1 V, with a uniform generation G_light = 1e27 1/(m^3 s) in the 2 um intrinsic absorber (photocurrent '
    '~ q G A = 6.4e-5 A/m, read from the contact tables); derived R_total.')
PN_DIODE_SUMMARY = ('A reverse-biased silicon photodiode in steady state: light creates electrons and holes in the '
                    'undoped middle, and the junction field sweeps them to the contacts.')


def silicon_pn_photodiode_spec():
    """Template 'silicon_pn_photodiode' (drift_diffusion, stationary)."""
    height = 0.2e-6
    return ModelSpec(
        title=PN_DIODE_LABEL,
        parameters=[ParameterSpec('V_reverse', 1.0, 'reverse bias of the cathode [V]'),
                    ParameterSpec('G_light', 1e27, 'pair generation rate in the absorber [1/(m^3 s)]')],
        domain=DomainSpec(size_x=4e-6, size_y=height),
        geometry=[_rectangle('p_region', 0.0, 1e-6, height), _rectangle('n_region', 3e-6, 4e-6, height),
                  _rectangle('absorber', 1e-6, 3e-6, height), _rectangle('anode', 0.0, 0.1e-6, height),
                  _rectangle('cathode', 3.9e-6, 4e-6, height)],
        materials=[MaterialSpec('silicon', library='silicon', selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS, DRIFT_DIFFUSION_TYPE_NAME, features=[
            FeatureSpec('background', 'doping', ['all'], {'donors': 1e21}),
            FeatureSpec('p_plus', 'doping', ['p_region'], {'acceptors': 1e24}),
            FeatureSpec('n_plus', 'doping', ['n_region'], {'donors': 1e24}),
            FeatureSpec('anode', 'ohmic_contact', ['anode'], {'voltage': 0.0}),
            FeatureSpec('cathode', 'ohmic_contact', ['cathode'], {'voltage': 'V_reverse'}),
            FeatureSpec('light', 'generation', ['absorber'], {'rate': 'G_light'})])],
        mesh=MeshSpec(mode='uniform', dx=1e-8),
        study=StudySpec(type='stationary'),
        derived_values=[DerivedValueSpec('R_total', 'recombination', [], 'integral'),
                        DerivedValueSpec('N_absorber', 'N', ['absorber'], 'mean')])
