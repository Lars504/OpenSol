"""drift_diffusion on triangle meshes (FEMDriftDiffusionPhysics, box method on the P1 nodes), cpu backend: the exact
ohmic bar and axial ring resistor, the carrier balance of an illuminated junction and the minority diffusion length;
read through contact tables and derived values. Registered by definition.register."""
import math

from ...core.constants import ELEMENTARY_CHARGE
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import worst_error
from ...model.spec import (DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MeshSpec, ModelSpec, PhysicsSpec,
                           SolverDefaultsSpec, StudySpec)
from .selftests import (_ACCEPTORS, _BAR_DONORS, _BAR_N0, _BAR_P0, _BAR_VOLTAGE, _DONORS, _MOBILITY_N, _MOBILITY_P,
                        _RING_CONDUCTIVITY, _RING_SETTINGS, _RING_VOLTAGE, _SETTINGS, _VT, _PHYSICS,
                        DRIFT_DIFFUSION_TYPE_NAME, _rectangle, _silicon)

UM = 1e-6


def _triangle_model(title, size, element, geometry, features, lifetime=1e-6, derived_values=(), extra_settings=None):
    """Stationary drift_diffusion-only model on a uniform triangle mesh (cpu backend, no PML ring)."""
    return ModelSpec(
        title=title,
        domain=DomainSpec(size_x=size[0], size_y=size[1]),
        geometry=list(geometry),
        materials=[_silicon(lifetime)],
        physics=[PhysicsSpec(_PHYSICS, DRIFT_DIFFUSION_TYPE_NAME, settings={**_SETTINGS, **(extra_settings or {})},
                             features=list(features))],
        meshes=[MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=element)],
        solver_defaults=SolverDefaultsSpec(backend='cpu'),
        study=StudySpec(type='stationary'),
        derived_values=list(derived_values))


def _table(results, feature):
    return results.result(_PHYSICS, feature)


# ---- semiconductor_fem_ohmic_bar ----
# The n-type bar material of selftests.py (N_D 1e22) on triangles (2 x 0.1 um, 20 nm elements), contacts on x < 0.1 um (0.3 V) and
# x > 1.9 um (0 V). The held nodes include the contact outlines x = 0.1 / 1.9 um (PSLG segments), so uniform n0, p0 with
# psi linear between them solve the box equations exactly (linear P1 fields: zero stiffness residual at free nodes and
# zero normal flux on the insulating faces): I = q (mu_n n0 + mu_p p0) H V / (1.8 um).
_FEM_BAR_SIZE = (2e-6, 1e-7)
_FEM_BAR_ELEMENT = 2e-8
_FEM_BAR_CONTACT = 1e-7
_FEM_BAR_LENGTH = _FEM_BAR_SIZE[0] - 2.0 * _FEM_BAR_CONTACT
_FEM_BAR_CURRENT = (ELEMENTARY_CHARGE * (_MOBILITY_N * _BAR_N0 + _MOBILITY_P * _BAR_P0) * _FEM_BAR_SIZE[1]
                    * _BAR_VOLTAGE / _FEM_BAR_LENGTH)
_EXACT_TOLERANCE = 1e-6   # Newton 1e-8; float64 tables (first estimate)


def _fem_ohmic_bar_specs():
    size_x, size_y = _FEM_BAR_SIZE
    return {'run': _triangle_model(
        'Self-test: ohmic n-type bar on triangles (drift-diffusion)', _FEM_BAR_SIZE, _FEM_BAR_ELEMENT,
        [_rectangle('left', 0.0, _FEM_BAR_CONTACT, size_y), _rectangle('right', size_x - _FEM_BAR_CONTACT, size_x, size_y)],
        [FeatureSpec('n_doping', 'doping', ['all'], {'donors': _BAR_DONORS}),
         FeatureSpec('left', 'ohmic_contact', ['left'], {'voltage': _BAR_VOLTAGE}),
         FeatureSpec('right', 'ohmic_contact', ['right'], {'voltage': 0.0})])}


def _check_fem_ohmic_bar(results_by_name):
    """Both contact currents against the exact box solution."""
    results = results_by_name['run']
    left, right = _table(results, 'left'), _table(results, 'right')
    errors = {'left_current': abs(left['current'] - _FEM_BAR_CURRENT) / _FEM_BAR_CURRENT,
              'right_current': abs(right['current'] + _FEM_BAR_CURRENT) / _FEM_BAR_CURRENT}
    return {'measured': left['current'], 'expected': _FEM_BAR_CURRENT, 'error': worst_error(errors),
            'tolerance': _EXACT_TOLERANCE,
            'details': {'errors': errors, 'unit_current': 'A/m', 'I_left': left['current'], 'I_right': right['current'],
                        'I_left_hole': left['hole_current'], 'L_held_m': _FEM_BAR_LENGTH}}


# ---- semiconductor_fem_axial_resistor ----
# geometry 'axisymmetric': a solid n-type cylinder r < 0.2 um, z 0..1 um on 20 nm triangles, contacts z < 0.1 um (0.3 V)
# and z > 0.9 um (0 V). psi linear in z is exact for the r-weighted P1 stiffness (centroid r integrates r exactly on
# each element), so I = sigma' pi R^2 V / (0.8 um) in whole-ring amperes.
_AXIAL_RADIUS = 2e-7
_AXIAL_HEIGHT = 1e-6
_AXIAL_CONTACT = 1e-7
_FEM_AXIAL_LENGTH = _AXIAL_HEIGHT - 2.0 * _AXIAL_CONTACT
_FEM_AXIAL_CURRENT = _RING_CONDUCTIVITY * math.pi * _AXIAL_RADIUS ** 2 * _RING_VOLTAGE / _FEM_AXIAL_LENGTH


def _fem_axial_specs():
    return {'run': _triangle_model(
        'Self-test: axisymmetric axial n-type resistor on triangles (drift-diffusion)', (_AXIAL_RADIUS, _AXIAL_HEIGHT),
        _FEM_BAR_ELEMENT,
        [GeometrySpec('bottom', 'rectangle', {'x0': 0.0, 'y0': 0.0, 'x1': _AXIAL_RADIUS, 'y1': _AXIAL_CONTACT}),
         GeometrySpec('top', 'rectangle', {'x0': 0.0, 'y0': _AXIAL_HEIGHT - _AXIAL_CONTACT, 'x1': _AXIAL_RADIUS,
                                           'y1': _AXIAL_HEIGHT})],
        [FeatureSpec('n_doping', 'doping', ['all'], {'donors': _BAR_DONORS}),
         FeatureSpec('bottom', 'ohmic_contact', ['bottom'], {'voltage': _RING_VOLTAGE}),
         FeatureSpec('top', 'ohmic_contact', ['top'], {'voltage': 0.0})],
        extra_settings=_RING_SETTINGS)}


def _check_fem_axial(results_by_name):
    """Whole-ring contact currents (unit 'A') against sigma' pi R^2 V / L."""
    results = results_by_name['run']
    bottom, top = _table(results, 'bottom'), _table(results, 'top')
    units = [table['units'][2] for table in (bottom, top)]
    errors = {'bottom_current': abs(bottom['current'] - _FEM_AXIAL_CURRENT) / _FEM_AXIAL_CURRENT,
              'top_current': abs(top['current'] + _FEM_AXIAL_CURRENT) / _FEM_AXIAL_CURRENT,
              'current_unit': 0.0 if units == ['A', 'A'] else math.inf}
    return {'measured': bottom['current'], 'expected': _FEM_AXIAL_CURRENT, 'error': worst_error(errors),
            'tolerance': _EXACT_TOLERANCE,
            'details': {'errors': errors, 'units': units, 'I_bottom_A': bottom['current'], 'I_top_A': top['current']}}


# ---- semiconductor_fem_photocurrent_balance ----
# The pn junction doping of selftests.py (N_A 1e23 | N_D 2e23 at x = 0.5 um) on a 1 um x 40 nm strip of 5 nm triangles,
# contacts on the outer 10 nm at 0 V, lifetime 1e-9 s, uniform G = 1e27. Summing every continuity row, the edge terms
# cancel, so the contact reactions add up to q (G A_free - sum of node volume R), and the element recombination field is
# the lumped mean of the node rates, so its integral is that sum.
_PHOTO_SIZE = (1e-6, 4e-8)
_PHOTO_ELEMENT = 5e-9
_PHOTO_CONTACT = 1e-8
_PHOTO_JUNCTION = 0.5e-6
_PHOTO_GENERATION = 1e27
_PHOTO_LIFETIME = 1e-9
_PHOTO_TOLERANCE = 1e-5   # Newton residual, float32 storage of R (first estimate)


def _fem_photocurrent_specs():
    size_x, size_y = _PHOTO_SIZE
    return {'run': _triangle_model(
        'Self-test: illuminated pn junction on triangles, carrier balance (drift-diffusion)', _PHOTO_SIZE,
        _PHOTO_ELEMENT,
        [_rectangle('p_side', 0.0, _PHOTO_JUNCTION, size_y), _rectangle('n_side', _PHOTO_JUNCTION, size_x, size_y),
         _rectangle('anode', 0.0, _PHOTO_CONTACT, size_y), _rectangle('cathode', size_x - _PHOTO_CONTACT, size_x, size_y)],
        [FeatureSpec('p_doping', 'doping', ['p_side'], {'acceptors': _ACCEPTORS}),
         FeatureSpec('n_doping', 'doping', ['n_side'], {'donors': _DONORS}),
         FeatureSpec('anode', 'ohmic_contact', ['anode'], {'voltage': 0.0}),
         FeatureSpec('cathode', 'ohmic_contact', ['cathode'], {'voltage': 0.0}),
         FeatureSpec('light', 'generation', ['all'], {'rate': _PHOTO_GENERATION})],
        lifetime=_PHOTO_LIFETIME,
        derived_values=[DerivedValueSpec('R_total', 'recombination', [], 'integral')])}


def _check_fem_photocurrent(results_by_name):
    """Electron / hole contact currents vs q (G A_free - integral of R) and Kirchhoff's current law."""
    results = results_by_name['run']
    anode, cathode = _table(results, 'anode'), _table(results, 'cathode')
    recombination_total = float(results.summary()['derived']['R_total'])
    free_area = (_PHOTO_SIZE[0] - 2.0 * _PHOTO_CONTACT) * _PHOTO_SIZE[1]
    expected = ELEMENTARY_CHARGE * (_PHOTO_GENERATION * free_area - recombination_total)
    electron_total = anode['electron_current'] + cathode['electron_current']
    hole_total = anode['hole_current'] + cathode['hole_current']
    errors = {'electron_balance': abs(electron_total - expected) / abs(expected),
              'hole_balance': abs(hole_total + expected) / abs(expected),
              'kirchhoff': abs(anode['current'] + cathode['current']) / abs(expected)}
    return {'measured': electron_total, 'expected': expected, 'error': worst_error(errors), 'tolerance': _PHOTO_TOLERANCE,
            'details': {'errors': errors, 'unit_current': 'A/m', 'I_anode': anode['current'],
                        'I_cathode': cathode['current'],
                        'generated_A_per_m': ELEMENTARY_CHARGE * _PHOTO_GENERATION * free_area,
                        'recombined_A_per_m': ELEMENTARY_CHARGE * recombination_total}}


# ---- semiconductor_fem_minority_diffusion ----
# p-type bar 4 um x 40 nm on 20 nm triangles, N_A = 1e22, lifetime 1e-10 s, G = 1e27, contacts on the outer 50 nm at
# 0 V. Low injection: dn = G tau (1 - cosh((x - x_m) / L) / cosh(L_half / L)) between the contact outlines (dn = 0 on
# them), L = sqrt(mu_n V_T tau); the contact current (the reaction: the flux through the outline) is q D_n H G tau / L
# tanh(L_half / L).
_DIFFUSION_SIZE = (4e-6, 4e-8)
_DIFFUSION_ELEMENT = 2e-8
_DIFFUSION_CONTACT = 5e-8
_DIFFUSION_ACCEPTORS = 1e22
_DIFFUSION_LIFETIME = 1e-10
_DIFFUSION_GENERATION = 1e27
_DIFFUSION_TOLERANCE = 2e-3   # (h / L)^2 ~ 1e-3 times an O(1 / 12) constant, O(G tau / N_A) = 1e-5 (first estimate)


def _fem_diffusion_current():
    """(electron current into each contact [A/m], diffusion length [m]) of the closed form."""
    diffusivity = _MOBILITY_N * _VT
    length = math.sqrt(diffusivity * _DIFFUSION_LIFETIME)
    half = 0.5 * (_DIFFUSION_SIZE[0] - 2.0 * _DIFFUSION_CONTACT)
    current = (ELEMENTARY_CHARGE * diffusivity * _DIFFUSION_SIZE[1] * _DIFFUSION_GENERATION * _DIFFUSION_LIFETIME
               / length * math.tanh(half / length))
    return current, length


def _fem_diffusion_specs():
    size_x, size_y = _DIFFUSION_SIZE
    return {'run': _triangle_model(
        'Self-test: minority-carrier diffusion length on triangles (drift-diffusion)', _DIFFUSION_SIZE,
        _DIFFUSION_ELEMENT,
        [_rectangle('left', 0.0, _DIFFUSION_CONTACT, size_y),
         _rectangle('right', size_x - _DIFFUSION_CONTACT, size_x, size_y)],
        [FeatureSpec('p_doping', 'doping', ['all'], {'acceptors': _DIFFUSION_ACCEPTORS}),
         FeatureSpec('left', 'ohmic_contact', ['left'], {'voltage': 0.0}),
         FeatureSpec('right', 'ohmic_contact', ['right'], {'voltage': 0.0}),
         FeatureSpec('light', 'generation', ['all'], {'rate': _DIFFUSION_GENERATION})],
        lifetime=_DIFFUSION_LIFETIME)}


def _check_fem_diffusion(results_by_name):
    """Electron current into each contact vs the closed form."""
    results = results_by_name['run']
    current, length = _fem_diffusion_current()
    left, right = _table(results, 'left'), _table(results, 'right')
    errors = {'left_electron_current': abs(abs(left['electron_current']) - current) / current,
              'right_electron_current': abs(abs(right['electron_current']) - current) / current}
    return {'measured': left['electron_current'], 'expected': current, 'error': worst_error(errors),
            'tolerance': _DIFFUSION_TOLERANCE,
            'details': {'errors': errors, 'unit_current': 'A/m', 'diffusion_length_m': length,
                        'I_left_electron': left['electron_current'], 'I_right_electron': right['electron_current']}}


FEM_SELFTEST_CASES = (
    SelfTestCase(
        name='semiconductor_fem_ohmic_bar', label="Drift-diffusion on triangles: ohmic n-type bar (Ohm's law)",
        description=(f'An n-type silicon-like bar (N_D = {_BAR_DONORS:g} 1/m^3) on a '
                     f'{_FEM_BAR_SIZE[0] / UM:g} x {_FEM_BAR_SIZE[1] / UM:g} um bar of {_FEM_BAR_ELEMENT * 1e9:g} nm '
                     f'triangles (cpu), contacts on the outer {_FEM_BAR_CONTACT / UM:g} um ({_BAR_VOLTAGE:g} V / 0 V). '
                     'The held nodes include the contact outlines, and uniform n0, p0 with psi linear between them '
                     'solve the box equations exactly (P1 reproduces linear fields; the Scharfetter-Gummel flux of a '
                     f'uniform density is pure drift; R = 0), so I = q (mu_n n0 + mu_p p0) H V / L, L = '
                     f'{_FEM_BAR_LENGTH / UM:g} um = {_FEM_BAR_CURRENT:.6g} A/m. error = max relative mismatch of both '
                     f'contact currents; tolerance {_EXACT_TOLERANCE:g} (first estimate). Not yet run.'),
        specs=_fem_ohmic_bar_specs, check=_check_fem_ohmic_bar),
    SelfTestCase(
        name='semiconductor_fem_axial_resistor',
        label='Drift-diffusion on triangles: axisymmetric axial n-type resistor',
        description=(f"geometry 'axisymmetric': a solid cylinder r < {_AXIAL_RADIUS / UM:g} um, z 0.."
                     f'{_AXIAL_HEIGHT / UM:g} um of {_FEM_BAR_ELEMENT * 1e9:g} nm triangles (cpu), N_D = '
                     f'{_BAR_DONORS:g}, contacts on the outer {_AXIAL_CONTACT / UM:g} um of z ({_RING_VOLTAGE:g} V / 0 '
                     "V, one bias step). psi linear in z is exact for the r-weighted P1 stiffness, so I = sigma' pi R^2 "
                     f'V / L = {_FEM_AXIAL_CURRENT:.6g} A (whole ring, L = {_FEM_AXIAL_LENGTH / UM:g} um). error = max '
                     "relative mismatch of both contacts, inf unless both units are 'A'; tolerance "
                     f'{_EXACT_TOLERANCE:g} (first estimate). Not yet run.'),
        specs=_fem_axial_specs, check=_check_fem_axial),
    SelfTestCase(
        name='semiconductor_fem_photocurrent_balance',
        label='Drift-diffusion on triangles: illuminated pn junction, carrier conservation',
        description=(f'An abrupt pn junction (N_A = {_ACCEPTORS:g} | N_D = {_DONORS:g} 1/m^3 at '
                     f'{_PHOTO_JUNCTION / UM:g} um) on a {_PHOTO_SIZE[0] / UM:g} um x {_PHOTO_SIZE[1] * 1e9:g} nm '
                     f'strip of {_PHOTO_ELEMENT * 1e9:g} nm triangles (cpu), contacts on the outer '
                     f'{_PHOTO_CONTACT * 1e9:g} nm at 0 V, lifetime {_PHOTO_LIFETIME:g} s, uniform G = '
                     f'{_PHOTO_GENERATION:g} 1/(m^3 s). The contact currents are the reactions of the held nodes and '
                     'the element recombination field is the lumped mean of the node rates, so the electron currents '
                     'add up exactly to q (G A_free - integral of R) (A_free = the strip without the contact elements), '
                     'the hole currents to minus that, and the contact currents cancel. error = max of the three '
                     f'relative mismatches; tolerance {_PHOTO_TOLERANCE:g} (float32 R storage; first estimate). Not yet '
                     'run.'),
        specs=_fem_photocurrent_specs, check=_check_fem_photocurrent),
    SelfTestCase(
        name='semiconductor_fem_minority_diffusion',
        label='Drift-diffusion on triangles: minority-carrier diffusion length',
        description=(f'A p-type bar {_DIFFUSION_SIZE[0] / UM:g} um x {_DIFFUSION_SIZE[1] * 1e9:g} nm of '
                     f'{_DIFFUSION_ELEMENT * 1e9:g} nm triangles (cpu), N_A = {_DIFFUSION_ACCEPTORS:g}, lifetime '
                     f'{_DIFFUSION_LIFETIME:g} s, G = {_DIFFUSION_GENERATION:g}, contacts on the outer '
                     f'{_DIFFUSION_CONTACT * 1e9:g} nm at 0 V. Low injection: the contact electron current (the reaction '
                     'of the held nodes = the flux through the contact outline) is q D_n H G tau / L tanh(L_half / L), '
                     f'L = sqrt(mu_n V_T tau) = {_fem_diffusion_current()[1] / UM:.3g} um, = '
                     f'{_fem_diffusion_current()[0]:.6g} A/m. error = max relative mismatch per contact; tolerance '
                     f'{_DIFFUSION_TOLERANCE:g} (second-order (h / L)^2 discretisation, O(G tau / N_A) corrections; '
                     'first estimate). Pins the edge Scharfetter-Gummel diffusion, the lumped recombination and '
                     'generation on triangles. Not yet run.'),
        specs=_fem_diffusion_specs, check=_check_fem_diffusion),
)
