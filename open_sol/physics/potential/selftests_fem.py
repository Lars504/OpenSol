"""Electrostatics / electric currents on triangle meshes (mesh plan stage T4, FEMPotentialPhysics): the coaxial
capacitor (curved held outlines, reaction charges, convergence order), the Joule-heated resistor strip with heat on
the same triangle mesh or on a tensor mesh (currents_heat_joule across meshes) and the linear electrolyte slit; model
runs read through feature results (held tables, probes), scalar outputs and derived integrals. Registered by
definition.register."""
import math

import numpy as np

from ...core.constants import ELEMENTARY_CHARGE, EPS0, K_B, N_A
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import last_value
from ...model.spec import (CouplingSpec, DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec,
                           MeshSpec, ModelSpec, PhysicsSpec, SolverDefaultsSpec, StudySpec)

ELECTROSTATICS_TYPE_NAME = 'electrostatics'   # = definition.ELECTROSTATICS_TYPE_NAME (definition imports this module)
CURRENTS_TYPE_NAME = 'electric_currents'
HEAT_TYPE_NAME = 'heat'
UM = 1e-6

# ---- electrostatics_fem_coax: inner conductor r = a at 1 V, outer r = b at 0 V, eps_dc everywhere ----
COAX_CENTRE = (2.7 * UM, 2.7 * UM)
INNER_RADIUS = 1.0 * UM            # a
OUTER_RADIUS = 2.5 * UM            # b
COAX_EPS = 4.0
COAX_VOLTAGE = 1.0
COAX_RUNS = {'coarse': INNER_RADIUS / 3.0, 'fine': INNER_RADIUS / 6.0}   # element sizes (ratio 2: order = log2)
COAX_TOLERANCE = 1e-5              # relative C' error at a / h = 6 (extrapolated ~2e-7 from 4.5e-8 measured at a / h = 10)
COAX_MIN_ORDER = 2.5               # observed 2.77 between a / h = 5 and 10 (relative errors 3.0e-7 / 4.5e-8): the two
#                                    uniform meshes are not nested and the element count ratio is 3.8, so the two-level
#                                    order estimate is noisy; P2 energy order ~4 asymptotically; a / h = 3 / 6 not run
COAX_BALANCE = 1e-8                # |Q_inner + Q_outer| / Q_inner (Gauss: exact up to the solve)

# ---- currents_heat_fem_joule: strip L x W, V on x_lo, 0 on x_hi; heat T0 on both x faces ----
STRIP_LENGTH = 2.1 * UM
STRIP_WIDTH = 0.6 * UM
STRIP_CELLS = 21                   # tensor heat cells along x (odd: the centre probe sits on a cell centre)
STRIP_ELEMENT = STRIP_LENGTH / STRIP_CELLS
SIGMA = 1e6                        # S/m
CONDUCTIVITY = 100.0               # W/(m K)
STRIP_VOLTAGE = 0.1                # V
REFERENCE_T = 293.15
# measured: current 2.7e-14, shared_rise 4.9e-7, two_mesh_rise 1.7e-4
JOULE_TOLERANCES = {'current': 1e-8, 'shared_rise': 1e-5, 'two_mesh_rise': 1e-3}

# ---- electrostatics_fem_electrolyte_slit: plates at V0 on x_lo / x_hi, water, 1:1 0.1 mM linear electrolyte ----
NM = 1e-9
SLIT_LENGTH = 100.0 * NM           # L (x)
SLIT_HEIGHT = 10.0 * NM            # H (y, insulated faces: 1D in x)
SLIT_ELEMENT = 2.5 * NM            # h ~ lambda_D / 12
SLIT_EPS = 80.0
SLIT_VOLTAGE = 0.01                # V
SLIT_TEMPERATURE = 298.15          # K
SLIT_SPECIES = ([1e-4, 1e-4], [1.0, -1.0])   # mol/L, valences
SLIT_SCREENING = 2.0 * 1e-4 * 1e3 * N_A * ELEMENTARY_CHARGE ** 2 / (K_B * SLIT_TEMPERATURE)   # k [F/m^3]
SLIT_DEBYE_LENGTH = math.sqrt(EPS0 * SLIT_EPS / SLIT_SCREENING)                                # 30.7 nm
# first estimates (not run): continuum V / plate charge ~(h / lambda)^4 ~ 4e-5 x a small constant; Gauss / the
# concentration identities exact up to float32 element storage; scalars exact
SLIT_TOLERANCES = {'continuum': 1e-4, 'identity': 1e-5, 'scalar': 1e-6}


# ---- coax ----
def coax_spec(element_size) -> ModelSpec:
    cx, cy = COAX_CENTRE
    return ModelSpec(
        title='Self-test: coaxial capacitor on triangles',
        domain=DomainSpec(size_x=2.0 * cx, size_y=2.0 * cy),
        geometry=[GeometrySpec('outer', 'circle', {'cx': cx, 'cy': cy, 'r': OUTER_RADIUS}),
                  GeometrySpec('inner', 'circle', {'cx': cx, 'cy': cy, 'r': INNER_RADIUS})],
        materials=[MaterialSpec('dielectric', properties={'eps_dc': COAX_EPS}, selection=['all'])],
        physics=[PhysicsSpec('electrostatics', ELECTROSTATICS_TYPE_NAME, features=[
            FeatureSpec('inner_electrode', 'voltage', ['inner'], {'voltage': COAX_VOLTAGE}),
            FeatureSpec('outer_electrode', 'voltage', ['outer'], {'voltage': 0.0})])],
        meshes=[MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=element_size)],
        solver_defaults=SolverDefaultsSpec(backend='cpu'),
        study=StudySpec(type='stationary'))


def _coax_runs():
    return {name: coax_spec(size) for name, size in COAX_RUNS.items()}


def _held(results, physics, feature, quantity):
    return float(results.result(physics, feature)[quantity])


def _check_coax(results_by_name):
    """C' = Q_inner / V against 2 pi eps0 eps / ln(b / a) at both sizes, the observed order and the charge balance;
    error = the fine relative error over COAX_TOLERANCE, + 1 per failed order / balance check; tolerance 1."""
    expected = 2.0 * math.pi * EPS0 * COAX_EPS / math.log(OUTER_RADIUS / INNER_RADIUS)
    errors, balance = {}, {}
    for name in COAX_RUNS:
        inner = _held(results_by_name[name], 'electrostatics', 'inner_electrode', 'charge')
        outer = _held(results_by_name[name], 'electrostatics', 'outer_electrode', 'charge')
        errors[name] = abs(inner / COAX_VOLTAGE / expected - 1.0)
        balance[name] = abs(inner + outer) / abs(inner)
    order = math.log2(errors['coarse'] / errors['fine']) if errors['fine'] > 0.0 else math.inf
    error = errors['fine'] / COAX_TOLERANCE + float(not order >= COAX_MIN_ORDER) \
        + float(not max(balance.values()) <= COAX_BALANCE)
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': 1.0,
            'details': {'relative_errors': errors, 'observed_order': order, 'charge_balance': balance,
                        'expected_capacitance_F_per_m': expected}}


# ---- Joule strip ----
def joule_strip_spec(two_mesh=False) -> ModelSpec:
    """Currents on a triangle mesh; heat on it too, or (two_mesh) on a uniform tensor mesh of STRIP_CELLS cells."""
    length, width = STRIP_LENGTH, STRIP_WIDTH
    meshes = [MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=STRIP_ELEMENT)]
    if two_mesh:
        meshes.append(MeshSpec(name='cells', mode='uniform', dx=STRIP_ELEMENT))
    return ModelSpec(
        title='Self-test: Joule-heated resistor strip on triangles',
        domain=DomainSpec(size_x=length, size_y=width),
        geometry=[GeometrySpec('probe_point', 'point', {'x': 0.5 * length, 'y': 0.5 * width})],
        materials=[MaterialSpec('resistor', properties={'sigma_dc': SIGMA, 'k': CONDUCTIVITY}, selection=['all'])],
        physics=[PhysicsSpec('currents', CURRENTS_TYPE_NAME, features=[
                     FeatureSpec('left', 'voltage', ['x_lo'], {'voltage': STRIP_VOLTAGE}),
                     FeatureSpec('right', 'voltage', ['x_hi'], {'voltage': 0.0})], mesh='triangles'),
                 PhysicsSpec('heat', HEAT_TYPE_NAME, settings={'T0': REFERENCE_T}, features=[
                     FeatureSpec('walls', 'temperature', ['x_lo', 'x_hi'], {'temperature': REFERENCE_T}),
                     FeatureSpec('centre', 'temperature_probe', ['probe_point'])],
                     mesh='cells' if two_mesh else 'triangles')],
        couplings=[CouplingSpec('joule', 'currents_heat_joule', 'currents', 'heat')],
        meshes=meshes,
        solver_defaults=SolverDefaultsSpec(backend='cpu'),
        study=StudySpec(type='stationary'))


def _joule_runs():
    return {'shared': joule_strip_spec(), 'two_mesh': joule_strip_spec(two_mesh=True)}


def _finite_volume_centre_rise(power_density):
    """Centre-cell rise [K] of the tensor heat: the 1D FV chain (k / dx interior, 2 k / dx at the held faces)."""
    dx = STRIP_ELEMENT
    operator = np.zeros((STRIP_CELLS, STRIP_CELLS))
    for cell in range(STRIP_CELLS - 1):
        operator[cell:cell + 2, cell:cell + 2] += CONDUCTIVITY / dx * np.array([[1.0, -1.0], [-1.0, 1.0]])
    operator[0, 0] += 2.0 * CONDUCTIVITY / dx
    operator[-1, -1] += 2.0 * CONDUCTIVITY / dx
    rises = np.linalg.solve(operator, np.full(STRIP_CELLS, power_density * dx))
    return float(rises[STRIP_CELLS // 2])


def _check_joule(results_by_name):
    """Terminal current sigma W V / L (both runs), the centre rise q L^2 / (8 k) on the shared triangle mesh (P2 exact
    for the parabola) and the FV chain value with heat on the tensor mesh; error = largest relative error over its
    tolerance; tolerance 1."""
    power_density = SIGMA * (STRIP_VOLTAGE / STRIP_LENGTH) ** 2
    expected = {'current': SIGMA * STRIP_WIDTH * STRIP_VOLTAGE / STRIP_LENGTH,
                'shared_rise': power_density * STRIP_LENGTH ** 2 / (8.0 * CONDUCTIVITY),
                'two_mesh_rise': _finite_volume_centre_rise(power_density)}
    measured = {'current': _held(results_by_name['shared'], 'currents', 'left', 'current'),
                'current_two_mesh': _held(results_by_name['two_mesh'], 'currents', 'left', 'current')}
    for name in ('shared', 'two_mesh'):
        measured[f'{name}_rise'] = last_value(results_by_name[name], 'heat', 'centre') - REFERENCE_T
    relative = {'current': abs(measured['current'] / expected['current'] - 1.0),
                'current_two_mesh': abs(measured['current_two_mesh'] / expected['current'] - 1.0),
                'shared_rise': abs(measured['shared_rise'] / expected['shared_rise'] - 1.0),
                'two_mesh_rise': abs(measured['two_mesh_rise'] / expected['two_mesh_rise'] - 1.0)}
    tolerances = dict(JOULE_TOLERANCES, current_two_mesh=JOULE_TOLERANCES['current'])
    scaled = [relative[name] / tolerances[name] for name in relative]
    error = max(scaled) if all(np.isfinite(scaled)) else math.nan
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': 1.0,
            'details': {'relative_errors': relative, 'tolerances': tolerances, 'measured': measured,
                        'expected': expected}}


# ---- electrolyte slit ----
def electrolyte_slit_spec() -> ModelSpec:
    """Plates plate_lo / plate_hi at V0 on x_lo / x_hi, a linear 1:1 electrolyte 'ions' on all, P2 triangles."""
    concentrations, valences = SLIT_SPECIES
    return ModelSpec(
        title='Self-test: linear electrolyte slit on triangles',
        domain=DomainSpec(size_x=SLIT_LENGTH, size_y=SLIT_HEIGHT),
        geometry=[GeometrySpec('midpoint', 'point', {'x': 0.5 * SLIT_LENGTH, 'y': 0.5 * SLIT_HEIGHT})],
        materials=[MaterialSpec('water', properties={'eps_dc': SLIT_EPS}, selection=['all'])],
        physics=[PhysicsSpec('electrostatics', ELECTROSTATICS_TYPE_NAME, features=[
            FeatureSpec('plate_lo', 'voltage', ['x_lo'], {'voltage': SLIT_VOLTAGE}),
            FeatureSpec('plate_hi', 'voltage', ['x_hi'], {'voltage': SLIT_VOLTAGE}),
            FeatureSpec('ions', 'electrolyte', ['all'], {'concentrations': list(concentrations),
                                                         'valences': list(valences),
                                                         'temperature': SLIT_TEMPERATURE, 'model': 'linear'}),
            FeatureSpec('mid', 'potential_probe', ['midpoint'])])],
        meshes=[MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=SLIT_ELEMENT)],
        solver_defaults=SolverDefaultsSpec(backend='cpu'),
        study=StudySpec(type='stationary'),
        derived_values=[DerivedValueSpec('Q_ion', 'rho_ion', [], 'integral'),
                        DerivedValueSpec('debye_length', 'electrostatics_debye_length_ions', [], 'mean'),
                        DerivedValueSpec('ionic_strength', 'electrostatics_ionic_strength_ions', [], 'mean')])


def _electrolyte_runs():
    return {'run': electrolyte_slit_spec()}


def _check_electrolyte_slit(results_by_name):
    """Plate charges and the midpoint V against the continuum slit, Gauss (Q_lo + Q_hi + int rho_ion = 0), the
    concentration identities on the field rasters (c_ion_1 + c_ion_2 = 2 c, e 1000 N_A (c_ion_1 - c_ion_2) = rho_ion:
    P0 fields, the same element per sample) and the scalars; error = largest relative error over its tolerance class;
    tolerance 1."""
    results = results_by_name['run']
    derived = results.summary()['derived']
    half_width = 0.5 * SLIT_LENGTH / SLIT_DEBYE_LENGTH
    plate_continuum = EPS0 * SLIT_EPS * SLIT_VOLTAGE * math.tanh(half_width) / SLIT_DEBYE_LENGTH * SLIT_HEIGHT
    midpoint_continuum = SLIT_VOLTAGE / math.cosh(half_width)
    charge_lo = _held(results, 'electrostatics', 'plate_lo', 'charge')
    charge_hi = _held(results, 'electrostatics', 'plate_hi', 'charge')
    midpoint = last_value(results, 'electrostatics', 'mid')
    ionic_charge = float(derived['Q_ion'])
    rasters = {name: np.asarray(results.field(f'electrostatics.{name}').values, dtype=np.float64)
               for name in ('c_ion_1', 'c_ion_2', 'rho_ion')}
    inside = np.isfinite(rasters['c_ion_1']) & np.isfinite(rasters['c_ion_2']) & np.isfinite(rasters['rho_ion'])
    bulk = SLIT_SPECIES[0][0]
    concentration_sum = (float(np.max(np.abs(rasters['c_ion_1'] + rasters['c_ion_2'] - 2.0 * bulk)[inside]))
                         / (2.0 * bulk) if inside.any() else math.nan)
    concentration_charge = ELEMENTARY_CHARGE * 1e3 * N_A * (rasters['c_ion_1'] - rasters['c_ion_2'])
    rho_scale = float(np.max(np.abs(rasters['rho_ion'])[inside])) if inside.any() else math.nan
    concentration_rho = (float(np.max(np.abs(concentration_charge - rasters['rho_ion'])[inside])) / rho_scale
                         if inside.any() else math.nan)
    relative = {
        'plate_lo': (abs(charge_lo / plate_continuum - 1.0), 'continuum'),
        'plate_hi': (abs(charge_hi / plate_continuum - 1.0), 'continuum'),
        'midpoint': (abs(midpoint / midpoint_continuum - 1.0), 'continuum'),
        'gauss': (abs(charge_lo + charge_hi + ionic_charge) / abs(charge_lo), 'identity'),
        'concentration_sum': (concentration_sum, 'identity'),
        'concentration_rho': (concentration_rho, 'identity'),
        'debye_length': (abs(float(derived['debye_length']) / SLIT_DEBYE_LENGTH - 1.0), 'scalar'),
        'ionic_strength': (abs(float(derived['ionic_strength']) / 1e-4 - 1.0), 'scalar')}
    scaled = [value / SLIT_TOLERANCES[kind] for value, kind in relative.values()]
    error = max(scaled) if all(np.isfinite(scaled)) else math.nan
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': 1.0,
            'details': {'relative_errors': {name: value for name, (value, _) in relative.items()},
                        'tolerances': SLIT_TOLERANCES, 'Q_lo': charge_lo, 'Q_hi': charge_hi, 'Q_ion': ionic_charge,
                        'Q_continuum': plate_continuum, 'V_mid': midpoint, 'V_mid_continuum': midpoint_continuum}}


FEM_SELFTEST_CASES = (
    SelfTestCase(
        name='electrostatics_fem_coax', label='Electrostatics on triangles: coaxial capacitor',
        description=(f'Electrostatics (P2, eps_dc = {COAX_EPS:g} everywhere) between a voltage of {COAX_VOLTAGE:g} V '
                     f'on the outline of the inner circle r = {INNER_RADIUS / UM:g} um and 0 V on the outer circle r = '
                     f'{OUTER_RADIUS / UM:g} um (curved outline edges, isoparametric P2; the regions inside / outside '
                     'solve to their held values), uniform triangles at a / h = 3 and 6. Expected C\' = 2 pi eps0 eps / '
                     "ln(b / a) from the inner electrode's charge (the reaction of its held dofs). error = the a / h = "
                     f'6 relative error over {COAX_TOLERANCE:g}, + 1 when the observed order log2(e_3 / e_6) is below '
                     f'{COAX_MIN_ORDER:g}, + 1 when |Q_inner + Q_outer| / Q_inner exceeds {COAX_BALANCE:g}; tolerance 1. '
                     'Measured at a / h = 5 / 10: relative errors 3.0e-7 / 4.5e-8, observed order 2.77 (the two '
                     'uniform meshes are not nested: a noisy two-level estimate of the P2 energy order ~4); '
                     'extrapolated to a / h = 3 / 6: ~1.3e-6 / ~2e-7. Not yet run at a / h = 3 and 6.'),
        specs=_coax_runs, check=_check_coax),
    SelfTestCase(
        name='currents_heat_fem_joule', label='Electric currents + heat on triangles: Joule-heated strip',
        description=(f'A resistor strip {STRIP_LENGTH / UM:g} x {STRIP_WIDTH / UM:g} um (sigma_dc = {SIGMA:g} S/m, k = '
                     f'{CONDUCTIVITY:g} W/(m K)), {STRIP_VOLTAGE:g} V on x_lo and 0 V on x_hi, heat held at T0 on both '
                     'x faces, currents_heat_joule; currents on a triangle mesh (P2). Run shared: heat on the same '
                     'mesh (q_joule per element, exactly uniform): the current sigma W V / L (reaction, 1e-8) and the '
                     'centre rise q L^2 / (8 k) (P2 exact for the parabola; float32 probe, 1e-5). Run two_mesh: heat on '
                     f'a uniform tensor mesh of {STRIP_CELLS} cells (q_joule moved conservatively triangles -> cells): '
                     'the centre rise of the 1D finite-volume chain (1e-3: the binned quadrature of the transfer). '
                     'error = largest relative error over its tolerance; tolerance 1. Measured: current 2.7e-14, '
                     'shared rise 4.9e-7, two-mesh rise 1.7e-4.'),
        specs=_joule_runs, check=_check_joule),
    SelfTestCase(
        name='electrostatics_fem_electrolyte_slit', label='Electrostatics on triangles: linear electrolyte slit',
        description=(f'Plates at {SLIT_VOLTAGE:g} V on x_lo and x_hi of a {SLIT_LENGTH / NM:g} x {SLIT_HEIGHT / NM:g} nm '
                     f'water slit (eps_dc {SLIT_EPS:g}, insulated y faces) filled with a linear 1:1 0.1 mM electrolyte '
                     f'(lambda_D = {SLIT_DEBYE_LENGTH / NM:.3g} nm), P2 on uniform {SLIT_ELEMENT / NM:g} nm triangles. '
                     'Checks: each plate charge against eps0 eps V0 tanh(L / (2 lambda_D)) / lambda_D H and the midpoint '
                     f"probe against V0 / cosh(L / (2 lambda_D)) ({SLIT_TOLERANCES['continuum']:g}); Gauss Q_lo + Q_hi "
                     '+ int rho_ion = 0 and, on the field rasters, c_ion_1 + c_ion_2 = 2 c and e 1000 N_A (c_ion_1 - '
                     'c_ion_2) = rho_ion (exact up to float32 element storage, '
                     f"{SLIT_TOLERANCES['identity']:g}); the debye_length / ionic_strength scalars "
                     f"({SLIT_TOLERANCES['scalar']:g}). error = largest relative error over its tolerance; tolerance 1. "
                     'Tolerances are first estimates: not yet run.'),
        specs=_electrolyte_runs, check=_check_electrolyte_slit),
)
