"""Electrostatics / electric currents on tetrahedron meshes (FEMPotentialPhysics3D, piezo_3d_plan stage 6): a layered
anisotropic parallel-plate capacitor along each axis (builtin Kuhn meshes, exact in P1 / P2), a coaxial capacitor on
gmsh tetrahedra (skipped without gmsh) and a resistor bar (current and Joule power); model runs read through feature
results (held tables), scalar outputs and derived integrals. Registered by definition.register."""
import math

from ...core.constants import EPS0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import float32_value, skip_result
from ...model.spec import (DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec,
                           PhysicsSpec, SolverDefaultsSpec, StudySpec)
from .selftests import _held_flux
from .selftests_fem import _scaled_result

ELECTROSTATICS_TYPE_NAME = 'electrostatics'   # = definition.ELECTROSTATICS_TYPE_NAME (definition imports this module)
CURRENTS_TYPE_NAME = 'electric_currents'
UM = 1e-6

# ---- electrostatics_fem3d_plates: two diagonal anisotropic layers (z < LAYER_Z, z > LAYER_Z) between plates ----
PLATE_SIZE = (1.0 * UM, 0.75 * UM, 1.0 * UM)
PLATE_CELL = 0.25 * UM              # builtin Kuhn cubes: the layer interface z = LAYER_Z is a node plane
LAYER_Z = 0.5 * UM
LAYER_PERMITTIVITIES = {'lower': {'eps_dc_xx': 3.0, 'eps_dc_yy': 5.0, 'eps_dc_zz': 7.0},
                        'upper': {'eps_dc_xx': 2.0, 'eps_dc_yy': 4.0, 'eps_dc_zz': 9.0}}
PLATE_VOLTAGE = 2.0                 # V on <axis>_lo, 0 V on <axis>_hi
PLATE_RUNS = {'z_quadratic': ('z', 'quadratic'), 'z_linear': ('z', 'linear'), 'x_quadratic': ('x', 'quadratic'),
              'y_linear': ('y', 'linear')}
# piecewise linear V: in the P1 and P2 spaces, so C and the balance are exact up to the float64 solve (first estimate)
PLATE_TOLERANCE = 1e-9

# ---- electrostatics_fem3d_coax: inner cylinder r = a at V0 (interface faces), outer shell r >= b a terminal at 0 V ----
COAX_INNER_RADIUS = 1.0 * UM        # a
COAX_OUTER_RADIUS = 2.0 * UM        # b
COAX_SIDE = 4.6 * UM                # square window around the axis
COAX_HEIGHT = 0.6 * UM              # insulated z faces: the 2D coax per length times the height
COAX_EPS = 4.0
COAX_VOLTAGE = 1.0
COAX_ELEMENT = 0.3 * UM             # h_max
COAX_CURVATURE = 0.15               # chords 0.15 r on both circles: equal facet ratios (their ln(b / a) error cancels)
COAX_TOLERANCE = 2e-3               # relative C error: faceted cylinders, P2 (first estimate, not yet run)
COAX_BALANCE = 1e-8                 # |Q_inner + Q_outer| / Q_inner (Gauss: exact up to the solve)

# ---- currents_fem3d_resistor: anisotropic bar, voltage on <axis>_lo, 0 V on <axis>_hi ----
BAR_SIZE = (1.0 * UM, 0.5 * UM, 0.75 * UM)
BAR_CELL = 0.25 * UM
BAR_CONDUCTIVITIES = {'sigma_dc_xx': 2e6, 'sigma_dc_yy': 3e6, 'sigma_dc_zz': 5e6}   # S/m
BAR_VOLTAGE = 0.1
BAR_RUNS = {'x_quadratic': ('x', 'quadratic'), 'z_linear': ('z', 'linear')}
# current: exact (linear V, float64 reactions); joule: the float32 per-element q_joule (first estimates)
BAR_TOLERANCES = {'current': 1e-9, 'balance': 1e-9, 'scalar': 1e-9, 'joule': 1e-6}

_AXES = 'xyz'


def _tetrahedra_mesh(cell):
    return MeshSpec(name='tetrahedra', kind='tetrahedra', mode='uniform', dx=cell, mesher='builtin')


def _model(title, size, physics_name, type_name, features, discretization, mesh, geometry=(), materials=(),
           derived=()):
    """Stationary single-physics model on a tetrahedra mesh (cpu backend, Automatic host SuperLU)."""
    return ModelSpec(
        title=title,
        domain=DomainSpec(size_x=size[0], size_y=size[1], size_z=size[2]),
        geometry=list(geometry),
        materials=list(materials),
        physics=[PhysicsSpec(physics_name, type_name, settings={'discretization': discretization},
                             features=list(features))],
        meshes=[mesh],
        solver_defaults=SolverDefaultsSpec(backend='cpu'),
        study=StudySpec(type='stationary'),
        derived_values=list(derived))


# ---- layered plate capacitor ----
def _layer_boxes(size):
    """GeometrySpecs 'lower' (z < LAYER_Z) and 'upper' (z > LAYER_Z) covering the window."""
    return [GeometrySpec(name, 'box', {'x0': 0.0, 'y0': 0.0, 'z0': z0, 'x1': size[0], 'y1': size[1], 'z1': z1})
            for name, z0, z1 in (('lower', 0.0, LAYER_Z), ('upper', LAYER_Z, size[2]))]


def plates_spec(axis, discretization) -> ModelSpec:
    """Plates on <axis>_lo (PLATE_VOLTAGE) and <axis>_hi (0 V) around the two layers, builtin Kuhn tetrahedra."""
    return _model(
        f'Self-test: layered anisotropic plate capacitor on tetrahedra ({axis}, {discretization})', PLATE_SIZE,
        'electrostatics', ELECTROSTATICS_TYPE_NAME,
        [FeatureSpec('plate_lo', 'voltage', [f'{axis}_lo'], {'voltage': PLATE_VOLTAGE}),
         FeatureSpec('plate_hi', 'voltage', [f'{axis}_hi'], {'voltage': 0.0})],
        discretization, _tetrahedra_mesh(PLATE_CELL), geometry=_layer_boxes(PLATE_SIZE),
        materials=[MaterialSpec(f'{name}_crystal', properties=dict(values), selection=[name])
                   for name, values in LAYER_PERMITTIVITIES.items()])


def _plates_expected_capacitance(axis):
    """C [F] of the layered plates along axis: z = the layers in series, x / y = in parallel (float32 eps)."""
    thicknesses = {'lower': LAYER_Z, 'upper': PLATE_SIZE[2] - LAYER_Z}
    eps = {name: float32_value(values[f'eps_dc_{axis}{axis}']) for name, values in LAYER_PERMITTIVITIES.items()}
    if axis == 'z':
        return EPS0 * PLATE_SIZE[0] * PLATE_SIZE[1] / sum(thicknesses[name] / eps[name] for name in eps)
    other = PLATE_SIZE[1 - _AXES.index(axis)]   # the in-plane side across the field (y for x plates, x for y plates)
    return EPS0 * other * sum(thicknesses[name] * eps[name] for name in eps) / PLATE_SIZE[_AXES.index(axis)]


def _check_plates(results_by_name):
    """C = Q_lo / V0 against the series / parallel closed form and |Q_lo + Q_hi| / Q_lo per run; tolerance 1 on the
    errors over PLATE_TOLERANCE."""
    relative, capacitances = {}, {}
    for run, (axis, _) in PLATE_RUNS.items():
        results = results_by_name[run]
        charge_lo = _held_flux(results, 'electrostatics', 'plate_lo', 'charge')
        charge_hi = _held_flux(results, 'electrostatics', 'plate_hi', 'charge')
        expected = _plates_expected_capacitance(axis)
        capacitances[run] = {'measured_F': charge_lo / PLATE_VOLTAGE, 'expected_F': expected}
        relative[f'{run}_capacitance'] = abs(charge_lo / PLATE_VOLTAGE / expected - 1.0)
        relative[f'{run}_balance'] = abs(charge_lo + charge_hi) / abs(charge_lo)
    return _scaled_result(relative, {name: PLATE_TOLERANCE for name in relative}, capacitances=capacitances)


# ---- coaxial capacitor (gmsh) ----
def coax_spec() -> ModelSpec:
    """Inner circle prism 'inner' at COAX_VOLTAGE (Dirichlet on its interface faces), the window minus the circle of
    radius b ('shell') a terminal at 0 V, eps_dc COAX_EPS everywhere, quadratic elements on gmsh tetrahedra."""
    centre = 0.5 * COAX_SIDE
    return _model(
        'Self-test: coaxial capacitor on gmsh tetrahedra', (COAX_SIDE, COAX_SIDE, COAX_HEIGHT), 'electrostatics',
        ELECTROSTATICS_TYPE_NAME,
        [FeatureSpec('inner_electrode', 'voltage', ['inner'], {'voltage': COAX_VOLTAGE}),
         FeatureSpec('outer_electrode', 'terminal', ['shell'], {'voltage': 0.0})],
        'quadratic',
        MeshSpec(name='tetrahedra', kind='tetrahedra', mode='uniform', dx=COAX_ELEMENT, mesher='gmsh',
                 curvature_factor=COAX_CURVATURE),
        geometry=[GeometrySpec('inner', 'circle', {'cx': centre, 'cy': centre, 'r': COAX_INNER_RADIUS}),
                  GeometrySpec('hole', 'circle', {'cx': centre, 'cy': centre, 'r': COAX_OUTER_RADIUS}),
                  GeometrySpec('frame', 'rectangle', {'x0': 0.0, 'y0': 0.0, 'x1': COAX_SIDE, 'y1': COAX_SIDE}),
                  GeometrySpec('shell', 'difference', {'base': 'frame', 'subtract': ['hole']})],
        materials=[MaterialSpec('dielectric', properties={'eps_dc': COAX_EPS}, selection=['all'])])


def _coax_runs():
    """The quadratic coax run on gmsh tetrahedra; {} without gmsh (the check then passes as skipped)."""
    from ...fem.gmsh_io import find_gmsh
    return {} if find_gmsh() is None else {'quadratic': coax_spec()}


def _check_coax(results_by_name):
    """C = Q_inner / V0 against 2 pi eps0 eps H / ln(b / a) and the charge balance; skipped without gmsh."""
    if not results_by_name:
        return skip_result('gmsh not found (set OPEN_SOL_GMSH or put gmsh on PATH)')
    results = results_by_name['quadratic']
    expected = (2.0 * math.pi * EPS0 * float32_value(COAX_EPS) * COAX_HEIGHT
                / math.log(COAX_OUTER_RADIUS / COAX_INNER_RADIUS))
    inner = _held_flux(results, 'electrostatics', 'inner_electrode', 'charge')
    outer = _held_flux(results, 'electrostatics', 'outer_electrode', 'charge')
    relative = {'capacitance': abs(inner / COAX_VOLTAGE / expected - 1.0), 'balance': abs(inner + outer) / abs(inner)}
    return _scaled_result(relative, {'capacitance': COAX_TOLERANCE, 'balance': COAX_BALANCE},
                          measured_capacitance_F=inner / COAX_VOLTAGE, expected_capacitance_F=expected)


# ---- resistor bar ----
def resistor_spec(axis, discretization) -> ModelSpec:
    """Anisotropic bar with BAR_VOLTAGE on <axis>_lo and 0 V on <axis>_hi; derived P_joule = int q_joule and the
    left current scalar output."""
    return _model(
        f'Self-test: anisotropic resistor bar on tetrahedra ({axis}, {discretization})', BAR_SIZE, 'currents',
        CURRENTS_TYPE_NAME,
        [FeatureSpec('left', 'voltage', [f'{axis}_lo'], {'voltage': BAR_VOLTAGE}),
         FeatureSpec('right', 'voltage', [f'{axis}_hi'], {'voltage': 0.0})],
        discretization, _tetrahedra_mesh(BAR_CELL),
        materials=[MaterialSpec('resistor', properties=dict(BAR_CONDUCTIVITIES), selection=['all'])],
        derived=[DerivedValueSpec('P_joule', 'q_joule', [], 'integral'),
                 DerivedValueSpec('I_left', 'currents_current_left', [], 'mean')])


def _bar_expected_current(axis):
    """I [A] = sigma_aa A V / L along axis (float32 sigma)."""
    index = _AXES.index(axis)
    area = math.prod(size for other, size in enumerate(BAR_SIZE) if other != index)
    return float32_value(BAR_CONDUCTIVITIES[f'sigma_dc_{axis}{axis}']) * area * BAR_VOLTAGE / BAR_SIZE[index]


def _check_resistor(results_by_name):
    """Per run: the left current against sigma A V / L, the balance I_left + I_right = 0, the current_left scalar
    output, and int q_joule = V I_left; tolerance 1 on the errors over BAR_TOLERANCES."""
    relative, tolerances, measured = {}, {}, {}
    for run, (axis, _) in BAR_RUNS.items():
        results = results_by_name[run]
        derived = results.summary()['derived']
        current_left = _held_flux(results, 'currents', 'left', 'current')
        current_right = _held_flux(results, 'currents', 'right', 'current')
        power = float(derived['P_joule'])
        measured[run] = {'I_left_A': current_left, 'I_expected_A': _bar_expected_current(axis), 'P_joule_W': power}
        for name, value in (('current', abs(current_left / _bar_expected_current(axis) - 1.0)),
                            ('balance', abs(current_left + current_right) / abs(current_left)),
                            ('scalar', abs(float(derived['I_left']) / current_left - 1.0)),
                            ('joule', abs(power / (BAR_VOLTAGE * current_left) - 1.0))):
            relative[f'{run}_{name}'] = value
            tolerances[f'{run}_{name}'] = BAR_TOLERANCES[name]
    return _scaled_result(relative, tolerances, measured=measured)


FEM_3D_SELFTEST_CASES = (
    SelfTestCase(
        name='electrostatics_fem3d_plates', label='Electrostatics on tetrahedra: layered anisotropic plate capacitor',
        description=(f'Electrostatics on builtin tetrahedra (Kuhn cubes of {PLATE_CELL / UM:g} um, no gmsh) of a '
                     f'{PLATE_SIZE[0] / UM:g} x {PLATE_SIZE[1] / UM:g} x {PLATE_SIZE[2] / UM:g} um box of two layers '
                     f'(z below / above {LAYER_Z / UM:g} um, a node plane) with diagonal eps_dc (xx, yy, zz) = (3, 5, '
                     f'7) and (2, 4, 9), plates at {PLATE_VOLTAGE:g} V and 0 V on the outer faces of one axis (z: the '
                     'layers in series, quadratic and linear; x quadratic and y linear: in parallel), the other faces '
                     'insulated. V is piecewise linear, so P1 and P2 hold it: C = Q_lo / V0 against eps0 A / sum(d / '
                     'eps_zz) and eps0 sum(eps_aa d) w / L, and Q_lo + Q_hi = 0 (Gauss). error = the largest relative '
                     f'error over {PLATE_TOLERANCE:g}; tolerance 1 (first estimate, not yet run).'),
        specs=lambda: {run: plates_spec(*settings) for run, settings in PLATE_RUNS.items()}, check=_check_plates),
    SelfTestCase(
        name='electrostatics_fem3d_coax', label='Electrostatics on tetrahedra: coaxial capacitor on gmsh tetrahedra',
        description=(f'Needs gmsh (else skipped and noted): eps_dc {COAX_EPS:g} between an inner circle prism r = '
                     f'{COAX_INNER_RADIUS / UM:g} um held at {COAX_VOLTAGE:g} V (voltage on its interface faces) and the '
                     f'window minus a circle of r = {COAX_OUTER_RADIUS / UM:g} um as a 0 V terminal, in a '
                     f'{COAX_SIDE / UM:g} x {COAX_SIDE / UM:g} x {COAX_HEIGHT / UM:g} um box with insulated z faces; '
                     f'quadratic elements on gmsh tetrahedra (h_max {COAX_ELEMENT / UM:g} um, curvature_factor '
                     f'{COAX_CURVATURE:g} on both circles, so their facet ratios match). C = Q_inner / V0 against 2 pi '
                     f'eps0 eps H / ln(b / a) ({COAX_TOLERANCE:g}: faceted cylinders; first estimate, not yet run) and '
                     f'|Q_inner + Q_outer| / Q_inner ({COAX_BALANCE:g}). error = the largest relative error over its '
                     'tolerance; tolerance 1.'),
        specs=_coax_runs, check=_check_coax),
    SelfTestCase(
        name='currents_fem3d_resistor', label='Electric currents on tetrahedra: anisotropic resistor bar',
        description=(f'Electric currents on builtin tetrahedra (Kuhn cubes of {BAR_CELL / UM:g} um) of a '
                     f'{BAR_SIZE[0] / UM:g} x {BAR_SIZE[1] / UM:g} x {BAR_SIZE[2] / UM:g} um bar with sigma_dc (xx, yy, '
                     f'zz) = (2e6, 3e6, 5e6) S/m, {BAR_VOLTAGE:g} V and 0 V on the x faces (quadratic) or z faces '
                     '(linear): the left current against sigma_aa A V / L, I_left + I_right = 0 (exact: linear V; '
                     f"{BAR_TOLERANCES['current']:g}), the current_left scalar output read by a derived value, and the "
                     f'derived integral of q_joule against V I_left (element dissipation densities, float32; '
                     f"{BAR_TOLERANCES['joule']:g}). error = the largest relative error over its tolerance; tolerance "
                     '1 (first estimates, not yet run).'),
        specs=lambda: {run: resistor_spec(*settings) for run, settings in BAR_RUNS.items()}, check=_check_resistor),
)
