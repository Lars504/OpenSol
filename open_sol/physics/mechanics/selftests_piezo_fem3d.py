"""Piezoelectric solid_mechanics_3d on tetrahedra (piezo_3d_plan stages 2 and 3, builtin Kuhn meshes): the coupled
operator identities (CPU, no model), a z-rotated lithium niobate bar in uniaxial strain under voltage / charge (exact P1 /
P2), its short / open thickness modes against the continuum and a two-layer capacitor with a void dielectric layer and a
floating island electrode (exact). The thickness bar with a resistive_load drive checks the damped modes against
the continuum. definition3d registers them."""
import math

import numpy as np

from ...core.constants import EPS0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import float32_value, lithium_niobate_entry, scaled_worst_result
from ...model.spec import FeatureSpec, MaterialSpec, StudySpec
from .selftests import _elastic
from .selftests_fem3d import UM, _box, _point, _probe, _probes, _reaction, _solid_spec

_SOLID = 'solid'
_PIEZO = {'piezoelectric': True}
_VALUE_TOLERANCE = 1e-9    # relative: electrode V / Q and reactions (float64 host solve, float32-exact constants)
_FIELD_TOLERANCE = 1e-6    # relative to each probe's scale (float32 probe samples; first estimate)


def _electrode(name, face, connection='voltage', voltage=0.0, charge=0.0):
    return FeatureSpec(name, 'electrode', [face], {'connection': connection, 'voltage': voltage, 'charge': charge})


def _relative(measured, expected, scale):
    return abs(float(measured) - float(expected)) / abs(float(scale))


# ---- solid_fem3d_piezo_operator: identities of the block operator for linear fields (CPU, no model) ----
# For u = G x and phi = -E . x (both in the P1 / P2 spaces) the bilinear forms are exact element sums of constants:
# u^T K_uu u = V S c S, u^T K_uphi phi = V grad(phi) . e S = -V E . e S, phi^T (-K_phiphi) phi = -V eps0 E . eps E, with a
# rotated crystal (all 21 c, 18 e and 6 eps non-zero); rigid u and constant phi lie in the null space; A is symmetric.
_OPERATOR_EULER = [31.0, 47.0, 73.0]
_OPERATOR_EDGES = (np.linspace(0.0, 0.75 * UM, 4), np.linspace(0.0, 0.5 * UM, 3), np.linspace(0.0, 0.5 * UM, 3))
_OPERATOR_GRADIENT = np.array([[1.0e-3, 0.3e-3, -0.2e-3], [0.5e-3, -0.4e-3, 0.6e-3], [-0.1e-3, 0.8e-3, 0.7e-3]])
_OPERATOR_FIELD = np.array([0.6e6, -1.1e6, 0.9e6])   # V/m
_OPERATOR_TOLERANCE = 1e-10                           # relative (round-off of the element sums; first estimate)


def _voigt_strain(gradient):
    return np.array([gradient[0, 0], gradient[1, 1], gradient[2, 2], gradient[1, 2] + gradient[2, 1],
                     gradient[0, 2] + gradient[2, 0], gradient[0, 1] + gradient[1, 0]])


def _check_operator(_):
    """Symmetry, the three energy identities and the null space of coupled_operator_3d, P1 and P2."""
    from ...fem.dofs3d import lagrange_dofs_3d
    from ...fem.elasticity3d import assemble_elastic_stiffness_3d
    from ...fem.mesh3d import structured_box_mesh
    from ...materials.crystal_tensors import full_crystal
    from .piezo_fem3d import block_layout, coupled_operator_3d
    crystal = full_crystal(lithium_niobate_entry(_OPERATOR_EULER))
    mesh = structured_box_mesh(*_OPERATOR_EDGES)
    count = mesh.n_elements
    stiffness = np.broadcast_to(crystal.stiffness, (count, 6, 6)).copy()
    piezo = np.broadcast_to(crystal.piezo, (count, 3, 6)).copy()
    permittivity = np.broadcast_to(crystal.permittivity, (count, 3, 3)).copy()
    everywhere = np.ones(count, dtype=bool)
    volume = float(np.prod([edges[-1] - edges[0] for edges in _OPERATOR_EDGES]))
    strain = _voigt_strain(_OPERATOR_GRADIENT)
    field = _OPERATOR_FIELD
    expected = {'uu': volume * strain @ crystal.stiffness @ strain,
                'u_phi': -volume * field @ crystal.piezo @ strain,
                'phi_phi': -volume * EPS0 * field @ crystal.permittivity @ field}
    errors = {}
    for order in (1, 2):
        dofs = lagrange_dofs_3d(mesh, order)
        operator = coupled_operator_3d(dofs, assemble_elastic_stiffness_3d(dofs, stiffness, elements=everywhere), piezo,
                                       permittivity, everywhere, everywhere)
        displacement_dofs, potential_dofs = block_layout(dofs.n_dofs)
        points = dofs.dof_points()
        linear_u, linear_phi, rigid_u, constant_phi = (np.zeros(operator.shape[0]) for _ in range(4))
        linear_u[displacement_dofs] = (points @ _OPERATOR_GRADIENT.T).reshape(-1)
        linear_phi[potential_dofs] = -points @ field
        spin = np.array([0.3, -0.7, 0.5]) * 1e-3
        rigid_u[displacement_dofs] = (np.array([1e-9, -2e-9, 0.5e-9]) + np.cross(spin, points)).reshape(-1)
        constant_phi[potential_dofs] = 1.0
        measured = {'uu': linear_u @ (operator @ linear_u), 'u_phi': linear_u @ (operator @ linear_phi),
                    'phi_phi': linear_phi @ (operator @ linear_phi)}
        for name, value in measured.items():
            errors[f'P{order}.{name}'] = abs(value - expected[name]) / abs(expected[name])
        peak = float(np.max(np.abs(operator.data)))
        errors[f'P{order}.symmetry'] = float(np.max(np.abs((operator - operator.T).data), initial=0.0)) / peak
        for name, vector in (('rigid_u', rigid_u), ('constant_phi', constant_phi)):
            errors[f'P{order}.{name}'] = (float(np.max(np.abs(operator @ vector)))
                                          / (peak * float(np.max(np.abs(vector)))))
    worst = float(np.max(list(errors.values())))
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': _OPERATOR_TOLERANCE,
            'details': {'errors': errors, 'expected': expected}}


# ---- solid_fem3d_piezo_bar_static: z-rotated LN bar, rollers on the four sides (uniaxial strain along z) ----
# euler (30, 0, 0) turns the crystal about its optic axis Z = model z: the c / e / eps entries with indices 4, 5, 6 mix
# (c14 -> c15, e15, e22 ..), but every coupling of S_zz and E_z keeps the 3m pattern (c34 = c35 = c36 = 0, e3J = (e31,
# e31, e33, 0, 0, 0), e13 = e23 = 0, eps13 = eps23 = 0), so u = (0, 0, S z) and phi linear in z solve it exactly: free
# (drive at V on z_hi, ground z_lo): T_zz = 0, S = e33 E / c33 with E = -V / L, D = (eps0 eps33 + e33^2 / c33) E, Q_drive =
# -D A (free capacitance), T_xx = c13 S - e31 E taken by the x rollers (insulating sides: D_x = 0); blocked (roller on
# z_hi too): S = 0, T_zz = -e33 E, Q_drive = eps0 eps33 A V / L (clamped); floating drive with charge Q: V = Q / C_free.
_BAR_SIZE = (0.5 * UM, 0.5 * UM, 1.0 * UM)
_BAR_EULER = [30.0, 0.0, 0.0]
_BAR_VOLTAGE = 1.0          # V on the drive electrode
_BAR_CHARGE = 1e-16         # C on the floating drive electrode
_BAR_POINT = (0.2 * UM, 0.3 * UM, 0.6 * UM)
_BAR_COMPONENTS = ('u_z', 'stress_xx', 'stress_zz', 'electric_potential', 'electric_field_x', 'electric_field_z',
                   'electric_displacement_x', 'electric_displacement_z')
_BAR_RUNS = {'free_quadratic': ('free', 'quadratic'), 'free_linear': ('free', 'linear'),
             'blocked': ('blocked', 'quadratic'), 'floating': ('floating', 'quadratic')}


def _bar_constants():
    """{c13, c23, c33, e31, e32, e33, eps33} of the rotated crystal as the float32 MaterialGrid stores them."""
    from ...materials.crystal_tensors import full_crystal
    properties = full_crystal(lithium_niobate_entry(_BAR_EULER)).properties
    return {name: float32_value(properties[key]) for name, key in (
        ('c13', 'c13'), ('c23', 'c23'), ('c33', 'c33'), ('e31', 'piezo_e31'), ('e32', 'piezo_e32'),
        ('e33', 'piezo_e33'), ('eps33', 'eps_dc_zz'))}


def _bar_spec(run):
    kind, discretization = _BAR_RUNS[run]
    features = [FeatureSpec('x_low', 'roller', ['x_lo']), FeatureSpec('x_high', 'roller', ['x_hi']),
                FeatureSpec('y_sides', 'roller', ['y_lo', 'y_hi']), FeatureSpec('anchor', 'roller', ['z_lo']),
                _electrode('ground', 'z_lo')]
    if kind == 'floating':
        features.append(_electrode('drive', 'z_hi', 'floating', charge=_BAR_CHARGE))
    else:
        features.append(_electrode('drive', 'z_hi', voltage=_BAR_VOLTAGE))
    if kind == 'blocked':
        features.append(FeatureSpec('clamp', 'roller', ['z_hi']))
    return _solid_spec(
        f'Self-test: piezoelectric LN bar on tetrahedra ({run})', _BAR_SIZE,
        [MaterialSpec('crystal', dispersion=[lithium_niobate_entry(_BAR_EULER)], selection=['all'])],
        [*features, *_probes('p', _BAR_COMPONENTS)], [_point('p', *_BAR_POINT)], discretization, settings=_PIEZO)


def _check_bar(results_by_name):
    """Probes, electrode V / Q and roller reactions of the free / blocked / floating bars against the exact fields."""
    constants = _bar_constants()
    width, depth, length = _BAR_SIZE
    area_z, area_x = width * depth, depth * length
    clamped = EPS0 * constants['eps33']
    free_permittivity = clamped + constants['e33'] ** 2 / constants['c33']
    free_capacitance = free_permittivity * area_z / length
    errors, tolerances, details = {}, {}, {}

    def add(item, error, tolerance):
        errors[item], tolerances[item] = error, tolerance

    stress_scale = constants['e33'] * _BAR_VOLTAGE / length
    for run, results in results_by_name.items():
        kind, _ = _BAR_RUNS[run]
        voltage = _BAR_CHARGE / free_capacitance if kind == 'floating' else _BAR_VOLTAGE
        field = -voltage / length
        strain = 0.0 if kind == 'blocked' else constants['e33'] * field / constants['c33']
        stress_zz = constants['c33'] * strain - constants['e33'] * field
        stress_xx = constants['c13'] * strain - constants['e31'] * field
        stress_yy = constants['c23'] * strain - constants['e32'] * field
        flux = constants['e33'] * strain + clamped * field
        charge = -flux * area_z
        scale = voltage / _BAR_VOLTAGE   # the floating run is ~Q / C_free volts
        expected = {'u_z': strain * _BAR_POINT[2], 'stress_xx': stress_xx, 'stress_zz': stress_zz,
                    'electric_potential': voltage * _BAR_POINT[2] / length, 'electric_field_x': 0.0,
                    'electric_field_z': field, 'electric_displacement_x': 0.0, 'electric_displacement_z': flux}
        scales = {'u_z': abs(constants['e33'] * field / constants['c33']) * length, 'stress_xx': scale * stress_scale,
                  'stress_zz': scale * stress_scale, 'electric_potential': voltage, 'electric_field_x': abs(field),
                  'electric_field_z': abs(field), 'electric_displacement_x': abs(flux),
                  'electric_displacement_z': abs(flux)}
        for name, value in expected.items():
            add(f'{run}.{name}', _relative(_probe(results, f'{name}_p'), value, scales[name]), _FIELD_TOLERANCE)
        drive, ground = results.result(_SOLID, 'drive'), results.result(_SOLID, 'ground')
        add(f'{run}.V_drive', _relative(drive['V'], voltage, voltage), _VALUE_TOLERANCE)
        add(f'{run}.Q_drive', _relative(drive['Q'], charge, charge), _VALUE_TOLERANCE)
        add(f'{run}.Q_ground', _relative(ground['Q'], -charge, charge), _VALUE_TOLERANCE)
        force_scale = scale * stress_scale * area_z
        add(f'{run}.F_x_low', _relative(_reaction(results, 'x_low')[0], -stress_xx * area_x, force_scale),
            _VALUE_TOLERANCE)
        add(f'{run}.F_x_high', _relative(_reaction(results, 'x_high')[0], stress_xx * area_x, force_scale),
            _VALUE_TOLERANCE)
        add(f'{run}.F_y_sides', _relative(_reaction(results, 'y_sides')[1], 0.0, force_scale), _VALUE_TOLERANCE)
        add(f'{run}.F_anchor', _relative(_reaction(results, 'anchor')[2], -stress_zz * area_z, force_scale),
            _VALUE_TOLERANCE)
        if kind == 'blocked':
            add(f'{run}.F_clamp', _relative(_reaction(results, 'clamp')[2], stress_zz * area_z, force_scale),
                _VALUE_TOLERANCE)
        details[run] = {'drive': [drive['V'], drive['Q']], 'expected_V_Q': [voltage, charge],
                        'stress_yy': stress_yy}
    details['constants'] = constants
    details['free_capacitance_F'] = free_capacitance
    return scaled_worst_result(errors, tolerances, **details)


# ---- solid_fem3d_piezo_thickness_modes: the z-rotated LN bar of the static case, longer, short / open modes ----
# Rollers on the four sides and z_lo, ground z_lo, drive z_hi: u = (0, 0, u_z(z)) and phi(z) solve the 3D problem (the 3m
# pattern of S_zz / E_z holds, see the static case), the half model of a free plate of thickness 2 L with electrodes on
# both faces. div D = 0 makes D_z uniform, so c^D u'' = -rho omega^2 u with c^D = c33 + e33^2 / eps (eps = eps0 eps33) and
# T_zz(L) = c^D u'(L) - (e33 / eps) D = 0. Open (drive floating, Q = 0): D = 0, f_a = (2 n - 1) v_D / (4 L), v_D =
# sqrt(c^D / rho), V_drive = (e33 / eps) u_z(L). Short (drive 0 V): D = e33 u_z(L) / L, so x cos x = k_t^2 sin x with x =
# omega L / v_D, k_t^2 = e33^2 / (eps c^D), and Q_drive = -D A. Mode 1 (the live one) peaks at z = L: u_z(L) = 1 m. The
# 0.125 um cross section puts every mode with lateral variation above ~14 GHz, far over the two longitudinal ones.
_THICKNESS_SIZE = (0.125 * UM, 0.125 * UM, 1.0 * UM)
_THICKNESS_CELL = 0.0625 * UM
_THICKNESS_MODES = 2
_THICKNESS_TOLERANCE = 1e-4      # relative: frequencies and mode-1 electrode values (P2, k h <= 0.3; first estimate)
_THICKNESS_TABLE = 'modes'
_THICKNESS_RUNS = ('short', 'open')


def _thickness_spec(run):
    drive = _electrode('drive', 'z_hi', 'floating') if run == 'open' else _electrode('drive', 'z_hi')
    features = [FeatureSpec('sides', 'roller', ['x_lo', 'x_hi', 'y_lo', 'y_hi']),
                FeatureSpec('anchor', 'roller', ['z_lo']), _electrode('ground', 'z_lo'), drive,
                FeatureSpec(_THICKNESS_TABLE, 'resonance_table')]
    return _solid_spec(
        f'Self-test: piezoelectric LN thickness modes on tetrahedra ({run} circuit)', _THICKNESS_SIZE,
        [MaterialSpec('crystal', dispersion=[lithium_niobate_entry(_BAR_EULER)], selection=['all'])], features,
        cell=_THICKNESS_CELL, study=StudySpec(type='eigenfrequency', settings={'n_resonances': _THICKNESS_MODES}),
        settings=_PIEZO)


def _thickness_references():
    """({'short': f_r, 'open': f_a} [Hz] of the first _THICKNESS_MODES thickness modes, constants as the float32
    MaterialGrid stores them, with c^D, k_t^2 and v_D)."""
    from scipy.optimize import brentq
    from ...materials.crystal_tensors import full_crystal
    properties = full_crystal(lithium_niobate_entry(_BAR_EULER)).properties
    constants = {name: float32_value(properties[key]) for name, key in (
        ('c33', 'c33'), ('e33', 'piezo_e33'), ('eps33', 'eps_dc_zz'), ('rho', 'rho'))}
    permittivity = EPS0 * constants['eps33']
    stiffened = constants['c33'] + constants['e33'] ** 2 / permittivity
    coupling = constants['e33'] ** 2 / (permittivity * stiffened)
    velocity = math.sqrt(stiffened / constants['rho'])
    per_root = velocity / (2.0 * math.pi * _THICKNESS_SIZE[2])   # f [Hz] per unit of x = omega L / v_D
    orders = range(1, _THICKNESS_MODES + 1)
    short = [per_root * brentq(lambda x: x * math.cos(x) - coupling * math.sin(x), (order - 0.75) * math.pi,
                               (order - 0.5) * math.pi, xtol=1e-15) for order in orders]
    open_circuit = [per_root * (order - 0.5) * math.pi for order in orders]
    constants.update({'c_D': stiffened, 'k_t_squared': coupling, 'v_D': velocity})
    return {'short': short, 'open': open_circuit}, constants


def _check_thickness(results_by_name):
    """Table frequencies against the open / short thickness modes; the live mode 1's electrode values: open V_drive =
    (e33 / eps) u_z(L), Q = 0; short Q_drive = -e33 A u_z(L) / L, Q_ground its negative."""
    expected, constants = _thickness_references()
    width, depth, length = _THICKNESS_SIZE
    voltage_scale = constants['e33'] / (EPS0 * constants['eps33'])   # V per m of u_z(L)
    charge_scale = constants['e33'] * width * depth / length          # C per m of u_z(L)
    errors, tolerances, details = {}, {}, {}

    def add(item, error):
        errors[item], tolerances[item] = error, _THICKNESS_TOLERANCE

    for run, results in results_by_name.items():
        frequencies = [float(row[1]) for row in results.result(_SOLID, _THICKNESS_TABLE)['rows']]
        for index, reference in enumerate(expected[run]):
            measured = frequencies[index] if index < len(frequencies) else math.nan
            add(f'{run}.f{index + 1}', abs(measured / reference - 1.0))
        drive, ground = results.result(_SOLID, 'drive'), results.result(_SOLID, 'ground')
        if run == 'open':
            add('open.V_drive', _relative(drive['V'], voltage_scale, voltage_scale))
            add('open.Q_ground', _relative(ground['Q'], 0.0, charge_scale))
        else:
            add('short.Q_drive', _relative(drive['Q'], -charge_scale, charge_scale))
            add('short.Q_ground', _relative(ground['Q'], charge_scale, charge_scale))
        details[run] = {'measured_Hz': frequencies, 'drive_V_Q': [drive['V'], drive['Q']],
                        'ground_V_Q': [ground['V'], ground['Q']]}
    details.update({'expected_Hz': expected, 'constants': constants,
                    'k_eff_squared_expected': 1.0 - (expected['short'][0] / expected['open'][0]) ** 2})
    return scaled_worst_result(errors, tolerances, **details)


# ---- solid_fem3d_piezo_loaded_modes: the thickness bar with the drive tied to ground through a resistor R ----
# Continuum (same constants and notation as the thickness case): Q_drive / V_drive = C0 / (1 - k_t^2 tan x / x), C0 = eps A /
# L, and the resistor closes the circuit, j omega Q = -V / R, so the damped modes are the complex roots of F(x) = 1 - k_t^2
# tan x / x + j g x = 0 with g x = omega C0 R, g = v_D C0 R / L (R -> 0: the short roots, R -> inf: the open ones; Im x > 0
# decays for exp(+j omega t)). Found by Newton along a continuation in R from the short root of mode 1. Runs: matched (R =
# 1 / (omega_m C0), omega_m the geometric mean of the short / open omega: Q of order 1 / k_t^2, tests the frozen pencil and
# the Newton step where the load is strong) and line (R = 50 Ohm: kappa_e / omega ~ 1e-7, the first-order regime of the
# LN resonator experiment).
_LOADED_RUNS = ('matched', 'line')
_LINE_RESISTANCE = 50.0                # Ohm
_LOADED_FREQUENCY_TOLERANCE = 1e-4     # |f - f_ref| / |f_ref| of the complex mode-1 frequency (as the thickness case)
_LOADED_DECAY_TOLERANCE = 1e-3         # relative: energy_decay_rate of the line run (first estimate)
_CONTINUATION_STEPS = 61
_NEWTON_STEPS = 60


def _loaded_reference(resistance):
    """(complex f [Hz] of loaded mode 1, constants with C0 [F]) of the continuum thickness bar with R [Ohm] to ground."""
    expected, constants = _thickness_references()
    width, depth, length = _THICKNESS_SIZE
    capacitance = EPS0 * constants['eps33'] * width * depth / length
    coupling, velocity = constants['k_t_squared'], constants['v_D']
    root = 2.0 * math.pi * length * expected['short'][0] / velocity   # x of the short mode 1
    full_load = velocity * capacitance * resistance / length
    for fraction in np.geomspace(1e-8, 1.0, _CONTINUATION_STEPS):
        load = fraction * full_load
        for _ in range(_NEWTON_STEPS):
            tangent = np.tan(root)
            value = 1.0 - coupling * tangent / root + 1j * load * root
            slope = -coupling * (root / np.cos(root) ** 2 - tangent) / root ** 2 + 1j * load
            step = value / slope
            root = root - step
            if abs(step) <= 1e-15 * abs(root):
                break
    return complex(root) * velocity / (2.0 * math.pi * length), {**constants, 'C0': capacitance}


def _loaded_resistance(run):
    """R [Ohm] of a loaded run: matched = 1 / (omega_m C0), line = _LINE_RESISTANCE."""
    if run == 'line':
        return _LINE_RESISTANCE
    expected, constants = _thickness_references()
    width, depth, length = _THICKNESS_SIZE
    capacitance = EPS0 * constants['eps33'] * width * depth / length
    return 1.0 / (2.0 * math.pi * math.sqrt(expected['short'][0] * expected['open'][0]) * capacitance)


def _loaded_spec(run):
    resistance = _loaded_resistance(run)
    frequency, _ = _loaded_reference(resistance)
    drive = FeatureSpec('drive', 'electrode', ['z_hi'], {'connection': 'resistive_load', 'load_resistance': resistance})
    features = [FeatureSpec('sides', 'roller', ['x_lo', 'x_hi', 'y_lo', 'y_hi']),
                FeatureSpec('anchor', 'roller', ['z_lo']), _electrode('ground', 'z_lo'), drive,
                FeatureSpec(_THICKNESS_TABLE, 'resonance_table')]
    return _solid_spec(
        f'Self-test: piezoelectric LN thickness mode with a resistive load ({run})', _THICKNESS_SIZE,
        [MaterialSpec('crystal', dispersion=[lithium_niobate_entry(_BAR_EULER)], selection=['all'])], features,
        cell=_THICKNESS_CELL, study=StudySpec(type='eigenfrequency', settings={'n_resonances': 1}),
        settings={**_PIEZO, 'target_frequency': frequency.real})


def _check_loaded(results_by_name):
    """Complex mode-1 frequency (frequency + j frequency_imag) of each run against the continuum root; line run also
    its energy_decay_rate against 4 pi Im f."""
    errors, tolerances, details = {}, {}, {}
    for run, results in results_by_name.items():
        table = results.result(_SOLID, _THICKNESS_TABLE)
        columns = table['columns']
        row = table['rows'][0] if table['rows'] else [math.nan] * len(columns)
        measured = complex(row[columns.index('frequency')], row[columns.index('frequency_imag')])
        expected, constants = _loaded_reference(_loaded_resistance(run))
        errors[f'{run}.f'] = abs(measured - expected) / abs(expected)
        tolerances[f'{run}.f'] = _LOADED_FREQUENCY_TOLERANCE
        decay = float(row[columns.index('energy_decay_rate')])
        if run == 'line':
            errors['line.decay'] = abs(decay / (4.0 * math.pi * expected.imag) - 1.0)
            tolerances['line.decay'] = _LOADED_DECAY_TOLERANCE
        details[run] = {'measured_Hz': [measured.real, measured.imag], 'expected_Hz': [expected.real, expected.imag],
                        'energy_decay_rate': decay, 'q_factor': row[columns.index('q_factor')],
                        'resistance_Ohm': _loaded_resistance(run), 'omega_C0_R': 2.0 * math.pi * expected.real
                        * constants['C0'] * _loaded_resistance(run)}
    return scaled_worst_result(errors, tolerances, **details)


# ---- solid_fem3d_capacitor: piezoelectric on, e = 0, two dielectric layers between z electrodes ----
# bottom (z < 0.25 um) and island (0.25 .. 0.5 um): an isotropic solid with eps_dc_xx / _yy / _zz = 6 / 5 / 4 and eps_xy
# = 1 (eps_xz = eps_yz = 0, so E along z leaves D along z: the insulating sides hold); top (0.5 .. 1 um): void (no
# stiffness), eps_dc 9. phi is piecewise linear in z with its kink on element faces, so P1 / P2 are exact: series runs C =
# eps0 A / (d_1 / 4 + d_2 / 9); island run: the island an Electrode (domain), floating with charge 0, so it is an
# equipotential conductor and C = eps0 A / (0.25 um / 4 + d_2 / 9); charged run: the drive floating with charge Q, V = Q /
# C. The solid needs its rollers (x_lo, y_lo, z_lo); with e = 0 it stays undeformed. Probes read node-averaged E / D:
# exact only in elements whose nodes see one field (the island run checks the low point's potential alone).
_CAPACITOR_SIZE = (0.5 * UM, 0.5 * UM, 1.0 * UM)
_LAYER_Z = (0.25 * UM, 0.5 * UM)                     # top of the bottom box, top of the island box
_BOTTOM_EPS = {'eps_dc_xx': 6.0, 'eps_dc_yy': 5.0, 'eps_dc_zz': 4.0, 'eps_dc_xy': 1.0}
_TOP_EPS = 9.0
_CAPACITOR_VOLTAGE = 2.0                             # V
_CAPACITOR_CHARGE = 2e-17                            # C (charged run)
_CAPACITOR_POINTS = {'low': (0.2 * UM, 0.3 * UM, 0.1 * UM), 'high': (0.2 * UM, 0.3 * UM, 0.8 * UM)}
_CAPACITOR_COMPONENTS = ('electric_potential', 'electric_field_x', 'electric_field_z', 'electric_displacement_x',
                         'electric_displacement_z', 'u_z')
_CAPACITOR_RUNS = {'series_quadratic': ('series', 'quadratic'), 'series_linear': ('series', 'linear'),
                   'island': ('island', 'quadratic'), 'charged': ('charged', 'quadratic')}


def _capacitor_spec(run):
    kind, discretization = _CAPACITOR_RUNS[run]
    width, depth, height = _CAPACITOR_SIZE
    geometry = [_box('bottom', 0.0, 0.0, 0.0, width, depth, _LAYER_Z[0]),
                _box('island', 0.0, 0.0, _LAYER_Z[0], width, depth, _LAYER_Z[1]),
                _box('top', 0.0, 0.0, _LAYER_Z[1], width, depth, height),
                *(_point(name, *point) for name, point in _CAPACITOR_POINTS.items())]
    materials = [MaterialSpec('film', properties=_elastic(100e9, 0.3, **_BOTTOM_EPS), selection=['bottom', 'island']),
                 MaterialSpec('gap', properties={'eps_dc': _TOP_EPS}, selection=['top'])]
    features = [FeatureSpec('x_roller', 'roller', ['x_lo']), FeatureSpec('y_roller', 'roller', ['y_lo']),
                FeatureSpec('z_roller', 'roller', ['z_lo']), _electrode('ground', 'z_lo')]
    if kind == 'charged':
        features.append(_electrode('drive', 'z_hi', 'floating', charge=_CAPACITOR_CHARGE))
    else:
        features.append(_electrode('drive', 'z_hi', voltage=_CAPACITOR_VOLTAGE))
    if kind == 'island':
        features.append(FeatureSpec('island_electrode', 'electrode_domain', ['island'],
                                    {'connection': 'floating', 'voltage': 0.0, 'charge': 0.0}))
    probes = [probe for name in _CAPACITOR_POINTS for probe in _probes(name, _CAPACITOR_COMPONENTS)]
    return _solid_spec(f'Self-test: two-layer capacitor, piezoelectric solid on tetrahedra ({run})', _CAPACITOR_SIZE,
                       materials, [*features, *probes], geometry, discretization, settings=_PIEZO)


def _check_capacitor(results_by_name):
    """Electrode V / Q (and the island's) and the low / high probes against the series-capacitor fields."""
    width, depth, height = _CAPACITOR_SIZE
    area = width * depth
    bottom_eps, top_eps = _BOTTOM_EPS['eps_dc_zz'], _TOP_EPS
    top_thickness = height - _LAYER_Z[1]
    errors, tolerances, details = {}, {}, {}

    def add(item, error, tolerance):
        errors[item], tolerances[item] = error, tolerance

    for run, results in results_by_name.items():
        kind, _ = _CAPACITOR_RUNS[run]
        bottom_thickness = _LAYER_Z[0] if kind == 'island' else _LAYER_Z[1]   # the island is equipotential
        resistance = bottom_thickness / bottom_eps + top_thickness / top_eps   # sum d / eps_r [m]
        capacitance = EPS0 * area / resistance
        voltage = _CAPACITOR_CHARGE / capacitance if kind == 'charged' else _CAPACITOR_VOLTAGE
        flux = -EPS0 * voltage / resistance          # D_z, uniform
        low_z, high_z = _CAPACITOR_POINTS['low'][2], _CAPACITOR_POINTS['high'][2]
        expected = {'low': {'electric_potential': voltage * (low_z / bottom_eps) / resistance,
                            'electric_field_z': flux / (EPS0 * bottom_eps)},
                    'high': {'electric_potential': voltage * (bottom_thickness / bottom_eps
                                                              + (high_z - _LAYER_Z[1]) / top_eps) / resistance,
                             'electric_field_z': flux / (EPS0 * top_eps)}}
        field_scale = abs(flux) / (EPS0 * min(bottom_eps, top_eps))
        for point, values in expected.items():
            values.update({'electric_field_x': 0.0, 'electric_displacement_x': 0.0, 'electric_displacement_z': flux,
                           'u_z': 0.0})
            if kind == 'island' and point == 'low':   # its element's z = 0.25 um nodes average in the island's E = 0
                values = {name: values[name] for name in ('electric_potential', 'u_z')}
            scales = {'electric_potential': voltage, 'electric_field_x': field_scale, 'electric_field_z': field_scale,
                      'electric_displacement_x': abs(flux), 'electric_displacement_z': abs(flux), 'u_z': 1e-12}
            for name, value in values.items():
                add(f'{run}.{point}.{name}', _relative(_probe(results, f'{name}_{point}'), value, scales[name]),
                    _FIELD_TOLERANCE)
        charge = capacitance * voltage
        drive, ground = results.result(_SOLID, 'drive'), results.result(_SOLID, 'ground')
        add(f'{run}.V_drive', _relative(drive['V'], voltage, voltage), _VALUE_TOLERANCE)
        add(f'{run}.Q_drive', _relative(drive['Q'], charge, charge), _VALUE_TOLERANCE)
        add(f'{run}.Q_ground', _relative(ground['Q'], -charge, charge), _VALUE_TOLERANCE)
        if kind == 'island':
            island = results.result(_SOLID, 'island_electrode')
            island_voltage = voltage * (_LAYER_Z[0] / bottom_eps) / resistance
            add(f'{run}.V_island', _relative(island['V'], island_voltage, voltage), _VALUE_TOLERANCE)
            add(f'{run}.Q_island', _relative(island['Q'], 0.0, charge), _VALUE_TOLERANCE)
        details[run] = {'drive': [drive['V'], drive['Q']], 'expected_V_Q': [voltage, charge],
                        'capacitance_F': capacitance}
    return scaled_worst_result(errors, tolerances, **details)


PIEZO_FEM_3D_CASES = (
    SelfTestCase(
        name='solid_fem3d_piezo_operator', label='Solid mechanics 3D on tetrahedra: piezoelectric operator identities',
        description=('piezo_fem3d.coupled_operator_3d assembled directly (no model run, P1 and P2) on a 0.75 x 0.5 x 0.5 '
                     f'um Kuhn box with lithium niobate rotated to euler_deg {_OPERATOR_EULER} (all 21 c, 18 e and 6 '
                     'eps non-zero): for u = G x and phi = -E . x the forms u^T K_uu u = V S c S, u^T K_uphi phi = -V E '
                     '. e S and phi^T (-K_phiphi) phi = -V eps0 E . eps E (exact sums of element constants), A '
                     'symmetric, rigid u and constant phi in the null space. error = the largest relative deviation, '
                     f'tolerance {_OPERATOR_TOLERANCE:g} (first estimate, not yet run). Pins the coupling block and its '
                     'sign / transposition against the full tensors.'),
        specs=dict, check=_check_operator),
    SelfTestCase(
        name='solid_fem3d_piezo_bar_static', label='Solid mechanics 3D on tetrahedra: piezoelectric bar, static',
        description=(f'Piezoelectric solid_mechanics_3d on builtin tetrahedra (0.25 um Kuhn cubes) of a '
                     f'{_BAR_SIZE[0] / UM:g} x {_BAR_SIZE[1] / UM:g} x {_BAR_SIZE[2] / UM:g} um bar of lithium niobate '
                     f"('full' crystal entry at euler_deg {_BAR_EULER}: turned about the optic axis z, so the 4 / 5 / 6 "
                     'entries mix but S_zz and E_z couple as in 3m), rollers on the four sides and z_lo, electrodes '
                     f'ground (z_lo) and drive (z_hi). Runs free_quadratic / free_linear (drive {_BAR_VOLTAGE:g} V): '
                     'S_zz = e33 E / c33, T_zz = 0, Q = (eps0 eps33 + e33^2 / c33) A V / L; blocked (roller on z_hi '
                     'too): T_zz = -e33 E, Q = eps0 eps33 A V / L, clamp / anchor reactions; floating (drive floating '
                     f'with {_BAR_CHARGE:g} C): V = Q / C_free. Exact in P1 / P2: probes u_z, stress_xx / _zz, potential, '
                     f'E and D within {_FIELD_TOLERANCE:g} of their scale, electrode V / Q and every roller reaction '
                     f'within {_VALUE_TOLERANCE:g} (float32-exact constants). measured = error = the worst deviation in '
                     'tolerance units, tolerance 1 (first estimates, not yet run).'),
        specs=lambda: {run: _bar_spec(run) for run in _BAR_RUNS}, check=_check_bar),
    SelfTestCase(
        name='solid_fem3d_piezo_thickness_modes',
        label='Solid mechanics 3D on tetrahedra: piezoelectric thickness modes, short / open',
        description=('eigenfrequency study (2 modes, target 0) of piezoelectric solid_mechanics_3d on quadratic builtin '
                     f'tetrahedra ({_THICKNESS_CELL / UM:g} um Kuhn cubes) of a {_THICKNESS_SIZE[0] / UM:g} x '
                     f"{_THICKNESS_SIZE[1] / UM:g} x {_THICKNESS_SIZE[2] / UM:g} um bar of the static case's z-rotated "
                     'lithium niobate, rollers on the four sides and z_lo, ground z_lo (shorted) and drive z_hi: run '
                     'short (drive 0 V: the roots of x cos x = k_t^2 sin x) and open (drive floating: f_a = (2 n - 1) v_D '
                     '/ (4 L), c^D = c33 + e33^2 / (eps0 eps33)), plus the live mode 1 electrode values (open V_drive = '
                     '(e33 / eps) x 1 m, Q_ground 0; short Q_drive = -e33 A / L x 1 m, Q_ground its negative). Pins the '
                     'coupled pencil (massless potential), the short / open electrode rules and the block-layout peak '
                     f'normalisation. Every item within {_THICKNESS_TOLERANCE:g} relative; measured = error = the worst '
                     'deviation in tolerance units, tolerance 1 (first estimate, not yet run).'),
        specs=lambda: {run: _thickness_spec(run) for run in _THICKNESS_RUNS}, check=_check_thickness),
    SelfTestCase(
        name='solid_fem3d_piezo_loaded_modes',
        label='Solid mechanics 3D on tetrahedra: piezoelectric thickness mode with a resistive load',
        description=('eigenfrequency study (1 mode, target = the continuum Re f) of the thickness-mode bar of '
                     'solid_fem3d_piezo_thickness_modes with the drive electrode connection resistive_load: the complex '
                     'mode-1 frequency against the continuum root of 1 - k_t^2 tan x / x + j omega C0 R = 0 (x = omega L '
                     '/ v_D, C0 = eps0 eps33 A / L). Runs matched (R = 1 / (omega_m C0), omega_m between the short and '
                     'open roots: strong damping, Q ~ 1 / k_t^2) and line (R = 50 Ohm: kappa_e / omega ~ 1e-7, the '
                     f'first-order regime). Complex f within {_LOADED_FREQUENCY_TOLERANCE:g} relative; the line run\'s '
                     f'energy_decay_rate within {_LOADED_DECAY_TOLERANCE:g} of 4 pi Im f. Pins the load term (sign, s^2 '
                     '/ R scaling), the frozen complex symmetric shift-invert, the Newton step on the Rayleigh '
                     'functional and the damped table columns. measured = error = the worst deviation in tolerance '
                     'units, tolerance 1 (first estimates, not yet run).'),
        specs=lambda: {run: _loaded_spec(run) for run in _LOADED_RUNS}, check=_check_loaded),
    SelfTestCase(
        name='solid_fem3d_capacitor', label='Solid mechanics 3D on tetrahedra: piezoelectric capacitor, void layer',
        description=('Piezoelectric solid_mechanics_3d with e = 0 on builtin tetrahedra of a 0.5 x 0.5 x 1 um box: an '
                     'isotropic solid film (z < 0.5 um, eps_dc_xx / _yy / _zz 6 / 5 / 4, eps_dc_xy 1) under a void '
                     'layer (eps_dc 9, no stiffness: the potential lives on void elements too), electrodes ground (z_lo) '
                     'and drive (z_hi). Runs series_quadratic / series_linear: C = eps0 A / (d_1 / 4 + d_2 / 9), the '
                     'piecewise-linear potential, E and D at a probe in each layer; island: an Electrode (domain) '
                     'floating with charge 0 on the film part 0.25 .. 0.5 um (an equipotential conductor: its V from '
                     'the divider, its Q 0, C of the shortened film); charged: the drive floating with '
                     f'{_CAPACITOR_CHARGE:g} C, V = Q / C. Exact in P1 / P2: probes within {_FIELD_TOLERANCE:g}, '
                     f'electrode values within {_VALUE_TOLERANCE:g}. measured = error = the worst deviation in '
                     'tolerance units, tolerance 1 (first estimates, not yet run).'),
        specs=lambda: {run: _capacitor_spec(run) for run in _CAPACITOR_RUNS}, check=_check_capacitor),
)
