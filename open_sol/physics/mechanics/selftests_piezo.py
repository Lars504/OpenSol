"""Model-run self-tests of piezoelectric solid_mechanics (piezoelectric plan stage 5): a bar in uniaxial strain (rollers
on the y faces) with electrodes on its x faces, static (free / blocked / floating) and modal (short / open circuit) runs
against the exact Q1 discrete solutions; no physics imports."""
import math

import numpy as np

from ...core.constants import EPS0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import float32_value, scaled_worst_result
from ...model.spec import DomainSpec, FeatureSpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec, StudySpec

_PHYSICS = 'solid'
_SOLID_TYPE = 'solid_mechanics'   # = selftests.SOLID_TYPE_NAME
_STATIC_STUDY = 'stationary'
_MODAL_STUDY = 'eigenfrequency'   # = model.study_eigenfrequency.STUDY_NAME
_TABLE = 'modes'
_FREQUENCY_COLUMN = 1             # frequency [Hz] in the resonance_table rows

# ---- bar: 10 x 1 mm, uniform 0.5 mm cells (20 x 2), E 70 GPa, nu 0.25 (c11 = lambda + 2 mu = 84 GPa), rho 2200, only
# piezo_e11 = 1.36 C/m^2 and eps_dc 10 (e^2 / (eps c11) = 0.249, K_t^2 = e^2 / (eps c_D) = 0.199). Rollers on y_lo / y_hi
# keep u_y = 0 and every field y-uniform, so the 2D Q1 solution is the 1D linear-element one (the y factors drop out);
# 'anchor' = roller u_x on x_lo, electrodes 'ground' (x_lo, 0 V) and 'drive' (x_hi).
# Static (linear u and phi lie in the element space, so Q1 is exact): free end, drive at V: T = 0, S = e E / c11 with E
# = -V / L, D = (eps + e^2 / c11) E, Q_drive = (eps + e^2 / c11) V W / L (the free capacitance), anchor reaction 0;
# blocked (roller 'clamp' on x_hi too): S = 0, T = -e E = e V / L, Q_drive = eps V W / L (clamped), clamp reaction +e V
# W / L, anchor -e V W / L; floating drive with charge Q: V = Q L / ((eps + e^2 / c11) W).
# Modal (fixed-free bar, u_x(0) = 0): the discrete D per element is uniform (Gauss rows) and phi(0) = 0, so open
# (floating drive, Q = 0) gives D = 0 and the plain chain at c_D = c11 + e^2 / eps: omega^2 = 6 c / (rho h^2) (1 - cos t)
# / (2 + cos t), t = (2 n - 1) pi / (2 N) (sin modes, the free end a mirror of the 2N chain; frequency_response uses
# it at c11 too); short (drive at 0 V) D = e u_N / L, i.e. the c_D chain minus e^2 / (eps L) on the end node:
# the reference is that N x N pencil solved densely. Open mode 1 has phi = (e / eps) u at every node.
_LENGTH, _WIDTH = 10e-3, 1e-3     # m
_CELL = 0.5e-3                    # m
_CELLS = (20, 2)
# material constants exact in float32 (the MaterialGrid storage; 1.36 would be off by 1e-8, beyond _VALUE_TOLERANCE)
_YOUNGS_MODULUS = float32_value(70e9)   # Pa
_POISSON_RATIO = 0.25
_DENSITY = 2200.0                 # kg/m^3
_PIEZO_E11 = float32_value(1.36)  # C/m^2
_EPS_DC = 10.0
_VOLTAGE = 100.0                  # V on the drive electrode
_CHARGE = 1e-9                    # C/m on the floating drive electrode
_N_MODES = 3
_VALUE_TOLERANCE = 1e-9           # relative: electrode V / Q and reactions (float64 host solve)
_FIELD_TOLERANCE = 1e-5           # relative to each field's scale (float32 outputs)
_FREQUENCY_TOLERANCE = 1e-8       # relative (float64 pencil, eigsh to round-off)


def _moduli():
    """(c11 [Pa], eps [F/m], e [C/m^2]) of the bar material."""
    mu = _YOUNGS_MODULUS / (2.0 * (1.0 + _POISSON_RATIO))
    lame_lambda = _YOUNGS_MODULUS * _POISSON_RATIO / ((1.0 + _POISSON_RATIO) * (1.0 - 2.0 * _POISSON_RATIO))
    return lame_lambda + 2.0 * mu, EPS0 * _EPS_DC, _PIEZO_E11


def _bar_spec(title, study, features):
    """The piezoelectric bar with rails (y rollers) plus `features`, in a stationary or eigenfrequency study."""
    study_settings = {'n_resonances': _N_MODES} if study == _MODAL_STUDY else {}
    if study == _MODAL_STUDY:
        features = features + [FeatureSpec(_TABLE, 'resonance_table')]
    return ModelSpec(
        title=f'Self-test: piezoelectric bar, {title} (solid mechanics)',
        domain=DomainSpec(size_x=_LENGTH, size_y=_WIDTH),
        materials=[MaterialSpec('piezo', properties={'youngs_modulus': _YOUNGS_MODULUS, 'poisson_ratio': _POISSON_RATIO,
                                                     'rho': _DENSITY, 'piezo_e11': _PIEZO_E11, 'eps_dc': _EPS_DC},
                                selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS, _SOLID_TYPE, settings={'piezoelectric': True},
                             features=[FeatureSpec('rails', 'roller', ['y_lo', 'y_hi']),
                                       FeatureSpec('anchor', 'roller', ['x_lo'])] + features)],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type=study, settings=study_settings))


def _ground():
    return FeatureSpec('ground', 'electrode', ['x_lo'], params={'connection': 'voltage', 'voltage': 0.0})


def _drive(connection, voltage=0.0, charge=0.0):
    return FeatureSpec('drive', 'electrode', ['x_hi'], params={'connection': connection, 'voltage': voltage,
                                                               'charge': charge})


def _static_specs():
    return {'free': _bar_spec('free end under voltage', _STATIC_STUDY, [_ground(), _drive('voltage', _VOLTAGE)]),
            'blocked': _bar_spec('blocked under voltage', _STATIC_STUDY,
                                 [FeatureSpec('clamp', 'roller', ['x_hi']), _ground(), _drive('voltage', _VOLTAGE)]),
            'floating': _bar_spec('free end, charged floating electrode', _STATIC_STUDY,
                                  [_ground(), _drive('floating', charge=_CHARGE)])}


def _field(results, name):
    values = np.asarray(results.field(f'{_PHYSICS}.{name}').values, dtype=np.float64)
    if values.shape != _CELLS:
        raise ValueError(f'{name} window shape {values.shape} differs from the {_CELLS} bar cells')
    return values


def _relative(measured, expected, scale):
    return abs(float(measured) - float(expected)) / abs(float(scale))


def _field_error(results, name, expected, scale):
    return float(np.max(np.abs(_field(results, name) - expected))) / abs(float(scale))


def _check_static(results_by_name):
    """Free / blocked / floating bars against the exact linear solution: fields (float32) and electrode V / Q and
    reactions (float64); measured = error = the worst deviation in tolerance units."""
    c11, eps, piezo_e = _moduli()
    field_x = -_VOLTAGE / _LENGTH
    free_strain = piezo_e * field_x / c11
    blocked_stress = -piezo_e * field_x
    free_capacitance = (eps + piezo_e ** 2 / c11) * _WIDTH / _LENGTH   # F/m per m of depth
    x_centres = ((np.arange(_CELLS[0]) + 0.5) * _CELL)[:, None]
    force_scale = blocked_stress * _WIDTH
    errors, tolerances = {}, {}

    def add(item, error, tolerance):
        errors[item], tolerances[item] = error, tolerance

    free = results_by_name['free']
    add('free strain_xx', _field_error(free, 'strain_xx', free_strain, free_strain), _FIELD_TOLERANCE)
    add('free u_x', _field_error(free, 'u_x', free_strain * x_centres, free_strain * _LENGTH), _FIELD_TOLERANCE)
    add('free stress_xx', _field_error(free, 'stress_xx', 0.0, blocked_stress), _FIELD_TOLERANCE)
    add('free potential', _field_error(free, 'electric_potential', _VOLTAGE * x_centres / _LENGTH, _VOLTAGE),
        _FIELD_TOLERANCE)
    add('free E_x', _field_error(free, 'electric_field_x', field_x, field_x), _FIELD_TOLERANCE)
    add('free E_y', _field_error(free, 'electric_field_y', 0.0, field_x), _FIELD_TOLERANCE)
    drive = free.result(_PHYSICS, 'drive')
    add('free Q drive', _relative(drive['Q'], free_capacitance * _VOLTAGE, free_capacitance * _VOLTAGE),
        _VALUE_TOLERANCE)
    add('free Q ground', _relative(free.result(_PHYSICS, 'ground')['Q'], -free_capacitance * _VOLTAGE,
                                   free_capacitance * _VOLTAGE), _VALUE_TOLERANCE)
    add('free anchor F_x', _relative(free.result(_PHYSICS, 'anchor')['F_x'], 0.0, force_scale), _VALUE_TOLERANCE)

    blocked = results_by_name['blocked']
    clamped_charge = eps * _WIDTH / _LENGTH * _VOLTAGE
    add('blocked strain_xx', _field_error(blocked, 'strain_xx', 0.0, free_strain), _FIELD_TOLERANCE)
    add('blocked stress_xx', _field_error(blocked, 'stress_xx', blocked_stress, blocked_stress), _FIELD_TOLERANCE)
    add('blocked Q drive', _relative(blocked.result(_PHYSICS, 'drive')['Q'], clamped_charge, clamped_charge),
        _VALUE_TOLERANCE)
    add('blocked clamp F_x', _relative(blocked.result(_PHYSICS, 'clamp')['F_x'], force_scale, force_scale),
        _VALUE_TOLERANCE)
    add('blocked anchor F_x', _relative(blocked.result(_PHYSICS, 'anchor')['F_x'], -force_scale, force_scale),
        _VALUE_TOLERANCE)

    floating = results_by_name['floating']
    floating_voltage = _CHARGE / free_capacitance
    floating_drive = floating.result(_PHYSICS, 'drive')
    add('floating V drive', _relative(floating_drive['V'], floating_voltage, floating_voltage), _VALUE_TOLERANCE)
    add('floating Q drive', _relative(floating_drive['Q'], _CHARGE, _CHARGE), _VALUE_TOLERANCE)
    add('floating Q ground', _relative(floating.result(_PHYSICS, 'ground')['Q'], -_CHARGE, _CHARGE), _VALUE_TOLERANCE)
    add('floating strain_xx', _field_error(floating, 'strain_xx', -piezo_e * floating_voltage / (_LENGTH * c11),
                                           free_strain), _FIELD_TOLERANCE)
    return scaled_worst_result(errors, tolerances, free_strain=free_strain, blocked_stress=blocked_stress,
                               free_capacitance=free_capacitance, floating_voltage=floating_voltage,
                               drive_free=[drive['V'], drive['Q']],
                               drive_floating=[floating_drive['V'], floating_drive['Q']])


def _modal_specs():
    return {'short': _bar_spec('short-circuit modes', _MODAL_STUDY, [_ground(), _drive('voltage')]),
            'open': _bar_spec('open-circuit modes', _MODAL_STUDY, [_ground(), _drive('floating')])}


def _chain_frequencies(modulus):
    """[Hz] of the lowest _N_MODES discrete fixed-free chain modes at axial modulus [Pa] (consistent mass)."""
    n_elements = _CELLS[0]
    frequencies = []
    for order in range(1, _N_MODES + 1):
        phase = (2 * order - 1) * math.pi / (2 * n_elements)
        omega_squared = 6.0 * modulus / (_DENSITY * _CELL ** 2) * (1.0 - math.cos(phase)) / (2.0 + math.cos(phase))
        frequencies.append(math.sqrt(omega_squared) / (2.0 * math.pi))
    return frequencies


def _short_frequencies(stiffened_modulus, end_stiffness):
    """[Hz] of the lowest _N_MODES modes of the dense 1D pencil: c_D chain (node 0 held) minus end_stiffness [Pa/m] on
    the free end node, consistent mass (per unit cross-section)."""
    n_elements = _CELLS[0]
    stiffness = np.zeros((n_elements + 1, n_elements + 1))
    mass = np.zeros((n_elements + 1, n_elements + 1))
    for element in range(n_elements):
        pair = np.ix_((element, element + 1), (element, element + 1))
        stiffness[pair] += stiffened_modulus / _CELL * np.array([[1.0, -1.0], [-1.0, 1.0]])
        mass[pair] += _DENSITY * _CELL / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
    stiffness[-1, -1] -= end_stiffness
    stiffness, mass = stiffness[1:, 1:], mass[1:, 1:]
    inverse_factor = np.linalg.inv(np.linalg.cholesky(mass))
    omega_squared = np.sort(np.linalg.eigvalsh(inverse_factor @ stiffness @ inverse_factor.T))
    return [math.sqrt(value) / (2.0 * math.pi) for value in omega_squared[:_N_MODES]]


def _continuum_short_root(coupling_squared):
    """Lowest x = k L of tan x = x / K_t^2 (continuum fixed-free bar, shorted electrodes), by bisection."""
    low, high = 1e-9, 0.5 * math.pi - 1e-12
    for _ in range(200):
        middle = 0.5 * (low + high)
        if math.tan(middle) - middle / coupling_squared < 0.0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _check_modal(results_by_name):
    """Lowest frequencies of the short / open runs against the discrete references, snapshot counts and the open
    mode-1 potential phi = (e / eps) u; measured = error = the worst deviation in tolerance units."""
    c11, eps, piezo_e = _moduli()
    stiffened = c11 + piezo_e ** 2 / eps
    expected_runs = {'open': _chain_frequencies(stiffened),
                     'short': _short_frequencies(stiffened, piezo_e ** 2 / (eps * _LENGTH))}
    errors, tolerances, details = {}, {}, {}
    for run, expected in expected_runs.items():
        results = results_by_name[run]
        rows = results.result(_PHYSICS, _TABLE)['rows']
        if len(rows) < len(expected):
            raise ValueError(f'{run}: expected {len(expected)} modes in the table, got {len(rows)}')
        measured = [float(row[_FREQUENCY_COLUMN]) for row in rows[:len(expected)]]
        errors[f'{run} frequency'] = max(abs(value / reference - 1.0) for value, reference in zip(measured, expected))
        tolerances[f'{run} frequency'] = _FREQUENCY_TOLERANCE
        labels = results.summary().get('snapshot_labels') or []
        errors[f'{run} snapshots'] = abs(len(labels) - _N_MODES)
        tolerances[f'{run} snapshots'] = 0.5
        details[run] = {'frequency_expected': expected, 'frequency_measured': measured}
    open_run = results_by_name['open']
    potential = _field(open_run, 'electric_potential')
    errors['open mode 1 potential'] = (float(np.max(np.abs(potential - piezo_e / eps * _field(open_run, 'u_x'))))
                                       / float(np.max(np.abs(potential))))
    tolerances['open mode 1 potential'] = _FIELD_TOLERANCE
    # continuum context (not checked): k_eff^2 of mode 1 and the shorted root tan x = x / K_t^2
    short_1, open_1 = details['short']['frequency_measured'][0], details['open']['frequency_measured'][0]
    coupling_squared = piezo_e ** 2 / (eps * stiffened)
    details['continuum'] = {
        'K_t^2': coupling_squared, 'k_eff^2 mode 1 measured': (open_1 ** 2 - short_1 ** 2) / open_1 ** 2,
        'short mode 1': _continuum_short_root(coupling_squared) * math.sqrt(stiffened / _DENSITY)
                        / (2.0 * math.pi * _LENGTH),
        'open mode 1': math.sqrt(stiffened / _DENSITY) / (4.0 * _LENGTH)}
    return scaled_worst_result(errors, tolerances, **details)


_BAR_TEXT = (f'a {_LENGTH * 1e3:g} x {_WIDTH * 1e3:g} mm bar of uniform {_CELL * 1e3:g} mm cells ({_CELLS[0]} x '
             f'{_CELLS[1]}; E {_YOUNGS_MODULUS:g} Pa, nu {_POISSON_RATIO:g}, rho {_DENSITY:g} kg/m^3, piezo_e11 '
             f'{_PIEZO_E11:g} C/m^2, eps_dc {_EPS_DC:g}) with rollers on y_lo / y_hi (uniaxial strain, y-uniform fields: '
             'the 2D Q1 solution is the 1D linear-element one), a roller on x_lo and electrodes ground (x_lo, 0 V) and '
             'drive (x_hi)')

PIEZO_CASES = (
    SelfTestCase(
        name='solid_piezo_bar_static',
        label='Solid mechanics: piezoelectric bar under voltage and charge',
        description=f'stationary study of a piezoelectric solid_mechanics on {_BAR_TEXT}. Runs: free (drive at '
                    f'{_VOLTAGE:g} V): strain_xx = d E = e E / c11 (E = -V / L), u_x = strain x, stress_xx = 0, phi = V '
                    'x / L, Q_drive = -Q_ground = (eps + e^2 / c11) V W / L (free capacitance), anchor reaction 0; '
                    'blocked (also a roller on x_hi): strain 0, stress_xx = e V / L, Q_drive = eps V W / L (clamped '
                    f'capacitance), clamp / anchor reactions +-e V W / L; floating (drive floating with {_CHARGE:g} '
                    'C/m): V_drive = Q L / ((eps + e^2 / c11) W). Linear u and phi are exact in Q1: the float32 '
                    f'fields within {_FIELD_TOLERANCE:g} of their scale, the float64 electrode values and reactions '
                    f'within {_VALUE_TOLERANCE:g} (relative). measured = error = the worst deviation in tolerance '
                    'units, tolerance 1: this pins the coupling block and its sign, the -e^T E stress output, the '
                    'electrode reduction (held, floating with charge) and the coupled reactions.',
        specs=_static_specs,
        check=_check_static),
    SelfTestCase(
        name='solid_piezo_bar_modes',
        label='Solid mechanics: piezoelectric bar resonances, short and open circuit',
        description=f'eigenfrequency study ({_N_MODES} lowest modes) of {_BAR_TEXT}, free at x_hi. Runs: open (drive '
                    'floating: the discrete D vanishes): the discrete fixed-free chain omega^2 = 6 c_D / (rho h^2) (1 - '
                    'cos t) / (2 + cos t), t = (2 n - 1) pi / (2 N), at c_D = c11 + e^2 / eps; short (drive at 0 V: D = '
                    'e u_N / L): the c_D chain minus e^2 / (eps L) on the end node, solved densely (1D, N x N); a '
                    'piezo material with piezoelectric off is pinned by solid_eigenfrequency_roller_box. Frequencies '
                    'within '
                    f'{_FREQUENCY_TOLERANCE:g} (relative), one snapshot per mode, and the live open mode 1 has '
                    f'electric_potential = (e / eps) u_x within {_FIELD_TOLERANCE:g} of its peak. measured = error = '
                    'the worst deviation in tolerance units, tolerance 1: this pins the coupled pencil with its '
                    'massless potential rows, the short / open electrode rules and the mode potential; details give '
                    'the measured k_eff^2 of mode 1 and the continuum values. Not yet run without the unstiffened '
                    'run.',
        specs=_modal_specs,
        check=_check_modal),
)
