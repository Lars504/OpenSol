"""Self-test of the nonlinear -> heat solution feedback (nonlinear_optics_plan stage 5 (a)): two-photon absorption of
an em_fdfd_nonlinear strip heats a heat physics through em_heat_joule, feed-forward and with heat_em_thermo_optic back;
exact discrete identities, no physics imports."""
import math

import numpy as np

from ...core.constants import C0, EPS0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import float_list, with_exact_cg
from ...model.spec import (CouplingSpec, DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec,
                           MeshSpec, ModelSpec, PhysicsSpec, StudySpec)

# nlfd_tpa_heat_feedback: y-periodic vacuum strip (4 rows, CPML along x, uniform 10 nm mesh; window cell k spans
# [k h, (k + 1) h]), one ez pump channel p at 1.55 um from a line source in cell 100, a chi3 slab on cells [200, 2200)
# with chi3 = 1e-19 - 1e-22 j m^2/V^2 (SPM + TPA; the THG product falls on no channel), pump_depleted. Heat (k = 100
# W/(m K) everywhere, T0 walls on x_lo / x_hi, insulated y faces: 1D in x) heated by em_heat_joule. The source amplitude
# gives (3/4) Re chi3 |E|^2 = 1e-3, so the TPA density q = -(w / 2) Im(conj(E) P) = (3/8) w eps0 |Im chi3| |E|^4 ~ 7e13
# W/m^3 and the peak rise ~45 K. Runs: picard and newton feed-forward (dndT 0: one background solve after the
# nonlinear one) and 'two_way' (dndT 1e-4 /K through heat_em_thermo_optic: dn ~ 5e-3 changes |E|^2 ~ 1 / n, so the
# rounds converge geometrically). Exact checks per run: heat Q = q of the final nl e_abs per cell (the absorption
# density of absorption.py; two-way: up to the last round's change), T = the Dirichlet Green sum of that Q (3-point FV
# identity, exact CG), sum Q dA = the nl_nonlinear_absorption scalar; two-way also eps_r = (1 + dndT theta)^2 and a
# converged feedback record with >= 2 rounds; feed-forward Q of picard = newton (lagged Kerr loss replaced exactly).
# The feed-forward runs also carry a sigma3 Kerr sheet_nonlinearity on the line in tail cell 2300 (outside the slab):
# Q = (3/8) Re sigma3 |E|^4 / h in that column (the site work booked in the line cell); its q (~1e17 W/m^3) dwarfs the
# bulk one, so the Q items are normalised per region (bulk cells by the bulk peak, the line cell by its own value).
_PHYSICS = 'nl'
_HEAT = 'heat'
_CELL = 10e-9                       # m
_ROWS = 4
_SOURCE_CELL = 100
_SLAB_START, _SLAB_CELLS = 200, 2000
_TAIL_CELLS = 200                   # cells after the slab
_N_X = _SLAB_START + _SLAB_CELLS + _TAIL_CELLS
_WAVELENGTH = 1.55e-6               # m
_OMEGA = 2.0 * math.pi * C0 / _WAVELENGTH
_CHI3_REAL, _CHI3_IMAG = 1e-19, -1e-22   # m^2/V^2
_KERR_DELTA_EPS = 1e-3              # (3/4) chi3_real |E|^2
_CONDUCTIVITY = 100.0               # W/(m K)
_T0 = 293.15                        # K
_DNDT = 1e-4                        # 1/K, two-way run
_RUNS = {'picard': ('picard', 0.0), 'newton': ('newton', 0.0), 'two_way': ('picard', _DNDT)}
_SHEET_CELL = 2300                  # tail cell of the sigma3 Kerr sheet line (feed-forward runs only)
_SHEET_SIGMA3 = (2e-23, 1e-23)      # S m^2/V^2 (real: absorbs); (3/4) eta0 |sigma3| |E|^2 ~ 1e-4
_NL_TOLERANCE = 1e-7                # pump_depleted tolerance: Q ~ |E|^4 moves ~4e-7, vs _METHOD_TOLERANCE 1e-5
_STUDY = {'tolerance': 1e-6, 'max_iterations': 30, 'feedback_rounds': 15}
_Q_TOLERANCE = 1e-4                 # relative to the peak q: float32 e_abs^4 and Q
_PROFILE_TOLERANCE = 1e-4           # relative to the peak rise: exact FV identity in float32
_POWER_TOLERANCE = 1e-5             # relative, sum Q dA vs the scalar (float32 Q)
_EPS_TOLERANCE = 1e-5               # absolute eps_r (float32)
_METHOD_TOLERANCE = 1e-5            # relative per region, picard vs newton Q (iteration tolerances 1e-7)
_MAX_ROUNDS = 10


def _amplitude():
    return math.sqrt(_KERR_DELTA_EPS / (0.75 * _CHI3_REAL))


def _feedback_spec(run_name):
    """Strip spec of one run (iteration method and dndT from _RUNS; feed-forward runs add the sigma3 sheet line)."""
    method, dndT = _RUNS[run_name]
    size_y = _ROWS * _CELL
    source_x = (_SOURCE_CELL + 0.5) * _CELL
    nl_settings = {'bc_y_lo': 'periodic', 'bc_y_hi': 'periodic', 'coupling_mode': 'pump_depleted',
                   'iteration_method': method, 'tolerance': _NL_TOLERANCE, 'max_iterations': 50}
    sheet_x = (_SHEET_CELL + 0.5) * _CELL
    features = [
        FeatureSpec('p', 'harmonic_channel', [], {'definition': 'wavelength', 'wavelength': _WAVELENGTH,
                                                  'polarization': 'ez'}),
        FeatureSpec('kerr', 'chi3_medium', ['nl_slab'], {'tensor': 'isotropic', 'strength': 'chi3',
                                                         'chi3_real': _CHI3_REAL, 'chi3_imag': _CHI3_IMAG}),
        FeatureSpec('pump', 'line_source', ['source_line'], {'channel': 'p', 'amplitude': _amplitude(),
                                                             'phase': 0.0})]
    if not dndT:
        features.append(FeatureSpec('kerr_sheet', 'sheet_nonlinearity', ['sheet_line'], {
            'sheet_model': 'sigma3', 'sigma3_real': _SHEET_SIGMA3[0], 'sigma3_imag': _SHEET_SIGMA3[1]}))
    couplings = [CouplingSpec('tpa_heat', 'em_heat_joule', _PHYSICS, _HEAT)]
    if dndT:
        couplings.append(CouplingSpec('thermo_optic', 'heat_em_thermo_optic', _HEAT, _PHYSICS))
    spec = ModelSpec(
        title=f'Self-test: em_fdfd_nonlinear two-photon heating ({run_name})',
        domain=DomainSpec(size_x=_N_X * _CELL, size_y=size_y),
        geometry=[GeometrySpec('source_line', 'line', {'x0': source_x, 'y0': 0.0, 'x1': source_x, 'y1': size_y}),
                  GeometrySpec('nl_slab', 'rectangle', {'x0': _SLAB_START * _CELL, 'y0': 0.0,
                                                        'x1': (_SLAB_START + _SLAB_CELLS) * _CELL, 'y1': size_y}),
                  GeometrySpec('sheet_line', 'line', {'x0': sheet_x, 'y0': 0.0, 'x1': sheet_x, 'y1': size_y})],
        materials=[MaterialSpec('vacuum', properties={'eps_r': 1.0, 'k': _CONDUCTIVITY, 'dndT': dndT},
                                selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS, 'em_fdfd_nonlinear', settings=nl_settings, features=features),
                 PhysicsSpec(_HEAT, 'heat', settings={'T0': _T0}, features=[
                     FeatureSpec('walls', 'temperature', ['x_lo', 'x_hi'], {'temperature': _T0})])],
        couplings=couplings,
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type='nonlinear_frequency_domain', settings=dict(_STUDY)),
        derived_values=[DerivedValueSpec('P_tpa', f'{_PHYSICS}_nonlinear_absorption', [], 'mean')])
    return with_exact_cg(spec, [_HEAT])


def _feedback_specs():
    return {run_name: _feedback_spec(run_name) for run_name in _RUNS}


def _window(results, name):
    """float64 y-mean per window column of a field; ValueError unless it has the strip's window shape."""
    values = np.asarray(results.field(name).values, dtype=np.float64)
    if values.shape != (_N_X, _ROWS):
        raise ValueError(f'{name}: expected the window shape {(_N_X, _ROWS)}, got {values.shape}')
    return values.mean(axis=1)


def _dirichlet_profile(power_density):
    """theta(x_i) of k theta'' = -q with theta = 0 on both window faces: sum_j G(x_i, s_j) q_j h / k (as the coupling
    strip self-tests)."""
    length = _N_X * _CELL
    centres = (np.arange(_N_X) + 0.5) * _CELL
    x, s = centres[:, None], centres[None, :]
    green = np.where(s <= x, s * (length - x), x * (length - s)) / length
    return green @ power_density * _CELL / _CONDUCTIVITY


def _region_deviation(q_measured, q_reference, has_sheet):
    """(bulk, sheet) max |q_measured - q_reference| relative to q_reference's bulk peak and its sheet-cell value
    (without the sheet: every cell is bulk, sheet 0)."""
    deviation = np.abs(q_measured - q_reference)
    if not has_sheet:
        return float(np.max(deviation)) / float(np.max(q_reference)), 0.0
    bulk_scale = float(np.max(np.delete(q_reference, _SHEET_CELL)))
    return (float(np.max(np.delete(deviation, _SHEET_CELL))) / bulk_scale,
            float(deviation[_SHEET_CELL]) / float(q_reference[_SHEET_CELL]))


def _run_errors(results, run_name):
    """({item: normalised error}, details, Q) of one run."""
    q_heat = _window(results, f'{_HEAT}.Q')
    e_abs = _window(results, f'{_PHYSICS}.e_abs')
    has_sheet = not _RUNS[run_name][1]
    q_expected = 0.375 * _OMEGA * EPS0 * abs(_CHI3_IMAG) * e_abs ** 4
    q_expected[:_SLAB_START] = 0.0
    q_expected[_SLAB_START + _SLAB_CELLS:] = 0.0
    if has_sheet:   # site work (3/8) Re sigma3 |E|^4 L in the line cell (area L h)
        q_expected[_SHEET_CELL] = 0.375 * _SHEET_SIGMA3[0] * e_abs[_SHEET_CELL] ** 4 / _CELL
    q_bulk_error, q_sheet_error = _region_deviation(q_heat, q_expected, has_sheet)
    q_peak = float(np.max(q_expected[_SLAB_START:_SLAB_START + _SLAB_CELLS]))
    theta = _window(results, f'{_HEAT}.T') - _T0
    theta_closed = _dirichlet_profile(q_heat)
    theta_peak = float(np.max(theta_closed))
    power_heat = float(np.sum(q_heat)) * _CELL * _ROWS * _CELL
    power_scalar = float(results.summary()['derived']['P_tpa'])
    errors = {f'{run_name} Q = q(|E|) bulk': q_bulk_error / _Q_TOLERANCE,
              f'{run_name} T profile': float(np.max(np.abs(theta - theta_closed))) / theta_peak / _PROFILE_TOLERANCE,
              f'{run_name} sum Q dA': abs(power_heat / power_scalar - 1.0) / _POWER_TOLERANCE}
    if has_sheet:
        errors[f'{run_name} Q = q(|E|) sheet line'] = q_sheet_error / _Q_TOLERANCE
    feedback = results.summary().get('feedback') or {}
    rounds = int(feedback.get('rounds', 0))
    if _RUNS[run_name][1]:
        eps_r = _window(results, f'{_PHYSICS}.eps_r')
        errors[f'{run_name} eps = (1 + dndT theta)^2'] = float(np.max(np.abs(eps_r - (1.0 + _DNDT * theta) ** 2))) \
            / _EPS_TOLERANCE
        errors[f'{run_name} feedback converged in 2..{_MAX_ROUNDS} rounds'] = (
            0.0 if feedback.get('converged') and 2 <= rounds <= _MAX_ROUNDS else math.inf)
    else:
        errors[f'{run_name} one feedback round'] = 0.0 if feedback.get('converged') and rounds == 1 else math.inf
    sample = slice(0, _N_X, _N_X // 24)
    details = {f'{run_name}_q_peak_W_m3': q_peak, f'{run_name}_theta_peak_K': theta_peak,
               f'{run_name}_P_heat_W_m': power_heat, f'{run_name}_P_tpa_W_m': power_scalar,
               f'{run_name}_feedback': feedback, f'{run_name}_theta_K': float_list(theta[sample]),
               f'{run_name}_Q_W_m3': float_list(q_heat[sample])}
    if has_sheet:
        details[f'{run_name}_q_sheet_line_W_m3'] = float(q_expected[_SHEET_CELL])
    return errors, details, q_heat


def _check_feedback(results_by_name):
    """Per run the Q / T / power identities (+ eps and the rounds for two_way), and picard Q = newton Q; measured =
    error = the worst error in tolerance units, tolerance 1."""
    errors, details, q_by_run = {}, {}, {}
    for run_name in _RUNS:
        run_errors, run_details, q_by_run[run_name] = _run_errors(results_by_name[run_name], run_name)
        errors.update(run_errors)
        details.update(run_details)
    method_bulk, method_sheet = _region_deviation(q_by_run['newton'], q_by_run['picard'], True)
    errors['picard Q = newton Q bulk'] = method_bulk / _METHOD_TOLERANCE
    errors['picard Q = newton Q sheet line'] = method_sheet / _METHOD_TOLERANCE
    worst_item = max(errors, key=lambda item: math.inf if math.isnan(errors[item]) else errors[item])
    details.update(normalized_errors=errors, worst_item=worst_item)
    worst = errors[worst_item]
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': 1.0, 'details': details}


def _check_feedback_memory(_results):
    """CPU: the run item of the feedback monitor's copies (model.meshing._feedback_copy_count) = 2 physics x float32 x
    cells for two_way, absent for the feed-forward run; error = the number of failed checks."""
    from ...model.meshing import estimate_memory, estimate_mesh
    from ...model.schema import get_registry
    registry = get_registry()
    measured, violations = {}, []
    for run_name, expected_copies in (('two_way', 2), ('picard', 0)):
        spec = _feedback_spec(run_name)
        mesh = estimate_mesh(spec, {}, registry)
        estimate = estimate_memory(spec, {}, registry, mesh)
        sizes = [size for label, size in estimate.run_items if 'solution-feedback field copies' in label]
        expected = 4 * expected_copies * int(mesh.n_cells)
        measured[run_name] = {'item_bytes': sizes, 'expected_bytes': expected}
        if sum(sizes) != expected or len(sizes) > 1:
            violations.append(f'{run_name}: feedback copy item(s) {sizes} B, expected {expected} B')
    return {'measured': len(violations), 'expected': 0, 'error': len(violations), 'tolerance': 0,
            'details': {'runs': measured, 'violations': violations}}


FEEDBACK_SELFTESTS = (
    SelfTestCase(
        name='nlfd_feedback_memory_estimate',
        label='NLFD solution feedback: memory estimate of the feedback copies (CPU)',
        description='CPU check of the memory estimate (no run) of the nlfd_tpa_heat_feedback specs: the two_way run (a '
                    'coupling loops back into the nonlinear physics) carries the run item of the feedback monitor\'s '
                    'primary-field copies, 2 physics x float32 x the grid cells; the feed-forward picard run (nothing '
                    'loops back: no monitor) has none. error = the number of failed checks, tolerance 0.',
        specs=lambda: {},
        check=_check_feedback_memory),
    SelfTestCase(
        name='nlfd_tpa_heat_feedback',
        label='NLFD two-photon heating: solution feedback into heat',
        description='y-periodic vacuum strip (4 rows, CPML along x, uniform 10 nm mesh), one ez pump channel at 1.55 '
                    'um, a 20 um chi3 slab (chi3 = 1e-19 - 1e-22 j m^2/V^2: SPM and two-photon absorption, '
                    '(3/4) Re chi3 |E|^2 = 1e-3), coupling_mode pump_depleted, and a heat physics (k = 100 W/(m K), '
                    'T0 walls on the x faces, exact CG) heated by em_heat_joule from the nonlinear physics in a '
                    'nonlinear_frequency_domain study (solution feedback). Runs picard and newton without '
                    'thermo-optic feedback (one background solve after the nonlinear solve) and two_way with '
                    f'heat_em_thermo_optic (dndT 1e-4 /K) back into the nonlinear physics (feedback rounds to the '
                    f'tolerance {_STUDY["tolerance"]:g}, nonlinear tolerance {_NL_TOLERANCE:g}). The picard and newton '
                    f'runs also carry a sigma3 Kerr sheet_nonlinearity ({_SHEET_SIGMA3[0]:g} + {_SHEET_SIGMA3[1]:g} j '
                    f'S m^2/V^2) on the vertical line in tail cell {_SHEET_CELL} (after the slab). Checks per run, '
                    'each an exact discrete identity: heat Q = (3/8) omega eps0 |Im chi3| |E|^4 per cell from the '
                    'final nl e_abs (the absorption density that replaces the lagged Kerr loss of the channel q_abs; '
                    '1e-4 of the bulk peak) and, in the sheet line cells, Q = (3/8) Re sigma3 |E|^4 / h (sheet SPM '
                    'absorption; 1e-4 of its own value), T - T0 = the Dirichlet Green sum of that Q (3-point FV, 1e-4 '
                    'of the peak rise), sum Q dA = the nl_nonlinear_absorption scalar (1e-5); two_way also eps_r = '
                    '(1 + dndT theta)^2 (1e-5) and a converged feedback record with 2 to 10 rounds; the picard and '
                    'newton Q agree to 1e-5 of the bulk peak and of the sheet-line value. measured = error = the '
                    'worst error in tolerance units, tolerance 1. Sheet merged into the feed-forward runs and '
                    'tolerances 1e-7 / 1e-6: not yet confirmed by a run.',
        specs=_feedback_specs,
        check=_check_feedback),
)
