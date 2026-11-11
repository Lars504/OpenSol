"""E13 self-test: two 'index' contributors (heat_em_thermo_optic + electrostatics_em_pockels) on one em_fdfd_hz of the
Pockels strip, superposed dn in eps_xx / eps_yy and the phase, in one case with the Pockels-alone check (shared
reference run). Run only inside pockels_cpu_gpu (selftests_backend; examples/multiphysics analytic_pockels_phase_vpi and
the index-sum examples check its parts); no physics imports."""
import math

import numpy as np

from ..model.schema import SelfTestCase
from ..model.selftest_helpers import with_exact_cg
from ..model.spec import CouplingSpec, DomainSpec, FeatureSpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec, \
    StudySpec
from .selftests import (_CELL, _CONDUCTIVITY, _CURRENTS_LINE_CELLS, _EM, _HEAT, _N_CELLS, _SIZE, _SLAB_X_RANGE, _T0,
                        _em_physics, _full_period_rectangle, _line_geometry, _monitor_phase_shift, _monitor_phases,
                        _strip_outcome, _summaries, _unconverged_failures, _window_field)
from .selftests_pockels import (_BIASED, _CRYSTAL, _CRYSTAL_CELLS, _ELECTROSTATICS, _FIELD_Y, _INDEX, _INDEX_CHANGE,
                                _R13, _R33, _REFERENCE, _VOLTAGE, POCKELS_DESCRIPTION, POCKELS_TYPE_NAME,
                                _check_pockels_strip, _pockels_strip_specs)

_COMBINED = 'combined'
# pockels_phase_index_sum combined run: the biased Pockels strip (n0 = 2.138 everywhere, crystal x = 2..6 um, 10 V
# across y: E_y = -5e7 V/m, dn_yy = +7.57e-3, dn_xx = +2.44e-3) plus a heat physics with both x walls at T0 + 20 K and
# no source: T = T0 + 20 K exactly in every cell (float64 CG), so dndT = 3e-4 /K gives a uniform dn_th = +6e-3,
# comparable to the Pockels shift. Both couplings write eps_xx_update / eps_yy_update as 'index' contributors. Its
# phase reference is the Pockels reference run (r = 0, no heat): the same EM problem, eps n0^2 through the same index
# square, so the two references would differ at most by float32 rounding (~1e-7 rad against the 1 % tolerance).
_TEMPERATURE_RISE = 20.0               # K, wall temperature above T0
_DNDT = 3e-4                           # 1/K, every cell (combined run)
_THERMAL_INDEX_CHANGE = _DNDT * _TEMPERATURE_RISE   # 6e-3
_EXACT_TOLERANCE = 1e-5                # theta / E_y: float32 storage of exact discrete solutions
_EPS_TOLERANCE = 1e-3                  # relative to the combined eps change of the crystal
_PHASE_TOLERANCE = 0.01                # relative to the predicted shift
_MAX_TICKS = 1                         # feed-forward: heat, electrostatics -> EM, one pass


# ---- specs ----
def _index_sum_spec():
    """em_fdfd_hz + electrostatics + heat strip with both index couplings on 'em': r_xx / r_yy of the crystal and dndT
    of every cell set."""
    slab_x0, slab_x1 = _SLAB_X_RANGE
    cladding = {'eps_r': _INDEX ** 2, 'eps_dc': 3.9, 'k': _CONDUCTIVITY, 'dndT': _DNDT}
    crystal = {**cladding, 'eps_dc': 43.0, 'eps_dc_yy': 28.0, 'r_yy': _R33, 'r_xx': _R13}
    wall_temperature = _T0 + _TEMPERATURE_RISE
    return with_exact_cg(ModelSpec(
        title=f'Self-test: thermo-optic + Pockels index sum ({_COMBINED})',
        domain=DomainSpec(size_x=_SIZE[0], size_y=_SIZE[1]),
        geometry=[_full_period_rectangle(_CRYSTAL, slab_x0, slab_x1), *_line_geometry(_CURRENTS_LINE_CELLS)],
        materials=[MaterialSpec('cladding', properties=cladding, selection=['all']),
                   MaterialSpec(_CRYSTAL, properties=crystal, selection=[_CRYSTAL])],
        physics=[_em_physics(_CURRENTS_LINE_CELLS, 'em_fdfd_hz'),
                 PhysicsSpec(_ELECTROSTATICS, _ELECTROSTATICS, features=[
                     FeatureSpec('top', 'voltage', ['y_hi'], {'voltage': _VOLTAGE}),
                     FeatureSpec('bottom', 'voltage', ['y_lo'], {'voltage': 0.0})]),
                 PhysicsSpec(_HEAT, 'heat', settings={'T0': _T0}, features=[
                     FeatureSpec('walls', 'temperature', ['x_lo', 'x_hi'], {'temperature': wall_temperature})])],
        couplings=[CouplingSpec('thermo_optic', 'heat_em_thermo_optic', _HEAT, _EM),
                   CouplingSpec('pockels', POCKELS_TYPE_NAME, _ELECTROSTATICS, _EM, params={'field_component': 'y'})],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type='stationary')))


def _pockels_index_sum_specs():
    """Pockels 'reference' (r = 0) and 'biased', then 'combined' (thermal + Pockels dn on both permittivity slots)."""
    return {**_pockels_strip_specs(), _COMBINED: _index_sum_spec()}


# ---- check (CPU, on Results-like objects) ----
def _combined_index(component):
    """(nx,) closed-form index n0 + dn_th + dn_Pockels_<component> (crystal cells only) of the combined run."""
    index = np.full(_N_CELLS[0], _INDEX + _THERMAL_INDEX_CHANGE)
    index[_CRYSTAL_CELLS] += _INDEX_CHANGE[component]
    return index


def _check_index_sum(results_by_name):
    """Uniform theta and E_y, eps_ii = (n0 + dn_th + dn_ii)^2 and the phase shift combined vs reference against the
    discrete-k prediction; error = max relative error in tolerance units + number of failed checks."""
    summaries = _summaries(results_by_name)
    combined = results_by_name[_COMBINED]

    # 1. the two sources: theta = rise and E_y = -V / d in every cell
    theta = _window_field(combined, f'{_HEAT}.T') - _T0
    err_theta = float(np.max(np.abs(theta - _TEMPERATURE_RISE))) / _TEMPERATURE_RISE
    field_y = _window_field(combined, f'{_ELECTROSTATICS}.E_y')
    err_field = float(np.max(np.abs(field_y - _FIELD_Y))) / abs(_FIELD_Y)

    # 2. the summed index slot: eps = (n0 + dn_th + dn_Pockels)^2 per component
    eps_errors = {}
    for component in ('xx', 'yy'):
        expected = _combined_index(component) ** 2
        change = (_INDEX + _THERMAL_INDEX_CHANGE + _INDEX_CHANGE[component]) ** 2 - _INDEX ** 2
        measured = _window_field(combined, f'{_EM}.eps_{component}')
        eps_errors[component] = float(np.max(np.abs(measured - expected[:, None]))) / change
    err_eps = max(eps_errors.values())

    # 3. phase between the mode monitors, combined minus reference (E_y tangential: eps_yy sets k)
    phases, _, dphi_measured = _monitor_phases(results_by_name, _COMBINED)
    dphi_predicted = _monitor_phase_shift(_combined_index('yy'), _INDEX)
    err_phase = abs(dphi_measured - dphi_predicted) / abs(dphi_predicted)

    # 4. feed-forward: one pass, no 'did not converge' warning
    failed = _unconverged_failures(summaries)
    for run_name, summary in summaries.items():
        if int(summary['ticks']) > _MAX_TICKS:
            failed.append(f"{run_name}: {int(summary['ticks'])} iterations > {_MAX_TICKS}")

    normalized_errors = {'theta': err_theta / _EXACT_TOLERANCE, 'field': err_field / _EXACT_TOLERANCE,
                         'eps': err_eps / _EPS_TOLERANCE, 'phase': err_phase / _PHASE_TOLERANCE}
    details = {'err_theta': err_theta, 'err_field': err_field, 'err_eps': eps_errors, 'err_phase': err_phase,
               'dn_thermal': _THERMAL_INDEX_CHANGE, 'dn_pockels': dict(_INDEX_CHANGE),
               'dphi_measured_rad': dphi_measured, 'dphi_predicted_rad': dphi_predicted,
               'phase_positive_rad': phases}
    return _strip_outcome(normalized_errors, failed, summaries, dphi_measured, dphi_predicted, details)


def _check_pockels_index_sum(results_by_name):
    """The Pockels-alone check (reference, biased) and the index-sum check (reference, combined); error = the worse
    (tolerance 1 each), both dphi compared cpu vs gpu by the backend wrapper (details 'parity_measured')."""
    reference = results_by_name[_REFERENCE]
    pockels = _check_pockels_strip({_REFERENCE: reference, _BIASED: results_by_name[_BIASED]})
    index_sum = _check_index_sum({_REFERENCE: reference, _COMBINED: results_by_name[_COMBINED]})
    errors = [float(pockels['error']), float(index_sum['error'])]
    error = math.nan if any(math.isnan(value) for value in errors) else max(errors)
    details = {'pockels': {'measured': pockels['measured'], 'expected': pockels['expected'],
                           'error': pockels['error'], **pockels['details']},
               'index_sum': {'measured': index_sum['measured'], 'expected': index_sum['expected'],
                             'error': index_sum['error'], **index_sum['details']},
               'parity_measured': {'dphi_biased': pockels['measured'], 'dphi_combined': index_sum['measured']}}
    return {'measured': index_sum['measured'], 'expected': index_sum['expected'], 'error': error, 'tolerance': 1.0,
            'details': details}


POCKELS_INDEX_SUM_CASE = SelfTestCase(
    name='pockels_phase_index_sum',
    label='Electrostatics (+ heat) + EM (H_z): Pockels phase, and thermo-optic + Pockels index sum on one slot '
          '(stationary)',
    description=POCKELS_DESCRIPTION + (
        ' Index sum (run combined, phase against the same reference): E13 index contributors on the same strip plus a '
        'heat physics (float64 CG to 1e-12, k = 0.2 W/(m K)) with both x walls at T0 + 20 K and no source, so theta = '
        '20 K exactly in every cell; dndT = 3e-4 /K everywhere gives dn_th = +6e-3, comparable to dn_yy = +7.57e-3 '
        'and dn_xx = +2.44e-3 of the Pockels coupling (field_component y). heat_em_thermo_optic and '
        'electrostatics_em_pockels both write eps_xx_update and eps_yy_update of em, summed by one SlotSum per slot. '
        'The reference has no heat physics: the same EM problem with eps n0^2 through the same index square, so a '
        'reference with a dndT = 0 heat would differ by float32 rounding only (~1e-7 rad). Checks: (1) err_theta = '
        'max |theta - 20 K| / 20 K and err_field = max |E_y + V / d| / (V / d), tolerance 1e-5 each (exact discrete '
        'solutions stored as float32). (2) err_eps = max over xx, yy of max |eps - (n0 + dn_th + dn_ii)^2| / ((n0 + '
        'dn_th + dn_ii)^2 - n0^2) with dn_ii in the crystal only, tolerance 1e-3: the sum of the two contributions '
        'inside one square, not a sum of two eps changes. (3) dphi = wrap([phi_down - phi_up]_combined - [phi_down - '
        'phi_up]_reference) against -sum [k_num(n_yy(x)) - k_num(n0)] dx (trapezoid over the cell centres between the '
        'mode monitor line cells), ~ -k0 (dn_th 5 um + dn_yy 4 um) = -0.24 rad, tolerance 1 % (the source and '
        'monitors re-solve in n0 + dn_th, the slab ends reflect ~(dn / 2n)^2 ~ 3e-6). (4) Feed-forward: 1 iteration, '
        'no \'did not converge\' warning; each violation is a failed check. Its error = max(err_theta / 1e-5, '
        'err_field / 1e-5, err_eps / 1e-3, err_phase / 0.01) + the number of failed checks, tolerance 1. Case error '
        '= the worse of the two, tolerance 1; measured / expected = the combined measured / predicted dphi in rad '
        '(the biased one in details; both are compared in pockels_cpu_gpu). Not yet run as one three-run case.'),
    specs=_pockels_index_sum_specs,
    check=_check_pockels_index_sum)
