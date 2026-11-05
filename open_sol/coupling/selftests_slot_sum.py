"""E13 self-test: two additive heat sources (em_heat_joule + currents_heat_joule) on one heat physics of the glass
strip of coupling/selftests.py; the summed slot and the superposition of the linear heat problem. No physics imports."""
import numpy as np

from ..model.schema import SelfTestCase
from ..model.selftest_helpers import max_relative_error, with_exact_cg
from ..model.spec import CouplingSpec, DerivedValueSpec, DomainSpec, FeatureSpec, MaterialSpec, MeshSpec, ModelSpec, \
    PhysicsSpec, StudySpec
from .selftests import (_CELL, _CONDUCTIVITY, _CURRENTS, _EM, _GLASS_EPS_R, _HEAT,
                        _LINE_CELLS, _RESISTOR_SIGMA, _SIZE, _SLAB, _SLAB_SIGMA, _SLAB_X_RANGE, _SOURCE, _STUDY,
                        _TERMINAL_A, _TERMINAL_B, _TERMINAL_LENGTH, _TERMINAL_VOLTAGE, _cell_line, _em_physics,
                        _full_period_rectangle, _heat_physics, _line_name, _summaries, _unconverged_failures,
                        _window_field)

_OPTICAL = 'optical'
_RESISTIVE = 'resistive'
_COMBINED = 'combined'
# Feed-forward chains (EM -> heat, currents -> heat, no index feedback): every physics is a sink, one pass each.
_MAX_TICKS = 1
_SUPERPOSITION_TOLERANCE = 1e-3   # relative to the peak combined rise (roadmap 10.1)
_SLOT_SUM_TOLERANCE = 1e-5        # relative to the peak source: float32 buffers summed on the GPU vs float64 here
_MIN_SHARE = 0.1                  # each source's peak rise relative to the combined one (else a trivial sum)


def _superposition_spec(run_name, optical, resistive):
    """Glass strip with the slab x = 2..6 um: optical adds em_fdfd_ez + mode source + sigma_e + em_heat_joule,
    resistive adds electric_currents + terminals + sigma_dc + currents_heat_joule; heat (T0 walls) always."""
    slab_x0, slab_x1 = _SLAB_X_RANGE
    glass, slab = {'k': _CONDUCTIVITY}, {'k': _CONDUCTIVITY}
    geometry = [_full_period_rectangle(_SLAB, slab_x0, slab_x1)]
    physics, couplings = [_heat_physics()], []
    if optical:
        glass['eps_r'] = slab['eps_r'] = _GLASS_EPS_R
        slab['sigma_e'] = _SLAB_SIGMA
        geometry.append(_cell_line(_line_name(_SOURCE), _LINE_CELLS[_SOURCE]))
        physics.insert(0, _em_physics({_SOURCE: _LINE_CELLS[_SOURCE]}))
        couplings.append(CouplingSpec('optical_joule', 'em_heat_joule', _EM, _HEAT))
    if resistive:
        slab['sigma_dc'] = _RESISTOR_SIGMA
        geometry += [_full_period_rectangle(_TERMINAL_A, slab_x0, slab_x0 + _TERMINAL_LENGTH),
                     _full_period_rectangle(_TERMINAL_B, slab_x1 - _TERMINAL_LENGTH, slab_x1)]
        physics.append(PhysicsSpec(_CURRENTS, 'electric_currents',
                                   features=[FeatureSpec(_TERMINAL_A, 'terminal', [_TERMINAL_A],
                                                         {'voltage': _TERMINAL_VOLTAGE}),
                                             FeatureSpec(_TERMINAL_B, 'terminal', [_TERMINAL_B], {'voltage': 0.0})]))
        couplings.append(CouplingSpec('resistive_joule', 'currents_heat_joule', _CURRENTS, _HEAT))
    return with_exact_cg(ModelSpec(
        title=f'Self-test: two heat sources on one heat physics ({run_name})',
        domain=DomainSpec(size_x=_SIZE[0], size_y=_SIZE[1]),
        geometry=geometry,
        materials=[MaterialSpec('glass', properties=glass, selection=['all']),
                   MaterialSpec('slab', properties=slab, selection=[_SLAB])],
        physics=physics,
        couplings=couplings,
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type='stationary', settings=dict(_STUDY)),
        derived_values=[DerivedValueSpec('T_max', 'T', [], 'max')]), [_CURRENTS] if resistive else [])


def _superposition_specs():
    """'optical' (absorbed light only), 'resistive' (Joule current only), 'combined' (both into one volumetric_power)."""
    return {_OPTICAL: _superposition_spec(_OPTICAL, True, False),
            _RESISTIVE: _superposition_spec(_RESISTIVE, False, True),
            _COMBINED: _superposition_spec(_COMBINED, True, True)}


def _check_superposition(results_by_name):
    """Superposition of the rises, the summed Q of the combined run and the lone-contributor Q of the single runs;
    error = max relative error in tolerance units + number of failed checks."""
    summaries = _summaries(results_by_name)
    rises = {run_name: _window_field(results, f'{_HEAT}.T_rise') for run_name, results in results_by_name.items()}
    optical_power = {run_name: _window_field(results_by_name[run_name], f'{_EM}.absorbed_power')
                     for run_name in (_OPTICAL, _COMBINED)}
    resistive_power = {run_name: _window_field(results_by_name[run_name], f'{_CURRENTS}.q_joule')
                       for run_name in (_RESISTIVE, _COMBINED)}
    heat_sources = {run_name: _window_field(results, f'{_HEAT}.Q') for run_name, results in results_by_name.items()}

    # 1. linear heat problem: the combined rise is the sum of the single-source rises
    err_superposition = max_relative_error(rises[_COMBINED], rises[_OPTICAL] + rises[_RESISTIVE])
    # 2. the slot sum itself: Q = absorbed_power + q_joule in the combined run, Q = its one source in the others
    err_slot_sum = max_relative_error(heat_sources[_COMBINED],
                                            optical_power[_COMBINED] + resistive_power[_COMBINED])
    err_single = max(max_relative_error(heat_sources[_OPTICAL], optical_power[_OPTICAL]),
                     max_relative_error(heat_sources[_RESISTIVE], resistive_power[_RESISTIVE]))

    peaks = {run_name: float(np.max(rise)) for run_name, rise in rises.items()}
    failed = _unconverged_failures(summaries)
    for run_name, summary in summaries.items():
        if int(summary['ticks']) > _MAX_TICKS:
            failed.append(f"{run_name}: {int(summary['ticks'])} iterations > {_MAX_TICKS} (feed-forward chain)")
    for run_name in (_OPTICAL, _RESISTIVE):
        if not peaks[run_name] >= _MIN_SHARE * peaks[_COMBINED]:   # NaN counts as failed
            failed.append(f'{run_name}: peak rise {peaks[run_name]:.3g} K below {_MIN_SHARE:g} of the combined '
                          f'{peaks[_COMBINED]:.3g} K')
    normalized_errors = {'superposition': err_superposition / _SUPERPOSITION_TOLERANCE,
                         'slot_sum': err_slot_sum / _SLOT_SUM_TOLERANCE, 'single': err_single / _SLOT_SUM_TOLERANCE}
    error = float(np.max(np.asarray(list(normalized_errors.values())))) + len(failed)   # NaN propagates
    details = {'err_superposition': err_superposition, 'err_slot_sum': err_slot_sum, 'err_single': err_single,
               'normalized_errors': normalized_errors, 'theta_max_K': peaks, 'failed_checks': failed,
               'ticks': {run_name: int(summary['ticks']) for run_name, summary in summaries.items()},
               'warnings': {run_name: list(summary['warnings']) for run_name, summary in summaries.items()}}
    return {'measured': peaks[_COMBINED], 'expected': peaks[_OPTICAL] + peaks[_RESISTIVE], 'error': error,
            'tolerance': 1.0, 'details': details}


SUPERPOSITION_CASE = SelfTestCase(
    name='heat_sources_superposition_strip',
    label='Two heat sources on one heat physics: summed Joule slot and superposition (stationary)',
    description='E13 additive source slots on the glass strip of em_heat_thermo_optic_strip (8.0 x 0.2 um, '
                'y-periodic, 20 nm mesh, k = 0.2 W/(m K) everywhere, heat walls at T0 = 293.15 K on x_lo / x_hi, '
                'insulated y faces). The slab x = 2..6 um carries sigma_e = 2.0e3 S/m (absorbing 86 % of the 1 W/m '
                'em_fdfd_ez mode launched at x = 1.01 um, n = 1.5, 1.55 um) and sigma_dc = 15 S/m between the '
                'terminals x = 2.0..2.1 um (1 V) and 5.9..6.0 um (0 V) of an electric_currents physics. Three '
                'stationary runs: optical (EM + heat + em_heat_joule), resistive (currents + heat + '
                'currents_heat_joule) and combined (both couplings into the one volumetric_power slot, summed by a '
                'SlotSum). dndT = 0, so every chain is feed-forward and the heat problem is linear in Q. Checks: (1) '
                'err_superposition = max |theta_combined - theta_optical - theta_resistive| / max theta_combined '
                'with theta = heat T_rise per cell, tolerance 1e-3 (the FV operator is identical in all runs; the '
                'float32 T_rise and the CG tolerance 1e-8 leave ~1e-5). (2) err_slot_sum = max |Q - absorbed_power - '
                'q_joule| / max |absorbed_power + q_joule| of the combined run and err_single = the same with the '
                'one source of each single run, tolerance 1e-5 (float32 sum of the contributor buffers). Failed '
                'checks: more than 1 iteration in a run (feed-forward: every physics a sink), a \'did not '
                'converge\' warning, or a single-source peak rise below 0.1 of the combined one (each rise is ~30 '
                'K). error = max(err_superposition / 1e-3, err_slot_sum / 1e-5, err_single / 1e-5) + the number of '
                'failed checks, tolerance 1; measured / expected = the combined peak rise / the sum of the single '
                'peaks in K (equal only when the peaks coincide, informative).',
    specs=_superposition_specs,
    check=_check_superposition)

SLOT_SUM_CASES = (SUPERPOSITION_CASE,)
