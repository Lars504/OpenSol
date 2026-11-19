"""Model definition of the FDFD 'adjoint_objective' feature (em_fdfd_ez / em_fdfd_hz): one weighted flux or mode-power
term of the objective J whose gradient the output field eps_sensitivity shows (adjoint.py)."""
from dataclasses import replace

from ...model.schema import FeatureType, Issue, Param, feature_path
from .adjoint import OBJECTIVE_QUANTITIES, objective_result
from .definition_common import MODE_NUMBER_TIP_START, SPAN_CELLS, mode_line, mode_number_param, object_result
from .definition_design import OPTIMIZATION_STUDY
from .definition_feature_validation import mode_line_issues, monitor_line_interface_issues, validate_line_feature
from .memory_estimates import adjoint_objective_memory_estimate

_MODE_QUANTITIES = tuple(quantity for quantity in OBJECTIVE_QUANTITIES if quantity != 'flux')
_PARAMS = (
    Param('quantity', 'choice', 'flux', label='Quantity', choices=OBJECTIVE_QUANTITIES,
          help='W/m: flux = time-averaged Poynting power through the line (as flux_monitor, > 0 toward +x / +y); '
               'mode_power_positive / _negative = |a|^2 of mode mode_number toward +normal / -normal (as mode_monitor '
               'power_positive / _negative).',
          tip='What this objective measures on its line.\n\n'
              '- Power through the line: all light crossing it, positive toward +x or +y.\n'
              '- Mode power forward or backward: the power of one guided mode travelling toward +x / +y or the other '
              'way.',
          choice_labels=(('flux', 'Power through the line'), ('mode_power_positive', 'Mode power, forward'),
                         ('mode_power_negative', 'Mode power, backward'))),
    replace(mode_number_param('Used by the mode_power quantities only; picked at the first solved wavelength and '
                              'followed by overlap, as mode_monitor.',
                              tip=MODE_NUMBER_TIP_START.format(verb='measure')),
            visible_if=(('quantity', _MODE_QUANTITIES),)),
    Param('weight', 'float', 1.0, label='Weight', unit='1',
          help='Factor in J = sum of weight * quantity over the enabled objectives of the physics: 1 maximises the '
               'quantity, -1 minimises it, 0 only records it.',
          tip='How much this measurement counts in the total objective: 1 to increase it, -1 to decrease it, 0 to only '
              'record it.'),
)
_DESCRIPTION = (
    'Objective term for the adjoint sensitivity: a flux_monitor or mode_monitor quantity on an axis-aligned line '
    'times weight. Output field eps_sensitivity = dJ/d eps_r per unit area [W/m^3] from one extra transposed solve on '
    'the forward factorisation, computed when read (snapshot, derived value): d eps over a region R changes J by the '
    'integral of eps_sensitivity d eps over R. em_fdfd_hz: the sum of the eps_xx and eps_yy terms (eps_xy fixed), plus '
    'eps_xy_sensitivity = dJ/d eps_xy per unit area (0 where the solve drops eps_xy: PEC cells, the outer two rows of '
    'a wall or mirror face). Line source calibrations and port / monitor modes are held fixed, so the gradient is '
    'exact only off every line source, port and mode monitor line and the cells past their hi faces; sigma_e, poles '
    'and mu_r are not differentiated. An optimization study maximises J, summed over every physics, over the '
    "design_region densities. Result 'xy' versus vacuum wavelength: series value (the quantity, W/m) and weighted "
    '(weight * value); extras frequency, wavelength, quantity, weight. Keep the line off a TFSF outline and a '
    f'mode_source interface face. The line covers {SPAN_CELLS}.')
_SUMMARY = ('Marks a power measurement on a line as a goal, so the solver also maps where a small change of the '
            'material would raise or lower it (the objective sensitivity field). Use it to guide device design.')


def _undesigned_objective_issues(vctx, physics_spec, feature_spec):
    """Optimization study: error when no enabled design_region of an em_fdfd type of this physics' polarisation
    exists, since no design then writes the permittivity it reads (ez eps_r, hz eps_xx / eps_yy): a constant J term."""
    if vctx.study_type != OPTIMIZATION_STUDY:
        return []
    if any(feature.enabled and feature.type == 'design_region'
           for physics in vctx.spec.physics if physics.type == physics_spec.type for feature in physics.features):
        return []
    return [Issue('error', feature_path(physics_spec, feature_spec),
                  f'no enabled design_region on an {physics_spec.type} physics writes the permittivity this objective '
                  'reads (em_fdfd_ez regions write eps_r, em_fdfd_hz regions eps_xx / eps_yy), so it adds a constant '
                  f'to J and no gradient; add a design_region to an {physics_spec.type} physics or disable this '
                  'objective')]


def _make_validate(polarization):
    """validate callback: line checks; mode quantities add the mode_monitor line checks; an optimization study needs
    a design_region of this polarisation."""
    def validate_objective(vctx, physics_spec, feature_spec, values):
        if values.get('quantity') in _MODE_QUANTITIES:
            issues = mode_line_issues(vctx, physics_spec, feature_spec, polarization, 'monitor', 'hi')
        else:
            issues = validate_line_feature(vctx, physics_spec, feature_spec, values)
        return (issues + monitor_line_interface_issues(vctx, physics_spec, feature_spec, time_domain=False)
                + _undesigned_objective_issues(vctx, physics_spec, feature_spec))
    return validate_objective


def _apply_objective(ctx, solver, feature_spec, values, selection):
    axis, position, span = mode_line(ctx, selection, feature_spec)
    return solver.add_objective(axis, position, span, quantity=values['quantity'],
                                mode_number=values['mode_number'], weight=values['weight'])


def adjoint_objective_feature(polarization):
    """FeatureType 'adjoint_objective' of em_fdfd_<polarization>."""
    return FeatureType('adjoint_objective', 'Optimization objective', 'monitor', 'line', params=_PARAMS,
                       object_kinds=('line',), max_entities=1, result_kind='xy', description=_DESCRIPTION,
                       summary=_SUMMARY, apply=_apply_objective, result=object_result(objective_result),
                       validate=_make_validate(polarization),
                       memory_estimate=adjoint_objective_memory_estimate(polarization))
