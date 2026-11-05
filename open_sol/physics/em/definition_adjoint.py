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
          help='Monitor quantity of this objective term, in W/m: flux = the time-averaged Poynting power through the '
               'line (as flux_monitor, > 0 toward +x / +y); mode_power_positive / mode_power_negative = |a|^2 of '
               'mode mode_number travelling toward +normal / -normal (as mode_monitor power_positive / '
               'power_negative). The objective J is the sum of weight * quantity over every enabled adjoint_objective '
               'of the physics.',
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
          help='Factor of this term in J = sum of weight * quantity: 1 to maximise the quantity, -1 to minimise it '
               '(the gradient points to increasing J), 0 to record the quantity without contributing.',
          tip='How much this measurement counts in the total objective: 1 to increase it, -1 to decrease it, 0 to only '
              'record it.'),
)
_DESCRIPTION = (
    'Objective term for the adjoint sensitivity: a flux_monitor or mode_monitor quantity on an axis-aligned line '
    'times weight. Every solve records the quantity; the output field eps_sensitivity is dJ/d eps_r per unit area '
    '[W/m^3] of J = sum of weight * quantity over the enabled objectives, from one extra sparse solve with the '
    'transposed operator on the forward factorisation (no new factorisation with scipy_splu), computed when the field '
    'is read (snapshot, derived value). A small permittivity change d eps in a region R changes J by the integral of '
    'eps_sensitivity * d eps over R; for em_fdfd_hz it is the sum of the eps_xx and eps_yy terms (an isotropic '
    'change, an off-diagonal eps_xy held fixed), and its output field eps_xy_sensitivity is dJ/d eps_xy per unit '
    'area from the same solve (zero where the solve drops eps_xy: PEC cells, the outer two rows of a wall or mirror '
    'face). For em_fdfd_hz the face fields an objective reads and the right-hand side of dipole and line source face '
    'currents are differentiated with the face permittivity; line source calibrations, port modes and a mode '
    'monitor\'s mode are held fixed, so the gradient is exact only in cells off every line source, port and mode '
    'monitor line and off the cells beyond their hi faces (they depend on the permittivity there too); sigma_e, '
    'poles and mu_r are not differentiated. An optimization study maximises J, summed over every physics, by varying '
    'the densities of the design_region features. Result \'xy\' versus vacuum wavelength: series value (the '
    'quantity, W/m) and weighted (weight * value, W/m); extras frequency, wavelength, quantity and weight. Keep the line off a TFSF '
    f'outline and a mode_source interface face, as for the monitors. The line covers {SPAN_CELLS}.')
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
