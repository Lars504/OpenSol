"""Model definition of the FDFD 'design_region' feature (em_fdfd_ez / em_fdfd_hz): the density-parametrised permittivity
an optimization study (model.study_optimization) varies; design_density.py holds the map and its chain rule."""
import numpy as np

from ...model.geometry import domain_object_rasterizer, expand_entities, preview_pixel_centers
from ...model.schema import FeatureType, Issue, Param, feature_path
from .design_density import design_region_result
from .definition_common import object_result
from .definition_feature_validation import _line_axes, _selected_shapes, preview_pixel
from .memory_estimates import design_region_memory_estimate

OPTIMIZATION_STUDY = 'optimization'   # = model.study_optimization.STUDY_NAME
# line features whose cells the adjoint gradient holds fixed (sources, ports, objectives)
_FIXED_LINE_FEATURES = ('line_source', 'gaussian_beam', 'mode_source', 'mode_monitor', 'adjoint_objective')

_PARAMS = (
    Param('eps_min', 'float', 2.085, label='Permittivity at density 0', unit='1', min=1.0,
          help='Relative permittivity of a design cell at density 0 (the void or cladding material, e.g. silica 2.085 '
               'at 1.55 um); a cell holds eps = eps_min + rho_bar (eps_max - eps_min) of its filtered, projected '
               'density rho_bar. em_fdfd_ez writes eps_r, em_fdfd_hz eps_xx and eps_yy.',
          tip='Permittivity of the design area where the design is empty, usually the cladding: 2.085 for glass '
              'at 1550 nm.'),
    Param('eps_max', 'float', 12.08, label='Permittivity at density 1', unit='1', min=1.0,
          help='Relative permittivity of a design cell at density 1 (the solid material, e.g. silicon 12.08 at 1.55 '
               'um). sigma_e, poles and mu_r of the design cells stay those of the material painted there and are not '
               'optimised: paint a lossless, non-dispersive material.',
          tip='Permittivity of the design area where the design is filled, usually the core material: 12.08 for '
              'silicon at 1550 nm.'),
    Param('initial_density', 'float', 0.5, label='Initial density', unit='1', min=0.0, max=1.0,
          help='Density rho of every design cell before the first iteration (also the fixed design outside an '
               'optimization study). 0.5 (grey) leaves the optimiser free in both directions; 0 or 1 starts from the '
               'void or solid design.',
          tip='Starting fill of the design area, from 0 (empty) to 1 (filled). 0.5 is a neutral start.'),
    Param('filter_radius', 'float', 1e-7, label='Filter radius', unit='m', min=0.0,
          help='Radius R of the conic density filter rho_tilde_a = sum_b w_ab A_b rho_b / sum_b w_ab A_b, w_ab = '
               'max(0, R - |c_a - c_b|) over the design cells (A_b cell area, c cell centre): it sets the smallest '
               'feature (~R) and suppresses checkerboards. 0 = no filter. Use >= 2 cells; typical 50-150 nm. Host '
               'memory grows as cells x (R / dx)^2.',
          tip='Smallest feature the design can form, roughly. Larger gives smoother, easier to make shapes; 0 turns '
              'smoothing off. Use at least two mesh cells.'),
    Param('projection_beta', 'float', 0.0, label='Projection sharpness', unit='1', min=0.0,
          help='Steepness beta of the tanh projection rho_bar = (tanh(beta eta) + tanh(beta (rho_tilde - eta))) / '
               '(tanh(beta eta) + tanh(beta (1 - eta))) that pushes the filtered density toward 0 or 1; 0 = no '
               'projection (rho_bar = rho_tilde). Larger beta gives a more binary design (the grayness series shows '
               'the remaining grey) but a stiffer problem with more local optima; typical 0-8, or 8-32 as the final '
               'value of the optimization study continuation_stages (earlier stages use a smaller beta).',
          tip='How strongly intermediate fills are pushed to empty or filled: 0 keeps them, 4 to 8 makes the design '
              'nearly black and white but harder to optimise.'),
    Param('projection_eta', 'float', 0.5, label='Projection threshold', unit='1', min=0.0, max=1.0,
          help='Threshold eta of the tanh projection (unused at projection_beta 0): filtered densities above it move '
               'toward 1, below it toward 0; 0.5 is neutral, lower erodes less (dilated design).',
          tip='Fill level that separates empty from filled when sharpening is on. 0.5 is neutral.'),
)
_DESCRIPTION = (
    'Density design region on the selected domains (window cells only): each cell carries a density rho in [0, 1], '
    'filtered by a conic filter of filter_radius and projected by a tanh of projection_beta / projection_eta to '
    'rho_bar, and gets eps = eps_min + rho_bar (eps_max - eps_min) in its permittivity arrays (em_fdfd_ez: eps_r; '
    'em_fdfd_hz: eps_xx and eps_yy, an isotropic change), overwriting the material permittivity there (sigma_e, poles '
    'and mu_r stay the material\'s). The optimization study varies rho to maximise the adjoint_objective sum J; any '
    'other study solves the initial_density design. Design regions of one physics must not share cells. Keep the '
    'region off every source, port and objective line and the cells beyond their hi faces (the adjoint gradient is '
    'not exact there). Output field design_density = rho_bar (0 outside every design region). Result \'xy\' versus '
    'iteration (0 = initial design, one point per accepted L-BFGS-B iteration): series objective (J of the whole '
    'model, W/m) and grayness (mean 4 rho_bar (1 - rho_bar) of this region, 0 = binary); extras n_cells, eps_min, '
    'eps_max, mean_density and final_grayness of the current design, and note (non-empty outside an optimization '
    'study, whose result is one point: iteration 0 with objective NaN).')
_SUMMARY = ('Marks an area whose material an optimization study may change cell by cell, between two permittivities, '
            'to improve the optimisation objectives.')


def _preview_selection_mask(vctx, selection):
    """bool (mx, my) mask of a domain selection on the material-preview pixels (pixel centres inside); None when the
    preview or the selection does not resolve."""
    material_map = vctx.material_map()
    if material_map is None:
        return None
    image, _, extent = material_map
    try:
        entities = expand_entities(vctx.spec, selection)
    except (ValueError, KeyError, TypeError):
        return None
    if 'all' in entities:
        return np.ones(image.shape, dtype=bool)
    x_centres, y_centres = preview_pixel_centers(extent, image.shape)
    object_mask = domain_object_rasterizer(vctx.shapes, x_centres, y_centres, np)
    mask = np.zeros(image.shape, dtype=bool)
    for entity in entities:
        try:
            window_mask = object_mask(entity)
        except (ValueError, KeyError, TypeError):   # a failed part of a composite: reported at the geometry
            continue
        if window_mask is not None:
            window, cropped = window_mask
            mask[window] |= cropped
    return mask


def _line_in_mask(vctx, shape, mask, extent):
    """True when an axis-aligned line shape passes through a pixel of the preview mask (not axis-aligned: False)."""
    line_axes = _line_axes(shape.values, vctx.domain_size)
    if line_axes is None:
        return False
    along_axis, _, position, span = line_axes
    along_index = 'xy'.index(along_axis)
    across_index = 1 - along_index
    first, last = (preview_pixel(coord, extent[2 * along_index], extent[2 * along_index + 1], mask.shape[along_index])
                   for coord in span)
    row = preview_pixel(position, extent[2 * across_index], extent[2 * across_index + 1], mask.shape[across_index])
    pixels = slice(first, last + 1)
    return bool((mask[pixels, row] if along_index == 0 else mask[row, pixels]).any())


def _earlier_design_regions(vctx, physics_spec, feature_spec):
    """[(physics, feature)] of the enabled design_region features before this one in spec order (each overlapping
    pair is reported once, at its later region)."""
    earlier = []
    for physics in vctx.spec.physics:
        for feature in physics.features:
            if physics.name == physics_spec.name and feature.name == feature_spec.name:
                return earlier
            if feature.enabled and feature.type == 'design_region':
                earlier.append((physics, feature))
    return earlier


def _optimization_overlap_issues(vctx, physics_spec, feature_spec, path):
    """Optimization study: errors for an earlier enabled design_region sharing preview pixels with this one, and
    warnings for source / port / objective lines of a physics of this type crossing it."""
    mask = _preview_selection_mask(vctx, feature_spec.selection)
    if mask is None or not mask.any() or vctx.domain_size is None:
        return []
    extent = vctx.material_map()[2]
    issues = []
    for physics, feature in _earlier_design_regions(vctx, physics_spec, feature_spec):
        other_mask = _preview_selection_mask(vctx, feature.selection)
        if other_mask is None or not (mask & other_mask).any():
            continue
        if physics.name == physics_spec.name:
            consequence = 'design regions of one physics must not share cells (the build refuses them)'
        elif physics.type == physics_spec.type:
            consequence = ('both write the same permittivity array there (the study refuses them); one region serves '
                           'both physics')
        else:
            consequence = (f'{physics.type} and {physics_spec.type} regions write different permittivity arrays '
                           '(eps_r vs eps_xx / eps_yy), so the cells get two independently optimised designs')
        issues.append(Issue('error', path, f"shares cells with design_region '{feature.name}' of '{physics.name}': "
                                           f'{consequence}; give them disjoint domains'))
    for physics in vctx.spec.physics:
        if physics.type != physics_spec.type:
            continue
        for feature in physics.features:
            if not feature.enabled or feature.type not in _FIXED_LINE_FEATURES:
                continue
            if any(_line_in_mask(vctx, shape, mask, extent) for shape in _selected_shapes(vctx, feature, 'line')):
                issues.append(Issue('warning', path, f"{feature.type} '{feature.name}' of '{physics.name}' crosses "
                                                     'the design region: the adjoint gradient holds its calibration, '
                                                     'port mode or line cells fixed, so the gradient there is not '
                                                     'exact; keep the region off source, port and objective lines '
                                                     '(em_fdfd_hz differentiates the face fields and face currents, '
                                                     'not the calibrations or modes)'))
    return issues


def _validate_design_region(vctx, physics_spec, feature_spec, values):
    """Info outside an optimization study; warning for equal permittivities (nothing to optimise); in an optimization
    study errors for overlapping design regions and warnings for fixed lines crossing the region."""
    path = feature_path(physics_spec, feature_spec)
    issues = []
    if vctx.study_type != OPTIMIZATION_STUDY:
        issues.append(Issue('info', path, f'design_region keeps its initial_density design in a {vctx.study_type} '
                                          f'study; only an {OPTIMIZATION_STUDY} study changes it and records its '
                                          'objective history'))
    else:
        issues += _optimization_overlap_issues(vctx, physics_spec, feature_spec, path)
    if values.get('eps_min') == values.get('eps_max'):
        issues.append(Issue('warning', path, f"eps_min equals eps_max ({values.get('eps_min')!r}): the density does "
                                             'not change the permittivity, so there is nothing to optimise'))
    return issues


def _apply_design_region(ctx, solver, feature_spec, values, selection):
    return solver.add_design_region(ctx.domain_mask(selection.entities), values['eps_min'], values['eps_max'],
                                    initial_density=values['initial_density'], filter_radius=values['filter_radius'],
                                    projection_beta=values['projection_beta'],
                                    projection_eta=values['projection_eta'])


def design_region_feature():
    """FeatureType 'design_region' of the em_fdfd_* types."""
    return FeatureType('design_region', 'Design region', 'domain_condition', 'domain', params=_PARAMS,
                       result_kind='xy', description=_DESCRIPTION, summary=_SUMMARY, apply=_apply_design_region,
                       result=object_result(design_region_result), validate=_validate_design_region,
                       memory_estimate=design_region_memory_estimate)
