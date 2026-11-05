"""Model feature 'effective_index' of the in-plane EM types (em_fdtd_* / em_fdfd_*): the effective-index 2.5D bridge
(B4) writing eps = n_eff^2 of a vertical layer stack into the selected domains at build time."""
from ...model.schema import FeatureType, Issue, Param, feature_path
from .definition_common import auto_mesh_issues
from .effective_index import SLAB_POLARIZATIONS, effective_index
from .placement import PERMITTIVITY_ARRAYS

FEATURE_NAME = 'effective_index'
# top-view polarisation -> the vertical slab polarisation whose dominant E lies in the same direction
MATCHING_SLAB_POLARIZATION = {'hz': 'te', 'ez': 'tm'}


def _params(polarization):
    """Stack, wavelength, slab polarisation (default: the one matching the top-view polarisation) and mode number."""
    default_slab = MATCHING_SLAB_POLARIZATION[polarization]
    return (
        Param('layer_indices', 'float_list', [1.444, 3.476, 1.444], label='Layer refractive indices', unit='1',
              min=0.0, min_exclusive=True,
              tip='Refractive indices of the vertical layers from bottom to top: the substrate first, then each film, '
                  'then the cover above.\n\nExample for 220 nanometre silicon on oxide under oxide: 1.444, 3.476, '
                  '1.444.',
              help='Real refractive indices of the vertical stack, bottom to top: substrate (semi-infinite), one or '
                   'more films, cover (semi-infinite); at least 3 entries. Losses are not modelled (real indices '
                   'only).'),
        Param('layer_thicknesses', 'float_list', [220e-9], label='Film thicknesses', unit='m', min=0.0,
              min_exclusive=True,
              tip='Thickness of each film between the substrate and the cover, from bottom to top, in metres.\n\n'
                  'One entry fewer than the film count plus two: 220e-9 for a single 220 nanometre film.',
              help='Thicknesses of the films (layer_indices without its first and last entry), bottom to top; '
                   'exactly len(layer_indices) - 2 entries.'),
        Param('wavelength', 'float', 1.55e-6, label='Wavelength', unit='m', min=0.0, min_exclusive=True,
              tip='Vacuum wavelength at which the layer stack is collapsed to one refractive index.\n\n'
                  'Use the centre wavelength of the light you simulate: the index is not recomputed for other '
                  'wavelengths.',
              help='Vacuum wavelength of the slab-mode solve. n_eff is computed once at this wavelength and used at '
                   'every frequency (FDTD pulses, frequency_domain sweeps), so keep the band narrow around it.'),
        Param('slab_polarization', 'choice', default_slab, label='Slab polarisation', choices=tuple(SLAB_POLARIZATIONS),
              choice_labels=(('te', 'Electric field along the layers'),
                             ('tm', 'Electric field across the layers')),
              tip='Polarisation of the vertical slab mode.\n\n'
                  '- Electric field along the layers: matches light whose electric field lies in the drawing plane.\n'
                  '- Electric field across the layers: matches light whose electric field points out of the '
                  'drawing.\n\nThe default matches this physics.',
              help="'te': slab mode with E parallel to the layers, the vertical-stack partner of the in-plane-E "
                   "top-view types em_fdtd_hz / em_fdfd_hz; 'tm': E mostly normal to the layers (H parallel), the "
                   f"partner of em_fdtd_ez / em_fdfd_ez. Default here: '{default_slab}' (validation notes a "
                   'mismatch).'),
        Param('mode_number', 'int', 1, label='Slab mode', min=1,
              tip='Which guided mode of the layer stack to use: 1 is the fundamental mode.',
              help='Guided slab mode by descending n_eff, 1 = fundamental; an error when the stack guides fewer.'),
    )


def _stack_index(values):
    """n_eff of the feature's stack (ValueError with the user-facing reason)."""
    return effective_index(values['layer_indices'], values['layer_thicknesses'], values['wavelength'],
                           values['slab_polarization'], values['mode_number'])


def _apply_effective_index(ctx, solver, feature_spec, values, selection):
    """Write n_eff^2 into every permittivity array of the solver on the selected cells (PML included under 'all')."""
    n_eff = _stack_index(values)
    mask = ctx.domain_mask(selection.entities)
    arrays = tuple(solver.permittivity_arrays)
    for array in arrays:
        ctx.domain.materials.assign(array, mask, n_eff ** 2)
    solver.materials_changed(arrays)
    ctx.log('info', f"feature '{feature_spec.name}': effective index n_eff = {n_eff:.6g} (eps = {n_eff ** 2:.6g})")
    return None


def _make_validate(polarization):
    """Stack errors, the computed n_eff (info), a slab / top-view polarisation mismatch and the auto mesh caveat."""
    def validate(vctx, physics_spec, feature_spec, values):
        path = feature_path(physics_spec, feature_spec)
        try:
            n_eff = _stack_index(values)
        except ValueError as exc:
            return [Issue('error', path, f'effective index: {exc}')]
        issues = [Issue('info', path, f"{values['slab_polarization'].upper()} mode {values['mode_number']} of the "
                                      f'stack: n_eff = {n_eff:.6g}, eps = {n_eff ** 2:.6g} on the selected cells')]
        matching = MATCHING_SLAB_POLARIZATION[polarization]
        if values['slab_polarization'] != matching:
            issues.append(Issue('info', path, f"slab polarization '{values['slab_polarization']}' does not match this "
                                              f"physics ('{polarization}' fields: '{matching}' has the same dominant "
                                              'E direction)'))
        issues.extend(auto_mesh_issues(vctx, physics_spec, path, 'the auto mesh sizes cells from the material indices, '
                                                                  f'not from this n_eff = {n_eff:.4g}: give the '
                                                                  'selected domains a material of at least that index '
                                                                  'or use a uniform / graded mesh'))
        return issues
    return validate


def effective_index_feature(polarization):
    """FeatureType 'effective_index' of an in-plane EM type of polarization 'ez' | 'hz'."""
    arrays = ' / '.join(PERMITTIVITY_ARRAYS[polarization])
    return FeatureType(
        FEATURE_NAME, 'Effective index (2.5D)', 'domain_condition', 'domain', params=_params(polarization),
        summary='Treats the drawing as the top view of a layered chip: the selected domains get the refractive index '
                'of the guided mode of their vertical layer stack.',
        description=f'Effective-index method (2.5D): at build time the guided slab mode mode_number of the vertical '
                    f'stack layer_indices / layer_thicknesses is solved at wavelength (the A3 slab-mode solver on a '
                    f'1D grid of ~wavelength / (100 n_max) cells, claddings deepened until the tails decay) and '
                    f'{arrays} = n_eff^2 is written on the selected cells, replacing the material background '
                    f'permittivity there (later features win on overlaps; the material\'s sigma_e and dispersion '
                    f'poles stay; a state-dependent permittivity expression of a material overwrites it). Approximate: '
                    f'vertical leakage, the n_eff dispersion away from wavelength and the vertical mode mismatch at '
                    f'stack boundaries are ignored. Validation reports n_eff (info); thermo-optic couplings take n_eff '
                    f'as their reference index n0.',
        apply=_apply_effective_index, validate=_make_validate(polarization))
