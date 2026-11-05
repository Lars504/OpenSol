"""Coupling types between elastic_waves and the frequency-domain EM types: acoustics_em_acousto_optic (elastic strain
-> EM index, an E13 'index' contributor) and em_acoustics_photoacoustic (absorbed light -> heating_power, a 'sum'
contributor), plus their self-tests; discovered and registered by definitions.register."""
from ..model.definition_helpers import selected_material_sets
from ..model.schema import CouplingType, Issue
from ..physics.acoustics.couplings import acousto_optic_shear_shift, make_acousto_optic_transform
from .coupling_term import CouplingTerm
from .index_terms import (after_source_step, has_off_diagonal_slot, index_contributor, index_slots, physics_and_type,
                          target_type, writes_index_and_off_diagonal_slots)

ACOUSTICS_TYPE_NAME = 'elastic_waves'   # = physics.acoustics.definition.PHYSICS_TYPE_NAME
ACOUSTO_OPTIC_TYPE_NAME = 'acoustics_em_acousto_optic'   # = selftests_acoustics.ACOUSTO_OPTIC_TYPE_NAME (not imported)
PHOTOACOUSTIC_TYPE_NAME = 'em_acoustics_photoacoustic'
HEATING_SLOT = 'heating_power'          # = physics.acoustics.solver.HEATING_SLOT
_PHOTOELASTIC_NAMES = ('photoelastic_p11', 'photoelastic_p12', 'photoelastic_p16', 'photoelastic_p21',
                       'photoelastic_p22', 'photoelastic_p26', 'photoelastic_p31', 'photoelastic_p32',
                       'photoelastic_p36', 'photoelastic_p61', 'photoelastic_p62',
                       'photoelastic_p66')   # = physics.acoustics.solver p11 / p12 + PHOTOELASTIC_TENSOR_ARRAYS
OFF_DIAGONAL_WARNING_RATIO = 1e-6   # dropped off-diagonal share of a crystal's index change that warns


# ---- create callbacks (worker thread, GPU) ----
def _create_acousto_optic(ctx, coupling_spec, values, every):
    """One 'index' contributor per index slot of the target type (+ the 'sum' eps_xy_update term of em_fdfd_hz), a
    CouplingGroup when several."""
    return index_contributor(ctx, coupling_spec, every, 'sxx', make_acousto_optic_transform,
                             off_diagonal_transform=acousto_optic_shear_shift)


def _create_photoacoustic(ctx, coupling_spec, values, every):
    """'sum' contributor copying the FDFD q_abs of the last solve into heating_power."""
    return CouplingTerm(coupling_spec.source, 'q_abs', coupling_spec.target, HEATING_SLOT, every=every,
                        name=coupling_spec.name, after_source_step=after_source_step(ctx, coupling_spec, every),
                        contribution='sum')


def _writes_heating(vctx, coupling_spec):
    return {HEATING_SLOT: 'sum'}


# ---- validation (CPU only) ----
def crystal_off_diagonal_issues(vctx, path, kind, what, physics_type):
    """Warning per selected material whose crystal entry has a dropped off-diagonal index change above
    OFF_DIAGONAL_WARNING_RATIO (kind 'photoelastic' | 'pockels'): the anti-plane yz / xz rows always, the xy row when
    the target type has in-plane E but no eps_xy slot; malformed entries are reported at the material."""
    from ..materials import crystal_entry   # lazy: physics -> materials stays out of module import time
    from ..materials.crystal_tensors import sagittal_crystal
    drops_shear = 'eps_xx_update' in physics_type.source_slots and not has_off_diagonal_slot(physics_type)
    components = '_yz / _xz / _xy' if drops_shear else '_yz / _xz'
    issues = []
    for material in vctx.spec.materials:
        entry = crystal_entry(material.dispersion)
        if entry is None or not material.selection:
            continue
        try:
            crystal = sagittal_crystal(entry)
        except ValueError:
            continue
        ratio = getattr(crystal, f'antiplane_{kind}_ratio')
        if drops_shear:
            ratio = max(ratio, getattr(crystal, f'shear_{kind}_ratio'))
        if ratio > OFF_DIAGONAL_WARNING_RATIO:
            issues.append(Issue('warning', path,
                                f"material '{material.name}': its crystal {what} has off-diagonal parts in model axes "
                                f'(Delta(1/n^2){components} up to {ratio:.3g} of its largest entry) which '
                                f'{physics_type.name} cannot hold (only em_fdfd_hz has an eps_xy slot; yz / xz would '
                                'couple E_z to the in-plane field): they are dropped'))
    return issues


def _validate_acousto_optic(vctx, coupling_spec, values):
    """Error without an index slot on the target; warnings when no material is photoelastic or a crystal's dropped
    off-diagonal part is not negligible; info on the model."""
    path = ('coupling', coupling_spec.name)
    physics_type = target_type(vctx, coupling_spec)
    if physics_type is None:
        return []
    if not index_slots(physics_type):
        return [Issue('error', path, f"target physics '{coupling_spec.target}' ({physics_type.name}) has no "
                                     'permittivity slot')]
    issues = []
    if not selected_material_sets(vctx, _PHOTOELASTIC_NAMES):
        issues.append(Issue('warning', path, 'no selected material sets photoelastic_p11 / photoelastic_p12 (or the '
                                             'tensor entries): the coupling changes no index (library silica and '
                                             'silicon carry them, crystal entries set them)'))
    issues += crystal_off_diagonal_issues(vctx, path, 'photoelastic', 'photoelastic response', physics_type)
    shear_text = ('the shear strain also through the xy row into eps_xy (isotropic p66 = (p11 - p12) / 2 or the tensor '
                  'entries photoelastic_p61 / _p62 / _p66)' if has_off_diagonal_slot(physics_type) else
                  'the shear strain only through the tensor entries photoelastic_p16 / _p26 / _p36 of the diagonal '
                  f'rows, no off-diagonal index change ({physics_type.name} has no eps_xy slot)')
    issues.append(Issue('info', path,
                        f"'{coupling_spec.target}' sees the elastic strain of '{coupling_spec.source}' at the steps it "
                        're-solves on (its every): photoelastic part only (no moving-boundary term), e_zz = 0, '
                        f'{shear_text}'))
    return issues


def _validate_photoacoustic(vctx, coupling_spec, values):
    """Warnings: the target's heating envelope is 'instantaneous' (a cw absorbed power then heats for the whole run);
    no selected material sets gruneisen."""
    path = ('coupling', coupling_spec.name)
    issues = []
    acoustics_spec, acoustics_type = physics_and_type(vctx, coupling_spec.target)
    if acoustics_type is None:
        return []
    settings = vctx.values(acoustics_type.settings, acoustics_spec.settings) or {}
    if settings.get('heating_envelope') == 'instantaneous':
        issues.append(Issue('warning', path,
                            f"'{coupling_spec.target}' takes the heating as delivered (heating_envelope "
                            "'instantaneous'): the time-averaged absorbed power of the frequency-domain solve then "
                            'heats steadily for the whole run (the pressure ramps up instead of forming a pulse); set '
                            "heating_envelope 'gaussian' for a laser pulse"))
    if not selected_material_sets(vctx, ('gruneisen',)):
        issues.append(Issue('warning', path, 'no selected material sets gruneisen: the absorbed power makes no sound'))
    return issues


# ---- types ----
def acoustics_coupling_types(optical_types, absorbing_optical_types):
    """The two coupling types for the registered frequency-domain EM type names (sorted tuples, non-empty)."""
    acousto_optic = CouplingType(
        ACOUSTO_OPTIC_TYPE_NAME, 'Acousto-optic effect (elastic waves -> electromagnetic waves)', ACOUSTICS_TYPE_NAME,
        optical_types,
        summary='Changes the refractive index of a frequency-domain wave physics from the strain of the elastic '
                'waves, as in an acousto-optic modulator. The optics is solved again on the frozen strain every few '
                'acoustic steps.',
        description='Acousto-optic (photoelastic) effect: each transfer writes, for every permittivity component ii '
                    'of the EM physics (E_z: zz through eps_r_update; H_z: xx and yy), dn_ii = -1/2 n0_ii^3 '
                    '(photoelastic_p11 e_ii + photoelastic_p12 (e_jj + e_kk)) of the elastic strain of the acoustic '
                    'stresses (elastic_strain_xx / _yy: the in-plane stiffness inverted at the cell centres, fluids '
                    'half the volumetric strain each, e_zz = 0 plane strain; the photoelastic math of '
                    'solid_em_strain_optic without its deformation term); n0 = the E13 '
                    'shared reference index. A crystal (a material with the photoelastic tensor entries photoelastic_p16 '
                    '.. _p66, set by a {"model": "crystal"} entry) instead gets the in-plane row of its model-axis '
                    'tensor, dn_ii = -1/2 n0_ii^3 (p_i1 e_xx + p_i2 e_yy + p_i6 g), g = 2 e_xy the engineering shear at '
                    'the cell centre (mean of its 4 corners), unset entries = the isotropic values. em_fdfd_hz also '
                    'gets the xy row in its eps_xy_update slot (a sum contributor): Delta eps_xy = -n0_xx^2 n0_yy^2 '
                    '(p_61 e_xx + p_62 e_yy + p_66 g), isotropic p_66 = (p11 - p12) / 2 (so a shear wave in glass '
                    'rotates the in-plane index ellipse; photoelastic_p44 is not read here: library silicon gets '
                    '(p11 - p12) / 2 = -0.0555 instead of its p44 = -0.051: set photoelastic_p66 = p44 on the material '
                    'for the cubic shear with cube axes along x, y); em_fdfd_ez needs no xy term (E_z does not see it). The '
                    'anti-plane change (Delta(1/n^2)_yz, _xz) is dropped (validation warns for crystals). With piezo '
                    'arrays the strain is the accumulated total strain. It is an E13 index contributor (adds to heat_em_thermo_optic / '
                    'electrostatics_em_pockels on the same physics, one every). Only frequency-domain EM types: in a '
                    'time_dependent study the EM physics re-solves on the frozen strain every k acoustic steps (its '
                    'every, which this coupling inherits: transfer just before each re-solve), a stroboscopic sequence '
                    'of optical states; an optical FDTD in the same run is refused (its step is ~1e5 times shorter). '
                    'Choose k so k dt samples the acoustic period finely enough, keeping in mind each re-solve is a '
                    'sparse direct solve.',
        create=_create_acousto_optic,
        validate=_validate_acousto_optic,
        writes=writes_index_and_off_diagonal_slots)
    photoacoustic = CouplingType(
        PHOTOACOUSTIC_TYPE_NAME, 'Photoacoustic heating (electromagnetic waves -> elastic waves)',
        absorbing_optical_types, ACOUSTICS_TYPE_NAME,
        summary='Turns the light absorbed in a frequency-domain wave physics into heating of the elastic waves, which '
                "makes sound (photoacoustics). Set the elastic waves' coupled heating envelope to a gaussian pulse.",
        description='Photoacoustic source: the absorbed power density q_abs [W/m^3] of the last solve of the EM '
                    'physics (time-averaged 1/2 omega eps0 (-Im eps) |E|^2, as em_heat_joule) is added to the '
                    'heating_power slot of elastic_waves, which multiplies it by its heating_envelope (gaussian: a '
                    'laser pulse of energy density q sqrt(pi) heating_pulse_width; instantaneous: steady heating for '
                    'the whole run, validation warns) and adds -gruneisen q to d(sxx)/dt and d(syy)/dt. It is a '
                    "'sum' contributor: several such couplings (e.g. one per wavelength) add; a field_transfer into "
                    'heating_power cannot be combined with them. Time_dependent study: transferred after each re-solve '
                    'of the EM physics (normally only its first, unless its permittivity changes).',
        create=_create_photoacoustic,
        validate=_validate_photoacoustic,
        writes=_writes_heating,
        mesh_transfer='value')
    return acousto_optic, photoacoustic


def _selftest_cases():
    """The acoustics-optics SelfTestCases (imported on the first registry.selftests(), not by get_registry)."""
    from .selftests_acoustics import ACOUSTICS_OPTICS_CASES
    return ACOUSTICS_OPTICS_CASES


def register(registry, physics):
    """The two coupling types, then their selftest loader, when elastic_waves and an absorbing frequency-domain EM type
    (em_fdfd_*) are registered (physics: definitions.RegisteredPhysics)."""
    if ACOUSTICS_TYPE_NAME not in physics.names or not physics.absorbing_optical_types:
        return
    for coupling_type in acoustics_coupling_types(physics.optical_types, physics.absorbing_optical_types):
        registry.register_coupling(coupling_type)
    registry.register_selftests(_selftest_cases)
