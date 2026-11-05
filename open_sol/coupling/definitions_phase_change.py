"""Coupling type heat -> EM: heat_em_phase_change, the heat temperature history driving the phase_change_medium features
(physics/em/phase_change.py) of an in-plane EM physics; discovered and registered by definitions.register."""
from ..model.schema import CouplingType, Issue, effective_coupling_every, valid_every
from ..physics.em.phase_change import FEATURE_NAME, update_phase_change
from ..physics.em.placement import PERMITTIVITY_SLOTS
from .coupling_term import CouplingGroup, CouplingTerm
from .definitions import HEAT_TYPE_NAME
from .index_terms import after_source_step, index_slots, physics_and_type, target_type

PHASE_CHANGE_COUPLING = 'heat_em_phase_change'   # = physics.em.selftests_phase_change (not imported)
_TIME_DEPENDENT_STUDY = 'time_dependent'


def _no_transfer(field, source_solver, target_solver):
    """Transform of the bookkeeping terms of the further permittivity slots: the first term updates the media."""
    return None


def _create(ctx, coupling_spec, values, every):
    """One term per permittivity slot of the target (for the runner's writer checks); the first hands heat T and time
    to the target's phase_change_media, which rewrite their own cells, so nothing is injected."""
    def transform(field, heat, em):
        update_phase_change(em, field, heat.time)
        return None

    slots = tuple(PERMITTIVITY_SLOTS[ctx.sim.physics[coupling_spec.target].polarization])
    follow = after_source_step(ctx, coupling_spec, every)
    terms = [CouplingTerm(coupling_spec.source, 'T', coupling_spec.target, slot,
                          transform if index == 0 else _no_transfer, every=every, name=coupling_spec.name,
                          after_source_step=follow)
             for index, slot in enumerate(slots)]
    if len(terms) == 1:
        return terms[0]
    return CouplingGroup(terms, every=every, name=coupling_spec.name, after_source_step=follow)


def _writes(vctx, coupling_spec):
    """Overwrites every permittivity slot of the target (the media rewrite those arrays on their cells)."""
    physics_type = target_type(vctx, coupling_spec)
    return {} if physics_type is None else dict.fromkeys(index_slots(physics_type), '')


def _validate(vctx, coupling_spec, values):
    """Errors outside a time_dependent study and for non-transient heat; warnings for a target without an enabled
    phase_change_medium and for an every other than the heat every."""
    path = ('coupling', coupling_spec.name)
    issues = []
    if vctx.study_type != _TIME_DEPENDENT_STUDY:
        issues.append(Issue('error', path, f'{PHASE_CHANGE_COUPLING} steps the crystallinity along the temperature '
                                           f'history, which only a {_TIME_DEPENDENT_STUDY} study has; other studies keep '
                                           'initial_crystallinity: disable this coupling'))
    heat_spec, heat_type = physics_and_type(vctx, coupling_spec.source)
    em_spec, _ = physics_and_type(vctx, coupling_spec.target)
    if heat_spec is None or em_spec is None:
        return issues
    heat_values = vctx.values(heat_type.settings, heat_spec.settings if isinstance(heat_spec.settings, dict) else {})
    if vctx.study_type == _TIME_DEPENDENT_STUDY and heat_values and heat_values.get('time_mode') != 'transient':
        issues.append(Issue('error', path, f"heat '{heat_spec.name}' is stationary: a steady solve has no time, so "
                                           "nothing would crystallise; set its time_mode to 'transient'"))
    if not any(feature.enabled and feature.type == FEATURE_NAME for feature in em_spec.features):
        issues.append(Issue('warning', path, f"'{em_spec.name}' has no enabled {FEATURE_NAME}: this coupling drives "
                                             'nothing'))
    every = effective_coupling_every(coupling_spec, heat_spec, em_spec)
    if every is not None and valid_every(heat_spec.every) and every != heat_spec.every:
        issues.append(Issue('warning', path, f"every = {every} differs from the every = {heat_spec.every} of heat "
                                             f"'{heat_spec.name}': the crystallinity then steps over several heat "
                                             'steps with the last temperature only (leave every blank)'))
    return issues


HEAT_EM_PHASE_CHANGE_DESCRIPTION = (
    'Phase change driven by temperature: right after every heat step it hands the heat temperature T (end of the step) '
    'and the heat time to every phase_change_medium feature of the EM physics, which advances its crystallinity over '
    'the elapsed heat time and rewrites eps / sigma_e on its cells (see phase_change_medium). Only in a time_dependent '
    'study with transient heat (validation errors otherwise); leave every blank (= the heat every). It overwrites the '
    'permittivity of those cells, so it cannot be combined with another coupling into the permittivity slots of the '
    'same EM physics (heat_em_thermo_optic, electrostatics_em_pockels, field_transfer: validation error), nor with '
    'a state-dependent permittivity material there. The absorbed power reaches heat through em_heat_joule as usual. '
    'With em_fdfd_* the FDFD re-solves after every update (stroboscopic, the time scale of GST switching); with '
    'em_fdtd_* every heat window of every * dt_EM advances it (femtoseconds: only very fast test kinetics or melting '
    'act). The cells are rewritten on the bounding box of each medium and the EM physics is told materials_changed '
    'once per update.')


def register(registry, physics):
    """heat_em_phase_change for the EM types with a phase_change_medium feature, when heat is registered (physics:
    definitions.RegisteredPhysics)."""
    targets = tuple(sorted(physics_type.name for physics_type in physics.em_physics_types
                           if any(feature.name == FEATURE_NAME for feature in physics_type.features)))
    if not targets or HEAT_TYPE_NAME not in physics.names:
        return
    registry.register_coupling(CouplingType(
        PHASE_CHANGE_COUPLING, 'Phase change (heat -> electromagnetic waves)', HEAT_TYPE_NAME, targets,
        summary='Lets the temperature of the heat physics crystallise, melt and re-amorphise the phase-change media of '
                'a wave physics over time, changing their refractive index and absorption.',
        description=HEAT_EM_PHASE_CHANGE_DESCRIPTION,
        create=_create, validate=_validate, writes=_writes, mesh_transfer='field'))
