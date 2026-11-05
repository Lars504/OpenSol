"""E13 index-contributor helpers shared by the coupling definitions and physics/mechanics (imports only
coupling_term and physics.em.index_reference, so a physics definition.py may import it); also the off-diagonal
eps_xy_update 'sum' term of the acousto-optic and Pockels couplings."""
from ..physics.em.index_reference import INDEX_SLOTS, OFF_DIAGONAL_SLOT
from .coupling_term import CouplingGroup, CouplingTerm


def after_source_step(ctx, coupling_spec, every):
    """True when the source steps every `every` ticks: transfer right after it steps (no extra lag), not just before
    the target steps (the phase of the window-averaged Joule terms)."""
    return ctx.sim.runner.strides[coupling_spec.source] == every


def index_slots(physics_type):
    """Source slots of a physics type that take 'index' contributions (index_reference.INDEX_SLOTS), in order."""
    return tuple(slot for slot in physics_type.source_slots if slot in INDEX_SLOTS)


def physics_and_type(vctx, physics_name):
    """(PhysicsSpec, PhysicsType) of a physics instance, or (None, None) when either is unknown (reported by the
    coupling endpoint checks; CPU)."""
    try:
        physics_spec = vctx.spec.physics_by_name(physics_name)
        return physics_spec, vctx.registry.physics_type(physics_spec.type)
    except KeyError:
        return None, None


def target_type(vctx, coupling_spec):
    """PhysicsType of a coupling spec's target, or None when the physics or its type is unknown (CPU)."""
    return physics_and_type(vctx, coupling_spec.target)[1]


def has_off_diagonal_slot(physics_type):
    """True when a physics type takes the off-diagonal eps_xy_update 'sum' slot (em_fdfd_hz)."""
    return OFF_DIAGONAL_SLOT in physics_type.source_slots


def writes_index_slots(vctx, coupling_spec):
    """CouplingType.writes of an index coupling into every index slot of its target: {slot: 'index'}."""
    physics_type = target_type(vctx, coupling_spec)
    return {} if physics_type is None else dict.fromkeys(index_slots(physics_type), 'index')


def writes_index_and_off_diagonal_slots(vctx, coupling_spec):
    """writes_index_slots plus {eps_xy_update: 'sum'} when the target has that slot (index_contributor with an
    off-diagonal transform)."""
    writes = writes_index_slots(vctx, coupling_spec)
    physics_type = target_type(vctx, coupling_spec)
    if physics_type is not None and has_off_diagonal_slot(physics_type):
        writes[OFF_DIAGONAL_SLOT] = 'sum'
    return writes


def index_contributor(ctx, coupling_spec, every, source_field, transform_for_slot, off_diagonal_transform=None):
    """'index' CouplingTerm source_field -> each index slot of the target type (transform_for_slot(slot)), plus a 'sum'
    term into eps_xy_update (off_diagonal_transform) when given and the target has that slot: one term with the
    coupling's every, or a CouplingGroup of the terms (every 1 inside the group; the off-diagonal one last, so the
    diagonal slot sums record n0 first) when several."""
    physics_type = ctx.registry.physics_type(ctx.spec.physics_by_name(coupling_spec.target).type)
    slots = index_slots(physics_type)
    transfers_after_source = after_source_step(ctx, coupling_spec, every)

    def term(slot, transform, contribution, term_every, name):
        return CouplingTerm(coupling_spec.source, source_field, coupling_spec.target, slot, transform,
                            every=term_every, name=name, after_source_step=transfers_after_source,
                            contribution=contribution)

    entries = [(slot, transform_for_slot(slot), 'index') for slot in slots]
    if off_diagonal_transform is not None and has_off_diagonal_slot(physics_type):
        entries.append((OFF_DIAGONAL_SLOT, off_diagonal_transform, 'sum'))
    if len(entries) == 1:
        return term(*entries[0], every, coupling_spec.name)
    return CouplingGroup([term(slot, transform, contribution, 1, f'{coupling_spec.name}.{slot}')
                          for slot, transform, contribution in entries], every=every,
                         name=coupling_spec.name, after_source_step=transfers_after_source)
