"""Factories of the standard EM <-> heat couplings for scripts; the model types em_heat_joule / heat_em_thermo_optic
(definitions.py) create theirs through them. Both are E13 contributors: add_coupling sums them with other writers."""
from ..physics.heat.coupling import (em_to_heat_power, em_to_heat_power_final, heat_to_em_dn, heat_to_em_dn_xx,
                                     heat_to_em_dn_yy, heat_to_em_dn_without_expansion)
from .coupling_term import CouplingGroup, CouplingTerm

_THERMO_OPTIC_TERMS = {   # EM permittivity array -> (EM source slot, heat transform giving its dn)
    'eps_r': ('eps_r_update', heat_to_em_dn),
    'eps_xx': ('eps_xx_update', heat_to_em_dn_xx),
    'eps_yy': ('eps_yy_update', heat_to_em_dn_yy),
}
THERMO_OPTIC_SLOTS = tuple(slot for slot, _ in _THERMO_OPTIC_TERMS.values())


def joule_heating(em: str = 'em', heat: str = 'heat', every: int = None, name: str = 'em_heat_joule'):
    """'sum' contributor: mean absorbed EM power density since the last transfer -> heat volumetric_power (flush merges
    the last partial window); every None = the heat stride. Reads q_accum only so runner.prepare enables the accounting."""
    return CouplingTerm(em, 'q_accum', heat, 'volumetric_power', em_to_heat_power, every=every, name=name,
                        flush_transform=em_to_heat_power_final, contribution='sum')


def thermo_optic(heat: str = 'heat', em: str = 'em', permittivity_arrays=('eps_r',), every: int = None,
                 after_source_step: bool = True, name: str = 'heat_em_thermo_optic', with_expansion: bool = True):
    """heat T -> EM permittivity: one 'index' contributor (dn) per array of `permittivity_arrays` (pass the EM solver's,
    e.g. em.permittivity_arrays for H_z), a CouplingGroup when several. Keep after_source_step while every = heat every.
    with_expansion: True = the (1 + alpha dT) factors on every array, False = none (a solid strain-optic coupling
    supplies the deformation), or a tuple of the arrays that get them (em_modes cross-sections: ('eps_r',))."""
    unknown = [array for array in permittivity_arrays if array not in _THERMO_OPTIC_TERMS]
    if unknown or not permittivity_arrays:
        raise ValueError(f"thermo_optic: permittivity arrays {tuple(permittivity_arrays)} of '{em}' are not all "
                         f"thermo-optic; supported: {', '.join(_THERMO_OPTIC_TERMS)}")

    expanded = set(permittivity_arrays) if with_expansion is True else set(with_expansion or ())

    def slot_and_transform(array):
        slot, transform = _THERMO_OPTIC_TERMS[array]
        return slot, (transform if array in expanded else heat_to_em_dn_without_expansion(slot))

    if len(permittivity_arrays) == 1:
        slot, transform = slot_and_transform(permittivity_arrays[0])
        return CouplingTerm(heat, 'T', em, slot, transform, every=every, name=name,
                            after_source_step=after_source_step, contribution='index')
    terms = [CouplingTerm(heat, 'T', em, slot, transform, every=1, name=f'{name}.{slot}', contribution='index')
             for slot, transform in (slot_and_transform(array) for array in permittivity_arrays)]
    return CouplingGroup(terms, every=every, name=name, after_source_step=after_source_step)
