"""Magnetic coupling types: magnetic_heat_induction (magnetic_fields_ac -> heat, E13 'sum' into volumetric_power) and
magnetic_solid_lorentz_force (magnetostatics / magnetic_fields_ac -> solid_mechanics, 'sum' into body_force_x / _y;
transforms physics/magnetic/couplings.py); discovered and registered by definitions.register."""
from ..model.definition_helpers import is_axisymmetric
from ..model.schema import CouplingType, Issue
from ..physics.magnetic.couplings import FORCE_SLOTS, HEAT_SLOT, InductionHeatSource, make_lorentz_force_transforms
from ..physics.magnetic.selftests_coupled import (INDUCTION_BILLET_DESCRIPTION, INDUCTION_BILLET_LABEL,
                                                  induction_billet_spec)
from .coupling_term import CouplingGroup, CouplingTerm
from .definitions import HEAT_SOURCE_SUM_TEXT, HEAT_TYPE_NAME, writes_heat_source
from .index_terms import after_source_step, physics_and_type

# = the physics.magnetic selftests / selftests_ac / selftests_coupled constants (not imported here)
MAGNETOSTATICS_TYPE_NAME = 'magnetostatics'
MAGNETIC_AC_TYPE_NAME = 'magnetic_fields_ac'
INDUCTION_COUPLING_NAME = 'magnetic_heat_induction'
LORENTZ_COUPLING_NAME = 'magnetic_solid_lorentz_force'
SOLID_TYPE_NAME = 'solid_mechanics'


# ---- create callbacks (worker thread, GPU) ----
def _create_induction(ctx, coupling_spec, values, every):
    """'sum' contributor of q_eddy + q_magnetic into heat volumetric_power, right after each magnetic solve."""
    return CouplingTerm(coupling_spec.source, 'q_eddy', coupling_spec.target, HEAT_SLOT, InductionHeatSource(),
                        every=every, name=coupling_spec.name, contribution='sum',
                        after_source_step=after_source_step(ctx, coupling_spec, every))


def _create_lorentz(ctx, coupling_spec, values, every):
    """CouplingGroup of the two 'sum' body-force terms sharing one LorentzForceDensity."""
    force_x, force_y = make_lorentz_force_transforms()
    terms = [CouplingTerm(coupling_spec.source, 'B_abs', coupling_spec.target, slot, transform, every=1,
                          name=f'{coupling_spec.name}.{slot}', contribution='sum')
             for slot, transform in zip(FORCE_SLOTS, (force_x, force_y))]
    return CouplingGroup(terms, every=every, name=coupling_spec.name,
                         after_source_step=after_source_step(ctx, coupling_spec, every))


def _writes_lorentz(vctx, coupling_spec):
    return {slot: 'sum' for slot in FORCE_SLOTS}


# ---- validation (CPU only) ----
def _validate_lorentz(vctx, coupling_spec, values):
    """Error for an axisymmetric magnetic source (solid_mechanics is plane strain)."""
    source_spec, source_type = physics_and_type(vctx, coupling_spec.source)
    if source_type is None:
        return []
    settings = vctx.values(source_type.settings, source_spec.settings) or {}
    if not is_axisymmetric(settings):
        return []
    return [Issue('error', ('coupling', coupling_spec.name),
                  f"magnetic physics '{source_spec.name}' is axisymmetric, but {SOLID_TYPE_NAME} is plane strain: the "
                  'Lorentz force of a ring current (radial hoop load) has no planar image; use geometry planar')]


# ---- types ----
MAGNETIC_HEAT_INDUCTION = CouplingType(
    INDUCTION_COUPLING_NAME, 'Induction heating (magnetic fields AC -> heat)', MAGNETIC_AC_TYPE_NAME, HEAT_TYPE_NAME,
    summary='Turns the eddy-current and magnetic losses of the alternating magnetic field into a heat source of the '
            'heat physics. It adds up with the other heating couplings of that heat physics.',
    description='q_eddy + q_magnetic [W/m^3] of the last magnetic_fields_ac solve (period averages) become the heat '
                'source. ' + HEAT_SOURCE_SUM_TEXT + ' Its window integral is eddy_loss plus the integral of '
                'q_magnetic (W/m planar, W axisymmetric: use the same geometry for both physics, else a validation '
                'warning). Stranded coils carry no eddy current, and their resistive loss is not transferred. '
                'Transfers after each magnetic solve, which repeats only when a material or source changed, e.g. a '
                'sigma_dc expression of heat_T (each change may cost a host refactorisation). Stationary study: one '
                'Picard fixed point; time-dependent: heat steps with the loss of the latest solve.',
    create=_create_induction,
    writes=writes_heat_source,
    mesh_transfer='value')

MAGNETIC_SOLID_LORENTZ_FORCE = CouplingType(
    LORENTZ_COUPLING_NAME, 'Lorentz force (magnetic fields -> solid mechanics)',
    (MAGNETOSTATICS_TYPE_NAME, MAGNETIC_AC_TYPE_NAME), SOLID_TYPE_NAME,
    summary='Applies the magnetic force on current-carrying conductors (J x B) to the solid mechanics. Planar models '
            'only.',
    description='f = J x B [N/m^3] of a planar magnetic physics: f_x = -J_z B_y, f_y = J_z B_x; magnetostatics J = '
                'J_coupled + J_applied, magnetic_fields_ac the time average Re(J_total B*) / 2 (the 2 omega part is '
                'dropped). Handed per cell to body_force_x / _y, additive (sums with em_solid_optical_force, one '
                'every; no field_transfer into those slots beside it); void solid cells (youngs_modulus 0) drop it, '
                'so a winding modelled as air carries no load. Axisymmetric magnetic physics: validation error. No '
                'magnetisation force (permeable or magnetised material, Maxwell stress at mu_dc jumps) and no '
                'magnetostriction. Transfers after each magnetic solve.',
    create=_create_lorentz,
    validate=_validate_lorentz,
    writes=_writes_lorentz,
    mesh_transfer='value')


def _induction_selftest_cases():
    """The induction-heating SelfTestCases (imported on the first registry.selftests(), not by get_registry)."""
    from ..physics.magnetic.selftests_coupled import INDUCTION_CASES
    return INDUCTION_CASES


def _lorentz_selftest_cases():
    """The Lorentz-force SelfTestCases (imported on the first registry.selftests(), not by get_registry)."""
    from ..physics.magnetic.selftests_coupled import LORENTZ_CASES
    return LORENTZ_CASES


def register(registry, physics):
    """magnetic_heat_induction (+ its selftest loader and the induction_billet template) when magnetic_fields_ac and
    heat are registered; magnetic_solid_lorentz_force (+ its selftest loader) when a magnetic type and solid_mechanics
    are (physics: definitions.RegisteredPhysics)."""
    physics_type_names = physics.names
    if {MAGNETIC_AC_TYPE_NAME, HEAT_TYPE_NAME} <= physics_type_names:
        registry.register_coupling(MAGNETIC_HEAT_INDUCTION)
        registry.register_template(
            'induction_billet', INDUCTION_BILLET_LABEL, INDUCTION_BILLET_DESCRIPTION, induction_billet_spec,
            summary='A coil heats an aluminium billet by induced eddy currents: the alternating field, the loss in the '
                    'billet skin and the temperature rise over 20 seconds. Rotationally symmetric.')
        registry.register_selftests(_induction_selftest_cases)
    if SOLID_TYPE_NAME in physics_type_names and (
            {MAGNETOSTATICS_TYPE_NAME, MAGNETIC_AC_TYPE_NAME} & physics_type_names):
        registry.register_coupling(MAGNETIC_SOLID_LORENTZ_FORCE)
        registry.register_selftests(_lorentz_selftest_cases)
