"""Coupling type laminar_flow -> heat: fluid_heat_convection, an E13 'sum' contributor of the advective heat source
(transform physics/fluid/convection.py); discovered and registered by definitions.register."""
from ..model.definition_helpers import evaluates_to
from ..model.schema import CouplingType, Issue
from ..physics.fluid.convection import TARGET_SLOT, AdvectiveHeatSource
from ..physics.fluid.definition import fluid_materials
from .coupling_term import CouplingTerm
from .definitions import HEAT_TYPE_NAME, writes_heat_source

CONVECTION_TYPE_NAME = 'fluid_heat_convection'   # = physics.fluid.selftests.CONVECTION_TYPE_NAME (not imported)
FLUID_TYPE_NAME = 'laminar_flow'                 # = physics.fluid.selftests.FLUID_TYPE_NAME (not imported)

_HEAT_SOURCE_WRITERS = ('em_heat_joule', 'currents_heat_joule',   # the other 'sum' contributors of volumetric_power
                        'magnetic_heat_induction')


def _create_convection(ctx, coupling_spec, values, every):
    """'sum' contributor of fluid u (u_x the nominal source field) and heat T: re-transfers when either solved."""
    return CouplingTerm(coupling_spec.source, 'u_x', coupling_spec.target, TARGET_SLOT,
                        AdvectiveHeatSource(coupling_spec.name), every=every, name=coupling_spec.name,
                        contribution='sum', also_reads=((coupling_spec.source, 'u_y'), (coupling_spec.target, 'T')))


def _validate_convection(vctx, coupling_spec, values):
    """Warning for fluid materials without a heat capacity (no heat carried there)."""
    issues = []
    for name, properties in fluid_materials(vctx):
        if not all(evaluates_to(properties.get(prop), vctx.parameters, lambda value: value > 0.0)
                   for prop in ('rho', 'cp')):
            issues.append(Issue('warning', ('coupling', coupling_spec.name),
                                f"fluid material '{name}' has no heat capacity (rho and cp): the flow carries no heat "
                                'there'))
    return issues


FLUID_HEAT_CONVECTION = CouplingType(
    CONVECTION_TYPE_NAME, 'Convective heat transport (laminar flow -> heat)', FLUID_TYPE_NAME, HEAT_TYPE_NAME,
    summary='Lets the flow carry heat: the heat physics gets the heat transported by the moving fluid.',
    description='Advection of heat by the flow: each transfer evaluates q = -rho cp u . grad T [W/m^3] on the fluid '
                'cells (rho, cp of the material, u of the laminar flow, T the solve-precision temperature of the heat '
                'physics) and contributes it to the heat source volumetric_power (an additive contribution: it sums '
                f"with {', '.join(_HEAT_SOURCE_WRITERS)}); it re-transfers whenever the flow or the temperature "
                'changed. The advective form uses face fluxes: interior faces '
                'between fluid cells with the mean of the two cell velocities and temperatures (second-order central, '
                'oscillation-free for a cell Peclet number rho cp u dx / k < 2), faces to solid cells carry nothing, '
                'outer faces only where heat holds a temperature (that face value is the inflow temperature; '
                'other outer faces carry no advective flux: an outlet with zero gradient). It vanishes for uniform T '
                'whatever the discrete divergence of u, so it conserves energy only to the accuracy of the flow. The '
                'advection is explicit (lagged): a stationary study iterates heat and this coupling (Picard, gain '
                '~ the Peclet number rho cp U L / k over the heat diffusion length / pi; use the study relaxation < 1 '
                'when it oscillates or diverges at Pe >~ 3); transient heat steps with the source of the previous '
                'state (stable for u dt / dx < ~1). Set a Temperature condition on the inflow face of heat for the '
                'inflow temperature.',
    create=_create_convection,
    validate=_validate_convection,
    writes=writes_heat_source,
    reads_target=True)   # also_reads heat T


def register(registry, physics):
    """fluid_heat_convection when laminar_flow and heat are registered (physics: definitions.RegisteredPhysics)."""
    if {FLUID_TYPE_NAME, HEAT_TYPE_NAME} <= physics.names:
        registry.register_coupling(FLUID_HEAT_CONVECTION)
