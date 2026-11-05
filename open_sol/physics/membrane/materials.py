"""Material vocabulary of the membrane physics (no cupy: the definition imports it): the tension tensor [N/m], the
areal density [kg/m^2], the optional in-plane shear tension, loss factor, plate bending stiffness and plate Poisson
ratio, and the coupled tension source slots."""

# Every element takes part unless it is a hole: areal_density <= 0 and every tension component 0 (material + coupled)
# and no bending stiffness.
MEMBRANE_SCHEMA = {'tension_xx': 0.0, 'tension_yy': 0.0, 'areal_density': 0.0}
OPTIONAL_MEMBRANE_SCHEMA = {'tension_xy': (0.0, ('tension_xy',)), 'loss_factor': (0.0, ('loss_factor',)),
                            'bending_stiffness': (0.0, ('bending_stiffness',)),
                            'plate_poisson_ratio': (0.0, ('plate_poisson_ratio',))}

# Material property -> MaterialGrid arrays it writes (isotropic entry first, components override).
MEMBRANE_PROPERTY_ARRAYS = {
    'tension': ('tension_xx', 'tension_yy'), 'tension_xx': ('tension_xx',), 'tension_yy': ('tension_yy',),
    'tension_xy': ('tension_xy',), 'areal_density': ('areal_density',), 'loss_factor': ('loss_factor',),
    'bending_stiffness': ('bending_stiffness',), 'plate_poisson_ratio': ('plate_poisson_ratio',),
}
TENSION_ARRAYS = ('tension_xx', 'tension_yy', 'tension_xy')
DENSITY_ARRAY = 'areal_density'
LOSS_ARRAY = 'loss_factor'
BENDING_ARRAY = 'bending_stiffness'     # present (any material sets it) = plate: Hermite line / C0 interior penalty
POISSON_ARRAY = 'plate_poisson_ratio'
COEFFICIENT_ARRAYS = frozenset((*TENSION_ARRAYS, DENSITY_ARRAY, LOSS_ARRAY, BENDING_ARRAY, POISSON_ARRAY))

# Source slots (new_modules_plan S9: solid -> membrane tension), added to the material tension per element / cell.
COUPLED_TENSION_SLOTS = ('coupled_tension_xx', 'coupled_tension_yy', 'coupled_tension_xy')
