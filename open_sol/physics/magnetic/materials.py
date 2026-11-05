"""Material vocabulary of the magnetic physics (no cupy: the definitions import it): relative permeability arrays
(always), remanence and saturation arrays (magnetostatics, allocated only when a material sets them), the static
conductivity shared
with electric_currents and the magnetic loss mu_dc_loss (magnetic_fields_ac, allocated only when set); the
magnetostatics_3d vocabulary (mu_dc per axis incl. zz, three remanence components, J_s)."""

# mu_dc = 1 by default: every cell takes part in the solve (vacuum), unlike sigma_dc = 0 of electric_currents.
MAGNETIC_SCHEMA = {'mu_dc_xx': 1.0, 'mu_dc_yy': 1.0}

# Remanent flux density B_r [T] per in-plane component (axisymmetric: x = r, y = z) and the saturation polarisation J_s
# [T] of the B-H curve (magnetostatics stage 4, physics/magnetic/saturation.py); optional_material_schema arrays.
REMANENCE_ARRAYS = ('remanence_x', 'remanence_y')
SATURATION_ARRAY = 'saturation_polarization'
OPTIONAL_MAGNETIC_SCHEMA = {name: (0.0, (name,)) for name in (*REMANENCE_ARRAYS, SATURATION_ARRAY)}
# Tabulated B-H curve (a 'table' material property, no array): magnetostatics create hands each material region's
# rows to MagneticPhysics.add_bh_curve.
BH_CURVE_PROPERTY = 'bh_curve'

# magnetic_fields_ac: the electric_currents arrays sigma_dc_xx / _yy (same names and default, so a sigma_dc(heat_T)
# update reaches both) and the imaginary relative permeability (mu = mu_dc - j mu_dc_loss).
CONDUCTIVITY_ARRAYS = frozenset({'sigma_dc_xx', 'sigma_dc_yy'})
MAGNETIC_AC_SCHEMA = {**MAGNETIC_SCHEMA, 'sigma_dc_xx': 0.0, 'sigma_dc_yy': 0.0}
OPTIONAL_MAGNETIC_AC_SCHEMA = {'mu_dc_loss': (0.0, ('mu_dc_loss',))}

# Material property -> MaterialGrid arrays it writes (isotropic entries first, components override).
MAGNETIC_PROPERTY_ARRAYS = {
    'mu_dc': ('mu_dc_xx', 'mu_dc_yy'), 'mu_dc_xx': ('mu_dc_xx',), 'mu_dc_yy': ('mu_dc_yy',),
    'remanence_x': ('remanence_x',), 'remanence_y': ('remanence_y',),
    SATURATION_ARRAY: (SATURATION_ARRAY,),
    'mu_dc_loss': ('mu_dc_loss',),
    'sigma_dc': ('sigma_dc_xx', 'sigma_dc_yy'), 'sigma_dc_xx': ('sigma_dc_xx',), 'sigma_dc_yy': ('sigma_dc_yy',),
}

PERMEABILITY_ARRAYS = frozenset(MAGNETIC_SCHEMA)

# magnetostatics_3d: c = mu_dc per axis (no xx / yy swap: the scalar-potential flux is B itself), the three remanence
# components and J_s (optional, allocated only when a material sets one); isotropic mu_dc also writes mu_dc_zz there.
MAGNETIC_3D_SCHEMA = {'mu_dc_xx': 1.0, 'mu_dc_yy': 1.0, 'mu_dc_zz': 1.0}
REMANENCE_3D_ARRAYS = (*REMANENCE_ARRAYS, 'remanence_z')
OPTIONAL_MAGNETIC_3D_SCHEMA = {name: (0.0, (name,)) for name in (*REMANENCE_3D_ARRAYS, SATURATION_ARRAY)}
MAGNETIC_3D_PROPERTY_ARRAYS = {
    'mu_dc': ('mu_dc_xx', 'mu_dc_yy', 'mu_dc_zz'), 'mu_dc_xx': ('mu_dc_xx',), 'mu_dc_yy': ('mu_dc_yy',),
    'mu_dc_zz': ('mu_dc_zz',), **{name: (name,) for name in (*REMANENCE_3D_ARRAYS, SATURATION_ARRAY)},
}
PERMEABILITY_3D_ARRAYS = frozenset(MAGNETIC_3D_SCHEMA)
