"""Thermal material vocabulary: THERMAL_SCHEMA (MaterialGrid arrays and defaults), THERMAL_PROPERTY_ARRAYS and
attach_thermal_properties."""
# rho = cp = 0 by default: a cell without k and capacity has an empty row in both time modes (pinned, ignores Q).
# k_zz is read only on a 3D domain (2D heat has no out-of-plane conduction).
THERMAL_SCHEMA = {
    'k_xx': 0.0, 'k_yy': 0.0, 'k_zz': 0.0, 'k_xy': 0.0,
    'rho': 0.0, 'cp': 0.0,
    'dndT_xx': 0.0, 'dndT_yy': 0.0, 'dndT_zz': 0.0,
    'alpha_xx': 0.0, 'alpha_yy': 0.0,
    'n0_zz': 1.0,
}

# Thermal property -> MaterialGrid arrays it writes; isotropic k expands to the xx, yy and zz arrays, alpha to xx and
# yy, isotropic dndT also to dndT_zz (the out-of-plane index).
THERMAL_PROPERTY_ARRAYS = {
    'k': ('k_xx', 'k_yy', 'k_zz'), 'k_xx': ('k_xx',), 'k_yy': ('k_yy',), 'k_zz': ('k_zz',), 'k_xy': ('k_xy',),
    'rho': ('rho',), 'cp': ('cp',),
    'dndT': ('dndT_xx', 'dndT_yy', 'dndT_zz'), 'dndT_xx': ('dndT_xx',), 'dndT_yy': ('dndT_yy',),
    'dndT_zz': ('dndT_zz',),
    'alpha': ('alpha_xx', 'alpha_yy'), 'alpha_xx': ('alpha_xx',), 'alpha_yy': ('alpha_yy',),
}


def attach_thermal_properties(materials):
    """Register the THERMAL_SCHEMA arrays on a MaterialGrid (no-op for existing ones)."""
    materials.add_properties(THERMAL_SCHEMA)
