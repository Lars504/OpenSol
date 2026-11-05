"""Heat package: re-exports HeatPhysics, TemperatureField2D, the heat boundaries, thermal schema, HeatRenderer and the
EM <-> heat coupling transforms."""
from .fields import TemperatureField2D
from .boundaries import HeatBoundaryConfig, BCSide
from .materials import (
    THERMAL_SCHEMA,
    THERMAL_PROPERTY_ARRAYS,
    attach_thermal_properties,
)
from .renderer import HeatRenderer
from .solver import HeatPhysics
from .coupling import em_to_heat_power, heat_to_em_dn, heat_to_em_dn_xx, heat_to_em_dn_yy

__all__ = [
    'HeatPhysics',
    'TemperatureField2D',
    'HeatBoundaryConfig', 'BCSide',
    'THERMAL_SCHEMA', 'THERMAL_PROPERTY_ARRAYS', 'attach_thermal_properties',
    'em_to_heat_power', 'heat_to_em_dn', 'heat_to_em_dn_xx', 'heat_to_em_dn_yy',
    'HeatRenderer',
]
