"""Solid mechanics package: re-exports SolidMechanicsPhysics (2D), SolidMechanics3DPhysics, the elastic systems, field
sets, renderers and mechanical schema."""
from .fields import SOURCE_SLOTS, MechanicsFields2D
from .fields3d import SOURCE_SLOTS_3D, MechanicsFields3D
from .materials import MECHANICAL_PROPERTY_ARRAYS, MECHANICAL_SCHEMA, lame_coefficients
from .renderer import SolidRenderer
from .renderer3d import SolidRenderer3D
from .solver import SolidMechanicsPhysics
from .solver3d import SolidMechanics3DPhysics
from .solver_common import ElasticSystemBase, SolidMechanicsBase, SupportRegion
from .system import ElasticSystem
from .system3d import ElasticSystem3D

__all__ = [
    'SolidMechanicsBase', 'SolidMechanicsPhysics', 'SolidMechanics3DPhysics', 'SupportRegion',
    'SOURCE_SLOTS', 'SOURCE_SLOTS_3D',
    'ElasticSystemBase', 'ElasticSystem', 'ElasticSystem3D', 'MechanicsFields2D', 'MechanicsFields3D',
    'SolidRenderer', 'SolidRenderer3D',
    'MECHANICAL_SCHEMA', 'MECHANICAL_PROPERTY_ARRAYS', 'lame_coefficients',
]
