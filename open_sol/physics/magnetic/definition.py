"""Registry entry of physics/magnetic: magnetostatics, magnetic_fields_ac (it shares the mu_dc Params, so the
magnetostatics ones stay the first per name), then magnetostatics_3d. Loaded lazily by the registry (__init__.py never
imports it)."""
from .definition3d import register as _register_3d
from .definition_ac import register as _register_ac
from .definition_magnetostatics import register as _register_magnetostatics


def register(registry):
    """magnetostatics, magnetic_fields_ac, then magnetostatics_3d (type, template, selftests each)."""
    _register_magnetostatics(registry)
    _register_ac(registry)
    _register_3d(registry)
