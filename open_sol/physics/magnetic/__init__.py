"""2D magnetostatics (MagneticPhysics on the DiffusionSystem) and time-harmonic magnetic fields (MagneticACPhysics,
host SuperLU). Names load on first access, so importing a definition module (vocabulary, renderer field infos) pulls
in no solver or cupy code through this package."""
import importlib

_EXPORTS = {
    'MagneticPhysics': '.solver', 'CoilRegion': '.solver', 'face_drive_terms': '.solver',
    'face_divergence_source': '.solver',
    'MagneticACPhysics': '.solver_ac', 'ACCoilRegion': '.solver_ac',
    'MagneticPhysics3D': '.solver3d',
    'MagneticFields2D': '.fields', 'MagneticFieldsAC2D': '.fields', 'MagneticFields3D': '.fields',
    'MagneticRenderer': '.renderer', 'MagneticACRenderer': '.renderer', 'MagneticRenderer3D': '.renderer',
    'MAGNETIC_SCHEMA': '.materials', 'MAGNETIC_PROPERTY_ARRAYS': '.materials',
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_name, __name__), name)
