"""Material library: REGISTRY of the library Materials, get, effective_material_data (library merge, 'crystal' entry,
'nk_fit' pole expansion) and fitted_library_poles."""
import functools

import numpy as np

from ._base import Material, NkTable
from .gold import GOLD
from .silver import SILVER
from .aluminum import ALUMINUM
from .silicon import SILICON
from .silica import SILICA
from .lithium_niobate import LITHIUM_NIOBATE_X_CUT, LITHIUM_NIOBATE_Z_CUT
from .air import AIR
from .water import WATER
from .heavy_water import HEAVY_WATER
from .gst import GST_225_AMORPHOUS, GST_225_CRYSTALLINE


REGISTRY = {
    m.name: m for m in (GOLD, SILVER, ALUMINUM, SILICON, SILICA, LITHIUM_NIOBATE_Z_CUT, LITHIUM_NIOBATE_X_CUT, AIR,
                        WATER, HEAVY_WATER, GST_225_AMORPHOUS, GST_225_CRYSTALLINE)
}


def get(name: str) -> Material:
    if name not in REGISTRY:
        raise KeyError(f"unknown material '{name}'")
    return REGISTRY[name]


def effective_material_data(library, properties, dispersion):
    """Merged (properties, dispersion): library model_properties(), then the properties of a 'crystal' entry
    (crystal_tensors.crystal_properties: sagittal or full by its plane; removed from the poles), then the non-None
    `properties`; dispersion None -> library poles;
    library None / '' = none; an 'nk_fit' entry is expanded (_expand_nk_fit). ValueError: unknown library, wrong
    property/dispersion types, a malformed crystal entry, a failed nk_fit."""
    if not isinstance(properties, dict):
        raise ValueError(f'properties must be a dict, got {type(properties).__name__}')
    if dispersion is not None and not isinstance(dispersion, (list, tuple)):
        raise ValueError(f'dispersion must be a list of pole dicts or None, got {type(dispersion).__name__}')
    merged_properties, poles = {}, []
    if library not in (None, ''):
        if not isinstance(library, str) or library not in REGISTRY:
            raise ValueError(f'unknown library material {library!r}; known: {sorted(REGISTRY)}')
        merged_properties.update(REGISTRY[library].model_properties())
        poles = REGISTRY[library].dispersion_models()
    if dispersion is not None:
        poles = list(dispersion)
    entries = crystal_entries(poles)
    if len(entries) > 1:
        raise ValueError(f'only one crystal entry per material, got {len(entries)}')
    overrides = {}
    if entries:
        from .crystal_tensors import crystal_properties   # lazy: only crystal models load the tensor data
        poles = [pole for pole in poles if not _is_crystal_entry(pole)]
        overrides.update(crystal_properties(entries[0]))
    overrides.update({name: value for name, value in properties.items() if value is not None})
    if any(_is_nk_fit(pole) for pole in poles):
        return _expand_nk_fit(library, overrides, poles)
    merged_properties.update(overrides)
    return merged_properties, poles


def crystal_entry(dispersion):
    """The 'crystal' dict of a MaterialSpec.dispersion list, None without one (or when dispersion is no list)."""
    entries = crystal_entries(dispersion)
    return entries[0] if entries else None


def crystal_entries(dispersion):
    """Every 'crystal' dict of a dispersion list ([] when dispersion is no list)."""
    return [pole for pole in dispersion if _is_crystal_entry(pole)] if isinstance(dispersion, (list, tuple)) else []


def _is_crystal_entry(pole):
    """crystal_tensors.is_crystal_entry without calling into it: every build runs this, and running crystal_tensors
    code would make every selftest depend on that file (model/selftest_cache)."""
    return isinstance(pole, dict) and pole.get('model') == 'crystal'   # = crystal_tensors.CRYSTAL_MODEL


def _is_nk_fit(pole):
    """True for an 'nk_fit' dispersion dict."""
    from ..physics.em.material_fit import NK_FIT_MODEL
    return isinstance(pole, dict) and pole.get('model') == NK_FIT_MODEL


def _expand_nk_fit(library, properties, poles):
    """effective_material_data with the one nk_fit entry replaced by its fitted poles (placed first): the library
    optical dict is dropped (the table replaces it) and eps_inf becomes eps_r when the fit keeps no Drude pole;
    `properties` (non-None overrides) win last."""
    from ..physics.em.material_fit import nk_fit_settings
    fit_entries = [pole for pole in poles if _is_nk_fit(pole)]
    if len(fit_entries) > 1:
        raise ValueError(f'only one nk_fit dispersion entry per material, got {len(fit_entries)}')
    material = REGISTRY.get(library) if isinstance(library, str) else None
    if material is None or material.nk_table is None:
        tabulated = ', '.join(sorted(name for name, entry in REGISTRY.items() if entry.nk_table is not None))
        raise ValueError(f'nk_fit needs a library material with tabulated n, k ({tabulated}), got library '
                         f'{library!r}')
    fit = fitted_library_poles(material.name, *nk_fit_settings(fit_entries[0]))
    merged_properties = {name: value for name, value in material.model_properties().items()
                         if name not in (material.optical or {})}
    if fit.drude is None:
        merged_properties['eps_r'] = fit.eps_inf
    merged_properties.update({name: value for name, value in properties.items() if value is not None})
    return merged_properties, fit.pole_dicts() + [pole for pole in poles if not _is_nk_fit(pole)]


@functools.lru_cache(maxsize=64)
def fitted_library_poles(name, n_lorentz, wavelength_start, wavelength_stop):
    """material_fit.PoleFit of library material `name`'s nk_table over [start, stop] [m] (cached; CPU only)."""
    from ..physics.em.material_fit import fit_drude_lorentz, table_samples
    table = REGISTRY[name].nk_table
    wavelengths, permittivity = table_samples(table.wavelengths, table.n, table.k, wavelength_start, wavelength_stop)
    try:
        return fit_drude_lorentz(wavelengths, permittivity, n_lorentz)
    except (ArithmeticError, RuntimeError, np.linalg.LinAlgError) as exc:
        raise ValueError(f"nk_fit of '{name}' failed: {exc}") from None


__all__ = [
    'Material', 'NkTable', 'fitted_library_poles',
    'GOLD', 'SILVER', 'ALUMINUM', 'SILICON', 'SILICA', 'LITHIUM_NIOBATE_Z_CUT', 'LITHIUM_NIOBATE_X_CUT', 'AIR',
    'WATER', 'HEAVY_WATER', 'GST_225_AMORPHOUS', 'GST_225_CRYSTALLINE',
    'REGISTRY', 'get', 'effective_material_data', 'crystal_entry', 'crystal_entries',
]
