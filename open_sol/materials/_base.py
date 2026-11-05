"""Material (library dataclass: EM poles, optical, thermal, mechanical, electrical and fluid property dicts) and
NkTable (tabulated n + j k per wavelength for the nk_fit dispersion fit)."""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class NkTable:
    """Tabulated complex index n + j k (k >= 0 extinction) per vacuum wavelength [m] (ascending) and its source; the
    'nk_fit' dispersion model fits poles to it (physics/em/material_fit)."""

    wavelengths: tuple
    n: tuple
    k: tuple
    reference: str

    def __post_init__(self):
        if not (len(self.wavelengths) == len(self.n) == len(self.k) >= 2):
            raise ValueError(f'NkTable needs >= 2 rows of equal length, got {len(self.wavelengths)}, {len(self.n)}, '
                             f'{len(self.k)}')
        if any(later <= earlier for earlier, later in zip(self.wavelengths, self.wavelengths[1:])):
            raise ValueError('NkTable wavelengths must increase')
        if min(self.wavelengths) <= 0.0 or min(self.n) <= 0.0 or min(self.k) < 0.0:
            raise ValueError('NkTable needs wavelengths > 0, n > 0 and k >= 0')

    @classmethod
    def from_micrometre_rows(cls, rows, reference):
        """NkTable from (wavelength [um], n, k) rows in any order."""
        rows = sorted(rows)
        return cls(tuple(row[0] * 1e-6 for row in rows), tuple(float(row[1]) for row in rows),
                   tuple(float(row[2]) for row in rows), reference)


@dataclass(frozen=True)
class Material:
    """Library material: EM dispersion pole(s), static optical (EM) dict, thermal dict (`cp_` = specific heat),
    mechanical dict (isotropic youngs_modulus / poisson_ratio), electrical dict (DC sigma_dc / eps_dc), fluid dict
    (viscosity) and the ASCII description describe('materials') shows."""

    name: str
    em: Optional[object] = None
    thermal: Optional[dict] = None
    mechanical: Optional[dict] = None
    optical: Optional[dict] = None
    electrical: Optional[dict] = None
    description: str = ''
    summary: str = ''         # GUI tooltip of the library entry (Add material menu, Library combo); '' -> none
    nk_table: Optional[NkTable] = None   # tabulated n, k for the 'nk_fit' dispersion model; None -> not offered
    fluid: Optional[dict] = None         # laminar_flow data (dynamic viscosity); density stays in thermal

    def model_properties(self) -> dict:
        """Model-vocabulary property dict (optical + thermal + mechanical + electrical + fluid items, `cp_` renamed
        `cp`) of builtin floats."""
        properties = {key: float(value) for key, value in (self.optical or {}).items()}
        for key, value in (self.thermal or {}).items():
            properties['cp' if key == 'cp_' else key] = float(value)
        properties.update({key: float(value) for key, value in (self.mechanical or {}).items()})
        properties.update({key: float(value) for key, value in (self.electrical or {}).items()})
        properties.update({key: float(value) for key, value in (self.fluid or {}).items()})
        return properties

    def dispersion_models(self) -> list:
        """EM poles as model dicts ({'model': 'drude'|'lorentz', ...} of builtin floats); [] when em is None."""
        if self.em is None:
            return []
        from ..physics.em.materials import DrudePole, LorentzPole
        poles = self.em if isinstance(self.em, (list, tuple)) else (self.em,)
        models = []
        for pole in poles:
            if isinstance(pole, DrudePole):
                models.append({'model': 'drude', 'omega_p': _float32_value(pole.omega_p),
                               'gamma': _float32_value(pole.gamma),
                               'eps_inf': _float32_value(pole.eps_inf)})
            elif isinstance(pole, LorentzPole):
                models.append({'model': 'lorentz', 'delta_eps': _float32_value(pole.delta_eps),
                               'omega_0': _float32_value(pole.omega_0),
                               'gamma': _float32_value(pole.gamma)})
            else:
                raise TypeError(f"material '{self.name}': unsupported EM pole type "
                                f"{type(pole).__name__}")
        return models


def _float32_value(value) -> float:
    """Builtin float of a float32-stored pole parameter, rounded to float32 precision (1.37e16, not 1.3700000272e16)."""
    return float(f"{float(value):.7g}")
