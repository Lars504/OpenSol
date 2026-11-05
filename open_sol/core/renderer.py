"""FieldInfo (label, unit, colormap, mirror parity of one field component), polar relabelling and the PhysicsRenderer
ABC that maps a solver's state to fields over the physical window."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class FieldInfo:
    """Display metadata of one renderable field component (label, unit, colormap, sign convention)."""

    name: str
    label: str
    unit: str = ''
    signed: bool = False     # True -> diverging colormap, symmetric limits about 0
    cmap: str = 'viridis'    # matplotlib colormap name
    primary: bool = False    # default field shown/stored for this physics
    requires_study: str = ''   # non-empty: exists only in that study type (Session hides it elsewhere, validate rejects it)
    parity: str = ''         # mirror image rule (core.mirror.PARITIES); '' = 'even' when unsigned, else unknown
    staggered: str = ''      # axes ('x', 'y', 'xy') along which the samples sit on the cells' hi faces (unfolding)
    plane_signs: str = ''    # 'x_lo:1 y_lo:-1': this field's own E parity on those mirror faces, replacing the
    #                          physics' face signs when unfolding (nlfd channels; core.mirror.plane_signs_text)


def polar_field_infos(field_infos, axisymmetric):
    """field_infos with the labels of in-plane components (names ending _x / _y / _xx / _yy, labels ending in the
    same ' x' / ' y' / ' xx' / ' yy') named radial / angular for a polar grid (names unchanged: the grid frame)."""
    angle = 'theta' if axisymmetric else 'phi'
    words = {'x': 'radial (r)', 'y': f'angular ({angle})', 'xx': 'rr (radial)', 'yy': f'{angle}{angle} (angular)'}
    relabelled = {}
    for name, info in field_infos.items():
        suffix = name.rpartition('_')[2]
        if suffix in words and info.label.endswith(f' {suffix}'):
            info = replace(info, label=f'{info.label[:-len(suffix)]}{words[suffix]}')
        relabelled[name] = info
    return relabelled


def primary_field_info(field_infos):
    """The FieldInfo flagged primary among field_infos (an iterable), else the first one; None when empty."""
    field_infos = tuple(field_infos)
    return next((info for info in field_infos if info.primary), field_infos[0] if field_infos else None)


class PhysicsRenderer(ABC):
    """Maps a solver's state to 2D fields over the physical window (PML ring trimmed)."""

    name: str = ''
    field_infos: dict = {}   # component -> FieldInfo, insertion-ordered (class attribute)

    @property
    def components(self) -> tuple:
        return tuple(self.field_infos)

    def field_info(self, component) -> FieldInfo:
        try:
            return self.field_infos[component]
        except KeyError:
            raise KeyError(
                f"{type(self).__name__} has no field '{component}'; "
                f"valid components: {list(self.field_infos)}") from None

    @abstractmethod
    def device_field(self, component):
        """2D float32 backend (cupy, or numpy on the cpu backend) array of `component` over the physical window (may
        be a view)."""

    def physical_field(self, component):
        """Host (numpy) copy of `device_field(component)`."""
        import numpy as np
        from .backend import to_host
        return np.array(to_host(self.device_field(component)))

    def time_label(self):
        return None
