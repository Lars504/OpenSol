"""MaterialGrid: per-cell float32 material arrays ((nx, ny), (nx, ny, nz) or one per triangle element) on the run
backend, allocated when a physics registers them."""
import numpy as np

from .backend import default_backend_name, get_backend, to_host
from .constants import DTYPE


class MaterialGrid:
    """Per-cell material arrays (float32 on the backend's xp, (nx, ny), or (nx, ny, nz) when nz is given; (nx,) when
    ny is None: one value per element of a triangle mesh, fem.domain), allocated only when a physics registers them via
    add_property (EM: physics.em.materials.register_em_arrays; heat / potential: their material_schema). backend
    None = core.backend.default_backend_name(). park(names) moves arrays the only solver reading them no longer needs
    on the device to host numpy (assign / assign_window write them there; device_array uploads a copy)."""

    def __init__(self, nx: int, ny: int = None, nz: int = None, backend: str = None):
        if ny is None and nz is not None:
            raise ValueError('MaterialGrid: nz needs ny (a per-element grid is 1D)')
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.shape = (nx,) if ny is None else (nx, ny) if nz is None else (nx, ny, nz)
        self._arrays = {}
        self._parked = set()
        self.backend = get_backend(backend or default_backend_name())

    def use_backend(self, name: str) -> None:
        """Switch an array-free grid to backend name; ValueError once a property is allocated on another one."""
        if name == self.backend.name:
            return
        if self._arrays:
            raise ValueError(f"MaterialGrid already holds {self.backend.name} arrays {list(self._arrays)}; "
                             f"cannot switch to backend '{name}'")
        self.backend = get_backend(name)

    def property_names(self) -> tuple:
        return tuple(self._arrays)

    @property
    def parked(self) -> frozenset:
        """Names of the properties held as host numpy arrays (park)."""
        return frozenset(self._parked)

    def park(self, names) -> None:
        """Move properties `names` to host numpy copies (the device arrays are dropped; nothing on the cpu backend).
        Only for arrays whose readers all take device_array or host arrays (the parking solver decides: em3d
        definition parked_material_arrays); writers go through assign / assign_window."""
        if self.backend.xp is np:
            return
        for name in names:
            if name in self._parked:
                continue
            host = to_host(self._array(name))
            setattr(self, name, host)
            self._arrays[name] = host
            self._parked.add(name)

    def device_array(self, name, index=None):
        """Property `name` (or its part `index`) on the backend: the array or a view of it, a new device copy of a
        parked one (only the indexed part is transferred)."""
        array = self._array(name)
        part = array if index is None else array[index]
        return self.backend.xp.asarray(part) if name in self._parked else part

    def add_property(self, name: str, default: float = 0.0):
        if name in self._arrays:
            return self._arrays[name]
        if hasattr(self, name):
            raise ValueError(f"MaterialGrid property name '{name}' clashes with an attribute")
        arr = self.backend.xp.full(self.shape, DTYPE(default), dtype=DTYPE)
        setattr(self, name, arr)
        self._arrays[name] = arr
        return arr

    def add_properties(self, schema: dict):
        for name, default in schema.items():
            self.add_property(name, default)

    def _array(self, name):
        if name not in self._arrays:
            raise AttributeError(
                f"MaterialGrid has no property '{name}'; registered: "
                f"{list(self.property_names())} (register it with add_property first)")
        return getattr(self, name)

    def assign(self, name: str, mask, value) -> None:
        """Write a scalar or full-grid array into property `name` where `mask` (bool, None = all cells) is True (a
        parked property on the host: mask and value transferred)."""
        if name in self._parked:
            self._assign_host(self._array(name), mask, value)
            return
        xp = self.backend.xp
        arr = self._array(name)
        if mask is None:   # numpy reads where=None as False (no write); cupy as no mask
            xp.copyto(arr, self._source(value))
            return
        if not isinstance(mask, xp.ndarray):
            mask = xp.asarray(mask)
        # copyto(where=) avoids the host sync of boolean-mask setitem.
        xp.copyto(arr, self._source(value), where=mask)

    def assign_window(self, name: str, window: tuple, mask, value) -> None:
        """Write a scalar or window-shaped array into property `name` inside the slices `window` where `mask` (bool
        broadcastable to the window, None = all) is True: the per-slab / bounding-box write of 3D builds."""
        if name in self._parked:
            self._assign_host(self._array(name)[window], mask, value)
            return
        xp = self.backend.xp
        view = self._array(name)[window]
        if mask is None:   # as assign: never where=None
            xp.copyto(view, self._source(value))
            return
        mask = xp.broadcast_to(xp.asarray(mask), view.shape)
        xp.copyto(view, self._source(value), where=mask)

    @staticmethod
    def _assign_host(view, mask, value):
        """assign / assign_window into a parked (host) array view."""
        source = DTYPE(value) if not hasattr(value, 'shape') or len(value.shape) == 0 else \
            np.asarray(to_host(value), dtype=DTYPE)
        if mask is None:
            np.copyto(view, source)
            return
        np.copyto(view, source, where=np.broadcast_to(np.asarray(to_host(mask), dtype=bool), view.shape))

    def set_mask(self, mask, **props):
        for name, val in props.items():
            if val is not None:
                self.assign(name, mask, val)

    def _source(self, value):
        """DTYPE scalar, or a float32 backend array (no copy when it already is one) of an array value."""
        xp = self.backend.xp
        if isinstance(value, xp.ndarray):
            return value.astype(DTYPE, copy=False)
        if hasattr(value, 'shape') and len(value.shape) > 0:
            return xp.asarray(value, dtype=DTYPE)
        return DTYPE(value)
