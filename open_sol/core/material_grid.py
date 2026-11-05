"""MaterialGrid: per-cell float32 material arrays ((nx, ny), (nx, ny, nz) or one per triangle element) on the run
backend, allocated when a physics registers them."""
from .backend import default_backend_name, get_backend
from .constants import DTYPE


class MaterialGrid:
    """Per-cell material arrays (float32 on the backend's xp, (nx, ny), or (nx, ny, nz) when nz is given; (nx,) when
    ny is None: one value per element of a triangle mesh, fem.domain), allocated only when a physics registers them via
    add_property (EM: physics.em.materials.register_em_arrays; heat / potential: their material_schema). backend
    None = core.backend.default_backend_name()."""

    def __init__(self, nx: int, ny: int = None, nz: int = None, backend: str = None):
        if ny is None and nz is not None:
            raise ValueError('MaterialGrid: nz needs ny (a per-element grid is 1D)')
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.shape = (nx,) if ny is None else (nx, ny) if nz is None else (nx, ny, nz)
        self._arrays = {}
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
        """Write a scalar or full-grid array into property `name` where `mask` (bool, None = all cells) is True."""
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
        xp = self.backend.xp
        view = self._array(name)[window]
        if mask is None:   # as assign: never where=None
            xp.copyto(view, self._source(value))
            return
        mask = xp.broadcast_to(xp.asarray(mask), view.shape)
        xp.copyto(view, self._source(value), where=mask)

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
