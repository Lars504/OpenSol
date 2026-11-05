"""BoxValues: float32 build values on a full-grid index box (3D z-dependent feature expressions of
BuildContext.eval_field), so no full-grid float array is needed."""
from dataclasses import dataclass

from .backend import array_module_of
from .constants import DTYPE


@dataclass(frozen=True, eq=False)
class BoxValues:
    """`values` (float32 backend array, box-shaped) on the full-grid cells `box` (one slice per axis); defined only
    inside the box, so a consumer reads them under a mask inside it (eval_field: the box is the mask's bounding box)."""
    box: tuple
    values: object

    def window_part(self, window):
        """(slices into a window-shaped array, values view) of the part of the box inside `window` (full-grid
        slices); empty slices when they do not overlap."""
        target, source = [], []
        for box_axis, window_axis in zip(self.box, window):
            start = max(box_axis.start, window_axis.start)
            stop = max(start, min(box_axis.stop, window_axis.stop))
            target.append(slice(start - window_axis.start, stop - window_axis.start))
            source.append(slice(start - box_axis.start, stop - box_axis.start))
        return tuple(target), self.values[tuple(source)]

    def add_masked_to(self, target, mask_window, window) -> None:
        """target += values under mask_window, in place on the box part only (target, mask_window: window-shaped);
        the temporary is box-sized, never window-sized."""
        target_part, source = self.window_part(window)
        if any(part.start == part.stop for part in target_part):
            return
        xp = array_module_of(self.values)
        target[target_part] += xp.where(mask_window[target_part], source.astype(target.dtype, copy=False),
                                        target.dtype.type(0))

    def copy_masked_to(self, target, mask_window, window) -> None:
        """target = values under mask_window (in place, box part only; as add_masked_to)."""
        target_part, source = self.window_part(window)
        if any(part.start == part.stop for part in target_part):
            return
        xp = array_module_of(self.values)
        xp.copyto(target[target_part], source.astype(target.dtype, copy=False), where=mask_window[target_part])

    def window_array(self, window, dtype=DTYPE):
        """Fresh window-shaped array: the values inside the box, 0 outside (window-sized, for consumers whose
        where() over the window allocates one anyway)."""
        array = array_module_of(self.values).zeros(tuple(axis.stop - axis.start for axis in window), dtype=dtype)
        target, source = self.window_part(window)
        array[target] = source
        return array
