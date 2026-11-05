"""EME sections (host numpy): one slice per cell layer normal to the propagation axis; consecutive slices equal to a
section's first slice within section_tolerance merge into it; a digest of the first slice's rows keys the mode cache."""
import hashlib
import math
from dataclasses import dataclass

import numpy as np

_KEY_BYTES = 16   # blake2b digest of a slice's material rows (the mode-set cache key at one omega)


@dataclass(frozen=True, eq=False)
class Section:
    """Consecutive slices sharing one mode set, solved on the first slice."""
    first_slice: int   # grid index along the propagation axis of the first slice (the mode-solve layer)
    n_slices: int
    start: float       # [m] from the window's lo edge along the propagation axis
    length: float      # [m] sum of the slice widths
    key: bytes         # digest of the first slice's material rows


def slice_key(rows):
    """Cache key of a slice's float64 material rows (n_rows, m)."""
    return hashlib.blake2b(np.ascontiguousarray(rows, dtype=np.float64).tobytes(), digest_size=_KEY_BYTES).digest()


def slice_stack(arrays, propagation_axis):
    """float64 (n_rows, n_along_propagation, n_along_line) stack of full-grid (nx, ny) host arrays."""
    if propagation_axis not in ('x', 'y'):
        raise ValueError(f"propagation_axis must be 'x' or 'y', got {propagation_axis!r}")
    stacked = np.stack([np.asarray(array, dtype=np.float64) for array in arrays])
    return np.ascontiguousarray(stacked if propagation_axis == 'x' else stacked.transpose(0, 2, 1))


def find_sections(slice_rows, widths, window, row_scales, tolerance=0.0):
    """Sections over the window slices (a slice of the propagation axis); slice_rows(index) -> float64 (n_rows, m)
    material rows of slice `index` (read once each, in order), widths = full-grid cell widths [m] along the axis,
    row_scales (n_rows,) = each row's largest |value| over the window; a slice joins the current section while every
    row differs from its first slice by <= tolerance x that row's scale (tolerance 0: identical rows only)."""
    tolerance = float(tolerance)
    if not (math.isfinite(tolerance) and tolerance >= 0.0):
        raise ValueError(f'section_tolerance must be finite and >= 0, got {tolerance!r}')
    first, stop = int(window.start), int(window.stop)
    if stop <= first:
        raise ValueError(f'the window has no slice along the propagation axis ({first}..{stop})')
    widths = np.asarray(widths, dtype=np.float64)
    scale = np.asarray(row_scales, dtype=np.float64)
    scale = np.where(scale > 0.0, scale, 1.0)[:, None]
    reference = np.array(slice_rows(first), dtype=np.float64)
    starts, keys = [first], [slice_key(reference)]
    for index in range(first + 1, stop):
        current = np.asarray(slice_rows(index), dtype=np.float64)
        if tolerance > 0.0:
            differs = bool(np.any(np.abs(current - reference) / scale > tolerance))
        else:
            differs = not np.array_equal(current, reference)
        if differs:
            reference = np.array(current, copy=True)
            starts.append(index)
            keys.append(slice_key(reference))
    sections = []
    position = 0.0
    for begin, end, key in zip(starts, starts[1:] + [stop], keys):
        length = float(np.sum(widths[begin:end]))
        sections.append(Section(first_slice=begin, n_slices=end - begin, start=position, length=length, key=key))
        position += length
    return sections


def stack_sections(stack, widths, window, tolerance=0.0):
    """find_sections of a slice_stack (row scales over the window)."""
    first, stop = int(window.start), int(window.stop)
    scales = np.max(np.abs(stack[:, first:stop, :]), axis=(1, 2)) if stop > first else np.ones(stack.shape[0])
    return find_sections(lambda index: stack[:, index, :], widths, window, scales, tolerance)
