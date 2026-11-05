"""uint16 material index of em_fdtd_3d: per MaterialGrid array a set of value levels (its exact distinct values, or
bins of a continuous array re-quantised to a level budget, representative = member mean), the joint level combinations
ranked stage by stage over x-slab chunks (no full-grid temporaries) and per-id value tables. Array module = the
input arrays' (numpy or cupy)."""
import math

import numpy as np

from ...core.backend import array_module_of
from ...core.constants import DTYPE

_MIN_REDUCIBLE_LEVELS = 256    # an array with more levels may be re-quantised further when the joint count is too large
_MAX_ATTEMPTS = 12             # joint-count retries (each halves the level budget of every reducible array)
_BISECTION_RATIO = 1.1         # bin width search stops when upper / lower < this (bins at most 10 % wider than needed)
_SEARCH_DEPTH = 2.0 ** 24      # smallest bin width tried without a hint = the guaranteed width / this


class _Levels:
    """Value levels of one array: exact sorted distinct values (bins None) or occupied bins of width `delta` above
    `minimum` (sorted int64 bin numbers, representative = member mean, error = max |value - representative|)."""

    def __init__(self, values, bins=None, minimum=0.0, delta=0.0, error=0.0):
        self.values = values
        self.bins = bins
        self.minimum = minimum
        self.delta = delta
        self.error = error

    @property
    def count(self) -> int:
        return int(self.values.size)

    def index(self, values):
        """int64 level of each value (flattened)."""
        flat = values.ravel()
        xp = array_module_of(flat)
        if self.bins is None:
            return xp.searchsorted(self.values, flat)
        return xp.searchsorted(self.bins, _bin_numbers(flat, self.minimum, self.delta))


def _bin_numbers(flat, minimum, delta):
    xp = array_module_of(flat)
    return xp.floor((flat.astype(np.float64) - minimum) / delta).astype(np.int64)


def _merged_unique(parts, limit):
    """Sorted distinct values of the chunk generator's arrays, None as soon as there are more than limit."""
    merged = None
    for part in parts:
        xp = array_module_of(part)
        values = xp.unique(part)
        merged = values if merged is None else xp.unique(xp.concatenate((merged, values)))
        if merged.size > limit:
            return None
    return merged


def _exact_levels(array, chunks, limit):
    values = _merged_unique((array[chunk] for chunk in chunks), limit)
    if values is None:
        return None
    if not bool(array_module_of(values).isfinite(values).all()):   # sorted: -inf first, NaN / inf last
        raise ValueError(f'non-finite material values ({float(values[0])}, {float(values[-1])}) in the em_fdtd_3d '
                         'material arrays')
    return _Levels(values.astype(DTYPE, copy=False))


def _occupied_bins(array, chunks, minimum, delta, limit):
    return _merged_unique((_bin_numbers(array[chunk].ravel(), minimum, delta) for chunk in chunks), limit)


def _binned_levels(array, chunks, budget, delta_hint=None):
    """_Levels of at most `budget` occupied bins of the smallest width found by geometric bisection (a hint from the
    previous rebuild narrows it), representative = mean of the member cells."""
    minimum = min(float(array[chunk].min()) for chunk in chunks)
    maximum = max(float(array[chunk].max()) for chunk in chunks)
    if not (math.isfinite(minimum) and math.isfinite(maximum)):
        raise ValueError(f'non-finite material values ({minimum}, {maximum}) in the em_fdtd_3d material arrays')
    upper = max(maximum - minimum, 1e-30) / max(budget - 1, 1)   # every value in bins 0 .. budget - 1
    lower = upper / _SEARCH_DEPTH if delta_hint is None else min(max(delta_hint / 4.0, upper / _SEARCH_DEPTH), upper)
    if _occupied_bins(array, chunks, minimum, lower, budget) is not None:
        upper = lower
    while upper / lower > _BISECTION_RATIO:
        middle = math.sqrt(upper * lower)
        if _occupied_bins(array, chunks, minimum, middle, budget) is None:
            lower = middle
        else:
            upper = middle
    bins = _occupied_bins(array, chunks, minimum, upper, budget)
    n_levels = int(bins.size)
    xp = array_module_of(array)
    sums = xp.zeros(n_levels, dtype=np.float64)
    counts = xp.zeros(n_levels, dtype=np.float64)
    for chunk in chunks:
        flat = array[chunk].ravel()
        rank = xp.searchsorted(bins, _bin_numbers(flat, minimum, upper))
        sums += xp.bincount(rank, weights=flat.astype(np.float64), minlength=n_levels)
        counts += xp.bincount(rank, minlength=n_levels)
    values = (sums / counts).astype(DTYPE)
    levels = _Levels(values, bins, minimum, upper)
    levels.error = max(float(xp.abs(array[chunk].ravel() - values[levels.index(array[chunk])]).max())
                       for chunk in chunks)
    return levels


def _stage_code(arrays, levels, order, stage_keys, chunk, stage):
    """int64 combination code of the arrays order[:stage + 1] on the chunk: the rank of the previous stage's code among
    its keys times this stage's level count plus this stage's level (ranks stay < the joint limit, so no overflow)."""
    code = levels[order[0]].index(arrays[order[0]][chunk])
    xp = array_module_of(code)
    for position in range(1, stage + 1):
        name = order[position]
        code = (xp.searchsorted(stage_keys[position - 1], code) * levels[name].count
                + levels[name].index(arrays[name][chunk]))
    return code


def _stage_keys(arrays, levels, order, chunks, limit):
    """Sorted distinct codes per stage, None as soon as a stage has more than limit combinations."""
    stage_keys = []
    for stage in range(len(order)):
        keys = _merged_unique((_stage_code(arrays, levels, order, stage_keys, chunk, stage) for chunk in chunks),
                              limit)
        if keys is None:
            return None
        stage_keys.append(keys)
    return stage_keys


def build_material_index(arrays, chunks, material_id, limit, delta_hints):
    """Write the material id of every cell into material_id (uint16, x-slab chunks) and return (tables, errors,
    deltas): float32 per-id values per array name, the max |value - table| of each re-quantised array and its bin
    width (the next rebuild's hint). Arrays with more than `limit` distinct values, or a joint count above `limit`,
    are re-quantised (halving budgets); ValueError when even that fails."""
    levels = {}
    for name, array in arrays.items():
        exact = _exact_levels(array, chunks, limit)
        levels[name] = exact if exact is not None else _binned_levels(array, chunks, limit, delta_hints.get(name))
    for _ in range(_MAX_ATTEMPTS):
        order = [name for name in arrays if levels[name].count > 1]
        stage_keys = _stage_keys(arrays, levels, order, chunks, limit) if order else []
        if stage_keys is not None:
            break
        reducible = [name for name in order if levels[name].count > _MIN_REDUCIBLE_LEVELS]
        if not reducible:
            break
        for name in reducible:
            levels[name] = _binned_levels(arrays[name], chunks, levels[name].count // 2,
                                          levels[name].delta or delta_hints.get(name))
    if stage_keys is None:
        raise ValueError(f'em_fdtd_3d stores materials as a uint16 index: more than {limit} distinct material '
                         f'combinations remain after re-quantising {", ".join(arrays)} (arrays varying together in '
                         f'many independent ways); use fewer field-dependent material properties')
    n_ids = int(stage_keys[-1].size) if order else 1
    xp = array_module_of(material_id)
    for chunk in chunks:
        target = material_id[chunk]
        if order:
            code = _stage_code(arrays, levels, order, stage_keys, chunk, len(order) - 1)
            target[...] = xp.searchsorted(stage_keys[-1], code).astype(np.uint16).reshape(target.shape)
        else:
            target.fill(0)
    level_of_id = {}
    if order:
        key = stage_keys[-1]
        for position in range(len(order) - 1, 0, -1):
            count = levels[order[position]].count
            level_of_id[order[position]] = key % count
            key = stage_keys[position - 1][key // count]
        level_of_id[order[0]] = key
    tables = {}
    for name in arrays:
        values = levels[name].values
        tables[name] = (values[level_of_id[name]] if name in level_of_id
                        else xp.repeat(values[:1], n_ids)).astype(DTYPE)
    errors = {name: levels[name].error for name in arrays if levels[name].bins is not None}
    deltas = {name: levels[name].delta for name in arrays if levels[name].bins is not None}
    return tables, errors, deltas
