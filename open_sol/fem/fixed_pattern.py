"""FixedPatternCsr: the CSR pattern of a sum of element blocks, built once with the data slot of every block entry and
the free-free restriction, so a re-assembly only sums values (constant pattern, explicit zeros kept); scatter_add, the
bincount form of np.add.at. DevicePatternCsr is its cupy twin (plan G5). Host numpy, scipy / core.backend inside
functions."""
# Vector maps with interleaved dofs (block d + component) take block=components: the pattern is built on the scalar
# element dofs and every scalar entry expanded to a dense block x block block (block^2 fewer COO entries and slot
# searches). kept_pattern keeps such patterns per owner (a mesh, weakly), so later sweep values on a reused mesh skip
# the build.
# FixedPatternCsr.add on the host: with numba a row-parallel sum (next-gains N2; each thread owns whole rows, so it is
# race-free and in a fixed order: the per-group (element, local row) pairs sorted by global row once), else bincount.
# FixedPatternCsr.add_elements (no kept slot map): with numba a serial scatter that finds each dof pair's scalar slot by
# a binary search in its row and adds the block x block entries there (no slot arrays, cost ~ the block entries); the
# numpy form sums one bincount over the chunk's slot window, which spans most of the matrix when a chunk's rows are far
# apart (P2 numbering: nodes first, edge dofs last; 72k P2 tetrahedra: 9.7 s against ~0.5 s).
# The numba loops (row sum, block scatter, block expansion, restriction) are core.kernels.host_twin twins of their
# numpy forms.
import copy
import hashlib
import weakref

import numpy as np

_BATCH_ENTRIES = 20_000_000   # block entries per COO -> CSR batch of the pattern build (bounded peak, as elasticity3d)
ROW_SUM_MIN_ENTRIES = 200_000  # block entries of an add from which the numba row sum runs (below: one bincount)
_ROW_SUM_CALLED = set()       # (data, blocks, slots) dtypes the numba row sum ran on in this process (later small adds
                              # take it too)
EXPAND_ENTRIES = 4_000_000    # expanded entries per batch of the block expansion (int64 temporaries ~ 3 x 8 B each)
# pattern entries from which the numba expansion / restriction loops run although numba is not imported yet (its
# import + cache load ~0.2-0.3 s against ~15 ns per entry saved); below, they run only once numba is loaded anyway
PATTERN_KERNEL_COLD_ENTRIES = 20_000_000
# pattern entries from which add_elements runs the numba scatter although numba is not imported yet: an assembly of
# such a pattern sums ~2.5 block entries per pattern entry at ~40 ns each by bincount (+ its window per chunk)
SCATTER_COLD_ENTRIES = 2_000_000
KEPT_BASE_PATTERNS = 2       # slot-free patterns kept per owner (kept_pattern; a PML shell splits the solid in two)
KEPT_GROUPED_PATTERNS = 1     # patterns with group slot maps / free restriction kept per owner (the latest)
_KEPT = weakref.WeakKeyDictionary()   # owner -> ({digest: base pattern}, {key: grouped pattern}), oldest first


def _add_bincount(window, indices, values, what) -> None:
    """window (a view) += values summed at the flat indices within it: one bincount per real / imaginary part; a real
    window refuses complex values."""
    values = np.asarray(values)
    complex_values = np.iscomplexobj(values)
    if complex_values and not np.iscomplexobj(window):
        raise TypeError(f'{what}: complex values into a real target would drop their imaginary part')
    window += np.bincount(indices, weights=values.real.astype(np.float64, copy=False).ravel(), minlength=window.size)
    if complex_values:
        window += 1j * np.bincount(indices, weights=values.imag.astype(np.float64, copy=False).ravel(),
                                   minlength=window.size)


def scatter_add(target, indices, values) -> None:
    """target[indices] += values with repeated indices summed (np.add.at) by one bincount over the index window (complex
    values: one per part; a real target refuses them)."""
    indices = np.asarray(indices).reshape(-1)
    if indices.size:
        low = int(indices.min())
        _add_bincount(target[low:int(indices.max()) + 1], indices - low, values, 'scatter_add')


def scatter_rows(n_rows, rows, values) -> np.ndarray:
    """Float64 (n_rows, ...) sums of real values (len(rows), ...) by row (np.add.at over a leading axis): one bincount
    per trailing column."""
    rows = np.asarray(rows).reshape(-1)
    values = np.asarray(values, dtype=np.float64)
    trailing = values.shape[1:]
    columns = values.reshape(rows.size, int(np.prod(trailing, dtype=np.int64)))
    summed = np.empty((n_rows, columns.shape[1]))
    for column in range(columns.shape[1]):
        summed[:, column] = np.bincount(rows, weights=columns[:, column], minlength=n_rows)
    return summed.reshape((n_rows,) + trailing)


def _summed_pattern(n, group_dofs) -> tuple:
    """Canonical (indptr, indices) int64 of the n x n sum of square element blocks over element dofs (m, r) per group
    (COO -> CSR in batches of _BATCH_ENTRIES block entries; a larger group is split by rows)."""
    import scipy.sparse
    total, batch, count = None, [], 0
    pieces = []
    for dofs in group_dofs:
        step = max(1, _BATCH_ENTRIES // max(dofs.shape[1] ** 2, 1))
        pieces.extend(dofs[start:start + step] for start in range(0, len(dofs), step))

    def flush():
        rows = np.concatenate([np.broadcast_to(dofs[:, :, None], dofs.shape + dofs.shape[1:]).ravel() for dofs in batch])
        columns = np.concatenate([np.broadcast_to(dofs[:, None, :], dofs.shape[:1] + dofs.shape[1:] * 2).ravel()
                                  for dofs in batch])
        counts = scipy.sparse.coo_matrix((np.ones(rows.size, dtype=np.float32), (rows, columns)), shape=(n, n)).tocsr()
        return counts if total is None else total + counts   # counts >= 1: the sum drops no entry

    for dofs in pieces:
        batch.append(dofs)
        count += dofs.shape[0] * dofs.shape[1] ** 2
        if count >= _BATCH_ENTRIES:
            total, batch, count = flush(), [], 0
    if batch:
        total = flush()
    if total is None:
        total = scipy.sparse.csr_matrix((n, n), dtype=np.float32)
    total.sum_duplicates()   # canonical: sorted column indices per row
    return np.asarray(total.indptr, dtype=np.int64), np.asarray(total.indices, dtype=np.int64)


def _summed_host(n, dofs) -> tuple:
    """numpy form of the row-parallel pattern build (its arguments): _summed_pattern of one element-dof array."""
    return _summed_pattern(int(n), [dofs])


def _summed_numba(numba):
    """Pattern build of one element-dof array (m, r): node -> element lists by a counting pass, then per row (prange)
    the sorted distinct dofs of its elements, counted in a first pass and written in a second."""
    @numba.njit(parallel=True, cache=True)
    def summed(n, dofs):
        m, r = dofs.shape
        starts = np.zeros(n + 1, dtype=np.int64)
        for element in range(m):
            for local in range(r):
                starts[dofs[element, local] + 1] += 1
        for row in range(n):
            starts[row + 1] += starts[row]
        fill = starts[:-1].copy()
        members = np.empty(m * r, dtype=np.int64)
        for element in range(m):
            for local in range(r):
                node = dofs[element, local]
                members[fill[node]] = element
                fill[node] += 1
        lengths = np.zeros(n, dtype=np.int64)
        for prange_row in numba.prange(n):
            row = np.int64(prange_row)
            candidates = np.empty((starts[row + 1] - starts[row]) * r, dtype=np.int64)
            count = 0
            for position in range(starts[row], starts[row + 1]):
                for local in range(r):
                    candidates[count] = dofs[members[position], local]
                    count += 1
            candidates.sort()
            distinct = 0
            for index in range(count):
                if index == 0 or candidates[index] != candidates[index - 1]:
                    distinct += 1
            lengths[row] = distinct
        indptr = np.zeros(n + 1, dtype=np.int64)
        for row in range(n):
            indptr[row + 1] = indptr[row] + lengths[row]
        indices = np.empty(indptr[n], dtype=np.int64)
        for prange_row in numba.prange(n):
            row = np.int64(prange_row)
            candidates = np.empty((starts[row + 1] - starts[row]) * r, dtype=np.int64)
            count = 0
            for position in range(starts[row], starts[row + 1]):
                for local in range(r):
                    candidates[count] = dofs[members[position], local]
                    count += 1
            candidates.sort()
            out = indptr[row]
            for index in range(count):
                if index == 0 or candidates[index] != candidates[index - 1]:
                    indices[out] = candidates[index]
                    out += 1
        return indptr, indices
    return summed


def scalar_pattern(n, group_dofs, numba=None) -> tuple:
    """Canonical (indptr, indices) int64 of the n x n sum of square element blocks over element dofs (m, r) per group:
    the row-parallel numba build (host twin 'fem.pattern_summed'; groups of one width, stacked) once numba is loaded or
    from PATTERN_KERNEL_COLD_ENTRIES / 4 block entries, else COO -> CSR (_summed_pattern); numba True / False forces."""
    import sys
    group_dofs = [np.ascontiguousarray(dofs, dtype=np.int64) for dofs in group_dofs]
    widths = {dofs.shape[1] for dofs in group_dofs if dofs.size}
    entries = sum(dofs.shape[0] * dofs.shape[1] ** 2 for dofs in group_dofs)
    if numba is None:
        numba = len(widths) == 1 and ('numba' in sys.modules or 4 * entries >= PATTERN_KERNEL_COLD_ENTRIES)
    if not numba or len(widths) != 1:
        if numba:
            raise ValueError('scalar_pattern(numba=True) needs element dofs of one width')
        return _summed_pattern(n, group_dofs)
    from ..core.kernels import host_twin
    twin = host_twin('fem.pattern_summed', _summed_host, _summed_numba)
    stacked = np.concatenate([dofs for dofs in group_dofs if dofs.size]) if len(group_dofs) > 1 else group_dofs[0]
    return twin(np.int64(n), stacked)


def _expand_host(scalar_indptr, scalar_indices, block, indptr, indices) -> None:
    """Fill the expanded column indices by numpy, in batches of scalar rows of at most EXPAND_ENTRIES entries."""
    n_scalar = len(scalar_indptr) - 1
    first = 0
    while first < n_scalar:
        target = scalar_indptr[first] + EXPAND_ENTRIES // (block * block)
        last = min(max(int(np.searchsorted(scalar_indptr, target, side='right')) - 1, first + 1), n_scalar)
        low, high = int(indptr[block * first]), int(indptr[block * last])
        rows = np.repeat(np.arange(block * first, block * last), np.diff(indptr[block * first:block * last + 1]))
        offset = np.arange(low, high) - indptr[rows]   # position within the expanded row
        indices[low:high] = block * scalar_indices[scalar_indptr[rows // block] + offset // block] + offset % block
        first = last


def _expand_numba(numba):
    @numba.njit(cache=True)
    def expand(scalar_indptr, scalar_indices, block, indptr, indices):
        for row in range(scalar_indptr.size - 1):
            first, last = scalar_indptr[row], scalar_indptr[row + 1]
            for component in range(block):
                position = indptr[block * row + component]
                for slot in range(first, last):
                    column = block * scalar_indices[slot]
                    for offset in range(block):
                        indices[position] = column + offset
                        position += 1
    return expand


def _restrict_host(indptr, indices, free, renumber, n_free) -> tuple:
    """(row starts, take of the data slots, renumbered columns), int64, of the free rows / columns of a pattern."""
    rows_of_slots = np.repeat(np.arange(free.size, dtype=np.int64), np.diff(indptr))
    kept = free[rows_of_slots] & free[indices]
    counts = np.bincount(renumber[rows_of_slots[kept]], minlength=n_free)
    return np.concatenate([[0], np.cumsum(counts)]), np.flatnonzero(kept), renumber[indices[kept]]


def _restrict_numba(numba):
    @numba.njit(cache=True)
    def restrict(indptr, indices, free, renumber, n_free):
        counts = np.zeros(n_free + 1, dtype=np.int64)
        for row in range(indptr.size - 1):
            if free[row]:
                kept = 0
                for slot in range(indptr[row], indptr[row + 1]):
                    if free[indices[slot]]:
                        kept += 1
                counts[renumber[row] + 1] = kept
        reduced_indptr = np.cumsum(counts)
        take = np.empty(reduced_indptr[-1], dtype=np.int64)
        reduced_indices = np.empty(reduced_indptr[-1], dtype=np.int64)
        position = 0
        for row in range(indptr.size - 1):
            if free[row]:
                for slot in range(indptr[row], indptr[row + 1]):
                    column = indices[slot]
                    if free[column]:
                        take[position] = slot
                        reduced_indices[position] = renumber[column]
                        position += 1
        return reduced_indptr, take, reduced_indices
    return restrict


_PATTERN_LOOPS = {'expand': (_expand_host, _expand_numba), 'restrict': (_restrict_host, _restrict_numba)}


def _pattern_loop(name, entries):
    """The pattern loop 'expand' / 'restrict' for `entries` pattern entries: its host twin (serial numba when
    available), or the numpy form while numba is not imported yet in this process and entries is under
    PATTERN_KERNEL_COLD_ENTRIES (the import would cost more than it saves)."""
    import sys
    from ..core.kernels import host_twin
    loop = host_twin(f'fem.pattern_{name}', *_PATTERN_LOOPS[name])
    return loop if 'numba' in sys.modules or entries >= PATTERN_KERNEL_COLD_ENTRIES else loop.cpu


def _block_expanded(scalar_indptr, scalar_indices, block) -> tuple:
    """(indptr, indices) int64 of the pattern with every scalar entry (i, j) a dense block x block block at rows block i
    + a, columns block j + b (canonical: row block i + a holds the columns of scalar row i, each expanded)."""
    lengths = np.diff(scalar_indptr)
    indptr = (block * block * scalar_indptr[:-1, None] + block * lengths[:, None] * np.arange(block)).ravel()
    indptr = np.append(indptr, block * block * scalar_indptr[-1])
    indices = np.empty(int(indptr[-1]), dtype=np.int64)
    _pattern_loop('expand', indices.size)(np.ascontiguousarray(scalar_indptr, dtype=np.int64),
                                          np.ascontiguousarray(scalar_indices, dtype=np.int64), np.int64(block),
                                          indptr, indices)
    return indptr, indices


def _sum_into(data, low, span, slots, blocks) -> None:
    """data[low:low + span] += bincount of the blocks at slots (relative to low); complex blocks need complex data."""
    if span:
        _add_bincount(data[low:low + span], slots.ravel(), blocks, 'FixedPatternCsr')


def _block_slots(scalar_indptr, keys, n_scalar, block, dofs, check) -> tuple:
    """(int64 data slots (m, block r, block r) of the element blocks over scalar element dofs (m, r), dof pairs outside
    the pattern (counted with check only: searchsorted alone gives them a wrong slot)); keys: ascending row * n_scalar +
    column of the scalar pattern, scalar_indptr its row starts (read for block > 1)."""
    query = dofs[:, :, None] * n_scalar + dofs[:, None, :]
    scalar_slots = np.searchsorted(keys, query)
    missing = 0
    if check and scalar_slots.size:
        missing = (scalar_slots.size if not len(keys) else
                   int(np.count_nonzero(keys[np.minimum(scalar_slots, len(keys) - 1)] != query)))
    if block == 1:
        return scalar_slots, missing
    starts = scalar_indptr[dofs]                                          # (m, r) scalar row starts
    lengths = scalar_indptr[dofs + 1] - starts
    component = np.arange(block)
    # slot(block i + a, block j + b) = block^2 P_i + block a r_i + block (s - P_i) + b, s the scalar slot of (i, j)
    base = (block * block - block) * starts[:, :, None] + block * scalar_slots   # (m, r, r)
    slots = (base[:, :, None, :, None] + (block * lengths)[:, :, None, None, None] * component[:, None, None]
             + component)                                                  # (m, r, block, r, block)
    return slots.reshape(len(dofs), block * dofs.shape[1], block * dofs.shape[1]), missing


def _scatter_host(data, scalar_indptr, keys, n_scalar, block, dofs, blocks) -> int:
    """numpy form of the block scatter (its arguments): slots by one searchsorted, one bincount over their window;
    returns the dof pairs outside the pattern (> 0: nothing was added)."""
    slots, missing = _block_slots(scalar_indptr, keys, int(n_scalar), int(block), dofs, check=True)
    if not missing and slots.size:
        low = int(slots.min())
        _sum_into(data, low, int(slots.max()) - low + 1, slots - low, blocks)
    return missing


def _scatter_numba(numba):
    """Block scatter data[slot(block i + a, block j + b)] += blocks[e, block p + a, block q + b] over the dof pairs
    (i, j) = dofs[e, (p, q)], the scalar slot of each pair by a binary search in row i; returns the pairs outside the
    pattern (skipped; the caller raises)."""
    @numba.njit(cache=True)
    def scatter(data, scalar_indptr, keys, n_scalar, block, dofs, blocks):
        n_local = dofs.shape[1]
        missing = 0
        for element in range(dofs.shape[0]):
            for local_row in range(n_local):
                row = dofs[element, local_row]
                first = scalar_indptr[row]
                length = scalar_indptr[row + 1] - first
                for local_column in range(n_local):
                    key = row * n_scalar + dofs[element, local_column]
                    low, high = first, first + length
                    while low < high:
                        middle = (low + high) >> 1
                        if keys[middle] < key:
                            low = middle + 1
                        else:
                            high = middle
                    if low >= first + length or keys[low] != key:
                        missing += 1
                        continue
                    base = block * block * first + block * (low - first)
                    for row_component in range(block):
                        slot = base + block * row_component * length
                        for column_component in range(block):
                            data[slot + column_component] += blocks[element, block * local_row + row_component,
                                                                    block * local_column + column_component]
        return missing
    return scatter


def _row_sum_host(data, low, slots, blocks, order, starts) -> None:
    """numpy form of the row sum (its arguments): one bincount over the slots."""
    _sum_into(data, int(low), int(slots.max()) + 1 if slots.size else 0, slots, blocks)


def _row_sum_numba(numba):
    """Row sum data[low + slots[p, c]] += blocks[p, c] over the pairs p of each row (prange over rows)."""
    @numba.njit(parallel=True, cache=True)
    def row_sum(data, low, slots, blocks, order, starts):
        width = slots.shape[1]
        for prange_row in numba.prange(starts.size - 1):
            row = np.int64(prange_row)
            for position in range(starts[row], starts[row + 1]):
                pair = order[position]
                for column in range(width):
                    data[low + slots[pair, column]] += blocks[pair, column]
    return row_sum


class FixedPatternCsr:
    """Canonical CSR pattern (indptr, indices: sorted, no duplicates) of the n x n sum of square element blocks over
    element dofs (m, r) per group, the data slot of every block entry per group (`add`), and for a free mask (n,) the
    restricted pattern of the free rows / columns with its take of the full data (`reduced`). Values are summed by
    bincount into fixed slots, so the pattern never depends on the values (no exact-zero entry is dropped).
    block > 1: group dofs are scalar element dofs of the n / block scalar map and the blocks (m, block r, block r) act on
    the interleaved dofs block d + component (module header). slots=False keeps no group slot map (add_elements /
    with_groups only)."""

    def __init__(self, n, group_dofs, free=None, block=1, slots=True):
        self.n, self.block = int(n), int(block)
        if self.block < 1 or self.n % self.block:
            raise ValueError(f'FixedPatternCsr: {self.n} rows are not a multiple of the block size {self.block}')
        group_dofs = [np.asarray(dofs, dtype=np.int64) for dofs in group_dofs]
        n_scalar = self.n // self.block
        scalar_indptr, scalar_indices = scalar_pattern(n_scalar, group_dofs)
        # ascending keys row * n_scalar + column of the scalar pattern (row-major, sorted): searched for slots
        keys = np.repeat(np.arange(n_scalar, dtype=np.int64), np.diff(scalar_indptr)) * n_scalar + scalar_indices
        if self.block == 1:
            indptr, indices = scalar_indptr, scalar_indices
            self._scalar_indptr = self._keys = None   # block 1: recomputed from the pattern when needed
        else:
            indptr, indices = _block_expanded(scalar_indptr, scalar_indices, self.block)
            self._scalar_indptr, self._keys = scalar_indptr, keys
        self.nnz = int(indptr[-1])
        self._index_dtype = np.int32 if max(self.n, self.nnz) < 2 ** 31 else np.int64
        self.indptr = indptr.astype(self._index_dtype)
        self.indices = indices.astype(self._index_dtype)
        del indptr, indices
        self._scatter = self._scatter_maps(group_dofs, keys, check=False) if slots else []
        self._row_orders = {}   # group -> (pairs sorted by row, row starts) of the numba row sum, built at first use
        self.free = None
        if free is not None:
            self._restrict(free)

    def _slot_keys(self) -> tuple:
        """(scalar row starts, ascending scalar slot keys) int64 of the pattern."""
        if self._keys is not None:
            return self._scalar_indptr, self._keys
        indptr = self.indptr.astype(np.int64)
        rows = np.repeat(np.arange(self.n, dtype=np.int64), np.diff(indptr))
        return indptr, rows * self.n + self.indices

    def _element_slots(self, dofs, keys, check) -> np.ndarray:
        """Int64 data slots (m, block r, block r) of the element blocks over scalar element dofs (m, r); check: refuse
        dofs pairs outside the pattern (searchsorted alone would give a wrong slot)."""
        slots, missing = _block_slots(self._scalar_indptr, keys, self.n // self.block, self.block, dofs, check)
        if missing:
            raise ValueError('FixedPatternCsr: element dofs couple entries outside the pattern')
        return slots

    def _scatter_maps(self, group_dofs, keys, check) -> list:
        """Per group (first slot, span, slots - first slot (m, block r, block r)) of every block entry."""
        scatter = []
        for dofs in group_dofs:
            slots = self._element_slots(dofs, keys, check)
            low = int(slots.min()) if slots.size else 0
            span = int(slots.max()) - low + 1 if slots.size else 0
            scatter.append((low, span, (slots - low).astype(np.int32 if span < 2 ** 31 else np.int64)))
        return scatter

    def _restrict(self, free) -> None:
        """Set the free mask (n,) and the restricted pattern of the free rows / columns with its take of the data."""
        free = np.array(free, dtype=bool).reshape(-1)   # own copy: kept patterns outlive the caller's array
        if free.shape != (self.n,):
            raise ValueError(f'FixedPatternCsr: free mask of {free.size} entries, expected {self.n}')
        self.free = free
        renumber = np.cumsum(free) - 1
        self.n_free = int(free.sum())
        reduced_indptr, self.reduced_take, reduced_indices = _pattern_loop('restrict', self.nnz)(
            self.indptr, self.indices, free, renumber, np.int64(self.n_free))
        reduced_index = self._index_dtype if max(self.n_free, self.reduced_take.size) < 2 ** 31 else np.int64
        self.reduced_indices = reduced_indices.astype(reduced_index)
        self.reduced_indptr = reduced_indptr.astype(reduced_index)

    def with_groups(self, group_dofs, free=None):
        """Copy sharing this pattern's index arrays with the slot maps of other groups (scalar element dofs (m, r) each,
        within the pattern) and the free restriction of `free` (None: none)."""
        grouped = copy.copy(self)
        grouped._scatter = self._scatter_maps([np.asarray(dofs, dtype=np.int64) for dofs in group_dofs],
                                              self._slot_keys()[1], check=True)
        grouped._row_orders = {}
        grouped.free = None
        for name in ('reduced_take', 'n_free', 'reduced_indices', 'reduced_indptr'):
            grouped.__dict__.pop(name, None)
        if free is not None:
            grouped._restrict(free)
        return grouped

    @property
    def index_bytes(self) -> int:
        """Host bytes of the cached index arrays (pattern, scalar keys, scatter maps, restriction)."""
        total = self.indptr.nbytes + self.indices.nbytes + sum(slots.nbytes for _, _, slots in self._scatter)
        total += sum(order.nbytes + starts.nbytes for order, starts in self._row_orders.values())
        if self._keys is not None:
            total += self._keys.nbytes + self._scalar_indptr.nbytes
        if self.free is not None:
            total += self.reduced_take.nbytes + self.reduced_indices.nbytes + self.reduced_indptr.nbytes
        return int(total)

    def zeros(self) -> np.ndarray:
        """Float64 data array (nnz,) to sum blocks into."""
        return np.zeros(self.nnz)

    def add(self, data, group, blocks, numba=None) -> None:
        """data += the element blocks (m, r, r) of group index `group` (constructor / with_groups order; complex blocks
        need complex data). numba None: the row sum from ROW_SUM_MIN_ENTRIES entries or once this process ran it on these
        dtypes (the threshold only decides a first compile / cache load), False: bincount, True: row sum."""
        low, span, slots = self._scatter[group]
        blocks = np.asarray(blocks)
        signature = (np.asarray(data).dtype, blocks.dtype, slots.dtype)
        use_kernel = ((span and (slots.size >= ROW_SUM_MIN_ENTRIES or signature in _ROW_SUM_CALLED))
                      if numba is None else bool(numba))
        if use_kernel:
            from ..core.kernels import host_twin
            row_sum = host_twin('fem.block_row_sum', _row_sum_host, _row_sum_numba)
            if row_sum.host_kind != 'numba':
                if numba:
                    raise RuntimeError('FixedPatternCsr.add(numba=True): the numba row sum is not available')
                use_kernel = False
        if not use_kernel:
            _sum_into(data, low, span, slots, blocks)
            return
        if np.iscomplexobj(blocks) and not np.iscomplexobj(data):
            raise TypeError('FixedPatternCsr: complex blocks into real data would drop their imaginary part')
        order, starts = self._row_order(group)
        width = slots.shape[-1]
        row_sum(data, np.int64(low), slots.reshape(-1, width), np.ascontiguousarray(blocks).reshape(-1, width), order,
                starts)
        _ROW_SUM_CALLED.add(signature)

    def _row_order(self, group) -> tuple:
        """(int64 pair indices sorted by global row (stable), int64 starts per touched row) of the group's (element,
        local row) pairs: all columns of a pair lie in one row, so threads owning rows never share a slot."""
        found = self._row_orders.get(group)
        if found is None:
            low, _, slots = self._scatter[group]
            first_slots = slots.reshape(-1, slots.shape[-1])[:, 0].astype(np.int64) + low
            rows = np.searchsorted(self.indptr, first_slots, side='right') - 1
            order = np.argsort(rows, kind='stable').astype(np.int64)
            counts = np.unique(rows[order], return_counts=True)[1]
            starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
            found = self._row_orders[group] = (order, starts)
        return found

    def add_elements(self, data, element_dofs, blocks, check=True, numba=None) -> None:
        """data += element blocks (m, block r, block r) over scalar element dofs (m, r) within the pattern, their slots
        searched per call (no kept slot map: one-off assemblies); complex blocks need complex data. numba None: the
        scatter kernel (always checked) once numba is loaded or from SCATTER_COLD_ENTRIES pattern entries, else one
        bincount over the slot window (check=False there only for element dofs (rows of) those the pattern was built
        from: a pair outside it would give a wrong slot); True / False force one."""
        import sys
        element_dofs = np.ascontiguousarray(element_dofs, dtype=np.int64)
        if not element_dofs.size:
            return
        blocks = np.asarray(blocks)
        scalar_indptr, keys = self._slot_keys()
        use_kernel = ('numba' in sys.modules or self.nnz >= SCATTER_COLD_ENTRIES) if numba is None else bool(numba)
        if use_kernel:
            from ..core.kernels import host_twin
            scatter = host_twin('fem.block_scatter', _scatter_host, _scatter_numba)
            if scatter.host_kind != 'numba':
                if numba:
                    raise RuntimeError('FixedPatternCsr.add_elements(numba=True): the numba scatter is not available')
                use_kernel = False
        if not use_kernel:
            slots = self._element_slots(element_dofs, keys, check=check)
            low = int(slots.min())
            _sum_into(data, low, int(slots.max()) - low + 1, slots - low, blocks)
            return
        if np.iscomplexobj(blocks) and not np.iscomplexobj(data):
            raise TypeError('FixedPatternCsr: complex blocks into real data would drop their imaginary part')
        n_scalar = self.n // self.block
        if int(element_dofs.min()) < 0 or int(element_dofs.max()) >= n_scalar:   # the kernel reads rows unchecked
            raise ValueError('FixedPatternCsr: element dofs couple entries outside the pattern')
        missing = scatter(data, scalar_indptr, keys, np.int64(n_scalar), np.int64(self.block), element_dofs,
                          np.ascontiguousarray(blocks))
        if missing:
            raise ValueError('FixedPatternCsr: element dofs couple entries outside the pattern')

    def summed(self, group, blocks) -> np.ndarray:
        """Data (nnz,) of the element blocks (m, r, r) of group index `group` alone, float64 or complex128 as blocks
        (real and imaginary parts by one bincount each)."""
        data = np.zeros(self.nnz, dtype=np.complex128 if np.iscomplexobj(blocks) else np.float64)
        _sum_into(data, *self._scatter[group], blocks)
        return data

    def full(self, data, copy_pattern=False):
        """Host CSR (n, n) of data on the pattern (shares the cached index arrays unless copy_pattern: callers that
        could modify the pattern in place get copies)."""
        import scipy.sparse
        indices, indptr = (self.indices.copy(), self.indptr.copy()) if copy_pattern else (self.indices, self.indptr)
        return scipy.sparse.csr_matrix((data, indices, indptr), shape=(self.n, self.n))

    def reduced(self, data):
        """Host canonical CSR (n_free, n_free) of the free rows / columns of data (the shared restricted pattern)."""
        import scipy.sparse
        if self.free is None:
            raise ValueError('FixedPatternCsr.reduced needs the free mask of the constructor')
        return scipy.sparse.csr_matrix((data[self.reduced_take], self.reduced_indices, self.reduced_indptr),
                                       shape=(self.n_free, self.n_free))

    def to_device(self):
        """DevicePatternCsr of this pattern (gpu backend)."""
        return DevicePatternCsr(self)


def _digest(*arrays) -> str:
    """Hex digest of the dtypes, shapes and bytes of arrays."""
    hasher = hashlib.blake2b(digest_size=16)
    for array in arrays:
        array = np.ascontiguousarray(array)
        hasher.update(f'{array.dtype.str}{array.shape}'.encode())
        hasher.update(array.data)
    return hasher.hexdigest()


def _keep(store, key, value, limit) -> None:
    """store[key] = value as the newest entry, the oldest dropped beyond limit entries."""
    store.pop(key, None)
    store[key] = value
    while len(store) > limit:
        store.pop(next(iter(store)))


def kept_pattern(owner, n, element_dofs, block=1, groups=None, free=None):
    """FixedPatternCsr(n, [element_dofs], block=block) without slot maps, kept per owner (weakly: a mesh lives across the
    sweep values that reuse it) and inputs; with `groups` (scalar element dofs per group) or `free` a with_groups copy,
    the latest one kept too. An owner that is not weak-referenceable keeps nothing."""
    element_dofs = np.asarray(element_dofs, dtype=np.int64)
    try:
        bases, grouped = _KEPT.setdefault(owner, ({}, {}))
    except TypeError:
        bases, grouped = {}, {}
    digest = _digest(np.array([n, block], dtype=np.int64), element_dofs)
    base = bases.get(digest)
    if base is None:
        base = FixedPatternCsr(n, [element_dofs], block=block, slots=False)
    _keep(bases, digest, base, KEPT_BASE_PATTERNS)
    if groups is None and free is None:
        return base
    groups = [element_dofs] if groups is None else [np.asarray(dofs, dtype=np.int64) for dofs in groups]
    key = digest + _digest(np.zeros(0, dtype=bool) if free is None else np.asarray(free, dtype=bool), *groups)
    found = grouped.get(key)
    if found is None:
        found = base.with_groups(groups, free=free)
    _keep(grouped, key, found, KEPT_GROUPED_PATTERNS)
    return found


def kept_pattern_if_any(owner, n, element_dofs, block=1):
    """The slot-free pattern kept_pattern(owner, n, element_dofs, block) holds already, None when it holds none (nothing
    is built: callers whose terms fit an assembled K's pattern, else their own COO -> CSR)."""
    try:
        state = _KEPT.get(owner)
    except TypeError:
        return None
    if state is None:
        return None
    return state[0].get(_digest(np.array([n, block], dtype=np.int64), np.asarray(element_dofs, dtype=np.int64)))


class DevicePatternCsr:
    """Device (cupy) twin of a FixedPatternCsr: the per-group scatter maps as global data slots, the full and free-free
    patterns and the free take on the GPU; blocks summed by cupyx.scatter_add (atomic, no host sync: cupy.bincount syncs
    on max(x) per call), CSRs as cupyx. The host pattern (`host`) keeps the host index arrays."""

    def __init__(self, pattern):
        from ..core.backend import cp
        self.host = pattern
        self.n, self.nnz = pattern.n, pattern.nnz
        slot_dtype = np.int32 if self.nnz < 2 ** 31 else np.int64
        self.indptr = cp.asarray(pattern.indptr)
        self.indices = cp.asarray(pattern.indices)
        self._slots = [cp.asarray((slots.astype(np.int64) + low).astype(slot_dtype).ravel())
                       for low, _, slots in pattern._scatter]
        self.free = pattern.free
        if pattern.free is not None:
            self.n_free = pattern.n_free
            self.reduced_take = cp.asarray(pattern.reduced_take.astype(slot_dtype))
            self.reduced_indptr = cp.asarray(pattern.reduced_indptr)
            self.reduced_indices = cp.asarray(pattern.reduced_indices)

    @property
    def device_bytes(self) -> int:
        """Pool bytes of the device index arrays."""
        arrays = [self.indptr, self.indices, *self._slots]
        if self.free is not None:
            arrays += [self.reduced_take, self.reduced_indptr, self.reduced_indices]
        return int(sum(array.nbytes for array in arrays))

    def zeros(self):
        """Float64 device data (nnz,) to sum blocks into."""
        from ..core.backend import cp
        return cp.zeros(self.nnz)

    def add(self, data, group, blocks) -> None:
        """data += the real device element blocks (m, r, r) of group index `group` (constructor order)."""
        import cupyx
        cupyx.scatter_add(data, self._slots[group], blocks.ravel())

    def full(self, data, copy_pattern=False):
        """cupyx CSR (n, n) of device data on the pattern (shared index arrays unless copy_pattern)."""
        from ..core.backend import cupyx_sparse
        indices, indptr = (self.indices.copy(), self.indptr.copy()) if copy_pattern else (self.indices, self.indptr)
        return cupyx_sparse.csr_matrix((data, indices, indptr), shape=(self.n, self.n))

    def reduced_values(self, data):
        """Device values (reduced nnz,) of the free rows / columns of data, in the host reduced pattern's order."""
        if self.free is None:
            raise ValueError('DevicePatternCsr.reduced_values needs the free mask of the host pattern')
        return data[self.reduced_take]

    def reduced(self, data):
        """cupyx canonical CSR (n_free, n_free) of the free rows / columns of device data."""
        from ..core.backend import cupyx_sparse
        return cupyx_sparse.csr_matrix((self.reduced_values(data), self.reduced_indices, self.reduced_indptr),
                                       shape=(self.n_free, self.n_free))
