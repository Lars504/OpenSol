"""FixedPatternCsr: the CSR pattern of a sum of element blocks, built once with the data slot of every block entry and
the free-free restriction, so a re-assembly only sums values (constant pattern, explicit zeros kept); scatter_add, the
bincount form of np.add.at. DevicePatternCsr is its cupy twin (plan G5). Host numpy, scipy / core.backend inside
functions."""
import numpy as np

_BATCH_ENTRIES = 20_000_000   # block entries per COO -> CSR batch of the pattern build (bounded peak, as elasticity3d)


def scatter_add(target, indices, values) -> None:
    """target[indices] += values with repeated indices summed (np.add.at) by one bincount over the index window (complex
    values: one per part; a real target refuses them)."""
    indices = np.asarray(indices).reshape(-1)
    if not indices.size:
        return
    low, high = int(indices.min()), int(indices.max()) + 1
    values = np.asarray(values).reshape(-1)
    summed = np.bincount(indices - low, weights=values.real.astype(np.float64, copy=False), minlength=high - low)
    if np.iscomplexobj(values):
        if not np.iscomplexobj(target):
            raise TypeError('scatter_add: complex values into a real target would drop their imaginary part')
        summed = summed + 1j * np.bincount(indices - low, weights=values.imag.astype(np.float64, copy=False),
                                           minlength=high - low)
    target[low:high] += summed


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


class FixedPatternCsr:
    """Canonical CSR pattern (indptr, indices: sorted, no duplicates) of the n x n sum of square element blocks over
    element dofs (m, r) per group, the data slot of every block entry per group (`add`), and for a free mask (n,) the
    restricted pattern of the free rows / columns with its take of the full data (`reduced`). Values are summed by
    bincount into fixed slots, so the pattern never depends on the values (no exact-zero entry is dropped)."""

    def __init__(self, n, group_dofs, free=None):
        import scipy.sparse
        self.n = int(n)
        group_dofs = [np.asarray(dofs, dtype=np.int64) for dofs in group_dofs]
        total, batch, count = None, [], 0

        def flush():
            rows = np.concatenate([np.broadcast_to(dofs[:, :, None], dofs.shape + dofs.shape[1:]).ravel()
                                   for dofs in batch])
            columns = np.concatenate([np.broadcast_to(dofs[:, None, :], dofs.shape[:1] + dofs.shape[1:] * 2).ravel()
                                      for dofs in batch])
            counts = scipy.sparse.coo_matrix((np.ones(rows.size, dtype=np.float32), (rows, columns)),
                                             shape=(self.n, self.n)).tocsr()
            return counts if total is None else total + counts   # counts >= 1: the sum drops no entry

        for dofs in group_dofs:
            batch.append(dofs)
            count += dofs.shape[0] * dofs.shape[1] ** 2
            if count >= _BATCH_ENTRIES:
                total, batch, count = flush(), [], 0
        if batch:
            total = flush()
        if total is None:
            total = scipy.sparse.csr_matrix((self.n, self.n), dtype=np.float32)
        total.sum_duplicates()   # canonical: sorted column indices per row
        self.nnz = int(total.nnz)
        index_dtype = np.int32 if max(self.n, self.nnz) < 2 ** 31 else np.int64
        self.indptr = np.asarray(total.indptr, dtype=index_dtype)
        self.indices = np.asarray(total.indices, dtype=index_dtype)
        rows_of_slots = np.repeat(np.arange(self.n, dtype=np.int64), np.diff(self.indptr))
        slot_keys = rows_of_slots * self.n + self.indices   # ascending: the CSR is row-major and sorted
        self._scatter = []   # per group: (first slot, slots - first slot (m, r, r)) of every block entry
        for dofs in group_dofs:
            keys = dofs[:, :, None] * self.n + dofs[:, None, :]
            slots = np.searchsorted(slot_keys, keys)
            low = int(slots.min()) if slots.size else 0
            span = int(slots.max()) - low + 1 if slots.size else 0
            self._scatter.append((low, span, (slots - low).astype(np.int32 if span < 2 ** 31 else np.int64)))
        self.free = None
        if free is not None:
            free = np.asarray(free, dtype=bool).reshape(-1)
            if free.shape != (self.n,):
                raise ValueError(f'FixedPatternCsr: free mask of {free.size} entries, expected {self.n}')
            self.free = free
            renumber = np.cumsum(free) - 1
            kept = free[rows_of_slots] & free[self.indices]
            self.reduced_take = np.flatnonzero(kept)
            self.n_free = int(free.sum())
            reduced_index = index_dtype if max(self.n_free, self.reduced_take.size) < 2 ** 31 else np.int64
            self.reduced_indices = renumber[self.indices[kept]].astype(reduced_index)
            counts = np.bincount(renumber[rows_of_slots[kept]], minlength=self.n_free)
            self.reduced_indptr = np.concatenate([[0], np.cumsum(counts)]).astype(reduced_index)

    @property
    def index_bytes(self) -> int:
        """Host bytes of the cached index arrays (pattern, scatter maps, restriction)."""
        total = self.indptr.nbytes + self.indices.nbytes + sum(slots.nbytes for _, _, slots in self._scatter)
        if self.free is not None:
            total += self.reduced_take.nbytes + self.reduced_indices.nbytes + self.reduced_indptr.nbytes
        return int(total)

    def zeros(self) -> np.ndarray:
        """Float64 data array (nnz,) to sum blocks into."""
        return np.zeros(self.nnz)

    def add(self, data, group, blocks) -> None:
        """data += the real element blocks (m, r, r) of group index `group` (constructor order; complex blocks raise:
        summed() keeps both parts)."""
        if np.iscomplexobj(blocks):
            raise TypeError('FixedPatternCsr.add takes real blocks (complex: summed)')
        low, span, slots = self._scatter[group]
        if span:
            data[low:low + span] += np.bincount(slots.ravel(), weights=np.asarray(blocks, dtype=np.float64).ravel(),
                                                minlength=span)

    def summed(self, group, blocks) -> np.ndarray:
        """Data (nnz,) of the element blocks (m, r, r) of group index `group` alone, float64 or complex128 as blocks
        (real and imaginary parts by one bincount each)."""
        low, span, slots = self._scatter[group]
        blocks = np.asarray(blocks)
        complex_values = np.iscomplexobj(blocks)
        data = np.zeros(self.nnz, dtype=np.complex128 if complex_values else np.float64)
        if span:
            flat = slots.ravel()
            data[low:low + span] = np.bincount(flat, weights=blocks.real.ravel(), minlength=span)
            if complex_values:
                data[low:low + span] += 1j * np.bincount(flat, weights=blocks.imag.ravel(), minlength=span)
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
