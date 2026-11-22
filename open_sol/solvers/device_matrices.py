"""Device copies of host sparse matrices for GPU products (DeviceMatrix: real values kept float64, a symmetric matrix as
its strict lower triangle + diagonal, a diagonal one as its vector) and their pool bytes for the memory estimators.
numpy only at module level (cupy, cupyx, scipy inside the functions)."""
# Real stays real: cupyx `real_csr @ complex_dense` casts the matrix to complex128 on every product (cusparse
# _cast_common_type: a 20 B per entry copy of a 12 B per entry matrix, each call), so a real matrix is applied to a
# complex operand through its float64 view: complex (n,) -> float64 (n, 2), C-contiguous complex (n, k) -> (n, 2 k)
# (real and imaginary parts as interleaved columns), one SpMM that reads the matrix once for both parts, then viewed
# back. The operand's column-major copy (cupyx SpMM takes F-order operands) and the complex result are vector-sized
# temporaries (device_product_bytes). A complex matrix applied to a float64 operand casts only the operand.
# Symmetric (B^T = B, unconjugated: complex symmetric counts): the strict lower triangle L (CSR) and the diagonal d,
# B X = L X + L^T X + d X, L^T read through cuSPARSE's transposed product on L's arrays (a CSC view, no copy). The
# transposed SpMV scatters with atomics and is slower than an untransposed one (unmeasured here): worth it beside a
# factor solve, not inside a tight stencil loop. bilinear(u, w) = u^T B w needs untransposed products only (u^T L^T w =
# w^T L u). Symmetry is probed by SYMMETRY_PROBES seeded random vectors before the upload (u^T B v - v^T B u relative
# to |u| |B v|; ValueError when it fails). One asymmetric pair of size delta among n rows is diluted by the vector
# norms to ~delta / (n |B|), so the probe guards against a wrong storage request (a non-symmetric operator), not
# against round-off-level asymmetry: callers ask for it only for matrices symmetric by construction (FEM masses).
import numpy as np

from .cudss_factor import canonical_csr, csr_rows

SYMMETRY_TOLERANCE = 1e-10   # |u^T B v - v^T B u| <= this x |u| |B v| (as eigenfrequency's mass probe)
SYMMETRY_PROBES = 2          # random probe vectors (pairs compared: the first against the second)
_PROBE_SEED = 29


def device_matrix(matrix, symmetric=False, real=None) -> 'DeviceMatrix':
    """DeviceMatrix of a host scipy sparse matrix (or a 1D array: a diagonal). real None: detected (a real dtype, or a
    complex one whose imaginary parts are all zero, counts as real), True: float64 storage (ValueError for nonzero
    imaginary parts), False: complex128 storage. symmetric: probed (ValueError when B^T != B) and stored as its strict
    lower triangle + diagonal. A matrix with off-diagonal entries absent is kept as its diagonal vector."""
    return DeviceMatrix(matrix, symmetric=symmetric, real=real)


def device_matrix_bytes(n_unknowns, entries, symmetric, real, diagonal=False) -> int:
    """Pool bytes of device_matrix for `entries` stored entries of the full matrix (diagonal included) on n_unknowns
    rows: values float64 (real) or complex128, int32 indices / indptr; symmetric: the strict lower triangle's CSR + the
    diagonal vector; diagonal: the diagonal vector alone."""
    n_unknowns, entries = int(n_unknowns), int(entries)
    value = 8 if real else 16
    if diagonal:
        return value * n_unknowns
    indptr = 4 * (n_unknowns + 1)
    if symmetric:
        return (value + 4) * max((entries - n_unknowns) // 2, 0) + indptr + value * n_unknowns
    return (value + 4) * entries + indptr


def device_product_bytes(n_unknowns, columns=1, symmetric=False, real=True, complex_operand=True) -> int:
    """Transient pool bytes at the peak of one DeviceMatrix product of an (n, columns) operand, the result included
    (a full or diagonal storage: the result; symmetric: + one L^T X or d X temporary beside the summed product; a real
    storage on a complex operand: + its column-major float64 parts, freed before the complex result is written)."""
    vector = (16 if complex_operand or not real else 8) * int(n_unknowns) * max(int(columns), 1)
    if real and complex_operand:
        return vector * (3 if symmetric else 2)
    return vector * (2 if symmetric else 1)


class DeviceMatrix:
    """Device copy of a host sparse matrix (device_matrix): `@` on cupy (n,) / (n, k) operands of float64 or
    complex128 (module header: a real storage applies to complex operands without a complex copy of the matrix),
    bilinear(left, right) the (k,) column sums of left * (M right) by untransposed products only; shape, dtype (the
    storage dtype), storage ('full', 'lower' or 'diagonal'), real, symmetric, nbytes."""

    def __init__(self, matrix, symmetric=False, real=None):
        import cupy as cp
        import cupyx.scipy.sparse
        import scipy.sparse
        if not scipy.sparse.issparse(matrix):   # a diagonal given as its values
            values = np.asarray(matrix)
            if values.ndim != 1:
                raise ValueError(f'device_matrix needs a scipy sparse matrix or a 1D diagonal, got shape {values.shape}')
            matrix = scipy.sparse.diags(values, 0, format='csr')
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f'device_matrix needs a square matrix, got shape {matrix.shape}')
        csr = canonical_csr(matrix)
        self.shape = tuple(int(size) for size in csr.shape)
        n_rows = self.shape[0]
        self.real = _real_storage(csr.data, real)
        self.dtype = np.dtype(np.float64 if self.real else np.complex128)
        self.symmetric = bool(symmetric)
        diagonal_only = csr.nnz <= n_rows and np.array_equal(csr.indices, csr_rows(csr.indptr, csr.indices.dtype))
        if self.symmetric and not diagonal_only:
            _check_symmetric(csr)
        self._lower = self._upper = self._full = None
        self._diagonal = None
        if diagonal_only:
            self.storage = 'diagonal'
            self._diagonal = cp.asarray(_values(csr.diagonal(), self.real))
        elif self.symmetric:
            self.storage = 'lower'
            lower = scipy.sparse.tril(csr, k=-1, format='csr')
            lower.sum_duplicates()
            self._lower = _uploaded(cupyx.scipy.sparse, lower, self.real)
            self._upper = self._lower.T   # a CSC view of the same arrays (transposed cuSPARSE product)
            self._diagonal = cp.asarray(_values(csr.diagonal(), self.real))
        else:
            self.storage = 'full'
            self._full = _uploaded(cupyx.scipy.sparse, csr, self.real)
        arrays = [self._diagonal] + [part for matrix_part in (self._lower, self._full) if matrix_part is not None
                                     for part in (matrix_part.data, matrix_part.indices, matrix_part.indptr)]
        self.nbytes = sum(int(array.nbytes) for array in arrays if array is not None)

    def __matmul__(self, operand):
        operand = self._checked(operand)
        if self.storage == 'diagonal':
            return (self._diagonal if operand.ndim == 1 else self._diagonal[:, None]) * operand
        return self._applied(self._product, operand)

    def bilinear(self, left, right):
        """Column sums of left * (M right) for (n, k) blocks ((k,); (n,) vectors: a 0-d array), unconjugated (pass
        conj(left) for a Hermitian form); a lower storage by untransposed L products only: u^T L^T w = w^T L u."""
        import cupy as cp
        left, right = self._checked(left), self._checked(right)
        if left.shape != right.shape:
            raise ValueError(f'DeviceMatrix.bilinear: left {left.shape} and right {right.shape} differ')
        diagonal = None if self._diagonal is None else (
            self._diagonal if left.ndim == 1 else self._diagonal[:, None])
        if self.storage == 'diagonal':
            return cp.sum(left * diagonal * right, axis=0)
        if self.storage == 'full':
            return cp.sum(left * (self @ right), axis=0)
        def lower(operand):
            return self._lower @ operand
        return (cp.sum(left * self._applied(lower, right), axis=0) + cp.sum(right * self._applied(lower, left), axis=0)
                + cp.sum(left * diagonal * right, axis=0))

    def _applied(self, apply, operand):
        """apply(operand) of a sparse part; a complex operand of a real storage through its float64 view (module
        header): (n,) -> (n, 2), (n, k) -> (n, 2 k), made column-major once (the L and L^T SpMMs share it)."""
        import cupy as cp
        if not (self.real and operand.dtype.kind == 'c'):
            return apply(operand)
        operand = cp.ascontiguousarray(operand)
        n_rows = self.shape[0]
        parts = cp.asfortranarray(operand.view(np.float64).reshape(n_rows, -1))
        products = apply(parts)
        del parts
        result = cp.empty(operand.shape, dtype=np.complex128)
        result.view(np.float64).reshape(n_rows, -1)[...] = products
        return result

    def _product(self, operand):
        """M operand in the operand's own layout and dtype rules (full CSR, or L X + L^T X + d X)."""
        if self.storage == 'full':
            return self._full @ operand
        product = self._lower @ operand
        product += self._upper @ operand
        product += (self._diagonal if operand.ndim == 1 else self._diagonal[:, None]) * operand
        return product

    def _checked(self, operand):
        """The cupy operand ((n,) or (n, k), float64 or complex128; TypeError / ValueError otherwise)."""
        import cupy as cp
        if not isinstance(operand, cp.ndarray):
            raise TypeError(f'DeviceMatrix products need cupy operands, got {type(operand).__name__}')
        if operand.dtype not in (np.float64, np.complex128):
            raise TypeError(f'DeviceMatrix products take float64 or complex128 operands, got {operand.dtype}')
        if operand.ndim not in (1, 2) or operand.shape[0] != self.shape[1]:
            raise ValueError(f'DeviceMatrix of shape {self.shape} cannot multiply an operand of shape {operand.shape}')
        return operand


def _real_storage(data, real) -> bool:
    """Whether values `data` are stored float64 (device_matrix's `real` rule)."""
    is_complex = np.iscomplexobj(data)
    if real is None:
        return not is_complex or not np.any(data.imag)
    if real and is_complex and np.any(data.imag):
        raise ValueError('device_matrix(real=True) got nonzero imaginary parts')
    return bool(real)


def _values(values, real):
    """Host values in the storage dtype (float64: the real parts, checked zero imaginary parts before)."""
    values = np.asarray(values)
    return np.ascontiguousarray(values.real if real else values, dtype=np.float64 if real else np.complex128)


def _uploaded(cupyx_sparse, csr, real):
    """cupyx CSR of a canonical host CSR in the storage dtype (int32 indices), flagged canonical: cupyx then never
    re-sorts it per product. cupyx's (data, indices, indptr) form takes cupy arrays only (host ones raise)."""
    import cupy as cp
    device = cupyx_sparse.csr_matrix((cp.asarray(_values(csr.data, real)),
                                      cp.asarray(csr.indices.astype(np.int32, copy=False)),
                                      cp.asarray(csr.indptr.astype(np.int32, copy=False))), shape=csr.shape)
    device.has_canonical_format = True
    return device


def _check_symmetric(csr) -> None:
    """ValueError unless u^T B v = v^T B u for seeded random vectors (to SYMMETRY_TOLERANCE of |u| |B v|; module
    header)."""
    probes = np.random.default_rng(_PROBE_SEED).standard_normal((csr.shape[0], SYMMETRY_PROBES))
    products = csr @ probes
    for first in range(SYMMETRY_PROBES - 1):
        second = first + 1
        forward = probes[:, first] @ products[:, second]
        backward = probes[:, second] @ products[:, first]
        scale = float(np.linalg.norm(probes[:, first]) * np.linalg.norm(products[:, second]))
        if not abs(forward - backward) <= SYMMETRY_TOLERANCE * scale:
            raise ValueError(f'device_matrix(symmetric=True) needs a symmetric matrix (B^T = B): u^T B v - v^T B u = '
                             f'{abs(forward - backward):.3g}, |u| |B v| = {scale:.3g}')
