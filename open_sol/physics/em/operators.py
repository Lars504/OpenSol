"""Yee difference operators of the 2D FDTD stencil for frequency-domain solvers (A1) and the FDFD Helmholtz assembly;
numpy / scipy or cupy / cupyx. Flat index i*ny + j; 'e' derivatives land on cell centres, 'h' on hi faces."""
# Inputs: grid_cell_widths(domain.grid) and cpml_grading_cell_size(domain). PML stretch: kappa when omega is None
# (exactly as the FDTD kernels fold it), else s = kappa + sigma/(alpha + j omega eps0) in exp(+j omega t). pec / pmc /
# mirror faces are plain non-periodic edges here (their polarisation-dependent outer rows are the caller's:
# pin_image_cells). Bloch k != 0 (F(x + L) = F(x) e^{-j k L}, L = sum of the cell widths, as the FDTD wraps): the 'h'
# wrap entry (row n-1, col 0) carries e^{-j k L}, the 'e' one (row 0, col n-1) e^{+j k L}, so the FDFD operator is no
# longer complex symmetric (lossless and PML-free: Hermitian up to the row scaling). Assumes cupyx.scipy.sparse (cupy
# 13.3) provides coo_matrix, csr_matrix((data, indices, indptr)), identity, kron, diags, issparse, astype, csr @ csr /
# csr + csr, csr @ vector and tocsr (which sums duplicates). Products with a diagonal are never sparse products here:
# _column_scaled / _kept_entries scale or select the stored entries (cuSPARSE SpGEMM workspaces grow with the rows).
import math
from dataclasses import dataclass

import numpy as np

from ...core.constants import EPS0
from .boundaries import cpml_axis_profiles
from .yee_grid import yee_inverse_spacings


def _sparse_module(xp):
    """scipy.sparse for numpy, cupyx.scipy.sparse for cupy (imported lazily); TypeError for another module."""
    module_name = getattr(xp, '__name__', '')
    if module_name == 'numpy':
        import scipy.sparse
        return scipy.sparse
    if module_name == 'cupy':
        import cupyx.scipy.sparse
        return cupyx.scipy.sparse
    raise TypeError(f"xp must be the numpy or the cupy module, got {xp!r}")


def pml_stretch(sigma, kappa, alpha, omega):
    """Complex128 coordinate stretch s = kappa + sigma/(alpha + j omega eps0) where sigma > 0, exactly kappa where
    sigma = 0 (Im s < 0 in exp(+j omega t)); sigma, alpha [S/m], omega [rad/s] > 0."""
    omega = float(omega)
    if not (math.isfinite(omega) and omega > 0.0):
        raise ValueError(f"pml_stretch needs a finite angular frequency omega > 0 [rad/s], got {omega!r}")
    sigma = np.asarray(sigma, dtype=np.float64)
    kappa = np.asarray(kappa, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    if np.any(sigma < 0.0):
        raise ValueError(f"pml_stretch needs sigma >= 0, got min {float(np.min(sigma))!r}")
    return np.where(sigma > 0.0, kappa + sigma / (alpha + 1j * omega * EPS0), kappa + 0j)


def difference_matrix_1d(inverse_spacing, location, periodic, stretch=None, xp=np, sparse=True, wrap_phase=0.0):
    """(n, n) Yee difference with weights w_i = inverse_spacing[i] / stretch[i]: 'e' = backward onto cell centres, 'h' =
    forward onto hi faces. Non-periodic: 'h' row n-1 is zero, 'e' row 0 reads the missing lo face as 0 (w_0 at (0, 0)).
    Periodic: wrap_phase = k L [rad] of a Bloch axis (wrapped sample times e^{-j k L} across the hi end)."""
    wrap_phase = float(wrap_phase)
    if wrap_phase != 0.0 and not periodic:
        raise ValueError(f"difference_matrix_1d: a Bloch wrap_phase ({wrap_phase!r} rad) needs a periodic axis")
    weights = np.asarray(inverse_spacing, dtype=np.float64)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError(f"inverse_spacing must be a non-empty 1D array, got shape {weights.shape}")
    n_cells = weights.size
    if stretch is not None:
        stretch = np.asarray(stretch)
        if stretch.shape != weights.shape:
            raise ValueError(f"stretch shape {stretch.shape} does not match inverse_spacing shape {weights.shape}")
        stretch_dtype = np.complex128 if np.iscomplexobj(stretch) else np.float64
        weights = weights / stretch.astype(stretch_dtype)
    if location == 'e':
        rows = np.arange(0 if periodic else 1, n_cells, dtype=np.int32)
        plus_cols, minus_cols = rows, (rows - 1) % n_cells
    elif location == 'h':
        rows = np.arange(n_cells if periodic else n_cells - 1, dtype=np.int32)
        plus_cols, minus_cols = (rows + 1) % n_cells, rows
    else:
        raise ValueError(f"location must be 'e' (cell centres) or 'h' (hi faces), got {location!r}")
    row_index = np.concatenate((rows, rows)).astype(np.int32)
    col_index = np.concatenate((plus_cols, minus_cols)).astype(np.int32)
    data = np.concatenate((weights[rows], -weights[rows]))
    if wrap_phase != 0.0:
        # 'h' row n-1 reads u[0] one period up (u e^{-j k L}); 'e' row 0 reads face n-1 one period down (e^{+j k L})
        data = data.astype(np.complex128)
        if location == 'h':
            data[n_cells - 1] *= np.exp(-1j * wrap_phase)
        else:
            data[n_cells] *= np.exp(1j * wrap_phase)
    if location == 'e' and not periodic:
        # the kernels read the missing lo face of cell 0 as 0 (ez PMC / hz PEC, like the never-updated hi face), so
        # dx_e W dx_h = -W_e dx_h^T W_h^-1 W dx_h: the FDFD operator is complex symmetric up to a diagonal row scaling
        row_index, col_index = np.append(row_index, 0).astype(np.int32), np.append(col_index, 0).astype(np.int32)
        data = np.append(data, weights[0])
    if sparse:
        sparse_module = _sparse_module(xp)
        if xp is not np:
            data, row_index, col_index = xp.asarray(data), xp.asarray(row_index), xp.asarray(col_index)
        return sparse_module.coo_matrix((data, (row_index, col_index)), shape=(n_cells, n_cells),
                                        dtype=data.dtype).tocsr()
    dense = np.zeros((n_cells, n_cells), dtype=data.dtype)
    np.add.at(dense, (row_index, col_index), data)
    return xp.asarray(dense)


@dataclass(frozen=True, eq=False)
class YeeOperators2D:
    """2D Yee derivatives (nx*ny, nx*ny) on the flat index i*ny + j: dx_e / dy_e act on hi-face samples and land on cell
    centres, dx_h / dy_h act on centre samples (E_z, H_z) and land on hi faces; omega None = real kappa stretch."""
    nx: int
    ny: int
    omega: float | None
    dx_e: object
    dx_h: object
    dy_e: object
    dy_h: object


def axis_stretch(profile, omega):
    """(stretch at cell centres, stretch at hi faces) of one axis: kappa (omega None) or pml_stretch at omega."""
    if omega is None:
        return profile.kappa_e, profile.kappa_h
    return (pml_stretch(profile.sigma_e, profile.kappa_e, profile.alpha_e, omega),
            pml_stretch(profile.sigma_h, profile.kappa_h, profile.alpha_h, omega))


def _kron_identity(matrix_1d, identity_size, identity_first, xp, sparse):
    """kron(I, D) when identity_first else kron(D, I), in the backend and dtype of the 1D matrix D."""
    if sparse:
        sparse_module = _sparse_module(xp)
        identity = sparse_module.identity(identity_size, dtype=matrix_1d.dtype, format='csr')
        factors = (identity, matrix_1d) if identity_first else (matrix_1d, identity)
        return sparse_module.kron(*factors, format='csr')
    identity = xp.eye(identity_size, dtype=matrix_1d.dtype)
    factors = (identity, matrix_1d) if identity_first else (matrix_1d, identity)
    return xp.kron(*factors)


def check_zero_bloch(boundary_config, label):
    """NotImplementedError for a Bloch face with k != 0 (EME: k = 0 periodic only)."""
    if boundary_config.has_bloch_phase:
        raise NotImplementedError(f"{label}: Bloch k != 0 is not supported (bloch_kx = {boundary_config.bloch_kx}, "
                                  f"bloch_ky = {boundary_config.bloch_ky}); use k = 0 (periodic)")


def bloch_wrap_phases(cell_widths, boundary_config):
    """(k_x L_x, k_y L_y) [rad] of the Bloch axes (0 elsewhere), L = the sum of the (dx_cells, dy_cells) widths [m]."""
    dx_cells, dy_cells = cell_widths
    return boundary_config.bloch_phase_angles(float(np.sum(dx_cells)), float(np.sum(dy_cells)))


def axis_difference_weights(cell_widths, boundary_config, pml_dx, axis, omega=None):
    """Host complex128 (w_e, w_h) = inverse spacing / stretch of axis 'x' | 'y', the entries difference_matrix_1d gives
    the 'e' (cell centres) and 'h' (hi faces) differences of yee_operators at omega (the FDFD TFSF incident line)."""
    spacings = yee_inverse_spacings(*cell_widths)
    inverse_e, inverse_h = ((spacings.inv_dx_e, spacings.inv_dx_h) if axis == 'x'
                            else (spacings.inv_dy_e, spacings.inv_dy_h))
    stretch_e, stretch_h = axis_stretch(cpml_axis_profiles(boundary_config, inverse_e.size, axis, pml_dx), omega)
    return (np.asarray(inverse_e / stretch_e, dtype=np.complex128),
            np.asarray(inverse_h / stretch_h, dtype=np.complex128))


def stretched_radii(dx_cells, stretch_e, radius_min):
    """Host (r~ at the nx + 1 x faces, r~ at the nx centres) of a polar x axis starting at radius_min [m]: running sums
    of the widths times the centre-row stretch (complex in a CPML at omega, so r~ enters every metric factor)."""
    widths = np.asarray(dx_cells, dtype=np.float64) * np.asarray(stretch_e)
    faces = float(radius_min) + np.concatenate((np.zeros(1, dtype=widths.dtype), np.cumsum(widths)))
    return faces, faces[:-1] + 0.5 * widths


def stretched_cell_areas(cell_widths, boundary_config, pml_dx, omega, polar_radius_min=None):
    """Host complex128 flat (nx*ny,) dx s_x dy s_y of the cells (polar: times r~_c): D with D A complex symmetric for a
    Helmholtz operator on yee_operators at omega without a Bloch phase (each axis term is -W_e G^T W W_h G)."""
    dx_cells, dy_cells = (np.asarray(widths, dtype=np.float64) for widths in cell_widths)
    stretch_x_e, _ = axis_stretch(cpml_axis_profiles(boundary_config, dx_cells.size, 'x', pml_dx), omega)
    stretch_y_e, _ = axis_stretch(cpml_axis_profiles(boundary_config, dy_cells.size, 'y', pml_dx), omega)
    along_x = dx_cells * stretch_x_e
    if polar_radius_min is not None:   # dx_e / dy_e / dy_h rows carry 1 / r~_c
        along_x = along_x * stretched_radii(dx_cells, stretch_x_e, polar_radius_min)[1]
    return np.outer(along_x, dy_cells * stretch_y_e).astype(np.complex128).ravel()


def _scaled(matrix, row_scale, column_scale, xp, sparse):
    """diag(row_scale) matrix diag(column_scale) (column_scale None = identity) of flat host scales."""
    dtype = np.result_type(matrix.dtype, row_scale.dtype,
                           row_scale.dtype if column_scale is None else column_scale.dtype)
    if not sparse:
        scaled = matrix.astype(dtype) * xp.asarray(row_scale)[:, None]
        return scaled if column_scale is None else scaled * xp.asarray(column_scale)[None, :]
    scaled = matrix.tocsr().astype(dtype)   # a copy: the radii are never 0, so no entry drops out
    scaled.data *= xp.asarray(row_scale, dtype=dtype)[_entry_rows(scaled, xp)]
    if column_scale is not None:
        scaled.data *= xp.asarray(column_scale, dtype=dtype)[scaled.indices]
    return scaled


def yee_operators(cell_widths, boundary_config, pml_dx, omega=None, xp=np, sparse=True, polar_radius_min=None):
    """YeeOperators2D of cell_widths = (dx_cells, dy_cells) [m] with the CPML profiles of boundary_config (pml_dx =
    grading cell size [m]) and its Bloch wrap phases; omega [rad/s] None -> real kappa stretch, > 0 -> pml_stretch.
    polar_radius_min [m]: x = r - polar_radius_min, y = phi [rad] (symmetry 2.16 / 2.18), physical derivatives below."""
    # Polar (integral-form Yee): dx_e -> (1 / r~_c) dx_e r~_f (flux through the arc faces r_f dphi over the sector area
    # r_c dr dphi), dy_e / dy_h -> (1 / r~_c) dy (arc lengths r_c dphi), dx_h unchanged; r~ = stretched_radii (the
    # complex r of a cylindrical PML on x_hi). Every consumer of the ops (operator, face fields) then sees d/dr and
    # (1 / r) d/dphi; the row scale keeps the operator complex symmetric up to diag(cell areas), as graded grids do.
    if omega is not None:
        omega = float(omega)
    dx_cells, dy_cells = cell_widths
    spacings = yee_inverse_spacings(dx_cells, dy_cells)
    nx, ny = spacings.inv_dx_e.size, spacings.inv_dy_e.size
    stretch_x_e, stretch_x_h = axis_stretch(cpml_axis_profiles(boundary_config, nx, 'x', pml_dx), omega)
    stretch_y_e, stretch_y_h = axis_stretch(cpml_axis_profiles(boundary_config, ny, 'y', pml_dx), omega)
    periodic_x, periodic_y = boundary_config.periodic_x, boundary_config.periodic_y
    phase_x, phase_y = bloch_wrap_phases(cell_widths, boundary_config)

    d1x_e = difference_matrix_1d(spacings.inv_dx_e, 'e', periodic_x, stretch_x_e, xp, sparse, phase_x)
    d1x_h = difference_matrix_1d(spacings.inv_dx_h, 'h', periodic_x, stretch_x_h, xp, sparse, phase_x)
    d1y_e = difference_matrix_1d(spacings.inv_dy_e, 'e', periodic_y, stretch_y_e, xp, sparse, phase_y)
    d1y_h = difference_matrix_1d(spacings.inv_dy_h, 'h', periodic_y, stretch_y_h, xp, sparse, phase_y)
    dx_e, dx_h = _kron_identity(d1x_e, ny, False, xp, sparse), _kron_identity(d1x_h, ny, False, xp, sparse)
    dy_e, dy_h = _kron_identity(d1y_e, nx, True, xp, sparse), _kron_identity(d1y_h, nx, True, xp, sparse)
    if polar_radius_min is not None:
        if periodic_x:
            raise ValueError('a polar grid has no periodic r axis')
        radius_faces, radius_centres = stretched_radii(dx_cells, stretch_x_e, polar_radius_min)
        inverse_centre_radii = np.repeat(1.0 / radius_centres, ny)   # flat i*ny + j: every cell row of column i
        dx_e = _scaled(dx_e, inverse_centre_radii, np.repeat(radius_faces[1:], ny), xp, sparse)
        dy_e = _scaled(dy_e, inverse_centre_radii, None, xp, sparse)
        dy_h = _scaled(dy_h, inverse_centre_radii, None, xp, sparse)
    return YeeOperators2D(nx=nx, ny=ny, omega=omega, dx_e=dx_e, dx_h=dx_h, dy_e=dy_e, dy_h=dy_h)


def scalar_laplacian(ops):
    """dx_e dx_h + dy_e dy_h on cell-centre samples; uniform periodic eigenvalue of cos(kx x + a) cos(ky y + b):
    -(2/dx^2)(1 - cos kx dx) - (2/dy^2)(1 - cos ky dy)."""
    return ops.dx_e @ ops.dx_h + ops.dy_e @ ops.dy_h


def _sparse_1d(data, row_index, col_index, n_cells, xp):
    """(n, n) CSR of COO entries (host numpy inputs) in the backend of xp."""
    if xp is not np:
        data, row_index, col_index = xp.asarray(data), xp.asarray(row_index), xp.asarray(col_index)
    return _sparse_module(xp).coo_matrix((data, (row_index, col_index)), shape=(n_cells, n_cells),
                                         dtype=data.dtype).tocsr()


def _centre_mean_matrix_1d(n_cells, periodic, wrap_factor, xp):
    """(n, n) hi faces -> centres: cell k = (face k + face k-1) / 2; cell 0 reads the wrapped last face times wrap_factor
    (periodic) or a missing lo face 0 (the grid-edge wall, as difference_matrix_1d 'e')."""
    rows = np.arange(n_cells, dtype=np.int32)
    lower_rows = rows if periodic else rows[1:]
    row_index = np.concatenate((rows, lower_rows)).astype(np.int32)
    col_index = np.concatenate((rows, (lower_rows - 1) % n_cells)).astype(np.int32)
    data = np.full(row_index.size, 0.5, dtype=np.complex128)
    if periodic:
        data[n_cells] *= wrap_factor   # row 0's lower face, one period below
    return _sparse_1d(data, row_index, col_index, n_cells, xp)


def _face_mean_matrix_1d(widths, periodic, wrap_factor, xp):
    """(n, n) centres -> hi faces: face k = (w_k c_k + w_(k+1) c_(k+1)) / (w_k + w_(k+1)) (the yee_grid.face_average
    weights); the last face takes cell 0 / wrap_factor (periodic) or nothing (non-periodic: never an active face)."""
    widths = np.asarray(widths, dtype=np.float64)
    n_cells = widths.size
    rows = np.arange(n_cells if periodic else n_cells - 1, dtype=np.int32)
    upper = ((rows + 1) % n_cells).astype(np.int32)
    total = widths[rows] + widths[upper]
    row_index = np.concatenate((rows, rows)).astype(np.int32)
    col_index = np.concatenate((rows, upper)).astype(np.int32)
    data = np.concatenate((widths[rows] / total, widths[upper] / total)).astype(np.complex128)
    if periodic:
        data[2 * rows.size - 1] /= wrap_factor   # the last face's upper cell, one period above
    return _sparse_1d(data, row_index, col_index, n_cells, xp)


def off_diagonal_mean_maps(cell_widths, periodic, wrap_factors, xp=np):
    """(centre_x, centre_y, face_x, face_y) CSR complex128 on the flat index: hi faces -> centres and centres -> hi
    faces along x / y, the factors of off_diagonal_face_couplings (the adjoint contracts them per cell)."""
    dx_cells, dy_cells = cell_widths
    nx, ny = len(dx_cells), len(dy_cells)
    periodic_x, periodic_y = periodic
    return (_kron_identity(_centre_mean_matrix_1d(nx, periodic_x, wrap_factors[0], xp), ny, False, xp, True),
            _kron_identity(_centre_mean_matrix_1d(ny, periodic_y, wrap_factors[1], xp), nx, True, xp, True),
            _kron_identity(_face_mean_matrix_1d(dx_cells, periodic_x, wrap_factors[0], xp), ny, False, xp, True),
            _kron_identity(_face_mean_matrix_1d(dy_cells, periodic_y, wrap_factors[1], xp), nx, True, xp, True))


def active_coupling_faces(weight_hi_x_faces, weight_hi_y_faces, periodic_x, periodic_y):
    """(hi-x, hi-y) bool (nx, ny) faces the eps_xy couplings keep: non-zero face weight (not PEC / even wall) and, on a
    non-periodic axis, not the never-updated last face."""
    active_hi_x_faces, active_hi_y_faces = weight_hi_x_faces != 0, weight_hi_y_faces != 0
    if not periodic_x:
        active_hi_x_faces[-1, :] = False
    if not periodic_y:
        active_hi_y_faces[:, -1] = False
    return active_hi_x_faces, active_hi_y_faces


def off_diagonal_face_couplings(cell_widths, periodic, wrap_factors, eta_xy, active_hi_x_faces, active_hi_y_faces,
                                xp=np):
    """(hi-x -> hi-y faces: E_x from D_y, hi-y -> hi-x faces: E_y from D_x) CSR complex128 of a cell off-diagonal
    inverse permittivity eta_xy (nx, ny), each face mean . diag(eta_xy) . centre mean, inactive faces dropped."""
    # face mean = W_f^-1 (centre mean)^T W_c (cell areas W_c, dual face areas W_f; Bloch: the conjugate transpose), so
    # W_f X is the (Hermitian) transpose of the other map's: the H_z operator keeps its symmetric structure.
    n_unknowns = len(cell_widths[0]) * len(cell_widths[1])
    centre_x, centre_y, face_x, face_y = off_diagonal_mean_maps(cell_widths, periodic, wrap_factors, xp)
    eta = _flat_values(eta_xy, n_unknowns, xp, 'off-diagonal eta_xy', np.complex128)
    active_x = _flat_values(active_hi_x_faces, n_unknowns, xp, 'active hi-x faces', np.bool_)
    active_y = _flat_values(active_hi_y_faces, n_unknowns, xp, 'active hi-y faces', np.bool_)

    def coupling(face_mean, active_rows, centre_mean, active_columns):
        """diag(active_rows) face_mean diag(eta) centre_mean diag(active_columns): one sparse product."""
        active_face_mean = _kept_entries(face_mean, active_rows[_entry_rows(face_mean, xp)], xp)
        product = (_column_scaled(active_face_mean, eta, xp) @ centre_mean).tocsr()
        return _kept_entries(product, active_columns[product.indices], xp)

    return coupling(face_y, active_y, centre_x, active_x), coupling(face_x, active_x, centre_y, active_y)


def _flat_values(values, n_unknowns, xp, label, dtype=None):
    """Flat (n,) xp array (in `dtype` when given) of an (nx, ny) or flat array on the i*ny + j index; ValueError on a
    wrong size."""
    values = xp.asarray(values)
    if values.size != n_unknowns:
        raise ValueError(f"{label} needs nx*ny = {n_unknowns} values ((nx, ny) or flat on i*ny + j), got shape "
                         f"{tuple(values.shape)}")
    return values.ravel() if dtype is None else values.astype(dtype, copy=False).ravel()


def _diagonal_matrix(values, n_unknowns, dtype, xp, label):
    """CSR diag(values) in `dtype` of an (nx, ny) or flat array on the i*ny + j index; ValueError on a wrong size."""
    values = _flat_values(values, n_unknowns, xp, label)
    return _sparse_module(xp).diags(values, 0, shape=(n_unknowns, n_unknowns), format='csr', dtype=dtype)


def _entry_rows(matrix, xp):
    """Row of every stored entry of a CSR matrix, in stored order."""
    return xp.searchsorted(matrix.indptr[1:], xp.arange(matrix.nnz, dtype=matrix.indptr.dtype), side='right')


def _kept_entries(matrix, keep_entries, xp):
    """New CSR of the stored entries of `matrix` where keep_entries (bool, one per entry) holds: a structural 0 / 1
    diagonal product without a sparse product or eliminate_zeros."""
    index_dtype = matrix.indptr.dtype
    kept_before = xp.concatenate((xp.zeros(1, dtype=index_dtype), xp.cumsum(keep_entries, dtype=index_dtype)))
    return _sparse_module(xp).csr_matrix(
        (matrix.data[keep_entries], matrix.indices[keep_entries], kept_before[matrix.indptr]), shape=matrix.shape)


def _column_scaled(matrix, column_scale, xp):
    """New CSR of matrix diag(column_scale) (CSR matrix, flat scale of its dtype); entries of 0 scale drop out, as in
    the product with a sparse diagonal."""
    entry_scale = column_scale[matrix.indices]
    nonzero = entry_scale != 0
    scaled = _kept_entries(matrix, nonzero, xp)
    scaled.data *= entry_scale[nonzero]
    return scaled


def _as_dtype(matrix, dtype):
    """The sparse matrix itself when it already has `dtype`, else a cast copy (cupyx products want equal dtypes)."""
    return matrix if matrix.dtype == dtype else matrix.astype(dtype)


def helmholtz_operator(ops, weight_hi_x_faces, weight_hi_y_faces, cell_term, xp=np):
    """dx_e diag(w_x) dx_h + dy_e diag(w_y) dy_h + diag(cell_term) as CSR (scipy / cupyx by xp); w_x on the hi-x
    faces, w_y on the hi-y faces, cell_term at the centres, each (nx, ny) or flat on the i*ny + j index."""
    if not _sparse_module(xp).issparse(ops.dx_e):
        raise TypeError("helmholtz_operator needs sparse YeeOperators2D of the same xp (yee_operators(..., "
                        "sparse=True))")
    n_unknowns = ops.nx * ops.ny
    weight_hi_x_faces, weight_hi_y_faces, cell_term = (xp.asarray(values) for values in
                                                       (weight_hi_x_faces, weight_hi_y_faces, cell_term))
    dtype = np.result_type(ops.dx_e.dtype, ops.dx_h.dtype, ops.dy_e.dtype, ops.dy_h.dtype, weight_hi_x_faces.dtype,
                           weight_hi_y_faces.dtype, cell_term.dtype)
    dx_e, dx_h, dy_e, dy_h = (_as_dtype(matrix, dtype) for matrix in (ops.dx_e, ops.dx_h, ops.dy_e, ops.dy_h))
    weight_x = _flat_values(weight_hi_x_faces, n_unknowns, xp, 'helmholtz_operator weight_hi_x_faces', dtype)
    weight_y = _flat_values(weight_hi_y_faces, n_unknowns, xp, 'helmholtz_operator weight_hi_y_faces', dtype)
    operator = _column_scaled(dx_e.tocsr(), weight_x, xp) @ dx_h   # (dx_e W_x) dx_h: the old product order
    operator = operator + _column_scaled(dy_e.tocsr(), weight_y, xp) @ dy_h
    return (operator + _diagonal_matrix(cell_term, n_unknowns, dtype, xp, 'helmholtz_operator cell_term')).tocsr()


def pin_cells(operator, pinned, xp=np):
    """D A D + (I - D) with D = diag(1 - pinned) (pinned bool (nx, ny) or flat): pinned unknowns decouple, row 1."""
    pinned = xp.asarray(pinned)
    if pinned.dtype != np.bool_:
        raise TypeError(f"pin_cells needs a bool mask of the pinned unknowns, got dtype {pinned.dtype}")
    n_unknowns = operator.shape[0]
    keep = ~_flat_values(pinned, n_unknowns, xp, 'pin_cells pinned')
    operator = operator.tocsr()
    kept = _kept_entries(operator, keep[_entry_rows(operator, xp)] & keep[operator.indices], xp)
    return (kept + _diagonal_matrix(pinned, n_unknowns, operator.dtype, xp, 'pin_cells pinned')).tocsr()


def pin_image_cells(operator, pinned, image_signs, xp=np):
    """pin_cells of A + diag(A s): a pinned cell p = s_p u_n of its only unpinned operator neighbour n is folded into
    n's diagonal (s = 0 Dirichlet, +-1 even / odd image; s must be 0 on unpinned cells)."""
    image_signs = xp.asarray(image_signs).ravel().astype(operator.dtype)
    n_unknowns = operator.shape[0]
    folded = _diagonal_matrix(operator @ image_signs, n_unknowns, operator.dtype, xp, 'pin_image_cells image_signs')
    return pin_cells((operator + folded).tocsr(), pinned, xp)
