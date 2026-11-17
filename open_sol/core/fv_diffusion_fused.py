"""Fused DiffusionSystem re-assembly on a kept 5 / 7-point pattern (no phi wrap, c_xy term or cut cells): one kernel
from the coefficients to the CSR values, face conductances, outer-face terms and the fixed-cell masks, one for the
elimination, Jacobi diagonal and rhs offset (core.kernels twins with numpy twins), plus the c > 0 pattern test of a
coefficient-only change. Same arithmetic order as fv_diffusion_3d.build_laplacian and DiffusionSystem.assemble."""
import numpy as np

from .backend import cp
from .kernels import real_raw_twin, twin

# block size and grid cap of the two grid-stride kernels
_BLOCK = 256
_MAX_BLOCKS = 4096
# outer-face kinds in the face parameter table (BCSide.kind)
_FACE_KINDS = {'neumann': 0, 'dirichlet': 1, 'robin': 2}

_SOURCE = r"""
__device__ __forceinline__ real_t area_scaled(real_t value, const real_t* factors, const int* offsets, int axis,
                                              int fi, int fj, int fk)
{
    // value x the three 1D face-area factors of `axis` in axis order (fv_diffusion_3d._scale_by; unit factors exact)
    value = value * factors[offsets[3 * axis] + fi];
    value = value * factors[offsets[3 * axis + 1] + fj];
    return value * factors[offsets[3 * axis + 2] + fk];
}

__device__ __forceinline__ real_t cell_length(const real_t* factors, const int* offsets, int axis, int i, int j, int k)
{
    // physical cell length along `axis`: (f0 f1) f2 of its 1D factors (FaceGeometry.length)
    return (factors[offsets[9 + 3 * axis] + i] * factors[offsets[9 + 3 * axis + 1] + j])
           * factors[offsets[9 + 3 * axis + 2] + k];
}

__device__ __forceinline__ real_t series_link(real_t c_lower, real_t c_upper, real_t length_lower, real_t length_upper,
                                              bool held_lower, bool held_upper, const real_t* factors,
                                              const int* offsets, int axis, int fi, int fj, int fk)
{
    // fv_diffusion_3d._series_link without a held wall: a held-free pair takes the free half cell
    if (held_lower && !held_upper)
        return area_scaled((real_t)2 * c_upper / length_upper, factors, offsets, axis, fi, fj, fk);
    if (held_upper && !held_lower)
        return area_scaled((real_t)2 * c_lower / length_lower, factors, offsets, axis, fi, fj, fk);
    real_t denominator = length_lower * c_upper + length_upper * c_lower;
    return area_scaled((real_t)2 * c_lower * c_upper / (denominator > 0 ? denominator : (real_t)1),
                       factors, offsets, axis, fi, fj, fk);
}

extern "C" __global__ void fv_reassemble_entries(
    const real_t* c_x, const real_t* c_y, const real_t* c_z, const bool* held, const real_t* factors,
    const int* offsets, const real_t* face_parameters, const real_t* azimuthal, const real_t* extra_diag,
    const real_t* capacity, const real_t* extra_rhs, const real_t* held_values, const real_t* unknown,
    const int* indptr, real_t* data, real_t* conduction_diagonal, real_t* interior, real_t* face_buffer,
    bool* fixed, real_t* fixed_values, real_t* free_mask, real_t* rhs_drive,
    int nx, int ny, int nz, int ndim, int has_azimuthal, int has_capacity)
{
    const int n_cells = nx * ny * nz;
    const int count[3] = {nx, ny, nz};
    const int stride[3] = {ny * nz, nz, 1};
    const real_t* coefficient[3] = {c_x, c_y, c_z};
    int interior_offset[3];
    int running = 0;
    for (int axis = 0; axis < 3; ++axis) {
        interior_offset[axis] = running;
        if (axis < ndim) running += (n_cells / count[axis]) * (count[axis] - 1);
    }
    int face_offset[6];
    running = 0;
    for (int face = 0; face < 2 * ndim; ++face) {
        face_offset[face] = running;
        running += n_cells / count[face / 2];
    }
    const int face_total = running;
    for (int cell = blockIdx.x * blockDim.x + threadIdx.x; cell < n_cells; cell += blockDim.x * gridDim.x) {
        const int k = cell % nz;
        const int j = (cell / nz) % ny;
        const int i = cell / (ny * nz);
        const int index[3] = {i, j, k};
        const bool held_here = held[cell];
        real_t lower_link[3] = {0, 0, 0};
        real_t upper_link[3] = {0, 0, 0};
        real_t diagonal = 0;
        for (int axis = 0; axis < ndim; ++axis) {
            const real_t c_here = coefficient[axis][cell];
            const real_t length_here = cell_length(factors, offsets, axis, i, j, k);
            int face_index[3] = {i, j, k};
            if (index[axis] < count[axis] - 1) {   // the face to the upper neighbour (this cell is its lower cell)
                const int upper = cell + stride[axis];
                int neighbour[3] = {i, j, k};
                neighbour[axis] += 1;
                face_index[axis] = index[axis] + 1;
                upper_link[axis] = series_link(
                    c_here, coefficient[axis][upper], length_here,
                    cell_length(factors, offsets, axis, neighbour[0], neighbour[1], neighbour[2]),
                    held_here, held[upper], factors, offsets, axis, face_index[0], face_index[1], face_index[2]);
                diagonal += upper_link[axis];
                int reduced[3] = {count[0], count[1], count[2]};
                reduced[axis] -= 1;
                interior[interior_offset[axis] + (i * reduced[1] + j) * reduced[2] + k] = upper_link[axis];
            }
            if (index[axis] > 0) {   // the face to the lower neighbour
                const int lower = cell - stride[axis];
                int neighbour[3] = {i, j, k};
                neighbour[axis] -= 1;
                face_index[axis] = index[axis];
                lower_link[axis] = series_link(
                    coefficient[axis][lower], c_here,
                    cell_length(factors, offsets, axis, neighbour[0], neighbour[1], neighbour[2]), length_here,
                    held[lower], held_here, factors, offsets, axis, face_index[0], face_index[1], face_index[2]);
                diagonal += lower_link[axis];
            }
        }
        real_t boundary_rhs = 0;
        for (int face = 0; face < 2 * ndim; ++face) {   // FACES / FACES_3D order: x_lo, x_hi, y_lo, ...
            const int axis = face / 2;
            const int edge = (face % 2) ? count[axis] - 1 : 0;
            if (index[axis] != edge) continue;
            const real_t kind = face_parameters[4 * face];
            const real_t factor_two = face_parameters[4 * face + 1];
            const real_t h = face_parameters[4 * face + 2];
            const real_t scalar = face_parameters[4 * face + 3];
            int face_index[3] = {i, j, k};
            face_index[axis] = (face % 2) ? count[axis] : 0;
            const real_t c_boundary = coefficient[axis][cell];
            const real_t width = cell_length(factors, offsets, axis, i, j, k);
            real_t conductance = 0;
            real_t rhs;
            if (kind == (real_t)1) {          // dirichlet
                conductance = area_scaled(factor_two * c_boundary / width, factors, offsets, axis,
                                          face_index[0], face_index[1], face_index[2]);
                rhs = conductance * scalar;
            } else if (kind == (real_t)2) {   // robin
                const real_t two_c = factor_two * c_boundary;
                const real_t denominator = two_c + h * width;
                conductance = area_scaled(h * two_c / (denominator > 0 ? denominator : (real_t)1), factors, offsets,
                                          axis, face_index[0], face_index[1], face_index[2]);
                rhs = conductance * scalar;
            } else {                          // neumann: inward flux x face area
                rhs = area_scaled(scalar, factors, offsets, axis, face_index[0], face_index[1], face_index[2]);
            }
            diagonal += conductance;
            boundary_rhs += rhs;
            const int within = axis == 0 ? j * nz + k : (axis == 1 ? i * nz + k : i * ny + j);
            face_buffer[face_offset[face] + within] = conductance;
            face_buffer[face_total + face_offset[face] + within] = rhs;
        }
        real_t conduction = diagonal;
        if (has_azimuthal) conduction = conduction + azimuthal[cell];
        conduction_diagonal[cell] = conduction;
        real_t extra = extra_diag[cell];
        if (has_capacity) extra = extra + capacity[cell];
        const real_t effective = conduction + extra;
        const bool unconnected = (effective <= 0) && !held_here;
        const bool fixed_here = held_here || unconnected;
        fixed[cell] = fixed_here;
        fixed_values[cell] = held_here ? held_values[cell] : (unconnected ? unknown[cell] : (real_t)0);
        free_mask[cell] = fixed_here ? (real_t)0 : (real_t)1;
        rhs_drive[cell] = boundary_rhs + extra_rhs[cell];
        // CSR row in the slot order of _SlotLayout: -x, -y, [-z,] diagonal, [+z,] +y, +x
        int position = indptr[cell];
        for (int axis = 0; axis < ndim; ++axis)
            if (index[axis] > 0) data[position++] = -lower_link[axis];
        data[position++] = effective;
        for (int axis = ndim - 1; axis >= 0; --axis)
            if (index[axis] < count[axis] - 1) data[position++] = -upper_link[axis];
    }
}

extern "C" __global__ void fv_reassemble_elimination(
    const real_t* data, const int* indptr, const int* indices, const int* diagonal_positions,
    const real_t* free_mask, const bool* fixed, const real_t* fixed_values, const real_t* rhs_drive,
    real_t* solve_data, real_t* inverse_diagonal, real_t* rhs_offset, real_t* unknown, int n_cells)
{
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < n_cells; row += blockDim.x * gridDim.x) {
        const real_t free_row = free_mask[row];
        real_t coupling = 0;   // (A_eff u_fix)[row]
        for (int position = indptr[row]; position < indptr[row + 1]; ++position) {
            const int column = indices[position];
            const real_t value = data[position];
            solve_data[position] = value * free_row * free_mask[column];
            coupling += value * fixed_values[column];
        }
        const int diagonal_position = diagonal_positions[row];
        const real_t diagonal = solve_data[diagonal_position] + (fixed[row] ? (real_t)1 : (real_t)0);
        solve_data[diagonal_position] = diagonal;
        inverse_diagonal[row] = (real_t)1 / diagonal;
        rhs_offset[row] = free_row * (rhs_drive[row] - coupling);
        if (fixed[row]) unknown[row] = fixed_values[row];
    }
}
"""


def _factor_views(factors, offsets, shape):
    """Per axis the (area factors, length factors) of the packed table as arrays broadcasting over `shape` (3 axes)."""
    sizes = [shape[0], shape[1], shape[2]]
    areas, lengths = [], []
    for axis in range(3):
        area_parts, length_parts = [], []
        for along in range(3):
            broadcast = [1, 1, 1]
            area_size = sizes[along] + (1 if along == axis else 0)
            broadcast[along] = area_size
            start = int(offsets[3 * axis + along])
            area_parts.append(factors[start:start + area_size].reshape(broadcast))
            broadcast[along] = sizes[along]
            start = int(offsets[9 + 3 * axis + along])
            length_parts.append(factors[start:start + sizes[along]].reshape(broadcast))
        areas.append(area_parts)
        lengths.append((length_parts[0] * length_parts[1]) * length_parts[2])
    return areas, lengths


def _scaled(value, parts):
    """value x each area factor part in axis order (the CUDA area_scaled)."""
    for part in parts:
        value = value * part
    return value


def _cut(axis, part):
    return (slice(None),) * axis + (part,)


def _entries_host(c_x, c_y, c_z, held, factors, offsets, face_parameters, azimuthal, extra_diag, capacity, extra_rhs,
                  held_values, unknown, indptr, data, conduction_diagonal, interior, face_buffer, fixed,
                  fixed_values, free_mask, rhs_drive, nx, ny, nz, ndim, has_azimuthal, has_capacity):
    """numpy twin of fv_reassemble_entries (same arguments; writes the outputs in place)."""
    shape = (int(nx), int(ny), int(nz))
    ndim = int(ndim)
    dtype = data.dtype
    n_cells = shape[0] * shape[1] * shape[2]
    coefficients = [array.reshape(shape) for array in (c_x, c_y, c_z)]
    held = held.reshape(shape)
    areas, lengths = _factor_views(factors, offsets, shape)
    two = dtype.type(2.0)
    one = dtype.type(1.0)
    diagonal = np.zeros(shape, dtype=dtype)
    links = []   # per axis the interior face conductances (n - 1 along it)
    interior_start = 0
    for axis in range(ndim):
        lower, upper = _cut(axis, slice(None, -1)), _cut(axis, slice(1, None))
        c_lower, c_upper = coefficients[axis][lower], coefficients[axis][upper]
        length_lower, length_upper = lengths[axis][lower], lengths[axis][upper]
        held_lower, held_upper = held[lower], held[upper]
        area = [part[_cut(axis, slice(1, -1))] if along == axis else part
                for along, part in enumerate(areas[axis])]
        denominator = length_lower * c_upper + length_upper * c_lower
        series = _scaled(two * c_lower * c_upper / np.where(denominator > 0, denominator, one), area)
        free_upper = _scaled(two * c_upper / length_upper, area)
        free_lower = _scaled(two * c_lower / length_lower, area)
        link = np.where(held_lower & ~held_upper, free_upper, np.where(held_upper & ~held_lower, free_lower, series))
        link = np.broadcast_to(link, c_lower.shape).astype(dtype, copy=False)
        diagonal[lower] += link
        diagonal[upper] += link
        interior[interior_start:interior_start + link.size] = link.ravel()
        interior_start += link.size
        links.append(link)
    boundary_rhs = np.zeros(shape, dtype=dtype)
    face_sizes = [n_cells // shape[face // 2] for face in range(2 * ndim)]
    face_total = sum(face_sizes)
    face_start = 0
    for face in range(2 * ndim):
        axis, side = divmod(face, 2)
        edge = shape[axis] - 1 if side else 0
        cells = _cut(axis, slice(edge, edge + 1))
        kind, factor_two, h, scalar = (dtype.type(value) for value in face_parameters[4 * face:4 * face + 4])
        area = [part[_cut(axis, slice(shape[axis], shape[axis] + 1) if side else slice(0, 1))] if along == axis
                else part for along, part in enumerate(areas[axis])]
        c_boundary = coefficients[axis][cells]
        width = lengths[axis][cells]
        if kind == 1:
            conductance = _scaled(factor_two * c_boundary / width, area)
            rhs = conductance * scalar
        elif kind == 2:
            two_c = factor_two * c_boundary
            denominator = two_c + h * width
            conductance = _scaled(h * two_c / np.where(denominator > 0, denominator, one), area)
            rhs = conductance * scalar
        else:
            conductance = np.zeros(c_boundary.shape, dtype=dtype)
            rhs = _scaled(np.full(c_boundary.shape, scalar, dtype=dtype), area)
        conductance = np.broadcast_to(conductance, c_boundary.shape).astype(dtype, copy=False)
        rhs = np.broadcast_to(rhs, c_boundary.shape).astype(dtype, copy=False)
        diagonal[cells] += conductance
        boundary_rhs[cells] += rhs
        size = face_sizes[face]
        face_buffer[face_start:face_start + size] = conductance.ravel()
        face_buffer[face_total + face_start:face_total + face_start + size] = rhs.ravel()
        face_start += size
    conduction = diagonal.ravel()
    if int(has_azimuthal):
        conduction = conduction + azimuthal.ravel()
    conduction_diagonal[...] = conduction.reshape(conduction_diagonal.shape)
    extra = extra_diag.ravel()
    if int(has_capacity):
        extra = extra + capacity.ravel()
    effective = conduction + extra
    held_flat = held.ravel()
    unconnected = (effective <= 0) & ~held_flat
    fixed_flat = held_flat | unconnected
    fixed[...] = fixed_flat.reshape(fixed.shape)
    fixed_values[...] = np.where(held_flat, held_values.ravel(),
                                 np.where(unconnected, unknown.ravel(), dtype.type(0.0))).reshape(fixed_values.shape)
    free_mask[...] = (~fixed_flat).astype(dtype).reshape(free_mask.shape)
    rhs_drive[...] = (boundary_rhs.ravel() + extra_rhs.ravel()).reshape(rhs_drive.shape)
    # CSR rows in the _SlotLayout order: -x, -y, [-z,] diagonal, [+z,] +y, +x
    position = np.asarray(indptr[:-1], dtype=np.int64).reshape(shape).copy()
    index = np.indices(shape, sparse=True)
    slots = []
    for axis in range(ndim):
        values = np.zeros(shape, dtype=dtype)
        values[_cut(axis, slice(1, None))] = -links[axis]
        slots.append((values, np.broadcast_to(index[axis] > 0, shape)))
    slots.append((effective.reshape(shape), np.ones(shape, dtype=bool)))
    for axis in reversed(range(ndim)):
        values = np.zeros(shape, dtype=dtype)
        values[_cut(axis, slice(None, -1))] = -links[axis]
        slots.append((values, np.broadcast_to(index[axis] < shape[axis] - 1, shape)))
    for values, present in slots:
        data[position[present]] = values[present]
        position += present


def _elimination_host(data, indptr, indices, diagonal_positions, free_mask, fixed, fixed_values, rhs_drive, solve_data,
                      inverse_diagonal, rhs_offset, unknown, n_cells):
    """numpy twin of fv_reassemble_elimination (same arguments; writes the outputs in place)."""
    import scipy.sparse as host_sparse
    n_cells = int(n_cells)
    free = free_mask.ravel()
    fixed_flat = fixed.ravel()
    rows = np.repeat(np.arange(n_cells), np.diff(np.asarray(indptr, dtype=np.int64)))
    solve_data[...] = data * free[rows] * free[indices]
    solve_data[diagonal_positions] += fixed_flat.astype(solve_data.dtype)
    inverse_diagonal[...] = solve_data.dtype.type(1.0) / solve_data[diagonal_positions]
    operator = host_sparse.csr_matrix((data, indices, indptr), shape=(n_cells, n_cells))
    coupling = operator.dot(fixed_values.ravel())
    rhs_offset[...] = (free * (rhs_drive.ravel() - coupling)).reshape(rhs_offset.shape)
    unknown_flat = unknown.reshape(-1)
    unknown_flat[fixed_flat] = fixed_values.ravel()[fixed_flat]


reassemble_entries = real_raw_twin('core.fv_reassemble_entries', _SOURCE, 'fv_reassemble_entries', _entries_host,
                                   dtype_argument=0)
reassemble_elimination = real_raw_twin('core.fv_reassemble_elimination', _SOURCE, 'fv_reassemble_elimination',
                                       _elimination_host, dtype_argument=0)


def _pattern_moved_host(c_x, kept_x, c_y, kept_y, c_z, kept_z, capacity, kept_capacity):
    """numpy twin of positive_pattern_moved (a 0-d bool array)."""
    moved = False
    for new, old in ((c_x, kept_x), (c_y, kept_y), (c_z, kept_z), (capacity, kept_capacity)):
        moved = moved | bool(np.any((new > 0) != (old > 0)))
    return np.asarray(moved)


# True when any cell's c > 0 (or C/dt > 0) differs between a new assembly and the kept one (a coefficient-only change)
positive_pattern_moved = twin('core.fv_positive_pattern_moved', cp.ReductionKernel(
    'T c_x, T kept_x, T c_y, T kept_y, T c_z, T kept_z, T capacity, T kept_capacity', 'bool moved',
    '((c_x > 0) != (kept_x > 0)) || ((c_y > 0) != (kept_y > 0)) || ((c_z > 0) != (kept_z > 0))'
    ' || ((capacity > 0) != (kept_capacity > 0))',
    'a || b', 'moved = a', 'false', 'fv_positive_pattern_moved', reduce_type='bool'), _pattern_moved_host)


def launch_shape(n_cells):
    """(grid, block) of the grid-stride re-assembly kernels."""
    return (max(1, min(_MAX_BLOCKS, -(-int(n_cells) // _BLOCK))),), (_BLOCK,)


def geometry_factors(metric, ndim, dtype, xp):
    """(packed 1D factors, int32 offsets (18,)) of a WindowMetric / WindowMetric3D in dtype: per axis the three face-area
    factors, then per axis the three length factors; a 2D window gets unit z factors (nz = 1)."""
    areas = [area.factors() for area in metric.face_area]
    lengths = [length.factors() for length in metric.length]
    sizes = [int(np.size(factor)) for factor in lengths[0]] + ([] if ndim == 3 else [1])
    blocks = []
    for axis in range(3):
        for along in range(3):
            if axis < ndim and along < ndim:
                blocks.append(np.asarray(areas[axis][along], dtype=np.float64).ravel())
            else:
                blocks.append(np.ones(sizes[along] + (1 if along == axis else 0)))
    for axis in range(3):
        for along in range(3):
            if axis < ndim and along < ndim:
                blocks.append(np.asarray(lengths[axis][along], dtype=np.float64).ravel())
            else:
                blocks.append(np.ones(sizes[along]))
    offsets = np.concatenate([[0], np.cumsum([block.size for block in blocks])[:-1]]).astype(np.int32)
    return xp.asarray(np.concatenate(blocks), dtype=dtype), xp.asarray(offsets)


def face_parameters(bcs, faces, reference):
    """Host float64 (4 x 6) table of the outer faces (kind, 2 x conductance_factor, h, value term) in face order: the
    value term is value - reference (dirichlet), external - reference (robin) or the inward flux (neumann)."""
    table = np.zeros((6, 4))
    for face_number, face in enumerate(faces):
        bc = getattr(bcs, face)
        if bc.kind == 'dirichlet':
            scalar = bc.value - reference
        elif bc.kind == 'robin':
            scalar = bc.external - reference
        else:
            scalar = bc.value
        table[face_number] = (_FACE_KINDS[bc.kind], bc.conductance_factor * 2.0, bc.h, scalar)
    return table.ravel()
