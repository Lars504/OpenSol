"""Host 3D Yee curl operators of the em3d layout (engine3d header) for frequency-domain solves (em_fdfd_3d) and the
conformal-PEC band checks: per-axis 1D differences (physics.em.operators.difference_matrix_1d: graded widths, CPML
stretch, Bloch wraps) kroned onto the flat index (i ny + j) nz + k, the live E unknowns and the curl-curl operator;
numpy / scipy (scipy inside functions)."""
# Stacked vectors: E = [E_x, E_y, E_z] (3 N, N = nx ny nz; E_c on the +c face of cell (i, j, k)), H = [H_x, H_y, H_z] on
# the edges (H_x on the (+y, +z) edge, ...). curl_e (E faces -> H edges: the 'h' forward differences) and curl_h (H edges
# -> E faces: the 'e' backward differences; a missing lo H of cell 0 on a non-periodic axis reads 0) are the stencils of
# the FDTD kernels: (curl E)_x = d_y E_z - d_z E_y, (curl H)_x = d_y H_z - d_z H_y, cyclic; the last 'h' row of a
# non-periodic axis is empty (ih[n-1] = 0). In exp(+j omega t) curl E = -j omega mu0 mu H and curl H = j omega eps0 eps E,
# so E solves curl_h diag(1 / mu) curl_e E = k^2 eps E (k = omega / C0). CPML at omega: d/da -> (1 / s_a) d/da with s at
# the rows the derivative lands on ('e' centres, 'h' hi faces); without omega the real kappa the FDTD folds in. Bloch:
# F(x + L) = F(x) e^{-j k L} in the wrap entries (L = the summed widths).
# Live E unknowns (live_unknowns_3d) = what em_fdtd_3d can drive:
# - a face between two PEC cells is pinned (kernels3d face rule; pec walls = their outermost window cell layer);
# - the normal E of the last row of a non-periodic axis (E_x of i = nx - 1) is dropped: with ih[n-1] = 0 it forms a
#   closed subsystem with that row's tangential H that the interior never drives (the FDTD keeps it 0);
# - symmetric mirror (E even, a PMC plane on the window edge): the normal E and the tangential H of the plane face row
#   are pinned (h_keep False), the ghost cell's E dropped (a closed subsystem behind the pinned H);
# - antisymmetric mirror (E odd, a PEC plane): the ghost's tangential E = -its inner neighbour's, folded into the inner
#   column by the prolongation (a corner ghost of two planes takes the image of the image: the signs multiply); the
#   plane's normal E stays live with half its dual volume in the window (plane_faces).
import math
from dataclasses import dataclass

import numpy as np

from ..em.operators import axis_stretch, difference_matrix_1d
from .boundaries3d import AXES, along, axis_cpml_profile, mirror_planes, pec_wall_layers


@dataclass(frozen=True, eq=False)
class YeeOperators3D:
    """Host scipy CSR curls of a 3D grid: differences = per axis (d_e onto centres, d_h onto hi faces) 1D matrices,
    curl_e (3 N, 3 N) stacked E faces -> H edges, curl_h (3 N, 3 N) H edges -> E faces; omega None = real kappa."""
    shape: tuple
    omega: float | None
    differences: tuple
    curl_e: object
    curl_h: object


@dataclass(frozen=True, eq=False)
class LiveUnknowns3D:
    """E unknowns of a compressed 3D system (module header): full_index (n,) int64 into the stacked E faces,
    prolongation (3 N, n) CSR unknowns -> stacked E (mirror images included), h_keep (3 N,) bool H edges not pinned by
    a symmetric plane, plane_faces (n,) bool normal E on an antisymmetric plane (half its dual volume in the window)."""
    shape: tuple
    full_index: np.ndarray
    prolongation: object
    h_keep: np.ndarray
    plane_faces: np.ndarray

    @property
    def n_unknowns(self) -> int:
        return int(self.full_index.size)


def axis_operator(matrix_1d, axis, shape):
    """(N, N) CSR of a 1D (n_axis, n_axis) matrix acting along axis 0 | 1 | 2 of the flat index (i ny + j) nz + k."""
    import scipy.sparse
    before, after = math.prod(shape[:axis]), math.prod(shape[axis + 1:])
    operator = scipy.sparse.kron(matrix_1d, scipy.sparse.identity(after, dtype=matrix_1d.dtype, format='csr'),
                                 format='csr')
    return scipy.sparse.kron(scipy.sparse.identity(before, dtype=matrix_1d.dtype, format='csr'), operator,
                             format='csr')


def _curl(dx, dy, dz):
    """(3 N, 3 N) CSR of the curl with the per-axis derivatives dx, dy, dz: rows x = d_y z - d_z y, cyclic."""
    import scipy.sparse
    return scipy.sparse.bmat([[None, -dz, dy], [dz, None, -dx], [-dy, dx, None]], format='csr')


def yee_operators_3d(cell_widths, config, pml_cell_sizes, omega=None):
    """YeeOperators3D of cell_widths = (dx, dy, dz) cell widths [m] with the CPML profiles of the BoundaryConfig3D
    config (pml_cell_sizes: sigma grading cell size per axis [m]) and its Bloch wrap phases; omega [rad/s] None -> the
    real kappa stretch, > 0 -> physics.em.operators.pml_stretch at omega."""
    if omega is not None:
        omega = float(omega)
    widths = tuple(np.asarray(axis_widths, dtype=np.float64) for axis_widths in cell_widths)
    shape = tuple(axis_widths.size for axis_widths in widths)
    phases = config.bloch_phase_angles(tuple(float(np.sum(axis_widths)) for axis_widths in widths))
    differences = []
    for axis, (axis_widths, name) in enumerate(zip(widths, AXES)):
        inverse_e = 1.0 / axis_widths
        inverse_h = 1.0 / (0.5 * (axis_widths + np.roll(axis_widths, -1)))   # last entry = the periodic wrap
        stretch_e, stretch_h = axis_stretch(axis_cpml_profile(config, name, axis_widths.size, pml_cell_sizes[axis]),
                                             omega)
        periodic = config.periodic(name)
        differences.append((difference_matrix_1d(inverse_e, 'e', periodic, stretch_e, wrap_phase=phases[axis]),
                            difference_matrix_1d(inverse_h, 'h', periodic, stretch_h, wrap_phase=phases[axis])))
    d_e = [axis_operator(pair[0], axis, shape) for axis, pair in enumerate(differences)]
    d_h = [axis_operator(pair[1], axis, shape) for axis, pair in enumerate(differences)]
    return YeeOperators3D(shape=shape, omega=omega, differences=tuple(differences), curl_e=_curl(*d_h),
                          curl_h=_curl(*d_e))


def weighted_curl_e(ops, face_cell, face_axis, face_extra):
    """ops.curl_e with the rows of conformal H faces (conformal_pec3d.ConformalPEC3D arrays on ops' grid: flat lower
    corner cells, components, (n, 4) weight - 1) weighted on their E edges E_b[p + e_a], E_b[p], E_a[p + e_b], E_a[p]."""
    curl = ops.curl_e.tocsr(copy=True)
    face_cell = np.asarray(face_cell, dtype=np.int64)
    if face_cell.size == 0:
        return curl
    curl.sort_indices()
    face_axis = np.asarray(face_axis, dtype=np.int64)
    face_extra = np.asarray(face_extra, dtype=np.float64)
    shape = ops.shape
    n_cells = math.prod(shape)
    strides = np.array((shape[1] * shape[2], shape[2], 1), dtype=np.int64)
    axis_a, axis_b = (face_axis + 1) % 3, (face_axis + 2) % 3
    rows = face_axis * n_cells + face_cell
    columns = (axis_b * n_cells + face_cell + strides[axis_a], axis_b * n_cells + face_cell,
               axis_a * n_cells + face_cell + strides[axis_b], axis_a * n_cells + face_cell)
    entry_rows = np.repeat(np.arange(curl.shape[0], dtype=np.int64), np.diff(curl.indptr))
    entry_keys = entry_rows * curl.shape[1] + curl.indices   # sorted: CSR rows in order, sorted columns
    for slot, column in enumerate(columns):
        position = np.searchsorted(entry_keys, rows * curl.shape[1] + column)
        if np.any(position >= entry_keys.size) or np.any(entry_keys[np.minimum(position, entry_keys.size - 1)]
                                                         != rows * curl.shape[1] + column):
            raise ValueError('weighted_curl_e: a conformal face edge is not an entry of the curl (a face on the grid '
                             'edge?)')
        curl.data[position] *= 1.0 + face_extra[:, slot]
    return curl


def with_pec_walls(shape, config, window, pec_cells=None):
    """New host bool PEC cells of a grid `shape`: pec_cells (None = none) plus the outermost window cell layer of every
    'pec' face of the BoundaryConfig3D config (engine3d._pec_wall_layers; window = the physical-window slices)."""
    pec = np.zeros(shape, dtype=bool) if pec_cells is None else np.array(pec_cells, dtype=bool)
    for layer in pec_wall_layers(config, window):
        pec[layer] = True
    return pec


def live_unknowns_3d(shape, config, window, pec_cells=None):
    """LiveUnknowns3D of a grid `shape` with the BoundaryConfig3D faces of config, the physical-window slices `window`
    (mirror planes on its edges) and host bool (nx, ny, nz) PEC cells (pec_cells None = none; pec wall layers added)."""
    import scipy.sparse
    shape = tuple(int(count) for count in shape)
    n_cells = math.prod(shape)
    pec = with_pec_walls(shape, config, window, pec_cells)
    live = np.ones((3,) + shape, dtype=bool)
    h_keep = np.ones((3,) + shape, dtype=bool)
    plane = np.zeros((3,) + shape, dtype=bool)
    image_target = np.arange(3 * n_cells, dtype=np.int64).reshape((3,) + shape)
    image_sign = np.ones((3,) + shape, dtype=np.int8)
    imaged = np.zeros((3,) + shape, dtype=bool)
    for axis, name in enumerate(AXES):
        periodic = config.periodic(name)
        partner = np.roll(pec, -1, axis=axis) if periodic else np.concatenate(
            (pec[along(axis, slice(1, None))], pec[along(axis, slice(-1, None))]), axis=axis)
        live[axis] &= ~(pec & partner)
        if not periodic:
            live[axis][along(axis, -1)] = False
    for axis, kind, ghost, inner, plane_face in mirror_planes(config, window):   # x faces first, as before
        others = [other for other in range(3) if other != axis]
        if kind == 'symmetric':
            live[axis][along(axis, plane_face)] = False
            live[(slice(None),) + along(axis, ghost)] = False
            for other in others:
                h_keep[other][along(axis, plane_face)] = False
        else:
            for other in others:   # an inner row imaged by an earlier plane passes its target and sign on
                image_target[other][along(axis, ghost)] = image_target[other][along(axis, inner)]
                image_sign[other][along(axis, ghost)] = -image_sign[other][along(axis, inner)]
                imaged[other][along(axis, ghost)] = True
                live[other][along(axis, ghost)] = False
            plane[axis][along(axis, plane_face)] = True
    image_target, image_sign, imaged, live = (values.ravel() for values in (image_target, image_sign, imaged, live))
    full_index = np.flatnonzero(live)
    numbers = np.full(3 * n_cells, -1, dtype=np.int64)
    numbers[full_index] = np.arange(full_index.size, dtype=np.int64)
    images = np.flatnonzero(imaged & (numbers[image_target] >= 0))
    rows = np.concatenate((full_index, images))
    columns = np.concatenate((numbers[full_index], numbers[image_target[images]]))
    values = np.concatenate((np.ones(full_index.size), image_sign[images].astype(np.float64)))
    prolongation = scipy.sparse.csr_matrix((values, (rows, columns)), shape=(3 * n_cells, full_index.size))
    return LiveUnknowns3D(shape=shape, full_index=full_index, prolongation=prolongation, h_keep=h_keep.ravel(),
                          plane_faces=plane.ravel()[full_index])


def nested_dissection_ordering(live):
    """(n,) elimination order of the live unknowns for the host factor: cells in sparse_direct.grid_nested_dissection
    order (curl-curl couples cells within +-1 per axis), the E components of a cell together."""
    from ...solvers.sparse_direct import grid_nested_dissection
    n_cells = math.prod(live.shape)
    cell_rank = np.empty(n_cells, dtype=np.int64)
    cell_rank[grid_nested_dissection(live.shape)] = np.arange(n_cells, dtype=np.int64)
    component, cell = np.divmod(live.full_index, n_cells)
    return np.argsort(3 * cell_rank[cell] + component, kind='stable')


def curl_curl_operator(ops, live, inverse_mu=None, curl_e=None):
    """(n, n) CSR of K = R curl_h diag(h_keep / mu) curl_e P on the live unknowns (R their rows, P live.prolongation):
    inverse_mu (3 N,) 1 / mu_r of every H edge (None = vacuum), curl_e None = ops.curl_e (e.g. weighted_curl_e)."""
    import scipy.sparse
    edge_weights = live.h_keep.astype(np.float64) if inverse_mu is None else np.where(live.h_keep, inverse_mu, 0.0)
    right = scipy.sparse.diags(edge_weights) @ ((ops.curl_e if curl_e is None else curl_e) @ live.prolongation)
    return (ops.curl_h[live.full_index] @ right).tocsr()


def face_stretches(shape, config, pml_cell_sizes, omega=None):
    """(3 N,) complex128 product of the CPML stretches at every stacked E face (its own axis at the hi face, the other
    two at the cell centre): with face_dual_volumes the row scaling D that makes D (-K + k^2 eps) complex symmetric."""
    # d_e / d_h weights are 1 / (width s_e) and 1 / (dual length s_h), so curl_h = V_E^-1 curl_e^T V_H with the stretched
    # face / edge volumes V (the 1D note in physics.em.operators.difference_matrix_1d); a Bloch phase breaks it.
    stretches = []
    for axis, name in enumerate(AXES):
        stretch_e, stretch_h = axis_stretch(axis_cpml_profile(config, name, int(shape[axis]), pml_cell_sizes[axis]),
                                             omega)
        stretches.append((np.asarray(stretch_e, dtype=np.complex128), np.asarray(stretch_h, dtype=np.complex128)))
    faces = []
    for component in range(3):
        factors = [stretches[axis][1 if axis == component else 0] for axis in range(3)]
        faces.append(np.multiply.outer(np.multiply.outer(factors[0], factors[1]), factors[2]).ravel())
    return np.concatenate(faces)


def face_dual_volumes(cell_widths, periodic):
    """(3 N,) dual volume [m^3] of every stacked E face: the link between its two centres (the last face of a
    non-periodic axis: its own width) times the two cell widths across it."""
    widths = tuple(np.asarray(axis_widths, dtype=np.float64) for axis_widths in cell_widths)
    volumes = []
    for axis in range(3):
        factors = list(widths)
        partner = np.roll(widths[axis], -1) if periodic[axis] else np.append(widths[axis][1:], widths[axis][-1])
        factors[axis] = 0.5 * (widths[axis] + partner)
        volumes.append(np.multiply.outer(np.multiply.outer(factors[0], factors[1]), factors[2]).ravel())
    return np.concatenate(volumes)
