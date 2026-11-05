"""Host (numpy / scipy) complex operator of the time-harmonic magnetic solve on the window: the 5-point finite-volume
operator of complex coefficients (same face rules as core.fv_diffusion: harmonic face means, held-free faces = the
free half cell, Dirichlet half cells times a factor), plus a per-cell diagonal; flat index i * ny + j."""
from dataclasses import dataclass

import numpy as np

_FACE_CELLS = {'x_lo': (0, slice(None)), 'x_hi': (-1, slice(None)), 'y_lo': (slice(None), 0),
               'y_hi': (slice(None), -1)}


@dataclass(frozen=True, eq=False)
class HostOperator:
    """Assembled window operator: CSR matrix [c], boundary rhs [c u] (window), interior face conductances (x faces
    (nx-1, ny), y faces (nx, ny-1)) and {face: (conductance, rhs)} along each outer face."""
    matrix: object
    boundary_rhs: np.ndarray
    conductance_x: np.ndarray
    conductance_y: np.ndarray
    face_terms: dict


def face_conductances(c_xx, c_yy, dx, dy, held):
    """Series conductances (complex allowed, Re > 0) of interior x / y faces; a held-free face gets the free half cell
    2 c_free L / w_free (as core.fv_diffusion.face_conductances)."""
    c_left, c_right = c_xx[:-1, :], c_xx[1:, :]
    conductance_x = 2.0 * c_left * c_right * dy[None, :] / (dx[:-1, None] * c_right + dx[1:, None] * c_left)
    c_below, c_above = c_yy[:, :-1], c_yy[:, 1:]
    conductance_y = 2.0 * c_below * c_above * dx[:, None] / (dy[None, :-1] * c_above + dy[None, 1:] * c_below)
    held_left, held_right = held[:-1, :], held[1:, :]
    conductance_x = np.where(held_left & ~held_right, 2.0 * c_right * dy[None, :] / dx[1:, None],
                             np.where(held_right & ~held_left, 2.0 * c_left * dy[None, :] / dx[:-1, None],
                                      conductance_x))
    held_below, held_above = held[:, :-1], held[:, 1:]
    conductance_y = np.where(held_below & ~held_above, 2.0 * c_above * dx[:, None] / dy[None, 1:],
                             np.where(held_above & ~held_below, 2.0 * c_below * dx[:, None] / dy[None, :-1],
                                      conductance_y))
    return conductance_x, conductance_y


def boundary_terms(face, bc, c_xx, c_yy, dx, dy):
    """(conductance, rhs) along one outer face: dirichlet bc.conductance_factor * 2 c L / w and its value drive, neumann
    none and value * L (inward flux); robin raises (no magnetic feature sets it)."""
    cells = _FACE_CELLS[face]
    if face in ('x_lo', 'x_hi'):
        c_boundary, face_length, normal_width = c_xx[cells], dy, dx[cells[0]]
    else:
        c_boundary, face_length, normal_width = c_yy[cells], dx, dy[cells[1]]
    if bc.kind == 'dirichlet':
        conductance = bc.conductance_factor * 2.0 * c_boundary * face_length / normal_width
        return conductance, conductance * bc.value
    if bc.kind == 'neumann':
        return np.zeros_like(c_boundary), bc.value * face_length + 0j
    raise NotImplementedError(f"face {face}: a robin condition has no magnetic meaning (use dirichlet / neumann)")


def assemble_operator(c_xx, c_yy, dx, dy, held, bcs, diagonal_extra):
    """HostOperator of -div(c grad u) + diagonal_extra u on the window (c complex window arrays, host widths, held
    bool window mask, bcs = {face: BCSide}, each with its conductance_factor)."""
    import scipy.sparse
    nx, ny = c_xx.shape
    conductance_x, conductance_y = face_conductances(c_xx, c_yy, dx, dy, held)
    diagonal = np.array(diagonal_extra, dtype=np.complex128)
    diagonal[:-1, :] += conductance_x
    diagonal[1:, :] += conductance_x
    diagonal[:, :-1] += conductance_y
    diagonal[:, 1:] += conductance_y
    boundary_rhs = np.zeros((nx, ny), dtype=np.complex128)
    face_terms = {}
    for face, bc in bcs.items():
        conductance, rhs = boundary_terms(face, bc, c_xx, c_yy, dx, dy)
        cells = _FACE_CELLS[face]
        diagonal[cells] += conductance
        boundary_rhs[cells] += rhs
        face_terms[face] = (conductance, rhs)
    index = np.arange(nx * ny).reshape(nx, ny)
    rows = np.concatenate((index.ravel(), index[:-1, :].ravel(), index[1:, :].ravel(),
                           index[:, :-1].ravel(), index[:, 1:].ravel()))
    columns = np.concatenate((index.ravel(), index[1:, :].ravel(), index[:-1, :].ravel(),
                              index[:, 1:].ravel(), index[:, :-1].ravel()))
    data = np.concatenate((diagonal.ravel(), -conductance_x.ravel(), -conductance_x.ravel(),
                           -conductance_y.ravel(), -conductance_y.ravel()))
    matrix = scipy.sparse.csr_matrix((data, (rows, columns)), shape=(nx * ny, nx * ny))
    return HostOperator(matrix, boundary_rhs, conductance_x, conductance_y, face_terms)


def face_fluxes_at_centres(operator, unknown, dx, dy, held):
    """(flux_x, flux_y) = -c grad u [c u / m] at the cell centres from the face flows of `operator` (complex unknown
    window array); 0 in held cells (as DiffusionSystem.flux_at_centres)."""
    nx, ny = unknown.shape
    flux_x_faces = np.zeros((nx + 1, ny), dtype=np.complex128)
    flux_x_faces[1:-1, :] = operator.conductance_x * (unknown[:-1, :] - unknown[1:, :])
    flux_y_faces = np.zeros((nx, ny + 1), dtype=np.complex128)
    flux_y_faces[:, 1:-1] = operator.conductance_y * (unknown[:, :-1] - unknown[:, 1:])
    for face, (conductance, rhs) in operator.face_terms.items():
        inward = rhs - conductance * unknown[_FACE_CELLS[face]]
        if face == 'x_lo':
            flux_x_faces[0, :] = inward
        elif face == 'x_hi':
            flux_x_faces[-1, :] = -inward
        elif face == 'y_lo':
            flux_y_faces[:, 0] = inward
        else:
            flux_y_faces[:, -1] = -inward
    flux_x_faces /= dy[None, :]
    flux_y_faces /= dx[:, None]
    free = ~held
    flux_x = np.where(free, 0.5 * (flux_x_faces[:-1, :] + flux_x_faces[1:, :]), 0.0)
    flux_y = np.where(free, 0.5 * (flux_y_faces[:, :-1] + flux_y_faces[:, 1:]), 0.0)
    return flux_x, flux_y


def face_power(operator, unknown, held):
    """sum of g |du|^2 [c u^2] (complex) over the faces touching a free cell: interior (held-free faces with their free
    half cell) and the outer faces of free cells (g |u - u_face|^2); = u^H K u of the free part."""
    free = ~held
    total = np.sum(np.where(free[:-1, :] | free[1:, :],
                            operator.conductance_x * np.abs(unknown[:-1, :] - unknown[1:, :]) ** 2, 0.0))
    total += np.sum(np.where(free[:, :-1] | free[:, 1:],
                             operator.conductance_y * np.abs(unknown[:, :-1] - unknown[:, 1:]) ** 2, 0.0))
    for face, (conductance, rhs) in operator.face_terms.items():
        cells = _FACE_CELLS[face]
        conducting = (conductance != 0) & free[cells]
        safe = np.where(conducting, conductance, 1.0)
        total += np.sum(np.where(conducting, np.abs(conductance * unknown[cells] - rhs) ** 2 / np.conj(safe), 0.0))
    return complex(total)
