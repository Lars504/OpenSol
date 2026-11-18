"""Bloch / Floquet unit cells of the 2D solid (solid_mechanics, periodic_x / periodic_y): the periodic identification of
the tensor mesh's Q1 corner nodes or of a triangle Lagrange map as a bloch3d.BlochCell3D (z never periodic), and the
rigid translations a periodic solid region keeps. Host numpy (scipy inside functions)."""
# The phase convention is bloch3d's (fem.dofs.bloch_master_map): u(x + L) = u(x) e^{-j k L} with exp(+j omega t), every
# component of a node (u_x, u_y, the guided u_z amplitude, phi) shares the node's master and phase, K_r = P^H K P and
# M_r = P^H M P (complex Hermitian; real when every k L is a multiple of pi). A solid region that wraps through a
# periodic side cannot rotate (u_y(x + L) = u_y(x) forbids w L); at a trivial phase (every periodic phase 1) it keeps
# the translations its held dofs leave free, at any other phase none. A region that touches no periodic partner keeps
# the ordinary rules of its system.
import numpy as np

from ...fem.dofs import bloch_master_map, bloch_masters
from .bloch3d import BlochCell3D

AXES_2D = ('x', 'y')


def _cell(master, phase, periodic, wavenumbers) -> BlochCell3D:
    """BlochCell3D of 2D masters (phase within 1e-15 of real made real by bloch_master_map; float64 when real)."""
    if not np.any(np.asarray(phase).imag):
        phase = np.ascontiguousarray(np.asarray(phase).real)
    trivial = bool(np.all(phase == 1.0))
    return BlochCell3D(master=master, phase=phase, wavenumbers=(*wavenumbers, 0.0), periodic=(*periodic, False),
                       trivial=trivial)


def _checked(periodic, wavenumbers) -> tuple:
    periodic = tuple(bool(value) for value in periodic)
    if len(periodic) != 2 or not any(periodic):
        raise ValueError(f'a 2D Bloch cell needs (periodic_x, periodic_y) with at least one True, got {periodic!r}')
    wavenumbers = tuple(float(value) if periodic[axis] else 0.0 for axis, value in enumerate(wavenumbers))
    if not all(np.isfinite(wavenumbers)):
        raise ValueError(f'bloch_k must be finite wavenumbers [rad/m], got {wavenumbers!r}')
    return periodic, wavenumbers


def tensor_bloch_cell(node_shape, extents, periodic, wavenumbers) -> BlochCell3D:
    """BlochCell3D over the Q1 corner nodes of a tensor window (node = i ny_nodes + j, i along x): the x_hi column
    stands for x_lo, the y_hi row for y_lo (corners chain both), phase e^{-j k L} per crossed period with L = extents
    [m] (the window size along that axis)."""
    periodic, wavenumbers = _checked(periodic, wavenumbers)
    nx_nodes, ny_nodes = (int(count) for count in node_shape)
    nodes = np.arange(nx_nodes * ny_nodes, dtype=np.int64).reshape(nx_nodes, ny_nodes)

    def partners_of(axis):
        partners = np.full(nodes.size, -1, dtype=np.int64)
        if axis == 0:
            partners[nodes[-1, :]] = nodes[0, :]
        else:
            partners[nodes[:, -1]] = nodes[:, 0]
        return partners
    master, phase = bloch_master_map(nodes.size, tuple(float(value) for value in extents),
                                     {axis: wavenumbers[axis] for axis in range(2) if periodic[axis]}, partners_of)
    return _cell(master, phase, periodic, wavenumbers)


def triangle_bloch_cell(dofs, periodic, wavenumbers) -> BlochCell3D:
    """BlochCell3D of a triangle Lagrange DofMap (fem.dofs.bloch_masters: L = the node extent along the axis);
    ValueError when the two sides of a periodic axis are meshed differently or the map's elements do not reach both
    sides alike."""
    periodic, wavenumbers = _checked(periodic, wavenumbers)
    master, phase = bloch_masters(dofs, {axis: wavenumbers[axis] for axis in range(2) if periodic[axis]})
    return _cell(master, phase, periodic, wavenumbers)


def joined_regions(labels, master) -> np.ndarray:
    """Region label per dof / node after joining the regions of periodic partners (labels int (n,), < 0 = no region;
    returned labels are 0-based compact, -1 kept)."""
    import scipy.sparse
    from scipy.sparse.csgraph import connected_components
    labels = np.asarray(labels, dtype=np.int64)
    inside = labels >= 0
    if not inside.any():
        return labels.copy()
    names, compact = np.unique(labels[inside], return_inverse=True)
    region = np.full(labels.size, -1, dtype=np.int64)
    region[inside] = compact
    moved = np.flatnonzero((master != np.arange(master.size)) & inside)
    moved = moved[region[master[moved]] >= 0]
    graph = scipy.sparse.coo_matrix((np.ones(moved.size, dtype=np.int8), (region[moved], region[master[moved]])),
                                    shape=(names.size, names.size))
    _, joined = connected_components(graph, directed=False)
    region[inside] = joined[region[inside]]
    return region


def wrapping_regions(region, master) -> np.ndarray:
    """Bool per region (0 .. region.max()) that holds a dof / node standing for another (it wraps through a side)."""
    count = int(region.max()) + 1 if region.size and region.max() >= 0 else 0
    moved = (master != np.arange(master.size)) & (region >= 0)
    return np.bincount(region[moved], minlength=count) > 0


def free_translations(region, held, wraps) -> np.ndarray:
    """Free translation components per wrapping region of a trivial-phase cell: the components (u_x, u_y; 2 columns of
    held (n, c) bool) no member dof holds (0 for a region that does not wrap)."""
    count = wraps.size
    free = np.zeros(count, dtype=np.int64)
    inside = region >= 0
    for component in range(2):
        held_any = np.bincount(region[inside & held[:, component]], minlength=count) > 0
        free += (~held_any).astype(np.int64)
    return np.where(wraps, free, 0)
