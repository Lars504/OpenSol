"""Coil drive of magnetostatics_3d (magnetostatics_3d_plan decision 5, section 4.2): T per ampere along the coil axis
from the winding mask, per slice normal to the axis 1 in the enclosed hole, a distance ramp across the winding and 0
outside, over the winding's axial length; on voxel windows (coil_drive_per_ampere) or as Nedelec-1 edge integrals on a
tetrahedron mesh (coil_edge_drive_per_ampere, the ramp from a winding lattice). Host numpy / scipy."""
import math

import numpy as np

_AXES = 'xyz'
_EIGHT_NEIGHBOURS = np.ones((3, 3), dtype=bool)
# tetrahedron coils: the ramp s on a uniform lattice over the winding's bounding box (uncalibrated choices)
LATTICE_PER_ELEMENT = 3          # lattice spacing = winding element size (6 V)^(1/3) / this
LATTICE_MAX_POINTS = 1_000_000   # the spacing grows until the lattice fits (memory estimate: build extra)
LATTICE_PADDING = 2              # lattice rows beyond the winding's transverse extent (the slice-edge rule)
LOCATE_CHUNK = 200_000           # lattice points per point-location call
_EDGE_RULE = np.polynomial.legendre.leggauss(3)   # Gauss-Legendre on the axial part of an edge


def _nearest_distance(points, targets):
    """Distance [m] from each point (n, 2) to its nearest target (m, 2) (k-d tree: any widths)."""
    from scipy.spatial import cKDTree   # scipy only inside functions
    return cKDTree(targets).query(points)[0]


def _slice_ramp(winding, centres, label):
    """s on one slice normal to the axis (2D bool winding, the two transverse centre coordinate arrays [m]): 1 in the
    holes the winding encloses, d_out / (d_in + d_out) on the winding (distances to the nearest hole / outside cell
    bordering it), 0 elsewhere; ValueError for a winding on the slice edge or one enclosing no hole."""
    from scipy import ndimage   # scipy only inside functions
    if winding[0].any() or winding[-1].any() or winding[:, 0].any() or winding[:, -1].any():
        raise ValueError(f'{label}: the winding touches the window edge (it must close around its hole inside the '
                         'window)')
    filled = ndimage.binary_fill_holes(winding)
    hole = filled & ~winding
    if not hole.any():
        raise ValueError(f'{label}: the winding encloses no hole (select a ring around the coil axis)')
    bordering = ndimage.binary_dilation(winding, _EIGHT_NEIGHBOURS)
    grids = np.stack(np.meshgrid(*centres, indexing='ij'), axis=-1)
    points = grids[winding]
    distance_in = _nearest_distance(points, grids[hole & bordering])
    distance_out = _nearest_distance(points, grids[bordering & ~filled])
    ramp = hole.astype(np.float64)
    ramp[winding] = distance_out / (distance_in + distance_out)
    return ramp


def coil_drive_per_ampere(winding_window, axis, widths, turns, label):
    """(T per ampere [1/m] float64 window array along axis 'x' / 'y' / 'z', axial length h [m]) of a stranded coil on
    the host bool window mask: turns / h * s per slice normal to the axis (identical slices computed once); h = the
    axis widths (widths: the three window width arrays [m]) of the slices holding winding cells."""
    axis_index = _AXES.index(axis)
    winding = np.moveaxis(np.asarray(winding_window, dtype=bool), axis_index, -1)
    transverse_widths = [np.asarray(widths[index], dtype=np.float64) for index in range(3) if index != axis_index]
    centres = [np.cumsum(width) - 0.5 * width for width in transverse_widths]
    axial_widths = np.asarray(widths[axis_index], dtype=np.float64)
    ramp = np.zeros(winding.shape)
    ramps_by_slice = {}
    length = 0.0
    for index in range(winding.shape[-1]):
        winding_slice = np.ascontiguousarray(winding[..., index])
        if not winding_slice.any():
            continue
        key = winding_slice.tobytes()
        if key not in ramps_by_slice:
            ramps_by_slice[key] = _slice_ramp(winding_slice, centres, f'{label}, window slice {index} along {axis}')
        ramp[..., index] = ramps_by_slice[key]
        length += float(axial_widths[index])
    if length == 0.0:
        raise ValueError(f'{label} selects no window cell')
    return np.moveaxis(ramp * (turns / length), -1, axis_index), length


def _winding_lattice(mesh, winding, axis_index, low, high):
    """Per-axis float64 lattice centres [m] over the winding's bounding box (low, high): transverse spacing d plus
    LATTICE_PADDING rows each side, centres offset half a spacing from the box (off the element faces of aligned
    meshes); axial >= 2 centres inside [low, high]; d grows until LATTICE_MAX_POINTS fit."""
    volumes = mesh.volumes(np.flatnonzero(winding))
    spacing = float(np.cbrt(6.0 * np.median(volumes))) / LATTICE_PER_ELEMENT
    while True:
        axes = []
        for index in range(3):
            extent = float(high[index] - low[index])
            if index == axis_index:
                count = max(2, int(math.ceil(extent / spacing)))
                axes.append(low[index] + (np.arange(count) + 0.5) * (extent / count))
            else:
                count = int(math.ceil(extent / spacing)) + 2 * LATTICE_PADDING
                axes.append(low[index] + (np.arange(count) - LATTICE_PADDING + 0.5) * spacing)
        total = math.prod(len(centres) for centres in axes)
        if total <= LATTICE_MAX_POINTS:
            return axes
        spacing *= 1.01 * (total / LATTICE_MAX_POINTS) ** (1.0 / 3.0)


def coil_edge_drive_per_ampere(mesh, winding_elements, axis, turns, label, locate):
    """(Nedelec-1 dofs (n_edges,) of T per ampere: per mesh edge the line integral (lower -> higher node) of turns / h
    s along the axis [1], axial length h [m]) of a stranded coil on a host bool element mask of a TetrahedronMesh: s
    by the slice ramp on a lattice over the winding (a lattice point in a winding element counts as winding; locate:
    points -> (elements, reference)), trilinear between lattice points, 0 beyond the winding's axial extent h (its
    nodes). The circulation of T around any mesh-edge loop is then turns x the current it encloses."""
    from scipy.interpolate import RegularGridInterpolator
    axis_index = _AXES.index(axis)
    winding = np.asarray(winding_elements, dtype=bool)
    if not winding.any():
        raise ValueError(f'{label} selects no window element')
    corners = mesh.points[np.unique(mesh.tets[winding])]
    low, high = corners.min(axis=0), corners.max(axis=0)
    length = float(high[axis_index] - low[axis_index])
    if not length > 0.0:
        raise ValueError(f'{label}: the winding has no extent along the axis {axis}')
    axes = _winding_lattice(mesh, winding, axis_index, low, high)
    grids = np.meshgrid(*axes, indexing='ij')
    points = np.stack([grid.ravel() for grid in grids], axis=1)
    inside = np.zeros(len(points), dtype=bool)
    for start in range(0, len(points), LOCATE_CHUNK):
        elements, _ = locate(points[start:start + LOCATE_CHUNK])
        inside[start:start + LOCATE_CHUNK] = (elements >= 0) & winding[np.maximum(elements, 0)]
    lattice_winding = np.moveaxis(inside.reshape(grids[0].shape), axis_index, -1)
    transverse = [axes[index] for index in range(3) if index != axis_index]
    ramp = np.zeros(lattice_winding.shape)
    ramps_by_slice = {}   # identical lattice slices computed once (as the voxel path)
    for index in range(lattice_winding.shape[-1]):
        winding_slice = np.ascontiguousarray(lattice_winding[..., index])
        if not winding_slice.any():
            raise ValueError(f'{label}: the winding has a gap along {axis} near {axes[axis_index][index]:.6g} m (on '
                             'a tetrahedron mesh a coil winding must be one piece along its axis)')
        key = winding_slice.tobytes()
        if key not in ramps_by_slice:
            ramps_by_slice[key] = _slice_ramp(winding_slice, transverse,
                                              f'{label}, lattice slice {index} along {axis}')
        ramp[..., index] = ramps_by_slice[key]
    interpolate = RegularGridInterpolator(axes, np.moveaxis(ramp, -1, axis_index), method='linear',
                                          bounds_error=False, fill_value=0.0)
    # edges with an axial part inside [low, high]: the clipped parameter range of each, Gauss points on it
    start_points, end_points = mesh.points[mesh.edges[:, 0]], mesh.points[mesh.edges[:, 1]]
    rise = end_points[:, axis_index] - start_points[:, axis_index]
    safe_rise = np.where(rise != 0.0, rise, 1.0)
    bounds = np.stack([(low[axis_index] - start_points[:, axis_index]) / safe_rise,
                       (high[axis_index] - start_points[:, axis_index]) / safe_rise], axis=1)
    t_low, t_high = np.clip(bounds.min(axis=1), 0.0, 1.0), np.clip(bounds.max(axis=1), 0.0, 1.0)
    candidates = np.flatnonzero((rise != 0.0) & (t_high > t_low))
    per_ampere = np.zeros(mesh.n_edges)
    if candidates.size == 0:
        return per_ampere, length
    nodes, weights = _EDGE_RULE
    span = t_high[candidates] - t_low[candidates]
    parameters = t_low[candidates, None] + 0.5 * span[:, None] * (nodes[None, :] + 1.0)   # (k, 3)
    direction = end_points[candidates] - start_points[candidates]
    samples = start_points[candidates, None, :] + parameters[..., None] * direction[:, None, :]
    samples[..., axis_index] = np.clip(samples[..., axis_index], axes[axis_index][0], axes[axis_index][-1])
    ramp_values = interpolate(samples.reshape(-1, 3)).reshape(samples.shape[:2])
    per_ampere[candidates] = (turns / length) * rise[candidates] * 0.5 * span * (ramp_values @ weights)
    return per_ampere, length
