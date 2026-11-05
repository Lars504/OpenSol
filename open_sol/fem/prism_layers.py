"""Swept prism layers (experiment plan D5): a triangle mesh extruded through z layers, each prism split into 3
tetrahedra by the minimum-vertex rule (conforming between prisms, periodic sides matched by an order key). Host numpy;
coordinates in m."""
import numpy as np

from .mesh3d import PLANE_TOLERANCE, build_tetrahedron_mesh


def _prism_permutations() -> np.ndarray:
    """(6, 6) int: row m = the prism vertex order (bottom a b c, top a' b' c' above them) whose first vertex is m (a
    rotation of the triangle, the layers swapped when m is on top)."""
    rows = []
    for first in range(6):
        layer, turn = divmod(first, 3)
        bottom = [3 * layer + (turn + step) % 3 for step in range(3)]
        rows.append(bottom + [3 * (1 - layer) + (turn + step) % 3 for step in range(3)])
    return np.array(rows, dtype=np.int64)


_PRISM_PERMUTATIONS = _prism_permutations()


def split_prisms(prisms, keys) -> np.ndarray:
    """(3 m, 4) int64 tetrahedra of prisms (m, 6) [bottom a b c, top a' b' c'] (tetrahedra 3 p .. 3 p + 2 of prism p):
    each quad face split by the diagonal from its smallest-key vertex (Dompierre et al. 1999), so prisms sharing a face
    agree; keys (m, 6) distinct within each row (ValueError otherwise)."""
    prisms = np.asarray(prisms, dtype=np.int64).reshape(-1, 6)
    keys = np.asarray(keys, dtype=np.int64).reshape(-1, 6)
    ordered = np.sort(keys, axis=1)
    repeated = (ordered[:, 1:] == ordered[:, :-1]).any(axis=1)
    if repeated.any():
        raise ValueError(f'split_prisms: {int(repeated.sum())} prism(s) repeat an order key, e.g. prism '
                         f'{int(np.flatnonzero(repeated)[0])} (a triangle touching both sides of a periodic axis?)')
    order = _PRISM_PERMUTATIONS[np.argmin(keys, axis=1)]
    v = np.take_along_axis(prisms, order, axis=1)   # v0 = the smallest vertex, v3 above / below it
    k = np.take_along_axis(keys, order, axis=1)
    # the quad face opposite v0 (v1 v2 v5 v4): diagonal v1-v5 when its smallest vertex is v1 or v5, else v2-v4
    diagonal_15 = (np.minimum(k[:, 1], k[:, 5]) < np.minimum(k[:, 2], k[:, 4]))[:, None]
    first = np.where(diagonal_15, v[:, [0, 1, 2, 5]], v[:, [0, 1, 2, 4]])
    second = np.where(diagonal_15, v[:, [0, 1, 5, 4]], v[:, [0, 2, 4, 5]])
    third = v[:, [0, 3, 4, 5]]
    return np.stack([first, second, third], axis=1).reshape(-1, 4)


def periodic_order_key(points_2d, periodic_axes) -> np.ndarray:
    """int64 (n,) order key of 2D nodes: each node on the hi side of a periodic axis ('x' / 'y') takes the key of its
    lo-side partner (corners: the lo-lo node), else its own index, so the split picks matching diagonals on both sides;
    ValueError when the sides' nodes do not match."""
    from scipy.spatial import cKDTree
    points_2d = np.asarray(points_2d, dtype=np.float64).reshape(-1, 2)
    keys = np.arange(len(points_2d), dtype=np.int64)
    low, high = points_2d.min(axis=0), points_2d.max(axis=0)
    slack = PLANE_TOLERANCE * float((high - low).max())
    for name in periodic_axes:
        axis = 'xy'.index(name)
        lo_nodes = np.flatnonzero(np.abs(points_2d[:, axis] - low[axis]) <= slack)
        hi_nodes = np.flatnonzero(np.abs(points_2d[:, axis] - high[axis]) <= slack)
        if len(lo_nodes) != len(hi_nodes) or not len(lo_nodes):
            raise ValueError(f'periodic {name}: {len(lo_nodes)} nodes on the {name}_lo side, {len(hi_nodes)} on '
                             f'{name}_hi; the triangle mesh is not periodic')
        distance, nearest = cKDTree(points_2d[lo_nodes][:, [1 - axis]]).query(points_2d[hi_nodes][:, [1 - axis]])
        if (distance > slack).any() or len(np.unique(nearest)) != len(hi_nodes):
            raise ValueError(f'periodic {name}: {int((distance > slack).sum())} {name}_hi nodes have no {name}_lo '
                             'node at the same position; the triangle mesh is not periodic')
        keys[hi_nodes] = keys[lo_nodes[nearest]]   # a corner already mapped along x keeps that chain
    return keys


def swept_tetrahedron_mesh(points_2d, triangles, z_edges, window=None, order_key=None, region=None, region_names=(),
                           entities=None):
    """fem.mesh3d.TetrahedronMesh of a triangle mesh (points (n, 2), triangles (t, 3)) extruded through the layers of
    increasing z_edges (L + 1,): node k n + i = (point i, z_k); element 3 p + j belongs to prism p = layer t + triangle;
    order_key (n,) int (periodic_order_key; None = the node index) ranks the 2D nodes, layers rank above within one;
    window / region / region_names / entities as build_tetrahedron_mesh."""
    points_2d = np.asarray(points_2d, dtype=np.float64).reshape(-1, 2)
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    z_edges = np.asarray(z_edges, dtype=np.float64).ravel()
    if len(z_edges) < 2 or not (np.diff(z_edges) > 0.0).all():
        raise ValueError('swept_tetrahedron_mesh needs increasing z edges with >= 2 entries')
    n_points, n_layers = len(points_2d), len(z_edges) - 1
    points = np.column_stack([np.tile(points_2d, (n_layers + 1, 1)), np.repeat(z_edges, n_points)])
    key_2d = (np.arange(n_points, dtype=np.int64) if order_key is None
              else np.asarray(order_key, dtype=np.int64).reshape(n_points))
    node_keys = (np.tile(key_2d, n_layers + 1) * (n_layers + 1)
                 + np.repeat(np.arange(n_layers + 1, dtype=np.int64), n_points))
    lower = triangles[None, :, :] + (np.arange(n_layers, dtype=np.int64) * n_points)[:, None, None]
    prisms = np.concatenate([lower, lower + n_points], axis=2).reshape(-1, 6)
    tets = split_prisms(prisms, node_keys[prisms])
    return build_tetrahedron_mesh(points, tets, window=window, region=region, region_names=region_names,
                                  entities=entities)


def swept_counts(n_triangles, n_points, n_edges, n_layers) -> dict:
    """{'elements', 'nodes', 'edges', 'faces'} of a swept mesh from its triangle mesh counts: 3 t L tetrahedra,
    n (L + 1) nodes, e (L + 1) + n L + e L edges (one diagonal per quad face, none inside a prism), faces by Euler."""
    elements = 3 * n_triangles * n_layers
    nodes = n_points * (n_layers + 1)
    edges = n_edges * (n_layers + 1) + n_points * n_layers + n_edges * n_layers
    return {'elements': elements, 'nodes': nodes, 'edges': edges, 'faces': 1 - nodes + edges + elements}
