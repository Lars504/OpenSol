"""Factor entry counts of a built element mesh from an elimination order made on it: nested dissection of the
elements with every region either cut again or swept front by front, whichever stores less; the a-priori size of a
sparse direct factor of a Lagrange operator on that mesh.

Cut: a region's elements are split at the plane through the mean of their centroids, normal to the axis of largest
spread: the elements lying wholly above it against the rest, so the interface is the mesh surface just above the
plane (a mesh line / layer where there is one) and elements stacked over one footprint stay together (the
tetrahedra of a swept prism column share their footprint, not their centroids: a centroid split tears the column
and its separator takes nodes of all three vertex columns). The nodes both halves share are the separator (S
nodes), eliminated after both halves: their factor columns hold the separator itself and every node of an earlier
separator the region touches (its border, B nodes): S (S + 1) / 2 + S B entries (George's count), plus the two
halves. Sweep: the region's nodes are eliminated in the order of their coordinate
along that axis (a band order), the border last: each node's row reaches back to its first neighbour (the envelope),
each border node's row from its first neighbour to the end; counted in both directions, the smaller kept. A region
costs the smaller of the two, bottom up. Cutting wins on compact 2D / 3D regions, sweeping on strips and thin slabs
(a cut pays its two end borders on every slice: 5 S^2 per element length of a strip against ~3.7 S^2 swept), so the
count follows the shape and grading of the mesh, which no growth law in the unknowns alone can. Both are orders a
solver could use: the count is an upper bound of what a good ordering stores, never a fit."""
import numpy as np

LEAF_ELEMENTS = 48   # a region of at most this many elements is swept, never cut again
MAX_LEVELS = 64      # cut levels at most (a cut leaves both halves smaller: ~log2(elements) levels)
_FAR = np.iinfo(np.int64).max // 4


def element_dissection_pairs(connectivity, node_points) -> int:
    """Factor entries of one triangle (diagonal included) in node pairs for the element -> node connectivity (m, local)
    of a mesh with node coordinates (n_nodes, dim); a vector operator stores (dofs per node)^2 entries per pair."""
    connectivity = np.asarray(connectivity, dtype=np.int64)
    n_elements = int(connectivity.shape[0])
    if n_elements == 0:
        return 0
    local = int(connectivity.shape[1])
    points = np.asarray(node_points, dtype=np.float64)
    n_nodes, dimensions = int(points.shape[0]), int(points.shape[1])
    in_mesh = np.zeros(n_nodes, dtype=bool)
    in_mesh[connectivity] = True
    points = points - points[in_mesh].mean(axis=0)
    scale = float(np.abs(points[in_mesh]).max())
    if scale > 0.0:
        points = points / scale
    element_points = points[connectivity]
    centroids = element_points.mean(axis=1)
    lowest = element_points.min(axis=1)             # an element's lowest coordinate along every axis
    del element_points
    # node -> element incidences of the mesh's nodes, in node order (reduceat segments)
    flat = connectivity.ravel()
    by_node = np.argsort(flat, kind='stable')
    incidence_cells = by_node // local
    mesh_nodes = np.flatnonzero(in_mesh)
    incidence_starts = np.searchsorted(flat[by_node], mesh_nodes)
    mesh_index = np.cumsum(in_mesh) - 1

    marked = np.zeros(n_nodes, dtype=bool)          # nodes of an earlier separator
    position = np.zeros(n_nodes, dtype=np.int64)    # sweep position of a node within its region
    node_region = np.zeros(n_nodes, dtype=np.int64)
    first_cell = np.empty(n_elements, dtype=np.int64)
    last_cell = np.empty(n_elements, dtype=np.int64)
    cell_ids = np.arange(n_elements)                # elements of the regions still being cut
    region = np.zeros(n_elements, dtype=np.int64)   # their region, compact per level
    n_regions = 1
    parents = np.zeros(1, dtype=np.int64)
    levels = []                                     # per level: (parent of each region, sweep cost, cut?, cut cost)
    for _ in range(MAX_LEVELS):
        cells = connectivity[cell_ids]
        interior = np.zeros(n_nodes, dtype=bool)    # nodes of these regions no separator holds
        interior[cells] = True
        interior &= ~marked
        inner = interior[cells]
        keep = inner.any(axis=1)                    # an element of separator nodes only adds nothing
        if not keep.all():
            cell_ids, region, cells, inner = cell_ids[keep], region[keep], cells[keep], inner[keep]
        if cell_ids.size == 0:
            break
        rows = np.arange(cell_ids.size)
        # an interior node lies in one region only (else an earlier separator would hold it)
        node_region[cells] = region[:, None]
        nodes = np.flatnonzero(interior)
        nodes_region = node_region[nodes]
        sizes = np.bincount(nodes_region, minlength=n_regions)
        element_counts = np.bincount(region, minlength=n_regions)
        counts = np.maximum(element_counts, 1).astype(np.float64)
        centres = centroids[cell_ids]
        means = np.stack([np.bincount(region, weights=centres[:, axis], minlength=n_regions) / counts
                          for axis in range(dimensions)], axis=1)
        spreads = np.stack([np.bincount(region, weights=centres[:, axis] ** 2, minlength=n_regions) / counts
                            for axis in range(dimensions)], axis=1) - means ** 2
        axis = np.argmax(spreads, axis=1)           # cut and sweep axis of every region

        # sweep: positions of the interior nodes along their region's axis, each element's first / last position
        order = np.lexsort((points[nodes, axis[nodes_region]], nodes_region))
        starts = np.cumsum(sizes) - sizes
        position[nodes[order]] = np.arange(nodes.size) - starts[nodes_region[order]]
        cell_positions = position[cells]
        first = np.where(inner, cell_positions, _FAR).min(axis=1)
        last = np.where(inner, cell_positions, -1).max(axis=1)
        first_cell[:] = _FAR
        last_cell[:] = -1
        first_cell[cell_ids] = first
        last_cell[cell_ids] = last
        slot = mesh_index[nodes]
        node_first = np.minimum.reduceat(first_cell[incidence_cells], incidence_starts)[slot]
        node_last = np.maximum.reduceat(last_cell[incidence_cells], incidence_starts)[slot]
        node_position = position[nodes]
        ascending = np.bincount(nodes_region, weights=node_position - node_first + 1, minlength=n_regions)
        descending = np.bincount(nodes_region, weights=node_last - node_position + 1, minlength=n_regions)
        # border: the earlier separator nodes each region's elements touch, once per region, with the first / last
        # interior position they are joined to
        borders = np.zeros(n_regions, dtype=np.int64)
        touched = marked[cells]
        keys = (region[:, None] * n_nodes + cells)[touched]
        if keys.size:
            by_key = np.argsort(keys, kind='stable')
            keys = keys[by_key]
            key_rows = np.broadcast_to(rows[:, None], cells.shape)[touched][by_key]
            key_starts = np.flatnonzero(np.concatenate(([True], keys[1:] != keys[:-1])))
            key_region = keys[key_starts] // n_nodes
            borders = np.bincount(key_region, minlength=n_regions)
            ascending += np.bincount(key_region, minlength=n_regions,
                                     weights=sizes[key_region] - np.minimum.reduceat(first[key_rows], key_starts))
            descending += np.bincount(key_region, minlength=n_regions,
                                      weights=np.maximum.reduceat(last[key_rows], key_starts) + 1)
        sweep = np.minimum(ascending, descending)

        # cut: the regions above the leaf size with elements wholly above their mean plane (none: swept as a leaf)
        element_axis = axis[region]
        upper = (element_counts[region] > LEAF_ELEMENTS) & (lowest[cell_ids, element_axis]
                                                            > means[region, element_axis])
        n_upper = np.bincount(region, weights=upper, minlength=n_regions)
        splits = (n_upper > 0) & (n_upper < element_counts)
        active = splits[region]
        cut = np.zeros(n_regions)
        if active.any():
            active_cells, active_upper = cells[active], upper[active]
            below = np.bincount(active_cells[~active_upper].ravel(), minlength=n_nodes) > 0
            above = np.bincount(active_cells[active_upper].ravel(), minlength=n_nodes) > 0
            separator = below & above & ~marked
            separator_sizes = np.bincount(node_region[separator], minlength=n_regions)
            cut = separator_sizes * (separator_sizes + 1) / 2.0 + separator_sizes * borders
            marked |= separator
        levels.append((parents, sweep, splits, cut))
        if not active.any():
            break
        child = 2 * region[active] + active_upper
        used = np.flatnonzero(np.bincount(child, minlength=2 * n_regions))
        relabel = np.zeros(2 * n_regions, dtype=np.int64)
        relabel[used] = np.arange(used.size)
        cell_ids, region = cell_ids[active], relabel[child]
        parents, n_regions = used // 2, int(used.size)

    # bottom up: a region stores the smaller of its sweep and its cut with both halves
    best, child_parents = None, None
    for level_parents, sweep, splits, cut in reversed(levels):
        if best is None:   # the deepest level: no halves counted below it
            total = sweep
        else:
            halves = np.bincount(child_parents, weights=best, minlength=sweep.size)
            total = np.where(splits, np.minimum(sweep, cut + halves), sweep)
        best, child_parents = total, level_parents
    return 0 if best is None else int(round(float(best.sum())))


class MeshDissection:
    """element_dissection_pairs of a built TriangleMesh / TetrahedronMesh per Lagrange order (1: its nodes, 2: nodes
    and edge midpoints) or Nedelec degree (tetrahedra), counted on first use and kept; holds the mesh's arrays."""

    def __init__(self, mesh):
        self._cells = mesh.tets if hasattr(mesh, 'tets') else mesh.triangles
        self._element_edges = mesh.element_edges
        self._edges = mesh.edges
        self._points = mesh.points
        self._faces = getattr(mesh, 'faces', None)                   # tetrahedra only
        self._element_faces = getattr(mesh, 'element_faces', None)
        self.n_points = int(mesh.n_points)
        self.n_edges = int(mesh.n_edges)
        self.n_faces = 0 if self._faces is None else int(self._faces.shape[0])
        self.digest = mesh.digest
        self._pairs = {}

    def __eq__(self, other):
        """Equal for the same mesh (its digest): GUI built counts delivered twice compare equal."""
        return isinstance(other, MeshDissection) and other.digest == self.digest

    __hash__ = None

    def n_nodes(self, order) -> int:
        """Lagrange nodes of the mesh at this order."""
        return self.n_points + (self.n_edges if int(order) == 2 else 0)

    def node_pairs(self, order) -> int:
        """Factor entries of one triangle in node pairs for Lagrange elements of this order (1 or 2)."""
        order = 2 if int(order) == 2 else 1
        if order not in self._pairs:
            cells = np.asarray(self._cells, dtype=np.int64)
            points = np.asarray(self._points, dtype=np.float64)
            if order == 2:
                cells = np.hstack([cells, self.n_points + np.asarray(self._element_edges, dtype=np.int64)])
                points = np.vstack([points, points[np.asarray(self._edges, dtype=np.int64)].mean(axis=1)])
            self._pairs[order] = element_dissection_pairs(cells, points)
        return self._pairs[order]

    def n_nedelec_sites(self, order) -> int:
        """Nedelec dof sites of the mesh: its edges (degree 1), edges and faces (degree 2)."""
        return self.n_edges + (self.n_faces if int(order) == 2 else 0)

    def nedelec_pairs(self, order) -> int:
        """Factor entries of one triangle in dof-site pairs for Nedelec elements of this degree (1: edge sites, 2: edge
        and face sites, at edge midpoints / face centroids; dofs couple exactly when their sites share an element)."""
        key = ('nedelec', 2 if int(order) == 2 else 1)
        if key not in self._pairs:
            points = np.asarray(self._points, dtype=np.float64)
            cells = np.asarray(self._element_edges, dtype=np.int64)
            sites = points[np.asarray(self._edges, dtype=np.int64)].mean(axis=1)
            if key[1] == 2:
                if self._element_faces is None:
                    raise ValueError('Nedelec degree 2 dof sites need a tetrahedron mesh (its faces)')
                cells = np.hstack([cells, self.n_edges + np.asarray(self._element_faces, dtype=np.int64)])
                sites = np.vstack([sites, points[np.asarray(self._faces, dtype=np.int64)].mean(axis=1)])
            self._pairs[key] = element_dissection_pairs(cells, sites)
        return self._pairs[key]
