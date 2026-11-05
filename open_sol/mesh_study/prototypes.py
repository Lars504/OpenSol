"""S7.8 external prototypes of the coax capacitor (CPU, numpy + scipy) -> records/accuracy_coax.jsonl beside the
tensor runs: cell-centred FV on a 2:1 balanced quadtree (S7.4) and P1 FEM on boundary-conforming triangles (S7.5).
Quadtree rules as DiffusionSystem: cells held by their centre (core r <= a, shield r > b), two-point fluxes eps * face /
centre distance, a held-free face takes the free half cell; hanging faces use the normal centre distance
(h_fine + h_coarse) / 2 (the simple, first-order-inconsistent choice). 'quadtree_uniform' refines every cell and must
reproduce the tensor uniform runs (prototype check). FEM: fem_polar (structured polar triangles, nodes on both
circles) and fem_delaunay (scipy Delaunay of boundary points + a jittered lattice: a stand-in for a generic mesher);
C' = 2 W / V0^2."""
import math
import time

import numpy as np

from ..core.constants import EPS0
from .common import MM, append_record, labels_to_skip, record_file, selected_cases

A_CORE, B_SHIELD, L_BOX = 0.5 * MM, 2.0 * MM, 5.0 * MM
CENTRE = L_BOX / 2
EPS_FILL, V0 = 2.1, 1.0
N_ROOT = 25                     # root cells per side: level l gives a / h = 2.5 * 2^l (l = 3: the example's a / 20)
C_EXACT = 2 * math.pi * EPS0 * EPS_FILL / math.log(B_SHIELD / A_CORE)
QUADTREE_LEVELS = (1, 2, 3, 4, 5, 6)
QUADTREE_UNIFORM_MAX_LEVEL = 4  # pure Python: the uniform tree at level 6 (2.6e6 cells) is slow
QUADTREE_BUFFER = 2.0           # refinement band half-width in cells of each level
FEM_A_OVER_H = (5, 10, 20, 40, 80, 160)


# ---- quadtree FV ----
def cell_size(level):
    return L_BOX / (N_ROOT * 2 ** level)


def near_circles(cell, buffer_cells):
    """True when r = a or r = b passes within buffer_cells cell widths of the cell square."""
    level, i, j = cell
    h = cell_size(level)
    x0, y0 = i * h - CENTRE, j * h - CENTRE
    r_min = math.hypot(max(x0, 0.0, -(x0 + h)), max(y0, 0.0, -(y0 + h)))
    r_max = max(math.hypot(x, y) for x in (x0, x0 + h) for y in (y0, y0 + h))
    pad = buffer_cells * h
    return any(r_min - pad <= radius <= r_max + pad for radius in (A_CORE, B_SHIELD))


def split(leaves, cell):
    level, i, j = cell
    leaves.discard(cell)
    leaves.update({(level + 1, 2 * i + di, 2 * j + dj) for di in (0, 1) for dj in (0, 1)})


def find_leaf(leaves, level, i, j):
    """Leaf covering cell (level, i, j) at that level or coarser; None when it is refined finer or outside."""
    n_side = N_ROOT * 2 ** level
    if not (0 <= i < n_side and 0 <= j < n_side):
        return None
    for coarser in range(level, -1, -1):
        shift = level - coarser
        key = (coarser, i >> shift, j >> shift)
        if key in leaves:
            return key
    return None


def build_tree(level_max, buffer_cells, uniform):
    leaves = {(0, i, j) for i in range(N_ROOT) for j in range(N_ROOT)}
    for level in range(level_max):
        for cell in [cell for cell in leaves if cell[0] == level and (uniform or near_circles(cell, buffer_cells))]:
            split(leaves, cell)
    while not uniform:   # 2:1 face balance
        to_split = set()
        for level, i, j in leaves:
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbour = find_leaf(leaves, level, i + di, j + dj)
                if neighbour is not None and neighbour[0] < level - 1:
                    to_split.add(neighbour)
        if not to_split:
            break
        for cell in to_split:
            split(leaves, cell)
    return leaves


def cell_state(cell):
    """V0 (core), 0.0 (shield) or None (free) by the cell centre."""
    level, i, j = cell
    h = cell_size(level)
    radius = math.hypot((i + 0.5) * h - CENTRE, (j + 0.5) * h - CENTRE)
    if radius <= A_CORE:
        return V0
    return 0.0 if radius > B_SHIELD else None


def faces(leaves):
    """(cell, neighbour, face length, normal centre distance) of every face once (from the finer or the +x/+y side)."""
    for cell in leaves:
        level, i, j = cell
        h = cell_size(level)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbour = find_leaf(leaves, level, i + di, j + dj)
            if neighbour is None:
                continue
            if neighbour[0] == level and (di + dj) > 0:
                yield cell, neighbour, h, h
            elif neighbour[0] == level - 1:
                yield cell, neighbour, h, 0.5 * (h + cell_size(level - 1))


def quadtree_capacitance(leaves):
    """C' per m (core charge / V0) and the free-cell count."""
    import scipy.sparse as sparse
    import scipy.sparse.linalg as sparse_linalg
    states = {cell: cell_state(cell) for cell in leaves}
    free = {cell: index for index, cell in enumerate(cell for cell in leaves if states[cell] is None)}
    rows, cols, values = [], [], []
    rhs = np.zeros(len(free))
    core_faces = []
    eps = EPS0 * EPS_FILL
    for cell, neighbour, length, distance in faces(leaves):
        first, second = states[cell], states[neighbour]
        if first is None and second is None:
            conductance = eps * length / distance
            p, q = free[cell], free[neighbour]
            rows += [p, q, p, q]
            cols += [p, q, q, p]
            values += [conductance, conductance, -conductance, -conductance]
        elif (first is None) != (second is None):
            free_cell, held_value = (cell, second) if first is None else (neighbour, first)
            conductance = eps * length / (0.5 * cell_size(free_cell[0]))
            p = free[free_cell]
            rows.append(p)
            cols.append(p)
            values.append(conductance)
            rhs[p] += conductance * held_value
            if held_value == V0:
                core_faces.append((p, conductance))
    matrix = sparse.csr_matrix((values, (rows, cols)), shape=(len(free), len(free)))
    potential = sparse_linalg.spsolve(matrix.tocsc(), rhs)
    charge = sum(conductance * (V0 - potential[p]) for p, conductance in core_faces)
    return charge / V0, len(free)


# ---- P1 FEM ----
def polar_mesh(h):
    """(points, triangles): n_r radial intervals, n_theta sectors at the mean radius, each quad split in two."""
    n_r = max(1, math.ceil((B_SHIELD - A_CORE) / h))
    n_theta = max(8, math.ceil(math.pi * (A_CORE + B_SHIELD) / h))
    radii = np.linspace(A_CORE, B_SHIELD, n_r + 1)
    angles = np.linspace(0.0, 2 * math.pi, n_theta, endpoint=False)
    radius_grid, angle_grid = np.meshgrid(radii, angles, indexing='ij')
    points = np.column_stack([(radius_grid * np.cos(angle_grid)).ravel(), (radius_grid * np.sin(angle_grid)).ravel()])

    def node(i, j):
        return i * n_theta + (j % n_theta)
    triangles = []
    for i in range(n_r):
        for j in range(n_theta):
            triangles += [(node(i, j), node(i + 1, j), node(i + 1, j + 1)),
                          (node(i, j), node(i + 1, j + 1), node(i, j + 1))]
    return points, np.array(triangles)


def delaunay_mesh(h, seed=0):
    """(points, triangles) of boundary circles at spacing h plus a jittered interior lattice kept h/2 off them."""
    from scipy.spatial import Delaunay
    rng = np.random.default_rng(seed)
    boundary = []
    for radius in (A_CORE, B_SHIELD):
        n = max(8, math.ceil(2 * math.pi * radius / h))
        angles = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
        boundary.append(np.column_stack([radius * np.cos(angles), radius * np.sin(angles)]))
    axis = np.arange(-B_SHIELD, B_SHIELD + h, h)
    lattice = np.array(np.meshgrid(axis, axis)).reshape(2, -1).T + rng.uniform(-0.2 * h, 0.2 * h, (axis.size ** 2, 2))
    radius = np.hypot(lattice[:, 0], lattice[:, 1])
    interior = lattice[(radius > A_CORE + 0.5 * h) & (radius < B_SHIELD - 0.5 * h)]
    points = np.vstack(boundary + [interior])
    triangles = Delaunay(points).simplices
    centroid_radius = np.hypot(*points[triangles].mean(axis=1).T)
    return points, triangles[(centroid_radius > A_CORE) & (centroid_radius < B_SHIELD)]


def fem_capacitance(points, triangles):
    """C' per m and the free-node count of the P1 solve with V0 on r = a and 0 on r = b."""
    import scipy.sparse as sparse
    import scipy.sparse.linalg as sparse_linalg
    corners = points[triangles]                                          # (n_t, 3, 2)
    edges = np.roll(corners, -1, axis=1) - np.roll(corners, 1, axis=1)   # edge opposite each vertex
    area = 0.5 * np.abs(edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0])
    local = EPS0 * EPS_FILL * np.einsum('tid,tjd->tij', edges, edges) / (4 * area)[:, None, None]
    rows = np.repeat(triangles, 3, axis=1).ravel()
    cols = np.tile(triangles, (1, 3)).ravel()
    stiffness = sparse.csr_matrix((local.ravel(), (rows, cols)), shape=(len(points),) * 2)
    radius = np.hypot(points[:, 0], points[:, 1])
    on_core = np.abs(radius - A_CORE) < 1e-9 * B_SHIELD
    on_shield = np.abs(radius - B_SHIELD) < 1e-9 * B_SHIELD
    potential = np.where(on_core, V0, 0.0)
    used = np.zeros(len(points), bool)
    used[triangles.ravel()] = True
    free = ~(on_core | on_shield) & used
    rhs = -stiffness[free][:, ~free] @ potential[~free]
    potential[free] = sparse_linalg.spsolve(stiffness[free][:, free].tocsc(), rhs)
    energy = 0.5 * potential @ (stiffness @ potential)
    return 2 * energy / V0 ** 2, int(free.sum())


# ---- runs ----
def _record(path, family, label, h, n_cells, n_unknowns, seconds, capacitance, parameters):
    error = capacitance / C_EXACT - 1.0
    append_record(path, {'case': 'coax', 'family': family, 'label': label, 'h': h, 'status': 'completed',
                         'n_cells': n_cells, 'n_unknowns': n_unknowns, 'wall_time': seconds, 'study_seconds': seconds,
                         'derived': {'C_per_m': capacitance}, 'derived_reference': {'C_per_m': C_EXACT},
                         'derived_error': {'C_per_m': abs(error)}, 'parameters': parameters})
    print(f"[coax] {label}: a/h {A_CORE / h:.1f}, {n_cells:,} cells, {n_unknowns:,} unknowns, C' {capacitance:.6e} "
          f'F/m, error {error:+.3e}, {seconds:.1f} s')


def run(options):
    if not selected_cases(options, 'prototypes', ('coax',)):
        return
    path = record_file(options, 'accuracy', 'coax')
    skip = labels_to_skip(path, options)
    for family, uniform in (('quadtree', False), ('quadtree_uniform', True)):
        for level in QUADTREE_LEVELS:
            h = cell_size(level)
            label = f'{family}_h{h:.4g}_buffer{QUADTREE_BUFFER:g}'
            if label in skip or (uniform and level > QUADTREE_UNIFORM_MAX_LEVEL):
                continue
            start = time.perf_counter()
            leaves = build_tree(level, QUADTREE_BUFFER, uniform)
            capacitance, n_free = quadtree_capacitance(leaves)
            _record(path, family, label, h, len(leaves), n_free, time.perf_counter() - start, capacitance,
                    {'level': level, 'buffer': QUADTREE_BUFFER})
    for family, mesher in (('fem_polar', polar_mesh), ('fem_delaunay', delaunay_mesh)):
        for n in FEM_A_OVER_H:
            h = A_CORE / n
            label = f'{family}_h{h:.4g}'
            if label in skip:
                continue
            start = time.perf_counter()
            points, triangles = mesher(h)
            capacitance, n_free = fem_capacitance(points, triangles)
            _record(path, family, label, h, len(triangles), n_free, time.perf_counter() - start, capacitance,
                    {'a_over_h': n})
