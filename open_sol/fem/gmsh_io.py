"""Optional gmsh mesher (decision D8): the PSLG written as a .geo (normalised coordinates, the size field as a
background view), `gmsh -2` run as an external process (never imported, linked or vendored), .msh 4.1 ASCII read
back into a RawMesh. Host numpy, scipy inside functions."""
import math
import os
import shutil
import subprocess
import tempfile

import numpy as np

from .mesher import MERGE_TOLERANCE, RawMesh

GMSH_ENVIRONMENT = 'OPEN_SOL_GMSH'   # path of the gmsh executable; else `gmsh` on PATH
GMSH_TIMEOUT = 3600.0                # s
_FACE_TAGS = frozenset(('x_lo', 'x_hi', 'y_lo', 'y_hi'))
_VIEW_CELLS = 400                    # background-view cells along the longest side at most
_TRIANGLE_TYPES = {2: 3, 9: 6}       # msh element type -> nodes (3-node / 6-node triangles)
_LINE_TYPES = {1: 2, 8: 3}
# written into every .geo (here and fem.gmsh3d): gmsh reads user option files (~/.gmsh-options, .gmshrc), and a saved
# scaling factor, binary or parametric output would silently change what the readers parse
PINNED_OUTPUT_OPTIONS = ('Mesh.ScalingFactor = 1;', 'Geometry.ScalingFactor = 1;', 'Mesh.Binary = 0;',
                         'Mesh.SaveParametric = 0;')


def geo_number(value) -> str:
    """Plain decimal text of a number for a .geo / .pos file: repr(float), never a numpy repr (numpy 2 writes
    'np.float64(0.25)', which gmsh reads as an unknown option: every point at 0, booleans failing)."""
    return repr(float(value))


def find_gmsh():
    """Path of the gmsh executable (environment OPEN_SOL_GMSH, else gmsh on PATH); None when not found."""
    configured = os.environ.get(GMSH_ENVIRONMENT)
    if configured:
        return configured if os.path.isfile(configured) else shutil.which(configured)
    return shutil.which('gmsh')


def mesh_with_gmsh(pslg, size_field, executable=None, timeout=GMSH_TIMEOUT) -> RawMesh:
    """RawMesh of the PSLG by an external gmsh run (Frontal-Delaunay, linear elements: P2 midside nodes and curve
    projection are done by open_sol); RuntimeError when gmsh is missing or fails."""
    executable = executable or find_gmsh()
    if executable is None:
        raise RuntimeError(f'mesher gmsh: executable not found (set {GMSH_ENVIRONMENT} or put gmsh on PATH)')
    with tempfile.TemporaryDirectory(prefix='open_sol_gmsh_') as folder:
        write_size_view(os.path.join(folder, 'size.pos'), pslg.window, size_field)
        write_geo(os.path.join(folder, 'model.geo'), pslg, 'size.pos')
        completed = subprocess.run([executable, 'model.geo', '-2', '-format', 'msh41', '-o', 'model.msh', '-v', '2'],
                                   cwd=folder, capture_output=True, text=True, timeout=timeout)
        msh_path = os.path.join(folder, 'model.msh')
        if completed.returncode != 0 or not os.path.isfile(msh_path):
            raise RuntimeError(f'mesher gmsh: exit code {completed.returncode}; gmsh says: {gmsh_errors(completed)}')
        try:
            nodes, triangles, line_blocks = read_msh(msh_path)
        except (ValueError, IndexError) as exc:   # gmsh may report a .geo error and still exit 0
            raise RuntimeError(f'mesher gmsh: unreadable mesh ({exc!r}); gmsh says: {gmsh_errors(completed)}') from None
    try:
        return _raw_mesh(pslg, nodes, triangles, line_blocks)
    except RuntimeError as exc:   # say what gmsh wrote: its coordinates and messages
        low, high = nodes.min(axis=0), nodes.max(axis=0)
        raise RuntimeError(f'{exc}; gmsh wrote {len(nodes)} nodes for {len(pslg.vertices)} PSLG vertices, normalised '
                           f'x [{low[0]:.4g}, {high[0]:.4g}], y [{low[1]:.4g}, {high[1]:.4g}] (the window is [0, 1] '
                           f'along its longest side); gmsh says: {gmsh_errors(completed)}') from None


def gmsh_errors(completed) -> str:
    """gmsh's 'Error' lines of a finished run (the last 5 output lines when there are none), joined by ' | '."""
    lines = [line.strip() for line in f'{completed.stdout or ""}\n{completed.stderr or ""}'.splitlines() if line.strip()]
    errors = [line for line in lines if 'error' in line.lower()]
    return ' | '.join((errors or lines)[-5:]) or '(no output)'


def _frame(window) -> tuple:
    """(origin (2,), scale) of the normalised coordinates (window corner 0, longest side 1)."""
    x0, y0, x1, y1 = window
    return np.array([x0, y0], dtype=np.float64), max(x1 - x0, y1 - y0)


def write_size_view(path, window, size_field):
    """Gmsh post-processing view (scalar triangles ST) of h / scale on a regular grid of the window (written by one
    np.savetxt, %.17g: plain decimals that round-trip, never a numpy repr)."""
    origin, scale = _frame(window)
    x0, y0, x1, y1 = window
    probe_x, probe_y = np.meshgrid(np.linspace(x0, x1, 129), np.linspace(y0, y1, 129), indexing='ij')
    smallest = float(size_field(np.column_stack([probe_x.ravel(), probe_y.ravel()])).min())
    spacing = max(smallest, scale / _VIEW_CELLS)
    n_x, n_y = (max(1, int(math.ceil((high - low) / spacing))) for low, high in ((x0, x1), (y0, y1)))
    grid_x, grid_y = np.meshgrid(np.linspace(x0, x1, n_x + 1), np.linspace(y0, y1, n_y + 1), indexing='ij')
    sizes = (size_field(np.column_stack([grid_x.ravel(), grid_y.ravel()])) / scale).reshape(grid_x.shape)
    unit_x, unit_y = (grid_x - origin[0]) / scale, (grid_y - origin[1]) / scale
    cell_i, cell_j = (index.ravel() for index in np.meshgrid(np.arange(n_x), np.arange(n_y), indexing='ij'))
    corners = ((cell_i, cell_j), (cell_i + 1, cell_j), (cell_i + 1, cell_j + 1), (cell_i, cell_j + 1))
    triangles = []   # per cell the triangles (0, 1, 2) then (0, 2, 3): columns x0 y0 x1 y1 x2 y2 v0 v1 v2
    for triple in ((0, 1, 2), (0, 2, 3)):
        picked = [corners[k] for k in triple]
        columns = [array[c] for c in picked for array in (unit_x, unit_y)] + [sizes[c] for c in picked]
        triangles.append(np.column_stack(columns))
    rows = np.stack(triangles, axis=1).reshape(-1, 9)
    with open(path, 'w', encoding='ascii') as handle:
        handle.write('View "size" {\n')
        np.savetxt(handle, rows, fmt='ST(%.17g, %.17g, 0, %.17g, %.17g, 0, %.17g, %.17g, 0){%.17g, %.17g, %.17g};')
        handle.write('};\n')


def write_geo(path, pslg, view_name):
    """Gmsh .geo of the PSLG: one Line per segment (tag = segment index + 1), the window loop as the surface, the
    other lines and lone points embedded, the size view as background mesh; window-side lines of pslg.periodic_axes
    get no interior nodes (Transfinite Curve = 2: the matched PSLG vertices are the side nodes)."""
    origin, scale = _frame(pslg.window)
    unit = (pslg.vertices - origin) / scale
    rows = ['// open_sol PSLG (fem/gmsh_io.py), coordinates normalised to the longest window side',
            'Mesh.MeshSizeExtendFromBoundary = 0;', 'Mesh.MeshSizeFromPoints = 0;', 'Mesh.MeshSizeFromCurvature = 0;',
            'Mesh.Algorithm = 6;', 'Mesh.ElementOrder = 1;', *PINNED_OUTPUT_OPTIONS]
    rows += [f'Point({index + 1}) = {{{geo_number(x)}, {geo_number(y)}, 0}};' for index, (x, y) in enumerate(unit)]
    rows += [f'Line({index + 1}) = {{{a + 1}, {b + 1}}};' for index, (a, b) in enumerate(pslg.segments)]
    on_window = [index for index, tags in enumerate(pslg.segment_tags) if _FACE_TAGS & set(tags)]
    rows.append(f"Curve Loop(1) = {{{', '.join(str(item) for item in _loop(pslg.segments, on_window))}}};")
    rows.append('Plane Surface(1) = {1};')
    periodic_faces = {f'{axis}_{side}' for axis in pslg.periodic_axes for side in ('lo', 'hi')}
    periodic_lines = [index + 1 for index, tags in enumerate(pslg.segment_tags) if periodic_faces & set(tags)]
    if periodic_lines:   # no Periodic Curve: its master / slave orientation rules differ between gmsh versions
        rows.append(f"Transfinite Curve{{{', '.join(str(tag) for tag in periodic_lines)}}} = 2;")
    embedded = sorted(set(range(len(pslg.segments))) - set(on_window))
    if embedded:
        rows.append(f"Curve{{{', '.join(str(index + 1) for index in embedded)}}} In Surface{{1}};")
    lone = sorted(set(range(len(pslg.vertices))) - set(pslg.segments.ravel().tolist()))
    if lone:
        rows.append(f"Point{{{', '.join(str(index + 1) for index in lone)}}} In Surface{{1}};")
    rows += [f'Merge "{view_name}";', 'Background Mesh View[0];']
    with open(path, 'w', encoding='ascii') as handle:
        handle.write('\n'.join(rows) + '\n')


def _loop(segments, members) -> list:
    """Signed line tags (+-(index + 1)) chaining the window segments into one closed loop."""
    at_vertex = {}
    for index in members:
        for vertex in segments[index]:
            at_vertex.setdefault(int(vertex), []).append(index)
    first = members[0]
    start, current = int(segments[first][0]), int(segments[first][1])
    loop, previous = [first + 1], first
    while current != start:
        following = next(index for index in at_vertex[current] if index != previous)
        forward = int(segments[following][0]) == current
        loop.append(following + 1 if forward else -(following + 1))
        current = int(segments[following][1] if forward else segments[following][0])
        previous = following
    return loop


def read_msh_blocks(path) -> tuple:
    """(node tags (n,), coordinates (n, 3), parametric, [(entity tag, element type, node tags (k, nodes))]) of a .msh
    4.1 ASCII file (2D and fem.gmsh3d): every block converted by one numpy parse, no Python loop per row; empty element
    blocks skipped. RuntimeError without $Nodes / $Elements."""
    with open(path, encoding='ascii', errors='replace') as handle:
        rows = [row.strip() for row in handle.read().split('\n')]
    if '$Nodes' not in rows or '$Elements' not in rows:
        raise RuntimeError('the .msh file has no $Nodes / $Elements section (no mesh was generated)')

    def block(start, count, dtype):
        return np.array(' '.join(rows[start:start + count]).split(), dtype=dtype).reshape(count, -1)

    position = rows.index('$Nodes') + 1
    n_blocks = int(rows[position].split()[0])
    position += 1
    tags, coordinates, parametric = [np.zeros(0, dtype=np.int64)], [np.zeros((0, 3))], False
    for _ in range(n_blocks):
        header = rows[position].split()
        count = int(header[3])
        parametric = parametric or bool(int(header[2]))
        position += 1
        if count:
            tags.append(block(position, count, np.int64).reshape(-1))
            coordinates.append(block(position + count, count, np.float64)[:, :3])
        position += 2 * count
    position = rows.index('$Elements') + 1
    n_blocks = int(rows[position].split()[0])
    position += 1
    blocks = []
    for _ in range(n_blocks):
        _, entity, element_type, count = (int(value) for value in rows[position].split())
        position += 1
        if count:
            blocks.append((entity, element_type, block(position, count, np.int64)[:, 1:]))
        position += count
    return np.concatenate(tags), np.concatenate(coordinates), parametric, blocks


def node_index(tags) -> np.ndarray:
    """Lookup node tag -> row of the coordinates (-1 for unused tags)."""
    index_of = np.full(int(tags.max()) + 1 if tags.size else 1, -1, dtype=np.int64)
    index_of[tags] = np.arange(len(tags))
    return index_of


def read_msh(path) -> tuple:
    """(nodes (n, 2), triangles (m, 3) node indices, [(entity tag, (k, 2) line node pairs)]) of a .msh 4.1 ASCII file;
    6-node triangles / 3-node lines give their corner nodes."""
    tags, coordinates, _, blocks = read_msh_blocks(path)
    index_of = node_index(tags)
    triangles = [nodes[:, :3] for _, kind, nodes in blocks if kind in _TRIANGLE_TYPES]
    line_blocks = [(entity, index_of[nodes[:, :2]]) for entity, kind, nodes in blocks if kind in _LINE_TYPES]
    triangles = np.concatenate(triangles) if triangles else np.zeros((0, 3), dtype=np.int64)
    return coordinates[:, :2], index_of[triangles], line_blocks


def _raw_mesh(pslg, unit_nodes, triangles, line_blocks) -> RawMesh:
    """RawMesh with the PSLG vertices first (exact coordinates), nodes on curved lines projected onto their curve."""
    from scipy.spatial import cKDTree

    origin, scale = _frame(pslg.window)
    nodes = unit_nodes * scale + origin
    distance, nearest = cKDTree(nodes).query(pslg.vertices)
    tolerance = 1e3 * MERGE_TOLERANCE * scale
    if not (distance <= tolerance).all():
        raise RuntimeError(f'mesher gmsh: {int((distance > tolerance).sum())} PSLG vertices are missing in the mesh')
    shared, counts = np.unique(nearest, return_counts=True)
    if (counts > 1).any():
        merged = [int(vertex) for vertex in np.flatnonzero(np.isin(nearest, shared[counts > 1]))[:6]]
        raise RuntimeError(f'mesher gmsh: gmsh merged PSLG vertices onto one node ({int((counts > 1).sum())} nodes, '
                           f'e.g. vertices {merged} at {pslg.vertices[merged[0]].tolist()}): vertices closer than '
                           "gmsh's geometry tolerance; refine the geometry or use the built-in mesher")
    order = np.concatenate([nearest, np.setdiff1d(np.arange(len(nodes)), nearest)])
    new_index = np.empty(len(nodes), dtype=np.int64)
    new_index[order] = np.arange(len(nodes))
    points = nodes[order]
    points[:len(pslg.vertices)] = pslg.vertices
    edges, segment_of = [], []
    for entity, pairs in line_blocks:
        if 1 <= entity <= len(pslg.segments):
            edges.append(new_index[pairs])
            segment_of.append(np.full(len(pairs), entity - 1, dtype=np.int64))
    edges = np.concatenate(edges) if edges else np.zeros((0, 2), dtype=np.int64)
    segment_of = np.concatenate(segment_of) if segment_of else np.zeros(0, dtype=np.int64)
    curve_of = pslg.segment_curve[segment_of]
    for curve_index in np.unique(curve_of[curve_of >= 0]):
        movable = np.setdiff1d(np.unique(edges[curve_of == curve_index]), np.arange(len(pslg.vertices)))
        points[movable] = pslg.curves[curve_index].project(points[movable])
    x0, y0, x1, y1 = pslg.window
    snap = MERGE_TOLERANCE * scale
    for axis, (low, high) in enumerate(((x0, x1), (y0, y1))):
        points[np.abs(points[:, axis] - low) <= snap, axis] = low
        points[np.abs(points[:, axis] - high) <= snap, axis] = high
    return RawMesh(points=points, triangles=new_index[triangles].astype(np.int32),
                   constraint_edges=edges.astype(np.int32), constraint_segment=segment_of.astype(np.int32), rounds=0)
