"""Optional gmsh tetrahedral mesher (igor3d_plan section 7, T3): primitive solids (box, sphere, z cylinder)
fragmented with the meshed box (window + PML shell) so interfaces conform, only the fragments inside the box kept,
written as an OpenCASCADE .geo in normalised coordinates, `gmsh -3` run as an external process (never imported,
linked or vendored, as fem.gmsh_io), the .msh 4.1 ASCII tetrahedra read back into a TetrahedronMesh. Host numpy."""
# Clipping: every volume is fragmented at once, gmsh meshes and saves them all (Mesh.SaveAll), and open_sol keeps the
# volume entities whose tetrahedron centroids lie mostly inside the box (_inside_fragments); the kept volume must equal
# the box volume. Never select inside gmsh: a per-solid BooleanIntersection with a box copy and a Volume In
# BoundingBox physical group both left the outside parts of the Mie eighth model's sphere in the mesh (nodes below
# x = 0). gmsh can report a .geo error and still exit 0, with an empty .msh or a mesh of the rows before the error (a
# MathEval string over 1024 characters: 'String too long', no size fields): any 'Error' line fails the run, quoted
# (gmsh_io.gmsh_errors), and MathEval texts stay below GEO_STRING_MAX; a failed selection also prints what gmsh wrote
# (_layout_text). Every .geo pins the output options a user gmsh option file could change
# (gmsh_io.PINNED_OUTPUT_OPTIONS).
# Sizes: Mesh.MeshSizeMax = h_max, Ball / Box / Cylinder background fields give each sized solid its size inside
# (linear transition outside: over (h_max - size) / (growth - 1), else `transition`), one MathEval field per box edge /
# cylinder rim of a solid with an edge_size (edge_size + (growth - 1) x distance, clipped by MeshSizeMax; analytic
# distances, so no Distance-field sampling option that changed name between gmsh versions), Mesh.MeshSizeFromCurvature
# resolves curved surfaces (2 pi / curvature_factor elements per 2 pi, default CURVATURE_ELEMENTS), Netgen optimisation
# removes slivers. Regions: centroid tags (centroids clamped into the window) of `region_of` (default
# fem.mesh3d.region_tags of the solids); a gmsh volume entity whose tags agree on at least mesh3d.ENTITY_MAJORITY of
# its elements takes that tag throughout (no misread faceting slivers at curved interfaces), others keep the
# per-element tags (objects the caller tags but did not pass as solids); the mesh keeps the entities. The window box is its own fragment when a PML shell
# exists, so the window faces are mesh faces. Periodic axes (Bloch cells): after the fragmenting every surface on the hi
# side of the box is made the translated image of the lo-side surface with the same bounding box (`Periodic Surface ...
# Translate`, gmsh tutorial t18), so gmsh copies the lo-side surface mesh and fem.mesh3d.periodic_node_partners pairs
# every node; a lo surface without a hi image (geometry not periodic) leaves the sides unmatched and the pairing raises.
# Composite sizes (D1): a SizedRegion is a field tree over its solids' fields (union Min, intersection Max, difference
# Max(base, inverted subtracted)); an inverted solid is size outside it, size + slope x depth inside (Min of one
# MathEval per face), and faces of a subtracted solid at / beyond the base's bounding box are left out of the depth (a
# hole through a film is no hole surface at the film's top and bottom). SizedRegion.sizes is the numpy twin.
# Surface sizes (D3b): a sized surface is Max(region tree with exact outside ramps (MathEval, no Box / Ball / Cylinder
# field), inverted tree) = size + slope x |distance to the region's surface| (CSG min / max of the primitives' analytic
# distances: exact for one primitive, a bound near inner faces of a composite); SizedRegion.surface_sizes is the twin.
# gmsh's Cylinder field is VIn where |(X - centre) . axis| < |axis|^2: its axis vector is the half length (_size_field).
# Disk cache (D4): with OPEN_SOL_MESH_CACHE (a folder) the raw `read_msh_tetrahedra` arrays (normalised coordinates,
# before clipping / snapping) are kept as .npz per sha256 of (format, gmsh path / size / mtime, .geo text).
# The model geometry export is model.tetrahedral_meshing.
import hashlib
import os
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from functools import reduce

import numpy as np

from .gmsh_io import (GMSH_ENVIRONMENT, GMSH_TIMEOUT, PINNED_OUTPUT_OPTIONS, find_gmsh, geo_number, gmsh_errors,
                      node_index, read_msh_blocks)
from .mesh3d import (OUTER_FACES_3D, PLANE_TOLERANCE, build_tetrahedron_mesh, entity_majority, region_tags,
                     span_adjugate)

SOLID_KINDS = ('box', 'sphere', 'cylinder')
CURVATURE_ELEMENTS = 16   # elements per 2 pi of surface curvature
GMSH_CACHE_SIZE = 4       # gmsh results kept per (executable, .geo text): a rebuild of the same mesh skips gmsh
_TETRAHEDRON_TYPE = 4     # msh element type of the 4-node tetrahedron
_FIRST_SOLID_TAG = 10
_EXTENT_SLACK = 1e-6      # x the longest box side: centroid inside test, nodes beyond the box by more are an error
_VOLUME_RTOL = 1e-6       # kept tetrahedra volume vs the box volume (planar box faces: exact up to rounding)
_BOX_EDGE_AXES = ((0, 1, 2), (1, 0, 2), (2, 0, 1))   # (axis along the edge, the two axes fixing it)
GEO_STRING_MAX = 1000     # characters per quoted .geo string (gmsh's lexer buffer is 1024: 'String too long')
REGION_KINDS = ('solid', 'union', 'intersection', 'difference')
_CLIP_RTOL = 1e-9         # x the base's longest side: a subtracted face this close to the base's box side is no hole face
MESH_CACHE_ENVIRONMENT = 'OPEN_SOL_MESH_CACHE'   # folder of the on-disk gmsh cache; unset or empty = off
MESH_CACHE_MAX_BYTES = 2 * 1024 ** 3              # oldest cache files (by mtime) removed above this total
_MESH_CACHE_FORMAT = 1    # bump when the cached arrays change meaning
_MESH_CACHE_PREFIX = 'gmsh3d_'


@dataclass(frozen=True)
class Solid:
    """Primitive of a gmsh tetrahedral mesh: 'box' values (x0, y0, z0, x1, y1, z1), 'sphere' (cx, cy, cz, r),
    'cylinder' (cx, cy, z0, r, height) along +z; size = element size inside, edge_size = along its edges (box: 12
    edges, cylinder: 2 rims, sphere: none) [m] (0: the mesh h_max)."""
    name: str
    kind: str
    values: tuple
    size: float = 0.0
    edge_size: float = 0.0

    def __post_init__(self):
        expected = {'box': 6, 'sphere': 4, 'cylinder': 5}.get(self.kind)
        if expected is None:
            raise ValueError(f"Solid kind must be one of {', '.join(SOLID_KINDS)}, got {self.kind!r}")
        if len(self.values) != expected:
            raise ValueError(f'Solid {self.name!r} ({self.kind}) needs {expected} values, got {len(self.values)}')

    def contains(self, points) -> np.ndarray:
        """Bool per point (n, 3) [m] inside the solid (closed)."""
        points = np.asarray(points, dtype=np.float64)
        if self.kind == 'box':
            low, high = np.array(self.values[:3]), np.array(self.values[3:])
            return ((points >= low) & (points <= high)).all(axis=-1)
        if self.kind == 'sphere':
            return np.linalg.norm(points - np.array(self.values[:3]), axis=-1) <= self.values[3]
        cx, cy, z0, radius, height = self.values
        radial = np.hypot(points[..., 0] - cx, points[..., 1] - cy)
        return (radial <= radius) & (points[..., 2] >= z0) & (points[..., 2] <= z0 + height)

    def edge_distance(self, points) -> np.ndarray:
        """Distance [m] of points (..., 3) to the nearest edge of the solid (inf for a sphere: no edges)."""
        points = np.asarray(points, dtype=np.float64)
        if self.kind == 'sphere':
            return np.full(points.shape[:-1], np.inf)
        if self.kind == 'cylinder':
            cx, cy, z0, radius, height = self.values
            radial = np.hypot(points[..., 0] - cx, points[..., 1] - cy) - radius
            return np.minimum(*(np.hypot(radial, points[..., 2] - rim) for rim in (z0, z0 + height)))
        low, high = np.array(self.values[:3]), np.array(self.values[3:])
        distance = np.full(points.shape[:-1], np.inf)
        for along, first, second in _BOX_EDGE_AXES:
            outside = np.maximum(np.maximum(low[along] - points[..., along], points[..., along] - high[along]), 0.0)
            for first_value in (low[first], high[first]):
                for second_value in (low[second], high[second]):
                    np.minimum(distance, np.sqrt(outside ** 2 + (points[..., first] - first_value) ** 2
                                                 + (points[..., second] - second_value) ** 2), out=distance)
        return distance

    def bounds(self) -> tuple:
        """(low (3,), high (3,)) [m] bounding box of the solid."""
        if self.kind == 'box':
            return np.array(self.values[:3], dtype=np.float64), np.array(self.values[3:], dtype=np.float64)
        if self.kind == 'sphere':
            centre, radius = np.array(self.values[:3], dtype=np.float64), self.values[3]
            return centre - radius, centre + radius
        cx, cy, z0, radius, height = self.values
        return np.array([cx - radius, cy - radius, z0]), np.array([cx + radius, cy + radius, z0 + height])

    def faces(self) -> list:
        """(axis, side 0 low / 1 high, coordinate [m]) per depth term, None for the curved side (box 6 planes, cylinder
        side + 2 ends, sphere its surface); the order of _depth_texts."""
        if self.kind == 'box':
            return [(axis, side, self.values[3 * side + axis]) for axis in range(3) for side in (0, 1)]
        if self.kind == 'sphere':
            return [None]
        z0, height = self.values[2], self.values[4]
        return [None, (2, 0, z0), (2, 1, z0 + height)]

    def depth(self, points, clip=None) -> np.ndarray:
        """Depth [m] of points (..., 3) below the nearest face (negative outside); clip (low, high) leaves out the faces
        at / beyond that box side (inf when none is left)."""
        points = np.asarray(points, dtype=np.float64)
        depth = np.full(points.shape[:-1], np.inf)
        for face, kept in zip(self.faces(), _kept_faces(self, clip)):
            if not kept:
                continue
            if face is not None:
                axis, side, coordinate = face
                term = points[..., axis] - coordinate if side == 0 else coordinate - points[..., axis]
            elif self.kind == 'sphere':
                term = self.values[3] - np.linalg.norm(points - np.array(self.values[:3]), axis=-1)
            else:
                term = self.values[3] - np.hypot(points[..., 0] - self.values[0], points[..., 1] - self.values[1])
            np.minimum(depth, term, out=depth)
        return depth


def _kept_faces(solid, clip) -> list:
    """Bool per Solid.faces entry: False for a plane at / beyond the clip box side it faces (no surface of the region
    the solid is subtracted from), True for the curved side and without a clip."""
    faces = solid.faces()
    if clip is None:
        return [True] * len(faces)
    low, high = (np.asarray(corner, dtype=np.float64) for corner in clip)
    slack = _CLIP_RTOL * float(np.max(high - low))
    return [face is None or (face[2] > low[face[0]] + slack if face[1] == 0 else face[2] < high[face[0]] - slack)
            for face in faces]


def _outside_distance(solid, points) -> np.ndarray:
    """Distance [m] of points (..., 3) outside a solid to it (0 inside): the Box / Ball field ramp, the cylinder's
    exact distance (_outside_distance_text)."""
    if solid.kind == 'sphere':
        return np.maximum(np.linalg.norm(points - np.array(solid.values[:3]), axis=-1) - solid.values[3], 0.0)
    if solid.kind == 'cylinder':
        cx, cy, z0, radius, height = solid.values
        radial = np.maximum(np.hypot(points[..., 0] - cx, points[..., 1] - cy) - radius, 0.0)
        return np.hypot(radial, np.maximum(np.maximum(z0 - points[..., 2], points[..., 2] - (z0 + height)), 0.0))
    low, high = solid.bounds()
    return np.linalg.norm(np.maximum(np.maximum(low - points, points - high), 0.0), axis=-1)


@dataclass(frozen=True)
class SizedRegion:
    """Element size `size` [m] inside a composite of gmsh3d Solids (plan D1): kind 'solid' (solid) or 'union' /
    'intersection' / 'difference' (children; difference: the base first), ramping outside like a sized solid."""
    kind: str
    size: float
    solid: Solid | None = None
    children: tuple = ()

    def __post_init__(self):
        if self.kind not in REGION_KINDS:
            raise ValueError(f"SizedRegion kind must be one of {', '.join(REGION_KINDS)}, got {self.kind!r}")
        if (self.kind == 'solid') != (self.solid is not None) or (self.kind != 'solid') != bool(self.children):
            raise ValueError("SizedRegion: kind 'solid' takes a solid and no children, a composite children only")

    def bounds(self) -> tuple:
        """(low (3,), high (3,)) [m] box holding the region (union: the joined boxes, intersection: their overlap,
        difference: the base's)."""
        if self.kind == 'solid':
            return self.solid.bounds()
        boxes = [child.bounds() for child in self.children]
        if self.kind == 'difference':
            return boxes[0]
        lows, highs = np.array([box[0] for box in boxes]), np.array([box[1] for box in boxes])
        if self.kind == 'union':
            return lows.min(axis=0), highs.max(axis=0)
        return lows.max(axis=0), highs.min(axis=0)

    def sizes(self, points, growth, h_max, transition=None) -> np.ndarray:
        """Element size [m] at points (..., 3): the numpy twin of the gmsh field tree (geo_text), capped at h_max;
        transition (default 2 h_max) sets the ramp when growth is None."""
        transition = 2.0 * float(h_max) if transition is None else float(transition)
        slope = (h_max - self.size) / transition if growth is None else float(growth) - 1.0
        values = _region_values(self, np.asarray(points, dtype=np.float64), False, None, slope, float(h_max))
        return np.minimum(values, h_max)

    def surface_distance(self, points) -> np.ndarray:
        """Distance [m] of points (..., 3) to the region's surface (plan D3b): the max of the outside distance and
        the depth trees (children's sizes unused; inf deep in a part whose every face is clipped)."""
        points = np.asarray(points, dtype=np.float64)
        return np.maximum(_region_distance(self, points, False, None), _region_distance(self, points, True, None))

    def surface_sizes(self, points, growth, h_max, transition=None) -> np.ndarray:
        """Element size [m] at points (..., 3) of this region as a sized surface: size + slope x surface_distance,
        capped at h_max (the numpy twin of geo_text's sized_surfaces); transition as in sizes."""
        transition = 2.0 * float(h_max) if transition is None else float(transition)
        slope = (h_max - self.size) / transition if growth is None else float(growth) - 1.0
        return np.minimum(self.size + slope * self.surface_distance(points), h_max)


def _region_distance(region, points, inverted, clip) -> np.ndarray:
    """Distance [m] of points (..., 3) from where a region's size holds: outside it (0 inside), inverted the depth
    inside it (0 outside); composites as _region_values (min / max by De Morgan, clipped subtracted faces)."""
    if region.kind == 'solid':
        if inverted:
            return np.maximum(region.solid.depth(points, clip), 0.0)
        return _outside_distance(region.solid, points)
    values = [_region_distance(child, points, inverted != (region.kind == 'difference' and index > 0), child_clip)
              for index, child, child_clip in _child_clips(region, clip)]
    use_min = (region.kind == 'union') != inverted
    return reduce(np.minimum if use_min else np.maximum, values)


def _region_values(region, points, inverted, clip, slope, h_max) -> np.ndarray:
    """Field values of a region (inverted: size outside it, ramping up inside; the De Morgan twin of
    _region_fields); a difference hands its subtracted children the base's box as clip."""
    if region.kind == 'solid':
        solid = region.solid
        if inverted:
            return region.size + slope * np.maximum(solid.depth(points, clip), 0.0)
        return region.size + slope * _outside_distance(solid, points)
    values = []
    for index, child, child_clip in _child_clips(region, clip):
        values.append(_region_values(child, points, inverted != (region.kind == 'difference' and index > 0),
                                     child_clip, slope, h_max))
    use_min = (region.kind == 'union') != inverted
    return reduce(np.minimum if use_min else np.maximum, values)


def _child_clips(region, clip) -> list:
    """(index, child, clip) of a composite region: a difference's subtracted children get the base's bounds, other
    children keep the inherited clip."""
    if region.kind != 'difference':
        return [(index, child, clip) for index, child in enumerate(region.children)]
    base_box = region.children[0].bounds()
    return [(index, child, clip if index == 0 else base_box) for index, child in enumerate(region.children)]


@dataclass(frozen=True)
class SizedFace:
    """Axis-aligned rectangle of a tetrahedron size field (new_modules_plan G1): the plane normal to axis 0 / 1 / 2 at
    coordinate, extents low / high [m] along the two other axes (ascending axis order); element size `size` within
    `distance` of the rectangle, size + (growth - 1) x the distance beyond the band (gmsh: Max of two MathEvals)."""
    axis: int
    coordinate: float
    low: tuple
    high: tuple
    size: float
    distance: float = 0.0

    def face_distance(self, points) -> np.ndarray:
        """Distance [m] of points (..., 3) to the rectangle."""
        points = np.asarray(points, dtype=np.float64)
        squared = (points[..., self.axis] - self.coordinate) ** 2
        for axis, low, high in zip((k for k in range(3) if k != self.axis), self.low, self.high):
            outside = np.maximum(np.maximum(low - points[..., axis], points[..., axis] - high), 0.0)
            squared = squared + outside ** 2
        return np.sqrt(squared)

    def sizes(self, points, growth, h_max=np.inf) -> np.ndarray:
        """Element size [m] of this source at points (..., 3): the numpy twin of its MathEval field, capped at h_max."""
        beyond = np.maximum(self.face_distance(points) - self.distance, 0.0)
        return np.minimum(self.size + (float(growth) - 1.0) * beyond, h_max)


def mesh_solids_with_gmsh(window, solids, h_max, pml_widths=None, h_min=None, transition=None, executable=None,
                          timeout=GMSH_TIMEOUT, growth=None, curvature_factor=None, region_of=None, region_names=None,
                          periodic_axes=(), sized_faces=(), size_points=None, sized_regions=(), disk_cache=None,
                          timings=None, sized_surfaces=()):
    """TetrahedronMesh of the window (x0, y0, z0, x1, y1, z1) [m] plus a PML shell of pml_widths {face: width [m]}
    with the solids conforming inside; h_max / h_min [m] (h_min default h_max / 100); size ramp outside a sized solid
    over (h_max - size) / (growth - 1) with a growth, else transition (default 2 h_max); region_of(points (m, 3)) ->
    int32 tags into region_names (default: the solids); periodic_axes ('x' / 'y' / 'z', no PML shell on them): hi-side
    surfaces meshed as images of the lo side; sized_faces (SizedFace), size_points ((n, 4) [x, y, z, h] [m], each
    growing like a sized solid) and sized_regions (SizedRegion: composites) refine near planes / points / inside
    composites, sized_surfaces (SizedRegion) near a region's surface only; disk_cache = cache folder ('' off, None:
    OPEN_SOL_MESH_CACHE); timings (a dict) gets the seconds of 'gmsh' ('source' gmsh / disk / memory), 'region_tags',
    'mesh_tables'. RuntimeError when gmsh is missing or fails, ValueError for a shell on a periodic axis."""
    executable = executable or find_gmsh()
    if executable is None:
        raise RuntimeError(f'mesher gmsh: executable not found (set {GMSH_ENVIRONMENT} or put gmsh on PATH)')
    window = tuple(float(value) for value in window)
    widths = {face: float((pml_widths or {}).get(face, 0.0)) for face in OUTER_FACES_3D}
    shelled = [axis for axis in periodic_axes if widths[f'{axis}_lo'] or widths[f'{axis}_hi']]
    if shelled:
        raise ValueError(f"mesher gmsh: periodic axes {', '.join(shelled)} carry a PML shell; a periodic axis has none")
    bounds = (window[0] - widths['x_lo'], window[1] - widths['y_lo'], window[2] - widths['z_lo'],
              window[3] + widths['x_hi'], window[4] + widths['y_hi'], window[5] + widths['z_hi'])
    h_min = float(h_max) / 100.0 if h_min is None else float(h_min)
    transition = 2.0 * float(h_max) if transition is None else float(transition)
    curvature_elements = (CURVATURE_ELEMENTS if curvature_factor is None
                          else max(int(np.ceil(2.0 * np.pi / float(curvature_factor))), 1))
    geo = geo_text(bounds, window, solids, float(h_max), h_min, transition, growth, curvature_elements, periodic_axes,
                   sized_faces, size_points, sized_regions, sized_surfaces)
    started = time.perf_counter()
    points, tets, entities = _cached_gmsh_mesh(executable, geo, bounds, window, timeout, disk_cache, timings)
    meshed = time.perf_counter()
    if region_of is None:
        region_names = tuple(solid.name for solid in solids)

        def region_of(centroids):
            return region_tags(centroids, solids)
    region = None
    if region_names:
        low, high = np.array(window[:3], dtype=np.float64), np.array(window[3:], dtype=np.float64)
        inset = PLANE_TOLERANCE * float((high - low).max())   # TetrahedronMesh.inset_centroids rule (builder masks)
        region = entity_majority(region_of(np.clip(points[tets].mean(axis=1), low + inset, high - inset)), entities)
    tagged = time.perf_counter()
    built = build_tetrahedron_mesh(points, tets, window=window, region=region, region_names=tuple(region_names or ()),
                                   entities=entities)
    if timings is not None:
        timings.update(gmsh=meshed - started, region_tags=tagged - meshed, mesh_tables=time.perf_counter() - tagged)
    return built


def mesh_cache_folder(disk_cache=None):
    """Absolute folder of the on-disk gmsh cache: disk_cache when given ('' = off), else the environment variable
    OPEN_SOL_MESH_CACHE (read per call; unset or empty = off); None when off."""
    folder = os.environ.get(MESH_CACHE_ENVIRONMENT, '') if disk_cache is None else os.fspath(disk_cache)
    return os.path.abspath(folder) if folder else None


def gmsh_run_count() -> int:
    """Number of `gmsh -3` runs in this process (self-tests: a mesh read from a cache runs none)."""
    return _GMSH_RUNS[0]


def _cached_gmsh_mesh(executable, geo, bounds, window, timeout, disk_cache=None, timings=None) -> tuple:
    """Copies of (points [m], tets, entities) of `gmsh -3` on the .geo text, clipped to the box and snapped to its
    planes; the last GMSH_CACHE_SIZE results kept per (executable, .geo text, box) in memory, the raw gmsh arrays on
    disk when a cache folder is set (mesh_cache_folder); timings['source'] says which."""
    # the .geo is normalised: a uniformly scaled box gives the same text, so the clipped metres also key on the box
    key = (os.path.abspath(executable), geo, tuple(bounds), tuple(window))
    with _GMSH_CACHE_LOCK:
        cached = _GMSH_CACHE.get(key)
        if cached is not None:
            _GMSH_CACHE.move_to_end(key)
    source = 'memory'
    if cached is None:
        folder = mesh_cache_folder(disk_cache)
        path = None if folder is None else _disk_cache_path(folder, executable, geo)
        raw = None if path is None else _read_disk_mesh(path)
        completed, source = None, 'disk'
        if raw is None:
            *raw, completed = _run_gmsh(executable, geo, timeout)
            source = 'gmsh'
            if path is not None:
                _write_disk_mesh(path, *raw)
        cached = _clipped_mesh(*raw, bounds, window, completed)
        with _GMSH_CACHE_LOCK:
            _GMSH_CACHE[key] = cached
            while len(_GMSH_CACHE) > GMSH_CACHE_SIZE:
                _GMSH_CACHE.popitem(last=False)
    if timings is not None:
        timings['source'] = source
    return tuple(array.copy() for array in cached)


def _disk_cache_path(folder, executable, geo):
    """Cache file of a (gmsh binary, .geo text) pair: sha256 of the format version, the binary's path, size and mtime
    and the text; None when the binary cannot be stat'ed."""
    try:
        status = os.stat(executable)
    except OSError:
        return None
    head = f'{_MESH_CACHE_FORMAT}\n{os.path.abspath(executable)}\n{status.st_size}\n{status.st_mtime_ns}\n'
    digest = hashlib.sha256((head + geo).encode('utf-8')).hexdigest()
    return os.path.join(folder, f'{_MESH_CACHE_PREFIX}{digest}.npz')


def _read_disk_mesh(path):
    """(unit nodes (n, 3), tets (m, 4), entities (m,)) of a cache file, None when missing, unreadable or inconsistent
    (the caller re-meshes and rewrites it); a hit refreshes the file's mtime (least recently used pruning)."""
    if not os.path.isfile(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            nodes, tets, entities = (np.array(data[name]) for name in ('nodes', 'tets', 'entities'))
    except Exception:   # noqa: BLE001 - truncated / corrupt zip, missing keys, permissions: re-mesh
        return None
    if not (nodes.ndim == 2 and nodes.shape[1] == 3 and tets.ndim == 2 and tets.shape[1] == 4 and len(tets)
            and entities.shape == (len(tets),) and np.issubdtype(tets.dtype, np.integer)
            and tets.min() >= 0 and tets.max() < len(nodes) and np.isfinite(nodes).all()):
        return None
    try:
        os.utime(path)
    except OSError:
        pass
    return nodes.astype(np.float64), tets.astype(np.int64), entities.astype(np.int64)


def _write_disk_mesh(path, nodes, tets, entities):
    """Save the raw gmsh arrays (a temporary file, then os.replace: readers never see a partial file) and prune the
    folder by mtime to MESH_CACHE_MAX_BYTES; failures are ignored (the cache is optional)."""
    folder = os.path.dirname(path)
    temporary = None
    try:
        os.makedirs(folder, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=folder, prefix=f'.{_MESH_CACHE_PREFIX}', suffix='.tmp')
        with os.fdopen(handle, 'wb') as stream:
            np.savez(stream, nodes=nodes, tets=tets, entities=entities)
        os.replace(temporary, path)
        temporary = None
    except (OSError, ValueError):
        return
    finally:
        if temporary is not None:
            try:
                os.remove(temporary)
            except OSError:
                pass
    _prune_disk_cache(folder, path)


def _prune_disk_cache(folder, kept_path):
    """Remove the least recently used cache files (oldest mtime first, never kept_path) while the folder's cache files
    exceed MESH_CACHE_MAX_BYTES."""
    try:
        files = [(entry.stat().st_mtime, entry.stat().st_size, entry.path) for entry in os.scandir(folder)
                 if entry.is_file() and entry.name.startswith(_MESH_CACHE_PREFIX) and entry.name.endswith('.npz')]
    except OSError:
        return
    total = sum(size for _, size, _ in files)
    for _, size, file_path in sorted(files):
        if total <= MESH_CACHE_MAX_BYTES:
            break
        if os.path.normcase(os.path.abspath(file_path)) == os.path.normcase(os.path.abspath(kept_path)):
            continue
        try:
            os.remove(file_path)
            total -= size
        except OSError:
            pass


def _run_gmsh(executable, geo, timeout) -> tuple:
    """(unit nodes, tets, entities, completed process) of `gmsh -3` on the .geo text as read from the .msh (normalised
    coordinates, every fragment); RuntimeError when gmsh fails."""
    _GMSH_RUNS[0] += 1
    with tempfile.TemporaryDirectory(prefix='open_sol_gmsh3d_') as folder:
        with open(os.path.join(folder, 'model.geo'), 'w', encoding='ascii') as handle:
            handle.write(geo)
        completed = subprocess.run([executable, 'model.geo', '-3', '-format', 'msh41', '-o', 'model.msh', '-v', '2'],
                                   cwd=folder, capture_output=True, text=True, timeout=timeout)
        msh_path = os.path.join(folder, 'model.msh')
        problem = None
        if completed.returncode != 0 or not os.path.isfile(msh_path):
            problem = f'exit code {completed.returncode}'
        elif any(line.lstrip().startswith('Error') for line in f'{completed.stdout}\n{completed.stderr}'.splitlines()):
            problem = 'errors (a .geo row gmsh could not read drops every later row, size fields included)'
        else:
            try:
                unit_nodes, tets, entities = read_msh_tetrahedra(msh_path)
            except (ValueError, RuntimeError) as exc:   # gmsh may report a .geo error and still exit 0
                problem = f'unreadable mesh ({exc})'
        if problem is not None:
            raise RuntimeError(f'mesher gmsh -3: {problem}; gmsh says: {gmsh_errors(completed)}')
    return unit_nodes, tets, entities, completed


def _clipped_mesh(unit_nodes, tets, entities, bounds, window, completed=None) -> tuple:
    """(points [m], tets, entities) of raw gmsh arrays: the fragments inside the box kept, nodes within the snap slack
    of a box or window plane moved onto it; completed = the gmsh process (None: read from the disk cache)."""
    origin, scale = _frame(bounds)
    try:
        points, tets, entities = _kept_mesh(unit_nodes, tets, entities, bounds, origin, scale)
    except RuntimeError as exc:   # say what gmsh wrote: its coordinates, volumes and messages
        said = 'nothing (read from the disk cache)' if completed is None else gmsh_errors(completed)
        raise RuntimeError(f'{exc}; {_layout_text(unit_nodes, tets, entities)}; gmsh says: {said}') from None
    slack = PLANE_TOLERANCE * scale * 1e3
    for axis in range(3):
        for value in (bounds[axis], bounds[3 + axis], window[axis], window[3 + axis]):
            points[np.abs(points[:, axis] - value) <= slack, axis] = value
    return points, tets, entities


_GMSH_CACHE = OrderedDict()
_GMSH_CACHE_LOCK = threading.Lock()   # the build worker and the GUI mesh preview may mesh at the same time
_GMSH_RUNS = [0]                      # gmsh_run_count


def _kept_mesh(unit_nodes, tets, entities, bounds, origin, scale) -> tuple:
    """(points [m], tets, entities) of the fragments inside the box, unused nodes dropped; RuntimeError when the kept
    mesh reaches beyond the box or does not fill its volume."""
    tets, entities = _inside_fragments(unit_nodes * scale + origin, tets, entities, bounds, scale)
    used, tets = np.unique(tets, return_inverse=True)
    tets = tets.reshape(-1, 4)
    points = unit_nodes[used] * scale + origin
    beyond = np.maximum(np.array(bounds[:3]) - points.min(axis=0), points.max(axis=0) - np.array(bounds[3:]))
    if (beyond > _EXTENT_SLACK * scale).any():
        raise RuntimeError(f'mesher gmsh -3: the mesh reaches {beyond.max():.3g} m beyond the box {bounds} '
                           '(a kept volume crosses the box faces: the solids were not fragmented with it)')
    volume = np.abs(span_adjugate(points[tets[:, 1:]] - points[tets[:, :1]])[0]).sum() / 6.0
    box_volume = float(np.prod(np.array(bounds[3:]) - np.array(bounds[:3])))
    if abs(volume - box_volume) > _VOLUME_RTOL * box_volume:
        raise RuntimeError(f'mesher gmsh -3: the kept tetrahedra fill {volume / box_volume:.8g} of the box volume '
                           '(fragments overlap or are missing)')
    return points, tets, entities


def _layout_text(unit_nodes, tets, entities) -> str:
    """What gmsh wrote, for failure messages: node extent in normalised units (the box is [0, 1] along its longest
    side) and per volume entity its tetrahedra and centroid extent."""
    def extent(points):
        low, high = points.min(axis=0), points.max(axis=0)
        return ' x '.join(f'[{a:.4g}, {b:.4g}]' for a, b in zip(low, high))

    centroids = unit_nodes[tets].mean(axis=1)
    parts = []
    for entity in np.unique(entities)[:12]:
        members = entities == entity
        parts.append(f'volume {int(entity)}: {int(members.sum())} tets, centroids {extent(centroids[members])}')
    return f'gmsh nodes (normalised) {extent(unit_nodes)}; ' + '; '.join(parts)


def _inside_fragments(points, tets, entities, bounds, scale) -> tuple:
    """(tets, entities) of the volume entities whose tetrahedron centroids lie mostly inside bounds; the solids'
    fragments outside the box are dropped."""
    slack = _EXTENT_SLACK * scale
    centroids = points[tets].mean(axis=1)
    inside = ((centroids >= np.array(bounds[:3]) - slack) & (centroids <= np.array(bounds[3:]) + slack)).all(axis=1)
    kept = [entity for entity in np.unique(entities) if inside[entities == entity].mean() > 0.5]
    if not kept:
        raise RuntimeError(f'mesher gmsh -3: no meshed volume lies inside the box {bounds}')
    keep = np.isin(entities, kept)
    return tets[keep], entities[keep]


def _frame(bounds) -> tuple:
    """(origin (3,), scale) of the normalised coordinates (box corner 0, longest side 1)."""
    origin = np.array(bounds[:3], dtype=np.float64)
    return origin, float(max(bounds[3] - bounds[0], bounds[4] - bounds[1], bounds[5] - bounds[2]))


def _numbers(*values) -> str:
    """', '-joined plain decimals (gmsh_io.geo_number: never a numpy repr)."""
    return ', '.join(geo_number(value) for value in values)


def _box_command(tag, low, high):
    return f'Box({tag}) = {{{_numbers(*low, *(high[k] - low[k] for k in range(3)))}}};'


def geo_text(bounds, window, solids, h_max, h_min, transition, growth=None,
             curvature_elements=CURVATURE_ELEMENTS, periodic_axes=(), sized_faces=(), size_points=None,
             sized_regions=(), sized_surfaces=()) -> str:
    """OpenCASCADE .geo (module header) in coordinates normalised to the longest side of bounds; sized faces as a Max
    of two MathEval fields, size points ((n, 4) [x, y, z, h] [m]) as Ball fields of radius 0, sized regions as Min /
    Max trees (_region_fields), sized surfaces as Max(ramped tree, inverted tree)."""
    origin, scale = _frame(bounds)

    def unit(point):
        return (np.asarray(point, dtype=np.float64) - origin) / scale

    outer_low, outer_high = unit(bounds[:3]), unit(bounds[3:])
    rows = ['// open_sol tetrahedral mesh (fem/gmsh3d.py), coordinates normalised to the longest box side',
            'SetFactory("OpenCASCADE");', 'Mesh.MeshSizeExtendFromBoundary = 0;', 'Mesh.MeshSizeFromPoints = 0;',
            f'Mesh.MeshSizeFromCurvature = {int(curvature_elements)};', f'Mesh.MeshSizeMax = {_numbers(h_max / scale)};',
            f'Mesh.MeshSizeMin = {_numbers(h_min / scale)};', 'Mesh.Algorithm = 6;', 'Mesh.Algorithm3D = 1;',
            'Mesh.Optimize = 1;', 'Mesh.OptimizeNetgen = 1;', 'Mesh.ElementOrder = 1;', 'Mesh.SaveAll = 1;',
            *PINNED_OUTPUT_OPTIONS,
            _box_command(1, outer_low, outer_high)]
    has_shell = tuple(window) != tuple(bounds)
    rows.append('tools[] = {};')
    if has_shell:
        rows += [_box_command(2, unit(window[:3]), unit(window[3:])), 'tools[] += {2};']
    fields = []
    for index, solid in enumerate(solids):
        tag = _FIRST_SOLID_TAG + index
        if solid.kind == 'box':
            rows.append(_box_command(tag, unit(solid.values[:3]), unit(solid.values[3:])))
        elif solid.kind == 'sphere':
            centre = unit(solid.values[:3])
            rows.append(f'Sphere({tag}) = {{{_numbers(*centre, solid.values[3] / scale)}}};')
        else:
            cx, cy, z0, radius, height = solid.values
            base = unit((cx, cy, z0))
            rows.append(f'Cylinder({tag}) = {{{_numbers(*base, 0.0, 0.0, height / scale, radius / scale)}}};')
        rows.append(f'tools[] += {{{tag}}};')
        if 0.0 < solid.size < h_max:
            ramp = transition if growth is None else (h_max - solid.size) / (float(growth) - 1.0)
            fields.append(_size_field(len(fields) + 1, solid, unit, scale, h_max, ramp))
        if 0.0 < solid.edge_size < h_max:
            slope = (h_max - solid.edge_size) / transition if growth is None else float(growth) - 1.0
            for distance in _edge_distance_texts(solid, unit, scale):
                fields.append(_math_eval_rows(len(fields) + 1, f'{geo_number(solid.edge_size / scale)} + '
                                                               f'{geo_number(slope)} * {distance}'))
    helpers = set()   # read by a Max field, never in the Min list; no MathEval reads another field (F<n>: gmsh hangs)
    for face in sized_faces:   # SizedFace.sizes is the numpy twin (meshing's estimate)
        if 0.0 < face.size < h_max:
            slope = (h_max - face.size) / transition if growth is None else float(growth) - 1.0
            size_number = len(fields) + 1
            for text in face_size_texts(face, origin, scale, slope):
                fields.append(_math_eval_rows(len(fields) + 1, text))
            fields.append([f'Field[{size_number + 2}] = Max;',
                           f'Field[{size_number + 2}].FieldsList = {{{size_number}, {size_number + 1}}};'])
            helpers.update((size_number, size_number + 1))
    for x, y, z, size in np.asarray(size_points if size_points is not None else (), dtype=np.float64).reshape(-1, 4):
        if 0.0 < size < h_max:   # h + (growth - 1) x distance up to h_max: Ball of radius 0 ramping over Thickness
            ramp = transition if growth is None else (h_max - size) / (float(growth) - 1.0)
            centre = unit((x, y, z))
            number = len(fields) + 1
            fields.append([f'Field[{number}] = Ball;', f'Field[{number}].Radius = 0;',
                           f'Field[{number}].Thickness = {_numbers(ramp / scale)};',
                           *(f'Field[{number}].{axis}Center = {_numbers(centre[k])};' for k, axis in enumerate('XYZ')),
                           f'Field[{number}].VIn = {_numbers(size / scale)};',
                           f'Field[{number}].VOut = {_numbers(h_max / scale)};'])
    for sized_region in sized_regions:   # the root goes into the Min list, its inner fields are helpers
        if 0.0 < sized_region.size < h_max:
            slope = (h_max - sized_region.size) / transition if growth is None else float(growth) - 1.0
            ramp = transition if growth is None else (h_max - sized_region.size) / (float(growth) - 1.0)
            _region_fields(sized_region, False, None, fields, helpers, unit, scale, h_max, slope, ramp)
    for sized_surface in sized_surfaces:   # size + slope |distance to the surface|: SizedRegion.surface_sizes
        if 0.0 < sized_surface.size < h_max:
            slope = (h_max - sized_surface.size) / transition if growth is None else float(growth) - 1.0
            outside = _region_fields(sized_surface, False, None, fields, helpers, unit, scale, h_max, slope, None,
                                     ramped=True)
            inside = _region_fields(sized_surface, True, None, fields, helpers, unit, scale, h_max, slope, None,
                                    ramped=True)
            _combined_field([outside, inside], 'Max', fields, helpers)
    if has_shell or solids:
        # every fragment is meshed and saved; the solids' parts outside the box are dropped after reading
        rows.append('BooleanFragments{ Volume{1}; Delete; }{ Volume{tools[]}; Delete; }')
    for field_rows in fields:
        rows += field_rows
    if fields:
        minimum = len(fields) + 1
        sizes = ', '.join(str(k) for k in range(1, minimum) if k not in helpers)
        rows += [f'Field[{minimum}] = Min;', f'Field[{minimum}].FieldsList = {{{sizes}}};',
                 f'Background Field = {minimum};']
    for axis in periodic_axes:
        rows += _periodic_rows('xyz'.index(axis), outer_low, outer_high)
    return '\n'.join(rows) + '\n'


def _periodic_rows(axis, low, high) -> list:
    """.geo rows making each surface on the hi side of the box (normalised low / high corners) along axis 0 / 1 / 2
    the image of the lo-side surface whose bounding box it matches after the translation by the box side."""
    name = 'xyz'[axis]
    shift = [0.0, 0.0, 0.0]
    shift[axis] = float(high[axis] - low[axis])
    slack = _EXTENT_SLACK
    side_high = [float(value) for value in high]
    side_high[axis] = float(low[axis])
    search = _numbers(*(value - slack for value in low), *(value + slack for value in side_high))
    lo, hi = f'periodic_lo_{name}', f'periodic_hi_{name}'
    image = ', '.join(f'lo_bounds({k}) {"-+"[sign]} {geo_number(slack)} + {geo_number(shift[k % 3])}'
                      for k, sign in zip(range(6), (0, 0, 0, 1, 1, 1)))
    matches = ' && '.join(f'Fabs(hi_bounds({k}) - {geo_number(shift[k % 3])} - lo_bounds({k})) < {geo_number(slack)}'
                          for k in range(6))
    return [f'// periodic {name}: hi-side surfaces are translated images of the lo side',
            f'{lo}() = Surface In BoundingBox{{{search}}};',
            f'For i In {{0 : #{lo}() - 1}}',
            f'  lo_bounds() = BoundingBox Surface{{{lo}(i)}};',
            f'  {hi}() = Surface In BoundingBox{{{image}}};',
            f'  For j In {{0 : #{hi}() - 1}}',
            f'    hi_bounds() = BoundingBox Surface{{{hi}(j)}};',
            f'    If ({matches})',
            f'      Periodic Surface{{{hi}(j)}} = {{{lo}(i)}} Translate{{{_numbers(*shift)}}};',
            '    EndIf',
            '  EndFor',
            'EndFor']


def _offset_text(axis, value) -> str:
    """MathEval text of the normalised coordinate along axis minus value."""
    return f'({"xyz"[axis]} - ({geo_number(value)}))'


def _positive_squared(text) -> str:
    """MathEval text of max(text, 0)^2, written (u * u + u * Sqrt(u * u)) / 2."""
    return f'(({text}) * ({text}) + ({text}) * Sqrt(({text}) * ({text}))) / 2'


def _edge_distance_texts(solid, unit, scale) -> list:
    """MathEval expressions (normalised x, y, z; only arithmetic and Sqrt: gmsh >= 4.8) of the distance to each edge
    of a box (12) or cylinder rim (2); max(a, 0) written (a + Sqrt(a * a)) / 2."""
    def squared(text):
        return f'{text} * {text}'

    if solid.kind == 'cylinder':
        cx, cy, z0, radius, height = solid.values
        centre = unit((cx, cy, z0))
        radial = (f'(Sqrt({squared(_offset_text(0, centre[0]))} + {squared(_offset_text(1, centre[1]))}) - '
                  f'{geo_number(radius / scale)})')
        return [f'Sqrt({squared(radial)} + {squared(_offset_text(2, rim))})'
                for rim in (centre[2], centre[2] + height / scale)]
    if solid.kind != 'box':
        return []
    low, high = unit(solid.values[:3]), unit(solid.values[3:])
    texts = []
    for along, first, second in _BOX_EDGE_AXES:
        below = f'({geo_number(low[along])} - {"xyz"[along]})'
        above = f'({"xyz"[along]} - {geo_number(high[along])})'
        outside = f'(({below} + Sqrt({squared(below)})) / 2 + ({above} + Sqrt({squared(above)})) / 2)'
        for first_value in (low[first], high[first]):
            for second_value in (low[second], high[second]):
                texts.append(f'Sqrt({squared(outside)} + {squared(_offset_text(first, first_value))} + '
                             f'{squared(_offset_text(second, second_value))})')
    return texts


def _region_fields(region, inverted, clip, fields, helpers, unit, scale, h_max, slope, ramp, ramped=False) -> int:
    """Number of the gmsh field of a SizedRegion (inverted: size outside, size + slope x depth inside), its inner
    fields appended to fields and named in helpers; De Morgan for inverted composites (SizedRegion.sizes is the twin);
    ramped: solids as size + slope x exact outside distance (MathEval) instead of Box / Ball / Cylinder fields;
    cylinder leaves always so (the Cylinder field's step left a refined ring's outer surface at h_max)."""
    if region.kind == 'solid':
        if not inverted and (ramped or region.solid.kind == 'cylinder'):
            outside = _outside_distance_text(region.solid, unit, scale)
            fields.append(_math_eval_rows(len(fields) + 1, f'{_short_number(region.size / scale)} + '
                                                           f'{_short_number(slope)} * {outside}'))
            return len(fields)
        if not inverted:
            fields.append(_size_field(len(fields) + 1, replace(region.solid, size=region.size), unit, scale, h_max,
                                      ramp))
            return len(fields)
        numbers = []
        texts = [text for text, kept in zip(_depth_texts(region.solid, unit, scale), _kept_faces(region.solid, clip))
                 if kept]
        for depth in texts:   # size + slope max(depth, 0): the Min over the faces is the ramp of the nearest one
            fields.append(_math_eval_rows(len(fields) + 1, f'{geo_number(region.size / scale)} + {geo_number(slope)} '
                                                           f'* ({depth} + Sqrt({depth} * {depth})) / 2'))
            numbers.append(len(fields))
        if not numbers:   # every face at / beyond the base's box: the base lies inside the solid
            fields.append(_math_eval_rows(len(fields) + 1, geo_number(h_max / scale)))
            numbers.append(len(fields))
        return _combined_field(numbers, 'Min', fields, helpers)
    numbers = [_region_fields(child, inverted != (region.kind == 'difference' and index > 0), child_clip, fields,
                              helpers, unit, scale, h_max, slope, ramp, ramped)
               for index, child, child_clip in _child_clips(region, clip)]
    return _combined_field(numbers, 'Min' if (region.kind == 'union') != inverted else 'Max', fields, helpers)


def _combined_field(numbers, kind, fields, helpers) -> int:
    """The one field of numbers, or a new Min / Max field over them (they become helpers)."""
    if len(numbers) == 1:
        return numbers[0]
    helpers.update(numbers)
    fields.append([f'Field[{len(fields) + 1}] = {kind};',
                   f"Field[{len(fields) + 1}].FieldsList = {{{', '.join(str(number) for number in numbers)}}};"])
    return len(fields)


def _short_number(value) -> str:
    """Plain decimal of a number at 12 significant digits, in parentheses (long MathEval texts stay below
    GEO_STRING_MAX; a negative value stays one operand)."""
    return f'({format(float(value), ".12g")})'


def _outside_distance_text(solid, unit, scale) -> str:
    """MathEval expression (normalised x, y, z; arithmetic and Sqrt only) of the exact distance outside a solid, 0
    inside (the twin _outside_distance); max(u, 0)^2 written (u * u + u * Sqrt(u * u)) / 2."""
    def offset(axis, value):
        return f'({"xyz"[axis]} - {_short_number(value)})'

    if solid.kind == 'sphere':
        centre = unit(solid.values[:3])
        radial = ' + '.join(f'{offset(k, centre[k])} * {offset(k, centre[k])}' for k in range(3))
        beyond = f'(Sqrt({radial}) - {_short_number(solid.values[3] / scale)})'
        return f'({beyond} + Sqrt({beyond} * {beyond})) / 2'
    if solid.kind == 'cylinder':
        cx, cy, z0, radius, height = solid.values
        base = unit((cx, cy, z0))
        radial = ' + '.join(f'{offset(k, base[k])} * {offset(k, base[k])}' for k in range(2))
        terms = [_positive_squared(f'Sqrt({radial}) - {_short_number(radius / scale)}'),
                 _positive_squared(f'{_short_number(base[2])} - z'),
                 _positive_squared(f'z - {_short_number(base[2] + height / scale)}')]
    else:
        low, high = unit(solid.values[:3]), unit(solid.values[3:])
        terms = [term for k, name in enumerate('xyz')
                 for term in (_positive_squared(f'{_short_number(low[k])} - {name}'),
                              _positive_squared(f'{name} - {_short_number(high[k])}'))]
    return f"Sqrt({' + '.join(terms)})"


def _depth_texts(solid, unit, scale) -> list:
    """MathEval expressions (normalised x, y, z; arithmetic and Sqrt only) of the depth below each face of a solid in
    Solid.faces order, negative outside; Solid.depth is their minimum."""
    texts = []
    for face in solid.faces():
        if face is not None:
            axis, side, coordinate = face
            value = unit(tuple(coordinate if k == axis else 0.0 for k in range(3)))[axis]
            texts.append(_offset_text(axis, value) if side == 0 else f'(({geo_number(value)}) - {"xyz"[axis]})')
            continue
        centre = unit(solid.values[:3]) if solid.kind == 'sphere' else unit((solid.values[0], solid.values[1], 0.0))
        axes = range(3) if solid.kind == 'sphere' else range(2)
        radial = ' + '.join(f'{_offset_text(axis, centre[axis])} * {_offset_text(axis, centre[axis])}' for axis in axes)
        texts.append(f'({geo_number(solid.values[3] / scale)} - Sqrt({radial}))')
    return texts


def _math_eval_rows(number, text) -> list:
    """.geo rows of MathEval field `number`; ValueError when the expression exceeds GEO_STRING_MAX (gmsh stops reading
    the .geo at a longer string and meshes without every later row)."""
    if len(text) > GEO_STRING_MAX:
        raise ValueError(f'mesher gmsh: MathEval field {number} has {len(text)} characters, gmsh reads at most '
                         f'{GEO_STRING_MAX} per .geo string')
    return [f'Field[{number}] = MathEval;', f'Field[{number}].F = "{text}";']


def face_size_texts(face, origin, scale, slope) -> tuple:
    """(size, ramp) MathEval expressions of a SizedFace (normalised x, y, z = (point - origin) / scale; arithmetic and
    Sqrt only); its size field is their Max: size + slope x max(distance to the rectangle - band, 0). max(u, 0)^2
    written (u * u + u * Sqrt(u * u)) / 2; below and above one side are never both positive, so their squares add."""
    def normalised(axis, value):
        return f'({geo_number((float(value) - float(origin[axis])) / scale)})'

    names = 'xyz'
    across = f'({names[face.axis]} - {normalised(face.axis, face.coordinate)})'
    terms = [f'{across} * {across}']
    for axis, low, high in zip((k for k in range(3) if k != face.axis), face.low, face.high):
        terms += [_positive_squared(normalised(axis, low) + ' - ' + names[axis]),
                  _positive_squared(names[axis] + ' - ' + normalised(axis, high))]
    distance = f"Sqrt({' + '.join(terms)})"
    size = geo_number(face.size / scale)
    return size, f'{size} + {geo_number(slope)} * ({distance} - ({geo_number(face.distance / scale)}))'


def _size_field(number, solid, unit, scale, h_max, transition) -> list:
    """Gmsh field rows giving `solid` its size inside, h_max beyond the transition."""
    def setting(name, value):
        return f'Field[{number}].{name} = {_numbers(value)};'

    common = [setting('VIn', solid.size / scale), setting('VOut', h_max / scale)]
    if solid.kind == 'sphere':
        centre = unit(solid.values[:3])
        return [f'Field[{number}] = Ball;', setting('Radius', solid.values[3] / scale),
                setting('Thickness', transition / scale),
                *(setting(f'{axis}Center', centre[k]) for k, axis in enumerate('XYZ')), *common]
    if solid.kind == 'box':
        low, high = unit(solid.values[:3]), unit(solid.values[3:])
        return [f'Field[{number}] = Box;', setting('Thickness', transition / scale),
                *(setting(f'{axis}Min', low[k]) for k, axis in enumerate('XYZ')),
                *(setting(f'{axis}Max', high[k]) for k, axis in enumerate('XYZ')), *common]
    cx, cy, z0, radius, height = solid.values
    centre = unit((cx, cy, z0 + 0.5 * height))
    # gmsh's Cylinder field holds |(X - centre) . axis| < |axis|^2: the axis vector is the HALF length
    return [f'Field[{number}] = Cylinder;', setting('Radius', radius / scale), setting('XAxis', 0.0),
            setting('YAxis', 0.0), setting('ZAxis', 0.5 * height / scale),
            *(setting(f'{axis}Center', centre[k]) for k, axis in enumerate('XYZ')), *common]


def read_msh_tetrahedra(path) -> tuple:
    """(nodes (n, 3), tetrahedra (m, 4) node indices, volume entity tag per tetrahedron (m,)) of a .msh 4.1 ASCII
    file (other element types skipped; gmsh_io.read_msh_blocks)."""
    tags, coordinates, parametric, blocks = read_msh_blocks(path)
    if parametric:
        raise RuntimeError('mesher gmsh -3: parametric node coordinates are not read (Mesh.SaveParametric = 0)')
    tets = [(entity, nodes[:, :4]) for entity, kind, nodes in blocks if kind == _TETRAHEDRON_TYPE]
    if not tets:
        raise RuntimeError('mesher gmsh -3: the mesh has no tetrahedra')
    return (coordinates, node_index(tags)[np.concatenate([nodes for _, nodes in tets])],
            np.concatenate([np.full(len(nodes), entity, dtype=np.int64) for entity, nodes in tets]))
