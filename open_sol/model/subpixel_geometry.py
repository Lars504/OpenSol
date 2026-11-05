"""Subpixel smoothing geometry (igor 1.5), computed once at build: supersampled rasters of the material regions, one x
strip at a time, give every E_z cell / E_x (hi-y) face / E_y (hi-x) face whose averaging volume two materials share
its fill fraction, normal and two representative cells (core.interface_geometry); 3D (em_fdtd_3d): that in-plane
geometry per run of equal cross-sections plus sampled horizontal interfaces; plus the CPU memory estimates. Backend
neutral: arrays on the module of the build's coarse index / the Domain (cupy on 'gpu', numpy on 'cpu')."""
import math
from types import SimpleNamespace

import numpy as np

from ..core.backend import array_module_of, to_host
from ..core.interface_geometry import (BYTES_PER_LOCATION, BYTES_PER_LOCATION_3D, InterfaceGeometry,
                                       InterfaceGeometry3D, InterfaceSet, InterfaceSet3D)
from ..core.memory_estimate import FLOAT32_BYTES, MemoryEstimate
from .geometry import (GEOMETRY_KINDS, cross_section_key, cross_section_runs, domain_object_boxes,
                       domain_object_rasterizer, paint_material_index, z_extent)
from .schema import subpixel_smoothing_physics

# Two rasters per build, (samples per cell along x, along y): fine along one axis, midpoint columns along the other, so
# an edge whose normal lies near the fine axis is located to 1 / (2 x fine) of a cell (a square s x s raster leaves
# axis-parallel edges quantised to 1 / s). Each location takes the raster whose fine axis is nearer its normal.
SAMPLES_FINE = 64
SAMPLES_COARSE = 4
RASTERS = ((SAMPLES_FINE, SAMPLES_COARSE), (SAMPLES_COARSE, SAMPLES_FINE))   # even: dual volumes are two half cells
STRIP_SAMPLES = 1 << 22        # fine samples per raster strip (int16 image + object masks of one strip)
LOCATION_CHUNK = 512           # interface locations per gather batch (normal windows of 9 x 256 samples each)
_HALO_CELLS = 2                # strip halo along x: hi-x dual volumes and the 3-cell normal windows
_PAINTED_KEPT = 4               # 3D: coarse (nx, ny) int16 cross-section images kept by key (2 per plain boundary)
_NORMAL_RADIUS_CELLS = 1.0     # normal: first moment of the dominant material, radial weight (1 - r^2 / R^2)+
_DEGENERATE_MOMENT = 1e-3      # |moment| below this x the window's |r| moment: normal undefined, n_d^2 = 1/2
_BUFFER_BYTES_PER_LOCATION = 4 * FLOAT32_BYTES   # physics/em/subpixel_smoothing buffers per location of a used kind
# int16 strip image + per-run images, cached object bool masks and one object's test (geometry._SEPARABLE_KINDS:
# its float64 distance sum and bool result over its bounding box, here taken as the whole strip)
_STRIP_BYTES_PER_SAMPLE = 12
_CHUNK_BYTES_PER_SAMPLE = 40   # normal window: int16 values, bool masks, float32 areas, float64 radii
_SET_FIELDS = {'location': 'int64', 'cell_a': 'int32', 'cell_b': 'int32', 'fill': 'float32',
               'normal_square': 'float32', 'upper_cell': 'int32', 'lower_share': 'float32', 'upper_share': 'float32'}
# Candidate representative cells per kind: (di, dj) from the location's lower cell, and the centre offset in cells.
_CANDIDATES = {
    'cells': [((di, dj), (di, dj)) for di in (0, -1, 1) for dj in (0, -1, 1)],
    'faces_y': [((di, dj), (di, dj - 0.5)) for di in (0, -1, 1) for dj in (0, 1, -1, 2)],
    'faces_x': [((di, dj), (di - 0.5, dj)) for di in (0, 1, -1, 2) for dj in (0, -1, 1)],
}


def interface_geometry(ctx):
    """InterfaceGeometry of a 2D BuildContext after its static materials (the materials' expanded selections painted
    supersampled (fill, normal) and at the build's cell centres (representative cells)); InterfaceGeometry3D in 3D."""
    expanded = [ctx.selection(material.selection, 'domain').entities for material in ctx.spec.materials]
    if getattr(ctx.domain, 'dimensions', 2) == 3:
        return build_interface_geometry_3d(ctx.domain, ctx.resolved_geometry, expanded)
    grid = ctx.domain.grid
    coarse_index = paint_material_index(expanded, ctx.object_window_mask, (grid.nx, grid.ny), ctx.domain.xp)
    return build_interface_geometry(ctx.domain, ctx.resolved_geometry, expanded, coarse_index)


def build_interface_geometry(domain, shapes, expanded_selections, coarse_index, rasters=RASTERS, z=None):
    """InterfaceGeometry of a 2D Domain from resolved shapes, the materials' expanded selections and the int16 (nx, ny)
    paint_material_index of the build (1 + winning material, 0 none; its array module is the result's); rasters =
    ((sx, sy), ...) even counts; z = the height of a 3D cross-section (None: 2D)."""
    from ..physics.em.yee_grid import face_shares, grid_cell_widths
    xp = array_module_of(coarse_index)
    nx, ny = domain.grid.nx, domain.grid.ny
    config = domain.boundary_config
    periodic = (bool(getattr(config, 'periodic_x', False)), bool(getattr(config, 'periodic_y', False)))
    x_edges, y_edges = domain.cell_edges()
    window_x, window_y = domain.physical_window()
    yee_dx, yee_dy = grid_cell_widths(domain.grid)
    shares = {'faces_y': tuple(xp.asarray(share) for share in face_shares(yee_dy, periodic[1], np.float32)),
              'faces_x': tuple(xp.asarray(share) for share in face_shares(yee_dx, periodic[0], np.float32))}
    candidates = {kind: [] for kind in _CANDIDATES}
    for sx, sy in rasters:
        if min(sx, sy) < 2 or sx % 2 or sy % 2:
            raise ValueError(f'raster samples per cell must be even integers >= 2, got {(sx, sy)!r}')
        fine_x = _FineAxis(x_edges, window_x, sx, periodic[0])
        fine_y = _FineAxis(y_edges, window_y, sy, periodic[1])
        context = _RasterContext(xp, (sx, sy), (nx, ny), periodic, fine_x, fine_y, coarse_index, shares,
                                 fine_axis=0 if sx >= sy else 1)
        strip_cells = max(1, STRIP_SAMPLES // (sx * sy * ny) - 2 * _HALO_CELLS)
        for i_start in range(0, nx, strip_cells):
            i_stop = min(i_start + strip_cells, nx)
            global_fine = np.arange((i_start - _HALO_CELLS) * sx, (i_stop + _HALO_CELLS) * sx)
            image = _paint(xp, expanded_selections, shapes, fine_x.wrapped(global_fine), fine_y.coordinates, z)
            strip = _Strip(context, image, i_start, i_stop, global_fine)
            for kind in _CANDIDATES:
                part = strip.locations(kind)
                if part is not None:
                    candidates[kind].append(part)
            del strip, image
    return InterfaceGeometry(**{kind: _merge(xp, candidates[kind]) for kind in _CANDIDATES},
                             supersampling=_supersampling(rasters))


def _supersampling(rasters) -> tuple:
    """(fine, coarse) samples per cell of the rasters used (InterfaceGeometry metadata)."""
    return max(max(raster) for raster in rasters), min(min(raster) for raster in rasters)


class _FineAxis:
    """Host float64 fine samples of one axis: s sub-centres per cell (positions, for moments and centres) and the
    raster coordinates (PML / ghost cells repeat the nearest window centre, as the clamped build centres), sample
    widths, unwrapped positions of any fine index."""

    def __init__(self, edges, window, s, periodic):
        widths = np.diff(edges)
        centres = 0.5 * (edges[:-1] + edges[1:])
        positions = edges[:-1, None] + widths[:, None] * ((np.arange(s) + 0.5) / s)[None, :]
        self.positions = positions.ravel()
        coordinates = positions.copy()
        coordinates[:window.start] = centres[window.start]
        coordinates[window.stop:] = centres[window.stop - 1]
        self.coordinates = coordinates.ravel()
        self.sample_widths = np.repeat(widths / s, s)
        self.size = self.coordinates.size
        self.period = float(edges[-1] - edges[0])
        self.periodic = periodic

    def index(self, fine_index):
        """Stored index of fine indices outside [0, size): wrapped (periodic) or clipped."""
        return np.mod(fine_index, self.size) if self.periodic else np.clip(fine_index, 0, self.size - 1)

    def wrapped(self, fine_index):
        """Sample coordinates inside the grid (the raster positions)."""
        return self.coordinates[self.index(fine_index)]

    def unwrapped(self, fine_index):
        """Unclamped sample positions, continuous across a periodic edge (+ period per wrap): the moment offsets, so
        an interface the clamped PML cells extend keeps its exact normal there."""
        if not self.periodic:
            return self.positions[self.index(fine_index)]
        return self.positions[self.index(fine_index)] + self.period * np.floor_divide(fine_index, self.size)


class _RasterContext:
    """Grid data shared by the strips of one raster (array module, sampling, run-backend copies of the y axis, coarse
    index, shares)."""

    def __init__(self, xp, samples, cells, periodic, fine_x, fine_y, coarse_index, shares, fine_axis):
        self.xp, self.samples, self.cells, self.periodic = xp, samples, cells, periodic
        self.fine_x, self.fine_y = fine_x, fine_y
        self.coarse_index, self.shares, self.fine_axis = coarse_index, shares, fine_axis
        self.y_widths = xp.asarray(fine_y.sample_widths, dtype=xp.float32)


def _paint(xp, expanded_selections, shapes, x_coordinates, y_coordinates, z=None):
    """int16 (len(x), len(y)) paint_material_index on fine coordinates (3D: the cross-section at height z); x split
    into non-decreasing runs (periodic wraps repeat cells) because rasterize_window needs non-decreasing centres."""
    breaks = np.flatnonzero(np.diff(x_coordinates) < 0.0) + 1
    images = []
    for run in np.split(x_coordinates, breaks):
        object_mask = domain_object_rasterizer(shapes, run, y_coordinates, xp, z=z)
        images.append(paint_material_index(expanded_selections, object_mask, (run.size, y_coordinates.size), xp))
    return images[0] if len(images) == 1 else xp.concatenate(images, axis=0)


class _Strip:
    """Fine raster of cells [i_start, i_stop) with _HALO_CELLS on each side and the interface locations it owns."""

    def __init__(self, context, image, i_start, i_stop, global_fine):
        xp = context.xp
        self.context, self.image, self.i_start, self.n_cells = context, image, i_start, i_stop - i_start
        self.x_widths = xp.asarray(context.fine_x.sample_widths[context.fine_x.index(global_fine)], dtype=xp.float32)
        self.x_unwrapped = xp.asarray(context.fine_x.unwrapped(global_fine))
        self.origin = _HALO_CELLS * context.samples[0]   # fine x index of cell i_start's first sample

    def locations(self, kind):
        """Candidate arrays (dict of run-backend arrays, 'score' included) of one kind's locations owned by the strip
        whose averaging volume holds two materials; None when there are none."""
        context, xp = self.context, self.context.xp
        sx, sy = context.samples
        n, (nx, ny) = self.n_cells, context.cells
        x_offset = self.origin + (sx // 2 if kind == 'faces_x' else 0)
        rows = self.image[x_offset:x_offset + n * sx]
        if kind == 'faces_y':
            rows = (xp.concatenate((rows[:, sy // 2:], rows[:, :sy // 2]), axis=1) if context.periodic[1]
                    else rows[:, sy // 2:sy // 2 + (ny - 1) * sy])
        blocks = rows.reshape(n, sx, rows.shape[1] // sy, sy)
        mixed = blocks.min(axis=(1, 3)) != blocks.max(axis=(1, 3))
        del blocks, rows
        if kind == 'faces_x' and not context.periodic[0] and self.i_start + n == nx:
            mixed[-1, :] = False   # the last hi-x face of a non-periodic axis is never updated
        block_i, block_j = xp.nonzero(mixed)
        chunks = [self._chunk(kind, block_i[start:start + LOCATION_CHUNK], block_j[start:start + LOCATION_CHUNK])
                  for start in range(0, int(block_i.size), LOCATION_CHUNK)]
        chunks = [chunk for chunk in chunks if chunk is not None]
        if not chunks:
            return None
        return {name: xp.concatenate([chunk[name] for chunk in chunks]) for name in chunks[0]}

    def _chunk(self, kind, block_i, block_j):
        """Fill fraction, normal, score, representatives and face-mean data of a batch of mixed locations."""
        context, xp = self.context, self.context.xp
        sx, sy = context.samples
        nx, ny = context.cells
        size_y = context.fine_y.size
        rows = xp.arange(block_i.size)
        x_start = self.origin + block_i * sx + (sx // 2 if kind == 'faces_x' else 0)
        y_start = block_j * sy + (sy // 2 if kind == 'faces_y' else 0)

        # area fractions of the two leading materials (sample areas weight the halves of a dual volume)
        volume_x = x_start[:, None] + xp.arange(sx)[None, :]
        volume_y = (y_start[:, None] + xp.arange(sy)[None, :]) % size_y
        values = self.image[volume_x[:, :, None], volume_y[:, None, :]]
        areas = self.x_widths[volume_x][:, :, None] * context.y_widths[volume_y][:, None, :]
        present = xp.unique(values)
        counts = xp.stack([(areas * (values == material)).sum(axis=(1, 2)) for material in present.tolist()])
        del values, areas
        first = xp.argmax(counts, axis=0)
        count_a = counts[first, rows]
        counts[first, rows] = -1.0
        second = xp.argmax(counts, axis=0)
        count_b = xp.maximum(counts[second, rows], 0.0)
        material_a, material_b = present[first], present[second]
        fill = (count_a / xp.maximum(count_a + count_b, 1e-30)).astype(xp.float32)

        # location centre (between the two middle samples), local cell widths, normal of material a
        centre_x = 0.5 * (self.x_unwrapped[x_start + sx // 2 - 1] + self.x_unwrapped[x_start + sx // 2])
        y_middle = to_host(y_start + sy // 2)
        centre_y = xp.asarray(0.5 * (context.fine_y.unwrapped(y_middle - 1) + context.fine_y.unwrapped(y_middle)))
        width_x = self.x_widths[x_start] * sx
        width_y = context.y_widths[y_start % size_y] * sy
        normal_x, normal_y = self._normal(material_a, x_start, y_start, centre_x, centre_y,
                                          xp.minimum(width_x, width_y) * _NORMAL_RADIUS_CELLS)
        score = xp.abs(normal_x if context.fine_axis == 0 else normal_y)
        if kind == 'cells':
            normal_square = xp.zeros(block_i.size, dtype=xp.float32)
        else:
            normal_square = ((normal_x if kind == 'faces_y' else normal_y) ** 2).astype(xp.float32)

        # representative cells: the nearest candidate whose build material is a (b)
        cell_i = self.i_start + block_i
        candidate_i = xp.stack([_wrap_cells(xp, cell_i + di, nx, context.periodic[0])
                                for (di, _), _ in _CANDIDATES[kind]], axis=1)
        candidate_j = xp.stack([_wrap_cells(xp, block_j + dj, ny, context.periodic[1])
                                for (_, dj), _ in _CANDIDATES[kind]], axis=1)
        distance = xp.stack([(offset_x * width_x) ** 2 + (offset_y * width_y) ** 2
                             for _, (offset_x, offset_y) in _CANDIDATES[kind]], axis=1)
        candidate_material = context.coarse_index[candidate_i, candidate_j]
        cell_a, found_a = _nearest(xp, candidate_i, candidate_j, distance, candidate_material == material_a[:, None],
                                   ny)
        cell_b, found_b = _nearest(xp, candidate_i, candidate_j, distance, candidate_material == material_b[:, None],
                                   ny)
        keep = found_a & found_b & (count_b > 0.0)
        if not bool(keep.any()):
            return None

        location = cell_i * ny + block_j
        if kind == 'cells':
            upper = location
            lower_share = xp.ones(block_i.size, dtype=xp.float32)
            upper_share = xp.zeros(block_i.size, dtype=xp.float32)
        else:
            if kind == 'faces_y':
                upper, along = cell_i * ny + _wrap_cells(xp, block_j + 1, ny, True), block_j
            else:
                upper, along = _wrap_cells(xp, cell_i + 1, nx, True) * ny + block_j, cell_i
            lower_share, upper_share = (share[along] for share in context.shares[kind])
        chunk = {'location': location, 'cell_a': cell_a, 'cell_b': cell_b, 'fill': fill,
                 'normal_square': normal_square, 'upper_cell': upper, 'lower_share': lower_share,
                 'upper_share': upper_share, 'score': score}
        return {name: array[keep] for name, array in chunk.items()}

    def _normal(self, material_a, x_start, y_start, centre_x, centre_y, radius):
        """(n_x, n_y) float64 unit normal from the first moment of material a with the radial weight (1 - r^2 / R^2)+
        over the 3 x 3 cell window of fine samples around the location centre (radial symmetry: exact direction for a
        straight interface); (sqrt(1/2), sqrt(1/2)) where the moment vanishes."""
        context, xp = self.context, self.context.xp
        sx, sy = context.samples
        window_x = (x_start - sx)[:, None] + xp.arange(3 * sx)[None, :]
        window_y_raw = to_host((y_start - sy)[:, None] + xp.arange(3 * sy)[None, :])
        window_y = xp.asarray(context.fine_y.index(window_y_raw))
        offset_x = self.x_unwrapped[window_x] - centre_x[:, None]
        offset_y = xp.asarray(context.fine_y.unwrapped(window_y_raw)) - centre_y[:, None]
        radius_square = offset_x[:, :, None] ** 2 + offset_y[:, None, :] ** 2
        weights = xp.maximum(1.0 - radius_square / (radius ** 2)[:, None, None], 0.0)
        weights *= self.x_widths[window_x][:, :, None] * context.y_widths[window_y][:, None, :]
        scale = (weights * xp.sqrt(radius_square)).sum(axis=(1, 2))
        del radius_square
        weights *= self.image[window_x[:, :, None], window_y[:, None, :]] == material_a[:, None, None]
        moment_x = (weights * offset_x[:, :, None]).sum(axis=(1, 2))
        moment_y = (weights * offset_y[:, None, :]).sum(axis=(1, 2))
        length = xp.sqrt(moment_x ** 2 + moment_y ** 2)
        degenerate = length <= _DEGENERATE_MOMENT * scale
        safe_length = xp.where(degenerate, 1.0, length)
        diagonal = math.sqrt(0.5)
        return (xp.where(degenerate, diagonal, moment_x / safe_length),
                xp.where(degenerate, diagonal, moment_y / safe_length))


def _wrap_cells(xp, index, count, periodic):
    """Cell indices wrapped (periodic) or clipped into [0, count)."""
    return index % count if periodic else xp.clip(index, 0, count - 1)


def _nearest(xp, candidate_i, candidate_j, distance, match, ny):
    """(int64 flat index of the nearest matching candidate, bool found) per row."""
    masked = xp.where(match, distance, np.inf)
    choice = xp.argmin(masked, axis=1)
    rows = xp.arange(choice.size)
    found = xp.isfinite(masked[rows, choice])
    return candidate_i[rows, choice] * ny + candidate_j[rows, choice], found


def _merge(xp, parts, fields=None, set_class=None):
    """InterfaceSet (fields _SET_FIELDS; 3D: _SET_FIELDS_3D / InterfaceSet3D) of the candidates: per location the one
    of the highest score (2D: the raster whose fine axis is nearest the normal), sorted by location; empty arrays when
    there are none."""
    fields = _SET_FIELDS if fields is None else fields
    set_class = InterfaceSet if set_class is None else set_class
    if not parts:
        return set_class(**{name: xp.zeros(0, dtype=dtype) for name, dtype in fields.items()})
    joined = {name: xp.concatenate([part[name] for part in parts]) for name in parts[0]}
    order = xp.lexsort(xp.stack((-joined['score'].astype(xp.float64), joined['location'].astype(xp.float64))))
    location = joined['location'][order]
    first = xp.ones(location.size, dtype=xp.bool_)
    first[1:] = location[1:] != location[:-1]
    chosen = order[first]
    return set_class(**{name: xp.ascontiguousarray(joined[name][chosen].astype(dtype)) for name, dtype in fields.items()})


# ---- 3D (em_fdtd_3d): in-plane geometry per z run + horizontal interfaces sampled across run boundaries ----
# Approximation: within a run of z cells sharing one cross-section the interfaces are vertical (exact for z prisms; a
# taper / sphere is a stack of one-layer runs); between runs the cell-centre columns are sampled at Z_SAMPLES heights per
# half cell. Per location the candidate of the highest score wins (_SCORES).
Z_SAMPLES = 16
_KINDS_3D = ('faces_x', 'faces_y', 'faces_z')
_SCORES = {'z_normal': 3.0, 'in_plane': 2.0, 'in_plane_upper': 1.5, 'z_tangential': 1.0}
_SET_FIELDS_3D = {'location': 'int64', 'cell_a': 'int64', 'cell_b': 'int64', 'fill': 'float32',
                  'normal_square': 'float32'}


class _PlaneView:
    """The (x, y) plane of a Domain3D as the 2D Domain build_interface_geometry reads (grid widths, periodic flags,
    cell edges, physical window)."""

    def __init__(self, domain):
        grid, config = domain.grid, domain.boundary_config
        self.grid = SimpleNamespace(nx=grid.nx, ny=grid.ny, dx_arr=grid.dx_arr, dy_arr=grid.dy_arr)
        self.boundary_config = SimpleNamespace(periodic_x=_periodic(config, 'x'), periodic_y=_periodic(config, 'y'))
        self._edges = domain.cell_edges()[:2]
        self._window = domain.physical_window()[:2]

    def cell_edges(self):
        return self._edges

    def physical_window(self):
        return self._window


def _periodic(config, axis) -> bool:
    return config is not None and bool(config.periodic(axis))


def build_interface_geometry_3d(domain, shapes, expanded_selections, rasters=RASTERS, z_samples=Z_SAMPLES):
    """InterfaceGeometry3D of a Domain3D (approximation in the section header, arrays on domain.xp): per run of z cells
    with one cross-section (cross_section_runs at the clamped build centres) the 2D geometry lifted to its layers (E_x /
    E_y faces with the other in-plane normal component, E_z faces the cells' area mean); per run boundary the E_z faces
    whose two cells differ (not in-plane mixed) get the sampled z fill with n_z = 1, and E_x / E_y faces of the two
    boundary layers whose two columns hold the same material pair a tangential z fill."""
    xp = domain.xp
    nx, ny, nz = domain.grid.shape
    x_centres, y_centres, z_centres = domain.host_centres()
    names = sorted({name for entities in expanded_selections if entities is not None for name in entities
                    if name in shapes and GEOMETRY_KINDS[shapes[name].kind].entity_kind == 'domain'})
    candidates = {kind: [] for kind in _KINDS_3D}
    if names:
        reuse, painted = {}, {}   # painted: the last few images by cross-section key (plain stacks repeat 2)

        def coarse_at(z):
            key = tuple(cross_section_key(name, shapes, float(z)) for name in names)
            if key not in painted:
                object_mask = domain_object_rasterizer(shapes, x_centres, y_centres, xp, z=float(z), reuse=reuse)
                if len(painted) >= _PAINTED_KEPT:
                    painted.pop(next(iter(painted)))
                painted[key] = paint_material_index(expanded_selections, object_mask, (nx, ny), xp)
            return painted[key]

        periodic_z = _periodic(domain.boundary_config, 'z')
        plane = _PlaneView(domain)
        runs = cross_section_runs(names, shapes, z_centres)
        run_data = []   # per run: (coarse int16 (nx, ny) at the run's centres, 2D InterfaceGeometry)
        for k_start, k_stop in runs:
            z = float(z_centres[k_start])
            coarse = coarse_at(z)
            geometry = build_interface_geometry(plane, shapes, expanded_selections, coarse, rasters, z=z)
            run_data.append((coarse, geometry))
            layers = xp.arange(k_start, k_stop, dtype=xp.int64)
            z_layers = layers if periodic_z or k_stop < nz else layers[:-1]   # the last z face is never updated
            for kind, part, layer_set, complement in (('faces_x', geometry.faces_x, layers, True),
                                                      ('faces_y', geometry.faces_y, layers, True),
                                                      ('faces_z', geometry.cells, z_layers, False)):
                candidates[kind].append(_lifted(xp, part, layer_set, layer_set, nz, complement, _SCORES['in_plane']))
        pairs = [(index - 1, index) for index in range(1, len(runs))]
        if periodic_z and len(runs) > 1:
            pairs.append((len(runs) - 1, 0))
        for lower, upper in pairs:
            _boundary_candidates(xp, domain, coarse_at, (runs[lower], runs[upper]), (run_data[lower], run_data[upper]),
                                 z_samples, candidates)
    return InterfaceGeometry3D(**{kind: _merge(xp, [part for part in candidates[kind] if part is not None],
                                               _SET_FIELDS_3D, InterfaceSet3D)
                                  for kind in _KINDS_3D},
                               supersampling=_supersampling(rasters), z_samples=int(z_samples))


def _lifted(xp, part, location_layers, cell_layers, nz, complement, score):
    """Candidate dict of a 2D InterfaceSet on z layers (flat 2D f -> f nz + k): locations on location_layers, cells on
    cell_layers; complement = 1 - n^2 (the other in-plane component); None when empty."""
    if part.size == 0 or location_layers.size == 0:
        return None

    def lift(flat, layers):
        return (flat.astype(xp.int64)[None, :] * nz + layers[:, None]).ravel()
    count = int(location_layers.size)
    normal_square = 1.0 - part.normal_square if complement else part.normal_square
    return {'location': lift(part.location, location_layers), 'cell_a': lift(part.cell_a, cell_layers),
            'cell_b': lift(part.cell_b, cell_layers), 'fill': xp.tile(part.fill, count),
            'normal_square': xp.tile(normal_square.astype(xp.float32), count),
            'score': xp.full(count * part.size, score, dtype=xp.float32)}


def _boundary_candidates(xp, domain, coarse_at, runs, data, z_samples, candidates):
    """Append the run boundary's candidates (build_interface_geometry_3d) between the last layer kb of the lower run
    and the first layer ku of the upper run (ku = 0 on a periodic z wrap)."""
    nx, ny, nz = domain.grid.shape
    (_, lower_stop), (upper_start, _) = runs
    (coarse_a, geometry_a), (coarse_b, geometry_b) = data
    kb, ku = lower_stop - 1, upper_start
    z_edges, z_centres = domain.cell_edges()[2], domain.cell_centers()[2]
    window = domain.physical_window()[2]
    period = float(z_edges[-1] - z_edges[0])
    centre_lo, edge = float(z_centres[kb]), float(z_edges[kb + 1])
    centre_hi = float(z_centres[ku]) + (period if ku <= kb else 0.0)
    clamp = (float(z_centres[window.start]), float(z_centres[window.stop - 1]))
    periodic_z = _periodic(domain.boundary_config, 'z')

    def fractions(start, length):
        """Fractions (nx, ny) of the samples in [start, start + length] holding the lower / upper run's material."""
        share_a = xp.zeros((nx, ny), dtype=xp.float32)
        share_b = xp.zeros((nx, ny), dtype=xp.float32)
        for sample in range(z_samples):
            height = start + length * (sample + 0.5) / z_samples
            if periodic_z:
                height = height - period if height > float(z_edges[-1]) else height
            else:
                height = min(max(height, clamp[0]), clamp[1])
            image = coarse_at(height)
            share_a += image == coarse_a
            share_b += image == coarse_b
        return share_a / z_samples, share_b / z_samples

    length_lo, length_hi = edge - centre_lo, centre_hi - edge
    lower_a, lower_b = fractions(centre_lo, length_lo)   # upper half of cell kb
    upper_a, upper_b = fractions(edge, length_hi)         # lower half of cell ku
    columns = xp.arange(nx * ny, dtype=xp.int64).reshape(nx, ny)

    # E_z faces (kb, kb + 1): z normal where the two cells differ and no run holds the column in-plane mixed
    in_plane = xp.zeros(nx * ny, dtype=xp.bool_)
    for geometry in (geometry_a, geometry_b):
        in_plane[geometry.cells.location] = True
    count_a = lower_a * length_lo + upper_a * length_hi
    count_b = lower_b * length_lo + upper_b * length_hi
    # an all-one-material sampling (interface between the last z sample and a cell centre) keeps fill 0 / 1: dropped,
    # the face would fall back to the arithmetic two-cell mean
    keep = (coarse_a != coarse_b) & ~in_plane.reshape(nx, ny) & (count_a + count_b > 0)
    flat = columns[keep]
    if flat.size:
        fill = count_a[keep] / (count_a[keep] + count_b[keep])
        candidates['faces_z'].append({'location': flat * nz + kb, 'cell_a': flat * nz + kb, 'cell_b': flat * nz + ku,
                                      'fill': fill.astype(xp.float32),
                                      'normal_square': xp.ones(flat.size, dtype=xp.float32),
                                      'score': xp.full(flat.size, _SCORES['z_normal'], dtype=xp.float32)})
    # columns in-plane mixed only in the upper run: its cell entries on the boundary E_z face
    candidates['faces_z'].append(_lifted(xp, geometry_b.cells, xp.full(1, kb, dtype=xp.int64),
                                         xp.full(1, ku, dtype=xp.int64), nz, False, _SCORES['in_plane_upper']))

    # E_x / E_y faces of layers kb and ku: both columns hold the same (lower, upper) material pair
    below = float(z_centres[kb] - z_edges[kb])        # lower half of cell kb: the lower run's material
    above = float(z_edges[ku + 1] - z_centres[ku])    # upper half of cell ku: the upper run's material
    for kind, axis in (('faces_x', 0), ('faces_y', 1)):
        periodic = _periodic(domain.boundary_config, 'xy'[axis])

        def partner(values):
            return xp.roll(values, -1, axis=axis)
        pair = (coarse_a == partner(coarse_a)) & (coarse_b == partner(coarse_b)) & (coarse_a != coarse_b)
        if not periodic:   # the last face of a non-periodic axis is never updated
            pair[(slice(None),) * axis + (-1,)] = False
        # (layer, other layer, z length of the layer's own material, of the other one) averaged over the two columns
        for layer, other, own_length, other_length in (
                (kb, ku, below + length_lo * 0.5 * (lower_a + partner(lower_a)),
                 length_lo * 0.5 * (lower_b + partner(lower_b))),
                (ku, kb, above + length_hi * 0.5 * (upper_b + partner(upper_b)),
                 length_hi * 0.5 * (upper_a + partner(upper_a)))):
            keep = pair & (other_length > 0)
            flat = columns[keep]
            if not flat.size:
                continue
            own, foreign = own_length[keep], other_length[keep]
            candidates[kind].append({'location': flat * nz + layer, 'cell_a': flat * nz + layer,
                                     'cell_b': flat * nz + other, 'fill': (own / (own + foreign)).astype(xp.float32),
                                     'normal_square': xp.zeros(flat.size, dtype=xp.float32),
                                     'score': xp.full(flat.size, _SCORES['z_tangential'], dtype=xp.float32)})


# ---- CPU memory estimate (GUI-thread safe) ----
def _local_width(edges, lo, hi) -> float:
    """Mean width of the mesh cells overlapping [lo, hi] along one axis (the finest cell when none does)."""
    edges = np.asarray(edges, dtype=np.float64)
    widths = np.diff(edges)
    overlapping = (edges[1:] > lo) & (edges[:-1] < hi)
    return float(widths[overlapping].mean()) if overlapping.any() else float(widths.min())


def estimated_location_count(spec, shapes, mesh) -> int:
    """Rough interface location count: 3 per cell along the bounding-box perimeter of every object a material paints
    (not 'all'), capped at 3 per cell; 3D: 3 per cell face over the bounding-box sides and top / bottom (z extent
    clipped to the mesh); cells of the mean local widths under the box (a graded mesh's finest cell overcounts)."""
    three_d = mesh.z_edges is not None
    interface_cells = 0.0
    selections = [material.selection for material in spec.materials]
    for name, (x0, y0, x1, y1) in domain_object_boxes(spec, shapes, selections, skip_all=True):
        width, depth = max(x1 - x0, 0.0), max(y1 - y0, 0.0)
        dx, dy = _local_width(mesh.x_edges, x0, x1), _local_width(mesh.y_edges, y0, y1)
        if not three_d:
            interface_cells += 2.0 * (depth / dy + width / dx)
            continue
        z_lo, z_hi = z_extent(name, shapes)
        z_lo, z_hi = max(z_lo, float(mesh.z_edges[0])), min(z_hi, float(mesh.z_edges[-1]))
        height = max(z_hi - z_lo, 0.0)
        dz = _local_width(mesh.z_edges, z_lo, z_hi)
        interface_cells += 2.0 * (depth * height / (dy * dz) + width * height / (dx * dz) + width * depth / (dx * dy))
    return int(min(3.0 * interface_cells, 3.0 * float(mesh.n_cells)))


def subpixel_memory_estimate(spec, registry, shapes, mesh) -> MemoryEstimate:
    """Interface geometry + the averaging buffers of every honouring physics (two of the three kinds each), build
    extra = the larger of one strip's painting and its image + a gather batch (one kind's share of the strip), the
    coarse paint image and both rasters' candidates; empty when off."""
    honouring, _ = subpixel_smoothing_physics(spec, registry)
    if not honouring:
        return MemoryEstimate()
    n_locations = estimated_location_count(spec, shapes, mesh)
    if mesh.z_edges is not None:
        return _memory_estimate_3d(n_locations, mesh, len(honouring))
    samples_per_cell = SAMPLES_FINE * SAMPLES_COARSE
    n_cells = int(mesh.n_cells)
    strip_samples = min(STRIP_SAMPLES + 2 * _HALO_CELLS * samples_per_cell * int(mesh.ny), n_cells * samples_per_cell)
    # a gather batch holds one kind's locations of one strip (n_locations counts 3 kinds over every strip); painting
    # is over by then, only the int16 strip image stays
    strip_share = min(1.0, strip_samples / (n_cells * samples_per_cell))
    batch_locations = min(LOCATION_CHUNK, max(math.ceil(n_locations / 3 * strip_share), 1))
    build_extra = (max(_STRIP_BYTES_PER_SAMPLE * strip_samples,
                       2 * strip_samples + _CHUNK_BYTES_PER_SAMPLE * batch_locations * 9 * samples_per_cell)
                   + 2 * n_cells + 3 * BYTES_PER_LOCATION * n_locations)
    buffers = _BUFFER_BYTES_PER_LOCATION * (2 * n_locations // 3) * len(honouring)
    return MemoryEstimate.from_items((
        ('interface geometry (approximate)', BYTES_PER_LOCATION * n_locations),
        (f'averaging buffers ({len(honouring)} physics, approximate)', buffers),
    ), build_extra=build_extra).labelled('subpixel smoothing')


_FACE_INDEX_BYTES_PER_CELL = 2        # em_fdtd_3d uint16 face override index (physics/em3d/subpixel_smoothing3d)
_FACE_BYTES_PER_LOCATION = 30         # its per-location kind positions (int64), values (3 x float32), ids, cells
_REFRESH_BYTES_PER_LOCATION = 48      # re-quantisation temporaries of a refresh (material_index3d on 1D arrays)


def _memory_estimate_3d(n_locations, mesh, n_physics) -> MemoryEstimate:
    """3D: interface geometry + per honouring physics the full-grid face index and its per-location arrays; build extra =
    one cross-section's 2D geometry build (its gather batch: one kind's locations of one z layer, at most
    LOCATION_CHUNK), both runs' z-sample fractions, the lifted candidates, a refresh."""
    n_plane = int(mesh.nx) * int(mesh.ny)
    samples_per_cell = SAMPLES_FINE * SAMPLES_COARSE
    strip_samples = min(STRIP_SAMPLES + 2 * _HALO_CELLS * samples_per_cell * int(mesh.ny), n_plane * samples_per_cell)
    section_locations = min(LOCATION_CHUNK, max(n_locations // (3 * max(int(mesh.nz), 1)), 1))
    build_extra = (_STRIP_BYTES_PER_SAMPLE * strip_samples + 4 * n_plane + 6 * FLOAT32_BYTES * n_plane
                   + 2 * (BYTES_PER_LOCATION_3D + FLOAT32_BYTES) * n_locations
                   + _CHUNK_BYTES_PER_SAMPLE * section_locations * 9 * samples_per_cell
                   + _REFRESH_BYTES_PER_LOCATION * n_locations)
    face_bytes = _FACE_INDEX_BYTES_PER_CELL * int(mesh.n_cells) + _FACE_BYTES_PER_LOCATION * n_locations
    return MemoryEstimate.from_items((
        ('interface geometry 3D (approximate)', BYTES_PER_LOCATION_3D * n_locations),
        (f'face override index ({n_physics} physics, approximate)', face_bytes * n_physics),
    ), build_extra=build_extra).labelled('subpixel smoothing')
