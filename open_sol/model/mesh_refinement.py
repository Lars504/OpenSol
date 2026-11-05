"""Refined MeshSpecs of the S7.7 refinement levels (CPU, no GPU work): a mesh frozen to explicit numeric sizes, the
'uniform' sequence (every size x factor), the 'adaptive' tensor bands from marked indicator cells and the adaptive
triangle sizes (MeshSpec.size_points at the nodes of marked elements, rebuilt by the mesher)."""
import math
from dataclasses import replace

import numpy as np

from .geometry import GEOMETRY_KINDS, axis_parallel_faces, domain_map, domain_size, domain_size_z
from .meshing import (MESH_ESTIMATE_ERRORS, UNIFORM_AUTO_DIVISORS, _auto_sizes, _axis_plans, _explicit_refinements,
                      _features_requiring_uniform_mesh, estimate_mesh, mesh_values)
from .spec import REFINEMENT_PLACES, MeshAxisSpec, MeshBandSpec, RefinementSpec

# an object refinement this close to the base size changes no cell size but makes the object's flat faces cell edges
SNAP_DX_FRACTION = 1.0 - 1e-6
BAND_MARGIN_CELLS = 2          # cells added on each side of a run of marked cells
ANISOTROPY_SHARE = 0.5         # anisotropic: an axis refines a marked cell when its part >= this x the largest part
COUNT_RTOL = 0.05              # the marked-cell bisection stops within this of the target cell count
MAX_BISECTIONS = 30
SIZE_POINT_RTOL = 1e-3         # triangles: a node gets a size point when its new size is this much below the field
GRADING_ROUNDS = 3             # triangles: threshold rounds correcting the plain count by the graded prediction


def refined_axes(spec, parameters) -> tuple:
    """Axes the adaptive bands refine: ('x', 'y') on a 2D Cartesian window, ('x',) on a 1D model or a polar /
    spherical window (radius only: the angle takes no bands); ValueError in 3D."""
    if spec.dimensions == 3:
        raise ValueError("adaptive refinement of 3D meshes is not supported yet; use adaptation method 'uniform'")
    if spec.dimensions == 1 or domain_map(spec, parameters).mapped:
        return ('x',)
    return ('x', 'y')


def frozen_mesh_spec(spec, mesh_spec, parameters, registry, shapes):
    """Copy of mesh_spec with every size a number and auto mode replaced by the grid it gives (nonuniform at its base
    size with its object refinements, or uniform for a lattice Boltzmann mesh); segments become custom edges; size
    points are kept."""
    mesh = mesh_values(mesh_spec, parameters)
    mode, dx = mesh['mode'], mesh['dx']
    refinements = _explicit_refinements(spec, mesh_spec, parameters, shapes, REFINEMENT_PLACES)   # sizing refuses
    if mode == 'auto':
        auto = _auto_sizes(spec, spec.physics_on(mesh_spec.name), parameters, registry, shapes, mesh)
        dx = auto.base_dx
        mode, refinements = ('uniform', []) if auto.rule == 'uniform_flow' else ('nonuniform',
                                                                                  auto.refinements + refinements)
    axes = []   # tensor meshes, builtin and swept tetrahedra (meshing.graded_window_edges) grade by them
    graded_builtin = mesh['kind'] == 'tetrahedra' and (mesh['mesher'] == 'builtin' or mesh['sweep_axis'] == 'z')
    if mode != 'uniform' and (mesh['kind'] == 'tensor' or graded_builtin):
        size_x, size_y = domain_size(spec, parameters)
        size_z = domain_size_z(spec, parameters)
        lengths = {'x': size_x, 'y': size_y} if size_z is None else {'x': size_x, 'y': size_y, 'z': size_z}
        given = {axis_spec.axis for axis_spec in mesh_spec.axes}
        for axis, plan in _axis_plans(mesh_spec, parameters, lengths).items():
            if axis in given:
                axes.append(MeshAxisSpec(axis, max_ratio=plan.max_ratio,
                                         bands=[MeshBandSpec(float(lo), float(hi), float(band_dx))
                                                for lo, hi, band_dx in plan.bands],
                                         edges=[] if plan.edges is None else [float(edge) for edge in plan.edges]))
    return replace(
        mesh_spec, kind=mesh['kind'], mode=mode, dx=float(dx), max_ratio=float(mesh['max_ratio']),
        wavelength_min=None, points_per_wavelength=mesh['points_per_wavelength'],
        refinements=[] if mode == 'uniform' else [_refinement_spec(refinement) for refinement in refinements],
        outer_dx=mesh['outer_dx'], outer_ratio=mesh['outer_ratio'], axes=axes, h_min=mesh['h_min'],
        curvature_factor=mesh['curvature_factor'], mesher=mesh['mesher'], pml_layers=mesh['pml_layers'])


def uniform_mesh_spec(frozen, factor):
    """The next 'uniform' level of a frozen MeshSpec: every size (dx, outer_dx, h_min, refinement, band and size-point
    sizes) x factor, custom axis edges resampled to 1 / factor as many cells; margins unchanged."""
    def scaled(value):
        return None if value is None else float(value) * factor

    refinements = [replace(refinement, dx=scaled(refinement.dx), dx_x=scaled(refinement.dx_x),
                           dx_y=scaled(refinement.dx_y), dx_z=scaled(refinement.dx_z))
                   for refinement in frozen.refinements]
    axes = [replace(axis_spec, bands=[replace(band, dx=scaled(band.dx)) for band in axis_spec.bands],
                    edges=_resampled_edges(axis_spec.edges, 1.0 / factor), segments=[])
            for axis_spec in frozen.axes]
    size_points = [[*map(float, entry[:-1]), float(entry[-1]) * factor] for entry in frozen.size_points]
    face_sizes = [replace(face_size, size=f'({face_size.size}) * {factor!r}' if isinstance(face_size.size, str)
                          else float(face_size.size) * factor) for face_size in frozen.face_sizes]
    return replace(frozen, dx=scaled(frozen.dx), outer_dx=scaled(frozen.outer_dx), h_min=scaled(frozen.h_min),
                   refinements=refinements, axes=axes, size_points=size_points, face_sizes=face_sizes)


def adaptive_mesh_spec(spec, frozen, estimate, parts, growth, anisotropic, parameters, registry, shapes,
                       triangle_mesh=None, element_order=2):
    """(MeshSpec, its estimate, marked cells) of the next 'adaptive' level of a tensor mesh: the cells of largest
    eta^2 (parts: per refined_axes axis, window-shaped) marked by bisection so the estimated cell count ~ growth x
    estimate.n_cells, turned into axis bands at about half the local width (BAND_MARGIN_CELLS on each side); None
    when no cell has a positive indicator. Triangles: adaptive_triangle_spec (parts = eta^2 per element of
    triangle_mesh, the built fem TriangleMesh). ValueError in 3D or for a mesh that must stay uniform."""
    if frozen.kind == 'triangles':
        if triangle_mesh is None:
            raise ValueError(f"mesh '{frozen.name}': adaptive refinement of a triangle mesh needs the built mesh")
        return adaptive_triangle_spec(spec, frozen, triangle_mesh, parts, element_order, growth, parameters, registry,
                                      shapes)
    axes = refined_axes(spec, parameters)
    on_mesh = spec.physics_on(frozen.name)
    uniform_features = _features_requiring_uniform_mesh(on_mesh, registry) + [
        f"{physics.name} ({physics.type}: lattice Boltzmann)" for physics in on_mesh
        if physics.type in UNIFORM_AUTO_DIVISORS]
    if uniform_features:
        raise ValueError(f"mesh '{frozen.name}': {', '.join(uniform_features)} need a uniform mesh, so adaptive bands "
                         "cannot apply; use adaptation method 'uniform'")
    windows = _window_axes(estimate)
    parts = np.asarray(parts, dtype=np.float64)
    if parts.shape != (2, windows['x'][0].size, windows['y'][0].size):
        raise ValueError(f"mesh '{frozen.name}': indicator parts of shape {parts.shape} do not match its window "
                         f"({windows['x'][0].size} x {windows['y'][0].size} cells)")
    custom = {axis_spec.axis for axis_spec in frozen.axes if axis_spec.edges}
    axes = tuple(axis for axis in axes if axis not in custom)   # a custom-edge axis ignores bands
    if not axes:
        raise ValueError(f"mesh '{frozen.name}': every refinable axis has custom edges; use adaptation method "
                         "'uniform'")
    base = replace(frozen, mode='nonuniform')
    if spec.dimensions == 2 and not domain_map(spec, parameters).mapped:
        base = replace(base, refinements=list(base.refinements) + snap_refinements(spec, shapes, base))
    total = parts.sum(axis=0)
    order = np.argsort(-total, axis=None)
    n_positive = int(np.count_nonzero(total > 0.0))
    if n_positive == 0:
        return None
    axis_index = {axis: index for index, axis in enumerate(('x', 'y'))}
    largest = np.max(parts[[axis_index[axis] for axis in axes]], axis=0)
    qualifies = {axis: (parts[axis_index[axis]] >= ANISOTROPY_SHARE * largest) if anisotropic
                 else np.ones(total.shape, dtype=bool) for axis in axes}
    target = growth * estimate.n_cells
    tried = {}

    def candidate(n_marked):
        if n_marked not in tried:
            marked = np.zeros(total.size, dtype=bool)
            marked[order[:n_marked]] = True
            marked = marked.reshape(total.shape)
            bands = {}
            for axis in axes:
                other = tuple(index for index in range(total.ndim) if index != axis_index[axis])
                indices = np.nonzero(np.any(marked & qualifies[axis], axis=other))[0]
                widths, edges = windows[axis]
                bands[axis] = axis_bands(indices, widths, edges, base.dx)
            mesh_spec = with_bands(base, bands)
            candidate_spec = replace(spec, meshes=[mesh_spec if mesh.name == frozen.name else mesh
                                                   for mesh in spec.meshes])
            try:
                count_estimate = estimate_mesh(candidate_spec, parameters, registry, frozen.name)
            except MESH_ESTIMATE_ERRORS:   # e.g. the cell-count guard: too many cells
                count_estimate = None
            tried[n_marked] = (mesh_spec, count_estimate)
        return tried[n_marked]

    def count(n_marked):
        found = candidate(n_marked)[1]
        return math.inf if found is None else found.n_cells

    low, high = 1, n_positive
    if count(high) <= target:
        low = high
    elif count(low) < target:
        for _ in range(MAX_BISECTIONS):
            if high - low <= 1:
                break
            middle = min(max(int(round(math.sqrt(low * high))), low + 1), high - 1)
            middle_count = count(middle)
            if abs(middle_count / target - 1.0) <= COUNT_RTOL:
                low = middle
                break
            if middle_count <= target:
                low = middle
            else:
                high = middle
    mesh_spec, count_estimate = candidate(low)
    if count_estimate is None:
        raise ValueError(f"mesh '{frozen.name}': even the refinement of the cell with the largest error indicator "
                         'gives an invalid mesh')
    return mesh_spec, count_estimate, low


def adaptive_triangle_spec(spec, frozen, mesh, eta_squared, element_order, growth, parameters, registry, shapes):
    """(MeshSpec, its estimate, marked elements) of the next 'adaptive' level of a triangle mesh (D14, plan 3.4): per
    window element h_new = h min(1, (eta_target / eta)^(1 / (p + 1))) (h its equilateral size, p the element order),
    eta_target by bisection so the predicted window elements ~ growth x the current ones; the new sizes go to the nodes
    of the marked elements as MeshSpec.size_points (where finer than the mesh's size field) and the mesh is rebuilt
    from them; None when no element has a positive indicator."""
    from ..fem.indicators import element_sizes
    from .meshing import quantised_size_points, size_point_function, triangle_sizing
    from .triangle_meshing import mesher_size_field

    eta_squared = np.nan_to_num(np.asarray(eta_squared, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if eta_squared.shape != (mesh.n_elements,):
        raise ValueError(f"mesh '{frozen.name}': indicator of shape {eta_squared.shape} does not match its "
                         f'{mesh.n_elements} elements')
    window = mesh.pml_faces == 0
    eta = np.sqrt(np.where(window, np.maximum(eta_squared, 0.0), 0.0))
    positive = eta > 0.0
    if not positive.any():
        return None
    sizing = triangle_sizing(spec, frozen, parameters, registry, shapes)
    old_field = mesher_size_field(sizing, shapes)
    sizes = element_sizes(mesh)
    exponent = 1.0 / (float(element_order) + 1.0)
    window_nodes = np.unique(mesh.triangles[window])
    node_field = np.full(mesh.n_points, np.inf)
    node_field[window_nodes] = old_field(mesh.points[window_nodes])
    corners = mesh.points[mesh.triangles[window]]   # graded prediction samples: centroid and corners
    samples = np.concatenate([corners.mean(axis=1, keepdims=True), corners], axis=1).reshape(-1, 2)
    old_samples = old_field(samples).reshape(-1, 4)
    current = int(window.sum())
    target = float(growth) * current

    def factors(eta_target):
        ratio = np.where(positive, eta_target / np.where(positive, eta, 1.0), np.inf)
        return np.minimum(1.0, ratio ** exponent)

    def plain_count(eta_target):   # every window element split by (h / h_new)^2, no grading around it
        new = np.maximum(sizes * factors(eta_target), sizing.h_min)
        return float(np.sum(np.maximum((sizes / new) ** 2, 1.0)[window]))

    def size_points(eta_target):
        factor = factors(eta_target)
        marked = window & (factor < 1.0)
        node_size = np.full(mesh.n_points, np.inf)
        np.minimum.at(node_size, mesh.triangles[marked].ravel(), np.repeat(sizes[marked] * factor[marked], 3))
        node_size = np.maximum(node_size, sizing.h_min)
        chosen = np.flatnonzero(node_size < (1.0 - SIZE_POINT_RTOL) * node_field)
        return np.column_stack([mesh.points[chosen], node_size[chosen]]), int(marked.sum())

    def graded_count(points):   # window elements of the old mesh x mean (h_old / h_new)^2 at their samples
        if not len(points):
            return float(current)
        new_field = size_point_function(quantised_size_points(points, sizing.h_min), sizing.growth, sizing.h_max)
        new_samples = np.maximum(np.minimum(old_samples, new_field(samples).reshape(-1, 4)), sizing.h_min)
        return float(np.sum(np.mean((old_samples / new_samples) ** 2, axis=1)))

    def threshold(wanted):   # eta_target with plain_count ~ wanted (geometric bisection; count falls as it grows)
        high = float(eta.max())
        low = high
        for _ in range(MAX_BISECTIONS * 4):
            if plain_count(low) >= wanted or low < 1e-300:
                break
            low *= 0.25
        for _ in range(MAX_BISECTIONS * 2):
            middle = math.sqrt(low * high)
            middle_count = plain_count(middle)
            if abs(middle_count / wanted - 1.0) <= COUNT_RTOL or high / low < 1.0 + 1e-12:
                return middle
            if middle_count >= wanted:
                low = middle
            else:
                high = middle
        return low

    correction = 1.0
    for _ in range(GRADING_ROUNDS):
        eta_target = threshold(max(target / correction, (1.0 + COUNT_RTOL) * current))
        points, n_marked = size_points(eta_target)
        graded = graded_count(points)
        if not len(points) or abs(graded / target - 1.0) <= COUNT_RTOL:
            break
        correction = max(graded / max(plain_count(eta_target), 1.0), 1e-3)
    mesh_spec = replace(frozen, size_points=list(frozen.size_points) + points.tolist())
    candidate_spec = replace(spec, meshes=[mesh_spec if other.name == frozen.name else other for other in spec.meshes])
    try:
        count_estimate = estimate_mesh(candidate_spec, parameters, registry, frozen.name)
    except MESH_ESTIMATE_ERRORS as exc:
        raise ValueError(f"mesh '{frozen.name}': the adapted triangle sizes give an invalid mesh: {exc}") from exc
    return mesh_spec, count_estimate, n_marked


def snap_refinements(spec, shapes, mesh_spec) -> list:
    """RefinementSpecs at SNAP_DX_FRACTION x dx (margin 0) of every domain object with flat x / y faces that the mesh
    does not refine yet, so its faces stay cell edges once bands grade the grid."""
    refined = {refinement.object for refinement in mesh_spec.refinements}
    snaps = []
    for obj in spec.geometry:
        shape = shapes.get(obj.name)
        if shape is None or obj.name in refined or GEOMETRY_KINDS[shape.kind].entity_kind != 'domain':
            continue
        faces = axis_parallel_faces(obj.name, shapes)
        if faces.get('x') or faces.get('y'):
            snaps.append(RefinementSpec(obj.name, float(mesh_spec.dx) * SNAP_DX_FRACTION, 0.0))
    return snaps


def axis_bands(indices, widths, edges, base_dx, margin=BAND_MARGIN_CELLS) -> list:
    """[(lo, hi, dx)] bands of marked window cell indices along one axis: dx = base_dx / 2^k nearest half the cell
    width (k >= 1; wider cells, e.g. outer grading, are skipped), runs of each size widened by margin cells and
    merged; lo / hi are window cell edges [m]."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return []
    widths = np.asarray(widths, dtype=np.float64)
    levels = np.rint(np.log2(base_dx / (0.5 * widths[indices])))
    bands = []
    for level in np.unique(levels):
        if level < 1:
            continue
        chosen = np.sort(indices[levels == level])
        starts = np.maximum(chosen - margin, 0)
        stops = np.minimum(chosen + margin, widths.size - 1)
        start, stop = int(starts[0]), int(stops[0])
        for next_start, next_stop in zip(starts[1:], stops[1:]):
            if next_start <= stop + 1:
                stop = max(stop, int(next_stop))
                continue
            bands.append((float(edges[start]), float(edges[stop + 1]), float(base_dx * 0.5 ** level)))
            start, stop = int(next_start), int(next_stop)
        bands.append((float(edges[start]), float(edges[stop + 1]), float(base_dx * 0.5 ** level)))
    return bands


def with_bands(mesh_spec, bands):
    """Copy of mesh_spec with the {axis: [(lo, hi, dx)]} bands added to its MeshAxisSpecs (a band inside an existing
    one at least as fine is left out)."""
    axes = [replace(axis_spec, bands=list(axis_spec.bands)) for axis_spec in mesh_spec.axes]
    by_axis = {axis_spec.axis: axis_spec for axis_spec in axes}
    for axis, axis_bands_list in bands.items():
        if not axis_bands_list:
            continue
        entry = by_axis.get(axis)
        if entry is None:
            entry = by_axis[axis] = MeshAxisSpec(axis)
            axes.append(entry)
        existing = [(float(band.lo), float(band.hi), float(band.dx)) for band in entry.bands]
        entry.bands += [MeshBandSpec(lo, hi, dx) for lo, hi, dx in axis_bands_list
                        if not any(old_lo <= lo and hi <= old_hi and old_dx <= dx for old_lo, old_hi, old_dx in existing)]
    return replace(mesh_spec, axes=axes)


def window_cells(estimate) -> int:
    """Cells of an estimate inside the physical window (padding excluded); triangles: the elements."""
    if estimate.kind != 'tensor':
        return int(estimate.n_cells)
    return int(math.prod(estimate.window_axis_cells))


def window_widths(estimate) -> tuple:
    """(dx window widths, dy window widths) float64 of a 2D tensor estimate."""
    windows = _window_axes(estimate)
    return windows['x'][0], windows['y'][0]


def _window_axes(estimate):
    """{'x' / 'y': (window widths, window edges [m])} of a tensor estimate (edges in its physical / logical frame)."""
    sides = estimate.npml_sides
    windows = {}
    for axis, widths, edges in (('x', estimate.dx_cells, estimate.x_edges), ('y', estimate.dy_cells, estimate.y_edges)):
        lo, hi = int(sides.get(f'{axis}_lo', 0)), int(sides.get(f'{axis}_hi', 0))
        windows[axis] = (np.asarray(widths[lo:len(widths) - hi], dtype=np.float64),
                         np.asarray(edges[lo:len(edges) - hi], dtype=np.float64))
    return windows


def _refinement_spec(refinement):
    """RefinementSpec of a resolved meshing refinement (numbers)."""
    return RefinementSpec(refinement.object_name, float(refinement.dx), refinement.margin,
                          dx_x=refinement.axis_dx.get('x'), dx_y=refinement.axis_dx.get('y'),
                          dx_z=refinement.axis_dx.get('z'), where=refinement.where)


def _resampled_edges(edges, cell_factor):
    """Custom axis edges with round(cells x cell_factor) cells (>= 1) following the same width distribution
    (piecewise-linear in the cell index); [] stays []."""
    if not edges:
        return []
    edges = np.asarray(edges, dtype=np.float64)
    n_cells = edges.size - 1
    n_new = max(1, int(round(n_cells * cell_factor)))
    return [float(edge) for edge in np.interp(np.linspace(0.0, n_cells, n_new + 1), np.arange(n_cells + 1), edges)]
