"""Field sampling on triangle meshes (mesh plan 2.9): point location (bucket grid over element boxes, barycentric
test, Newton on the P2 map of curved elements), evaluation of Lagrange / Nedelec fields at points, the display
raster (pixel centres -> values, NaN outside; cached per mesh and raster), quadrature sample sets for derived values
and the display triangulation (P2 split 1 -> 4). Host numpy."""
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .elements import (curved_elements, element_geometry, lagrange_basis, lagrange_values, nedelec_basis,
                       physical_gradients, piola_vectors)
from .quadrature import triangle_rule

POINT_CHUNK = 65536           # points located per vectorised block
INSIDE_TOLERANCE = 1e-10      # reference-coordinate slack of the inside test (points on shared edges)
_NEWTON_STEPS = 12
_NEWTON_STEP_TOLERANCE = 1e-12   # reference-coordinate step below which a point's Newton stops (quadratic: done)
CURVED_BOX_SLACK = 0.25       # x the node box's longer side: a curved candidate outside its widened box skips Newton
_RASTER_CACHE_SIZE = 4


@dataclass(frozen=True, eq=False)
class Location:
    """Containing element (mesh index, -1 outside) and reference coordinates (xi, eta) of points."""
    element: np.ndarray       # int64 (n,)
    reference: np.ndarray     # float64 (n, 2)

    @property
    def found(self) -> np.ndarray:
        return self.element >= 0


class LocatedPoints:
    """A located point set (read-only Location) with its Lagrange basis values per order, computed on first use and
    read-only, so reads of several fields on one point set share them (TriangleDomain.located keeps these)."""

    def __init__(self, location: Location):
        location.element.flags.writeable = False
        location.reference.flags.writeable = False
        self.location = location
        self._values = {}      # order -> read-only (n, 1, k) basis values of every point
        self._gradients = {}   # order -> read-only (n, 1, k, 2) reference gradients of every point

    def __len__(self) -> int:
        return len(self.location.element)

    def lagrange(self, order, gradient=False) -> tuple:
        """(values (n, 1, k), reference gradients (n, 1, k, 2) or None) of the order-`order` Lagrange basis at every
        point (points outside: values at reference (0, 0)); read-only."""
        order = int(order)
        values = self._values.get(order)
        if values is None:
            values = lagrange_values(order, self.location.reference[:, None, :])
            values.flags.writeable = False
            self._values[order] = values
        gradients = self._gradients.get(order) if gradient else None
        if gradient and gradients is None:
            gradients = lagrange_basis(order, self.location.reference[:, None, :])[1]
            gradients.flags.writeable = False
            self._gradients[order] = gradients
        return values, gradients


class PointLocator:
    """Point location on a TriangleMesh: buckets of about one element each over the element bounding boxes (curved
    elements' boxes include their midside nodes when geometry_order is 2)."""

    def __init__(self, mesh, geometry_order=2):
        self.mesh = mesh
        self.geometry_order = int(geometry_order)
        corners = mesh.points[mesh.triangles]
        low, high = corners.min(axis=1), corners.max(axis=1)
        self.curved = (curved_elements(mesh) if self.geometry_order == 2
                       else np.zeros(mesh.n_elements, dtype=bool))
        if self.curved.any():
            middles = mesh.midside_points[mesh.element_edges[self.curved]]
            low[self.curved] = np.minimum(low[self.curved], middles.min(axis=1))
            high[self.curved] = np.maximum(high[self.curved], middles.max(axis=1))
        slack = CURVED_BOX_SLACK * (high - low).max(axis=1, keepdims=True)
        self._near_low, self._near_high = low - slack, high + slack   # widened node boxes (curved candidates)
        self.low = low.min(axis=0)
        span = np.maximum(high.max(axis=0) - self.low, 1e-300)
        cell = float(np.sqrt(span[0] * span[1] / max(mesh.n_elements, 1)))
        self.shape = np.maximum(np.ceil(span / cell).astype(np.int64), 1)
        self.cell = span / self.shape
        first = self._bucket_index(low)
        last = self._bucket_index(high)
        counts = (last[:, 0] - first[:, 0] + 1) * (last[:, 1] - first[:, 1] + 1)
        owner = np.repeat(np.arange(mesh.n_elements, dtype=np.int64), counts)
        offset = np.arange(len(owner)) - np.repeat(np.cumsum(counts) - counts, counts)
        width = (last[:, 0] - first[:, 0] + 1)[owner]
        bucket_x = first[owner, 0] + offset % width
        bucket_y = first[owner, 1] + offset // width
        bucket = bucket_x * self.shape[1] + bucket_y
        order = np.argsort(bucket * mesh.n_elements + owner)   # unique keys: the stable bucket order, by quicksort
        self.bucket_elements = owner[order]
        self.bucket_start = np.searchsorted(bucket[order], np.arange(self.shape[0] * self.shape[1] + 1))
        origin = corners[:, 0, :]
        axes = np.stack([corners[:, 1, :] - origin, corners[:, 2, :] - origin], axis=2)   # columns x1 - x0, x2 - x0
        determinant = (axes[:, 0, 0] * axes[:, 1, 1] - axes[:, 0, 1] * axes[:, 1, 0])[:, None, None]
        rows = (np.stack([axes[:, 1, 1], -axes[:, 0, 1]], axis=1), np.stack([-axes[:, 1, 0], axes[:, 0, 0]], axis=1))
        self.origin = origin
        self.inverse = np.stack(rows, axis=1) / determinant   # affine (x - x0) -> (xi, eta)

    def _bucket_index(self, points):
        index = np.floor((points - self.low) / self.cell).astype(np.int64)
        return np.clip(index, 0, self.shape - 1)

    def locate(self, points) -> Location:
        """Location of (n, 2) points [m]; among several containing elements the one the point lies deepest in."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        element = np.full(len(points), -1, dtype=np.int64)
        reference = np.zeros((len(points), 2))
        for start in range(0, len(points), POINT_CHUNK):
            chunk = slice(start, start + POINT_CHUNK)
            element[chunk], reference[chunk] = self._locate_chunk(points[chunk])
        return Location(element, reference)

    def _locate_chunk(self, points):
        count = len(points)
        element = np.full(count, -1, dtype=np.int64)
        reference = np.zeros((count, 2))
        index = np.floor((points - self.low) / self.cell).astype(np.int64)
        valid = ((index >= 0) & (index <= self.shape - 1)).all(axis=1)
        # points on the far box edge belong to the last bucket
        on_far_edge = ~valid & (index >= 0).all(axis=1) & (points <= self.low + self.cell * self.shape).all(axis=1)
        valid |= on_far_edge
        index = np.clip(index, 0, self.shape - 1)
        bucket = index[:, 0] * self.shape[1] + index[:, 1]
        starts = np.where(valid, self.bucket_start[bucket], 0)
        counts = np.where(valid, self.bucket_start[bucket + 1] - starts, 0)
        point_of = np.repeat(np.arange(count), counts)
        if not len(point_of):
            return element, reference
        offsets = np.arange(len(point_of)) - np.repeat(np.cumsum(counts) - counts, counts)
        candidate = self.bucket_elements[np.repeat(starts, counts) + offsets]
        local = np.einsum('kcd,kd->kc', self.inverse[candidate], points[point_of] - self.origin[candidate])
        curved = np.flatnonzero(self.curved[candidate])
        if curved.size:   # Newton only where the point lies near the element (a bucket shares many far candidates)
            elements, located = candidate[curved], points[point_of[curved]]
            near = ((located >= self._near_low[elements]) & (located <= self._near_high[elements])).all(axis=1)
            local[curved[~near]] = np.inf
            curved = curved[near]
            if curved.size:
                local[curved] = self._newton(candidate[curved], points[point_of[curved]], local[curved])
        depth = np.minimum(np.minimum(local[:, 0], local[:, 1]), 1.0 - local[:, 0] - local[:, 1])
        depth = np.where(np.isfinite(depth), depth, -np.inf)
        # per point (candidates are grouped by point) the first candidate of largest depth: segmented argmax
        group_counts = counts[counts > 0]
        group_starts = np.cumsum(group_counts) - group_counts
        deepest = np.repeat(np.maximum.reduceat(depth, group_starts), group_counts)
        first = np.minimum.reduceat(np.where(depth == deepest, np.arange(len(depth)), len(depth)), group_starts)
        hit = depth[first] >= -INSIDE_TOLERANCE
        element[point_of[first[hit]]] = candidate[first[hit]]
        reference[point_of[first[hit]]] = local[first[hit]]
        return element, reference

    def _newton(self, elements, points, start):
        """Reference coordinates of points under the P2 map of curved elements (Newton from the affine guess; a
        point the map cannot reach ends far outside the reference triangle)."""
        local = start.copy()
        running = np.arange(len(local))   # points whose last step exceeded _NEWTON_STEP_TOLERANCE
        with np.errstate(divide='ignore', invalid='ignore'):
            for _ in range(_NEWTON_STEPS):
                geometry = element_geometry(self.mesh, elements[running], local[running, None, :], 2, check=False)
                residual = points[running] - geometry.points[:, 0, :]
                step = np.einsum('kcd,kd->kc', geometry.inverse[:, 0], residual)
                local[running] = np.clip(local[running] + step, -1.0, 2.0)
                running = running[np.abs(step).max(axis=1) > _NEWTON_STEP_TOLERANCE]   # NaN steps stop (missed below)
                if not running.size:
                    break
            geometry = element_geometry(self.mesh, elements, local[:, None, :], 2, check=False)
            size = np.sqrt(np.abs(geometry.determinant[:, 0]))
            missed = ~(np.linalg.norm(points - geometry.points[:, 0, :], axis=1) <= 1e-9 * size)
        local[missed] = np.inf
        return local


def locator(mesh, geometry_order=2) -> PointLocator:
    """Cached PointLocator of a mesh (kept on the mesh object's identity for the mesh's lifetime)."""
    key = (id(mesh), int(geometry_order))
    cached = _LOCATORS.get(key)
    if cached is None or cached.mesh is not mesh:
        cached = PointLocator(mesh, geometry_order)
        _LOCATORS[key] = cached
        while len(_LOCATORS) > _RASTER_CACHE_SIZE:
            _LOCATORS.popitem(last=False)
    else:
        _LOCATORS.move_to_end(key)
    return cached


_LOCATORS = OrderedDict()


# ---- evaluation ----
def evaluate(dofs, coefficients, points=None, *, location=None, gradient=False):
    """Values of a field (coefficients per dof of `dofs`) at (n, 2) points, a Location or LocatedPoints (its Lagrange
    basis reused): Lagrange (n,) [+ gradients (n, 2) when gradient], Nedelec vectors (n, 2) [+ curls (n,)]; NaN outside
    the dof map's elements."""
    coefficients = np.asarray(coefficients)
    located = location if isinstance(location, LocatedPoints) else None
    if located is not None:
        location = located.location
    elif location is None:
        location = locator(dofs.mesh, dofs.geometry_order).locate(points)
    position = np.where(location.found, dofs.element_position[np.maximum(location.element, 0)], -1)
    inside = position >= 0
    lagrange = dofs.family == 'lagrange'
    count = len(position)
    dtype = np.result_type(coefficients.dtype, np.float64)
    values = np.full((count,) if lagrange else (count, 2), np.nan, dtype=dtype)
    derivative = np.full((count, 2) if lagrange else (count,), np.nan, dtype=dtype)
    if inside.any():
        elements = dofs.elements[position[inside]]
        reference = location.reference[inside][:, None, :]   # (k, 1, 2)
        local = coefficients[dofs.element_dofs[position[inside]]]   # (k, n)
        need_geometry = gradient or not lagrange
        geometry = element_geometry(dofs.mesh, elements, reference, dofs.geometry_order) if need_geometry else None
        if lagrange:
            if located is None:
                if gradient:
                    basis, reference_gradients = lagrange_basis(dofs.order, reference)
                else:
                    basis, reference_gradients = lagrange_values(dofs.order, reference), None
            else:
                basis, reference_gradients = located.lagrange(dofs.order, gradient)
                if not inside.all():
                    basis = basis[inside]
                    reference_gradients = None if reference_gradients is None else reference_gradients[inside]
            values[inside] = np.einsum('kqn,kn->k', basis, local)
            if gradient:
                gradients = physical_gradients(reference_gradients, geometry.inverse)
                derivative[inside] = np.einsum('kqnc,kn->kc', gradients, local)
        else:
            signs = dofs.mesh.edge_signs[elements][:, None, :]
            vectors, curls = nedelec_basis(dofs.order, reference, signs)
            values[inside] = np.einsum('kqnc,kn->kc', piola_vectors(vectors, geometry.inverse), local)
            if gradient:
                derivative[inside] = np.einsum('kqn,kn->k', curls / geometry.determinant[..., None], local)
    return (values, derivative) if gradient else values


# ---- display raster ----
class RasterSampler:
    """Located pixel centres of a raster (x_centres (nx,), y_centres (ny,)) on a mesh: sample() fills (nx, ny)
    images with NaN outside the mesh or the field's elements."""

    def __init__(self, mesh, x_centres, y_centres, geometry_order=2):
        self.mesh = mesh
        self.shape = (len(x_centres), len(y_centres))
        grid_x, grid_y = np.meshgrid(np.asarray(x_centres, dtype=np.float64), np.asarray(y_centres, dtype=np.float64),
                                     indexing='ij')
        self.location = locator(mesh, geometry_order).locate(np.column_stack([grid_x.ravel(), grid_y.ravel()]))

    def sample(self, dofs, coefficients) -> np.ndarray:
        values = evaluate(dofs, coefficients, location=self.location)
        if values.ndim == 2:
            raise ValueError('RasterSampler.sample: scalar fields only (sample Nedelec components separately)')
        return values.reshape(self.shape)


def raster_sampler(mesh, x_centres, y_centres, geometry_order=2) -> RasterSampler:
    """Cached RasterSampler per (mesh digest, raster centres, geometry order)."""
    x_centres = np.ascontiguousarray(x_centres, dtype=np.float64)
    y_centres = np.ascontiguousarray(y_centres, dtype=np.float64)
    key = (mesh.digest, x_centres.tobytes(), y_centres.tobytes(), int(geometry_order))
    cached = _RASTERS.get(key)
    if cached is None:
        cached = RasterSampler(mesh, x_centres, y_centres, geometry_order)
        _RASTERS[key] = cached
        while len(_RASTERS) > _RASTER_CACHE_SIZE:
            _RASTERS.popitem(last=False)
    else:
        _RASTERS.move_to_end(key)
    return cached


_RASTERS = OrderedDict()


# ---- quadrature samples and display triangulation ----
@dataclass(frozen=True, eq=False)
class QuadratureSamples:
    """Flattened quadrature points of elements: integrals = sum(values * weight)."""
    x: np.ndarray
    y: np.ndarray
    weight: np.ndarray        # w_q det J [m^2]
    element: np.ndarray       # int64 mesh element of each sample
    reference: np.ndarray     # (n, 2) reference coordinates (evaluate(location=...))

    def location(self) -> Location:
        return Location(self.element, self.reference)


def quadrature_samples(mesh, degree=4, elements=None, geometry_order=2) -> QuadratureSamples:
    """Degree-`degree` quadrature samples of mesh elements (None = all) in the given geometry order."""
    elements = np.arange(mesh.n_elements) if elements is None else np.asarray(elements)
    elements = np.flatnonzero(elements) if elements.dtype == bool else elements.astype(np.int64)
    rule = triangle_rule(degree)
    geometry = element_geometry(mesh, elements, rule.points, geometry_order)
    weight = rule.weights[None, :] * geometry.determinant
    count = len(rule.weights)
    return QuadratureSamples(x=geometry.points[..., 0].ravel(), y=geometry.points[..., 1].ravel(),
                             weight=weight.ravel(), element=np.repeat(elements, count),
                             reference=np.tile(rule.points, (len(elements), 1)))


def display_triangulation(mesh, order=2) -> tuple:
    """(nodes (n, 2), triangles (k, 3) int32) for plots: order 1 the mesh itself; order 2 the nodes plus midside nodes
    (index n_points + edge = the full-mesh P2 dof numbering), each element split into four."""
    if order == 1:
        return mesh.points, mesh.triangles
    nodes = np.concatenate([mesh.points, mesh.midside_points])
    corner = mesh.triangles.astype(np.int64)
    middle = mesh.n_points + mesh.element_edges.astype(np.int64)   # middle[:, k] = midside of edge k (opposite k)
    triangles = np.concatenate([
        np.column_stack([corner[:, 0], middle[:, 2], middle[:, 1]]),
        np.column_stack([middle[:, 2], corner[:, 1], middle[:, 0]]),
        np.column_stack([middle[:, 1], middle[:, 0], corner[:, 2]]),
        np.column_stack([middle[:, 0], middle[:, 1], middle[:, 2]])]).astype(np.int32)
    return nodes, triangles


def node_values(dofs, coefficients) -> np.ndarray:
    """Values of a P1 / P2 Lagrange field at the display_triangulation(mesh, dofs.order) nodes without point location
    (their numbering = the Lagrange global ids); NaN at nodes outside the dof map's elements."""
    if dofs.family != 'lagrange' or dofs.order not in (1, 2):
        raise ValueError('node_values: P1 / P2 Lagrange maps only')
    mesh = dofs.mesh
    coefficients = np.asarray(coefficients)
    count = mesh.n_points + (mesh.n_edges if dofs.order == 2 else 0)
    values = np.full(count, np.nan, dtype=np.result_type(coefficients.dtype, np.float64))
    values[dofs.global_ids] = coefficients
    return values
