"""Point-membership tests of the drawn geometry for solver_defaults.conformal_boundaries (solver_plan S7.6): host numpy
callables handed to the honouring solvers by the builder (2D inside(x, y), 3D inside(x, y, z)), plus their CPU memory
estimate."""
import math

import numpy as np

from ..core.memory_estimate import FLOAT32_BYTES, FLOAT64_BYTES, MemoryEstimate
from .geometry import (GEOMETRY_KINDS, contains_points_3d, domain_object_boxes, material_paint_order, rasterize,
                       z_extent)
from .schema import conformal_boundaries_physics

_INT32_BYTES = 4
_INT64_BYTES = 8
_EZ_FACE_BYTES = 2 * _INT32_BYTES + FLOAT32_BYTES      # cell, axis, extra
_HZ_CELL_BYTES = _INT32_BYTES + 4 * FLOAT32_BYTES      # cell, four face weights
_H3D_FACE_BYTES = _INT64_BYTES + _INT32_BYTES + 4 * FLOAT32_BYTES   # em_fdtd_3d: flat cell, H component, four weights
_H3D_FACES_PER_SURFACE_CELL = 3.0                      # cut H faces per surface cell of a pec_domain box (rough bound)
# em_fdtd_3d stability check (conformal_pec3d): ~3 x 5 live band E edges per surface cell, ~300 host bytes each (20
# ARPACK float64 vectors, D / G CSR, numbering); + 4 bool box masks per grid cell (upper bound)
_H3D_BAND_BYTES_PER_SURFACE_CELL = 3 * 5 * 300
_H3D_BAND_MASK_BYTES = 4
_WALL_LINK_ROW_BYTES = 8 * FLOAT32_BYTES               # laminar_flow: eight link fractions per wall cell
_HELD_WALL_BYTES = 2 * FLOAT64_BYTES                   # heat / potential: t_x, t_y per cell (system dtype <= float64)
# apertures a_x, a_y, the volume-scaled cell area and the bool conducting pattern the walls were located on, per cell
_INSULATED_WALL_BYTES = 3 * FLOAT64_BYTES + 1
_SURFACE_LOSS_BYTES = 2 * FLOAT64_BYTES                # heat cut-cell convection: surface_diag, surface_rhs per cell
_OUTLINE_SOURCE_BYTES = FLOAT32_BYTES                  # the outline part of the applied source density per cell
_DUAL_CUT_BYTES = 3 * FLOAT64_BYTES                    # em_modes: link_x, link_y, dual-cell area (host) per grid cell
# em_fdfd_*: the conformal cell mask (bool) + conformal_pec.ConformalFDFD per grid cell (device; built on the host first)
_FDFD_CONFORMAL_BYTES = {'em_fdfd_ez': 1 + 2 * FLOAT64_BYTES, 'em_fdfd_hz': 2 + 3 * FLOAT64_BYTES}
_CONFORMAL_TYPES = {'em_fdtd_ez': 'pec', 'em_fdtd_hz': 'pec', 'laminar_flow': 'walls', 'heat': 'held',
                    'electric_currents': 'held', 'electrostatics': 'held', 'em_modes': 'dual',
                    'em_fdfd_ez': 'fdfd', 'em_fdfd_hz': 'fdfd', 'em_fdtd_3d': 'pec3d'}   # estimate kinds
_HELD_FEATURE_TYPES = frozenset({'temperature', 'fixed_temperature_domain', 'voltage', 'terminal'})
_OUTLINE_FLUX_TYPES = frozenset({'heat_flux', 'current_density', 'surface_charge'})


class GeometryLocator:
    """Resolved shapes of a 2D or 3D build and the materials' expanded selections, evaluated at arbitrary host points
    with the rules of the build masks (rectangles / prisms half-open, circles / spheres closed; 3D:
    geometry.contains_points_3d). Points
    are Domain coordinates: on a mapped 2D window (domain_map: polar / spherical) the logical (r - radius_min, angle -
    angle_min), mapped to the drawing plane before the shapes are tested."""

    def __init__(self, shapes, expanded_selections, dimensions=2, domain_map=None):
        self.shapes = shapes
        self.expanded_selections = expanded_selections
        self.dimensions = int(dimensions)
        self.domain_map = domain_map if domain_map is not None and domain_map.mapped else None

    def _domain_names(self, entities):
        return [name for name in entities if name in self.shapes
                and GEOMETRY_KINDS[self.shapes[name].kind].entity_kind == 'domain']

    def _points(self, coordinates):
        """float64 broadcast drawing-plane coordinate arrays; ValueError unless one per model dimension."""
        if len(coordinates) != self.dimensions:
            raise ValueError(f'a {self.dimensions}D geometry locator takes {self.dimensions} coordinates, got '
                             f'{len(coordinates)}')
        points = np.broadcast_arrays(*(np.asarray(value, dtype=np.float64) for value in coordinates))
        if self.domain_map is not None:
            points = np.broadcast_arrays(*self.domain_map.to_physical(*points))
        return points

    def _member(self, name, points):
        if self.dimensions == 3:
            return contains_points_3d(name, self.shapes, *points)
        return np.asarray(rasterize(name, self.shapes, *points, np), dtype=bool)

    def inside(self, entities):
        """inside(x, y) (3D: inside(x, y, z)) -> bool array of the union of the domain objects in `entities` ('all' =
        everywhere)."""
        whole = 'all' in entities
        names = self._domain_names(entities)

        def test(*coordinates):
            points = self._points(coordinates)
            result = np.full(points[0].shape, whole, dtype=bool)
            for name in names if not whole else ():
                result |= self._member(name, points)
            return result
        return test

    def material_index(self, *coordinates):
        """int16 1 + winning material index (0 = none) at the points (x, y) / (x, y, z), painted in
        material_paint_order."""
        points = self._points(coordinates)
        index = np.zeros(points[0].shape, dtype=np.int16)
        for material_index in material_paint_order(self.expanded_selections):
            entities = self.expanded_selections[material_index]
            if 'all' in entities:
                index[...] = material_index + 1
                continue
            for name in self._domain_names(entities):
                index[self._member(name, points)] = material_index + 1
        return index

    def material_test(self, material_index):
        """inside(*coordinates) -> bool where spec material `material_index` (0-based) wins the paint (its effective
        region, as the build's material masks)."""
        def test(*coordinates):
            return self.material_index(*coordinates) == material_index + 1
        return test


def geometry_locator(ctx) -> GeometryLocator:
    """GeometryLocator of a BuildContext (its resolved shapes and the spec materials' expanded selections; 3D for a
    model with size_z; Domain coordinates on a polar / spherical window)."""
    expanded = [ctx.selection(material.selection, 'domain').entities for material in ctx.spec.materials]
    is_3d = ctx.spec.is_3d
    return GeometryLocator(ctx.resolved_geometry, expanded, dimensions=3 if is_3d else 2,
                           domain_map=None if is_3d else ctx.domain_map)


# ---- CPU memory estimate (GUI-thread safe) ----
def _perimeter_cells(spec, shapes, entity_lists, finest) -> float:
    """Bounding-box perimeters of the domain objects in entity_lists over the finest cell (rough cut count)."""
    perimeter = sum(2.0 * (max(x1 - x0, 0.0) + max(y1 - y0, 0.0))
                    for _, (x0, y0, x1, y1) in domain_object_boxes(spec, shapes, entity_lists))
    return perimeter / finest


def _surface_cells(spec, shapes, entity_lists, finest, z_range) -> float:
    """Bounding-box surface areas of the 3D domain objects in entity_lists over the finest cell face (z extents
    clipped to the meshed z_range)."""
    area = 0.0
    for name, (x0, y0, x1, y1) in domain_object_boxes(spec, shapes, entity_lists):
        z_lo, z_hi = z_extent(name, shapes)
        z_lo, z_hi = max(z_lo, z_range[0]), min(z_hi, z_range[1])
        sides = (max(x1 - x0, 0.0), max(y1 - y0, 0.0), max(z_hi - z_lo, 0.0))
        if math.isfinite(sides[2]):
            area += 2.0 * (sides[0] * sides[1] + sides[1] * sides[2] + sides[0] * sides[2])
    return area / finest ** 2


def conformal_memory_estimate(spec, registry, shapes, mesh) -> MemoryEstimate:
    """Device arrays of every honouring physics: E_z faces / H_z cells along its pec_domain outlines (2 per perimeter
    cell), laminar_flow's int32 wall row per cell plus 8 fractions per wall cell, the two held-edge face arrays of a
    heat / potential physics with held features plus its insulated-wall arrays and perimeter-flux / convection arrays
    (upper bounds), em_modes' host dual-cell cuts and the em_fdfd_* full-grid weights with pec_domain, em_fdtd_3d's
    cut H faces (3 per bounding-box surface cell of its pec_domain objects) + its host stability check; empty when off."""
    honouring, _ = conformal_boundaries_physics(spec, registry)
    if not honouring:
        return MemoryEstimate()
    finest = min(float(np.diff(np.asarray(edges, dtype=np.float64)).min()) for edges in (mesh.x_edges, mesh.y_edges))
    n_cells = int(mesh.n_cells)
    items = []
    host_items = []
    for physics_spec in spec.physics:
        kind = _CONFORMAL_TYPES.get(physics_spec.type)
        if physics_spec.name not in honouring or kind is None:
            continue
        if kind == 'pec':
            selections = [feature.selection for feature in physics_spec.features
                          if feature.enabled and feature.type == 'pec_domain']
            count = min(2.0 * _perimeter_cells(spec, shapes, selections, finest), 2.0 * n_cells)
            per_entry = _EZ_FACE_BYTES if physics_spec.type == 'em_fdtd_ez' else _HZ_CELL_BYTES
            items.append((f"conformal PEC '{physics_spec.name}' (approximate)", int(per_entry * count)))
        elif kind == 'held':
            if any(feature.enabled and feature.type in _HELD_FEATURE_TYPES for feature in physics_spec.features):
                items.append((f"held edges '{physics_spec.name}'", _HELD_WALL_BYTES * n_cells))
            # insulated walls (allocated only when a free cell borders a closed one): two apertures + the scaled area
            items.append((f"insulated-wall cut cells '{physics_spec.name}' (upper bound)",
                          _INSULATED_WALL_BYTES * n_cells))
            enabled_types = {feature.type for feature in physics_spec.features if feature.enabled}
            if enabled_types & _OUTLINE_FLUX_TYPES:
                items.append((f"outline flux densities '{physics_spec.name}' (upper bound)",
                              _OUTLINE_SOURCE_BYTES * n_cells))
            if 'convection' in enabled_types:
                items.append((f"outline convection '{physics_spec.name}' (upper bound)",
                              _SURFACE_LOSS_BYTES * n_cells))
        elif kind == 'dual':
            if any(feature.enabled and feature.type == 'pec_domain' for feature in physics_spec.features):
                host_items.append((f"conformal PEC dual-cell cuts '{physics_spec.name}'", _DUAL_CUT_BYTES * n_cells))
        elif kind == 'fdfd':
            if any(feature.enabled and feature.type == 'pec_domain' for feature in physics_spec.features):
                size = _FDFD_CONFORMAL_BYTES[physics_spec.type] * n_cells
                items.append((f"conformal PEC weights '{physics_spec.name}'", size))
                host_items.append((f"conformal PEC weights '{physics_spec.name}' (host build)", size))
        elif kind == 'pec3d':
            selections = [feature.selection for feature in physics_spec.features
                          if feature.enabled and feature.type == 'pec_domain']
            z_edges = np.asarray(mesh.z_edges, dtype=np.float64)
            finest_3d = min(finest, float(np.diff(z_edges).min()))
            surface = _surface_cells(spec, shapes, selections, finest_3d, (float(z_edges[0]), float(z_edges[-1])))
            count = min(_H3D_FACES_PER_SURFACE_CELL * surface, 3.0 * n_cells)
            items.append((f"conformal PEC H faces '{physics_spec.name}' (approximate)", int(_H3D_FACE_BYTES * count)))
            host_items.append((f"conformal PEC stability check '{physics_spec.name}' (approximate)",
                               int(_H3D_BAND_BYTES_PER_SURFACE_CELL * min(surface, n_cells)
                                   + _H3D_BAND_MASK_BYTES * n_cells)))
        else:
            selections = [material.selection for material in spec.materials]
            count = min(_perimeter_cells(spec, shapes, selections, finest), float(n_cells))
            items.append((f"wall links '{physics_spec.name}' (approximate)",
                          _INT32_BYTES * n_cells + int(_WALL_LINK_ROW_BYTES * count)))
    return MemoryEstimate.from_items(tuple(items), host_items=tuple(host_items)).labelled('conformal boundaries')
