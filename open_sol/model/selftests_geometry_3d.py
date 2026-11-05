"""Self-test of the 3D prism extensions (igor 1.1: extrusion axis x / y, sidewall angle): rasterised volumes against
closed forms, an x / y extrusion against the equivalent slab stack, sections, mesh snap faces and the validation
errors. CPU only (specs() is empty, check({}) calls the geometry / meshing / validate functions directly)."""
import math

import numpy as np

from .schema import SelfTestCase
from .selftest_helpers import SubChecks
from .spec import DomainSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, RefinementSpec

_CELL = 25e-9                        # m, uniform (was 20 nm; not yet confirmed by a run)
_SIZE = (2e-6, 2e-6, 1e-6)           # m; 80 x 80 x 40 cells (change with _CELL)
_CELLS = (80, 80, 40)
_VOLUME_RTOL = 0.02                  # staircase of shapes >= 12 cells across (~0.5 % at 20 nm)
_ROD = (1.0e-6, 0.5e-6, 0.3e-6, 0.2e-6, 1.8e-6)              # circle along y: x, z of the axis, r, y ends [m]
_CONE = (1.0e-6, 1.0e-6, 0.4e-6, 0.2e-6, 0.8e-6, 0.2e-6)     # cx, cy, base r, z0, z1, top r [m]
_TRIANGLE = (1.0e-6, 1.0e-6, 1.2e-6, 0.1e-6, 0.7e-6, 0.15e-6)   # centroid x, y, side, z0, z1, top inward offset [m]
_BAR = (0.6e-6, 0.3e-6, 1.4e-6, 0.7e-6, 0.25e-6, 1.75e-6)    # rectangle along x: y0, z0, y1, z1, x ends [m]
_WEDGE = ((0.3e-6, 0.1e-6), (1.7e-6, 0.1e-6), (1.1e-6, 0.85e-6), (0.9e-6, 0.4e-6), (0.7e-6, 0.6e-6))   # (x, z) [m]
_WEDGE_Y = (0.4e-6, 1.6e-6)          # y ends of the wedge [m]
_CONE_X = (1.0e-6, 0.5e-6, 0.35e-6, 0.2e-6, 1.8e-6, 0.15e-6)   # circle along x: axis y, z, r, x ends, end r [m]
_TAPER_Y = (0.4e-6, 0.2e-6, 1.6e-6, 0.8e-6, 0.3e-6, 1.7e-6, 0.1e-6)   # along y: x0, z0, x1, z1, y ends, offset [m]
_SECTION_RTOL = 1e-9                 # relative tolerance of analytic section sizes
_STACK_MISMATCH = 1e-3               # wedge vs slab stack: differing cells / wedge cells (boundary ties only)
_EXACT = 1e-15                       # m: absolute tolerance of closed-form coordinates (float64 rounding)
_MESH_BASE, _MESH_FINE, _MESH_RATIO = 50e-9, 10e-9, 1.2


def _triangle_points():
    """Clockwise (tests the orientation) equilateral triangle of _TRIANGLE: centroid, side."""
    cx, cy, side = _TRIANGLE[:3]
    inradius = side / (2.0 * math.sqrt(3.0))
    return [(cx, cy + 2.0 * inradius), (cx + 0.5 * side, cy - inradius), (cx - 0.5 * side, cy - inradius)]


def geometry_3d_spec():
    """3D spec without physics: rod along y, cone, tapered triangle, bar along x with its equivalent box, wedge along
    y, cone along x, tapered rectangle along y; one material on the cone (vertical-cut preview)."""
    rod_x, rod_z, rod_r, rod_y0, rod_y1 = _ROD
    cone_x, cone_y, cone_r, cone_z0, cone_z1, cone_top_r = _CONE
    _, _, _, tri_z0, tri_z1, tri_offset = _TRIANGLE
    bar_y0, bar_z0, bar_y1, bar_z1, bar_x0, bar_x1 = _BAR
    cone_x_y, cone_x_z, cone_x_r, cone_x_x0, cone_x_x1, cone_x_top_r = _CONE_X
    taper_x0, taper_z0, taper_x1, taper_z1, taper_y0, taper_y1, taper_offset = _TAPER_Y
    spec = ModelSpec(title='Self-test: 3D prism extensions', domain=DomainSpec(size_x=_SIZE[0], size_y=_SIZE[1],
                                                                               size_z=_SIZE[2]))
    spec.mesh = MeshSpec(mode='uniform', dx=_CELL)
    spec.geometry = [
        GeometrySpec('rod', 'circle', {'cx': rod_x, 'cy': rod_z, 'r': rod_r, 'z0': rod_y0, 'z1': rod_y1,
                                       'axis': 'y'}),
        GeometrySpec('cone', 'circle', {'cx': cone_x, 'cy': cone_y, 'r': cone_r, 'z0': cone_z0, 'z1': cone_z1,
                                        'sidewall_angle': math.degrees(math.atan((cone_r - cone_top_r)
                                                                                 / (cone_z1 - cone_z0)))}),
        GeometrySpec('triangle', 'polygon', {'points': _triangle_points(), 'z0': tri_z0, 'z1': tri_z1,
                                             'sidewall_angle': math.degrees(math.atan(tri_offset
                                                                                      / (tri_z1 - tri_z0)))}),
        GeometrySpec('bar', 'rectangle', {'x0': bar_y0, 'y0': bar_z0, 'x1': bar_y1, 'y1': bar_z1, 'z0': bar_x0,
                                          'z1': bar_x1, 'axis': 'x'}),
        GeometrySpec('bar_box', 'box', {'x0': bar_x0, 'y0': bar_y0, 'z0': bar_z0, 'x1': bar_x1, 'y1': bar_y1,
                                        'z1': bar_z1}),
        GeometrySpec('wedge', 'polygon', {'points': list(_WEDGE), 'z0': _WEDGE_Y[0], 'z1': _WEDGE_Y[1], 'axis': 'y'}),
        GeometrySpec('cone_x', 'circle', {'cx': cone_x_y, 'cy': cone_x_z, 'r': cone_x_r, 'z0': cone_x_x0,
                                          'z1': cone_x_x1, 'axis': 'x',
                                          'sidewall_angle': math.degrees(math.atan((cone_x_r - cone_x_top_r)
                                                                                   / (cone_x_x1 - cone_x_x0)))}),
        GeometrySpec('taper_y', 'rectangle', {'x0': taper_x0, 'y0': taper_z0, 'x1': taper_x1, 'y1': taper_z1,
                                              'z0': taper_y0, 'z1': taper_y1, 'axis': 'y',
                                              'sidewall_angle': math.degrees(math.atan(taper_offset
                                                                                       / (taper_y1 - taper_y0)))}),
    ]
    spec.materials = [MaterialSpec('metal', properties={'k': 3.0}, selection=['cone'])]
    return spec


def _full_mask(window, mask):
    """Full (nx, ny, nz) bool grid of a rasterize_box result."""
    full = np.zeros(_CELLS, dtype=bool)
    full[window] = mask
    return full


def _volume_checks(shapes, centres, add):
    """Rod, cone, tapered-triangle, x-cone and y-tapered rectangle staircase volumes against their closed forms; bar =
    box; wedge = slab stack."""
    from .geometry import ResolvedShape, rasterize, rasterize_box

    def volume(name):
        return float(rasterize_box(name, shapes, *centres, np)[1].sum()) * _CELL ** 3

    rod_r, rod_length = _ROD[2], _ROD[4] - _ROD[3]
    rod_ratio = volume('rod') / (math.pi * rod_r ** 2 * rod_length)
    add('rod_y_volume', abs(rod_ratio - 1.0) < _VOLUME_RTOL, rod_ratio)
    _, _, base_r, z0, z1, top_r = _CONE
    cone_ratio = volume('cone') / (math.pi * (z1 - z0) / 3.0 * (base_r ** 2 + base_r * top_r + top_r ** 2))
    add('cone_frustum_volume', abs(cone_ratio - 1.0) < _VOLUME_RTOL, cone_ratio)
    _, _, side, tri_z0, tri_z1, tri_offset = _TRIANGLE
    inradius = side / (2.0 * math.sqrt(3.0))
    scale = (inradius - tri_offset) / inradius   # mitred offset of a triangle = similar triangle
    base_area = math.sqrt(3.0) / 4.0 * side ** 2
    triangle_ratio = volume('triangle') / ((tri_z1 - tri_z0) / 3.0 * base_area * (1.0 + scale + scale ** 2))
    add('tapered_triangle_volume', abs(triangle_ratio - 1.0) < _VOLUME_RTOL, triangle_ratio)
    _, _, cone_x_r, cone_x_x0, cone_x_x1, cone_x_top_r = _CONE_X
    cone_x_ratio = volume('cone_x') / (math.pi * (cone_x_x1 - cone_x_x0) / 3.0
                                       * (cone_x_r ** 2 + cone_x_r * cone_x_top_r + cone_x_top_r ** 2))
    add('cone_x_frustum_volume', abs(cone_x_ratio - 1.0) < _VOLUME_RTOL, cone_x_ratio)
    taper_x0, taper_z0, taper_x1, taper_z1, taper_y0, taper_y1, taper_offset = _TAPER_Y
    width, height, length = taper_x1 - taper_x0, taper_z1 - taper_z0, taper_y1 - taper_y0
    taper_ratio = volume('taper_y') / (length * (width * height - (width + height) * taper_offset
                                                 + 4.0 / 3.0 * taper_offset ** 2))   # sides shrink by 2 x offset
    add('tapered_rectangle_y_volume', abs(taper_ratio - 1.0) < _VOLUME_RTOL, taper_ratio)

    bar = _full_mask(*rasterize_box('bar', shapes, *centres, np))
    box = _full_mask(*rasterize_box('bar_box', shapes, *centres, np))
    add('bar_x_equals_box', bool(np.array_equal(bar, box)) and bool(box.any()),
        {'bar_cells': int(bar.sum()), 'box_cells': int(box.sum())})
    x_centres, y_centres, z_centres = centres
    plane = {'plane': ResolvedShape('plane', 'polygon', {'points': list(_WEDGE)})}
    X, Z = np.meshgrid(x_centres, z_centres, indexing='ij')
    in_y = (y_centres >= _WEDGE_Y[0]) & (y_centres < _WEDGE_Y[1])
    stack = rasterize('plane', plane, X, Z, np)[:, None, :] & in_y[None, :, None]
    wedge = _full_mask(*rasterize_box('wedge', shapes, *centres, np))
    mismatch = int(np.count_nonzero(wedge != stack))
    add('wedge_y_equals_slab_stack', mismatch <= _STACK_MISMATCH * max(int(stack.sum()), 1) and bool(stack.any()),
        {'mismatch': mismatch, 'stack_cells': int(stack.sum())})


def _section_checks(spec, shapes, centres, add):
    """Runs per z cell of a taper, x / y extrusion extents, vertical sections and the vertical-cut material preview."""
    from .geometry import bounding_box, cross_section_runs, material_preview, vertical_section, z_extent

    runs = cross_section_runs(['cone'], shapes, centres[2])
    cone_cells = round((_CONE[4] - _CONE[3]) / _CELL)
    add('taper_run_per_z_cell', len(runs) == cone_cells + 2, {'n_runs': len(runs), 'expected': cone_cells + 2})
    rod_x, rod_z, rod_r, rod_y0, rod_y1 = _ROD
    rod_box, rod_extent = bounding_box('rod', shapes), z_extent('rod', shapes)
    add('rod_y_extents', np.allclose(rod_box, (rod_x - rod_r, rod_y0, rod_x + rod_r, rod_y1), atol=_EXACT)
        and np.allclose(rod_extent, (rod_z - rod_r, rod_z + rod_r), atol=_EXACT),
        {'box': list(rod_box), 'z': list(rod_extent)})
    cone_x, cone_y, cone_r = _CONE[:3]
    add('taper_footprint_is_base', np.allclose(bounding_box('cone', shapes), (cone_x - cone_r, cone_y - cone_r,
                                                                              cone_x + cone_r, cone_y + cone_r),
                                               atol=_EXACT))

    across = vertical_section('rod', shapes, ('y', 1.0e-6), _SIZE[2])   # normal to the rod: its circle in (x, z)
    radii = np.hypot(across[0][:, 0] - rod_x, across[0][:, 1] - rod_z) if len(across) == 1 else np.array([np.nan])
    add('rod_cut_across_is_circle', bool(np.allclose(radii, rod_r, atol=_EXACT)), float(np.nanmax(radii)))
    along = vertical_section('rod', shapes, ('x', rod_x), _SIZE[2])        # through the axis: a rectangle in (y, z)
    along_box = ([float(along[0][:, 0].min()), float(along[0][:, 1].min()), float(along[0][:, 0].max()),
                  float(along[0][:, 1].max())] if len(along) == 1 else [])
    add('rod_cut_along_is_rectangle', bool(along_box) and np.allclose(
        along_box, (rod_y0, rod_z - rod_r, rod_y1, rod_z + rod_r), atol=_EXACT), along_box)
    trapezoid = vertical_section('cone', shapes, ('y', cone_y), _SIZE[2])
    top_half_width = (float(np.max(np.abs(trapezoid[0][trapezoid[0][:, 1] == trapezoid[0][:, 1].max(), 0] - cone_x)))
                      if len(trapezoid) == 1 else math.nan)
    add('cone_cut_is_trapezoid', len(trapezoid) == 1 and abs(top_half_width - _CONE[5]) < 1e-3 * cone_r,
        top_half_width)
    preview = material_preview(spec, {}, max_cells=100, shapes=shapes, cut=('y', cone_y))[0]
    add('cut_preview_has_cone', bool((preview == 1).any()), int((preview == 1).sum()))
    _taper_x_section_checks(shapes, add)


def _taper_x_section_checks(shapes, add):
    """Cone along x: footprint box and z extent of its base, the offset circle across it, its trapezoid along it."""
    from .geometry import bounding_box, vertical_section, z_extent

    axis_y, axis_z, base_r, x0, x1, top_r = _CONE_X
    cone_box, cone_extent = bounding_box('cone_x', shapes), z_extent('cone_x', shapes)
    add('cone_x_extents', np.allclose(cone_box, (x0, axis_y - base_r, x1, axis_y + base_r), atol=_EXACT)
        and np.allclose(cone_extent, (axis_z - base_r, axis_z + base_r), atol=_EXACT),
        {'box': list(cone_box), 'z': list(cone_extent)})
    cut_x = 0.5 * (x0 + x1)
    across = vertical_section('cone_x', shapes, ('x', cut_x), _SIZE[2])   # normal to the axis: circle in (y, z)
    expected_r = base_r - (cut_x - x0) * (base_r - top_r) / (x1 - x0)
    radii = np.hypot(across[0][:, 0] - axis_y, across[0][:, 1] - axis_z) if len(across) == 1 else np.array([np.nan])
    add('cone_x_cut_across_is_offset_circle', bool(np.allclose(radii, expected_r, rtol=_SECTION_RTOL, atol=0.0)),
        {'max_radius': float(np.nanmax(radii)), 'expected': expected_r})
    along = vertical_section('cone_x', shapes, ('y', axis_y), _SIZE[2])   # through the axis: trapezoid in (x, z)
    if len(along) == 1:
        polygon = along[0]
        end_half_height = float(np.max(np.abs(polygon[polygon[:, 0] == polygon[:, 0].max(), 1] - axis_z)))
        start_half_height = float(np.max(np.abs(polygon[polygon[:, 0] == polygon[:, 0].min(), 1] - axis_z)))
    else:
        end_half_height = start_half_height = math.nan
    add('cone_x_cut_along_is_trapezoid', abs(end_half_height - top_r) < _SECTION_RTOL * base_r
        and abs(start_half_height - base_r) < _SECTION_RTOL * base_r,
        {'start_half_height': start_half_height, 'end_half_height': end_half_height})


def _mesh_checks(registry, spec, add):
    """A refined bar along x snaps its x ends, y sides and z sides to cell edges; a refined cone its z ends only."""
    from .meshing import estimate_mesh

    refined = spec.copy()
    refined.mesh = MeshSpec(mode='nonuniform', dx=_MESH_BASE, max_ratio=_MESH_RATIO,
                            refinements=[RefinementSpec('bar', _MESH_FINE, 0.0), RefinementSpec('cone', _MESH_FINE,
                                                                                                 0.0)])
    mesh = estimate_mesh(refined, {}, registry)
    bar_y0, bar_z0, bar_y1, bar_z1, bar_x0, bar_x1 = _BAR
    faces = {'x': (mesh.x_edges, (bar_x0, bar_x1)), 'y': (mesh.y_edges, (bar_y0, bar_y1)),
             'z': (mesh.z_edges, (bar_z0, bar_z1, _CONE[3], _CONE[4]))}
    misses = {axis: max(float(np.min(np.abs(np.asarray(edges) - face))) for face in positions)
              for axis, (edges, positions) in faces.items()}
    add('extrusion_faces_snapped', max(misses.values()) < 1e-3 * _MESH_FINE, misses)


def _validation_checks(registry, add):
    """Collapsing (axis z and x), open-ended x / y and self-crossing tapered prisms fail to resolve; a set sidewall
    angle is rejected in 2D while axis 'z' / angle 0 are not."""
    from .geometry import resolve_geometry
    from .validate import validate

    um = 1e-6
    bad = ModelSpec(title='Self-test: bad prisms', domain=DomainSpec(size_x=_SIZE[0], size_y=_SIZE[1],
                                                                     size_z=_SIZE[2]))
    bad.geometry = [
        GeometrySpec('collapsing', 'rectangle', {'x0': 0.0, 'y0': 0.0, 'x1': 0.2 * um, 'y1': 1.0 * um, 'z0': 0.0,
                                                 'z1': 0.5 * um, 'sidewall_angle': 30.0}),
        GeometrySpec('open_x', 'circle', {'cx': um, 'cy': 0.5 * um, 'r': 0.2 * um, 'z0': 0.1 * um, 'axis': 'x'}),
        GeometrySpec('collapsing_x', 'rectangle', {'x0': 0.0, 'y0': 0.0, 'x1': um, 'y1': 0.5 * um, 'z0': 0.0,
                                                   'z1': um, 'axis': 'x', 'sidewall_angle': 30.0}),
        GeometrySpec('bow_tie', 'polygon', {'points': [(0.0, 0.0), (um, um), (um, 0.0), (0.0, um)], 'z0': 0.0,
                                            'z1': 0.1 * um, 'sidewall_angle': 5.0}),
    ]
    errors = {}
    resolve_geometry(bad, {}, errors=errors)
    expected = [obj.name for obj in bad.geometry]
    add('bad_prisms_rejected', sorted(errors) == sorted(expected), errors)

    flat = ModelSpec(title='Self-test: 2D taper', domain=DomainSpec(size_x=_SIZE[0], size_y=_SIZE[1]))
    flat.geometry = [GeometrySpec('tilted', 'circle', {'cx': um, 'cy': um, 'r': 0.3 * um, 'sidewall_angle': 10.0}),
                     GeometrySpec('neutral', 'circle', {'cx': um, 'cy': um, 'r': 0.2 * um, 'axis': 'z',
                                                        'sidewall_angle': 0.0})]
    rejected = [(issue.path, issue.message) for issue in validate(flat, registry)
                if 'only used in 3D' in issue.message]
    add('sidewall_rejected_in_2d', any(path == ('geometry', 'tilted') and 'sidewall_angle' in message
                                       for path, message in rejected)
        and not any(path == ('geometry', 'neutral') for path, _ in rejected), [message for _, message in rejected])


def _check_geometry_3d(results_by_name):
    """Every sub-check pass / fail; error = number of failed sub-checks, tolerance 0."""
    from .geometry import resolve_geometry
    from .schema import get_registry

    registry = get_registry()
    spec = geometry_3d_spec()
    shapes = resolve_geometry(spec, {})
    centres = [((np.arange(n) + 0.5) * _CELL).astype(np.float32) for n in _CELLS]
    checks = SubChecks()
    add = checks.add

    _volume_checks(shapes, centres, add)
    _section_checks(spec, shapes, centres, add)
    _mesh_checks(registry, spec, add)
    _validation_checks(registry, add)
    return checks.result()


GEOMETRY_3D_SELFTEST_CASES = (
    SelfTestCase(
        name='geometry_extrusion_taper',
        label='3D prisms: extrusion axis and sidewall angle',
        description='CPU checks of the 3D prism extensions on a 2 x 2 x 1 um grid of 25 nm cells (80 x 80 x 40): '
                    'staircase volumes within 2 % of the closed forms of a cylinder along y (circle, axis y), a '
                    'cone frustum (circle, sidewall_angle), a tapered clockwise triangle (mitred polygon offset = '
                    'similar triangles), a cone along x and a tapered rectangle along y (sampled sweeps); a '
                    'rectangle along x identical to the equivalent box and a polygon along y equal to its slab stack '
                    '(2D polygon raster per z, < 0.1 % boundary ties); one cross-section run per z cell of the taper; '
                    'the x / y extrusion bounding box and z extent, the footprint of a narrowing taper = its base '
                    '(also along x); vertical sections (the rod circle across it, its rectangle along it, the cone '
                    'trapezoid top half-width, the x cone offset circle across it and trapezoid ends along it) and a '
                    'vertical-cut material preview; on a nonuniform mesh the bar faces snapped along x, y, z and the '
                    'cone ends along z; collapsing (axis z and x), open x and self-crossing tapered prisms rejected, '
                    'a sidewall angle rejected in 2D while axis z / angle 0 pass. error = number of failed '
                    'sub-checks, tolerance 0. 25 nm cells not yet confirmed by a run.',
        specs=lambda: {},
        check=_check_geometry_3d,
        max_wall_time=60.0),
)
