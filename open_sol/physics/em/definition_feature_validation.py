"""CPU-only feature validation of the 2D EM types (placement on the MeshEstimate as the runtime looks cells up: lines,
boxes, wall-pinned rows, mode lines, TFSF / mode-source interfaces, source media); physics validation: definition_common."""
import math
from dataclasses import dataclass

import numpy as np

from ...core.constants import C0
from ...core.domain import containing_cell
from ...core.mesh import has_uniform_spacing
from ...core.mirror import MIRROR_SIGNS, plane_wave_mirror_issue
from ...model.definition_helpers import evaluates_to, validate_points_inside, wavelength_range_issues
from ...model.expressions import evaluate_scalar
from ...model.geometry import bounding_box, line_orientation, preview_pixel_centers
from ...model.schema import PERMITTIVITY_PROPERTY_NAMES, Issue, feature_path, resolve_values
from .boundaries import PINNED_FACE_TYPES, WALL_SIGNS
from .definition_common import (FOCUSED_BEAM_FEATURE, _effective_material, _refractive_index, _window_axes,
                                eigen_only_feature_issues, raw_settings, resolved_solved_wavelengths)
from .focused_beam import FOCUS_TOLERANCE, MIN_SPAN_CELLS, NORMAL_RANGE, continuum_end_levels
from .memory_estimates import _DFT_SPECTRA
from .placement import TFSF_POLARIZATION_AXES

_VACUUM_PROPERTIES = {**dict.fromkeys(PERMITTIVITY_PROPERTY_NAMES, 1.0), 'mu_r': 1.0, 'sigma_e': 0.0, 'sigma_m': 0.0}


# ---- shape helpers ----
def _selected_shapes(vctx, feature_spec, kind):
    """Resolved shapes of geometry kind `kind` in the feature selection (other problems are reported elsewhere)."""
    shapes = (vctx.shapes.get(name) for name in vctx.objects(feature_spec.selection))
    return [shape for shape in shapes if shape is not None and shape.kind == kind]


def _outside_message(points, size):
    """Problem text when any (x, y) lies outside [0, size_x] x [0, size_y], else None."""
    size_x, size_y = size
    for x, y in points:
        if not (0.0 <= x <= size_x and 0.0 <= y <= size_y):
            return (f'has ({x:.4g}, {y:.4g}) m outside the physical domain '
                    f'[0, {size_x:.4g}] x [0, {size_y:.4g}] m')
    return None


def _shape_corners(shape):
    """(x, y) points of a point shape, or the two end / corner points of a line / rectangle shape."""
    values = shape.values
    if shape.kind == 'point':
        return ((values['x'], values['y']),)
    return ((values['x0'], values['y0']), (values['x1'], values['y1']))


def _line_axes(line, size):
    """(axis along, normal axis, position across, (start, end) sorted along) of an axis-aligned line's values; None
    when it is not axis-aligned."""
    orientation = line_orientation(line, *size)
    if orientation is None:
        return None
    along_axis, normal_axis = ('x', 'y') if orientation == 'horizontal' else ('y', 'x')
    position = 0.5 * (line[f'{normal_axis}0'] + line[f'{normal_axis}1'])
    start, end = sorted((line[f'{along_axis}0'], line[f'{along_axis}1']))
    return along_axis, normal_axis, position, (start, end)


def _shape_issues(vctx, physics_spec, feature_spec, kind, shape_messages, needs_mesh=True):
    """Issues '<kind> '<name>' <text>' of shape_messages(shape) -> [(level, text)] per selected `kind` shape inside the
    domain (outside ones are reported by the inside checks); [] without a domain size (or a mesh, with needs_mesh)."""
    size = vctx.domain_size
    if size is None or (needs_mesh and vctx.mesh is None):
        return []
    path = feature_path(physics_spec, feature_spec)
    issues = []
    for shape in _selected_shapes(vctx, feature_spec, kind):
        if _outside_message(_shape_corners(shape), size):
            continue
        issues.extend(Issue(level, path, f"{kind} '{shape.name}' {text}") for level, text in shape_messages(shape))
    return issues


def _is_vacuum(material_spec, parameters):
    """True when the material evaluates to unit permittivities and mu_r, no conductivities and no dispersion."""
    try:
        properties, dispersion = _effective_material(material_spec)
    except ValueError:   # unknown library / malformed properties (reported at the material): not provably vacuum
        return False
    if dispersion:
        return False
    for name, vacuum_value in _VACUUM_PROPERTIES.items():
        if name not in properties:
            continue
        try:
            value = evaluate_scalar(properties[name], parameters)
        except ValueError:   # field- / state-dependent or broken: not provably vacuum
            return False
        if not math.isclose(value, vacuum_value, rel_tol=1e-6, abs_tol=1e-12):
            return False
    return True


# ---- points, lines, boxes ----
def electric_dipole_issues(physics_spec, feature_spec, values):
    """Error of an H_z point_source with excitation electric_dipole unless dipole_moment is finite and non-zero and
    orientation finite ([] otherwise, also for a magnetic_current)."""
    if values['excitation'] != 'electric_dipole':
        return []
    moment, orientation = values['dipole_moment'], values['orientation']
    if math.isfinite(moment) and moment != 0.0 and math.isfinite(orientation):
        return []
    return [Issue('error', feature_path(physics_spec, feature_spec),
                  f'electric_dipole needs a finite non-zero dipole_moment and a finite orientation, got dipole_moment '
                  f'= {moment:g}, orientation = {orientation:g}')]


def validate_hz_point_source(vctx, physics_spec, feature_spec, values):
    """Point inside the domain, electric_dipole_issues, then (finite orientation) the wall warnings of
    validate_hz_point_walls (FDTD: overwritten every step; FDFD: the same pinned rows)."""
    issues = validate_points_inside(vctx, physics_spec, feature_spec, values)
    dipole_issues = electric_dipole_issues(physics_spec, feature_spec, values)
    if dipole_issues:   # the wall check needs a finite orientation
        return issues + dipole_issues
    return issues + validate_hz_point_walls(vctx, physics_spec, feature_spec, values)


def _line_issues(vctx, physics_spec, feature_spec):
    """Line objects axis-aligned and inside the physical domain."""
    path = feature_path(physics_spec, feature_spec)
    size = vctx.domain_size
    if size is None:
        return []
    issues = []
    for shape in _selected_shapes(vctx, feature_spec, 'line'):
        if line_orientation(shape.values, *size) is None:
            issues.append(Issue('error', path, f"line '{shape.name}' must be horizontal or vertical"))
        message = _outside_message(_shape_corners(shape), size)
        if message:
            issues.append(Issue('error', path, f"line '{shape.name}' {message}"))
    return issues


def _box_issues(vctx, physics_spec, feature_spec):
    """(issues, rectangles inside the domain): rectangle corners must lie inside the physical domain."""
    path = feature_path(physics_spec, feature_spec)
    size = vctx.domain_size
    if size is None:
        return [], []
    issues, inside = [], []
    for shape in _selected_shapes(vctx, feature_spec, 'rectangle'):
        message = _outside_message(_shape_corners(shape), size)
        if message:
            issues.append(Issue('error', path, f"rectangle '{shape.name}' {message}"))
        else:
            inside.append(shape)
    return issues, inside


def _tfsf_spans_period_y(mesh, box, settings):
    """True when the box covers every y cell of a periodic y axis (an FDFD TFSF box without y edges)."""
    if settings.get('bc_y_lo') != 'periodic' or settings.get('bc_y_hi') != 'periodic':
        return False
    widths, edges, first, last = _window_axes(mesh)[1]
    lo, hi = sorted((box['y0'], box['y1']))
    return (containing_cell(lo, edges, first, last) == 0
            and containing_cell(hi, edges, first, last) == widths.size - 1)


def _tfsf_reached_faces(mesh, box):
    """Window faces whose edge cell the TFSF box covers (as placement.box_window_faces)."""
    reached = set()
    for axis, (_, edges, first, last) in zip('xy', _window_axes(mesh)):
        lo, hi = sorted((box[f'{axis}0'], box[f'{axis}1']))
        if containing_cell(lo, edges, first, last) <= first:
            reached.add(f'{axis}_lo')
        if containing_cell(hi, edges, first, last) >= last:
            reached.add(f'{axis}_hi')
    return reached


def _tfsf_margin_message(mesh, box, full_period_y=False, open_faces=()):
    """Problem text unless one scattered-field cell stays inside the window on every side (as add_tfsf) but the
    open_faces (y mirror faces the box opens onto); full_period_y (FDFD) accepts every y cell of a periodic y axis."""
    for axis, (_, edges, first, last) in zip('xy', _window_axes(mesh)):
        if axis == 'y' and full_period_y:
            continue
        lo_cell = containing_cell(box[f'{axis}0'], edges, first, last)
        hi_cell = containing_cell(box[f'{axis}1'], edges, first, last)
        lo_cell, hi_cell = sorted((lo_cell, hi_cell))
        if (lo_cell <= first and f'{axis}_lo' not in open_faces) or (hi_cell >= last and f'{axis}_hi' not in open_faces):
            return (f'needs one scattered-field cell inside the domain on every side (a y mirror face of the plane '
                    f"wave's parity excepted): keep {axis}0 >= {edges[first + 1]:.4g} m and {axis}1 < "
                    f'{edges[last]:.4g} m')
    return None


_TFSF_POLARIZATIONS = {'em_fdtd_ez': 'ez', 'em_fdfd_ez': 'ez', 'em_fdtd_hz': 'hz', 'em_fdfd_hz': 'hz'}


def _tfsf_open_faces(mesh, box, physics_spec, settings):
    """(mirror faces the TFSF box opens onto, problem texts of reached mirror faces it may not open onto) as
    placement.tfsf_open_faces; nothing opens for other types (nlfd channels) or without a mesh estimate."""
    polarization = _TFSF_POLARIZATIONS.get(physics_spec.type)
    if mesh is None or polarization is None:
        return frozenset(), []
    open_faces, problems = set(), []
    for face in sorted(_tfsf_reached_faces(mesh, box)):
        sign = MIRROR_SIGNS.get(settings.get(f'bc_{face}'))
        if sign is None:
            continue
        problem = plane_wave_mirror_issue(face, sign, 'x', TFSF_POLARIZATION_AXES[polarization])
        if problem is None:
            open_faces.add(face)
        else:
            problems.append(problem)
    return frozenset(open_faces), problems


def _far_field_margin_message(mesh, box):
    """Problem text when the NF2FF samples beyond x1 / y1 fall outside the window (as add_near_to_far)."""
    for axis, (_, edges, first, last) in zip('xy', _window_axes(mesh)):
        if containing_cell(box[f'{axis}1'], edges, first, last) >= last:
            return (f'samples the cell beyond its {axis}1 edge, which must stay inside the domain: '
                    f'keep {axis}1 < {edges[last]:.4g} m')
    return None


# ---- total-field / scattered-field interfaces (TFSF outlines, mode-source interface faces) ----
# Cell sets are ((x_lo, x_hi), (y_lo, y_hi)) half-open grid-cell ranges (math.inf for an unbounded half plane). TFSF
# and mode phasors / corrections enter only on their interface, so the solved field jumps there: a sample reading
# cells on both sides mixes total and scattered field. The FDTD stores interface faces as scattered field, so an FDTD
# sample is consistently scattered whenever its own cells (line / sample cells) lie outside the total field.
def _rectangle_cells(mesh, box):
    """Half-open cells [cell(lo), cell(hi) + 1) per axis of a rectangle, as placement.box_cells looks them up."""
    cells = []
    for axis, (_, edges, first, last) in zip('xy', _window_axes(mesh)):
        lo, hi = sorted((box[f'{axis}0'], box[f'{axis}1']))
        cells.append((containing_cell(lo, edges, first, last), containing_cell(hi, edges, first, last) + 1))
    return tuple(cells)


def _line_cells(mesh, line, size):
    """(normal axis, line cell, half-open span cells along the line) of an axis-aligned line, as
    placement.flux_line_cells / port_cells look them up; None when the line is not axis-aligned."""
    line_axes = _line_axes(line, size)
    if line_axes is None:
        return None
    along_axis, normal_axis, position, (start, end) = line_axes
    axes = dict(zip('xy', _window_axes(mesh)))
    _, normal_edges, normal_first, normal_last = axes[normal_axis]
    _, along_edges, along_first, along_last = axes[along_axis]
    return (normal_axis, containing_cell(position, normal_edges, normal_first, normal_last),
            (containing_cell(start, along_edges, along_first, along_last),
             containing_cell(end, along_edges, along_first, along_last) + 1))


def _cells_on_axes(normal_axis, normal_cells, along_cells):
    """Cell set from the cell ranges across (along normal_axis) and along a line."""
    return (normal_cells, along_cells) if normal_axis == 'x' else (along_cells, normal_cells)


def _cells_inside(cells, box):
    """True when every cell of the cell set lies in the cell set box."""
    return all(box_lo <= lo and hi <= box_hi for (lo, hi), (box_lo, box_hi) in zip(cells, box))


def _cells_disjoint(cells, box):
    """True when the two cell sets share no cell."""
    return any(hi <= box_lo or lo >= box_hi for (lo, hi), (box_lo, box_hi) in zip(cells, box))


# one-sided TFSF line sources (total field downstream of the interface face: lo face of the line cell when positive)
_ONE_SIDED_LINE_TYPES = ('mode_source', 'focused_beam')


def _total_field_regions(vctx, physics_spec):
    """[(kind 'tfsf' | 'mode', feature name, interface text, total-field cells, jump cells)] of the enabled
    plane_wave_tfsf boxes and mode_source / focused_beam lines of the physics inside the domain: a one-sided line's
    total field is its downstream half plane, its jump cells the two cells beside its interface face over the span."""
    size = vctx.domain_size
    regions = []
    for feature in physics_spec.features:
        if not feature.enabled or feature.type not in ('plane_wave_tfsf', *_ONE_SIDED_LINE_TYPES):
            continue
        kind = 'rectangle' if feature.type == 'plane_wave_tfsf' else 'line'
        for shape in _selected_shapes(vctx, feature, kind):
            values = shape.values
            if _outside_message(_shape_corners(shape), size):
                continue   # reported at that feature
            if kind == 'rectangle':
                box = _rectangle_cells(vctx.mesh, values)
                regions.append(('tfsf', feature.name, f"the outline of the plane_wave_tfsf box '{feature.name}'",
                                box, box))
                continue
            placement = _line_cells(vctx.mesh, values, size)
            if placement is None:
                continue   # reported at that feature
            normal_axis, line_cell, span = placement
            params = feature.params if isinstance(feature.params, dict) else {}
            positive = params.get('direction', 'positive') == 'positive'
            face = line_cell - 1 if positive else line_cell   # interface face = hi face of cell `face`
            face_coordinate = _window_axes(vctx.mesh)['xy'.index(normal_axis)][1][face + 1]
            downstream = (line_cell, math.inf) if positive else (-math.inf, line_cell + 1)
            regions.append(('mode', feature.name, f"the interface face of {feature.type} '{feature.name}' at "
                                                  f'{normal_axis} = {face_coordinate:.4g} m',
                            _cells_on_axes(normal_axis, downstream, (-math.inf, math.inf)),
                            _cells_on_axes(normal_axis, (face, face + 2), span)))
    return regions


def _mixed_region_texts(regions, read_cells, own_cells, time_domain):
    """Texts of the regions whose interface a sample straddles: it reads read_cells (every cell its values come from);
    FDTD samples whose own_cells lie outside the total field are consistently scattered."""
    if not time_domain:
        own_cells = read_cells
    return [text for _, _, text, total, jump in regions
            if not _cells_disjoint(read_cells, jump) and not _cells_inside(read_cells, total)
            and not _cells_disjoint(own_cells, total)]


def monitor_line_interface_issues(vctx, physics_spec, feature_spec, time_domain):
    """Warnings for a flux / mode monitor line whose line cells and hi faces read both sides of a TFSF outline or a
    mode-source interface face."""
    if vctx.domain_size is None or vctx.mesh is None:
        return []
    regions = _total_field_regions(vctx, physics_spec)

    def messages(shape):
        placement = _line_cells(vctx.mesh, shape.values, vctx.domain_size)
        if placement is None:
            return []   # reported by _line_issues
        normal_axis, line_cell, span = placement
        read_cells = _cells_on_axes(normal_axis, (line_cell, line_cell + 2), span)   # line cells + beyond the hi face
        own_cells = _cells_on_axes(normal_axis, (line_cell, line_cell + 1), span)
        return [('warning', f'measures across {text} (its line cells and the cells beyond their hi face lie on both '
                            f'sides), so it mixes total and scattered field; move it entirely to one side')
                for text in _mixed_region_texts(regions, read_cells, own_cells, time_domain)]
    return _shape_issues(vctx, physics_spec, feature_spec, 'line', messages) if regions else []


# FDFD probe face components -> (x, y) cell offsets their value comes from (ez H on the +y / +x face; hz E collocated
# from the two faces of the cell)
_HARMONIC_PROBE_OFFSETS = {'hx': ((0, 1), (0, 2)), 'hy': ((0, 2), (0, 1)),
                           'ex': ((0, 1), (-1, 2)), 'ey': ((-1, 2), (0, 1))}


def harmonic_probe_interface_issues(vctx, physics_spec, feature_spec, values):
    """Warnings for an FDFD face-component probe (ez hx / hy, hz ex / ey) whose value comes from cells on both sides
    of a TFSF outline or a mode-source interface face."""
    offsets = _HARMONIC_PROBE_OFFSETS.get(values['component'])
    if offsets is None or vctx.domain_size is None or vctx.mesh is None:
        return []
    regions = _total_field_regions(vctx, physics_spec)

    def messages(shape):
        read_cells = []
        for coord, (_, edges, first, last), (lo_offset, hi_offset) in zip(_shape_corners(shape)[0],
                                                                           _window_axes(vctx.mesh), offsets):
            cell = containing_cell(coord, edges, first, last)
            read_cells.append((cell + lo_offset, cell + hi_offset))
        return [('warning', f"takes {values['component']} from cells on both sides of {text}, so it mixes total and "
                            f'scattered field; move it a cell away from that face')
                for text in _mixed_region_texts(regions, read_cells, read_cells, time_domain=False)]
    return _shape_issues(vctx, physics_spec, feature_spec, 'point', messages) if regions else []


def _far_field_tfsf_message(mesh, far_cells, tfsf_cells, time_domain):
    """Problem text unless the far-field box cells enclose the TFSF box cells with the margins its samples need (each
    averages the faces on both sides of its cell): FDTD 1 cell before the lo sides, x1 / y1 edges in the last TFSF cell
    or beyond (outline faces are scattered field); FDFD 2 cells before, x1 / y1 edges 1 cell past it."""
    lo_margin, hi_margin = (1, 0) if time_domain else (2, 1)
    if all(far_lo <= tfsf_lo - lo_margin and far_hi >= tfsf_hi + hi_margin
           for (far_lo, far_hi), (tfsf_lo, tfsf_hi) in zip(far_cells, tfsf_cells)):
        return None
    bounds = []
    for axis, (tfsf_lo, tfsf_hi), (_, edges, _, _) in zip('xy', tfsf_cells, _window_axes(mesh)):
        lo_edge = edges[min(max(tfsf_lo - lo_margin + 1, 0), edges.size - 1)]
        hi_edge = edges[min(max(tfsf_hi - 1 + hi_margin, 0), edges.size - 1)]
        bounds.append(f'{axis}0 < {lo_edge:.4g} m and {axis}1 >= {hi_edge:.4g} m')
    hi_text = 'in the last TFSF cell or beyond' if time_domain else 'at least one cell past the TFSF box'
    return (f'with {lo_margin} cell{"s" if lo_margin > 1 else ""} to spare before its lo-x / lo-y sides and the x1 / '
            f'y1 edges {hi_text}, so every sample (it averages the faces on both sides of its cell) sees scattered '
            f'field only: keep {", ".join(bounds)}')


def _far_field_interface_issues(vctx, physics_spec, shape, path, time_domain):
    """Warnings for a far-field box not enclosing a TFSF box with the sample margins, or sampling across a
    mode-source interface face."""
    regions = _total_field_regions(vctx, physics_spec)
    if not regions:
        return []
    far_cells = _rectangle_cells(vctx.mesh, shape.values)
    issues = []
    for kind, feature_name, _, total, _ in regions:
        message = _far_field_tfsf_message(vctx.mesh, far_cells, total, time_domain) if kind == 'tfsf' else None
        if message:
            issues.append(Issue('warning', path, f"far-field box '{shape.name}' must enclose the plane_wave_tfsf box "
                                                 f"'{feature_name}' {message}"))
    (x0, x1), (y0, y1) = far_cells   # sample columns x0, x1 and rows y0, y1 (x1 / y1 one beyond the box)
    sides = (((x0 - 1, x0 + 2), (y0, y1 + 1), (x0, x0 + 1), (y0, y1 + 1)),
             ((x1 - 1, x1 + 2), (y0, y1 + 1), (x1, x1 + 1), (y0, y1 + 1)),
             ((x0, x1 + 1), (y0 - 1, y0 + 2), (x0, x1 + 1), (y0, y0 + 1)),
             ((x0, x1 + 1), (y1 - 1, y1 + 2), (x0, x1 + 1), (y1, y1 + 1)))
    mode_regions = [region for region in regions if region[0] == 'mode']
    texts = [text for read_x, read_y, own_x, own_y in sides
             for text in _mixed_region_texts(mode_regions, (read_x, read_y), (own_x, own_y), time_domain)]
    for text in dict.fromkeys(texts):
        issues.append(Issue('warning', path, f"far-field box '{shape.name}' samples across {text}, so its contour "
                                             f'mixes total and scattered field; move the box edge off that face'))
    return issues


# ---- media on source lines and box outlines (material preview) ----
def preview_pixel(coord, lo, hi, count):
    """Index of the material-preview pixel containing coord, clamped to [0, count - 1]."""
    return min(max(int(math.floor((coord - lo) / (hi - lo) * count)), 0), count - 1)


def _tfsf_edge_materials(vctx, box, uniform=False, full_period_y=False, open_faces=()):
    """Names of the materials the material preview finds on the outline of box (no y edges with full_period_y, none on
    an open_faces side): the non-vacuum ones, or with uniform (FDFD) all of them unless they share one medium ([] when
    unavailable)."""
    material_map = vctx.material_map()
    if material_map is None:
        return []
    image, names, (x_min, x_max, y_min, y_max) = material_map
    i0, i1 = sorted(preview_pixel(box[key], x_min, x_max, image.shape[0]) for key in ('x0', 'x1'))
    j0, j1 = sorted(preview_pixel(box[key], y_min, y_max, image.shape[1]) for key in ('y0', 'y1'))
    edges = [image[i0, j0:j1 + 1], image[i1, j0:j1 + 1]]
    if not full_period_y:
        edges += [image[i0:i1 + 1, row] for row, face in ((j0, 'y_lo'), (j1, 'y_hi')) if face not in open_faces]
    outline_indices = {int(index) for index in np.unique(np.concatenate(edges))}
    if uniform:
        return _distinct_media(vctx, outline_indices, names)
    materials = vctx.spec.materials
    return [names[index - 1] for index in sorted(outline_indices)
            if index > 0 and not _is_vacuum(materials[index - 1], vctx.parameters)]


def validate_line_feature(vctx, physics_spec, feature_spec, values):
    return _line_issues(vctx, physics_spec, feature_spec)


_BEAM_EDGE_LEVEL = 1e-2   # gaussian-beam profile / peak above which a span end truncates it (and the media check counts)


def _medium_key(material_spec, parameters):
    """Comparable EM medium of a material (evaluated permittivities, mu_r, conductivities, poles); its name when it
    does not evaluate to plain numbers (field-dependent or malformed)."""
    try:
        properties, dispersion = _effective_material(material_spec)
        values = tuple(evaluate_scalar(properties[name], parameters) if properties.get(name) is not None else default
                       for name, default in _VACUUM_PROPERTIES.items())
    except (ValueError, TypeError):
        return material_spec.name
    return values, repr(list(dispersion))


def _line_source_media(vctx, along_axis, position, span, beam, polarization):
    """Names of the distinct media the material preview finds on the line cells (a beam: where its profile is >=
    _BEAM_EDGE_LEVEL; hz: also on the cells past their hi face, which share the face eps); [] when uniform / unknown."""
    found = _line_material_indices(vctx, along_axis, position, span, beam, polarization)
    return [] if found is None else _distinct_media(vctx, *found)


def _line_material_indices(vctx, along_axis, position, span, beam, polarization, extra_across=()):
    """(material-preview indices (0 = vacuum), names) on the line cells as _line_source_media reads them, plus the rows
    at the extra_across coordinates [m]; None without a preview."""
    material_map = vctx.material_map()
    if material_map is None:
        return None
    image, names, extent = material_map
    along_index = 'xy'.index(along_axis)
    across_index = 1 - along_index
    along_lo, along_hi = extent[2 * along_index], extent[2 * along_index + 1]
    across_lo, across_hi = extent[2 * across_index], extent[2 * across_index + 1]
    first_pixel, last_pixel = (preview_pixel(coord, along_lo, along_hi, image.shape[along_index]) for coord in span)
    pixels = np.arange(first_pixel, last_pixel + 1)
    if beam is not None:
        center, waist = beam
        along_centers = preview_pixel_centers(extent, image.shape)[along_index][pixels]
        pixels = pixels[np.exp(-((along_centers - center) / waist) ** 2) >= _BEAM_EDGE_LEVEL]
    across_coords = [position, *extra_across]
    if polarization == 'hz' and vctx.mesh is not None:   # the cell past the sheet's hi face (window-clamped)
        _, edges, first, last = _window_axes(vctx.mesh)[across_index]
        next_cell = min(containing_cell(position, edges, first, last) + 1, last)
        across_coords.append(0.5 * (edges[next_cell] + edges[next_cell + 1]))
    indices = set()
    for coord in across_coords:
        row = preview_pixel(coord, across_lo, across_hi, image.shape[across_index])
        indices.update(int(index) for index in (image[pixels, row] if along_index == 0 else image[row, pixels]))
    return indices, names


def _distinct_media(vctx, indices, names):
    """Names of the materials of material-preview indices (0 = vacuum) when they hold more than one EM medium, else
    []."""
    materials = vctx.spec.materials
    vacuum_key = (tuple(_VACUUM_PROPERTIES.values()), repr([]))
    media = {}   # medium key -> names of the materials that have it
    for index in sorted(indices):
        key = _medium_key(materials[index - 1], vctx.parameters) if index > 0 else vacuum_key
        media.setdefault(key, []).append(names[index - 1] if index > 0 else 'vacuum')
    return [name for medium_names in media.values() for name in medium_names] if len(media) > 1 else []


def _end_in_outer_cell(mesh, settings, axis, side, coord, size, bc_types):
    """True when a line end on a side whose bc is one of bc_types lies in the outermost window cell of the mesh
    estimate `mesh` (None: on the domain edge)."""
    if settings.get(f'bc_{axis}_{side}', 'cpml') not in bc_types:
        return False
    axis_index = 'xy'.index(axis)
    if mesh is None:
        edge = 0.0 if side == 'lo' else size[axis_index]
        return abs(coord - edge) <= 1e-9 * size[axis_index]
    _, edges, first, last = _window_axes(mesh)[axis_index]
    return containing_cell(coord, edges, first, last) == (first if side == 'lo' else last)


def _end_continues_into_pml(mesh, settings, axis, side, coord, size):
    """A source line end on a cpml side in the outermost window cell: the line continues through the PML
    (placement.sheet_line)."""
    return _end_in_outer_cell(mesh, settings, axis, side, coord, size, ('cpml',))


def _end_on_mirror(mesh, settings, axis, side, coord, size):
    """A source line end on a symmetric / antisymmetric side in the outermost window cell, next to the mirror plane on
    the window edge: the mirror images the line, nothing is cut there."""
    return _end_in_outer_cell(mesh, settings, axis, side, coord, size, ('symmetric', 'antisymmetric'))


def _beam_placement_messages(vctx, settings, along_axis, span, center, waist, size):
    """Problem texts of a gaussian beam centred off its line, or whose profile exceeds _BEAM_EDGE_LEVEL at a span end
    that stops inside the domain (not on a mirror plane)."""
    start, end = span
    if not start <= center <= end:
        return [f'has its beam centre {along_axis} = {center:.4g} m off the line ({along_axis} = {start:.4g} to '
                f'{end:.4g} m), so only the tail of the profile is launched; set center on the line']
    messages = []
    reach = waist * math.sqrt(-math.log(_BEAM_EDGE_LEVEL))   # distance from the centre where the profile hits the level
    for side, coord in (('lo', start), ('hi', end)):
        level = math.exp(-((coord - center) / waist) ** 2)
        if level > _BEAM_EDGE_LEVEL and not (
                _end_continues_into_pml(vctx.mesh, settings, along_axis, side, coord, size)
                or _end_on_mirror(vctx.mesh, settings, along_axis, side, coord, size)):
            messages.append(f'truncates the beam at {along_axis} = {coord:.4g} m, where the profile is still '
                            f'{level:.2g} of its peak (above {_BEAM_EDGE_LEVEL:g}), so the cut edge diffracts; extend '
                            f'the line to at least {reach:.4g} m from the centre')
    return messages


def _source_line_issues(vctx, physics_spec, feature_spec, values, polarization):
    """Warnings of a line_source / gaussian_beam: different media along its line (see _line_source_media) and, for a
    beam, a centre off the line or a truncated profile."""
    size = vctx.domain_size
    settings = raw_settings(physics_spec)

    def warnings(shape):
        line_axes = _line_axes(shape.values, size)
        if line_axes is None:
            return []   # reported by _line_issues
        along_axis, _, position, span = line_axes
        beam, messages = None, []
        if 'waist' in values:
            center = values['center'] if values['center'] is not None else 0.5 * (span[0] + span[1])
            if not (math.isfinite(center) and values['waist'] > 0):
                return []
            messages = _beam_placement_messages(vctx, settings, along_axis, span, center, values['waist'], size)
            beam = (center, values['waist'])
        media = _line_source_media(vctx, along_axis, position, span, beam, polarization)
        if media:
            where = f' where the profile exceeds {_BEAM_EDGE_LEVEL:g}' if beam else ''
            past_face = ' (or on the cells past their hi face, which share the face eps)' if polarization == 'hz' else ''
            messages.append(f"has different media on its line cells{where}{past_face} ({', '.join(media)}): each "
                            f'line cell is calibrated to radiate the amplitude in its own medium, so the current '
                            f'steps at the interfaces, the steps diffract and the field there is not the amplitude; '
                            f'keep the line in one medium')
        return [('warning', message) for message in messages]
    return _shape_issues(vctx, physics_spec, feature_spec, 'line', warnings, needs_mesh=False)


# ---- wall-pinned rows ----
# Wall BCs of BoundaryManager, re-applied after source injection, per side: the bc types that overwrite the
# out-of-plane field (E_z: apply_ez_walls, H_z: apply_hz_walls) in the outermost grid cell, alike for both
# polarisations, and per polarisation those that pin the in-plane field on the inner face of that cell (face 0 / n - 2,
# face k = hi face of cell k, n = cells along the axis): H_z pec / antisymmetric zero the tangential E there
# (apply_exy_walls); E_z pmc / symmetric copy the E_z row across it, so its curl vanishes and the tangential H stays
# frozen (apply_h_walls, re-applied after injection, also zeroes the pmc faces 0 / n - 2). The FDFD pins the same cells
# and zeroes the same faces' weights. Both sets come from boundaries.WALL_SIGNS / PINNED_FACE_TYPES (also read by
# mode_ports for scripts). A mirror face's outermost cell is the ghost padded outside the window, so only its plane
# face can hold a feature.
_PINNED_CELL_BCS = frozenset(bc.value for bc in WALL_SIGNS['ez'])
_PINNED_FACE_BCS = {polarization: frozenset(bc.value for bc in types)
                    for polarization, types in PINNED_FACE_TYPES.items()}


def _pinned_positions(settings, axis, n_cells, polarization):
    """({grid cell: side} whose out-of-plane field, {face: side} whose in-plane field (face k = hi face of cell k) the
    wall BCs of `polarization` overwrite or freeze every step along `axis`; side 'lo' / 'hi' of bc_<axis>_<side>."""
    pinned_cells, pinned_faces = {}, {}
    for side, outer_cell, wall_face in (('lo', 0, 0), ('hi', n_cells - 1, n_cells - 2)):
        bc_type = settings.get(f'bc_{axis}_{side}', 'cpml')
        if bc_type in _PINNED_CELL_BCS:
            pinned_cells[outer_cell] = side
        if bc_type in _PINNED_FACE_BCS[polarization]:
            pinned_faces[wall_face] = side
    return pinned_cells, pinned_faces


def _wall_text(settings, axis, side, edges, lo_edge_index, hi_edge_index):
    """Wall label 'bc_<axis>_<side> = <type>' and the bound that clears it: <axis> >= edges[lo_edge_index] at a lo
    wall, <axis> < edges[hi_edge_index] at a hi wall."""
    label = f"bc_{axis}_{side} = {settings.get(f'bc_{axis}_{side}')}"
    if side == 'lo':
        return label, f'keep {axis} >= {edges[lo_edge_index]:.4g} m'
    return label, f'keep {axis} < {edges[hi_edge_index]:.4g} m'


def _hz_sheet_messages(mesh, line, size, settings):
    """[(level, problem text)] of an H_z sheet (hi face of the line cell): error on the outer grid wall of a
    non-periodic hi face without PML (as placement.sheet_line refuses), warning on a face a wall BC pins."""
    line_axes = _line_axes(line, size)
    if line_axes is None:
        return []   # reported by _line_issues
    along_axis, axis, position, _ = line_axes
    component = f'E_{along_axis}'
    if settings.get(f'bc_{axis}_hi') == 'periodic':   # the last face wraps; periodic axes pin nothing
        return []
    widths, edges, first, last = _window_axes(mesh)['xy'.index(axis)]
    line_cell = containing_cell(position, edges, first, last)
    last_grid_cell = widths.size - 1   # a window cell only without PML on the hi face
    if line_cell == last_grid_cell:
        return [('error', f'at {axis} = {position:.4g} m puts the H_z sheet on the hi face of the last grid cell, '
                          f'the outer grid wall; keep {axis} < {edges[last_grid_cell]:.4g} m')]
    side = _pinned_positions(settings, axis, widths.size, 'hz')[1].get(line_cell)
    if side is None:
        return []
    label, keep = _wall_text(settings, axis, side, edges, line_cell + 1, line_cell)
    return [('warning', f'at {axis} = {position:.4g} m puts the H_z sheet ({component}) on the inner face of the '
                        f'outermost cell, which {label} pins to 0 every step, so it radiates nothing; {keep}')]


def validate_hz_line_feature(vctx, physics_spec, feature_spec, values):
    """_line_issues and _source_line_issues plus the H_z sheet (hi face of the line cell): error on the outer grid
    wall, warning on a face a wall BC pins to 0."""
    settings = raw_settings(physics_spec)
    return (_line_issues(vctx, physics_spec, feature_spec)
            + _source_line_issues(vctx, physics_spec, feature_spec, values, 'hz')
            + _shape_issues(vctx, physics_spec, feature_spec, 'line', lambda shape: _hz_sheet_messages(
                vctx.mesh, shape.values, vctx.domain_size, settings)))


def validate_hz_point_walls(vctx, physics_spec, feature_spec, values):
    """Warnings for an H_z point source whose injected values a wall BC overwrites every step (orientation must be
    finite): a magnetic_current in a pinned outermost cell, electric_dipole faces on a pinned inner face."""
    settings = raw_settings(physics_spec)
    return _shape_issues(vctx, physics_spec, feature_spec, 'point', lambda shape: [
        ('warning', message) for message in _hz_point_wall_messages(vctx.mesh, _shape_corners(shape)[0], settings,
                                                                    values)])


def _hz_point_wall_messages(mesh, point, settings, values):
    """Problem texts (at most one per axis) of an H_z point source whose injected values a wall BC overwrites."""
    dipole = values['excitation'] == 'electric_dipole'
    if dipole:   # current share on the faces across each axis: E_y faces across x (sin), E_x faces across y (cos)
        theta = math.radians(values['orientation'])
        shares = {'x': abs(math.sin(theta)), 'y': abs(math.cos(theta))}
    messages = []
    for axis, position, (widths, edges, first, last) in zip('xy', point, _window_axes(mesh)):
        cell = containing_cell(position, edges, first, last)
        pinned_cells, pinned_faces = _pinned_positions(settings, axis, widths.size, 'hz')
        if not dipole:
            side = pinned_cells.get(cell)
            if side is not None:
                label, keep = _wall_text(settings, axis, side, edges, cell + 1, cell)
                messages.append(f'at {axis} = {position:.4g} m lies in the outermost cell, whose H_z {label} '
                                f'overwrites every step, so the magnetic current radiates nothing; {keep}')
            continue
        if shares[axis] <= 1e-12 * max(shares.values()):   # no current along this axis's faces
            continue
        faces = [face for face in (cell, cell - 1) if 0 <= face <= widths.size - 2]   # as placement.dipole_faces
        pinned = [face for face in faces if face in pinned_faces]
        if not pinned:
            continue
        component = 'E_y' if axis == 'x' else 'E_x'
        lost = 'all' if len(pinned) == len(faces) else 'half'
        label, keep = _wall_text(settings, axis, pinned_faces[pinned[0]], edges, pinned[0] + 2, pinned[0])
        messages.append(f'at {axis} = {position:.4g} m drives {component} on the inner face of the outermost cell, '
                        f'which {label} pins to 0 every step, so {lost} of the dipole {component} current radiates '
                        f'nothing; {keep}')
    return messages


def _ez_pinned_cell_message(mesh, settings, axis, position, source_text):
    """Problem text when `position` along `axis` lies in an outermost cell whose E_z a wall BC overwrites (else None)."""
    widths, edges, first, last = _window_axes(mesh)['xy'.index(axis)]
    cell = containing_cell(position, edges, first, last)
    side = _pinned_positions(settings, axis, widths.size, 'ez')[0].get(cell)
    if side is None:
        return None
    label, keep = _wall_text(settings, axis, side, edges, cell + 1, cell)
    return (f'at {axis} = {position:.4g} m lies in the outermost cell, whose E_z {label} overwrites every step, so '
            f'{source_text} radiates nothing; {keep}')


def validate_ez_point_feature(vctx, physics_spec, feature_spec, values):
    """validate_points_inside plus a warning for an E_z point source in an outermost cell a wall BC overwrites."""
    settings = raw_settings(physics_spec)

    def warnings(shape):
        messages = (_ez_pinned_cell_message(vctx.mesh, settings, axis, position, 'the filament current')
                    for axis, position in zip('xy', _shape_corners(shape)[0]))
        return [('warning', message) for message in messages if message is not None]
    return (validate_points_inside(vctx, physics_spec, feature_spec, values)
            + _shape_issues(vctx, physics_spec, feature_spec, 'point', warnings))


def validate_ez_line_feature(vctx, physics_spec, feature_spec, values):
    """_line_issues and _source_line_issues plus a warning for an E_z sheet whose line cells lie in an outermost cell
    a wall BC overwrites."""
    settings = raw_settings(physics_spec)

    def warnings(shape):
        line_axes = _line_axes(shape.values, vctx.domain_size)
        if line_axes is None:
            return []   # reported by _line_issues
        _, normal_axis, position, _ = line_axes
        message = _ez_pinned_cell_message(vctx.mesh, settings, normal_axis, position, 'the sheet current')
        return [] if message is None else [('warning', message)]
    return (_line_issues(vctx, physics_spec, feature_spec)
            + _source_line_issues(vctx, physics_spec, feature_spec, values, 'ez')
            + _shape_issues(vctx, physics_spec, feature_spec, 'line', warnings))


def conducting_sheet_line_issues(vctx, physics_spec, feature_spec, polarization):
    """conducting_sheet line checks: _line_issues; hz error on the outer grid wall (placement.sheet_line refuses it);
    infos for a sheet a wall BC pins (no effect) and for span ends continuing through a CPML (an infinite sheet)."""
    settings = raw_settings(physics_spec)
    size = vctx.domain_size

    def messages(shape):
        line_axes = _line_axes(shape.values, size)
        if line_axes is None:
            return []   # reported by _line_issues
        along_axis, normal_axis, position, span = line_axes
        found = []
        widths, edges, first, last = _window_axes(vctx.mesh)['xy'.index(normal_axis)]
        line_cell = containing_cell(position, edges, first, last)
        pinned_cells, pinned_faces = _pinned_positions(settings, normal_axis, widths.size, polarization)
        if polarization == 'hz':
            if line_cell == widths.size - 1 and settings.get(f'bc_{normal_axis}_hi') != 'periodic':
                found.append(('error', f'at {normal_axis} = {position:.4g} m puts the sheet (on the hi face of the line '
                                       f'cells) on the outer grid wall; keep {normal_axis} < {edges[line_cell]:.4g} m'))
            side = pinned_faces.get(line_cell)
            where = 'its face (the hi face of the line cells)'
        else:
            side = pinned_cells.get(line_cell)
            where = 'its line cells'
        if side is not None:
            label, keep = _wall_text(settings, normal_axis, side, edges, line_cell + 1, line_cell)
            found.append(('info', f'at {normal_axis} = {position:.4g} m: {label} pins the field on {where}, so the '
                                  f'sheet has no effect; {keep}'))
        continued = [side for side, coord in (('lo', span[0]), ('hi', span[1]))
                     if _end_continues_into_pml(vctx.mesh, settings, along_axis, side, coord, size)]
        if continued:
            found.append(('info', f"continues through the {' and '.join(f'{along_axis}_{side}' for side in continued)} "
                                  'PML (its end lies in the outermost window cell next to a cpml face), so it acts as '
                                  'a sheet without an end there'))
        return found
    return (_line_issues(vctx, physics_spec, feature_spec)
            + _shape_issues(vctx, physics_spec, feature_spec, 'line', messages))


def validate_flux_monitor(vctx, physics_spec, feature_spec, values):
    """FDTD flux monitor: _line_issues, TFSF / mode-source interface warnings, the wavelength range."""
    return (_line_issues(vctx, physics_spec, feature_spec)
            + monitor_line_interface_issues(vctx, physics_spec, feature_spec, time_domain=True)
            + wavelength_range_issues(values, feature_path(physics_spec, feature_spec)))


# ---- mode lines ----
# polarisation -> (out-of-plane field, transverse partner on the faces across a line of normal axis 'x' / 'y')
_MODE_FIELDS = {'ez': ('E_z', {'x': 'H_y', 'y': 'H_x'}), 'hz': ('H_z', {'x': 'E_y', 'y': 'E_x'})}
# mode feature role -> (name of its face, consequence of a wall-pinned row, consequence of a cell-width change)
_MODE_ROLES = {
    'source': ('interface', 'the launch is neither complete nor one-sided', 'the one-sided launch leaks backward'),
    'monitor': ('measuring', 'the decomposition reads the wall value instead of the field',
                'the forward / backward separation degrades'),
}
_MODE_WIDTH_SPREAD = 0.01   # largest relative width change along the normal over the line cell and its neighbours
# mode feature role -> consequence of its face inside a CPML layer (no psi term in the launch, unstretched mode)
_MODE_STRETCHED_CONSEQUENCES = {
    'source': 'the one-sided launch ignores, so it leaks backward and misses power',
    'monitor': 'the decomposition ignores, so its forward / backward powers are wrong',
}
# (_MODE_ROLES, _MODE_WIDTH_SPREAD, _MODE_STRETCHED_CONSEQUENCES: runtime twins in mode_ports.PortLine
# .placement_warnings.)


def bloch_period_line_issues(vctx, physics_spec, feature_spec, bloch_axes):
    """FDFD errors for a port line along a Bloch axis (k != 0, in bloch_axes) spanning its whole period: fdfd_ports
    refuses it (the periodic slab mode would need the Bloch phase)."""
    def messages(shape):
        cells = _line_cells(vctx.mesh, shape.values, vctx.domain_size)
        if cells is None:
            return []   # reported by _line_issues
        normal_axis, _, (span_lo, span_hi) = cells
        along_axis = 'y' if normal_axis == 'x' else 'x'
        n_along = dict(zip('xy', _window_axes(vctx.mesh)))[along_axis][0].size
        if along_axis not in bloch_axes or span_lo != 0 or span_hi != n_along:
            return []
        return [('error', f'spans the whole {along_axis} period with Bloch k{along_axis} != 0, where no port mode is '
                          f'available (it would need the Bloch phase); shorten the line to the guide, or set '
                          f'bloch_k{along_axis} = 0')]
    return _shape_issues(vctx, physics_spec, feature_spec, 'line', messages)


def make_validate_mode_source(polarization):
    """validate callback of a mode_source in `polarization` ('ez' | 'hz'): mode_line_issues with the interface face
    on the lo face of the line cell (direction positive) or its hi face (negative)."""
    def validate_mode_source(vctx, physics_spec, feature_spec, values):
        face_side = 'lo' if values['direction'] == 'positive' else 'hi'
        return mode_line_issues(vctx, physics_spec, feature_spec, polarization, 'source', face_side)
    return validate_mode_source


def make_validate_mode_monitor(polarization, time_domain=True):
    """validate callback of a mode_monitor in `polarization`: mode_line_issues (measuring face = hi face of the line
    cell), TFSF / mode-source interface warnings, the wavelength range (time_domain: FDTD params), and an info unless
    exactly one mode_source is enabled."""
    def validate_mode_monitor(vctx, physics_spec, feature_spec, values):
        path = feature_path(physics_spec, feature_spec)
        issues = mode_line_issues(vctx, physics_spec, feature_spec, polarization, 'monitor', 'hi')
        issues += monitor_line_interface_issues(vctx, physics_spec, feature_spec, time_domain)
        if time_domain:
            issues += wavelength_range_issues(values, path)
        n_sources = sum(1 for feature in physics_spec.features if feature.enabled and feature.type == 'mode_source')
        if n_sources != 1:
            issues.append(Issue('info', path, f"the physics has {n_sources or 'no'} enabled mode_source features, so "
                                              'this monitor reports power_positive / power_negative only '
                                              '(fraction_* and phase_* need exactly one mode source)'))
        return issues
    return validate_mode_monitor


_FOCUSED_BEAM_END_LEVEL = 1e-6   # beam |u| / its peak on the line above which a line end truncates it (warning)


def make_validate_focused_beam(polarization):
    """validate callback of an FDFD focused_beam in `polarization`: the line axis-aligned inside the domain; errors
    first for what the runtime refuses (focused_beam.focused_beam_line / FocusedBeamPhasor: periodic axes, a mirror
    face normal to the beam axis, a reached mirror face of the wrong parity or a focus off it, fewer than
    MIN_SPAN_CELLS or non-uniform cells along the line, rows within NORMAL_RANGE of the window edge or non-uniform
    across it, several media on those rows, a numerical aperture above the line index); warnings for a beam above
    _FOCUSED_BEAM_END_LEVEL of its peak at an open line end (the continuum beam at the shortest and longest solved
    wavelength) and a focus off the span; info for a focus upstream and a mirror end; the eigen-solved warning."""
    def validate_focused_beam(vctx, physics_spec, feature_spec, values):
        issues = _line_issues(vctx, physics_spec, feature_spec) + eigen_only_feature_issues(vctx, physics_spec,
                                                                                            feature_spec)
        settings = raw_settings(physics_spec)
        physics_type = vctx.registry.physics_type(physics_spec.type)
        wavelengths = resolved_solved_wavelengths(vctx, physics_spec, physics_type)
        return issues + _shape_issues(
            vctx, physics_spec, feature_spec, 'line', lambda shape: _focused_beam_messages(
                vctx, physics_spec, shape.values, settings, polarization, values, physics_type, wavelengths),
            needs_mesh=False)
    return validate_focused_beam


def _line_medium_index(vctx, along_axis, position, span, physics_type):
    """Static index of the material the preview finds at the middle of a line (vacuum 1), NaN when unknown."""
    material_map = vctx.material_map()
    if material_map is None:
        return math.nan
    image, _, extent = material_map
    along_index = 'xy'.index(along_axis)
    coords = [0.0, 0.0]
    coords[along_index], coords[1 - along_index] = 0.5 * (span[0] + span[1]), position
    pixel = tuple(preview_pixel(coords[axis], extent[2 * axis], extent[2 * axis + 1], image.shape[axis])
                  for axis in (0, 1))
    index = int(image[pixel])
    if index == 0:
        return 1.0
    return _refractive_index(vctx.spec.materials[index - 1], vctx.parameters, [physics_type.material_properties])


_FOCUSED_BEAM_REFUSED_PROPERTIES = ('eps_xy', 'gyration_z')   # off-diagonal media the beam's plane waves ignore
_FOCUSED_BEAM_REFUSED_FEATURES = {'pec_domain': 'PEC', 'conducting_sheet': 'conducting sheet'}


def _focused_beam_row_messages(vctx, physics_spec, along_axis, position, span, neighbours, normal_range):
    """Errors for several media or an eps_xy / gyration_z material on a focused beam's line cells and the rows at
    `neighbours` (material preview); warnings for a pec_domain / conducting_sheet object of the physics whose bounding
    box reaches the band span x normal_range [m] (the run refuses PEC or a sheet on those cells)."""
    messages = []
    found = _line_material_indices(vctx, along_axis, position, span, None, 'ez', neighbours)
    if found is not None:
        indices, names = found
        media = _distinct_media(vctx, indices, names)
        if media:
            messages.append(('error', f"crosses different media {', '.join(media)} on its cells or the "
                                      f'{NORMAL_RANGE} rows either side: the beam is computed for one homogeneous '
                                      'medium there; move the line into a uniform region'))
        for index in sorted(index for index in indices if index > 0):
            material = vctx.spec.materials[index - 1]
            try:
                properties, _ = _effective_material(material)
            except ValueError:   # reported at the material
                continue
            for name in _FOCUSED_BEAM_REFUSED_PROPERTIES:   # a field-dependent value counts as non-zero
                if evaluates_to(properties.get(name), vctx.parameters, lambda value: value != 0.0):
                    messages.append(('error', f"has material '{material.name}' with {name} on its rows: the beam is a "
                                              'sum of plane waves of a diagonal medium; keep the line in an isotropic, '
                                              'unrotated, unmagnetised medium'))
    along_index = 'xy'.index(along_axis)
    band = [None, None]
    band[along_index], band[1 - along_index] = tuple(sorted(span)), tuple(normal_range)
    for feature in physics_spec.features:
        kind = _FOCUSED_BEAM_REFUSED_FEATURES.get(feature.type)
        if kind is None or not feature.enabled:
            continue
        for name in vctx.objects(feature.selection):
            if vctx.shapes.get(name) is None:
                continue   # reported at that feature
            x0, y0, x1, y1 = bounding_box(name, vctx.shapes)
            box = ((min(x0, x1), max(x0, x1)), (min(y0, y1), max(y0, y1)))
            if all(lo <= band_hi and band_lo <= hi for (lo, hi), (band_lo, band_hi) in zip(box, band)):
                messages.append(('warning', f"may put {kind} cells of '{feature.name}' (object '{name}') on its line "
                                            f'or the {NORMAL_RANGE} rows either side, which the run refuses (the beam '
                                            'needs one dielectric medium there); keep the object away from the line'))
    return messages


def _focused_beam_messages(vctx, physics_spec, line, settings, polarization, values, physics_type, wavelengths):
    """(level, text) of one focused-beam line inside the domain (make_validate_focused_beam); the cell checks need
    the mesh estimate, the beam checks the material preview."""
    size = vctx.domain_size
    line_axes = _line_axes(line, size)
    if line_axes is None:
        return []   # reported by _line_issues
    along_axis, normal_axis, position, (start, end) = line_axes
    messages = []
    for axis in (along_axis, normal_axis):
        if settings.get(f'bc_{axis}_lo') == 'periodic':
            messages.append(('error', f'the {axis} axis is periodic, which a focused beam does not support (its '
                                      'Gaussian spectrum is not periodic); use cpml or wall faces on that axis'))
    for side in ('lo', 'hi'):
        if settings.get(f'bc_{normal_axis}_{side}') in MIRROR_SIGNS:
            messages.append(('error', f'bc_{normal_axis}_{side} is a mirror face normal to the beam axis {normal_axis}: '
                                      'its image beam would travel the other way; use cpml or a wall face there'))
    focus = (values['focus_x'], values['focus_y'])
    focus_along, focus_normal = focus['xy'.index(along_axis)], focus['xy'.index(normal_axis)]
    sign = 1.0 if values['direction'] == 'positive' else -1.0
    mesh = vctx.mesh
    tensor = mesh is not None and getattr(mesh, 'kind', 'tensor') == 'tensor'   # triangles: refused by the grids
    axes = dict(zip('xy', _window_axes(mesh))) if tensor else None
    along_size = size['xy'.index(along_axis)]
    mirror_ends = set()
    for side, face, problem in _focused_beam_mirror_ends(mesh, settings, along_axis, normal_axis, polarization,
                                                         (start, end), size):
        if problem is not None:
            messages.append(('error', problem))
            continue
        mirror_ends.add(side)
        if axes is not None:   # the meshed window edge (a uniform mesh may round the size up) and its cell
            widths, edges, first, last = axes[along_axis]
            plane, cell = float(edges[first if side == 'lo' else last + 1]), float(widths[first])
        else:
            plane, cell = (0.0 if side == 'lo' else along_size), 1e-3 * along_size
        if abs(focus_along - plane) > FOCUS_TOLERANCE * cell:
            messages.append(('error', f'ends on the {face} mirror plane at {along_axis} = {plane:.4g} m, so the focus '
                                      f'must lie on it (focus {along_axis} = {focus_along:.6g} m): only then does the '
                                      'image complete the beam'))
        else:
            messages.append(('info', f'ends on the {face} mirror plane through the focus: its image completes the '
                                     'beam, and power counts both halves'))
    if axes is not None:
        along_widths, along_edges, along_first, along_last = axes[along_axis]
        span_lo = containing_cell(start, along_edges, along_first, along_last)
        span_hi = containing_cell(end, along_edges, along_first, along_last) + 1
        if span_hi - span_lo < MIN_SPAN_CELLS:
            messages.append(('error', f'covers {span_hi - span_lo} cell(s) along {along_axis}; a focused beam needs at '
                                      f'least {MIN_SPAN_CELLS} (and should span the whole beam)'))
        elif not has_uniform_spacing(along_widths[span_lo:span_hi]):
            messages.append(('error', f'has non-uniform cells along {along_axis} ('
                                      f'{float(np.min(along_widths[span_lo:span_hi])):.4g} .. '
                                      f'{float(np.max(along_widths[span_lo:span_hi])):.4g} m): the beam is a sum of '
                                      'plane waves of uniform cells; keep the line in a uniform part of the mesh'))
        widths, edges, first, last = axes[normal_axis]
        line_cell = containing_cell(position, edges, first, last)
        rows = np.arange(line_cell - NORMAL_RANGE, line_cell + NORMAL_RANGE + 1)
        if rows[0] < first or rows[-1] > last:
            messages.append(('error', f'lies within {NORMAL_RANGE} cells of the {normal_axis} window edge: the launch '
                                      f'reads {NORMAL_RANGE} uniform cells on either side of the line; move it inside'))
        elif not has_uniform_spacing(widths[rows]):
            messages.append(('error', f'lies where the cell widths along {normal_axis} change ('
                                      f'{float(widths[rows].min()):.4g} .. {float(widths[rows].max()):.4g} m over the '
                                      f'{NORMAL_RANGE} cells either side): the one-sided launch leaks upstream on '
                                      'graded rows; keep the line in a uniform part of the mesh'))
        else:
            neighbours = [0.5 * (edges[row] + edges[row + 1]) for row in rows if row != line_cell]
            messages += _focused_beam_row_messages(vctx, physics_spec, along_axis, position, (start, end),
                                                   neighbours, (float(edges[rows[0]]), float(edges[rows[-1] + 1])))
    aperture_error, end_messages = _focused_beam_end_messages(vctx, along_axis, position, (start, end), focus, sign,
                                                              values, physics_type, wavelengths, mirror_ends)
    if aperture_error is not None:
        messages.append(('error', aperture_error))
    return messages + end_messages + upstream_focus_messages(sign, focus_normal, position, normal_axis)


def _focused_beam_mirror_ends(mesh, settings, along_axis, normal_axis, polarization, span, size):
    """[(side, face, parity error text or None)] of the ends of a 2D focused-beam line on a mirror plane (in the
    outermost window cell of the mesh estimate `mesh`; None: on the domain edge)."""
    polarization_axis = 'z' if polarization == 'ez' else along_axis   # the incident E
    ends = []
    for side, coord in zip(('lo', 'hi'), span):
        if not _end_on_mirror(mesh, settings, along_axis, side, coord, size):
            continue
        face = f'{along_axis}_{side}'
        problem = None
        if plane_wave_mirror_issue(face, MIRROR_SIGNS[settings[f'bc_{face}']], normal_axis, polarization_axis):
            wanted = ('symmetric' if plane_wave_mirror_issue(face, 1, normal_axis, polarization_axis) is None
                      else 'antisymmetric')
            problem = (f'ends on the {face} mirror plane, about which the beam (E along {polarization_axis}) is even, '
                       f'so bc_{face} must be {wanted}; else end the line at least one cell before the plane')
        ends.append((side, face, problem))
    return ends


def _focused_beam_end_messages(vctx, along_axis, position, span, focus, sign, values, physics_type, wavelengths,
                               mirror_ends):
    """(numerical_aperture error text or None, [(level, text)] of the line-end and focus-off-the-line warnings) of a
    2D focused-beam line for the drawing-frame focus (x, y) [m]: the continuum beam at focused_beam_end_wavelengths,
    sides in mirror_ends exempt; no line-end warning after the aperture error (or with an unknown line index)."""
    start, end = span
    focus_along, focus_normal = focus['xy'.index(along_axis)], focus[1 - 'xy'.index(along_axis)]
    index = _line_medium_index(vctx, along_axis, position, span, physics_type)
    aperture = values['numerical_aperture']
    aperture_error = None
    messages = []
    if aperture is not None and math.isfinite(index) and aperture > index:
        aperture_error = (f'has numerical_aperture {aperture:g} above the index {index:.4g} of the medium at the line '
                          '(NA = n sin(theta) <= n); lower it')
    elif math.isfinite(index):
        worst = {}
        for wavelength in focused_beam_end_wavelengths(wavelengths):
            levels = continuum_end_levels(2.0 * math.pi * float(C0) / wavelength, index, values['waist'], aperture,
                                          span, position, focus_along, focus_normal, sign)
            for side, level in zip(('lo', 'hi'), levels):
                if side not in mirror_ends and math.isfinite(level):
                    worst[side] = max(worst.get(side, 0.0), level)
        for side, coord in (('lo', start), ('hi', end)):
            if worst.get(side, 0.0) > _FOCUSED_BEAM_END_LEVEL:
                messages.append(('warning', f'stops at {along_axis} = {coord:.4g} m where the beam is still '
                                            f'{worst[side]:.2g} of its peak on the line (above '
                                            f'{_FOCUSED_BEAM_END_LEVEL:g}; continuum beam at the solved wavelengths), '
                                            'so the cut edge launches a diffracted wave and light leaks upstream; '
                                            'lengthen the line'))
    if not start <= focus_along <= end:
        messages.append(('warning', f'has its focus at {along_axis} = {focus_along:.4g} m, off the line ({start:.4g} '
                                    f'to {end:.4g} m), so the line launches only the edge of the beam; put the focus '
                                    'over the line'))
    return aperture_error, messages


# ---- public helpers of brillouin_scattering validation (physics/mechanics/definition_brillouin) ----
_FOCUSED_BEAM_POLARIZATIONS = {'em_fdfd_ez': 'ez', 'em_fdfd_hz': 'hz', 'em_fdfd_bor': None}   # None: the BOR beam
_DIRECTION_SIGNS = {'positive': 1, 'negative': -1}


@dataclass(frozen=True)
class FocusedBeamLineBound:
    """A focused_beam line as validation sees it (focused_beam_lines): feature and line object names, the normal axis
    'x' | 'y' of the drawing frame (the beam axis), the direction sign +1 / -1 along it (None when the direction does
    not resolve), the line position along the normal [m] and total_field_from [m]: the coordinate along the normal from
    which every drive-field sample and its interpolation stencil is total field (the downstream edge of the line
    cell; None without a tensor mesh estimate or a sign). Coordinates in the drawing frame (window lower edge 0)."""
    feature: str
    line: str
    normal_axis: str
    sign: int | None
    position: float
    total_field_from: float | None


def _focused_beam_features(vctx, physics_spec):
    """[(feature spec, its FeatureType)] of the enabled focused_beam features of an em_fdfd_ez / _hz / em_fdfd_bor
    physics spec; ValueError for another physics type."""
    if physics_spec.type not in _FOCUSED_BEAM_POLARIZATIONS:
        raise ValueError(f"focused beams belong to {', '.join(_FOCUSED_BEAM_POLARIZATIONS)}, not to physics "
                         f"'{physics_spec.name}' of type {physics_spec.type}")
    physics_type = vctx.registry.physics_type(physics_spec.type)
    return [(feature, physics_type.feature_type(feature.type)) for feature in physics_spec.features
            if feature.enabled and feature.type == FOCUSED_BEAM_FEATURE]


def _resolved_lines(vctx, feature):
    """[(shape, (along axis, normal axis, position, (start, end)))] of the axis-aligned line objects of a feature
    inside the domain (the others are reported at the feature)."""
    size = vctx.domain_size
    if size is None:
        return []
    lines = []
    for shape in _selected_shapes(vctx, feature, 'line'):
        line_axes = _line_axes(shape.values, size)
        if line_axes is not None and not _outside_message(_shape_corners(shape), size):
            lines.append((shape, line_axes))
    return lines


def focused_beam_lines(vctx, physics_spec, mesh) -> list:
    """[FocusedBeamLineBound] of the enabled focused_beam features of an optical physics spec (em_fdfd_ez / _hz /
    em_fdfd_bor; ValueError for another type), one per axis-aligned line object inside the domain. mesh = THAT
    physics' mesh estimate (vctx.meshes of its mesh; never vctx.mesh of a caller validating another physics):
    total_field_from takes the line cell by the runtime rule (containing_cell on its window edges) and is its
    downstream edge: from there on the line row's own values (ez / BOR E_r, u at the row centre), the faces past it
    (hz E_x / E_y from the curl of two total-field cells; BOR E_z on its z edges) and every interpolation between
    neighbouring Yee points are total field, never the upstream row or the interface face (2D hz: the curl across
    it mixes a scattered-field cell)."""
    tensor = mesh is not None and getattr(mesh, 'kind', 'tensor') == 'tensor'
    axes = dict(zip('xy', _window_axes(mesh))) if tensor else None
    lines = []
    for feature, feature_type in _focused_beam_features(vctx, physics_spec):
        params = feature.params if isinstance(feature.params, dict) else {}
        direction_default = next(param.default for param in feature_type.params if param.name == 'direction')
        direction = params.get('direction', direction_default)
        sign = _DIRECTION_SIGNS.get(direction) if isinstance(direction, str) else None
        for shape, (_, normal_axis, position, _) in _resolved_lines(vctx, feature):
            bound = None
            if axes is not None and sign is not None:
                _, edges, first, last = axes[normal_axis]
                cell = containing_cell(position, edges, first, last)
                bound = float(edges[cell + 1] if sign > 0 else edges[cell])
            lines.append(FocusedBeamLineBound(feature=feature.name, line=shape.name, normal_axis=normal_axis,
                                              sign=sign, position=float(position), total_field_from=bound))
    return lines


def focused_beam_scan_messages(vctx, physics_spec, feature_spec, mesh, offset=(0.0, 0.0)) -> list:
    """[(level, "line '<name>' <text>")] of the focus-dependent warnings of one focused_beam feature (beam above its
    end level at an open line end; 2D also a focus off the line) for its focus moved by the drawing-frame offset
    (dx, dy) [m] (em_fdfd_bor: (0, dz); a radial offset gives [], it is refused elsewhere): the same continuum beam,
    thresholds and wavelengths (focused_beam_end_wavelengths) as the beam's own validation, whose messages are these at
    offset (0, 0). brillouin_scattering validation calls it once per scan position. mesh = the optical physics' own
    mesh estimate (as focused_beam_lines); [] when the feature values do not resolve (reported at the beam), no line
    warning after a numerical_aperture error (reported at the beam)."""
    dx, dy = (float(value) for value in offset)
    feature_type = next((found for feature, found in _focused_beam_features(vctx, physics_spec)
                         if feature is feature_spec), None)
    if feature_type is None:
        raise ValueError(f"'{feature_spec.name}' is not an enabled focused_beam of physics '{physics_spec.name}'")
    polarization = _FOCUSED_BEAM_POLARIZATIONS[physics_spec.type]
    try:
        values = resolve_values(feature_type.params, feature_spec.params if isinstance(feature_spec.params, dict)
                                else {}, vctx.parameters)
    except ValueError:   # reported at the beam
        return []
    focus_names = ('focus_z',) if polarization is None else ('focus_x', 'focus_y')
    if values['direction'] not in _DIRECTION_SIGNS or any(values[name] is None for name in focus_names):
        return []   # reported at the beam
    physics_type = vctx.registry.physics_type(physics_spec.type)
    wavelengths = resolved_solved_wavelengths(vctx, physics_spec, physics_type)
    tensor_mesh = mesh if mesh is not None and getattr(mesh, 'kind', 'tensor') == 'tensor' else None
    messages = []
    for shape, (along_axis, normal_axis, position, span) in _resolved_lines(vctx, feature_spec):
        if polarization is None:   # em_fdfd_bor (lazy: definition_bor imports this module)
            from .definition_bor import bor_focused_beam_scan_messages
            found = bor_focused_beam_scan_messages(vctx, along_axis, position, span, values, physics_type,
                                                   wavelengths, (dx, dy))
        else:
            sign = float(_DIRECTION_SIGNS[values['direction']])
            mirror_ends = {side for side, _, problem in _focused_beam_mirror_ends(
                tensor_mesh, raw_settings(physics_spec), along_axis, normal_axis, polarization, span,
                vctx.domain_size) if problem is None}
            _, found = _focused_beam_end_messages(vctx, along_axis, position, span,
                                                  (values['focus_x'] + dx, values['focus_y'] + dy), sign, values,
                                                  physics_type, wavelengths, mirror_ends)
        messages.extend((level, f"line '{shape.name}' {text}") for level, text in found)
    return messages


def focused_beam_end_wavelengths(wavelengths):
    """The solved wavelengths [m] a focused beam's line-end check samples (2D and BOR): the shortest and the longest,
    ascending; () without any."""
    return tuple(sorted({min(wavelengths), max(wavelengths)})) if wavelengths else ()


def upstream_focus_messages(sign, focus_normal, position, axis_name):
    """[info] when a focused beam toward sign (+1 / -1 along axis_name) has its focus upstream of its line at
    `position` [m] (2D and BOR), else []."""
    if not sign * (focus_normal - position) < 0.0:
        return []
    return [('info', f'has its focus upstream of the line ({axis_name} = {focus_normal:.4g} m): the beam diverges from '
                     'a virtual focus')]


def mode_line_issues(vctx, physics_spec, feature_spec, polarization, role, face_side):
    """_line_issues plus the mode-line checks of _mode_line_messages on the mesh estimate (`role` 'source' / 'monitor',
    its face on the face_side 'lo' / 'hi' face of the line cell)."""
    settings = raw_settings(physics_spec)
    return _line_issues(vctx, physics_spec, feature_spec) + _shape_issues(
        vctx, physics_spec, feature_spec, 'line', lambda shape: _mode_line_messages(
            vctx.mesh, shape.values, vctx.domain_size, settings, polarization, role, face_side))


def _mode_line_messages(mesh, line, size, settings, polarization, role, face_side):
    """(level, text) of one mode line inside the domain: errors for a wall-cell-only span and a missing / never-updated
    face; warnings for a wall-pinned line cell or face, a face inside a CPML layer and > 1% normal width changes."""
    # cells as the runtime looks them up: containing cell clamped to the window, the span [cell(start), cell(end) + 1)
    line_axes = _line_axes(line, size)
    if line_axes is None:
        return []   # reported by _line_issues
    along_axis, normal_axis, position, (span_start, span_end) = line_axes
    axes = dict(zip('xy', _window_axes(mesh)))
    messages = []
    along_widths, along_edges, along_first, along_last = axes[along_axis]
    span_lo = containing_cell(span_start, along_edges, along_first, along_last)
    span_hi = containing_cell(span_end, along_edges, along_first, along_last) + 1
    lo_type, hi_type = settings.get(f'bc_{along_axis}_lo', 'cpml'), settings.get(f'bc_{along_axis}_hi', 'cpml')
    # a span end on the grid edge (a face without PML) of a non-periodic axis leaves the wall cell out of the solve
    solve_lo = span_lo + 1 if span_lo == 0 and lo_type != 'periodic' else span_lo
    solve_hi = span_hi - 1 if span_hi == along_widths.size and hi_type != 'periodic' else span_hi
    if solve_hi <= solve_lo:
        side, wall_type = ('lo', lo_type) if span_lo == 0 else ('hi', hi_type)
        messages.append(('error', f'spans only the outermost cell at the bc_{along_axis}_{side} = {wall_type} wall, '
                                  f'which the mode solve leaves out (the wall pins it), so no cell is left to solve; '
                                  f'lengthen the line along {along_axis}'))
    face_name, pinned_consequence, spread_consequence = _MODE_ROLES[role]
    out_of_plane, partners = _MODE_FIELDS[polarization]
    partner = partners[normal_axis]
    widths, edges, first, last = axes[normal_axis]
    n_cells = widths.size
    line_cell = containing_cell(position, edges, first, last)
    face = line_cell - 1 if face_side == 'lo' else line_cell
    periodic = settings.get(f'bc_{normal_axis}_hi') == 'periodic'   # faces wrap; periodic axes pin nothing
    where = f'at {normal_axis} = {position:.4g} m'
    if not periodic and not 0 <= face <= n_cells - 2:   # face -1 does not exist, the last face is never updated
        _, keep = _wall_text(settings, normal_axis, face_side, edges, line_cell + 1, line_cell)
        messages.append(('error', f'{where} puts its {face_name} face ({partner}, the {face_side} face of the line '
                                  f'cell) on the outer grid wall, which is never updated; {keep}'))
    elif not periodic:
        pinned_cells, pinned_faces = _pinned_positions(settings, normal_axis, n_cells, polarization)
        cell_side, face_pinned_side = pinned_cells.get(line_cell), pinned_faces.get(face)
        pinned = [text for text, side in ((f'{out_of_plane} of the line cell', cell_side),
                                          (f'{partner} on its {face_name} face', face_pinned_side)) if side]
        if pinned:
            label, keep = _wall_text(settings, normal_axis, cell_side or face_pinned_side, edges, line_cell + 1,
                                     line_cell)
            messages.append(('warning', f'{where} has {" and ".join(pinned)} pinned every step by the wall '
                                        f'{label}, so {pinned_consequence}; {keep}'))
        # CPML H rows (boundaries.cpml_axis_profiles): lo faces 0..first-1, hi faces last..n-2 (face k = hi face of
        # cell k), so the face of a line in the outermost window cell next to a cpml face is stretched.
        stretched_side = next((side for side, inside in (('lo', face < first), ('hi', face >= last))
                               if inside and settings.get(f'bc_{normal_axis}_{side}', 'cpml') == 'cpml'), None)
        if stretched_side is not None:
            _, keep = _wall_text(settings, normal_axis, stretched_side, edges, line_cell + 1, line_cell)
            messages.append(('warning', f'{where} lies in the outermost window cell, so its {face_name} face '
                                        f'({partner}) is inside the bc_{normal_axis}_{stretched_side} = cpml layer, '
                                        f'whose stretch {_MODE_STRETCHED_CONSEQUENCES[role]}; {keep}'))
    neighbours = np.arange(line_cell - 1, line_cell + 2)
    neighbours = neighbours % n_cells if periodic else neighbours[(neighbours >= 0) & (neighbours < n_cells)]
    local_widths = widths[neighbours]
    spread = float(local_widths.max() / local_widths.min()) - 1.0
    if spread > _MODE_WIDTH_SPREAD:
        messages.append(('warning', f'{where} lies where the cell widths along {normal_axis} change by '
                                    f'{100.0 * spread:.2g}% around the line cell (more than '
                                    f'{100.0 * _MODE_WIDTH_SPREAD:g}%): the mode is matched to one spacing, so '
                                    f'{spread_consequence}; place the line on a locally uniform mesh'))
    return messages


# ---- boxes: TFSF, DFT, far field ----
def _tfsf_issues(vctx, physics_spec, feature_spec, harmonic):
    """Box inside the domain, one scattered-field cell on every side (FDFD: or every y cell of a periodic y axis), the
    box outline in vacuum (FDTD) or in one uniform medium (FDFD) (warning)."""
    path = feature_path(physics_spec, feature_spec)
    settings = raw_settings(physics_spec)
    issues, inside = _box_issues(vctx, physics_spec, feature_spec)
    for shape in inside:
        full_period_y = (harmonic and vctx.mesh is not None
                         and _tfsf_spans_period_y(vctx.mesh, shape.values, settings))
        open_faces, mirror_problems = _tfsf_open_faces(vctx.mesh, shape.values, physics_spec, settings)
        issues.extend(Issue('error', path, f"TFSF box '{shape.name}' {problem}") for problem in mirror_problems)
        message = (_tfsf_margin_message(vctx.mesh, shape.values, full_period_y, open_faces)
                   if vctx.mesh is not None and not mirror_problems else None)
        if message:
            if harmonic and settings.get('bc_y_lo') == 'periodic':
                message += ' (or cover every cell of the periodic y axis: y0 = 0 and y1 = the domain height)'
            issues.append(Issue('error', path, f"TFSF box '{shape.name}' {message}"))
        if open_faces:
            issues.append(Issue('info', path, f"TFSF box '{shape.name}' opens onto the mirror plane(s) "
                                              f"{', '.join(sorted(open_faces))}: its image completes the box there"))
        edge_materials = _tfsf_edge_materials(vctx, shape.values, uniform=harmonic, full_period_y=full_period_y,
                                              open_faces=open_faces)
        if edge_materials and harmonic:
            issues.append(Issue('warning', path, f"TFSF box '{shape.name}' edge crosses different media "
                                                 f"{', '.join(edge_materials)}: the incident wave is computed for "
                                                 f"one uniform background (the medium beside its lo-x edge), so "
                                                 f"incident light leaks into the scattered field"))
        elif edge_materials:
            issues.append(Issue('warning', path, f"TFSF box '{shape.name}' edge crosses non-vacuum material "
                                                 f"{', '.join(edge_materials)}: the incident field is computed for "
                                                 f"vacuum, so incident light leaks into the scattered field"))
    return issues


def validate_tfsf(vctx, physics_spec, feature_spec, values):
    """FDTD TFSF box: inside the domain, one scattered-field cell on every side, vacuum along its outline (warning)."""
    return _tfsf_issues(vctx, physics_spec, feature_spec, harmonic=False)


def validate_harmonic_tfsf(vctx, physics_spec, feature_spec, values):
    """FDFD TFSF box: as validate_tfsf, but it may cover a periodic y axis and needs one uniform outline medium; an
    oblique wave (Bloch ky != 0 on periodic y) needs uniform y cells (error on a graded y axis)."""
    issues = _tfsf_issues(vctx, physics_spec, feature_spec, harmonic=True)
    settings = raw_settings(physics_spec)
    if vctx.mesh is None or settings.get('bc_y_lo') != 'periodic' or settings.get('bc_y_hi') != 'periodic':
        return issues
    try:
        bloch_ky = evaluate_scalar(settings.get('bloch_ky', 0.0), vctx.parameters)
    except (ValueError, TypeError):   # reported at the setting
        return issues
    if bloch_ky != 0.0 and not has_uniform_spacing(vctx.mesh.dy_cells):
        dy_cells = np.asarray(vctx.mesh.dy_cells, dtype=np.float64)
        issues.append(Issue('error', feature_path(physics_spec, feature_spec),
                            f'an oblique plane wave (bloch_ky = {bloch_ky:.4g} rad/m) needs uniform y cells, but the '
                            f'mesh grades y ({dy_cells.min():.4g} .. {dy_cells.max():.4g} m): exp(-j ky y) is no Bloch '
                            'wave of graded rows, so incident light would leak; refine along x only (refinement dx_y '
                            'equal to the base cell, or a vertical refinement line) or set bloch_ky = 0'))
    return issues


_DFT_WARN_BYTES = 2 ** 30


def make_validate_dft_monitor(polarization):
    """validate callback of a dft_monitor in `polarization`: box inside the domain, the wavelength range, and a warning
    when its spectra on the mesh estimate exceed 1 GiB of GPU memory."""
    n_spectra = _DFT_SPECTRA[polarization]

    def validate_dft_monitor(vctx, physics_spec, feature_spec, values):
        path = feature_path(physics_spec, feature_spec)
        issues, inside = _box_issues(vctx, physics_spec, feature_spec)
        if vctx.mesh is not None:
            axes = _window_axes(vctx.mesh)
            grid_cells = axes[0][0].size * axes[1][0].size
            stride = values['downsample']
            for shape in inside:
                box_cells = 1
                for axis, (_, edges, first, last) in zip('xy', axes):   # as placement.box_cells
                    lo, hi = sorted((shape.values[f'{axis}0'], shape.values[f'{axis}1']))
                    axis_cells = containing_cell(hi, edges, first, last) + 1 - containing_cell(lo, edges, first, last)
                    box_cells *= -(-(axis_cells - min(stride, axis_cells) // 2) // stride)   # monitors.dft_sample_slice
                spectrum_values = n_spectra * values['n_wavelengths'] * box_cells
                if 8 * spectrum_values > _DFT_WARN_BYTES:
                    issues.append(Issue('warning', path, f"box '{shape.name}' ({box_cells} recorded cells) needs "
                                                         f'{8 * spectrum_values / 2 ** 30:.3g} GiB of GPU memory for '
                                                         f"{values['n_wavelengths']} wavelengths, and each step "
                                                         f'accumulates {spectrum_values / (3 * grid_cells):.2g}x the '
                                                         f'values of the field update; shrink the box, use fewer '
                                                         f'wavelengths or a larger downsample'))
        return issues + wavelength_range_issues(values, path)
    return validate_dft_monitor


def validate_far_field(vctx, physics_spec, feature_spec, values, time_domain=True):
    """Box inside the domain with one cell to spare beyond x1 / y1, warnings for TFSF / mode-source interfaces it
    samples across and for non-vacuum material on its outline (the projection is for vacuum), at most one mirror face
    per axis (the pattern unfolds over it), valid wavelength range (time_domain: FDTD params and outline faces)."""
    path = feature_path(physics_spec, feature_spec)
    issues, inside = _box_issues(vctx, physics_spec, feature_spec)
    for shape in inside if vctx.mesh is not None else ():
        message = _far_field_margin_message(vctx.mesh, shape.values)
        if message:
            issues.append(Issue('error', path, f"far-field box '{shape.name}' {message}"))
        issues += _far_field_interface_issues(vctx, physics_spec, shape, path, time_domain)
        outline_materials = _tfsf_edge_materials(vctx, shape.values)
        if outline_materials:
            issues.append(Issue('warning', path, f"far-field box '{shape.name}' outline crosses non-vacuum material "
                                                 f"{', '.join(outline_materials)}: the projection assumes vacuum "
                                                 f"outside the box, so the pattern and its power are off; enclose "
                                                 f"the structure in vacuum"))
    settings = raw_settings(physics_spec)
    for axis in 'xy':
        if all(settings.get(f'bc_{axis}_{side}') in MIRROR_SIGNS for side in ('lo', 'hi')):
            issues.append(Issue('error', path, f'mirror faces on both {axis} sides make an infinite array of images, '
                                               'which the far field cannot unfold: keep a mirror on one side at most'))
    if time_domain:
        issues += wavelength_range_issues(values, path)
    return issues
