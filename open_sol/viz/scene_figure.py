"""Perspective 3D picture of a model.scene_3d.Scene3D on a pyplot-free matplotlib Axes3D (view_3d C): translucent
material faces in one depth-sorted collection, rim edges, dashed composite inputs, point / line glyphs; optionally
field cut planes (FieldData-like, duck-typed) in one colour scale with the objects as edges only."""
import math

import numpy as np

from .common import color_limits, edges_from_centers, length_unit

DEFAULT_VIEW = (25.0, -60.0)       # (elevation, azimuth) [deg]
PLANE_CELLS = 4096                 # cells per cut plane above which every k-th cell is drawn (each cell is one quad)
_NO_MATERIAL_COLOR = '#b0b0b0'
_GLYPH_COLOR = '#555555'
_DOMAIN_EDGE_COLOR = '#7f7f7f'
_FACE_AXES = {'x': 0, 'y': 1, 'z': 2}


def new_scene_axes(figure):
    """Axes3D added to a matplotlib Figure (importing mplot3d registers the '3d' projection)."""
    import mpl_toolkits.mplot3d  # noqa: F401
    return figure.add_subplot(projection='3d')


def scene_figure(scene, width_in=6.0, height_in=5.0, dpi=100, **options):
    """New pyplot-free Figure holding draw_scene_3d(scene, **options) (scripts; savefig renders it with Agg)."""
    from matplotlib.figure import Figure
    figure = Figure(figsize=(width_in, height_in), dpi=dpi)
    draw_scene_3d(scene, new_scene_axes(figure), **options)
    return figure


def draw_scene_3d(scene, axes, material_colors=None, category_colors=None, highlight_color='#ffa000', z_stretch=1.0,
                  alpha=0.35, focal_length=1.5, view=DEFAULT_VIEW, field_planes=(), cmap=None, symmetric=None,
                  plane_cells=PLANE_CELLS):
    """Draw scene on an Axes3D in the length unit of its largest size; z_stretch scales the box height only;
    material_colors per material index (None: tab10), category_colors {feature category: colour} for glyphs.
    field_planes: FieldData cuts (plane set) drawn opaque in one colour scale with a colour bar, the objects then as
    edges only; cmap / symmetric None = the first plane's FieldInfo cmap / signed; plane_cells as PLANE_CELLS."""
    from matplotlib import colormaps
    from matplotlib.colors import to_rgb, to_rgba
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    scale, unit = length_unit(max(scene.size))
    size = np.asarray(scene.size, dtype=np.float64) / scale
    colors = list(material_colors) if material_colors is not None else list(colormaps['tab10'].colors)
    category_colors = category_colors or {}
    field_planes = list(field_planes or ())

    def material_color(index):
        return _NO_MATERIAL_COLOR if index is None else colors[index % len(colors)]

    whole_domain = 'all' in scene.highlighted_faces
    axes.add_collection3d(Line3DCollection(_box_edges(size), colors=highlight_color if whole_domain
                                           else _DOMAIN_EDGE_COLOR, linewidths=2.0 if whole_domain else 0.8))
    faces, face_colors = [], []
    edges, edge_colors, edge_widths = [], [], []
    dashed, dashed_colors = [], []
    for solid in scene.solids:
        color = highlight_color if solid.highlighted else material_color(solid.material)
        polylines = [polyline / scale for polyline in solid.edges]
        if not solid.solid:
            dashed.extend(polylines)
            dashed_colors.extend([color] * len(polylines))
            continue
        if not field_planes:   # with cut planes the objects are edges only (opaque planes, no face sorting errors)
            face_rgba = to_rgba(material_color(solid.material), min(1.0, 2.0 * alpha) if solid.highlighted else alpha)
            faces.extend(face / scale for face in solid.faces)
            face_colors.extend([face_rgba] * len(solid.faces))
        edges.extend(polylines)
        edge_colors.extend([color if solid.highlighted else tuple(0.6 * np.asarray(to_rgb(color)))] * len(polylines))
        edge_widths.extend([2.0 if solid.highlighted else 0.6] * len(polylines))
    for face in scene.highlighted_faces:
        if face != 'all':
            faces.append(_outer_face(face, size))
            face_colors.append(to_rgba(highlight_color, 0.3))
    if field_planes:
        plane_faces, plane_colors = _field_plane_faces(axes, field_planes, scale, cmap, symmetric, plane_cells)
        faces.extend(plane_faces)
        face_colors.extend(plane_colors)
        axes.computed_zorder = False   # by zorder, then add order: box / edges / glyphs always over the opaque planes
    if faces:   # one collection, so mplot3d depth-sorts every face against every other
        axes.add_collection3d(Poly3DCollection(faces, facecolors=face_colors, edgecolors='none'))
    if edges:
        axes.add_collection3d(Line3DCollection(edges, colors=edge_colors, linewidths=edge_widths))
    if dashed:
        axes.add_collection3d(Line3DCollection(dashed, colors=dashed_colors, linewidths=1.0, linestyles='dashed'))
    _draw_glyphs(axes, scene.glyphs, scale, category_colors, highlight_color)

    axes.set_xlim(0.0, size[0])
    axes.set_ylim(0.0, size[1])
    axes.set_zlim(0.0, size[2])
    axes.set_box_aspect((size[0], size[1], size[2] * z_stretch))
    axes.set_proj_type('persp', focal_length=focal_length)
    axes.set_xlabel(f'x ({unit})')
    axes.set_ylabel(f'y ({unit})')
    axes.set_zlabel(f'z ({unit})' if z_stretch == 1.0 else f'z ({unit}, stretched x{z_stretch:.3g})')
    axes.view_init(*view)
    if field_planes:   # the colour bar explains the colours; material edges need no legend
        return
    shown = sorted({solid.material for solid in scene.solids if solid.material is not None})
    handles = [Patch(facecolor=to_rgba(material_color(index), 0.6), edgecolor='none',
                     label=scene.material_names[index]) for index in shown]
    handles += [Patch(facecolor='none', edgecolor=material_color(index),
                      label=f'{scene.material_names[index]} (background)') for index in scene.background]
    if handles:
        axes.legend(handles=handles, loc='upper left', fontsize='small', frameon=False)


def _box_edges(size) -> list:
    """The 12 (2, 3) edges of the box [0, size]."""
    corners = np.array([[x, y, z] for x in (0.0, size[0]) for y in (0.0, size[1]) for z in (0.0, size[2])])
    return [corners[[first, second]] for first in range(8) for second in range(first + 1, 8)
            if np.count_nonzero(corners[first] != corners[second]) == 1]


def _outer_face(face, size) -> np.ndarray:
    """(4, 3) quad of an outer face name ('x_lo', ..., 'z_hi') of the box [0, size]."""
    axis = _FACE_AXES[face[0]]
    level = 0.0 if face.endswith('_lo') else size[axis]
    first, second = (other for other in range(3) if other != axis)
    quad = np.zeros((4, 3))
    quad[:, axis] = level
    quad[:, first] = (0.0, size[first], size[first], 0.0)
    quad[:, second] = (0.0, 0.0, size[second], size[second])
    return quad


def _field_plane_faces(axes, field_planes, scale, cmap, symmetric, plane_cells):
    """(quads, rgba colours) of every cut plane in one colour scale (color_limits over all planes), plus the colour
    bar and the field / time title on the axes' figure."""
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    info = field_planes[0].info
    signed = info.signed if symmetric is None else bool(symmetric)
    vmin, vmax = color_limits(np.concatenate([np.ravel(plane.values) for plane in field_planes]), signed)
    norm, colormap = Normalize(vmin, vmax), colormaps[cmap or info.cmap]
    quads, colors = [], []
    for plane in field_planes:
        plane_quads, plane_values = _plane_quads(plane, scale, plane_cells)
        quads.extend(plane_quads)
        colors.extend(colormap(norm(plane_values)))
    label = f'{info.label} ({info.unit})' if info.unit and info.unit != '1' else info.label
    axes.figure.colorbar(ScalarMappable(norm, colormap), ax=axes, shrink=0.6, pad=0.12, label=label)
    time_label = field_planes[0].time_label
    axes.set_title(f'{info.label}  |  {time_label}' if time_label else info.label, fontsize='medium')
    return quads, colors


def _plane_quads(plane, scale, plane_cells):
    """((N, 4, 3) quads in scene units, (N,) values) of the finite cells of one cut plane; above plane_cells cells
    every k-th cell (first value of each k x k block) over the exact edges of its native or uniform cells."""
    axis, position = plane.plane
    values = np.asarray(plane.values, dtype=np.float64)
    u_low, u_high, v_low, v_high = (float(edge) for edge in plane.extent)
    u_edges = (edges_from_centers(plane.x_centers, u_low, u_high) if plane.x_centers is not None
               else np.linspace(u_low, u_high, values.shape[0] + 1))
    v_edges = (edges_from_centers(plane.y_centers, v_low, v_high) if plane.y_centers is not None
               else np.linspace(v_low, v_high, values.shape[1] + 1))
    stride = max(1, math.ceil(math.sqrt(values.size / plane_cells)))
    if stride > 1:
        values = values[::stride, ::stride]
        u_edges = np.append(u_edges[:-1:stride], u_edges[-1])
        v_edges = np.append(v_edges[:-1:stride], v_edges[-1])
    u_lo, v_lo = np.meshgrid(u_edges[:-1], v_edges[:-1], indexing='ij')
    u_hi, v_hi = np.meshgrid(u_edges[1:], v_edges[1:], indexing='ij')
    u_index, v_index = (_FACE_AXES[name] for name in plane.axes)
    quads = np.empty(values.shape + (4, 3))
    quads[..., _FACE_AXES[axis]] = position
    for corner, (u_corner, v_corner) in enumerate(((u_lo, v_lo), (u_hi, v_lo), (u_hi, v_hi), (u_lo, v_hi))):
        quads[..., corner, u_index] = u_corner
        quads[..., corner, v_index] = v_corner
    finite = np.isfinite(values)
    return quads[finite] / scale, values[finite]


def _draw_glyphs(axes, glyphs, scale, category_colors, highlight_color):
    """Points as markers, lines as segments, coloured by feature category (highlighted: the highlight colour)."""
    points = [glyph for glyph in glyphs if len(glyph.vertices) == 1]
    if points:
        xyz = np.vstack([glyph.vertices for glyph in points]) / scale
        colors = [category_colors.get(glyph.category, _GLYPH_COLOR) for glyph in points]
        edge_colors = [highlight_color if glyph.highlighted else 'white' for glyph in points]
        sizes = [90.0 if glyph.highlighted else 36.0 for glyph in points]
        axes.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=sizes, edgecolors=edge_colors, linewidths=1.5,
                     depthshade=False)
    for glyph in glyphs:
        if len(glyph.vertices) > 1:
            xyz = glyph.vertices / scale
            color = highlight_color if glyph.highlighted else category_colors.get(glyph.category, _GLYPH_COLOR)
            axes.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, linewidth=3.0 if glyph.highlighted else 2.0)
