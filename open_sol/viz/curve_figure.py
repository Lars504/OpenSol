"""Article-quality export of a 2D curve plot (pyplot-free matplotlib Figure): PlotStyle (fonts, ticks, line widths,
legend; the GUI preview follows it on screen) and save_curve_figure of plain curve data at a given physical size,
optionally over a SceneData (RGBA images, texts, a colour bar: the GUI canvas)."""
import math
from dataclasses import asdict, dataclass, fields

import numpy as np

TICK_DIRECTIONS = ('in', 'out')
LEGEND_LOCATIONS = ('best', 'upper right', 'upper left', 'lower left', 'lower right', 'outside right', 'none')
SCALE_BAR_LOCATIONS = ('lower left', 'lower right')
_CHOICES = {'tick_direction': TICK_DIRECTIONS, 'legend_location': LEGEND_LOCATIONS,
            'scale_bar_location': SCALE_BAR_LOCATIONS}
_MINOR_TICK_LENGTH = 2.0 / 3.0   # minor / major tick length (pyqtgraph's second tick level, gui preview)
# scale bar geometry in pt, shared by the GUI preview and the export (scale_bar_layout)
SCALE_BAR_MARGIN = 9.0       # corner of the shown content (else of the view) to the scale bar's box
SCALE_BAR_PAD = 3.0          # box edge to its content
SCALE_BAR_GAP = 2.0          # a bar to its text
_SCALE_BAR_ZORDER = 1.0e6    # above every scene layer (their z orders count the overlay items)


@dataclass
class PlotStyle:
    """Look of an exported curve plot: sizes in pt; '' / 0 fields mean automatic."""
    font_family: str = ''            # '' = matplotlib's default sans-serif
    title: str = ''
    x_label: str = ''                # '' = the plot's own label (units are appended either way)
    y_label: str = ''
    label_size: float = 10.0
    tick_label_size: float = 9.0
    title_size: float = 11.0
    si_prefix: bool = True           # linear axes: values in an SI-prefixed unit (MHz) instead of 1e6 ticks
    axis_width: float = 0.8
    tick_width: float = 0.8
    tick_length: float = 3.5
    tick_direction: str = 'in'
    minor_ticks: bool = True
    boxed: bool = True               # frame with ticks on all four sides
    grid: bool = False
    grid_alpha: float = 0.25
    foreground: str = '#000000'      # axes, ticks and text
    background: str = '#ffffff'
    legend_location: str = 'best'
    legend_size: float = 8.0
    legend_frame: bool = False
    legend_columns: int = 1
    scale_bar: bool = True           # scene plots (the GUI canvas) only: a scale bar replaces the axes
    scale_bar_location: str = 'lower right'
    scale_bar_length: float = 25.0   # % of the view side the bar may span (the longest 1-2-5 length within it)
    scale_bar_width: float = 2.0     # bar thickness
    scale_bar_line_color: str = ''   # '' = scale_bar_text_color (bar_color())
    scale_bar_font_size: float = 10.0
    scale_bar_bold: bool = True      # thin grey text fades over a dark field
    scale_bar_text_color: str = '#808080'   # mid grey: readable over dark and light fields without a box
    scale_bar_box: bool = False      # background-coloured box behind the bar and its text
    dpi: int = 600                  # raster formats only
    transparent: bool = False

    def bar_color(self) -> str:
        """Colour of the scale bar itself: its own, else the scale bar text colour."""
        return self.scale_bar_line_color or self.scale_bar_text_color

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data) -> 'PlotStyle':
        """PlotStyle of the known keys of a stored dict; a value of the wrong type or choice keeps the default."""
        style = cls()
        for style_field in fields(cls):
            value = (data or {}).get(style_field.name)
            default = getattr(style, style_field.name)
            if value is None or isinstance(value, bool) != isinstance(default, bool):
                continue
            try:
                value = type(default)(value)
            except (TypeError, ValueError):
                continue
            if value in _CHOICES.get(style_field.name, (value,)):
                setattr(style, style_field.name, value)
        return style


@dataclass(frozen=True, eq=False)
class CurveData:
    """One plotted curve: data in axis units, matplotlib line style ('' = no line) and marker ('' = none), line width
    and marker size in pt."""
    x: np.ndarray
    y: np.ndarray
    label: str | None
    color: str
    line_style: str | tuple = '-'
    marker: str = ''
    marker_color: str = ''
    line_width: float = 1.5
    marker_size: float = 4.0
    marker_edge_color: str = ''      # '' = marker_color
    zorder: float | None = None      # None = matplotlib's default


@dataclass(frozen=True, eq=False)
class ImageLayer:
    """RGBA image (rows from the lowest y up, floats 0..1) over extent (x0, x1, y0, y1) in data units."""
    rgba: np.ndarray
    extent: tuple
    zorder: float = 0.0


@dataclass(frozen=True)
class TextLabel:
    """Text at (x, y) in data units: matplotlib ha / va, angle in degrees, size in pt (0 = tick label size),
    background colour ('' = none)."""
    x: float
    y: float
    text: str
    color: str
    ha: str = 'left'
    va: str = 'top'
    angle: float = 0.0
    size: float = 0.0
    background: str = ''
    zorder: float = 3.0


@dataclass(frozen=True)
class AxisTicks:
    """Fixed ticks of one axis in data units with their texts ('' = unlabelled); minor ticks are shorter."""
    major: tuple
    major_labels: tuple
    minor: tuple = ()
    minor_labels: tuple = ()


@dataclass(frozen=True, eq=False)
class ColorbarData:
    """Colour bar beside the axes: lookup table (N, 4) floats 0..1 from low to high (log: spaced in log10), fixed
    ticks with their texts (None: the tick step, 0 = automatic; linear bars only)."""
    lut: np.ndarray
    low: float
    high: float
    tick_step: float = 0.0
    log: bool = False
    ticks: AxisTicks | None = None


@dataclass(frozen=True)
class ScaleBarData:
    """Scale bar drawn instead of the axes: a horizontal bar x_length long (data units) with x_text and, where the y
    scale differs, a vertical bar y_length long with y_text (y_length 0: one bar, the x and y scales are equal);
    content = (x0, x1, y0, y1) in data units of what the view shows (the bar sits in its corner), None = the view."""
    x_length: float
    x_text: str
    y_length: float = 0.0
    y_text: str = ''
    content: tuple | None = None


@dataclass(frozen=True, eq=False)
class SceneData:
    """Non-curve content of a plot: images, texts, an optional colour bar, an equal data aspect and a scale bar
    replacing the axes (place and box from the PlotStyle)."""
    images: tuple = ()
    texts: tuple = ()
    colorbar: ColorbarData | None = None
    equal_aspect: bool = False
    scale_bar: ScaleBarData | None = None


@dataclass(frozen=True)
class AxesData:
    """Axis texts (units included), log flags, view ranges [min, max] in data units, the data -> shown factors and
    fixed ticks (None = matplotlib's automatic ticks)."""
    x_label: str
    y_label: str
    x_log: bool = False
    y_log: bool = False
    x_range: tuple | None = None
    y_range: tuple | None = None
    x_scale: float = 1.0
    y_scale: float = 1.0
    x_ticks: AxisTicks | None = None
    y_ticks: AxisTicks | None = None


def scale_bar_layout(line_width, x_bar, x_text_size, y_bar=0.0, y_text_size=(0.0, 0.0)):
    """Scale bar box in pt, origin at its lower left corner, y up: ((width, height), bar centre lines [(x0, y0, x1,
    y1)], text centres [(x, y)]) of a horizontal bar x_bar long and line_width thick under its text (width, height);
    with y_bar > 0 a vertical bar from the same corner too, the texts then below / left of their bars (the vertical
    one's size is given before its quarter turn)."""
    half_line = 0.5 * line_width
    x_text_width, x_text_height = x_text_size
    y_text_width, y_text_height = y_text_size
    if y_bar <= 0.0:   # one bar along the bottom, its text centred above it
        bars = [(0.0, half_line, x_bar, half_line)]
        centres = [(0.5 * x_bar, line_width + SCALE_BAR_GAP + 0.5 * x_text_height)]
        left, right = min(0.0, 0.5 * (x_bar - x_text_width)), max(x_bar, 0.5 * (x_bar + x_text_width))
        bottom, top = 0.0, line_width + SCALE_BAR_GAP + x_text_height
    else:   # two bars from one corner, each text outside it: below the horizontal bar, left of the vertical one
        corner_x = y_text_height + SCALE_BAR_GAP + half_line   # centre line of the vertical bar
        corner_y = x_text_height + SCALE_BAR_GAP + half_line   # centre line of the horizontal bar
        # the horizontal bar starts half a line early so the two bars meet in a square corner
        bars = [(corner_x - half_line, corner_y, corner_x + x_bar, corner_y),
                (corner_x, corner_y, corner_x, corner_y + y_bar)]
        centres = [(corner_x + 0.5 * x_bar, 0.5 * x_text_height), (0.5 * y_text_height, corner_y + 0.5 * y_bar)]
        left = min(0.0, centres[0][0] - 0.5 * x_text_width)
        right = max(corner_x + x_bar, centres[0][0] + 0.5 * x_text_width)
        bottom = min(0.0, centres[1][1] - 0.5 * y_text_width)
        top = max(corner_y + y_bar, centres[1][1] + 0.5 * y_text_width)
    shift_x, shift_y = SCALE_BAR_PAD - left, SCALE_BAR_PAD - bottom
    return ((right - left + 2.0 * SCALE_BAR_PAD, top - bottom + 2.0 * SCALE_BAR_PAD),
            [(x0 + shift_x, y0 + shift_y, x1 + shift_x, y1 + shift_y) for x0, y0, x1, y1 in bars],
            [(x + shift_x, y + shift_y) for x, y in centres])


def scale_bar_origin(location, box_size, view_size, content=None):
    """Lower left corner of a scale bar box, in pt from the view's lower left corner (sizes (width, height) in pt):
    in the `location` bottom corner of content (x0, x1, y0, y1 in the same pt: what the view shows, e.g. the field
    without the empty margin around it) cut to the view, or of the whole view without content or where the box does
    not fit in it."""
    if location not in SCALE_BAR_LOCATIONS:
        raise ValueError(f'scale bar location {location!r} is not one of {SCALE_BAR_LOCATIONS}')
    x0, x1, y0 = 0.0, view_size[0], 0.0
    if content is not None:
        inner_x0, inner_x1 = max(content[0], 0.0), min(content[1], view_size[0])
        inner_y0, inner_y1 = max(content[2], 0.0), min(content[3], view_size[1])
        if (inner_x1 - inner_x0 >= box_size[0] + 2.0 * SCALE_BAR_MARGIN
                and inner_y1 - inner_y0 >= box_size[1] + 2.0 * SCALE_BAR_MARGIN):
            x0, x1, y0 = inner_x0, inner_x1, inner_y0
    x = x0 + SCALE_BAR_MARGIN if location == 'lower left' else x1 - SCALE_BAR_MARGIN - box_size[0]
    return x, y0 + SCALE_BAR_MARGIN


def save_curve_figure(path, curves, axes, style, size_inches, scene=None):
    """Draw curves (over the images / texts / colour bar of scene) on one axes with style at exactly size_inches
    (width, height; constrained layout) and save; the format follows the suffix (pdf, svg, eps, png, ...), raster
    formats at style.dpi; fonts stay text in pdf / eps / svg. A scene scale bar replaces the axes (ticks, labels,
    frame, grid); a single bar holds for both directions, so a data limit is widened until the scales are equal."""
    from matplotlib import rc_context
    from matplotlib.figure import Figure
    foreground = style.foreground
    rc = {'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'font.size': style.label_size,
          'axes.edgecolor': foreground, 'axes.labelcolor': foreground, 'text.color': foreground,
          'xtick.color': foreground, 'ytick.color': foreground}
    if style.font_family:
        rc['font.family'] = style.font_family
    with rc_context(rc):
        figure = Figure(figsize=size_inches, dpi=style.dpi, layout='constrained')
        figure.patch.set_facecolor(style.background)
        ax = figure.add_subplot()
        ax.set_facecolor(style.background)
        scene = scene or SceneData()
        for layer in scene.images:
            x0, x1, y0, y1 = layer.extent
            ax.imshow(layer.rgba, origin='lower', interpolation='nearest', zorder=layer.zorder,
                      extent=(x0 * axes.x_scale, x1 * axes.x_scale, y0 * axes.y_scale, y1 * axes.y_scale))
        legend_entries = []   # explicit (line, label): matplotlib's own legend drops labels starting with '_'
        for curve in curves:
            z_options = {} if curve.zorder is None else {'zorder': curve.zorder}
            (line,) = ax.plot(np.asarray(curve.x, dtype=np.float64) * axes.x_scale,
                              np.asarray(curve.y, dtype=np.float64) * axes.y_scale,
                              linestyle=curve.line_style or 'None', color=curve.color, linewidth=curve.line_width,
                              marker=curve.marker or 'None', markersize=curve.marker_size,
                              markerfacecolor=curve.marker_color or curve.color,
                              markeredgecolor=curve.marker_edge_color or curve.marker_color or curve.color,
                              **z_options)
            if curve.label:
                legend_entries.append((line, curve.label))
        for label in scene.texts:
            bbox = {'facecolor': label.background, 'edgecolor': 'none', 'pad': 1.0} if label.background else None
            ax.text(label.x * axes.x_scale, label.y * axes.y_scale, label.text, color=label.color, ha=label.ha,
                    va=label.va, rotation=label.angle, rotation_mode='anchor', zorder=label.zorder, clip_on=True,
                    fontsize=label.size or style.tick_label_size, bbox=bbox)
        scale_bar = scene.scale_bar
        if scale_bar is not None and (axes.x_log or axes.y_log):
            raise ValueError('a scale bar needs linear axes')
        _style_axes(ax, axes, style, frame=scale_bar is None)
        ax.set_aspect('equal' if scene.equal_aspect else 'auto')
        if scale_bar is not None and scale_bar.y_length <= 0.0:
            # one bar for both directions: equal scales, reached by widening a data limit (the axes box stays, so
            # the colour bar still spans it)
            ax.set_aspect(axes.x_scale / axes.y_scale, adjustable='datalim')
        if scene.colorbar is not None:
            _add_colorbar(figure, ax, scene.colorbar, style)
        _add_legend(figure, ax, style, legend_entries)
        if scale_bar is not None:
            _add_scale_bar(figure, ax, scale_bar, axes, style)   # last: it needs the final layout
        figure.savefig(path, dpi=style.dpi, transparent=style.transparent,
                       facecolor='none' if style.transparent else style.background)


def _style_axes(ax, axes, style, frame=True):
    """Scales, limits, ticks, spines, grid, labels and title of the single axes; frame False (a scale bar replaces
    the axes) keeps only the limits and the title."""
    from matplotlib.ticker import FixedFormatter, FixedLocator
    for log, set_scale in ((axes.x_log, ax.set_xscale), (axes.y_log, ax.set_yscale)):
        if log:
            set_scale('log', nonpositive='mask')
    for value_range, scale, set_limits in ((axes.x_range, axes.x_scale, ax.set_xlim),
                                           (axes.y_range, axes.y_scale, ax.set_ylim)):
        if value_range is not None and all(math.isfinite(value) for value in value_range):
            set_limits(value_range[0] * scale, value_range[1] * scale)
    if style.title:
        ax.set_title(style.title, fontsize=style.title_size)
    if not frame:
        ax.tick_params(which='both', bottom=False, top=False, left=False, right=False, labelbottom=False,
                       labelleft=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        return
    for axis, ticks, scale in ((ax.xaxis, axes.x_ticks, axes.x_scale), (ax.yaxis, axes.y_ticks, axes.y_scale)):
        if ticks is None:
            continue
        axis.set_major_locator(FixedLocator(np.asarray(ticks.major, dtype=np.float64) * scale))
        axis.set_major_formatter(FixedFormatter(list(ticks.major_labels)))
        axis.set_minor_locator(FixedLocator(np.asarray(ticks.minor, dtype=np.float64) * scale))
        axis.set_minor_formatter(FixedFormatter(list(ticks.minor_labels)))
    ax.tick_params(which='both', direction=style.tick_direction, width=style.tick_width, top=style.boxed,
                   right=style.boxed, labelsize=style.tick_label_size, colors=style.foreground)
    ax.tick_params(which='major', length=style.tick_length)
    ax.tick_params(which='minor', length=style.tick_length * _MINOR_TICK_LENGTH)
    for side, spine in ax.spines.items():
        spine.set_linewidth(style.axis_width)
        spine.set_color(style.foreground)
        spine.set_visible(style.boxed or side in ('left', 'bottom'))
    if style.grid:
        ax.grid(True, which='major', alpha=style.grid_alpha, linewidth=0.5 * style.axis_width,
                color=style.foreground)
    ax.set_xlabel(axes.x_label, fontsize=style.label_size)
    ax.set_ylabel(axes.y_label, fontsize=style.label_size)


def _add_scale_bar(figure, ax, scale_bar, axes, style):
    """Scale bar (scale_bar_layout) in the style's corner of the axes. Drawn in axes fractions after a layout pass,
    so the bar lengths follow the final limits and axes size; its artists stay out of the layout."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    figure.draw_without_rendering()   # constrained layout and the aspect: the final axes box and limits
    axes_box = ax.get_window_extent()
    pt_per_px = 72.0 / figure.dpi
    view_size = (axes_box.width * pt_per_px, axes_box.height * pt_per_px)
    if not (view_size[0] > 0.0 and view_size[1] > 0.0):
        raise ValueError(f'scale bar: the axes are {view_size[0]:g} x {view_size[1]:g} pt after the layout')
    (x_low, x_high), (y_low, y_high) = ax.get_xlim(), ax.get_ylim()
    if not (x_high > x_low and y_high > y_low):
        raise ValueError(f'scale bar: needs rising axis limits, got x {x_low:g} to {x_high:g}, y {y_low:g} to '
                         f'{y_high:g}')
    x_bar = scale_bar.x_length * axes.x_scale / (x_high - x_low) * view_size[0]
    y_bar = scale_bar.y_length * axes.y_scale / (y_high - y_low) * view_size[1]
    has_y_bar = scale_bar.y_length > 0.0
    text_options = dict(transform=ax.transAxes, ha='center', va='center', fontsize=style.scale_bar_font_size,
                        fontweight='bold' if style.scale_bar_bold else 'normal',
                        color=style.scale_bar_text_color, zorder=_SCALE_BAR_ZORDER + 1.0)
    texts = [ax.text(0.5, 0.5, scale_bar.x_text, **text_options)]
    extent = texts[0].get_window_extent()
    x_text_size = (extent.width * pt_per_px, extent.height * pt_per_px)
    y_text_size = (0.0, 0.0)
    if has_y_bar:
        texts.append(ax.text(0.5, 0.5, scale_bar.y_text, rotation=90.0, **text_options))
        extent = texts[1].get_window_extent()   # of the turned text: its height is the text's length
        y_text_size = (extent.height * pt_per_px, extent.width * pt_per_px)
    box_size, bars, centres = scale_bar_layout(style.scale_bar_width, x_bar, x_text_size,
                                               y_bar if has_y_bar else 0.0, y_text_size)
    content = None
    if scale_bar.content is not None:   # data units -> pt from the axes' lower left corner, at the final limits
        content_x0, content_x1, content_y0, content_y1 = scale_bar.content
        x_pt_per_unit, y_pt_per_unit = view_size[0] / (x_high - x_low), view_size[1] / (y_high - y_low)
        content = ((content_x0 * axes.x_scale - x_low) * x_pt_per_unit,
                   (content_x1 * axes.x_scale - x_low) * x_pt_per_unit,
                   (content_y0 * axes.y_scale - y_low) * y_pt_per_unit,
                   (content_y1 * axes.y_scale - y_low) * y_pt_per_unit)
    origin = scale_bar_origin(style.scale_bar_location, box_size, view_size, content)

    def axes_fraction(x, y):   # pt from the box's lower left corner -> axes fractions
        return (origin[0] + x) / view_size[0], (origin[1] + y) / view_size[1]

    added = []
    if style.scale_bar_box:
        added.append(Rectangle(axes_fraction(0.0, 0.0), box_size[0] / view_size[0], box_size[1] / view_size[1],
                               transform=ax.transAxes, facecolor=style.background, edgecolor='none',
                               zorder=_SCALE_BAR_ZORDER))
    for x0, y0, x1, y1 in bars:
        (start_x, start_y), (end_x, end_y) = axes_fraction(x0, y0), axes_fraction(x1, y1)
        added.append(Line2D([start_x, end_x], [start_y, end_y], transform=ax.transAxes, color=style.bar_color(),
                            linewidth=style.scale_bar_width, solid_capstyle='butt',
                            zorder=_SCALE_BAR_ZORDER + 1.0))
    for artist in added:
        ax.add_artist(artist)
    for text, centre in zip(texts, centres):
        text.set_position(axes_fraction(*centre))
    for artist in (*added, *texts):
        artist.set_in_layout(False)   # savefig lays the figure out again: the same axes box as measured here


def _add_colorbar(figure, ax, colorbar, style):
    """Colour bar of the lookup table beside the axes, ticks and outline in the axes style; fixed ticks keep their
    texts (no offset text above the bar)."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import ListedColormap, LogNorm, Normalize
    norm = (LogNorm if colorbar.log else Normalize)(colorbar.low, colorbar.high)
    mappable = ScalarMappable(norm=norm, cmap=ListedColormap(np.asarray(colorbar.lut, dtype=np.float64)))
    bar = figure.colorbar(mappable, ax=ax)
    ticks = colorbar.ticks
    if ticks is not None:
        from matplotlib.ticker import FixedFormatter, FixedLocator
        bar.locator = FixedLocator(np.asarray(ticks.major, dtype=np.float64))
        bar.formatter = FixedFormatter(list(ticks.major_labels))
        bar.minorlocator = FixedLocator(np.asarray(ticks.minor, dtype=np.float64))
        bar.minorformatter = FixedFormatter(list(ticks.minor_labels))
    elif colorbar.tick_step > 0.0 and not colorbar.log:
        from matplotlib.ticker import MultipleLocator
        bar.ax.yaxis.set_major_locator(MultipleLocator(colorbar.tick_step))
    bar.ax.tick_params(which='both', direction=style.tick_direction, width=style.tick_width,
                       labelsize=style.tick_label_size, colors=style.foreground)
    bar.ax.tick_params(which='major', length=style.tick_length)
    bar.ax.tick_params(which='minor', length=style.tick_length * _MINOR_TICK_LENGTH)
    bar.outline.set_linewidth(style.axis_width)
    bar.outline.set_edgecolor(style.foreground)


def _add_legend(figure, ax, style, legend_entries):
    """Legend of the labelled curves [(line, label)] at style.legend_location ('outside right': beside the axes, in
    the figure)."""
    if style.legend_location == 'none' or not legend_entries:
        return
    handles, labels = zip(*legend_entries)
    options = dict(handles=list(handles), labels=list(labels), fontsize=style.legend_size,
                   frameon=style.legend_frame, ncols=max(1, style.legend_columns))
    if style.legend_location == 'outside right':
        legend = figure.legend(loc='outside right upper', **options)
    else:
        legend = ax.legend(loc=style.legend_location, **options)
    if style.legend_frame:
        legend.get_frame().set_edgecolor(style.foreground)
        legend.get_frame().set_linewidth(style.axis_width)
