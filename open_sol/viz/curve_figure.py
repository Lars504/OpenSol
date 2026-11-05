"""Article-quality export of a 2D curve plot (pyplot-free matplotlib Figure): PlotStyle (fonts, ticks, line widths,
legend; the GUI preview follows it on screen) and save_curve_figure of plain curve data at a given physical size,
optionally over a SceneData (RGBA images, texts, a colour bar: the GUI canvas)."""
import math
from dataclasses import asdict, dataclass, fields

import numpy as np

TICK_DIRECTIONS = ('in', 'out')
LEGEND_LOCATIONS = ('best', 'upper right', 'upper left', 'lower left', 'lower right', 'outside right', 'none')
_CHOICES = {'tick_direction': TICK_DIRECTIONS, 'legend_location': LEGEND_LOCATIONS}
_MINOR_TICK_LENGTH = 2.0 / 3.0   # minor / major tick length (pyqtgraph's second tick level, gui preview)


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
    dpi: int = 600                  # raster formats only
    transparent: bool = False

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


@dataclass(frozen=True, eq=False)
class ColorbarData:
    """Colour bar beside the axes: lookup table (N, 4) floats 0..1 from low to high (log: spaced in log10), tick step
    (0 = automatic; linear bars only)."""
    lut: np.ndarray
    low: float
    high: float
    tick_step: float = 0.0
    log: bool = False


@dataclass(frozen=True, eq=False)
class SceneData:
    """Non-curve content of a plot: images, texts, an optional colour bar and an equal data aspect."""
    images: tuple = ()
    texts: tuple = ()
    colorbar: ColorbarData | None = None
    equal_aspect: bool = False


@dataclass(frozen=True)
class AxisTicks:
    """Fixed ticks of one axis in data units with their texts ('' = unlabelled); minor ticks are shorter."""
    major: tuple
    major_labels: tuple
    minor: tuple = ()
    minor_labels: tuple = ()


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


def save_curve_figure(path, curves, axes, style, size_inches, scene=None):
    """Draw curves (over the images / texts / colour bar of scene) on one axes with style at exactly size_inches
    (width, height; constrained layout) and save; the format follows the suffix (pdf, svg, eps, png, ...), raster
    formats at style.dpi; fonts stay text in pdf / eps / svg."""
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
        _style_axes(ax, axes, style)
        ax.set_aspect('equal' if scene.equal_aspect else 'auto')
        if scene.colorbar is not None:
            _add_colorbar(figure, ax, scene.colorbar, style)
        _add_legend(figure, ax, style, legend_entries)
        figure.savefig(path, dpi=style.dpi, transparent=style.transparent,
                       facecolor='none' if style.transparent else style.background)


def _style_axes(ax, axes, style):
    """Scales, limits, ticks, spines, grid, labels and title of the single axes."""
    from matplotlib.ticker import FixedFormatter, FixedLocator
    for log, set_scale in ((axes.x_log, ax.set_xscale), (axes.y_log, ax.set_yscale)):
        if log:
            set_scale('log', nonpositive='mask')
    for value_range, scale, set_limits in ((axes.x_range, axes.x_scale, ax.set_xlim),
                                           (axes.y_range, axes.y_scale, ax.set_ylim)):
        if value_range is not None and all(math.isfinite(value) for value in value_range):
            set_limits(value_range[0] * scale, value_range[1] * scale)
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
    if style.title:
        ax.set_title(style.title, fontsize=style.title_size)


def _add_colorbar(figure, ax, colorbar, style):
    """Colour bar of the lookup table beside the axes, ticks and outline in the axes style."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import ListedColormap, LogNorm, Normalize
    norm = (LogNorm if colorbar.log else Normalize)(colorbar.low, colorbar.high)
    mappable = ScalarMappable(norm=norm, cmap=ListedColormap(np.asarray(colorbar.lut, dtype=np.float64)))
    bar = figure.colorbar(mappable, ax=ax)
    if colorbar.tick_step > 0.0 and not colorbar.log:
        from matplotlib.ticker import MultipleLocator
        bar.ax.yaxis.set_major_locator(MultipleLocator(colorbar.tick_step))
    bar.ax.tick_params(direction=style.tick_direction, width=style.tick_width, length=style.tick_length,
                       labelsize=style.tick_label_size, colors=style.foreground)
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
