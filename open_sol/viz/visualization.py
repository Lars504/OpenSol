"""Matplotlib plots of open_sol simulations (pyplot-free Figures): geometry, mesh, fields and EM monitor results (FDFD:
annotations, plot_flux / plot_transmission, plot_radiation_pattern; FDFD probes have no time series)."""

import weakref
from dataclasses import dataclass, replace

import numpy as np

from ..core.constants import C0, FACES
from ..core.mesh import has_uniform_spacing
from ..core.renderer import primary_field_info
from .common import (color_limits, edges_from_centers, length_unit, normalised_db, probe_spectrum, safe_ratio,
                     time_unit)

def _new_figure(figsize):
    """A Figure outside pyplot: never switches the user's backend or joins pyplot's registry (garbage-collected when
    dropped); savefig renders with Agg."""
    from matplotlib.figure import Figure
    return Figure(figsize=figsize)


_STYLE = {
    'source_color': '#FF4444',
    'probe_color': '#44AA44',
    'dft_color': '#4488FF',
    'flux_color': '#FF8800',
    'mode_color': '#00A0A0',
    'nf2ff_color': '#AA44FF',
    'pec_color': '#808080',
    'conductor_color': '#2266CC',
    'material_cmap': 'YlOrBr',
    'intensity_cmap': 'inferno',
    'mesh_cmap': 'viridis_r',
    'dpi': 150,
    'figsize_single': (8, 6),
    'figsize_wide': (14, 5),
    'figsize_dashboard': (16, 12),
}

# EM BCType value -> (colour, alpha, label) of the face bar.
_EM_BC_STYLE = {
    'cpml': ('#333333', 0.25, 'PML'),
    'pec': ('#CC0000', 0.5, 'PEC'),
    'pmc': ('#0066CC', 0.5, 'PMC'),
    'periodic': ('#00AA00', 0.5, 'Periodic'),
    'bloch': ('#AA00AA', 0.5, 'Bloch'),
    'symmetric': ('#FF8800', 0.5, 'Sym'),
    'antisymmetric': ('#FF4400', 0.5, 'ASym'),
}

# Annotation kind -> Line2D legend style (dict order = legend order).
_LEGEND_STYLES = {
    'source': dict(marker='*', color='w', markerfacecolor=_STYLE['source_color'], markersize=10,
                   label='Point source'),
    'line_source': dict(color=_STYLE['source_color'], linestyle='--', linewidth=1.5, label='Line source'),
    'mode_source': dict(color=_STYLE['source_color'], linestyle='--', linewidth=3.0, label='Mode source'),
    'tfsf': dict(color=_STYLE['source_color'], linestyle='-.', linewidth=1.5, label='TFSF box'),
    'probe': dict(marker='o', color='w', markerfacecolor=_STYLE['probe_color'], markersize=6, label='Probe'),
    'flux': dict(color=_STYLE['flux_color'], linewidth=2.5, label='Flux monitor'),
    'mode_monitor': dict(color=_STYLE['mode_color'], linewidth=2.5, label='Mode monitor'),
    'dft': dict(color=_STYLE['dft_color'], linestyle=':', linewidth=1.5, label='DFT monitor'),
    'nf2ff': dict(color=_STYLE['nf2ff_color'], linestyle=':', linewidth=1.5, label='NF2FF box'),
    'pec': dict(marker='s', color='w', markerfacecolor=_STYLE['pec_color'], markersize=8, label='PEC'),
    'conductor': dict(color=_STYLE['conductor_color'], linestyle='--', linewidth=1.0, label='σ_e > 0'),
}

# Dashboard fields shown after every physics' primary field, in this order, when a renderer exposes them.
_SECONDARY_FIELDS = ('e_abs', 'T_rise', 'absorbed_power', 'heat_flux_abs', 'q_joule')

_MAX_DFT_PANELS = 8   # plot_dft_fields default panel cap (evenly spaced monitor frequencies)

DARK_CMAPS = frozenset({'inferno', 'magma', 'plasma', 'viridis', 'cividis'})   # overlays drawn light on these


@dataclass(frozen=True, eq=False)
class _Frame:
    """Plot geometry of a Domain: physical extent [m], axis length unit, centres / edges of ALL cells [m]."""
    extent: tuple
    scale: float
    unit: str
    x_centers: np.ndarray
    y_centers: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    window: tuple
    non_uniform: bool

    def scaled_extent(self):
        return tuple(value / self.scale for value in self.extent)

    def window_centers(self):
        sx, sy = self.window
        return self.x_centers[sx], self.y_centers[sy]

    def window_edges(self):
        sx, sy = self.window
        return self.x_edges[sx.start:sx.stop + 1], self.y_edges[sy.start:sy.stop + 1]

    def point(self, ix, iy):
        """Scaled (x, y) of the centre of cell (ix, iy)."""
        return self.x_centers[ix] / self.scale, self.y_centers[iy] / self.scale

    def segment(self, along, fixed_cell, span, hi_face=False):
        """Scaled (xs, ys) of a line along `along` through the centre (or the hi face) of cell `fixed_cell` over the
        half-open cell range `span` (None = whole grid line), clipped to the physical window."""
        x_min, x_max, y_min, y_max = self.extent
        if along == 'x':
            lo, hi = span if span is not None else (0, self.x_edges.size - 1)
            start, stop = max(self.x_edges[lo], x_min), min(self.x_edges[hi], x_max)
            y = (self.y_edges[fixed_cell + 1] if hi_face else self.y_centers[fixed_cell]) / self.scale
            return (start / self.scale, stop / self.scale), (y, y)
        if along == 'y':
            lo, hi = span if span is not None else (0, self.y_edges.size - 1)
            start, stop = max(self.y_edges[lo], y_min), min(self.y_edges[hi], y_max)
            x = (self.x_edges[fixed_cell + 1] if hi_face else self.x_centers[fixed_cell]) / self.scale
            return (x, x), (start / self.scale, stop / self.scale)
        raise ValueError(f"line axis must be 'x' or 'y', got {along!r}")

    def box(self, ix0, ix1, iy0, iy1):
        """Scaled (x0, y0, width, height) of the cell-edge box [ix0, ix1) x [iy0, iy1)."""
        x0, x1 = self.x_edges[ix0] / self.scale, self.x_edges[ix1] / self.scale
        y0, y1 = self.y_edges[iy0] / self.scale, self.y_edges[iy1] / self.scale
        return x0, y0, x1 - x0, y1 - y0

    def center_box(self, ix0, ix1, iy0, iy1):
        """Scaled (x0, y0, width, height) of the box through the centres of columns ix0, ix1 and rows iy0, iy1."""
        x0, x1 = self.x_centers[ix0] / self.scale, self.x_centers[ix1] / self.scale
        y0, y1 = self.y_centers[iy0] / self.scale, self.y_centers[iy1] / self.scale
        return x0, y0, x1 - x0, y1 - y0


# ------------------------------------------------------------------ module helpers

def _finish(fig, filename):
    """tight_layout, save to `filename` when given, return the figure."""
    fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=_STYLE['dpi'], bbox_inches='tight', facecolor='white')
        print(f"  Saved: {filename}")
    return fig


def _mesh_ratio_limits(mesh_config, axis):
    """[(name, ratio)] limiting adjacent cells along `axis`: its max_ratio, plus outer_ratio where outer graded; [] for
    a uniform grid or a custom axis (its cells are given, not marched)."""
    if mesh_config is None or axis in mesh_config.custom_axes():
        return []
    limits = [('max_ratio', mesh_config.ratio(axis))]
    outer_ratio = mesh_config.axis_outer_ratio(axis)
    if mesh_config.outer_core(axis) is not None and outer_ratio != limits[0][1]:
        limits.append(('outer_ratio', outer_ratio))
    return limits


def _extent_length_unit(extent):
    """(scale, unit) for axes spanning extent (x0, x1, y0, y1) [m]."""
    x0, x1, y0, y1 = extent
    return length_unit(max(x1 - x0, y1 - y0))


WAVELENGTH_UNIT = (1e-9, 'nm')   # (scale [m], unit) of every wavelength axis and label, as in the GUI


def _field_label(info):
    """Colour-bar label of a FieldInfo: 'label (unit)', unit dropped when dimensionless."""
    return _with_unit(info.label, info.unit)


def _field_title(renderer, info):
    time_label = renderer.time_label()
    return f'{info.label}  |  {time_label}' if time_label else info.label


def _primary_component(renderer):
    """Component flagged primary in renderer.field_infos, else the first one (ValueError when it has none)."""
    info = primary_field_info(renderer.field_infos.values())
    if info is None:
        raise ValueError(f'{type(renderer).__name__} has no field components')
    return info.name


def _contour_color(cmap):
    return 'white' if cmap in DARK_CMAPS else 'black'


def _draw_cells(ax, values, extent, scale, cmap, vmin, vmax, x_edges=None, y_edges=None):
    """Draw a (nx, ny) cell array with x horizontal over its cell edges [m] (None: uniform cells over the extent) by
    pcolorfast: an image whose cost follows the output pixels, exact cell boundaries on graded edges."""
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"expected a 2D (nx, ny) array, got shape {values.shape}")
    if (x_edges is None) != (y_edges is None):
        raise ValueError("pass both x_edges and y_edges, or neither")
    if x_edges is None:
        x0, x1, y0, y1 = extent
        x_edges = np.linspace(x0, x1, values.shape[0] + 1)
        y_edges = np.linspace(y0, y1, values.shape[1] + 1)
    x_edges = np.asarray(x_edges, dtype=np.float64)
    y_edges = np.asarray(y_edges, dtype=np.float64)
    if x_edges.shape != (values.shape[0] + 1,) or y_edges.shape != (values.shape[1] + 1,):
        raise ValueError(f"edges {x_edges.shape}, {y_edges.shape} do not bound a {values.shape} array")
    mappable = ax.pcolorfast(x_edges / scale, y_edges / scale, values.T, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect('equal')
    return mappable


def _draw_triangles(ax, triangulation, extent, scale, cmap, vmin, vmax, contours=0):
    """Draw a FieldData.triangulation (nodes [m], triangles, node values) by Gouraud tripcolor (P2: four linear
    sub-triangles per element), clipped to extent [m]; triangles with a non-finite node value are left out;
    contours > 0 adds that many tricontour lines between vmin and vmax."""
    from matplotlib.tri import Triangulation
    nodes, triangles, node_values = triangulation
    nodes = np.asarray(nodes, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    node_values = np.asarray(node_values, dtype=np.float64)
    finite = np.isfinite(node_values)
    hidden = ~finite[triangles].all(axis=1)
    if hidden.all():
        raise ValueError('the triangulated field has no triangle with finite values')
    shown_values = np.where(finite, node_values, 0.0)   # the 0 fill reaches only hidden triangles
    mesh = Triangulation(nodes[:, 0] / scale, nodes[:, 1] / scale, triangles, mask=hidden)
    mappable = ax.tripcolor(mesh, shown_values, shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax)
    if contours > 0 and vmax > vmin:
        ax.tricontour(mesh, shown_values, levels=np.linspace(vmin, vmax, int(contours) + 2)[1:-1],
                      colors=_contour_color(cmap), linewidths=0.5)
    x0, x1, y0, y1 = extent
    ax.set_xlim(x0 / scale, x1 / scale)
    ax.set_ylim(y0 / scale, y1 / scale)
    ax.set_aspect('equal')
    return mappable


def _colorbar(ax, mappable, label, size='3%'):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cax = make_axes_locatable(ax).append_axes('right', size=size, pad=0.08)
    colorbar = ax.figure.colorbar(mappable, cax=cax)
    if label:
        colorbar.set_label(label)
    return colorbar


def _set_frame_axes(ax, frame):
    """Clamp `ax` to the physical window and label x / y in the frame's length unit."""
    x0, x1, y0, y1 = frame.scaled_extent()
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel(f'x ({frame.unit})')
    ax.set_ylabel(f'y ({frame.unit})')


def _blank(ax, message, fontsize=9):
    """Replace a panel's content by a centred message."""
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha='center', va='center', fontsize=fontsize,
            color='gray')
    ax.set_xticks([])
    ax.set_yticks([])


def _legend(ax, kinds):
    """Legend for the annotation kinds drawn on `ax` (nothing when empty)."""
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], **style) for kind, style in _LEGEND_STYLES.items() if kind in kinds]
    if handles:
        ax.legend(handles=handles, loc='upper right', fontsize=7, framealpha=0.8, edgecolor='gray')


def _draw_box(ax, box, color, linestyle):
    import matplotlib.patches as patches
    x0, y0, width, height = box
    ax.add_patch(patches.Rectangle((x0, y0), width, height, linewidth=1.5, edgecolor=color,
                                   facecolor='none', linestyle=linestyle, alpha=0.8, zorder=9))


def _contour_levels(values, max_levels=12, min_gap_fraction=0.02):
    """Contour levels at the midpoints of the gaps between distinct material values; a continuous map (heated eps)
    keeps only gaps above min_gap_fraction of its range, the max_levels largest (None when no level remains)."""
    distinct = np.unique(values)
    if distinct.size < 2:
        return None
    gaps = np.diff(distinct)
    if distinct.size > max_levels + 1:
        gap_index = np.flatnonzero(gaps > min_gap_fraction * float(distinct[-1] - distinct[0]))
        gap_index = np.sort(gap_index[np.argsort(gaps[gap_index])[::-1][:max_levels]])
        if gap_index.size == 0:
            return None
    else:
        gap_index = np.arange(gaps.size)
    return 0.5 * (distinct[gap_index] + distinct[gap_index + 1])


def _heat_bc_style(side):
    """(colour, alpha, label) of a heat BCSide (kind dirichlet | neumann | robin)."""
    if side.kind == 'dirichlet':
        return '#CC3300', 0.45, f'T = {side.value:.5g} K'
    if side.kind == 'robin':
        return '#0077BB', 0.45, f'h = {side.h:.3g} W/(m²·K), T∞ = {side.external:.5g} K'
    if side.kind == 'neumann':
        if side.value == 0.0:
            return '#888888', 0.25, 'insulated'
        return '#888888', 0.45, f'q = {side.value:.3g} W/m²'
    raise ValueError(f"unknown heat BC kind '{side.kind}'")


def _potential_bc_style(side, kind):
    """(colour, alpha, label) of a PotentialPhysics BCSide; kind 'currents' (neumann = inward J) | 'electrostatics'
    (neumann = surface charge)."""
    currents = kind == 'currents'
    if side.kind == 'dirichlet':
        return '#CC3300', 0.45, f'V = {side.value:.5g} V'
    if side.kind == 'robin':
        return '#0077BB', 0.45, f'h = {side.h:.3g} {"S/m²" if currents else "F/m²"}, V∞ = {side.external:.5g} V'
    if side.kind == 'neumann':
        if side.value == 0.0:
            return '#888888', 0.25, 'insulated' if currents else 'D_n = 0'
        return '#888888', 0.45, (f'J = {side.value:.3g} A/m²' if currents else f'σ_s = {side.value:.3g} C/m²')
    raise ValueError(f"unknown potential BC kind '{side.kind}'")


def _time_axis(times_list):
    """(time scale or None, axis label) shared by sample-time arrays; None = plot against the sample index."""
    times = np.concatenate([np.asarray(t, dtype=np.float64) for t in times_list])
    if times.size < 2 or np.ptp(times) == 0:
        return None, 'Sample'
    t_scale, t_unit = time_unit(float(np.max(np.abs(times))))
    return t_scale, f'Time ({t_unit})'


def _scaled_times(times, t_scale):
    return times / t_scale if t_scale is not None else np.arange(times.size)


def _flux_spectrum(monitor):
    """'flux_spectrum' result of an FDTD FluxMonitor (unit a.u.) or an FDFD HarmonicFluxMonitor (unit W/m)."""
    from ..physics.em.fdfd_monitors import HarmonicFluxMonitor, harmonic_flux_result
    from ..physics.em.results import flux_result
    return harmonic_flux_result(monitor) if isinstance(monitor, HarmonicFluxMonitor) else flux_result(monitor)


def _far_field_pattern(monitor, freq_index, n_angles):
    """(angles [rad from +x], |pattern|^2 unscaled, wavelength [m]) at one frequency only: FDTD NearToFarField index in
    monitor order, FDFD HarmonicNearToFar index in ascending solved wavelength; IndexError outside [0, n)."""
    from ..physics.em.fdfd_monitors import HarmonicNearToFar
    from ..physics.em.monitors import near_to_far_pattern
    n_angles = int(n_angles)
    if n_angles < 1:
        raise ValueError(f"n_angles must be >= 1, got {n_angles}")
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    if isinstance(monitor, HarmonicNearToFar):
        omegas, faces = monitor.recorded()
        frequencies = omegas / (2.0 * np.pi)
    else:
        frequencies = np.asarray(monitor.frequencies, dtype=np.float64)
    freq_index = int(freq_index)
    if not 0 <= freq_index < frequencies.size:
        raise IndexError(f"freq_idx {freq_index} outside [0, {frequencies.size}) of the monitor's frequencies")
    if isinstance(monitor, HarmonicNearToFar):
        one = slice(freq_index, freq_index + 1)
        one_faces = tuple((e_faces[one], h_faces[one]) for e_faces, h_faces in faces)
        pattern = near_to_far_pattern(monitor.polarization, frequencies[one], angles, one_faces, monitor.geometry)[0]
    else:
        pattern = monitor.far_field(angles, freq_index=freq_index)
    return angles, np.abs(pattern) ** 2, float(C0 / frequencies[freq_index])


def _probe_spectrum(result):
    """(wavelength [m], |FFT|²) of a probe result (probe_spectrum); None if not possible."""
    spectrum = probe_spectrum(result['t'], result['values'])
    if spectrum is None:
        return None
    frequencies, magnitude = spectrum
    return C0 / frequencies, magnitude ** 2


def _spectrum_mask(wavelength, power, wavelength_range):
    """Wavelengths inside `wavelength_range` [m], else the band holding >= 1e-3 of the peak power."""
    if wavelength_range is not None:
        return (wavelength >= wavelength_range[0]) & (wavelength <= wavelength_range[1])
    return power >= 1e-3 * _finite_peak(power)


def _finite_peak(values) -> float:
    """Largest finite value (0 when none): one NaN sample must not hide the peak."""
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return float(finite.max()) if finite.size else 0.0


def field_data_edges(field_data):
    """(x_edges, y_edges) [m] of a FieldData on non-uniform native cells, rebuilt from its centres; (None, None) on
    uniform cells or without centres (uniform cells over the extent are then exact)."""
    if field_data.x_centers is None or field_data.y_centers is None:
        return None, None
    extent = [float(edge) for edge in field_data.extent]
    x_edges = edges_from_centers(field_data.x_centers, extent[0], extent[1])
    y_edges = edges_from_centers(field_data.y_centers, extent[2], extent[3])
    if has_uniform_spacing(np.diff(x_edges)) and has_uniform_spacing(np.diff(y_edges)):
        return None, None
    return x_edges, y_edges


def dft_map_edges(result):
    """(x_edges, y_edges) [m] of a dft_fields result map: the box cells, or with downsample > 1 midpoints between the
    recorded centres (their blocks), outer edges pinned to the extent."""
    extent = [float(edge) for edge in result['extent']]
    if int(result.get('downsample', 1)) == 1:
        return (edges_from_centers(result['x_centers'], extent[0], extent[1]),
                edges_from_centers(result['y_centers'], extent[2], extent[3]))
    edges = []
    for centers, low_edge, high_edge in ((result['x_centers'], extent[0], extent[1]),
                                         (result['y_centers'], extent[2], extent[3])):
        centers = np.asarray(centers, dtype=np.float64)
        edges.append(np.concatenate(([low_edge], 0.5 * (centers[1:] + centers[:-1]), [high_edge])))
    return tuple(edges)


# ------------------------------------------------------------------ result panels (Visualizer and result_plots)

def _ascii_unit(scale_and_unit):
    """(scale, unit) of the unit helpers with the micro sign written as ASCII 'u' (um, us)."""
    scale, unit = scale_and_unit
    return scale, unit if unit.isascii() else 'u' + unit[1:]


def _with_unit(name, unit):
    """'name (unit)', or name alone for '' and the dimensionless '1'."""
    return f'{name} ({unit})' if unit and unit != '1' else name


def _draw_map(ax, values, extent, scale_and_unit, cmap, limits, colorbar_label, title, x_edges=None, y_edges=None,
              colorbar_size='3%', triangulation=None, contours=0, axes=('x', 'y')):
    """Cell map (_draw_cells; with a FieldData triangulation _draw_triangles instead) with colour bar, labels of the
    value axes (axes, e.g. ('x', 'z') of a 3D y cut) in the axis length unit (scale, unit) and title."""
    scale, unit = scale_and_unit
    if triangulation is not None:
        mappable = _draw_triangles(ax, triangulation, extent, scale, cmap, *limits, contours=contours)
    else:
        mappable = _draw_cells(ax, values, extent, scale, cmap, *limits, x_edges, y_edges)
    _colorbar(ax, mappable, colorbar_label, size=colorbar_size)
    ax.set_xlabel(f'{axes[0]} ({unit})')
    ax.set_ylabel(f'{axes[1]} ({unit})')
    ax.set_title(title)


def field_data_axes(field_data) -> tuple:
    """Value axes of a FieldData for map labels; ValueError for native polar / spherical cells (an angle axis in rad
    cannot share the length scale: plot the containing-cell image, field(native=False), instead)."""
    axes = tuple(getattr(field_data, 'axes', None) or ('x', 'y'))
    if any(axis in ('phi', 'theta') for axis in axes):
        raise ValueError(f'cannot draw a field on native {axes[0]}, {axes[1]} cells as a map; use the containing-cell '
                         'image over the drawing plane (native=False)')
    return axes


def _draw_timeseries(ax, result, title, ascii_text=False):
    """Timeseries result values vs time (auto unit), or vs the 1-based iteration when all samples share one time
    (stationary iterations); a 'no samples' panel for an empty series."""
    times = np.asarray(result['t'], dtype=np.float64)
    values = np.asarray(result['values'])
    line_style = '.-' if values.size <= 200 else '-'
    if times.size == 0:
        _blank(ax, 'no samples')
    elif times.size < 2 or np.ptp(times) == 0:
        ax.plot(np.arange(1, values.size + 1), values, line_style, linewidth=0.8, color='#2166AC')
        ax.set_xlabel(f'Iteration (all at t = {float(times[0]):.6g} s)')
    else:
        t_scale, t_unit = time_unit(float(np.max(np.abs(times))))
        if ascii_text:
            t_scale, t_unit = _ascii_unit((t_scale, t_unit))
        ax.plot(times / t_scale, values, line_style, linewidth=0.8, color='#2166AC')
        ax.set_xlabel(f'Time ({t_unit})')
    if times.size:
        ax.grid(True, alpha=0.2)
    ax.set_ylabel(_with_unit(result.get('component', ''), result.get('unit', '')))
    ax.set_title(title)


def _draw_flux_panel(ax, results, title, labels=None, direction='+x / +y'):
    """Flux power vs ascending wavelength (nm, ASCII), one curve per flux_spectrum result (legend from labels when
    several); > 0 is power toward `direction`. Returns the wavelength axis scale [m per unit]."""
    wavelengths = [np.asarray(result['wavelength'], dtype=np.float64) for result in results]
    if not wavelengths or any(wavelength.size == 0 for wavelength in wavelengths):
        raise ValueError('a flux spectrum has no wavelengths')
    wl_scale, wl_unit = WAVELENGTH_UNIT
    single = len(results) == 1
    for index, (result, wavelength) in enumerate(zip(results, wavelengths)):
        order = np.argsort(wavelength)
        ax.plot(wavelength[order] / wl_scale, np.asarray(result['power'], dtype=np.float64)[order], '-o',
                markersize=3, linewidth=1.2, color=_STYLE['flux_color'] if single else None,
                label=None if labels is None else labels[index])
    units = {result.get('unit', 'a.u.') for result in results}   # FDTD 'a.u.', FDFD 'W/m'
    unit = units.pop() if len(units) == 1 else 'mixed units'
    ax.axhline(0.0, color='gray', linewidth=0.5)
    ax.set_xlabel(f'Wavelength ({wl_unit})')
    ax.set_ylabel(f'Flux ({unit}, > 0 toward {direction})')
    ax.set_title(title)
    if labels is not None and not single:
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)
    return wl_scale


def _draw_far_field_panel(ax, angles, power_rows, labels, title):
    """Polar plot (angle counter-clockwise from +x) of far-field power rows, each normalised to its maximum; a single
    row is filled, a legend shows the labels that are not None."""
    closed_angles = np.append(angles, angles[0])
    single = len(power_rows) == 1
    for power, label in zip(power_rows, labels):
        normalised, _ = normalised_db(power)
        closed_power = np.append(normalised, normalised[0])
        ax.plot(closed_angles, closed_power, linewidth=1.2, label=label, color='#0072BD' if single else None)
        if single:
            ax.fill(closed_angles, closed_power, alpha=0.15, color='#0072BD')
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_ylim(0.0, 1.05)
    if any(label is not None for label in labels):
        ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.08, 1.0))
    ax.set_title(title, pad=16)


def plot_field_array(values, extent=None, info=None, title='', filename=None, x_edges=None, y_edges=None, vmax=None,
                     contours=0):
    """Plot a 2D field ([i, j] = x, y) over extent (x0, x1, y0, y1) [m] styled by a FieldInfo, or a FieldData alone
    (extent None: its extent, info unless given, native cell edges; a triangle-mesh FieldData by tripcolor over its
    triangulation, `contours` lines). Cell edges [m] draw non-uniform cells (None: uniform cells over the extent);
    `vmax` overrides the colour limit."""
    triangulation, axes = None, ('x', 'y')
    if extent is None:
        field_data = values
        axes = field_data_axes(field_data)
        values, extent, info = field_data.values, field_data.extent, info or field_data.info
        triangulation = getattr(field_data, 'triangulation', None)
        if x_edges is None and y_edges is None:
            x_edges, y_edges = field_data_edges(field_data)
    return _finish(_field_map_figure(values, extent, info, title, x_edges, y_edges, vmax, triangulation, contours,
                                     axes), filename)


def _field_map_figure(values, extent, info, title, x_edges, y_edges, vmax, triangulation=None, contours=0,
                      axes=('x', 'y')):
    """Unfinished Figure of plot_field_array (no layout pass yet: callers add overlays, then _finish once)."""
    fig = _new_figure(_STYLE['figsize_single'])
    _draw_map(fig.subplots(), values, extent, _extent_length_unit(extent), info.cmap,
              color_limits(values, info.signed, vmax), _field_label(info), title or info.label, x_edges, y_edges,
              triangulation=triangulation, contours=contours, axes=axes)
    return fig


class Visualizer:
    """Matplotlib plots of a Simulation from sim.domain, sim.physics and each solver's renderer(); EM physics are FDTD
    EMPhysics and FDFDPhysics. Holds sim weakly (no sim <-> sim.viz cycle keeping a dropped Simulation's GPU arrays)."""

    def __init__(self, sim):
        self._sim_ref = weakref.ref(sim)
        self.renderers = {}
        self._renderer_key = ()

    @property
    def sim(self):
        sim = self._sim_ref()
        if sim is None:
            raise RuntimeError('Visualizer: its Simulation was garbage-collected (keep a reference to the Simulation)')
        return sim

    @property
    def domain(self):
        return self.sim.domain

    # ------------------------------------------------------------ data access

    def _sync_renderers(self):
        """Rebuild self.renderers when the simulation's physics set changed; return it."""
        physics = self.sim.physics
        key = tuple((name, id(solver)) for name, solver in physics.items())
        if key != self._renderer_key:
            self.renderers = {name: solver.renderer() for name, solver in physics.items()}
            self._renderer_key = key
        return self.renderers

    def _resolve_renderer(self, component, physics=None):
        """Renderer of physics `physics` if given, else the unique renderer exposing `component`."""
        renderers = self._sync_renderers()
        if physics is not None:
            if physics not in renderers:
                raise KeyError(f"no physics named '{physics}'; attached: {list(renderers)}")
            return renderers[physics]
        owners = [name for name, renderer in renderers.items() if component in renderer.components]
        if len(owners) == 1:
            return renderers[owners[0]]
        if owners:
            raise ValueError(f"component '{component}' is exposed by several physics {owners}; "
                             f"pass physics='<name>'")
        available = {name: list(renderer.components) for name, renderer in renderers.items()}
        raise KeyError(f"no attached physics exposes component '{component}'; available: {available}")

    def _component_unit(self, component):
        """Unit of `component` when exactly one attached physics exposes it, else ''."""
        owners = [r for r in self._sync_renderers().values() if component in r.components]
        return owners[0].field_info(component).unit if len(owners) == 1 else ''

    def _resolve_field(self, component, physics):
        """(renderer, component); component None = the primary field of `physics` (default: the first physics)."""
        if component is None and physics is None:
            renderers = self._sync_renderers()
            if not renderers:
                raise ValueError("no physics attached; no field to show")
            physics = next(iter(renderers))
        renderer = self._resolve_renderer(component, physics)
        return renderer, component if component is not None else _primary_component(renderer)

    def _get_phys_field(self, component=None, physics=None):
        """Renderer field over the physical window (numpy); component None as in plot_field."""
        renderer, component = self._resolve_field(component, physics)
        return renderer.physical_field(component)

    def _get_phys_eps(self):
        """Permittivity over the physical window (numpy): eps_r with a physics reading it (E_z, em_modes) or no EM,
        eps_xx with only H_z physics; vacuum (ones) when that array is not registered (no EM physics)."""
        from ..core.backend import to_host
        em_solvers = self._em_solvers()
        name = 'eps_r' if not em_solvers or self._reads_eps_r(em_solvers) else 'eps_xx'
        sx, sy = self.domain.physical_window()
        if name not in self.domain.materials.property_names():
            return np.ones((sx.stop - sx.start, sy.stop - sy.start), dtype=np.float32)
        return np.array(to_host(getattr(self.domain.materials, name)[sx, sy]))

    def _material_map(self):
        """(window values, FieldInfo, contour levels) of the plotted material: eps_r with an E_z / em_modes physics,
        eps_xx with only H_z physics, else k_xx if registered, else sigma_dc_xx / eps_dc_xx of the first potential
        physics (eps_r when none exists); levels computed once per map."""
        from ..core.backend import to_host
        from ..core.renderer import FieldInfo
        em_solvers = self._em_solvers()
        k_xx = getattr(self.domain.materials, 'k_xx', None)
        potential = self._potential_physics()
        sx, sy = self.domain.physical_window()
        if self._reads_eps_r(em_solvers) or (not em_solvers and k_xx is None and potential is None):
            values, info = self._get_phys_eps(), FieldInfo(name='eps_r', label='Relative permittivity ε_r',
                                                           cmap=_STYLE['material_cmap'])
        elif em_solvers:
            values, info = self._get_phys_eps(), FieldInfo(name='eps_xx', label='Relative permittivity eps_xx',
                                                           cmap=_STYLE['material_cmap'])
        elif k_xx is not None:
            values, info = np.array(to_host(k_xx[sx, sy])), FieldInfo(name='k_xx', label='Thermal conductivity k_xx',
                                                                      unit='W/(m·K)', cmap=_STYLE['material_cmap'])
        elif potential.kind == 'currents':
            values = np.array(to_host(self.domain.materials.sigma_dc_xx[sx, sy]))
            info = FieldInfo(name='sigma_dc_xx', label='DC conductivity σ_dc,xx', unit='S/m',
                             cmap=_STYLE['material_cmap'])
        else:
            values = np.array(to_host(self.domain.materials.eps_dc_xx[sx, sy]))
            info = FieldInfo(name='eps_dc_xx', label='DC relative permittivity ε_dc,xx', cmap=_STYLE['material_cmap'])
        return values, info, _contour_levels(values)

    def _frame(self):
        domain = self.domain
        if getattr(domain, 'dimensions', 2) != 2 or getattr(domain, 'grid_kind', 'tensor') != 'tensor':
            raise NotImplementedError('the Visualizer draws 2D tensor-grid Simulations only; plot 3D or triangle / '
                                      'tetrahedron fields from Session / Results FieldData (plot_field_array)')
        extent = tuple(float(value) for value in domain.physical_extent())
        scale, unit = _extent_length_unit(extent)
        x_centers, y_centers = domain.cell_centers()
        x_edges, y_edges = domain.cell_edges()
        return _Frame(extent, scale, unit, x_centers, y_centers, x_edges, y_edges,
                      domain.physical_window(), bool(domain.non_uniform))

    def _window_widths(self):
        """Cell widths (dx, dy) [m] of the physical window."""
        dx_cells, dy_cells = self.domain.cell_widths()
        sx, sy = self.domain.physical_window()
        return dx_cells[sx], dy_cells[sy]

    def _probes(self):
        """Every time-series FieldProbe: probes among each physics' `monitors` (EM FDTD) plus each physics' `probes`
        list (heat); FDFD probes: _harmonic_probes."""
        from ..core.monitors import FieldProbe
        probes = []
        for solver in self.sim.physics.values():
            probes.extend(m for m in getattr(solver, 'monitors', ()) if isinstance(m, FieldProbe))
            probes.extend(getattr(solver, 'probes', ()))
        return probes

    def _harmonic_probes(self):
        """Every FDFD HarmonicProbe (one phasor per solved wavelength, no time series): drawn, never plotted here."""
        from ..physics.em.fdfd_monitors import HarmonicProbe
        return [monitor for solver in self.sim.physics.values() for monitor in getattr(solver, 'monitors', ())
                if isinstance(monitor, HarmonicProbe)]

    def _probe_result(self, probe):
        from ..core.monitors import probe_result
        from ..physics.em.fdfd_monitors import HarmonicProbe
        if isinstance(probe, HarmonicProbe):
            raise TypeError('an FDFD HarmonicProbe holds one phasor per solved wavelength, not a time series: use '
                            'fdfd_monitors.harmonic_probe_result (Results.result in a model)')
        return probe_result(probe, unit=self._component_unit(probe.field_component))

    def _probe_label(self, probe, frame):
        x, y = frame.point(probe.ix, probe.iy)
        return f'{probe.field_component} at ({x:.3g}, {y:.3g}) {frame.unit}'

    def _heat_physics(self):
        from ..physics.heat import HeatPhysics
        return next((s for s in self.sim.physics.values() if isinstance(s, HeatPhysics)), None)

    def _potential_physics(self):
        from ..physics.potential import PotentialPhysics
        return next((s for s in self.sim.physics.values() if isinstance(s, PotentialPhysics)), None)

    def _em_solvers(self):
        """Every attached EM physics, FDTD EMPhysics, FDFDPhysics (both polarisations) and ModePhysics, in attachment
        order."""
        from ..physics.em import EMPhysics, FDFDPhysics, ModePhysics
        return [solver for solver in self.sim.physics.values()
                if isinstance(solver, (EMPhysics, FDFDPhysics, ModePhysics))]

    @staticmethod
    def _reads_eps_r(em_solvers):
        """True when an EM solver reads the eps_r array (E_z physics, em_modes)."""
        return any('eps_r' in em.permittivity_arrays for em in em_solvers)

    # ------------------------------------------------------------ annotations

    def _annotate_sources(self, ax, frame):
        """Draw the sources of every EM physics (FDTD and FDFD phasor sources alike) at their physical positions;
        return the legend kinds drawn."""
        em_solvers = self._em_solvers()
        if not em_solvers:
            return set()
        from ..core.sources import PointSource, LineSource
        from ..physics.em.fdfd_ports import ModeSourcePhasor
        from ..physics.em.fdfd_sources import (ElectricDipolePhasor, MagneticCurrentPhasor, PlaneWavePhasor,
                                               PointCurrentPhasor, SheetCurrentPhasor)
        from ..physics.em.mode_ports import ModeSource
        from ..physics.em.sources import TFSFSource, TFSFSourceHz
        color = _STYLE['source_color']
        kinds = set()
        for source in (source for em in em_solvers for source in em.sources):
            if isinstance(source, (PointSource, PointCurrentPhasor, MagneticCurrentPhasor, ElectricDipolePhasor)):
                x, y = frame.point(source.ix, source.iy)
                ax.plot(x, y, '*', color=color, markersize=12, markeredgecolor='white', markeredgewidth=0.5,
                        zorder=10)
                ax.annotate('src', (x, y), color=color, fontsize=7, fontweight='bold', xytext=(4, 4),
                            textcoords='offset points')
                kinds.add('source')
            elif isinstance(source, (LineSource, SheetCurrentPhasor)):
                # H_z tangential sheet currents (hi_face) sit on the hi face of the line cell.
                xs, ys = frame.segment(source.axis, source.position, source.span,
                                       hi_face=getattr(source, 'hi_face', False))
                ax.plot(xs, ys, color=color, linewidth=1.5, linestyle='--', alpha=0.8, zorder=9)
                kinds.add('line_source')
            elif isinstance(source, (TFSFSource, TFSFSourceHz, PlaneWavePhasor)):
                # Total-field E_z / H_z cells: [x0, x1) x [y0, y1) (half-open).
                _draw_box(ax, frame.box(source.x0, source.x1, source.y0, source.y1), color, '-.')
                kinds.add('tfsf')
            elif isinstance(source, (ModeSource, ModeSourcePhasor)):
                # Through the line-cell centres over its solve range (clipped to the window when PML-extended).
                xs, ys = frame.segment(source.axis, source.position, source.span)
                ax.plot(xs, ys, color=color, linewidth=3.0, linestyle='--', alpha=0.8, zorder=9)
                kinds.add('mode_source')
            else:
                raise TypeError(f"cannot annotate EM source of type {type(source).__name__}")
        return kinds

    def _annotate_monitors(self, ax, frame):
        """Draw the flux (FDTD and FDFD) / DFT / NF2FF / mode monitors of every EM physics (probes: _annotate_probes);
        return the legend kinds drawn."""
        em_solvers = self._em_solvers()
        if not em_solvers:
            return set()
        from ..core.monitors import FieldProbe
        from ..physics.em.fdfd_monitors import HarmonicFluxMonitor, HarmonicNearToFar, HarmonicProbe
        from ..physics.em.fdfd_ports import HarmonicModeMonitor
        from ..physics.em.mode_ports import ModeMonitor
        from ..physics.em.monitors import DFTMonitor, FluxMonitor, NearToFarField
        from ..physics.em.solvers.modes_2d import ModeDispersionMonitor, ModeTableMonitor
        kinds = set()
        for monitor in (monitor for em in em_solvers for monitor in em.monitors):
            if isinstance(monitor, (FieldProbe, HarmonicProbe, ModeTableMonitor, ModeDispersionMonitor)):
                continue   # probes: _annotate_probes; em_modes monitors have no geometry
            if isinstance(monitor, (FluxMonitor, HarmonicFluxMonitor)):
                # Flux axis 'x' = line at fixed x spanning y; it measures the hi face of its line cell.
                along = 'y' if monitor.axis == 'x' else 'x'
                xs, ys = frame.segment(along, monitor.position, monitor.span, hi_face=True)
                ax.plot(xs, ys, color=_STYLE['flux_color'], linewidth=2.5, alpha=0.9, zorder=9)
                kinds.add('flux')
            elif isinstance(monitor, DFTMonitor):
                box = frame.box(monitor.x_slice.start or 0, monitor.x_slice.stop,
                                monitor.y_slice.start or 0, monitor.y_slice.stop)
                _draw_box(ax, box, _STYLE['dft_color'], ':')
                kinds.add('dft')
            elif isinstance(monitor, (NearToFarField, HarmonicNearToFar)):
                # The sampled contour runs through the centres of columns x0, x1 and rows y0, y1 (x1 = cell(hi)+1).
                box = frame.center_box(monitor.x0, monitor.x1, monitor.y0, monitor.y1)
                _draw_box(ax, box, _STYLE['nf2ff_color'], ':')
                kinds.add('nf2ff')
            elif isinstance(monitor, (ModeMonitor, HarmonicModeMonitor)):
                # Mode monitor axis = direction along the line; its transverse field sits on the hi face (normal).
                xs, ys = frame.segment(monitor.axis, monitor.position, monitor.span, hi_face=True)
                ax.plot(xs, ys, color=_STYLE['mode_color'], linewidth=2.5, alpha=0.9, zorder=9)
                kinds.add('mode_monitor')
            else:
                raise TypeError(f"cannot annotate EM monitor of type {type(monitor).__name__}")
        return kinds

    def _annotate_probes(self, ax, frame):
        """Draw every time-series and FDFD probe at its cell centre; return the legend kinds drawn."""
        probes = self._probes() + self._harmonic_probes()
        color = _STYLE['probe_color']
        for probe in probes:
            x, y = frame.point(probe.ix, probe.iy)
            ax.plot(x, y, 'o', color=color, markersize=6, markeredgecolor='white', markeredgewidth=0.5, zorder=10)
            ax.annotate(probe.field_component, (x, y), color=color, fontsize=6, xytext=(4, 4),
                        textcoords='offset points')
        return {'probe'} if probes else set()

    def _boundary_styles(self):
        """{face: (colour, alpha, label)} from the first EM physics' BoundaryConfig, else the heat face BCs, else the
        first potential physics' face BCs, else {}."""
        em_solvers = self._em_solvers()
        if em_solvers:
            boundary_config = em_solvers[0].boundary_config
            return {face: _EM_BC_STYLE[getattr(boundary_config, face).value] for face in FACES}
        heat = self._heat_physics()
        if heat is not None:
            return {face: _heat_bc_style(side) for face, side in heat.bcs.sides().items()}
        potential = self._potential_physics()
        if potential is not None:
            return {face: _potential_bc_style(side, potential.kind) for face, side in potential.bcs.sides().items()}
        return {}

    def _annotate_boundaries(self, ax, frame):
        """Colour bar and label on each window face."""
        styles = self._boundary_styles()
        if not styles:
            return
        import matplotlib.patches as patches
        x0, x1, y0, y1 = frame.scaled_extent()
        bar = 0.015 * max(x1 - x0, y1 - y0)
        bars = {
            'x_lo': ((x0, y0), bar, y1 - y0),
            'x_hi': ((x1 - bar, y0), bar, y1 - y0),
            'y_lo': ((x0, y0), x1 - x0, bar),
            'y_hi': ((x0, y1 - bar), x1 - x0, bar),
        }
        # (axes-fraction position, rotation, ha, va) of each face label, just inside the bar.
        labels = {
            'x_lo': ((0.025, 0.5), 90, 'left', 'center'),
            'x_hi': ((0.975, 0.5), 90, 'right', 'center'),
            'y_lo': ((0.5, 0.025), 0, 'center', 'bottom'),
            'y_hi': ((0.5, 0.975), 0, 'center', 'top'),
        }
        for face, (color, alpha, text) in styles.items():
            corner, width, height = bars[face]
            ax.add_patch(patches.Rectangle(corner, width, height, facecolor=color, alpha=alpha, edgecolor='none',
                                           zorder=8))
            (u, v), rotation, ha, va = labels[face]
            ax.text(u, v, text, transform=ax.transAxes, rotation=rotation, ha=ha, va=va, fontsize=6, color=color,
                    fontweight='bold', zorder=11)

    def _draw_material_contours(self, ax, frame, color, material_map=None, cell_box=None):
        """Overlay material-interface contours (material_map from _material_map); cell_box = full-grid (x_slice,
        y_slice) contours only that box plus one margin cell."""
        values, _, levels = material_map if material_map is not None else self._material_map()
        x_centers, y_centers = frame.window_centers()
        if cell_box is not None:
            # Window-relative cell ranges, clamped to the window.
            (sx, sy), (box_x, box_y) = frame.window, cell_box
            x_cells = slice(min(max(box_x.start - sx.start - 1, 0), values.shape[0]),
                            min(max(box_x.stop - sx.start + 1, 0), values.shape[0]))
            y_cells = slice(min(max(box_y.start - sy.start - 1, 0), values.shape[1]),
                            min(max(box_y.stop - sy.start + 1, 0), values.shape[1]))
            values, x_centers, y_centers = values[x_cells, y_cells], x_centers[x_cells], y_centers[y_cells]
        if levels is None or x_centers.size < 2 or y_centers.size < 2:
            return
        ax.contour(x_centers / frame.scale, y_centers / frame.scale, values.T, levels=levels, colors=color,
                   linewidths=0.6, alpha=0.5)

    def _draw_conductors(self, ax, frame):
        """Grey fill on the union of every EM physics' pec_cells and a dashed outline of sigma_e > 0 (neither shows
        in the permittivity map); return the legend kinds drawn."""
        em_solvers = self._em_solvers()
        if not em_solvers:
            return set()
        from matplotlib.colors import ListedColormap
        from ..core.backend import to_host
        sx, sy = frame.window
        kinds = set()
        pec_masks = [to_host(em.pec_cells[sx, sy]) for em in em_solvers if em.pec_cells is not None]
        pec_window = np.logical_or.reduce(pec_masks) if pec_masks else None
        if pec_window is not None and pec_window.any():
            # NaN cells stay transparent; zorder above the uniform-material placeholder patch.
            pec_image = _draw_cells(ax, np.where(pec_window, 1.0, np.nan), frame.extent, frame.scale,
                                    ListedColormap([_STYLE['pec_color']]), 0.0, 1.0, *frame.window_edges())
            pec_image.set_zorder(2)
            kinds.add('pec')
        materials = self.domain.materials
        if 'sigma_e' in materials.property_names():
            conducting = to_host(materials.sigma_e[sx, sy] > 0)
            x_centers, y_centers = frame.window_centers()
            if conducting.any() and not conducting.all() and x_centers.size >= 2 and y_centers.size >= 2:
                ax.contour(x_centers / frame.scale, y_centers / frame.scale, conducting.T.astype(np.float32),
                           levels=[0.5], colors=_STYLE['conductor_color'], linewidths=1.0, linestyles='--',
                           zorder=3)
                kinds.add('conductor')
        return kinds

    # ------------------------------------------------------------ panels

    def _draw_geometry(self, ax, frame, title, material_map=None, annotate=True):
        """Material map, PEC / sigma_e overlay, plus boundary bars, sources, monitors, probes and legend."""
        values, info, _ = material_map if material_map is not None else self._material_map()
        if np.ptp(values) > 0:
            vmin, vmax = color_limits(values, signed=False)
            mappable = _draw_cells(ax, values, frame.extent, frame.scale, info.cmap, vmin, vmax,
                                   *frame.window_edges())
            _colorbar(ax, mappable, _field_label(info))
        else:
            import matplotlib.patches as patches
            x0, x1, y0, y1 = frame.scaled_extent()
            ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor='#f2f2f2', edgecolor='none'))
            ax.text(0.5, 0.5, f'uniform {info.name} = {float(values.flat[0]):.4g}', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='gray')
            ax.set_aspect('equal')
        conductor_kinds = self._draw_conductors(ax, frame)
        if annotate:
            self._annotate_boundaries(ax, frame)
            kinds = (conductor_kinds | self._annotate_sources(ax, frame) | self._annotate_monitors(ax, frame)
                     | self._annotate_probes(ax, frame))
            _legend(ax, kinds)
        _set_frame_axes(ax, frame)
        ax.set_title(title, fontsize=11)

    def _draw_cell_area(self, ax, frame, title, material_map=None):
        """Cell-area map of the physical window (finer cells = smaller area)."""
        dx_w, dy_w = self._window_widths()
        cell_scale, cell_unit = length_unit(max(dx_w.max(), dy_w.max()))
        area = np.outer(dx_w, dy_w) / cell_scale ** 2
        vmin, vmax = color_limits(area, signed=False)
        mappable = _draw_cells(ax, area, frame.extent, frame.scale, _STYLE['mesh_cmap'], vmin, vmax,
                               *frame.window_edges())
        _colorbar(ax, mappable, f'Cell area ({cell_unit}²)')
        self._draw_material_contours(ax, frame, 'white', material_map)
        _set_frame_axes(ax, frame)
        smallest = min(dx_w.min(), dy_w.min()) / cell_scale
        largest = max(dx_w.max(), dy_w.max()) / cell_scale
        ax.set_title(f'{title} ({smallest:.3g}–{largest:.3g} {cell_unit})', fontsize=10)

    def _draw_uniform_mesh_summary(self, ax):
        dx_w, dy_w = self._window_widths()
        cell_scale, cell_unit = length_unit(max(dx_w.max(), dy_w.max()))
        grid = self.domain.grid
        lines = ['Uniform grid', f'dx = {dx_w[0] / cell_scale:.3g} {cell_unit}',
                 f'{dx_w.size}×{dy_w.size} physical cells']
        if (grid.nx, grid.ny) != (dx_w.size, dy_w.size):
            lines.append(f'{grid.nx}×{grid.ny} cells incl. PML')
        _blank(ax, '\n'.join(lines), fontsize=11)
        ax.set_title('Mesh', fontsize=10)

    def _draw_field_panel(self, ax, frame, renderer, component, material_map=None):
        info = renderer.field_info(component)
        values = renderer.physical_field(component)
        vmin, vmax = color_limits(values, info.signed)
        mappable = _draw_cells(ax, values, frame.extent, frame.scale, info.cmap, vmin, vmax, *frame.window_edges())
        _colorbar(ax, mappable, _field_label(info))
        self._draw_material_contours(ax, frame, _contour_color(info.cmap), material_map)
        _set_frame_axes(ax, frame)
        ax.set_title(_field_title(renderer, info), fontsize=10)

    def _draw_probe_signals(self, ax, probes, frame):
        ax.set_title('Probe signals', fontsize=10)
        results = [(probe, self._probe_result(probe)) for probe in probes]
        results = [(probe, result) for probe, result in results if result['t'].size > 0]
        if not results:
            _blank(ax, 'no probe data')
            return
        t_scale, x_label = _time_axis([result['t'] for _, result in results])
        for probe, result in results:
            ax.plot(_scaled_times(result['t'], t_scale), result['values'], linewidth=0.6,
                    label=self._probe_label(probe, frame))
        units = {result['unit'] for _, result in results}
        unit = units.pop() if len(units) == 1 else 'mixed units'
        ax.set_xlabel(x_label)
        ax.set_ylabel(f'Probe value ({unit})' if unit else 'Probe value')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    def _draw_probe_spectra(self, ax, probes, frame):
        """Normalised probe spectra vs wavelength, labelled as the probe signals panel (component and position);
        returns False (ax untouched) when no probe has one."""
        spectra = []   # (probe, spectrum)
        for probe in probes:
            spectrum = _probe_spectrum(self._probe_result(probe))
            if spectrum is not None and _finite_peak(spectrum[1]) > 0:
                spectra.append((probe, spectrum))
        if not spectra:
            return False
        wl_scale, wl_unit = WAVELENGTH_UNIT
        for probe, (wavelength, power) in spectra:
            keep = _spectrum_mask(wavelength, power, None)
            ax.plot(wavelength[keep] / wl_scale, power[keep] / _finite_peak(power[keep]), linewidth=1,
                    label=self._probe_label(probe, frame))
        ax.set_xlabel(f'Wavelength ({wl_unit})')
        ax.set_ylabel('Normalised |FFT|²')
        ax.set_title('Probe spectra', fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
        return True

    def _draw_flux_spectra(self, ax, monitors, title):
        """_draw_flux_panel of FDTD / FDFD flux monitors (power toward +x / +y for axis 'x' / 'y'); returns the
        wavelength axis scale [m per unit]."""
        axes = {monitor.axis for monitor in monitors}
        return _draw_flux_panel(ax, [_flux_spectrum(monitor) for monitor in monitors], title,
                                labels=[f'flux {index} (+{monitor.axis})' for index, monitor in enumerate(monitors)],
                                direction=f'+{axes.pop()}' if len(axes) == 1 else '+x / +y')

    def _dashboard_fields(self, component):
        """[(renderer, component)]: `component`, each physics' primary field, then _SECONDARY_FIELDS."""
        renderers = self._sync_renderers()
        candidates = [(self._resolve_renderer(component), component)] if component is not None else []
        candidates += [(renderer, _primary_component(renderer)) for renderer in renderers.values()]
        candidates += [(renderer, name) for name in _SECONDARY_FIELDS for renderer in renderers.values()
                       if name in renderer.components]
        chosen = []
        for pair in candidates:
            if pair not in chosen:
                chosen.append(pair)
        return chosen

    def _draw_dashboard_field(self, ax, frame, panel_fields, index, material_map):
        if index >= len(panel_fields):
            _blank(ax, 'no field')
            return
        renderer, component = panel_fields[index]
        self._draw_field_panel(ax, frame, renderer, component, material_map)

    # ------------------------------------------------------------ geometry / mesh plots

    def plot_geometry(self, filename=None, show_annotations=True):
        frame = self._frame()
        fig = _new_figure(_STYLE['figsize_single'])
        self._draw_geometry(fig.subplots(), frame, 'Geometry & Setup', annotate=show_annotations)
        return _finish(fig, filename)

    def plot_setup(self, filename=None):
        """Geometry with annotations, plus the cell-area map on non-uniform grids."""
        frame = self._frame()
        material_map = self._material_map()
        if frame.non_uniform:
            fig = _new_figure(_STYLE['figsize_wide'])
            ax_geometry, ax_mesh = fig.subplots(1, 2)
            self._draw_cell_area(ax_mesh, frame, 'Mesh density', material_map)
        else:
            fig = _new_figure(_STYLE['figsize_single'])
            ax_geometry = fig.subplots()
        self._draw_geometry(ax_geometry, frame, 'Geometry', material_map)
        return _finish(fig, filename)

    def plot_mesh(self, filename=None):
        """Cell widths along x / y, cell-area map and cell-size histogram of the physical window."""
        frame = self._frame()
        dx_w, dy_w = self._window_widths()
        x_centers, y_centers = frame.window_centers()
        cell_scale, cell_unit = length_unit(max(dx_w.max(), dy_w.max()))
        fig = _new_figure((12, 9))
        axes = fig.subplots(2, 2)

        for ax, centers, widths, axis, color in ((axes[0, 0], x_centers, dx_w, 'x', '#2166AC'),
                                                 (axes[0, 1], y_centers, dy_w, 'y', '#B2182B')):
            ax.plot(centers / frame.scale, widths / cell_scale, '-', color=color, linewidth=1.2)
            ax.fill_between(centers / frame.scale, 0, widths / cell_scale, alpha=0.15, color=color)
            ax.set_xlabel(f'{axis} ({frame.unit})')
            ax.set_ylabel(f'd{axis} ({cell_unit})')
            ax.set_title(f'Cell width along {axis}')
            ax.grid(True, alpha=0.2)
            ax.set_ylim(bottom=0)

        self._draw_cell_area(axes[1, 0], frame, 'Cell area')

        ax = axes[1, 1]
        sizes = np.concatenate([dx_w, dy_w]) / cell_scale
        ax.hist(sizes, bins=50, color='#4DBEEE', edgecolor='white', linewidth=0.5, alpha=0.85)
        ax.axvline(sizes.min(), color='red', linestyle='--', linewidth=1, label=f'min: {sizes.min():.3g} {cell_unit}')
        ax.axvline(sizes.max(), color='blue', linestyle='--', linewidth=1, label=f'max: {sizes.max():.3g} {cell_unit}')
        ax.set_xlabel(f'Cell size ({cell_unit})')
        ax.set_ylabel('Count')
        ax.set_title('Cell size distribution')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

        fig.suptitle(f'Mesh: {dx_w.size}×{dy_w.size} = {dx_w.size * dy_w.size:,} cells (physical window)',
                     fontsize=12, fontweight='bold')
        return _finish(fig, filename)

    def plot_mesh_quality(self, filename=None):
        """Adjacent cell-size ratios along x / y against each axis' MeshConfig ratio limits (non-uniform grids)."""
        frame = self._frame()
        dx_w, dy_w = self._window_widths()
        x_edges, y_edges = frame.window_edges()
        ratio_x = dx_w[1:] / dx_w[:-1]
        ratio_y = dy_w[1:] / dy_w[:-1]
        mesh_config = self.domain.mesh_config
        fig = _new_figure((15, 4))
        axes = fig.subplots(1, 3)
        histogram_limits = {}   # (name, ratio) -> axes it limits

        for ax, interior_edges, ratio, axis, index, color in (
                (axes[0], x_edges[1:-1], ratio_x, 'x', 'i', '#2166AC'),
                (axes[1], y_edges[1:-1], ratio_y, 'y', 'j', '#B2182B')):
            ax.plot(interior_edges / frame.scale, ratio, '-', color=color, linewidth=0.8)
            ax.axhline(1.0, color='gray', linestyle='-', linewidth=0.5)
            limits = _mesh_ratio_limits(mesh_config, axis)
            for (name, limit), line_color in zip(limits, ('orange', 'purple')):
                ax.axhline(limit, color=line_color, linestyle='--', linewidth=1, label=f'{name} {limit:g}')
                ax.axhline(1 / limit, color=line_color, linestyle='--', linewidth=1)
                histogram_limits.setdefault((name, limit), []).append(axis)
            if limits:
                ax.legend(fontsize=7)
            worst = float(np.max(np.maximum(ratio, 1 / ratio))) if ratio.size else 1.0
            custom = ' (custom cells)' if mesh_config is not None and axis in mesh_config.custom_axes() else ''
            ax.set_xlabel(f'{axis} ({frame.unit})')
            ax.set_ylabel(f'd{axis}[{index}+1] / d{axis}[{index}]')
            ax.set_title(f'Adjacent ratio ({axis}){custom}, max={worst:.4f}')
            ax.grid(True, alpha=0.2)

        ax = axes[2]
        all_ratios = np.concatenate([ratio_x, ratio_y])
        if all_ratios.size:
            ax.hist(all_ratios, bins=60, color='#77AC30', edgecolor='white', linewidth=0.3, alpha=0.85)
        for ((name, limit), limited_axes), line_color in zip(histogram_limits.items(),
                                                              ('orange', 'purple', 'brown', 'teal')):
            ax.axvline(limit, color=line_color, linestyle='--', label=f"{name} {' / '.join(limited_axes)} {limit:g}")
            ax.axvline(1 / limit, color=line_color, linestyle='--')
        if histogram_limits:
            ax.legend(fontsize=7)
        ax.set_xlabel('Adjacent cell ratio')
        ax.set_ylabel('Count')
        ax.set_title('Ratio distribution')
        ax.grid(True, alpha=0.2)

        fig.suptitle('Mesh Quality', fontsize=12, fontweight='bold')
        return _finish(fig, filename)

    # ------------------------------------------------------------ field plots

    def plot_field(self, component=None, vmax=None, cmap=None, show_geometry=True, title=None, filename=None,
                   physics=None):
        """Plot a renderer field over the physical window (component None = primary field of `physics`, default the
        first physics)."""
        renderer, component = self._resolve_field(component, physics)
        info = renderer.field_info(component)
        if cmap is not None:
            info = replace(info, cmap=cmap)
        frame = self._frame()
        x_edges, y_edges = frame.window_edges()
        fig = _field_map_figure(renderer.physical_field(component), frame.extent, info,
                                title if title is not None else _field_title(renderer, info), x_edges, y_edges, vmax)
        if show_geometry:
            self._draw_material_contours(fig.axes[0], frame, _contour_color(info.cmap))
        return _finish(fig, filename)

    # ------------------------------------------------------------ probe / EM monitor plots

    def plot_probe(self, probe, label=None, filename=None):
        result = self._probe_result(probe)
        if result['t'].size == 0:
            raise ValueError("probe has no samples; run the simulation first")
        fig = _new_figure((10, 3.5))
        _draw_timeseries(fig.subplots(), result, label or f"Field probe ({result['component']})")
        return _finish(fig, filename)

    def plot_probes(self, probes=None, labels=None, filename=None):
        """One time-series panel per probe (default: every probe of every physics)."""
        probes = self._probes() if probes is None else list(probes)
        if not probes:
            raise ValueError("no probes to plot")
        frame = self._frame()
        fig = _new_figure((10, 2.5 * len(probes)))
        axes = fig.subplots(len(probes), 1, squeeze=False)
        for index, (ax, probe) in enumerate(zip(axes[:, 0], probes)):
            title = labels[index] if labels and index < len(labels) else self._probe_label(probe, frame)
            _draw_timeseries(ax, self._probe_result(probe), title)
            ax.title.set_fontsize(10)
        fig.suptitle('Probes', fontsize=12, fontweight='bold')
        return _finish(fig, filename)

    def plot_spectrum(self, probe, wavelength_range=None, label=None, filename=None):
        """Normalised |FFT|² of a probe signal vs wavelength (auto band unless wavelength_range [m] is given)."""
        spectrum = _probe_spectrum(self._probe_result(probe))
        if spectrum is None:
            raise ValueError("probe needs >= 4 samples with a positive time step for a spectrum; run the simulation first")
        wavelength, power = spectrum
        keep = _spectrum_mask(wavelength, power, wavelength_range)
        peak = _finite_peak(power[keep])
        if not keep.any() or peak <= 0:
            raise ValueError("probe spectrum is empty in the plotted wavelength range")
        wl_scale, wl_unit = WAVELENGTH_UNIT
        normalised = power[keep] / peak
        fig = _new_figure((9, 4))
        ax = fig.subplots()
        ax.plot(wavelength[keep] / wl_scale, normalised, '-', color='#7E2F8E', linewidth=1.2)
        ax.fill_between(wavelength[keep] / wl_scale, 0, normalised, alpha=0.15, color='#7E2F8E')
        ax.set_xlabel(f'Wavelength ({wl_unit})')
        ax.set_ylabel(f'Normalised |{probe.field_component}(λ)|²')
        ax.set_title(label or 'Spectrum')
        ax.grid(True, alpha=0.2)
        return _finish(fig, filename)

    def plot_flux(self, flux_monitor, wavelength_range=None, label=None, filename=None):
        """Flux spectrum vs wavelength; positive = power toward +x (axis 'x') or +y (axis 'y')."""
        fig = _new_figure((9, 4))
        ax = fig.subplots()
        wl_scale = self._draw_flux_spectra(ax, [flux_monitor], label or 'Flux spectrum')
        if wavelength_range:
            ax.set_xlim(wavelength_range[0] / wl_scale, wavelength_range[1] / wl_scale)
        return _finish(fig, filename)

    def plot_transmission(self, flux_monitor, reference_flux, wavelength_range=None, label=None, filename=None):
        """Flux / reference flux vs wavelength (reference_flux: the 'power' of the same monitor's flux_spectrum in a
        reference run)."""
        result = _flux_spectrum(flux_monitor)
        reference = np.asarray(reference_flux, dtype=np.float64)
        if reference.shape != result['power'].shape:
            raise ValueError(f"reference_flux shape {reference.shape} != monitor spectrum shape "
                             f"{result['power'].shape}")
        transmission = safe_ratio(result['power'], reference)   # NaN where the reference vanishes
        wl_scale, wl_unit = WAVELENGTH_UNIT
        fig = _new_figure((9, 4))
        ax = fig.subplots()
        ax.plot(result['wavelength'] / wl_scale, transmission, '-', color='#0072BD', linewidth=1.5)
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_xlabel(f'Wavelength ({wl_unit})')
        ax.set_ylabel('Transmission')
        ax.set_title(label or 'Transmission spectrum T(λ)')
        ax.grid(True, alpha=0.2)
        if wavelength_range:
            ax.set_xlim(wavelength_range[0] / wl_scale, wavelength_range[1] / wl_scale)
        return _finish(fig, filename)

    def plot_dft_fields(self, dft_monitor, show_geometry=True, freq_indices=None, filename=None):
        """DFT intensity over the DFT monitor box, one panel per freq_indices entry (default: at most 8 evenly spaced
        monitor frequencies)."""
        from ..physics.em.results import dft_result
        frame = self._frame()
        metadata = dft_result(dft_monitor, self.domain)
        wavelengths = np.atleast_1d(metadata['wavelength'])
        quantity = metadata['quantity']
        if wavelengths.size == 0:
            raise ValueError("DFT monitor has no frequencies")
        if freq_indices is None:
            n_panels = min(_MAX_DFT_PANELS, wavelengths.size)
            freq_indices = np.unique(np.linspace(0, wavelengths.size - 1, n_panels).round().astype(int))
        freq_indices = [int(index) for index in np.atleast_1d(freq_indices)]
        if not freq_indices:
            raise ValueError("freq_indices is empty")
        material_map = self._material_map() if show_geometry else None
        wl_scale, wl_unit = WAVELENGTH_UNIT
        cols = min(len(freq_indices), 4)
        rows = int(np.ceil(len(freq_indices) / cols))
        x_start, y_start = dft_monitor.x_slice.start or 0, dft_monitor.y_slice.start or 0
        box_x_edges, box_y_edges = dft_map_edges(metadata)
        box_cells = (slice(x_start, dft_monitor.x_slice.stop), slice(y_start, dft_monitor.y_slice.stop))
        fig = _new_figure((4 * cols, 3.5 * rows))
        axes = fig.subplots(rows, cols, squeeze=False)
        for panel_index, ax in enumerate(axes.flat):
            if panel_index >= len(freq_indices):
                ax.set_visible(False)
                continue
            index = freq_indices[panel_index]
            intensity = dft_result(dft_monitor, self.domain, freq_index=index)['intensity']
            _draw_map(ax, intensity, frame.extent, (frame.scale, frame.unit), _STYLE['intensity_cmap'],
                      color_limits(intensity, signed=False), f'{quantity} (a.u.)',
                      f'λ = {wavelengths[index] / wl_scale:.6g} {wl_unit}', box_x_edges, box_y_edges,
                      colorbar_size='4%')
            ax.title.set_fontsize(9)
            if material_map is not None:
                x_limits, y_limits = ax.get_xlim(), ax.get_ylim()
                self._draw_material_contours(ax, frame, 'white', material_map, cell_box=box_cells)
                ax.set_xlim(x_limits)
                ax.set_ylim(y_limits)
        fig.suptitle(f'{quantity} at the DFT frequencies', fontsize=12, fontweight='bold')
        return _finish(fig, filename)

    def plot_radiation_pattern(self, nf2ff_monitor, freq_idx=0, n_angles=360, filename=None):
        """Normalised far-field power, polar and dB vs angle, at freq_idx in [0, n): FDTD monitor frequency order, FDFD
        ascending solved wavelength (only that one frequency is evaluated)."""
        angles, power, wavelength = _far_field_pattern(nf2ff_monitor, freq_idx, n_angles)
        _, power_db = normalised_db(power)
        wl_scale, wl_unit = WAVELENGTH_UNIT
        wavelength_text = f'λ = {wavelength / wl_scale:.6g} {wl_unit}'

        fig = _new_figure((12, 5))
        _draw_far_field_panel(fig.add_subplot(1, 2, 1, projection='polar'), angles, [power], [None],
                              f'Radiation pattern\n{wavelength_text}')

        ax_db = fig.add_subplot(1, 2, 2)
        ax_db.plot(np.degrees(angles), power_db, '-', color='#0072BD', linewidth=1.2)
        ax_db.set_xlabel('Angle from +x (degrees)')
        ax_db.set_ylabel('Normalised power (dB)')
        ax_db.set_title(f'Far-field pattern  |  {wavelength_text}')
        ax_db.set_ylim(-30, 1)
        ax_db.grid(True, alpha=0.2)
        return _finish(fig, filename)

    # ------------------------------------------------------------ dashboard

    def plot_dashboard(self, probes=None, flux_monitors=None, component=None, filename=None):
        """2x3 overview: geometry, mesh, fields of every physics, probe signals and EM FDTD spectra (works heat-only;
        FDFD probes and flux monitors are only drawn on the geometry by default)."""
        from matplotlib.gridspec import GridSpec
        frame = self._frame()
        material_map = self._material_map()
        em_monitors = [monitor for em in self._em_solvers() for monitor in em.monitors]
        probes = self._probes() if probes is None else list(probes)
        if flux_monitors is None:   # FDTD FluxMonitors only (a.u.; FDFD W/m spectra: plot_flux)
            from ..physics.em.monitors import FluxMonitor
            flux_monitors = [monitor for monitor in em_monitors if isinstance(monitor, FluxMonitor)]
        em_probes = [probe for probe in probes if any(probe is monitor for monitor in em_monitors)]
        panel_fields = self._dashboard_fields(component)

        fig = _new_figure(_STYLE['figsize_dashboard'])
        grid = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

        self._draw_geometry(fig.add_subplot(grid[0, 0]), frame, 'Geometry', material_map)
        ax = fig.add_subplot(grid[0, 1])
        if frame.non_uniform:
            self._draw_cell_area(ax, frame, 'Mesh density', material_map)
        else:
            self._draw_uniform_mesh_summary(ax)
        self._draw_dashboard_field(fig.add_subplot(grid[0, 2]), frame, panel_fields, 0, material_map)
        self._draw_probe_signals(fig.add_subplot(grid[1, 0]), probes, frame)
        ax = fig.add_subplot(grid[1, 1])
        if flux_monitors:
            self._draw_flux_spectra(ax, flux_monitors, 'Flux spectra')
            ax.title.set_fontsize(10)
        elif not self._draw_probe_spectra(ax, em_probes, frame):
            self._draw_dashboard_field(ax, frame, panel_fields, 2, material_map)
        self._draw_dashboard_field(fig.add_subplot(grid[1, 2]), frame, panel_fields, 1, material_map)

        fig.suptitle('Open_sol — Simulation Dashboard', fontsize=14, fontweight='bold', y=0.98)
        return _finish(fig, filename)
