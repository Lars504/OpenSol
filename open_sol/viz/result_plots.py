"""PNG report figures of fields and feature-result dicts (pyplot-free Figures); ASCII text, never prints."""
import math
import re
from pathlib import Path

import numpy as np

from ..core.wavelengths import wavelength_label
from .common import color_limits, length_unit, normalised_db, time_unit
from .visualization import (_STYLE, WAVELENGTH_UNIT, _ascii_unit, _blank, _draw_far_field_panel, _draw_flux_panel,
                            _draw_map, _draw_timeseries, _extent_length_unit, _field_label, _new_figure, _with_unit,
                            dft_map_edges, field_data_axes, field_data_edges)

# result kind -> figure size (inches); xy: one unit panel (+3 inches per further panel); table: minimum (rows add height)
_RESULT_FIGURE_SIZES = {'timeseries': (10, 4), 'flux_spectrum': (9, 4), 'dft_fields': _STYLE['figsize_single'],
                        'far_field': (8, 6.5), 'xy': (9, 4), 'table': (10, 2.5)}
_MAX_FAR_FIELD_CURVES = 6
_MAX_TABLE_FIGURE_ROWS = 30
# ASCII SI prefixes by power of ten; only units in _PREFIXABLE_UNITS get one (kg, rad, ppm, K, dB, ... keep scale 1)
_SI_PREFIXES = {-24: 'y', -21: 'z', -18: 'a', -15: 'f', -12: 'p', -9: 'n', -6: 'u', -3: 'm', 0: '', 3: 'k', 6: 'M',
                9: 'G', 12: 'T', 15: 'P', 18: 'E', 21: 'Z', 24: 'Y'}
_PREFIXABLE_UNITS = frozenset({'Hz', 'W', 'V', 'A', 'J', 'N', 'Pa', 'F', 'S', 'C', 'T', 'Wb', 'H', 'eV', 'Ohm', 'ohm'})
_UNLABELLED = re.compile(r'_child\d+|')   # matplotlib's default label of an unlabelled line (or none)


def save_field_figure(field_data, path, title='', show_mesh=False, contours=0):
    """PNG of a FieldData styled by its FieldInfo (as plot_field_array); non-uniform native cells (x_centers /
    y_centers) draw over their exact cell edges, a triangle-mesh field (triangulation) by tripcolor with `contours`
    lines; show_mesh: the default title names the field's mesh (models with several meshes). Creates the parent
    directory; returns str(path)."""
    info = field_data.info
    axes = field_data_axes(field_data)
    extent = tuple(float(edge) for edge in field_data.extent)
    mesh = getattr(field_data, 'mesh', '')
    parts = [info.label] + [text for text in (field_data.time_label, f'mesh {mesh}' if show_mesh and mesh else '')
                            if text]
    fig = _new_figure(_STYLE['figsize_single'])
    _draw_map(fig.subplots(), field_data.values, extent, _ascii_unit(_extent_length_unit(extent)), info.cmap,
              color_limits(field_data.values, info.signed), _field_label(info), title or '  |  '.join(parts),
              *field_data_edges(field_data), triangulation=getattr(field_data, 'triangulation', None),
              contours=contours, axes=axes)
    return _save_png(fig, path)


def save_result_figure(result, path, title='', freq_index=None, curves=()):
    """PNG of a feature-result dict: timeseries, flux spectrum, far-field polar plot, DFT intensity map (freq_index
    picks a frequency of 'intensity_all'), xy curves or table; curves [(label, abscissa in the result's units, values,
    series)] are drawn dashed (far field: normalised, curve k in the colour of far_field_wavelengths(result)[k % n];
    a table draws none: plot analytic.table_xy of it instead). Creates the parent directory; returns str(path)."""
    if not isinstance(result, dict):
        raise TypeError(f'result must be a dict, got {type(result).__name__}')
    kind = result.get('kind')
    if kind not in _RESULT_FIGURE_SIZES:
        raise ValueError(f"cannot plot result kind {kind!r}; supported kinds: {', '.join(_RESULT_FIGURE_SIZES)}")
    if freq_index is not None and kind != 'dft_fields':
        raise ValueError(f'freq_index applies to dft_fields results only, not to {kind}')
    fig = _new_figure(_figure_size(result, kind))
    if kind == 'timeseries':
        ax = fig.add_subplot()
        _draw_timeseries(ax, result, title or f"Probe {result.get('component', '')}", ascii_text=True)
        times = np.asarray(result['t'], dtype=np.float64)
        if times.size > 1 and np.ptp(times) > 0:
            _draw_curves(ax, curves, _ascii_unit(time_unit(float(np.max(np.abs(times)))))[0], '#2166AC')
        elif times.size:   # 1-based iteration axis: curves of `iteration`
            _draw_curves(ax, curves, 1.0, '#2166AC')
    elif kind == 'flux_spectrum':
        ax = fig.add_subplot()
        wavelength_scale = _draw_flux_panel(ax, [result], title or 'Flux spectrum')
        _draw_curves(ax, curves, wavelength_scale, _STYLE['flux_color'])
    elif kind == 'far_field':
        ax = fig.add_subplot(projection='polar')
        wavelength_lines = _draw_far_field(ax, result, title)
        for position, (label, angles, values, series) in enumerate(curves):   # far_field_wavelengths order
            color = wavelength_lines[position % len(wavelength_lines)].get_color()
            _draw_curves(ax, [(label, angles, normalised_db(values)[0], series)], 1.0, color)
    elif kind == 'xy':
        _draw_xy(fig, result, title, curves)
    elif kind == 'table':
        _draw_table(fig.add_subplot(), result, title)
    else:
        _draw_dft_fields(fig.add_subplot(), result, title, freq_index)
    return _save_png(fig, path)


def save_line_profile_figure(distance, values, label, unit, path, title='', curves=()):
    """PNG of a field profile along a line (distance [m]), legend label as in the GUI, with analytic curves [(label,
    distance [m], values)] dashed in its colour as 'label (analytic)'. Creates the parent directory; returns str(path)."""
    distance = np.asarray(distance, dtype=np.float64)
    fig = _new_figure((9, 4))
    ax = fig.add_subplot()
    scale, unit_text = _ascii_unit(length_unit(float(np.max(distance)) if distance.size else 1e-6))
    (line,) = ax.plot(distance / scale, np.asarray(values, dtype=np.float64), '-', linewidth=1.2, label=label)
    for curve_label, curve_distance, curve_values in curves:
        ax.plot(np.asarray(curve_distance) / scale, curve_values, '--', linewidth=1.2, color=line.get_color(),
                label=f'{curve_label} (analytic)')
    ax.set_xlabel(f'Distance along the line ({unit_text})')
    ax.set_ylabel(_with_unit(label, unit))
    ax.set_title(title)
    _legend(ax)
    ax.grid(True, alpha=0.2)
    return _save_png(fig, path)


def xy_series_groups(result):
    """[(unit, series names, y label)] of an xy result, one per distinct series unit in series order; the first is
    labelled y_label, the others by the leading '_'-separated words their names share (else 'Value')."""
    series_units = result.get('series_units') or {}
    names_by_unit = {}
    for name in result['series']:
        names_by_unit.setdefault(series_units.get(name, ''), []).append(name)
    groups = []
    for index, (unit, names) in enumerate(names_by_unit.items()):
        label = (result.get('y_label') if index == 0 else '') or _shared_leading_words(names) or 'Value'
        groups.append((unit, names, label))
    return groups


def table_cell_text(value):
    """Display text of a table cell: None -> '', bools 'true' / 'false', floats '%.6g', anything else str()."""
    if value is None:
        return ''
    if isinstance(value, (bool, np.bool_)):
        return 'true' if value else 'false'
    if isinstance(value, (float, np.floating)):
        return f'{float(value):.6g}'
    return str(value)


# ---- result panels (timeseries, flux and map panels: visualization's shared _draw_* helpers) ----
def far_field_wavelengths(result):
    """The wavelengths [m] a far-field figure draws (at most 6, evenly chosen), in drawing order."""
    wavelengths = np.atleast_1d(np.asarray(result['wavelength'], dtype=np.float64))
    return wavelengths[_far_field_indices(wavelengths.size)]


def _far_field_indices(n_wavelengths):
    n_curves = min(_MAX_FAR_FIELD_CURVES, n_wavelengths)
    return np.linspace(0, n_wavelengths - 1, n_curves).round().astype(int)


def _draw_far_field(ax, result, title):
    """Polar plot (angle counter-clockwise from +x) of the far-field power of the result 'quantity' normalised per
    wavelength, for the far_field_wavelengths -> their lines in that order."""
    angles = np.asarray(result['angles'], dtype=np.float64)
    wavelengths = np.atleast_1d(np.asarray(result['wavelength'], dtype=np.float64))
    power = np.asarray(result['power'], dtype=np.float64).reshape(wavelengths.size, angles.size)
    if wavelengths.size == 0 or angles.size == 0:
        raise ValueError(f'the far-field result has {wavelengths.size} wavelengths and {angles.size} angles')
    indices = _far_field_indices(wavelengths.size)
    quantity = result.get('quantity', 'abs(E_z)^2')
    _draw_far_field_panel(ax, angles, power[indices], [wavelength_label(wavelengths[index]) for index in indices],
                          title or f'Far-field power {quantity}, normalised per wavelength')
    return ax.get_lines()[:indices.size]


def _draw_dft_fields(ax, result, title, freq_index):
    """DFT intensity map of the result 'quantity' on the DFT box cells over 'extent', with the wavelength in the
    title."""
    intensity, index = _dft_intensity(result, freq_index)
    extent = tuple(float(edge) for edge in result['extent'])
    heading = title or 'DFT field intensity'
    if index is not None:
        heading = f"{heading}  |  {wavelength_label(float(np.atleast_1d(result['wavelength'])[index]))}"
    _draw_map(ax, intensity, extent, _ascii_unit(_extent_length_unit(extent)), _STYLE['intensity_cmap'],
              color_limits(intensity, signed=False), f"{result.get('quantity', 'abs(E_z)^2')} (a.u.)", heading,
              *dft_map_edges(result))


def _draw_xy(fig, result, title, curves=()):
    """One panel per distinct series unit (xy_series_groups), each series of that unit vs x with a legend; NaN values
    leave gaps; axes get an SI prefix (_axis_scale: wavelengths in nm); result keys log_x / log_y (derived curve
    plots) make the axes logarithmic; series_x {name: own abscissa} draws a series over its own x (sweep overlays).
    Curves go dashed onto the panel, scale and colour of their series (series '' or unknown: the first series)."""
    groups = xy_series_groups(result)
    if not groups:
        ax = fig.add_subplot()
        _blank(ax, 'no series')
        ax.set_title(title or result.get('y_label', ''))
        return
    x = np.asarray(result['x'], dtype=np.float64)
    x_label = result.get('x_label') or 'x'
    x_scale, x_unit = _axis_scale(x, result.get('x_unit', ''), x_label)
    own_x = {name: np.asarray(values, dtype=np.float64) for name, values in (result.get('series_x') or {}).items()}
    marker_style = '-o' if x.size <= 60 else '-'
    axes = fig.subplots(len(groups), 1, sharex=True, squeeze=False)[:, 0]
    series_places = {}   # series name -> (axes, line colour, value scale)
    for ax, (unit, names, label) in zip(axes, groups):
        panel_values = {name: np.asarray(result['series'][name], dtype=np.float64) for name in names}
        y_scale, y_unit = _axis_scale(np.concatenate(list(panel_values.values())), unit, label)
        for name, values in panel_values.items():
            (line,) = ax.plot(own_x.get(name, x) / x_scale, values / y_scale, marker_style, markersize=3,
                              linewidth=1.2, label=name)
            series_places[name] = (ax, line.get_color(), y_scale)
        ax.set_ylabel(_with_unit(label, y_unit))
        ax.grid(True, alpha=0.2)
        if result.get('log_x'):
            ax.set_xscale('log')
        if result.get('log_y'):
            ax.set_yscale('log')
    first_place = series_places[groups[0][1][0]]
    for label, abscissa, values, series in curves:
        ax, color, y_scale = series_places.get(series, first_place)
        _draw_curves(ax, [(label, abscissa, values, series)], x_scale, color, y_scale)
    for ax in axes:
        _legend(ax)
    axes[-1].set_xlabel(_with_unit(x_label, x_unit))
    axes[0].set_title(title or result.get('y_label', ''))


def _draw_curves(ax, curves, abscissa_scale, color, value_scale=1.0):
    """Analytic curves [(label, abscissa, values, series)] dashed in color over axes scaled by abscissa_scale /
    value_scale."""
    for label, abscissa, values, _series in curves:
        ax.plot(np.asarray(abscissa, dtype=np.float64) / abscissa_scale,
                np.asarray(values, dtype=np.float64) / value_scale, '--', linewidth=1.2, color=color,
                label=f'{label} (analytic)')
    if curves:
        _legend(ax)


def _legend(ax):
    """Legend of every labelled line given explicitly (matplotlib's automatic legend drops labels starting with '_',
    a valid curve / series name); unlabelled lines carry matplotlib's '_child<n>' and stay out."""
    lines = [line for line in ax.get_lines() if not _UNLABELLED.fullmatch(str(line.get_label()))]
    if lines:
        ax.legend(handles=lines, labels=[line.get_label() for line in lines], fontsize=7)


def _draw_table(ax, result, title):
    """matplotlib table of the first 30 rows under 'column (unit)' headers; the title notes a cut."""
    columns = [str(column) for column in result['columns']]
    units = [str(unit) for unit in result.get('units') or [''] * len(columns)]
    rows = list(result['rows'])
    ax.axis('off')
    heading = title or 'Table'
    if len(rows) > _MAX_TABLE_FIGURE_ROWS:
        heading = f'{heading}  |  first {_MAX_TABLE_FIGURE_ROWS} of {len(rows)} rows'
    ax.set_title(heading)
    if not columns or not rows:
        _blank(ax, f"no rows (columns: {', '.join(columns) or 'none'})")
        return
    table = ax.table(cellText=[[table_cell_text(value) for value in row] for row in rows[:_MAX_TABLE_FIGURE_ROWS]],
                     colLabels=[_with_unit(column, unit) for column, unit in zip(columns, units)],
                     loc='upper center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(list(range(len(columns))))
    table.scale(1.0, 1.3)


# ---- helpers ----
def _figure_size(result, kind):
    """Figure size (inches) of a result: _RESULT_FIGURE_SIZES, grown by the xy unit panels or the table rows shown."""
    width, height = _RESULT_FIGURE_SIZES[kind]
    if kind == 'xy':
        return width, height + 3.0 * max(0, len(xy_series_groups(result)) - 1)
    if kind == 'table':
        return width, max(height, 1.2 + 0.3 * (min(len(result['rows']), _MAX_TABLE_FIGURE_ROWS) + 1))
    return width, height


def _axis_scale(values, unit, label=''):
    """(scale, ASCII unit) of an axis from the largest finite |value|: wavelengths (label 'Wavelength...', unit m) in
    nm, m / s by length_unit / time_unit, the SI units of _PREFIXABLE_UNITS (Hz, W, V, ...) an SI prefix; the rest 1."""
    if unit == 'm' and str(label).strip().lower().startswith('wavelength'):
        return WAVELENGTH_UNIT
    finite_magnitudes = np.abs(values[np.isfinite(values)])
    largest = float(finite_magnitudes.max()) if finite_magnitudes.size else 0.0
    if largest <= 0.0:
        return 1.0, unit
    if unit == 'm':
        return _ascii_unit(length_unit(largest))
    if unit == 's':
        return _ascii_unit(time_unit(largest))
    if unit not in _PREFIXABLE_UNITS:
        return 1.0, unit
    exponent = min(24, max(-24, 3 * math.floor(math.log10(largest) / 3)))
    return 10.0 ** exponent, f'{_SI_PREFIXES[exponent]}{unit}'


def _shared_leading_words(names):
    """Leading '_'-separated words shared by every name, joined by spaces ('' when none)."""
    shared_words = []
    for words in zip(*(str(name).split('_') for name in names)):
        if len(set(words)) != 1:
            break
        shared_words.append(words[0])
    return ' '.join(shared_words)


def _dft_intensity(result, freq_index):
    """(intensity (mx, my), frequency index or None) of a dft_fields result: the fetched 'intensity', else
    'intensity_all'[freq_index] (default: the frequency with the largest maximum)."""
    if 'intensity' in result and freq_index in (None, result.get('freq_index')):
        return np.asarray(result['intensity']), result.get('freq_index')
    if 'intensity_all' in result:
        stack = np.asarray(result['intensity_all'])
        if freq_index is None:
            freq_index = int(np.argmax(stack.max(axis=(1, 2))))
        elif not 0 <= freq_index < stack.shape[0]:
            raise IndexError(f'freq_index {freq_index} out of range for {stack.shape[0]} DFT frequencies')
        return stack[freq_index], int(freq_index)
    requested = '' if freq_index is None else f' at freq_index {freq_index}'
    raise ValueError(f'this dft_fields result holds no intensity{requested}: fetch one frequency with freq_index '
                     f'or pass a result holding intensity_all (n_f, mx, my)')


def _save_png(fig, path):
    """tight_layout and save at the report dpi on a white background (parent directories created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=_STYLE['dpi'], bbox_inches='tight', facecolor='white')
    return str(path)
