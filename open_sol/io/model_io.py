"""Model files (UTF-8 JSON) and result export (.npz / .csv); CPU only (numpy), safe on the GUI thread."""
import csv
import json
import os
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np

from ..model.analytic import is_iteration_series, numeric_table_columns, table_xy
from ..model.spec import ModelSpec
from ..viz.common import edges_from_centers, normalised_db

_EXPORT_FORMATS = ('.npz', '.csv')
# export_result kinds read_curve_data also plots ('line_profile': the GUI's 'Raw profile…')
_RAW_CURVE_KINDS = ('timeseries', 'flux_spectrum', 'far_field', 'line_profile')
_FAR_FIELD_DB_FLOOR = -60.0   # dB floor of a read far field (as the GUI far-field view draws it)
# .npz archives up to this many array bytes are zipped in memory and written once: np.savez through a file object
# seeks and flushes twice per member (~45 us each, 0.46 s for a 2571-member results file)
_NPZ_BUFFER_BYTES = 256 * 1024 ** 2
_CSV_TEXT_KEYS = frozenset({'kind', 'x_label', 'x_unit', 'y_label', 'model', 'plot'})   # read back as text, not JSON
_CURVE_PLOT_KEYS = {'x_label': '', 'x_unit': '', 'y_label': '', 'series_units': {}, 'log_x': False,
                    'log_y': False}   # per-plot metadata of an export_curve_plots .npz (key: default)


def save_model(spec, path):
    """Write the spec as UTF-8 JSON (indent 2); the file is replaced only after the full text was written."""
    path = Path(path)
    try:
        text = json.dumps(spec.to_dict(), indent=2, ensure_ascii=False, allow_nan=False)
    except ValueError as exc:
        raise ValueError(f'cannot save the model to {path}: {exc}') from exc
    _write_text_atomic(path, text + '\n')


def load_model(path):
    """ModelSpec from a JSON model file; ValueError naming the file for invalid JSON or an invalid model."""
    path = Path(path)
    try:
        return ModelSpec.from_dict(json.loads(path.read_text(encoding='utf-8-sig')))
    except ValueError as exc:   # JSONDecodeError and UnicodeDecodeError are ValueErrors; OSError propagates
        raise ValueError(f'{path}: {exc}') from exc


def export_field(field_data, path):
    """Write a FieldData: .npz (values, extent, x/y centres and edges = the first / second axis, metadata JSON) or
    .csv (u, v, value rows, u-major); the metadata flags a display resampling ('resampled', 'native_shape'), names a 3D cut
    ('plane') and the value axes ('axes', e.g. ['x', 'z'] for a y cut; ['r', 'phi' | 'theta'] on native polar /
    spherical cells, which add the drawing-plane centres: .npz 'x_mapped' / 'y_mapped' (nu, nv), .csv x / y columns),
    the model mesh ('mesh', S7.2); a triangle-mesh field's .npz adds its display triangulation 'nodes' (n, 2),
    'triangles' (k, 3) and 'node_values' (values = the raster)."""
    path = Path(path)
    file_format = _export_format(path)
    values = np.asarray(field_data.values)
    if values.ndim != 2:
        raise ValueError(f'field values must be 2D (nx, ny), got shape {values.shape}')
    extent = tuple(float(edge) for edge in field_data.extent)
    if len(extent) != 4:
        raise ValueError(f'field extent must be (x0, x1, y0, y1), got {field_data.extent!r}')
    x_centers = _axis_centers(field_data.x_centers, extent[0], extent[1], values.shape[0], 'x')
    y_centers = _axis_centers(field_data.y_centers, extent[2], extent[3], values.shape[1], 'y')
    info = field_data.info
    plane = getattr(field_data, 'plane', None)   # duck-typed field objects without one: 2D
    axes = [str(axis) for axis in getattr(field_data, 'axes', ('x', 'y'))]   # side cuts: ['x', 'z'] / ['y', 'z']
    metadata = {'physics': field_data.physics, 'component': field_data.component, 'label': info.label,
                'unit': info.unit, 'time': field_data.time, 'time_label': field_data.time_label,
                'extent': list(extent), 'resampled': field_data.native_shape is not None,
                'plane': None if plane is None else list(plane), 'axes': axes,   # 3D: [normal axis, position m]
                'mesh': str(getattr(field_data, 'mesh', '') or '')}
    if field_data.native_shape is not None:   # nearest-cell display data: not for overlaps / integrals
        metadata['native_shape'] = [int(count) for count in field_data.native_shape]
    mapped = getattr(field_data, 'mapped_centres', None)   # native polar / spherical cells
    mapped = None if mapped is None else [np.asarray(centres, dtype=np.float64) for centres in mapped]
    if mapped is not None and any(centres.shape != values.shape for centres in mapped):
        raise ValueError(f'mapped centres {[centres.shape for centres in mapped]} do not match values {values.shape}')
    if file_format == '.npz':
        arrays = {'values': values, 'extent': np.array(extent, dtype=np.float64), 'x_centers': x_centers,
                  'y_centers': y_centers,   # + cell edges, as results.npz fields
                  'x_edges': edges_from_centers(x_centers, extent[0], extent[1]),
                  'y_edges': edges_from_centers(y_centers, extent[2], extent[3])}
        if mapped is not None:
            arrays.update(x_mapped=mapped[0], y_mapped=mapped[1])
        triangulation = getattr(field_data, 'triangulation', None)   # triangle meshes (S7.5)
        if triangulation is not None:
            arrays.update(nodes=np.asarray(triangulation[0], dtype=np.float64),
                          triangles=np.asarray(triangulation[1], dtype=np.int32),
                          node_values=np.asarray(triangulation[2], dtype=np.float32))
        _write_npz(path, arrays, metadata)
        return
    x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing='ij')
    names, columns = [*axes], [x_grid.ravel(), y_grid.ravel()]
    if mapped is not None:
        names += ['x', 'y']
        columns += [centres.ravel() for centres in mapped]
    _write_csv(path, [*names, 'value'], [*columns, values.ravel()], metadata)


def export_volume(volume, path):
    """Write a session.VolumeData as .npz only: values (wx, wy, wz) float32, x / y / z cell centres [m] float64 and
    the metadata JSON (kind 'volume')."""
    path = Path(path)
    if _export_format(path) != '.npz':
        raise ValueError(f'a 3D volume exports to .npz only ({path.name}); .csv takes cut planes (export_field)')
    values = np.asarray(volume.values)
    centres = [np.asarray(axis_centres, dtype=np.float64) for axis_centres in volume.centres]
    if values.ndim != 3 or tuple(axis_centres.size for axis_centres in centres) != values.shape:
        raise ValueError(f'volume values {values.shape} and centres {[c.shape for c in centres]} do not match')
    info = volume.info
    _write_npz(path, {'values': values, 'x_centers': centres[0], 'y_centers': centres[1], 'z_centers': centres[2]},
               {'kind': 'volume', 'physics': volume.physics, 'component': volume.component, 'label': info.label,
                'unit': info.unit, 'time': volume.time, 'time_label': volume.time_label})


def export_result(result, path):
    """Write a feature result dict: .npz (numeric arrays, dict values' arrays as '<key>/<name>', metadata JSON) or .csv
    (columns; far field: angle + power per wavelength; DFT: x, y, intensity rows; xy: x + series; table: rows)."""
    path = Path(path)
    file_format = _export_format(path)
    if not isinstance(result, dict):
        raise TypeError(f'result must be a dict, got {type(result).__name__}')
    kind = result.get('kind')
    arrays, metadata = _export_entries(result, kind)
    if file_format == '.npz':
        _write_npz(path, arrays, metadata)
        return
    if kind == 'table':   # numeric extras (arrays) go to the metadata lines: a table has no array columns
        table_metadata = {key: value for key, value in metadata.items() if key not in ('columns', 'rows')}
        table_metadata.update((key, array.tolist()) for key, array in arrays.items())
        _write_table_csv(path, result['columns'], result['rows'], table_metadata)
        return
    names, columns, used_keys = _result_columns(kind, arrays)
    for key, array in arrays.items():
        if key not in used_keys:   # e.g. every DFT frequency: the header keeps them (no silent loss)
            metadata[key] = array.tolist()
    _write_csv(path, names, columns, metadata)


def export_curve_plots(plots, path, model=''):
    """Write curve plots {plot name: xy result} with the model title and the first plot's parameters: .npz (every
    plot: 'plot<i>/x', 'plot<i>/series<j>' arrays, names / axes / units in the metadata) or .csv (one plot only)."""
    path = Path(path)
    file_format = _export_format(path)
    if not plots:
        raise ValueError('no curve plots to export')
    parameters = dict(next(iter(plots.values())).get('parameters') or {})
    if file_format == '.csv':
        if len(plots) != 1:
            raise ValueError(f'{len(plots)} plots do not fit one .csv table; export .npz')
        (name, xy), = plots.items()
        export_result(dict(xy, plot=name, model=model, parameters=parameters), path)
        return
    arrays, entries = {}, []
    for plot_index, (name, xy) in enumerate(plots.items()):
        arrays[f'plot{plot_index}/x'] = np.asarray(xy['x'], dtype=np.float64)
        for series_index, values in enumerate(xy['series'].values()):
            arrays[f'plot{plot_index}/series{series_index}'] = np.asarray(values, dtype=np.float64)
        entries.append({'name': name, 'series': list(xy['series']),
                        **{key: xy.get(key, default) for key, default in _CURVE_PLOT_KEYS.items()}})
    _write_npz(path, arrays, {'kind': 'curve_plots', 'model': model, 'parameters': parameters, 'plots': entries})


def read_curve_data(path):
    """{plot name: xy result} of a curve file: an export_curve_plots .npz (every plot), an exported xy, timeseries,
    flux_spectrum, far_field (one plot per wavelength, dB), line_profile or table result (.csv / .npz; named by its
    'plot' metadata, else the file name) or any numeric .csv (first column x, the others series; '#' lines skipped,
    optional column-name row); ValueError naming the file when it holds no curves."""
    path = Path(path)
    try:
        if _export_format(path) == '.npz':
            return _read_npz_curves(path)
        return _read_csv_curves(path)
    except (ValueError, KeyError) as exc:
        raise ValueError(f'{path.name}: {exc}') from exc


# ---- helpers ----
def _write_text_atomic(path, text):
    """Write UTF-8 text to '<file>.partial', then os.replace it over `path` (never a half-written file)."""
    partial_path = path.with_name(path.name + '.partial')
    try:
        partial_path.write_text(text, encoding='utf-8')
        os.replace(partial_path, path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise


def _export_format(path):
    """'.npz' or '.csv' from the file suffix; ValueError otherwise."""
    suffix = path.suffix.lower()
    if suffix not in _EXPORT_FORMATS:
        raise ValueError(f"unsupported export format {path.suffix!r} ({path.name}); use {' or '.join(_EXPORT_FORMATS)}")
    return suffix


def _axis_centers(centers, low_edge, high_edge, n_cells, axis):
    """Given cell centres (float64), or uniform centres spanning [low_edge, high_edge]."""
    if centers is None:
        return low_edge + (np.arange(n_cells) + 0.5) * ((high_edge - low_edge) / n_cells)
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (n_cells,):
        raise ValueError(f'{axis}_centers has shape {centers.shape}, expected ({n_cells},)')
    return centers


def _numeric_array(value):
    """numpy array of a numeric (>= 1-D) ndarray / non-empty numeric list or tuple; None for anything else."""
    if isinstance(value, np.ndarray):
        if value.ndim == 0:   # a scalar: metadata, not a column
            return None
        array = value
    elif isinstance(value, (list, tuple)) and value:
        try:
            array = np.asarray(value)
        except ValueError:   # ragged nested lists
            return None
    else:
        return None
    return array if array.dtype.kind in 'biufc' else None


def _export_entries(result, kind):
    """(arrays, metadata) of a result: numeric arrays, also one level into dict values as '<key>/<name>'; everything
    else (a table's rows always) is metadata."""
    arrays, metadata = {}, {}
    for key, value in result.items():
        if isinstance(value, dict):
            nested_metadata = {}
            for name, item in value.items():
                array = _numeric_array(item)
                if array is None:
                    nested_metadata[name] = item
                else:
                    arrays[f'{key}/{name}'] = array
            if nested_metadata or not value:
                metadata[key] = nested_metadata
            continue
        array = None if kind == 'table' and key == 'rows' else _numeric_array(value)
        if array is None:
            metadata[key] = value
        else:
            arrays[key] = array
    return arrays, metadata


def _result_columns(kind, arrays):
    """(column names, 1D columns, array keys used) of a result for .csv; ValueError when it has no table form."""
    if kind == 'xy':
        x = arrays['x']
        if x.ndim != 1:
            raise ValueError(f'xy x has shape {x.shape}, expected (n,); export .npz')
        series_keys = [key for key in arrays if key.startswith('series/')]
        mismatched = {key: arrays[key].shape for key in series_keys if arrays[key].shape != x.shape}
        if mismatched:
            raise ValueError(f'xy series shapes {mismatched} differ from x {x.shape}; export .npz')
        names = ['x'] + [key[len('series/'):] for key in series_keys]
        return names, [x] + [arrays[key] for key in series_keys], {'x', *series_keys}
    if kind == 'far_field':
        angles = arrays['angles']
        power = np.atleast_2d(arrays['power'])
        wavelengths = np.atleast_1d(arrays['wavelength'])
        if power.shape != (wavelengths.size, angles.size):
            raise ValueError(f'far-field power has shape {power.shape}, expected '
                             f'({wavelengths.size}, {angles.size}) (wavelengths, angles)')
        power_names = _wavelength_column_names(wavelengths)
        wavelength_named = all(name.endswith('nm') for name in power_names)   # else the metadata lists them
        return (['angle_rad'] + power_names, [angles, *power],
                {'angles', 'power'} | ({'wavelength'} if wavelength_named else set()))
    if kind == 'dft_fields':
        if 'intensity' not in arrays:
            raise ValueError('this DFT result holds no field data: fetch one frequency (freq_index) before a .csv '
                             'export, or export .npz')
        intensity = arrays['intensity']
        x_centers, y_centers = arrays['x_centers'], arrays['y_centers']
        if intensity.shape != (x_centers.size, y_centers.size):
            raise ValueError(f'DFT intensity has shape {intensity.shape}, expected ({x_centers.size}, '
                             f'{y_centers.size})')
        x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing='ij')
        names, columns = ['x', 'y', 'intensity'], [x_grid.ravel(), y_grid.ravel(), intensity.ravel()]
        used_keys = {'x_centers', 'y_centers', 'intensity'}
        if 'hz_intensity' in arrays:   # H_z DFT: |H_z(omega)|^2 on the same cell centres
            hz_intensity = arrays['hz_intensity']
            if hz_intensity.shape != intensity.shape:
                raise ValueError(f'DFT hz_intensity has shape {hz_intensity.shape}, expected {intensity.shape}')
            names.append('hz_intensity')
            columns.append(hz_intensity.ravel())
            used_keys.add('hz_intensity')
        return names, columns, used_keys
    multi_dimensional = [key for key, array in arrays.items() if array.ndim > 1]
    if multi_dimensional:
        raise ValueError(f"arrays {multi_dimensional} are not 1D and cannot be written as .csv columns; "
                         f"export .npz")
    lengths = {key: array.size for key, array in arrays.items()}
    if not lengths:
        raise ValueError('the result has no arrays to write as .csv columns')
    if len(set(lengths.values())) > 1:
        raise ValueError(f'the result arrays have different lengths {lengths}; export .npz')
    return list(arrays), list(arrays.values()), set(arrays)


def _wavelength_column_names(wavelengths):
    """'power_<nm>nm' far-field column names with the fewest digits (6 or more) that keep close wavelengths apart."""
    for digits in (6, 9, 12, 17):
        names = [f'power_{wavelength * 1e9:.{digits}g}nm' for wavelength in wavelengths]
        if len(set(names)) == len(names):
            return names
    return [f'power_{index}' for index in range(len(wavelengths))]   # repeated wavelengths


def _write_csv(path, names, columns, metadata):
    """Comma-separated numeric table with '# key: value' metadata lines above the column-name row."""
    for name, column in zip(names, columns):
        if np.iscomplexobj(column):
            raise ValueError(f"column '{name}' is complex; export .npz")
    table = np.column_stack([np.asarray(column, dtype=np.float64) for column in columns])
    header_lines = _csv_metadata_lines(metadata) + [_csv_row_text(names)]
    np.savetxt(path, table, fmt='%.9g', delimiter=',', header='\n'.join(header_lines), comments='',
               encoding='utf-8')


def _write_table_csv(path, columns, rows, metadata):
    """Mixed-type table (csv module; floats %.9g, None empty, bools true / false) below '# key: value' lines."""
    column_names = [str(column) for column in columns]
    for index, row in enumerate(rows):
        if len(row) != len(column_names):
            raise ValueError(f'table row {index} has {len(row)} values for {len(column_names)} columns')
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        handle.write(''.join(f'{line}\n' for line in _csv_metadata_lines(metadata)))
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(column_names)
        writer.writerows([_csv_cell_text(value) for value in row] for row in rows)


def _csv_metadata_lines(metadata):
    """'# key: value' lines: text as is when it reads back unchanged (_plain_metadata_text), anything else as JSON."""
    return [f'# {key}: {value if _plain_metadata_text(key, value) else _to_json(value)}'
            for key, value in metadata.items()]


def _plain_metadata_text(key, value):
    """True for a string that _metadata_value reads back as itself: one line, no edge spaces, no leading quote and,
    outside _CSV_TEXT_KEYS, not valid JSON ('1', 'true')."""
    if not isinstance(value, str) or value != value.strip() or value.startswith('"'):
        return False
    if value and value.splitlines() != [value]:   # any line break (also \r, \x0b and the Unicode separators)
        return False
    return key in _CSV_TEXT_KEYS or not _parses_as_json(value)


def _parses_as_json(text):
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def _csv_row_text(cells):
    """One CSV row without line end, cells quoted where they hold a comma, quote or line break."""
    buffer = StringIO()
    csv.writer(buffer, lineterminator='').writerow(cells)
    return buffer.getvalue()


def _read_csv_curves(path):
    """{plot name: xy dict} of a .csv: leading '# key: value' metadata (JSON, _CSV_TEXT_KEYS as text), optional
    column-name row (csv-quoted, line breaks included; kept exactly in an open_sol export, else stripped), numeric rows
    (empty cells NaN; a 'table' export's cells as written); later '#' rows are skipped. ValueError for repeated curve
    column names."""
    with open(path, encoding='utf-8-sig', newline='') as handle:   # newline='': quoted line breaks stay in a cell
        lines = handle.read().replace('\r\n', '\n').split('\n')   # np.savetxt writes \r\n on Windows, also in cells
    metadata, first_data_line = {}, 0
    for line in lines:
        if line.strip() and not line.startswith('#'):
            break
        key, separator, value = line[1:].partition(':')
        if line.startswith('#') and separator:
            metadata[key.strip()] = _metadata_value(key.strip(), value.strip())
        first_data_line += 1
    rows = [row for row in csv.reader(StringIO('\n'.join(lines[first_data_line:])))
            if any(cell.strip() for cell in row) and not row[0].startswith('#')]
    kind = metadata.get('kind', 'xy')
    if rows and (kind == 'table' or not all(_is_number(cell) for cell in rows[0])):
        exact_names = 'kind' in metadata   # written by export_result: names are free text, spaces included
        names, rows = [cell if exact_names else cell.strip() for cell in rows[0]], rows[1:]
    else:
        names = ['x'] + [f'column {index}' for index in range(1, len(rows[0]) if rows else 0)]
    if kind == 'table':
        return _result_plots(dict(metadata, columns=names, rows=[[_table_cell_value(cell) for cell in row]
                                                                 for row in rows]), path)
    if len(names) < 2 or not rows:
        raise ValueError('expected a column of x values and at least one column of curve values')
    repeated = sorted({name for name in names[1:] if names[1:].count(name) > 1})
    if repeated:
        raise ValueError(f"repeated column names {', '.join(map(repr, repeated))}: each curve needs its own name")
    widths = sorted({len(row) for row in rows})
    if widths != [len(names)]:
        raise ValueError(f'rows have {widths} values for {len(names)} columns')
    table = np.array([[float(cell) if cell.strip() else np.nan for cell in row] for row in rows], dtype=np.float64)
    if kind in _RAW_CURVE_KINDS:   # an export_result of a probe, spectrum or far field: columns by name
        return _result_plots(dict(metadata, **dict(zip(names, table.T))), path)
    return _named_plot(_curve_dict(metadata, names[0], table[:, 0], dict(zip(names[1:], table[:, 1:].T))), path)


def _read_npz_curves(path):
    """{plot name: xy dict} of an export_curve_plots .npz, an exported xy .npz ('x', 'series/<name>') or another
    exported curve result (_RAW_CURVE_KINDS, 'table': metadata only)."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata'])) if 'metadata' in archive.files else {}
        if metadata.get('kind') == 'curve_plots':
            shared = {key: metadata[key] for key in ('model', 'parameters') if key in metadata}
            plots = {}
            for plot_index, entry in enumerate(metadata['plots']):
                series = {name: np.asarray(archive[f'plot{plot_index}/series{series_index}'], dtype=np.float64)
                          for series_index, name in enumerate(entry['series'])}
                plot_metadata = dict(shared, **{key: value for key, value in entry.items()
                                                if key not in ('name', 'series')})
                plots[str(entry['name'])] = _curve_dict(
                    plot_metadata, 'x', np.asarray(archive[f'plot{plot_index}/x'], dtype=np.float64), series)
            return plots
        if metadata.get('kind') in _RAW_CURVE_KINDS or metadata.get('kind') == 'table':
            return _result_plots(dict(metadata, **{key: np.asarray(archive[key]) for key in archive.files
                                                   if key != 'metadata'}), path)
        if 'x' not in archive.files:
            raise ValueError("no 'x' array: not an exported curve (xy) result")
        x = np.asarray(archive['x'], dtype=np.float64)
        series = {key[len('series/'):]: np.asarray(archive[key], dtype=np.float64)
                  for key in archive.files if key.startswith('series/')}
    return _named_plot(_curve_dict(metadata, 'x', x, series), path)


def _result_plots(result, path):
    """{plot name: xy} of a read result of _RAW_CURVE_KINDS or 'table', drawn as the GUI result views draw it: probe
    vs time (stationary iterations vs the 1-based index, no units recorded), power vs wavelength, far field in dB
    below its peak vs angle in degrees per wavelength, a line profile vs distance along the line, table number columns
    vs its first (with two or more)."""
    kind = result['kind']
    shared = {key: result[key] for key in ('model', 'parameters') if key in result}
    if kind == 'line_profile':
        name = str(result.get('component') or 'values')
        return _named_plot(dict(shared, plot=result.get('plot', path.stem), **_curve_dict(
            {'x_label': 'Distance along the line', 'x_unit': 'm', 'y_label': name,
             'series_units': {name: str(result.get('unit', ''))}},
            'x', np.asarray(result['distance'], dtype=np.float64),
            {name: np.asarray(result['values'], dtype=np.float64)})), path)
    if kind == 'table':
        numeric_columns = list(numeric_table_columns(result))
        xy = table_xy(result, numeric_columns[0] if len(numeric_columns) > 1 else '')
        if not xy['series']:
            raise ValueError('the table has no number columns to plot')
        return _named_plot(dict(shared, plot=result.get('plot', path.stem), **xy), path)
    if kind == 'timeseries':
        times = np.asarray(result['t'], dtype=np.float64)
        name = str(result.get('component') or 'values')
        if is_iteration_series(times):   # the GUI labels this axis with the shared t: no units, drawn on any axis
            return _named_plot(dict(shared, plot=result.get('plot', path.stem), **_curve_dict(
                {'x_label': 'Iteration'}, 'x', np.arange(1.0, times.size + 1.0),
                {name: np.asarray(result['values'], dtype=np.float64)})), path)
        return _named_plot(dict(shared, plot=result.get('plot', path.stem), **_curve_dict(
            {'x_label': 'Time', 'x_unit': 's', 'y_label': 'Value', 'series_units': {name: str(result.get('unit', ''))}},
            'x', times, {name: np.asarray(result['values'], dtype=np.float64)})), path)
    if kind == 'flux_spectrum':
        series = {'power': np.asarray(result['power'], dtype=np.float64)}
        units = {'power': str(result.get('unit', 'a.u.'))}
        if 'power_per_cw' in result:
            series['power_per_cw'] = np.asarray(result['power_per_cw'], dtype=np.float64)
            units['power_per_cw'] = 'W/m'
        return _named_plot(dict(shared, plot=result.get('plot', path.stem), **_curve_dict(
            {'x_label': 'Wavelength', 'x_unit': 'm', 'y_label': 'Power', 'series_units': units}, 'x',
            np.asarray(result['wavelength'], dtype=np.float64), series)), path)
    if 'angles' in result:   # far field .npz: power (n_f, n_angles)
        angles = np.asarray(result['angles'], dtype=np.float64)
        powers = np.atleast_2d(np.asarray(result['power'], dtype=np.float64))
        names = [f'λ = {wavelength * 1e9:.6g} nm' for wavelength in np.atleast_1d(result['wavelength'])]
    else:                    # far field .csv: angle_rad + one power_<nm>nm column per wavelength
        angles = np.asarray(result['angle_rad'], dtype=np.float64)
        columns = [key for key in result if key.startswith('power_')]
        powers = [np.asarray(result[key], dtype=np.float64) for key in columns]
        names = [f'λ = {key[6:-2]} nm' if key.endswith('nm') else key for key in columns]
    plots = {}
    for name, power in zip(names, powers):
        _, db = normalised_db(power, _FAR_FIELD_DB_FLOOR)
        plots[name] = dict(shared, **_curve_dict(
            {'x_label': 'Angle from +x', 'x_unit': 'deg', 'y_label': 'Normalised power',
             'series_units': {'normalised power': 'dB'}}, 'x', np.degrees(angles), {'normalised power': db}))
    if not plots:
        raise ValueError('the far field holds no power columns')
    return plots


def _table_cell_value(cell):
    """Value of a table .csv cell as _csv_cell_text wrote it: '' -> None, true / false, a number, else the text."""
    if not cell.strip():
        return None
    if cell in ('true', 'false'):
        return cell == 'true'
    try:
        return float(cell)
    except ValueError:
        return cell


def _named_plot(xy, path):
    """{plot name: xy} of a one-plot file: its 'plot' metadata (removed from xy), else the file name."""
    return {str(xy.pop('plot', path.stem)): xy}


def _curve_dict(metadata, x_name, x, series):
    """xy result of read curves: the file's metadata plus x / series; series_units only as recorded."""
    if metadata.get('kind', 'xy') != 'xy':
        raise ValueError(f"holds a {metadata['kind']!r} result, not curves (xy, {', '.join(_RAW_CURVE_KINDS)} or "
                         f"table)")
    if not series or any(np.shape(values) != x.shape for values in series.values()) or x.ndim != 1:
        raise ValueError('expected 1D x and curve values of the same length')
    result = dict(metadata, kind='xy', x=x, series=series)
    result.setdefault('x_label', x_name)
    result.setdefault('x_unit', '')
    result.setdefault('y_label', '')
    result.setdefault('series_units', {})
    return result


def _metadata_value(key, text):
    """Value of a '# key: value' line: JSON, else the text; _CSV_TEXT_KEYS: the text, or the JSON string the writer
    quoted (line breaks, edge spaces, a leading quote)."""
    if key in _CSV_TEXT_KEYS:
        if text.startswith('"'):
            try:
                decoded = json.loads(text)
            except ValueError:   # a hand-written quote: plain text
                return text
            return decoded if isinstance(decoded, str) else text
        return text
    try:
        return json.loads(text)
    except ValueError:
        return text


def _is_number(cell):
    """True for a float literal (nan / inf included) or an empty cell."""
    try:
        float(cell.strip() or 'nan')
    except ValueError:
        return False
    return True


def _csv_cell_text(value):
    """Text of one table cell: None -> '', bools 'true' / 'false', floats '%.9g', anything else str()."""
    if value is None:
        return ''
    if isinstance(value, (bool, np.bool_)):
        return 'true' if value else 'false'
    if isinstance(value, (float, np.floating)):
        return f'{float(value):.9g}'
    return str(value)


def _write_npz(path, arrays, metadata):
    """np.savez of the arrays plus 'metadata' (0-d unicode JSON array, loads without pickle) to '<file>.partial',
    then os.replace over exactly `path` (never a truncated archive); up to _NPZ_BUFFER_BYTES zipped in memory first."""
    if 'metadata' in arrays:
        raise ValueError("array key 'metadata' clashes with the .npz metadata entry")
    entries = dict(arrays, metadata=np.array(_to_json(metadata)))
    array_bytes = sum(int(getattr(value, 'nbytes', 0)) for value in entries.values())
    partial_path = path.with_name(path.name + '.partial')
    try:
        with open(partial_path, 'wb') as handle:   # a file object: numpy does not append '.npz' to the name
            if array_bytes <= _NPZ_BUFFER_BYTES:
                buffer = BytesIO()
                np.savez(buffer, **entries)
                handle.write(buffer.getbuffer())
            else:
                np.savez(handle, **entries)
        os.replace(partial_path, path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise


def _to_json(value):
    return json.dumps(value, default=_json_default)


def _json_default(value):
    """numpy scalars / arrays -> builtin values for json.dumps."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'{type(value).__name__} is not JSON serialisable')
