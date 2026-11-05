"""Saved run results: summary.json (digests), results.npz (arrays), log.txt and model.json; CPU only (numpy)."""
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePath

import numpy as np

from ..core.renderer import FieldInfo
from ..model.session import PLANE_AXES, FieldData, LineProfile, Snapshot
from ..model.spec import ModelSpec
from .model_io import _write_npz, _write_text_atomic, load_model, save_model

RESULTS_FORMAT = 'open_sol.results'     # results.npz metadata 'format'
RESULTS_FORMAT_VERSION = 1
NONFINITE_STRINGS = {math.nan: 'nan', math.inf: 'inf', -math.inf: '-inf'}   # JSON text of non-finite floats
_NONFINITE_FLOATS = {text: number for number, text in NONFINITE_STRINGS.items()}
_MAX_JSON_ARRAY_ELEMENTS = 10_000
_MAX_DIGEST_TABLE_ROWS = 50
SUMMARY_FILE = 'summary.json'
LOG_FILE = 'log.txt'
_ARRAYS_FILE = 'results.npz'
_MODEL_FILE = 'model.json'
_CONVERGENCE_FILE = 'convergence.json'
_MESHES_FILE = 'meshes.npz'
MESHES_FORMAT = 'open_sol.meshes'       # meshes.npz metadata 'format' (S7.2: the meshes the fields live on)
MESHES_FORMAT_VERSION = 1
_MESH_ARRAY_NAMES = {'tensor': ('x_edges', 'y_edges', 'z_edges'), 'triangles': ('points', 'triangles', 'pml_faces'),
                     'tetrahedra': ('points', 'tets')}


# ---- JSON ----
def json_safe(value):
    """JSON-able copy: builtin scalars, str keys, lists, NONFINITE_STRINGS for non-finite floats, arrays of at most
    10,000 elements as lists; TypeError for larger arrays (they belong in results.npz) and unsupported types."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):   # also numpy float64
        value = float(value)
        if math.isfinite(value):
            return value
        return NONFINITE_STRINGS[math.nan if math.isnan(value) else value]   # the math.nan key matches by identity
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        if value.size > _MAX_JSON_ARRAY_ELEMENTS:
            raise TypeError(f'an array of shape {value.shape} has more than {_MAX_JSON_ARRAY_ELEMENTS} elements and '
                            f'does not belong in a JSON report: store it in results.npz and report a digest')
        return json_safe(value.tolist())
    if isinstance(value, PurePath):
        return str(value)
    raise TypeError(f'{type(value).__name__} is not JSON serialisable')


def write_json(path, data):
    """Write json_safe(data) as ASCII JSON (indent 2) through '<file>.partial'; creates the parent directory."""
    path = Path(path)
    text = json.dumps(json_safe(data), indent=2, ensure_ascii=True, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(path, text + '\n')


def read_json(path):
    """Parsed JSON file (utf-8-sig); the strings 'nan', 'inf' and '-inf' stay strings."""
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


# ---- records ----
@dataclass(eq=False)
class FieldRecord:
    """One saved output field on its native window cells (polar / spherical: the image pixels over the sector's box):
    values float32 (nx, ny), cell edges [m] float64."""
    physics: str
    component: str
    values: np.ndarray          # float32 (nx, ny)
    x_edges: np.ndarray         # float64 (nx + 1,) [m]
    y_edges: np.ndarray         # float64 (ny + 1,) [m]
    info: FieldInfo
    time: float
    time_label: str | None
    plane: tuple | None = None  # 3D models: ('z', cut height [m]) as FieldData.plane; None in 2D
    mesh: str = ''              # model mesh the field lives on (S7.2)
    triangulation: tuple | None = None   # triangle meshes: FieldData.triangulation (nodes, triangles, node values)

    @property
    def key(self):
        return f'{self.physics}.{self.component}'

    def field_data(self) -> FieldData:
        """session.FieldData (fresh values copy) with extent = outer edges and x / y centres = edge midpoints (a
        triangle-mesh raster: its pixel centres) and the triangulation."""
        x_edges = np.asarray(self.x_edges, dtype=np.float64)
        y_edges = np.asarray(self.y_edges, dtype=np.float64)
        extent = (float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1]))
        return FieldData(self.physics, self.component, np.array(self.values, dtype=np.float32), extent, self.info,
                         self.time, self.time_label, x_centers=0.5 * (x_edges[:-1] + x_edges[1:]),
                         y_centers=0.5 * (y_edges[:-1] + y_edges[1:]), plane=self.plane, axes=_plane_axes(self.plane),
                         mesh=self.mesh, triangulation=self.triangulation)


@dataclass
class ResultsData:
    """Everything a results directory holds (host data only); see save_results / load_results."""
    summary: dict
    fields: dict = field(default_factory=dict)          # 'physics.component' -> FieldRecord
    results: dict = field(default_factory=dict)         # 'physics.feature' -> result dict (dft: + 'intensity_all')
    sweep_results: list = field(default_factory=list)   # per sweep value {'physics.feature': result dict}
    sector_results: list = field(default_factory=list)  # per parity sector (summary parity_sectors labels), the same
    sector_combined_results: dict = field(default_factory=dict)   # the drawn source's (core.mirror.combine_sector_result)
    snapshots: list = field(default_factory=list)       # session.Snapshot
    sweep_snapshots: list = field(default_factory=list) # per sweep value [session.Snapshot] (SweepSpec.keep_fields)
    log_lines: list = field(default_factory=list)
    spec: ModelSpec | None = None
    figure_files: list = field(default_factory=list)    # relative 'figures/<name>.png'
    meshes: dict = field(default_factory=dict)          # mesh name -> {kind, arrays} (Session.mesh_arrays; meshes.npz)


# ---- digests ----
def field_digest(record):
    """Statistics of a FieldRecord over its finite cells ('mean' area-weighted, 'abs_max_at' = cell centre [m]);
    the statistics are None when no cell is finite."""
    values = np.asarray(record.values)
    x_edges = np.asarray(record.x_edges, dtype=np.float64)
    y_edges = np.asarray(record.y_edges, dtype=np.float64)
    if values.ndim != 2 or x_edges.shape != (values.shape[0] + 1,) or y_edges.shape != (values.shape[1] + 1,):
        raise ValueError(f"field '{record.key}': values {values.shape} and edges {x_edges.shape}, {y_edges.shape} "
                         f"do not match (nx, ny), (nx + 1,), (ny + 1,)")
    finite = np.isfinite(values)
    n_finite = int(np.count_nonzero(finite))
    digest = {'physics': record.physics, 'component': record.component, 'label': record.info.label,
              'unit': record.info.unit, 'shape': [int(n) for n in values.shape], 'min': None, 'max': None,
              'mean': None, 'abs_max': None, 'abs_max_at': None, 'n_nonfinite': int(values.size - n_finite),
              'time': float(record.time), 'time_label': record.time_label}
    if n_finite == 0:
        return digest
    finite_values = values[finite].astype(np.float64)
    finite_areas = np.outer(np.diff(x_edges), np.diff(y_edges))[finite]
    abs_values = np.where(finite, np.abs(values), -1.0)
    i, j = np.unravel_index(int(np.argmax(abs_values)), values.shape)
    abs_max_center = [float(0.5 * (x_edges[i] + x_edges[i + 1])), float(0.5 * (y_edges[j] + y_edges[j + 1]))]
    digest.update(min=float(finite_values.min()), max=float(finite_values.max()),
                  mean=float(np.dot(finite_values, finite_areas) / finite_areas.sum()),
                  abs_max=float(abs_values[i, j]), abs_max_at=abs_max_center)
    return digest


def result_digest(result, max_intensity=None):
    """Compact digest of a feature-result dict per kind (builtin numbers, None if not computable; unknown: kind, keys);
    xy, flux and far-field power statistics skip non-finite values; dft_fields max_intensity: the argument, else from
    'intensity_all'."""
    kind = result.get('kind')
    if kind == 'timeseries':
        times = _float_vector(result['t'])
        values = _float_vector(result['values'])
        return {'kind': kind, 'component': result.get('component'), 'unit': result.get('unit'),
                'n_samples': int(values.size), 't_start': _element(times, 0), 't_stop': _element(times, -1),
                'min': _reduce(values, np.min), 'max': _reduce(values, np.max), 'final': _element(values, -1)}
    if kind == 'flux_spectrum':
        wavelength = _float_vector(result['wavelength'])
        power = _float_vector(result['power'])
        peak_index = _finite_argmax(power)
        finite_power = power[np.isfinite(power)]
        return {'kind': kind, 'unit': result.get('unit'), 'n_wavelengths': int(wavelength.size),
                'wavelength_min': _reduce(wavelength, np.min), 'wavelength_max': _reduce(wavelength, np.max),
                'power_min': _reduce(finite_power, np.min), 'power_max': _reduce(finite_power, np.max),
                'wavelength_at_max': None if peak_index is None else float(wavelength[peak_index])}
    if kind == 'dft_fields':
        wavelength = _float_vector(result['wavelength'])
        if max_intensity is None and 'intensity_all' in result:
            max_intensity = [_reduce(np.asarray(frame), np.max) for frame in result['intensity_all']]
        return {'kind': kind, 'quantity': result.get('quantity'), 'shape': [int(n) for n in result['shape']],
                'n_wavelengths': int(wavelength.size), 'wavelengths': wavelength.tolist(),
                'intensity_max': None if max_intensity is None else [None if value is None else float(value)
                                                                     for value in max_intensity]}
    if kind == 'far_field':
        wavelength = _float_vector(result['wavelength'])
        angles = _float_vector(result['angles'])
        power = np.asarray(result['power'], dtype=np.float64).reshape(wavelength.size, angles.size)
        peak_indices = [_finite_argmax(row) for row in power]
        power_max = [None if index is None else float(row[index]) for row, index in zip(power, peak_indices)]
        angle_at_max_deg = [None if index is None else float(np.degrees(angles[index])) for index in peak_indices]
        return {'kind': kind, 'quantity': result.get('quantity'), 'unit': result.get('unit'),
                'n_wavelengths': int(wavelength.size),
                'n_angles': int(angles.size), 'wavelengths': wavelength.tolist(), 'power_max': power_max,
                'angle_at_max_deg': angle_at_max_deg}
    if kind == 'xy':
        x = _float_vector(result['x'])
        series_units = result.get('series_units') or {}
        series_digests = {}
        for name, values in result['series'].items():
            values = _float_vector(values)
            if values.size != x.size:
                raise ValueError(f"xy series '{name}' has {values.size} values for {x.size} x values")
            finite = np.isfinite(values)
            peak_index = _finite_argmax(values)
            series_digests[str(name)] = {'unit': series_units.get(name), 'min': _reduce(values[finite], np.min),
                                         'max': _reduce(values[finite], np.max),
                                         'x_at_max': None if peak_index is None else float(x[peak_index])}
        finite_x = x[np.isfinite(x)]
        return {'kind': kind, 'x_label': result.get('x_label'), 'x_unit': result.get('x_unit'),
                'n_points': int(x.size), 'x_min': _reduce(finite_x, np.min), 'x_max': _reduce(finite_x, np.max),
                'series': series_digests}
    if kind == 'table':
        rows = list(result['rows'])
        return {'kind': kind, 'columns': [str(column) for column in result['columns']],
                'units': [str(unit) for unit in result.get('units') or []], 'n_rows': len(rows),
                'rows': json_safe(rows[:_MAX_DIGEST_TABLE_ROWS])}
    return {'kind': kind, 'keys': sorted(str(key) for key in result)}


# ---- results directory ----
def save_results(directory, data, save_arrays=True, written_files=None):
    """Write results.npz and meshes.npz (when save_arrays; meshes.npz only with data.meshes), log.txt, model.json (when
    data.spec) and, last, summary.json after setting data.summary['files']; creates the directory and returns that
    files dict. written_files (dict) receives 'arrays' / 'meshes' / 'model' -> file name once that file is written (a
    failure or interrupt leaves the rest out)."""
    directory = Path(directory)
    written_files = {} if written_files is None else written_files
    directory.mkdir(parents=True, exist_ok=True)
    files = {'summary': SUMMARY_FILE, 'log': LOG_FILE, 'arrays': _ARRAYS_FILE if save_arrays else None,
             'meshes': _MESHES_FILE if save_arrays and data.meshes else None,
             'model': _MODEL_FILE if data.spec is not None else None, 'figures': list(data.figure_files)}
    if isinstance(data.summary.get('convergence'), dict):   # S7.7 mesh refinement report, also in summary.json
        write_json(directory / _CONVERGENCE_FILE, data.summary['convergence'])
        files['convergence'] = _CONVERGENCE_FILE
    data.summary['files'] = files
    if save_arrays:
        arrays, metadata = _results_arrays(data)
        _write_npz(directory / _ARRAYS_FILE, arrays, json_safe(metadata))
        written_files['arrays'] = _ARRAYS_FILE
    if files['meshes']:
        _write_npz(directory / _MESHES_FILE, *_mesh_arrays(data.meshes))
        written_files['meshes'] = _MESHES_FILE
    (directory / LOG_FILE).write_text(''.join(f'{line}\n' for line in data.log_lines), encoding='utf-8',
                                       newline='\n')
    if data.spec is not None:
        save_model(data.spec, directory / _MODEL_FILE)
        written_files['model'] = _MODEL_FILE
    write_json(directory / SUMMARY_FILE, data.summary)
    return files


def load_summary(directory):
    """summary.json of a results directory alone, derived (+ reference / error) / sweep derived (+ reference / error) /
    curve_error (+ sweep) max, rms / fixed_point (+ wavelength_fixed_points) relative_change / derived_series values
    (+ sweep) / convergence level value, relative_change, derived non-finite strings back as floats (as live);
    FileNotFoundError when absent."""
    summary_path = Path(directory) / SUMMARY_FILE
    if not summary_path.is_file():
        raise FileNotFoundError(f'{summary_path} not found: {directory} is not an open_sol results directory')
    summary = read_json(summary_path)
    for key in ('derived', 'derived_reference', 'derived_error'):
        if isinstance(summary.get(key), dict):
            summary[key] = {name: _nonfinite_float(value) for name, value in summary[key].items()}
    sweep = summary.get('sweep')
    for key in ('derived', 'reference', 'error'):
        if isinstance(sweep, dict) and isinstance(sweep.get(key), dict):
            sweep[key] = {name: [_nonfinite_float(value) for value in values] for name, values in sweep[key].items()}
    records = list((summary.get('curve_error') or {}).values())
    if isinstance(sweep, dict):
        records += [record for values in (sweep.get('curve_error') or {}).values() for record in values]
    for record in records:
        if isinstance(record, dict):
            record.update({key: _nonfinite_float(record[key]) for key in ('max', 'rms') if key in record})
    for fixed_point in [summary.get('fixed_point'), *(summary.get('wavelength_fixed_points') or [])]:
        if isinstance(fixed_point, dict) and 'relative_change' in fixed_point:
            fixed_point['relative_change'] = _nonfinite_float(fixed_point['relative_change'])
    series_records = [summary.get('derived_series')]
    if isinstance(sweep, dict):
        series_records += list(sweep.get('derived_series') or [])
    for series_record in series_records:
        if isinstance(series_record, dict) and isinstance(series_record.get('series'), dict):
            series_record['series'] = {name: [_nonfinite_float(value) for value in values]
                                       for name, values in series_record['series'].items()}
    convergence = summary.get('convergence')
    for level in (convergence.get('levels') or []) if isinstance(convergence, dict) else []:
        if isinstance(level, dict):
            level.update({key: _nonfinite_float(level[key]) for key in ('value', 'relative_change') if key in level})
            if isinstance(level.get('derived'), dict):
                level['derived'] = {name: _nonfinite_float(value) for name, value in level['derived'].items()}
    return summary


def load_results(directory):
    """ResultsData of a results directory: summary.json (required), log.txt, and the arrays and model that
    summary['files'] lists (stale files of an earlier run are ignored); missing optional files stay empty / None."""
    directory = Path(directory)
    summary = load_summary(directory)
    files = summary.get('files') if isinstance(summary.get('files'), dict) else {}
    data = ResultsData(summary=summary, figure_files=list(files.get('figures') or []))
    if files.get('arrays') and (directory / files['arrays']).is_file():
        _read_arrays(directory / files['arrays'], data)
    if files.get('meshes') and (directory / files['meshes']).is_file():
        data.meshes = read_meshes(directory / files['meshes'])
    log_path = directory / LOG_FILE
    if log_path.is_file():
        data.log_lines = log_path.read_text(encoding='utf-8').splitlines()
    if files.get('model') and (directory / files['model']).is_file():
        data.spec = load_model(directory / files['model'])
    return data


def read_meshes(path):
    """{mesh name: {'kind': 'tensor' | 'triangles' | 'tetrahedra', array name: array}} of a meshes.npz (tensor:
    x_edges, y_edges(, z_edges) window cell edges in Domain coordinates; triangles: points, triangles, pml_faces;
    tetrahedra: points, tets)."""
    with np.load(path, allow_pickle=False) as npz:
        metadata = json.loads(str(npz['metadata']))
        if metadata.get('format') != MESHES_FORMAT or metadata.get('version') != MESHES_FORMAT_VERSION:
            raise ValueError(f"{path}: format {metadata.get('format')!r} version {metadata.get('version')!r} is not "
                             f"{MESHES_FORMAT!r} version {MESHES_FORMAT_VERSION}")
        meshes = {}
        for name, entry in metadata['meshes'].items():
            kind = entry['kind']
            meshes[name] = {'kind': kind, **{array_name: np.array(npz[f'{name}/{array_name}'])
                                             for array_name in _MESH_ARRAY_NAMES[kind]
                                             if f'{name}/{array_name}' in npz.files}}
    return meshes


def _mesh_arrays(meshes):
    """(npz arrays '<mesh>/<array name>', metadata) of ResultsData.meshes."""
    arrays, entries = {}, {}
    for name, mesh in meshes.items():
        kind = mesh['kind']
        if kind not in _MESH_ARRAY_NAMES:
            raise ValueError(f"mesh '{name}': kind must be one of {', '.join(_MESH_ARRAY_NAMES)}, got {kind!r}")
        for array_name in _MESH_ARRAY_NAMES[kind]:
            if mesh.get(array_name) is not None:
                arrays[f'{_npz_name_part(name)}/{array_name}'] = np.asarray(mesh[array_name])
        entries[name] = {'kind': kind}
    return arrays, {'format': MESHES_FORMAT, 'version': MESHES_FORMAT_VERSION, 'meshes': entries}


# ---- helpers ----
def _float_vector(values):
    return np.asarray(values, dtype=np.float64).ravel()


def _element(array, index):
    """Builtin float of array[index]; None for an empty array."""
    return float(array[index]) if array.size else None


def _reduce(array, reduction):
    """Builtin float of reduction(array); None for an empty array."""
    return float(reduction(array)) if array.size else None


def _finite_argmax(values):
    """Index of the largest finite value of a 1D float array; None when none is finite."""
    finite = np.isfinite(values)
    return int(np.argmax(np.where(finite, values, -np.inf))) if finite.any() else None


def _nonfinite_float(value):
    """Float of exactly 'nan' / 'inf' / '-inf' (the json_safe text of a non-finite float); any other value unchanged."""
    return _NONFINITE_FLOATS.get(value, value) if isinstance(value, str) else value


def _plane(entries):
    """FieldData.plane (axis, position [m]) of a saved field / snapshot entry; None in 2D."""
    plane = entries['plane']
    return None if plane is None else (str(plane[0]), float(plane[1]))


def _plane_axes(plane):
    """FieldData.axes of a cut plane (session.PLANE_AXES; ('x', 'y') in 2D)."""
    return PLANE_AXES['z' if plane is None else plane[0]]


def _restore_nonfinite(result):
    """Result with the non-finite strings of its top-level scalar extras (e.g. resonance_monitor start_time) and
    table cells back as floats (as live)."""
    for key, value in result.items():
        if isinstance(value, str):
            result[key] = _nonfinite_float(value)
    if result.get('kind') == 'table' and 'rows' in result:
        result['rows'] = [[_nonfinite_float(cell) for cell in row] for row in result['rows']]
    return result


def _results_arrays(data):
    """(npz arrays, metadata dict) of the fields, feature results, sweep / sector (+ combined sector) results,
    snapshots and kept sweep snapshots of a ResultsData."""
    arrays, digests = {}, {}   # digests: (id(nodes), id(triangles)) -> table digest (shared display tables)
    metadata = {'format': RESULTS_FORMAT, 'version': RESULTS_FORMAT_VERSION, 'fields': {}, 'results': {},
                'sweep_results': [], 'sector_results': [], 'sector_combined_results': {}, 'snapshots': [],
                'sweep_snapshots': []}
    for key, record in data.fields.items():
        arrays[f'field/{key}/values'] = np.asarray(record.values)
        arrays[f'field/{key}/x_edges'] = np.asarray(record.x_edges, dtype=np.float64)
        arrays[f'field/{key}/y_edges'] = np.asarray(record.y_edges, dtype=np.float64)
        metadata['fields'][key] = {'physics': record.physics, 'component': record.component,
                                   'info': asdict(record.info), 'time': record.time, 'time_label': record.time_label,
                                   'plane': record.plane, 'mesh': record.mesh,
                                   'triangulation': _triangulation_entry(record.triangulation, f'field/{key}/',
                                                                         arrays, digests)}
    for key, result in data.results.items():
        metadata['results'][key] = _split_result(result, f'result/{_npz_name_part(key)}/', arrays)
    for index, value_results in enumerate(data.sweep_results):
        metadata['sweep_results'].append({key: _split_result(result, f'sweep/{index}/{_npz_name_part(key)}/',
                                                             arrays)
                                          for key, result in value_results.items()})
    for index, sector_results in enumerate(data.sector_results):
        metadata['sector_results'].append({key: _split_result(result, f'sector/{index}/{_npz_name_part(key)}/',
                                                              arrays)
                                           for key, result in sector_results.items()})
    for key, result in data.sector_combined_results.items():
        metadata['sector_combined_results'][key] = _split_result(result, f'sector_combined/{_npz_name_part(key)}/',
                                                                 arrays)
    metadata['snapshots'] = [_snapshot_entries(snapshot, f'snapshot/{index}/', arrays, digests)
                             for index, snapshot in enumerate(data.snapshots)]
    metadata['sweep_snapshots'] = [[_snapshot_entries(snapshot, f'sweep_snapshot/{value}/{index}/', arrays, digests)
                                    for index, snapshot in enumerate(snapshots)]
                                   for value, snapshots in enumerate(data.sweep_snapshots)]
    return arrays, metadata


def _snapshot_entries(snapshot, prefix, arrays, digests):
    """Metadata entry {time, label, fields, cuts, segments} of one Snapshot; its arrays go into arrays under prefix."""
    segments = {}
    for name, profile in snapshot.segments.items():   # 3D-segment line profiles: arrays under prefix segment/<name>/
        segment_prefix = f'{prefix}segment/{_npz_name_part(name)}/'
        arrays[segment_prefix + 'distance'] = np.asarray(profile.distance, dtype=np.float64)
        arrays[segment_prefix + 'values'] = np.asarray(profile.values, dtype=np.float64)
        for axis, centres in zip('xyz', profile.centres):
            arrays[f'{segment_prefix}centre_{axis}'] = np.asarray(centres, dtype=np.float64)
        segments[name] = {'physics': profile.physics, 'component': profile.component, 'start': list(profile.start),
                          'end': list(profile.end), 'info': asdict(profile.info), 'time_label': profile.time_label}
    return {'time': snapshot.time, 'label': snapshot.label,
            'fields': _snapshot_field_entries(snapshot.fields, prefix, arrays, digests),
            'cuts': [_snapshot_field_entries(fields, f'{prefix}cut{cut}/', arrays, digests)
                     for cut, fields in enumerate(snapshot.cuts)],
            'segments': segments}


def _snapshot_field_entries(fields, prefix, arrays, digests):
    """Metadata entries of one snapshot cut's {key: FieldData}; values (and native cell centres) go into arrays under
    prefix + key."""
    entries = {}
    for key, field_data in fields.items():
        arrays[f'{prefix}{key}/values'] = np.asarray(field_data.values)
        if field_data.x_centers is not None and field_data.y_centers is not None:   # native (graded) cells
            arrays[f'{prefix}{key}/x_centers'] = np.asarray(field_data.x_centers, dtype=np.float64)
            arrays[f'{prefix}{key}/y_centers'] = np.asarray(field_data.y_centers, dtype=np.float64)
        if field_data.mapped_centres is not None:   # native polar / spherical cells: drawing-plane centres
            arrays[f'{prefix}{key}/mapped_x'] = np.asarray(field_data.mapped_centres[0], dtype=np.float64)
            arrays[f'{prefix}{key}/mapped_y'] = np.asarray(field_data.mapped_centres[1], dtype=np.float64)
        entries[key] = {'extent': field_data.extent, 'info': asdict(field_data.info),
                        'time_label': field_data.time_label, 'native_shape': field_data.native_shape,
                        'plane': field_data.plane, 'axes': list(field_data.axes), 'unfolded': field_data.unfolded,
                        'mesh': field_data.mesh,
                        'triangulation': _triangulation_entry(field_data.triangulation, f'{prefix}{key}/', arrays,
                                                              digests)}
    return entries


def _triangulation_entry(triangulation, field_prefix, arrays, digests):
    """Metadata of a FieldData triangulation: its node values go to field_prefix + 'node_values', the node table to
    'triangulation/<digest>/nodes' / 'triangles', written once per distinct table (snapshots share it; hashed once
    per table object through `digests`); None without a triangulation."""
    if triangulation is None:
        return None
    identity = (id(triangulation[0]), id(triangulation[1]))   # the objects stay alive during the save
    table = digests.get(identity)
    if table is None:
        nodes = np.ascontiguousarray(triangulation[0], dtype=np.float64)
        triangles = np.ascontiguousarray(triangulation[1], dtype=np.int32)
        table = digests[identity] = hashlib.sha256(nodes.tobytes() + triangles.tobytes()).hexdigest()[:16]
    if f'triangulation/{table}/nodes' not in arrays:
        nodes = np.ascontiguousarray(triangulation[0], dtype=np.float64)
        triangles = np.ascontiguousarray(triangulation[1], dtype=np.int32)
        arrays[f'triangulation/{table}/nodes'] = nodes
        arrays[f'triangulation/{table}/triangles'] = triangles
    arrays[field_prefix + 'node_values'] = np.asarray(triangulation[2], dtype=np.float32)
    return table


def _read_triangulation(table, field_prefix, npz):
    """FieldData.triangulation of a _triangulation_entry (None: none saved)."""
    if not table:
        return None
    return (npz[f'triangulation/{table}/nodes'], npz[f'triangulation/{table}/triangles'],
            npz[field_prefix + 'node_values'])


def _is_npz_array(value):
    """True for a >= 1-D numeric ndarray (stored in results.npz, not in the metadata JSON)."""
    return isinstance(value, np.ndarray) and value.ndim >= 1 and value.dtype.kind in 'biufc'


def _npz_name_part(key):
    """str(key) as one level of a results.npz array name; ValueError for a '/' (the level separator)."""
    text = str(key)
    if '/' in text:
        raise ValueError(f"result key {text!r} contains '/', which separates the levels of results.npz names")
    return text


def _split_result(result, array_prefix, arrays):
    """Non-array entries of a result dict; its >= 1-D numeric ndarrays go into arrays under array_prefix + key, those
    inside a dict value (one level, e.g. xy 'series') under array_prefix + key + '/' + name."""
    entries = {}
    for key, value in result.items():
        if _is_npz_array(value):
            arrays[array_prefix + _npz_name_part(key)] = value
        elif isinstance(value, dict) and any(_is_npz_array(item) for item in value.values()):
            nested_prefix = f'{array_prefix}{_npz_name_part(key)}/'
            entries[key] = {name: item for name, item in value.items() if not _is_npz_array(item)}
            for name, item in value.items():
                if _is_npz_array(item):
                    arrays[nested_prefix + _npz_name_part(name)] = item
        else:
            entries[key] = value
    return entries


def _result_array_buckets(array_names):
    """{'result/<key>/', 'sweep/<i>/<key>/', 'sector/<i>/<key>/' or 'sector_combined/<key>/': its npz array names in
    archive order}, one pass over the archive."""
    prefix_levels = {'result': 2, 'sweep': 3, 'sector': 3, 'sector_combined': 2}
    buckets = {}
    for array_name in array_names:
        levels = array_name.split('/')
        level_count = prefix_levels.get(levels[0])
        if level_count is not None and len(levels) > level_count:
            buckets.setdefault('/'.join(levels[:level_count]) + '/', []).append(array_name)
    return buckets


def _join_result(entries, array_prefix, buckets, npz):
    """Result dict from its metadata entries plus the npz arrays of buckets[array_prefix] ('<key>/<name>' goes back
    into the dict entry key, in archive order)."""
    result = dict(entries)
    for array_name in buckets.get(array_prefix, ()):
        key, _, name = array_name[len(array_prefix):].partition('/')
        if name:
            result.setdefault(key, {})[name] = npz[array_name]
        else:
            result[key] = npz[array_name]
    return result


def _read_arrays(path, data):
    """Fill data.fields, results, sweep_results, sector_results, sector_combined_results, snapshots and
    sweep_snapshots from a results.npz."""
    with np.load(path, allow_pickle=False) as npz:
        metadata = json.loads(str(npz['metadata']))
        if metadata.get('format') != RESULTS_FORMAT or metadata.get('version') != RESULTS_FORMAT_VERSION:
            raise ValueError(f"{path}: format {metadata.get('format')!r} version {metadata.get('version')!r} is not "
                             f"{RESULTS_FORMAT!r} version {RESULTS_FORMAT_VERSION}")
        for key, field_entries in metadata['fields'].items():
            prefix = f'field/{key}/'
            data.fields[key] = FieldRecord(
                field_entries['physics'], field_entries['component'], npz[prefix + 'values'],
                npz[prefix + 'x_edges'], npz[prefix + 'y_edges'], FieldInfo(**field_entries['info']),
                float(field_entries['time']), field_entries['time_label'], _plane(field_entries),
                mesh=str(field_entries['mesh']),
                triangulation=_read_triangulation(field_entries['triangulation'], prefix, npz))
        buckets = _result_array_buckets(npz.files)
        data.results = {key: _restore_nonfinite(_join_result(entries, f'result/{key}/', buckets, npz))
                        for key, entries in metadata['results'].items()}
        data.sweep_results = [{key: _restore_nonfinite(_join_result(entries, f'sweep/{index}/{key}/', buckets, npz))
                               for key, entries in value_results.items()}
                              for index, value_results in enumerate(metadata['sweep_results'])]
        data.sector_results = [{key: _restore_nonfinite(_join_result(entries, f'sector/{index}/{key}/', buckets,
                                                                     npz))
                                for key, entries in sector_results.items()}
                               for index, sector_results in enumerate(metadata['sector_results'])]
        data.sector_combined_results = {
            key: _restore_nonfinite(_join_result(entries, f'sector_combined/{key}/', buckets, npz))
            for key, entries in metadata['sector_combined_results'].items()}
        array_names = set(npz.files)
        data.snapshots = [_read_snapshot(entries, f'snapshot/{index}/', npz, array_names)
                          for index, entries in enumerate(metadata['snapshots'])]
        data.sweep_snapshots = [[_read_snapshot(entries, f'sweep_snapshot/{value}/{index}/', npz, array_names)
                                 for index, entries in enumerate(value_entries)]
                                for value, value_entries in enumerate(metadata['sweep_snapshots'])]


def _read_snapshot(snapshot_entries, prefix, npz, array_names):
    """Snapshot of one _snapshot_entries record and its npz arrays under prefix."""
    snapshot_time = float(snapshot_entries['time'])
    fields = _snapshot_fields(snapshot_entries['fields'], prefix, snapshot_time, npz, array_names)
    cuts = [_snapshot_fields(entries, f'{prefix}cut{cut}/', snapshot_time, npz, array_names)
            for cut, entries in enumerate(snapshot_entries['cuts'])]
    segments = {}
    for name, entries in snapshot_entries['segments'].items():
        segment_prefix = f'{prefix}segment/{name}/'
        segments[name] = LineProfile(
            entries['physics'], entries['component'], npz[segment_prefix + 'distance'],
            npz[segment_prefix + 'values'], tuple(npz[f'{segment_prefix}centre_{axis}'] for axis in 'xyz'),
            tuple(float(value) for value in entries['start']), tuple(float(value) for value in entries['end']),
            FieldInfo(**entries['info']), snapshot_time, entries['time_label'])
    return Snapshot(time=snapshot_time, fields=fields, cuts=cuts, segments=segments,
                    label=snapshot_entries['label'])


def _snapshot_fields(entries, prefix, snapshot_time, npz, array_names):
    """{key: FieldData} of one snapshot cut from its metadata entries and the npz arrays under prefix + key."""
    fields = {}
    for key, field_entries in entries.items():
        physics, _, component = key.partition('.')
        field_prefix = f'{prefix}{key}/'
        native_shape = field_entries['native_shape']
        has_centers = field_prefix + 'x_centers' in array_names   # native-cell snapshot
        has_mapped = field_prefix + 'mapped_x' in array_names     # native polar / spherical cells
        plane = _plane(field_entries)
        axes = field_entries['axes']
        fields[key] = FieldData(
            physics, component, npz[field_prefix + 'values'],
            tuple(float(edge) for edge in field_entries['extent']), FieldInfo(**field_entries['info']),
            snapshot_time, field_entries['time_label'],
            x_centers=npz[field_prefix + 'x_centers'] if has_centers else None,
            y_centers=npz[field_prefix + 'y_centers'] if has_centers else None,
            native_shape=None if native_shape is None else tuple(native_shape), plane=plane,
            axes=tuple(axes), unfolded=bool(field_entries['unfolded']),
            mapped_centres=((npz[field_prefix + 'mapped_x'], npz[field_prefix + 'mapped_y']) if has_mapped
                            else None),
            mesh=str(field_entries['mesh']),
            triangulation=_read_triangulation(field_entries['triangulation'], field_prefix, npz))
    return fields
