"""Self-contained HTML mode player: animated complex mode shapes on a surface mesh beside their band diagram.

`write_mode_player` packs plain numpy data (surface nodes and triangles, complex nodal displacements per mode,
optionally the complex stress / strain / potential per mode and the nodal density, the optional band table and lattice
period) into `mode_player.html`, a single page with no network use (hand-written WebGL2, data embedded as base64); the
page derives what it colours by from them (components, von Mises, principal stresses, energy densities).
`mode_player_from_results` reads it from an api.Results (duck-typed: viz never imports api or model), from a saved run
or a live session.
"""

import base64
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TEMPLATE_FILE = 'mode_player.html'
DATA_MARKER = '/*__MODE_DATA__*/null'
FORMAT = 'open_sol.mode_player'
FORMAT_VERSION = 1
_INT16_MAX = 32767
# optional per-mode groups beside the displacement: page group -> (PlayerMode attribute, components per node); the
# piezoelectric potential travels with the stress (the mode_shapes feature keeps both or neither)
FIELD_GROUPS = {'stress': ('stress', 6), 'strain': ('strain', 6), 'potential': ('potential', 1)}
PAGE_FIELDS = ('stress', 'strain')        # what a caller chooses (fields=); 'potential' follows 'stress'
_BASE64 = 4.0 / 3.0
_LENGTH_UNITS = ((1e-9, 'nm'), (1e-6, 'um'), (1e-3, 'mm'), (1.0, 'm'))


@dataclass(eq=False)
class PlayerMode:
    """One mode: complex nodal displacement (n_nodes, 3) in any consistent unit (only its shape is drawn)."""
    frequency: float                       # Hz
    displacement: np.ndarray               # complex (n_nodes, 3)
    k: float | None = None                 # abscissa in the band table's k unit (None: no band diagram position)
    k_index: int | None = None             # row of the band table this mode belongs to
    band: int | None = None                # 1-based rank of its frequency among the modes solved at its k
    cell_phase: float = 0.0                # phase step per lattice period [rad]: open_sol Bloch modes satisfy
    #                                        u(x + L) = u(x) exp(-j k L) (time exp(+j omega t)), so -k L
    q_factor: float | None = None
    label: str = ''
    stress: np.ndarray | None = None       # complex (n_nodes, 6): xx, yy, zz, xy, xz, yz (only its shape is drawn)
    strain: np.ndarray | None = None       # complex (n_nodes, 6), same order, tensor shear strain
    potential: np.ndarray | None = None    # complex (n_nodes,)


@dataclass(eq=False)
class BandTable:
    """Frequencies per k (rows may differ in length; NaN allowed) for the band diagram."""
    k: list
    frequencies: list                      # [[Hz, ...] per k]
    k_label: str = 'Wavenumber'
    k_symbol: str = 'k'
    k_unit: str = ''
    k_ticks: list = field(default_factory=list)   # [(k, label)], e.g. [(0, 'Gamma'), (0.5, 'X')]
    k_range: tuple | None = None


def length_scale(extent):
    """(factor, unit) so extent (m) reads between 1 and 1000 units."""
    for scale, unit in _LENGTH_UNITS:
        if extent < 1000 * scale:
            return scale, unit
    return 1.0, 'm'


def _b64(array):
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode('ascii')


def _aligned(displacement):
    """(re, im, rotation): the mode rotated in complex phase so its largest component is real (rotation = that unit
    factor, 1 for a zero mode), scaled to max |u| = 1 (|u| over x, y, z)."""
    u = np.asarray(displacement, dtype=np.complex128).reshape(-1, 3)
    u = np.where(np.isfinite(u), u, 0.0)
    magnitude = np.sqrt(np.sum(np.abs(u) ** 2, axis=1))
    peak = float(magnitude.max()) if magnitude.size else 0.0
    if peak <= 0.0:
        return u.real * 0.0, u.real * 0.0, 1.0
    flat = u.ravel()
    largest = flat[int(np.argmax(np.abs(flat)))]
    rotation = abs(largest) / largest
    u = u * rotation / peak
    return u.real, u.imag, rotation


def _rotated_group(values, rotation, n_nodes, width):
    """{'re', 'im'} base64 int16 of a complex per-node group turned by the displacement's rotation and scaled to max
    |component| = 1 (each derived quantity is drawn normalised, so the group's own scale is not needed)."""
    values = np.asarray(values, dtype=np.complex128).reshape(n_nodes, width)
    values = np.where(np.isfinite(values), values, 0.0) * rotation
    peak = float(np.abs(values).max()) if values.size else 0.0
    if peak > 0.0:
        values = values / peak
    return {'re': _b64(_quantised(values.real)), 'im': _b64(_quantised(values.imag))}


def _quantised(values):
    return np.clip(np.round(values * _INT16_MAX), -_INT16_MAX, _INT16_MAX).astype('<i2')


def _finite(value):
    return None if value is None or not math.isfinite(float(value)) else float(value)


def page_bytes(n_nodes, n_modes, groups=()) -> dict:
    """{'displacement' | group: approximate page bytes} of the mode data (int16 re + im per component, base64)."""
    per_component = 2 * 2 * n_nodes * n_modes * _BASE64
    sizes = {'displacement': int(3 * per_component)}
    sizes.update({group: int(FIELD_GROUPS[group][1] * per_component) for group in groups})
    return sizes


class ModePageTooLarge(ValueError):
    """The page with every stored group would pass max_bytes: `sizes` {'displacement' | group: bytes} lets the caller
    choose (mode_player_from_results(fields=))."""

    def __init__(self, sizes, max_bytes):
        self.sizes = dict(sizes)
        self.max_bytes = int(max_bytes)
        super().__init__(f'the mode page would hold about {sum(sizes.values()) / 1e6:.0f} MB of mode data (limit '
                         f'{max_bytes / 1e6:.0f} MB): choose fewer fields (fields=)')


def player_data(positions, triangles, modes, bands=None, lattice_period=None, lattice_axis=(1.0, 0.0, 0.0),
                title='', subtitle='', dimensions=3, up=(0.0, 0.0, 1.0), view=None, default_mode=0, density=None):
    """The JSON-able dict the page reads. positions (n, 3) in m (2D: z = 0), triangles (m, 3) node indices,
    modes [PlayerMode] (a stress / strain / potential group goes on the page when every mode has it, else none may),
    bands BandTable or None, lattice_period (m) with lattice_axis tiles the cell (None: one cell). view = (azimuth,
    elevation) degrees about `up`, None = the page default. density (n,) kg/m^3 or None: the kinetic energy density."""
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if not len(modes):
        raise ValueError('no modes to export')
    if positions.shape[0] == 0 or triangles.shape[0] == 0:
        raise ValueError('the mode mesh has no surface triangles')
    if triangles.min() < 0 or triangles.max() >= positions.shape[0]:
        raise ValueError('triangle node index outside the node list')
    extent = float(np.max(np.ptp(positions, axis=0))) or 1.0
    scale, unit = length_scale(extent)
    n_nodes = positions.shape[0]
    groups = []
    for group, (attribute, width) in FIELD_GROUPS.items():
        present = [getattr(mode, attribute) is not None for mode in modes]
        if any(present) and not all(present):
            raise ValueError(f'{group} is given for {sum(present)} of {len(modes)} modes: every mode or none')
        if all(present):
            groups.append(group)
    out_modes = []
    for mode in modes:
        u = np.asarray(mode.displacement)
        if u.shape != positions.shape:
            raise ValueError(f'mode displacement shape {u.shape} does not match the nodes {positions.shape}')
        re, im, rotation = _aligned(u)
        entry = {'frequency': float(mode.frequency), 'k': _finite(mode.k), 'k_index': mode.k_index,
                 'band': mode.band, 'cell_phase': float(mode.cell_phase), 'q_factor': _finite(mode.q_factor),
                 'label': mode.label, 're': _b64(_quantised(re)), 'im': _b64(_quantised(im))}
        for group in groups:
            attribute, width = FIELD_GROUPS[group]
            values = np.asarray(getattr(mode, attribute))
            if values.size != n_nodes * width:
                raise ValueError(f'mode {group} has {values.size} values, expected {n_nodes} nodes x {width}')
            entry[group] = _rotated_group(values, rotation, n_nodes, width)
        out_modes.append(entry)
    data = {
        'format': FORMAT, 'version': FORMAT_VERSION, 'title': title, 'subtitle': subtitle, 'length_unit': unit,
        'geometry': {'dimensions': int(dimensions), 'positions': _b64((positions / scale).astype('<f4')),
                     'triangles': _b64(triangles.astype('<u4')), 'up': [float(v) for v in up]},
        'lattice': None, 'bands': None, 'modes': out_modes, 'fields': groups, 'density': None,
        'default_mode': int(min(max(default_mode, 0), len(out_modes) - 1))}
    if view is not None:
        data['geometry']['view'] = [float(view[0]), float(view[1])]
    if density is not None:
        density = np.asarray(density, dtype=np.float64).reshape(-1)
        if density.size != n_nodes:
            raise ValueError(f'density has {density.size} values for {n_nodes} nodes')
        density = np.where(np.isfinite(density) & (density > 0.0), density, 0.0)
        if density.max() > 0.0:   # relative: every energy density is drawn normalised
            data['density'] = _b64((density / density.max()).astype('<f4'))
    if lattice_period:
        axis = np.asarray(lattice_axis, dtype=np.float64)
        data['lattice'] = {'period': float(lattice_period) / scale, 'axis': (axis / np.linalg.norm(axis)).tolist()}
    if bands is not None:
        data['bands'] = {
            'k': [float(k) for k in bands.k],
            'frequencies': [[float(f) if math.isfinite(f) else None for f in row] for row in bands.frequencies],
            'k_label': bands.k_label, 'k_symbol': bands.k_symbol, 'k_unit': bands.k_unit,
            'k_ticks': [[float(k), str(name)] for k, name in bands.k_ticks],
            'k_range': None if bands.k_range is None else [float(v) for v in bands.k_range]}
    return data


def render_html(data):
    """The page text with data embedded ('</' escaped so the JSON cannot close its script element)."""
    template = (Path(__file__).resolve().parent / TEMPLATE_FILE).read_text(encoding='utf-8')
    if DATA_MARKER not in template:
        raise RuntimeError(f'{TEMPLATE_FILE} has no data marker')
    text = json.dumps(data, separators=(',', ':'), allow_nan=False).replace('</', '<\\/')
    return template.replace(DATA_MARKER, text, 1)


def write_mode_player(path, positions, triangles, modes, **options) -> str:
    """Write the player page for plain arrays (arguments of `player_data`); returns the path."""
    path = Path(path)
    if path.suffix.lower() not in ('.html', '.htm'):
        path = path.with_suffix('.html')
    text = render_html(player_data(positions, triangles, modes, **options))
    partial = path.with_name(path.name + '.partial')
    partial.write_text(text, encoding='utf-8')
    partial.replace(path)
    return str(path)


# ---- from open_sol results (duck-typed api.Results) ----
MODE_SHAPES_KIND = 'mode_shapes'


def mode_shape_keys(results) -> list:
    """'physics.feature' keys of the mode_shapes results of a Results (its last / only value)."""
    keys = []
    for key in results.result_names():
        try:
            if results.result(key).get('kind') == MODE_SHAPES_KIND:
                keys.append(key)
        except (KeyError, ValueError, TypeError):
            continue
    return keys


def _entries(results, key):
    """[(sweep index or None, result dict)] of key: every successful sweep value, else the run's own result."""
    try:
        per_value = results.sweep_result(key)
    except KeyError:
        per_value = None
    if per_value and any(entry is not None for entry in per_value):
        return [(index, entry) for index, entry in enumerate(per_value) if entry is not None]
    return [(None, results.result(key))]


def _same_mesh(first, other):
    return (other['nodes'].shape == first['nodes'].shape and other['triangles'].shape == first['triangles'].shape
            and np.array_equal(other['triangles'], first['triangles'])
            and np.allclose(other['nodes'], first['nodes'], rtol=0.0, atol=1e-9 * max(np.ptp(first['nodes']), 1e-30)))


def _k_ticks(k_values):
    lo, hi = min(k_values), max(k_values)
    ticks = []
    if lo <= 1e-9 <= hi + 1e-9:
        ticks.append((0.0, 'Γ'))
    if lo - 1e-9 <= 0.5 <= hi + 1e-9:
        ticks.append((0.5, 'X'))
    return ticks


def stored_fields(entries) -> list:
    """PAGE_FIELDS every mode_shapes result dict of entries keeps (mode_shapes feature setting 'fields')."""
    return [group for group in PAGE_FIELDS if all(f'{group}_re' in entry for entry in entries)]


def _page_groups(entries, fields):
    """FIELD_GROUPS of the page: the chosen PAGE_FIELDS (None = every stored one; ValueError for one not stored or
    not a field) plus the potential with the stress when stored."""
    stored = stored_fields(entries)
    if fields is None:
        fields = stored
    fields = list(fields)
    unknown = [group for group in fields if group not in PAGE_FIELDS]
    if unknown:
        raise ValueError(f"fields= takes {', '.join(PAGE_FIELDS)}, got {', '.join(map(repr, unknown))}")
    missing = [group for group in fields if group not in stored]
    if missing:
        raise ValueError(f"the mode_shapes result keeps no {' or '.join(missing)}: set the mode_shapes feature's "
                         "fields to 'stress' or 'stress_strain' and compute again")
    groups = [group for group in PAGE_FIELDS if group in fields]
    if 'stress' in groups and all('potential_re' in entry for entry in entries):
        groups.append('potential')
    return groups


def mode_page_sizes(results, key=None) -> dict:
    """{'displacement' | group: approximate bytes} of the player page of a mode_shapes result with every stored group
    (the GUI asks which to keep when the sum is large)."""
    key = _mode_key(results, key)
    entries = _mesh_entries(results, key)
    first = entries[-1][1]
    return page_bytes(first['nodes'].shape[0], _mode_count(entries),
                      _page_groups([entry for _, entry in entries], None))


def _mode_count(entries):
    """Modes the page shows: the finite frequencies over every entry."""
    return sum(int(np.isfinite(np.asarray(entry['frequency'], dtype=np.float64)).sum()) for _, entry in entries)


def _mode_key(results, key):
    if key is None:
        keys = mode_shape_keys(results)
        if not keys:
            raise ValueError('the results hold no mode shapes: add a mode_shapes output feature to the solid '
                             'physics (eigenfrequency study on a tetrahedron mesh) and compute again')
        if len(keys) > 1:
            raise ValueError(f"several mode_shapes results ({', '.join(keys)}): choose one")
        return keys[0]
    if key not in results.result_names() or results.result(key).get('kind') != MODE_SHAPES_KIND:
        raise KeyError(f'{key!r} is not a mode_shapes result')   # only this key is fetched (live: one GPU read)
    return key


def _mesh_entries(results, key):
    """_entries of key on the last value's surface mesh."""
    entries = _entries(results, key)
    first = entries[-1][1]
    return [(index, entry) for index, entry in entries if _same_mesh(first, entry)]


def _group_values(entry, group, j):
    """Complex per-node values of `group` of mode j of a mode_shapes result dict."""
    real, imaginary = entry[f'{group}_re'][j], entry[f'{group}_im'][j]
    return np.asarray(real, dtype=np.float64) + 1j * np.asarray(imaginary, dtype=np.float64)


def mode_player_from_results(results, path, key=None, title=None, subtitle=None, fields=None, max_bytes=None) -> str:
    """Write the player page of a mode_shapes result (key 'physics.feature'; None = the only one) of a Results,
    live or loaded: in a sweep every successful value on the same surface mesh, its Bloch wavenumber as the band
    diagram abscissa (another swept parameter otherwise); fields = the stored PAGE_FIELDS to put on the page (None =
    every stored one); max_bytes: ModePageTooLarge (nothing written) when the mode data would pass it. Returns the
    written path."""
    key = _mode_key(results, key)
    entries = _mesh_entries(results, key)
    first = entries[-1][1]
    groups = _page_groups([entry for _, entry in entries], fields)
    if max_bytes is not None:   # before any mode is packed
        sizes = page_bytes(first['nodes'].shape[0], _mode_count(entries), groups)
        if sum(sizes.values()) > max_bytes:
            raise ModePageTooLarge(sizes, max_bytes)
    summary = results.summary()
    sweep = summary.get('sweep') or {}
    swept_values = list(sweep.get('values') or [])
    periodic = np.asarray(first.get('periodic', np.zeros(3)), dtype=bool)
    period = np.asarray(first.get('period', np.zeros(3)), dtype=np.float64)
    k_all = np.array([np.asarray(entry.get('bloch_k', np.zeros(3)), dtype=np.float64) for _, entry in entries])
    axis = None
    if periodic.any():
        spread = np.where(periodic, np.abs(k_all).max(axis=0), -1.0)
        axis = int(np.argmax(spread))
    k_fractions = [float(k[axis] * period[axis] / (2 * math.pi)) if axis is not None else None for k in k_all]
    by_k = axis is not None and len(entries) > 1 and np.ptp(k_fractions) > 1e-12
    by_parameter = not by_k and len(entries) > 1 and all(index is not None for index, _ in entries)
    modes, rows, xs = [], [], []
    for row, (index, entry) in enumerate(entries):
        frequency = np.asarray(entry['frequency'], dtype=np.float64)
        q = np.asarray(entry.get('q_factor', np.full(frequency.shape, np.nan)), dtype=np.float64)
        re = np.asarray(entry['displacement_re'], dtype=np.float64)
        im = np.asarray(entry['displacement_im'], dtype=np.float64)
        if by_k:
            x = k_fractions[row]
        elif by_parameter:
            x = float(swept_values[index]) if index < len(swept_values) else float(row)
        else:
            x = None
        xs.append(x)
        rows.append(frequency.tolist())
        rank = np.argsort(np.argsort(np.where(np.isfinite(frequency), frequency, np.inf)))
        cell_phase = -float(k_all[row][axis] * period[axis]) if axis is not None else 0.0
        for j in range(frequency.size):
            if not math.isfinite(frequency[j]):
                continue
            extra = {FIELD_GROUPS[group][0]: _group_values(entry, group, j) for group in groups}
            modes.append(PlayerMode(frequency=float(frequency[j]), displacement=re[j] + 1j * im[j],
                                    k=x, k_index=row if x is not None else None, band=int(rank[j]) + 1,
                                    cell_phase=cell_phase, q_factor=float(q[j]) if j < q.size else None,
                                    label=f'resonance {j + 1}', **extra))
    bands = None
    if by_k:
        bands = BandTable(k=xs, frequencies=rows, k_label='Wavenumber  k a / 2π', k_symbol='k a/2π',
                          k_ticks=_k_ticks(xs), k_range=(min(xs), max(xs)))
    elif by_parameter:
        bands = BandTable(k=xs, frequencies=rows, k_label=str(sweep.get('parameter') or 'swept value'),
                          k_symbol=str(sweep.get('parameter') or 'value'))
    middle = min(range(len(modes)), key=lambda i: (abs((modes[i].k_index or 0) - (len(entries) - 1) / 2),
                                                   modes[i].band or 0))
    spec = getattr(results, 'spec', None)
    model_title = (summary.get('model') or {}).get('title') or (getattr(spec, 'title', None) if spec else None)
    if title is None:
        title = model_title or 'Mode shapes'
    if subtitle is None:
        physics = key.partition('.')[0]
        counts = f'{len(entries)} wavenumbers' if by_k else (f'{len(entries)} values of {sweep.get("parameter")}'
                                                              if by_parameter else 'one solve')
        subtitle = f'{physics}: {len(modes)} modes, {counts} (open_sol)'
    dimensions = 2 if np.ptp(first['nodes'][:, 2]) == 0.0 else 3
    lattice = {} if axis is None else {'lattice_period': float(period[axis]),
                                       'lattice_axis': tuple(float(i == axis) for i in range(3))}
    return write_mode_player(path, first['nodes'], first['triangles'], modes, bands=bands, title=title,
                             subtitle=subtitle, dimensions=dimensions, default_mode=middle,
                             density=first.get('density'), **lattice)
