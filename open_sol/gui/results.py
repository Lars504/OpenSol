"""ResultsPanel: field selection, snapshots and export plus result tabs (probes, spectra, far field, DFT, curves & tables,
derived values, derived curves, sweep, line profile; hidden without content). Every curve plot is a plot_view
ResultPlotView fed from GUI-thread data; session access only via bridge.call (CPU data back)."""
import copy
import math
import os

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from ..core.mirror import mirror_faces, unfold_field
from ..core.renderer import FieldInfo
from ..io.model_io import export_field, export_result, export_volume
from ..model.analytic import (DEGREE_FIT_OPERATIONS, FFT_SERIES, RATIO_SERIES, built_parameters, derived_curve_plots, derived_curve_result,
                              feature_curve, field_plane_height, is_curve_source, is_segment_3d, is_wavelength_xy,
                              line_curve,
                              line_endpoints, line_plane, numeric_table_columns, parameters_with, relative_error,
                              result_value, snapshot_index, split_key, sweep_curve, sweep_table_xy, sweep_value_curve,
                              table_xy)
from ..model.interface_sampling import interface_line_profile, line_curve_field_keys, projected_info
from ..model.expressions import evaluate_parameters, evaluate_scalar, error_message
from ..model.meshing import MESH_ESTIMATE_ERRORS, estimate_mesh
from ..model.session import FieldData
from ..model.spec import ModelSpec
from ..physics.em.results import _DFT_QUANTITY
from ..viz.result_plots import table_cell_text
from ..viz.common import length_unit, line_profile, normalised_db, probe_spectrum, safe_ratio, time_unit
from .estimation_results import EstimationTab, estimation_curve_options, estimation_plots, estimation_report
from .document import results_inputs
from .forms import (derived_operation_text, display_label, fill_table, read_only_table, rich_tooltip, selectable,
                    table_number)
from .helpers import (USER_ERRORS, ask_save_path, evaluation_mesh, field_mesh_name, format_duration,
                      physics_mesh_name, remember_directory, several_meshes, study_summary_text)
from .plot_view import (METRES_PER_NM, PLOT_FILE_TYPES, POINT_OPTIONS, SERIES_COLORS, TABLE_FILE_TYPES, ResultPlotView,
                        curve_xy, dashed_pen, file_stem, write_export)

_COLORMAPS = ('viridis', 'inferno', 'magma', 'plasma', 'cividis', 'turbo', 'RdBu_r', 'coolwarm', 'seismic', 'gray',
              'YlOrBr')
_FAR_FIELD_DB_FLOOR = -60.0
_DERIVED_TABLE_SECTIONS = ('all', 'results', 'physics', 'mesh', 'parameters')   # document edits that change its rows
_CSV_FIRST_FILE_TYPES = TABLE_FILE_TYPES[::-1]   # table results: their .npz holds metadata only
_DERIVED_EXPORT_COLUMNS = ('name', 'expression', 'operation', 'value', 'reference', 'relative_error', 'tolerance')
_PROFILE_HINT = 'Draw a line over the field view to plot a profile, or choose the line of an analytic curve.'
_FLUX_QUANTITY = 'flux toward +x / +y'   # Spectra quantity name of a flux_spectrum's power (sign convention)
_SPECTRUM_SHOW_CHOICES = ('Results', 'Ratio to reference')   # Spectra Show combo; index 1 = ratios
_FLUX_SERIES = 'power'   # AnalyticCurveSpec.series of a flux spectrum's power (_FLUX_QUANTITY)
_HZ_DFT_QUANTITY = 'abs(H_z)^2'   # DFT tab label of an H_z monitor's 'hz_intensity'
_CURVE_ERRORS = USER_ERRORS + (TypeError,)   # a failing analytic curve: reported in the view's error line, not drawn
_DRAWN_PROFILE = '(drawn line)'   # Line profile combo entry of the canvas profile tool
_PLANE_AXIS_LABELS = {'x': 'Side (yz at x)', 'y': 'Front (xz at y)', 'z': 'Horizontal (xy at z)'}   # cut axis combo
_RUN_NOTE = 'Study running: this plot shows earlier results (or none) until the run ends.'
_BUILD_FAILED_REASON = 'the last build failed'   # banner cause after BuildFailed (kept by the StudyFailed that follows)
# derived references (Sweep, Derived curves): grey dotted, off the SERIES_COLORS palette
_REFERENCE_PEN = pg.mkPen('#9a9a9a', width=2.5, style=QtCore.Qt.PenStyle.DotLine)
# Convergence tab (mesh refinement levels, S7.7)
_NO_CONVERGENCE = 'No mesh refinement in the last study (Study settings: Mesh refinement).'
_CONVERGENCE_REFERENCE_SERIES = ('Richardson extrapolation', 'tolerance')   # drawn with _REFERENCE_PEN
_STOP_REASON_TEXTS = {'max_levels': 'maximum number of refinements reached', 'max_cells': 'maximum number of cells '
                      'reached', 'memory': 'the next level would not fit in memory', 'no_refinement': 'no further '
                      'refinement possible', 'not_finite': 'the watched value is not a finite number', 'cancelled':
                      'stopped', 'failed': 'a level failed'}
_KEEP_MESH_TIP = ("Add the meshes of the last refinement level to the model as new meshes named '<mesh>_adapted' and "
                  'run the physics of each refined mesh on them, so later runs start from the adapted mesh.\n\n'
                  'The original meshes stay in the model, unused. Undo removes the new ones again.')
_CONVERGENCE_TABLE_TIP = ('One row per mesh refinement level; level 0 is the run on the model meshes.\n\n'
                          '- Cells: every cell of the mesh, absorbing layers included (triangle meshes: elements)\n'
                          '- Relative change: change of the watched derived value from the previous level, relative '
                          'to its value; the refinement stops when it is within the tolerance')


# ---- worker-side helpers (run inside bridge.call on the worker thread; return CPU data) ----
def _fetch_feature_results(session, keys):
    """{(physics, feature): 1D result dict (dft_fields: metadata only) or error text} of the given features."""
    results = {}
    for physics, feature in keys:
        try:
            results[(physics, feature)] = session.feature_result(physics, feature)
        except Exception as exc:   # one failing feature must not hide the others; reported in the panel
            results[(physics, feature)] = f'{type(exc).__name__}: {exc}'
    return results


def _evaluate_derived_values(session, derived_specs, curves):
    """{name: (_derived_definition, value, error text or None)} of DerivedValueSpec copies; NaN for a failing one. A
    derived-curve source is built from `curves` (the document's DerivedCurveSpec copies, not the built spec's)."""
    curves_by_name = {curve.name: curve for curve in curves}
    fetch = _feature_fetcher(session)
    values = {}
    for derived in derived_specs:
        definition = _derived_definition(derived, curves_by_name)
        earlier = {name: entry[1] for name, entry in values.items()}   # names of the derived values listed above
        try:
            if is_curve_source(derived.source):
                curve = curves_by_name.get(derived.source)
                if curve is None:
                    raise KeyError(f'no derived curve named {derived.source!r}')
                value = result_value(derived_curve_result(curve, fetch, session.parameters), derived,
                                     {**earlier, **session.parameters})
            else:
                value = session.derived_value(derived, fetch, earlier)
            values[derived.name] = (definition, value, None)
        except Exception as exc:   # one failing value must not hide the others; shown in the table
            values[derived.name] = (definition, math.nan, f'{type(exc).__name__}: {exc}')
    return values


def _evaluate_derived_curves(session, curves):
    """analytic.derived_curve_plots of DerivedCurveSpec copies over the session's feature results (each fetched once)
    and parameters."""
    return derived_curve_plots(curves, _feature_fetcher(session), session.parameters)


def _feature_fetcher(session):
    """fetch('physics.feature') -> the session's feature result, each fetched once."""
    fetched = {}

    def fetch(key):
        if key not in fetched:
            fetched[key] = session.feature_result(*split_key(key))
        return fetched[key]
    return fetch


def _session_field(session, field_key, plane=None, native=False):
    """Session.field of a 'physics.component' key (native cells with centres; plane: 3D cut (axis, position); native:
    the (r, angle) cells of a polar / spherical window)."""
    physics, component = split_key(field_key, 'physics.component')
    return session.field(component, physics, plane=plane, native=native)


def _snapshot_field(snapshot, field_key, plane=None):
    """FieldData of a snapshot: the default cut (plane None), else (3D) the cut it holds on plane's axis nearest plane's
    position (the mid-height z cut or a stored cut, so each sweep value's own positions); KeyError when none."""
    if plane is None:
        return snapshot.fields[field_key]
    held = [fields[field_key] for fields in [snapshot.fields, *snapshot.cuts]
            if field_key in fields and fields[field_key].plane is not None and fields[field_key].plane[0] == plane[0]]
    if not held:
        raise KeyError(f'{field_key} on a {plane[0]} cut (not among the stored cut planes)')
    return min(held, key=lambda field_data: abs(float(field_data.plane[1]) - plane[1]))


def _session_segment_profile(session, field_key, start, end, sampling='nearest'):
    """Session.segment_profile of a 'physics.component' key along a 3D segment."""
    physics, component = split_key(field_key, 'physics.component')
    return session.segment_profile(component, start, end, physics=physics, sampling=sampling)


def _dft_field_data(session, physics, feature, freq_index, intensity_key='intensity'):
    """DFT intensity_key map ('intensity' = the monitor quantity, 'hz_intensity') of a DFT feature at one frequency as
    a float64 FieldData for the canvas (|H_z|^2 ~ 1e-35 would underflow float32; the canvas casts for display)."""
    result = session.feature_result(physics, feature, freq_index=freq_index)
    if intensity_key not in result:
        raise ValueError(f"DFT result of {physics}.{feature} has no '{intensity_key}'")
    quantity = _HZ_DFT_QUANTITY if intensity_key == 'hz_intensity' else result.get('quantity', 'intensity')
    info = FieldInfo(name=f'dft_{intensity_key}', label=f'{quantity} at {feature}', unit='a.u.', cmap='inferno')
    return FieldData(physics=physics, component=f'dft_{intensity_key}',
                     values=np.asarray(result[intensity_key], dtype=np.float64), extent=tuple(result['extent']),
                     info=info, time=float(session.time),
                     time_label=_wavelength_text(np.atleast_1d(result['wavelength'])[freq_index]),
                     x_centers=np.asarray(result['x_centers']), y_centers=np.asarray(result['y_centers']))


# ---- shared helpers ----
def _derived_definition(derived, curves):
    """What a DerivedValueSpec's value depends on (expression, selection, operation, source, at, window, wavelength,
    fit, degree; a derived-curve source adds that curve's expression and inputs, curves = {name: DerivedCurveSpec}):
    a stored value is shown only while it is unchanged (reference / tolerance are judged on the GUI thread)."""
    curve = curves.get(derived.source) if is_curve_source(derived.source) else None
    curve_definition = None
    if curve is not None:   # input order matters: the first array input gives the abscissa
        curve_definition = (str(curve.expression),
                            tuple((str(alias), str(key)) for alias, key in curve.inputs.items()))
    return (str(derived.expression), tuple(derived.selection), str(derived.operation), derived.source,
            _optional_text(derived.at), tuple(str(bound) for bound in derived.window or ()),
            _optional_text(derived.wavelength), str(derived.fit), str(derived.degree), str(derived.point),
            curve_definition)


def _optional_text(value):
    return None if value is None else str(value)


def _curves_by_name(spec):
    """{name: DerivedCurveSpec} of a spec (derived-value curve sources)."""
    return {curve.name: curve for curve in spec.derived_curves}


def _operation_text(derived, labelled=False):
    """Text of the operation with its at / point / fit / degree / window / wavelength; labelled: the settings
    drop-down text of the operation (table), else its key (CSV export)."""
    text = derived_operation_text(derived.operation) if labelled else str(derived.operation)
    if derived.operation == 'at':   # a source abscissa, or a field's point object
        text += f' {derived.at if derived.at is not None else derived.point}'
    if derived.fit:
        text += f' {derived.fit}'
    if derived.operation in DEGREE_FIT_OPERATIONS:
        text += f' (degree {derived.degree})'
    if derived.window:
        text += f" over [{', '.join(str(bound) for bound in derived.window)}]"
    if derived.wavelength is not None:
        text += f' at λ = {derived.wavelength}'
    return text


def _key_text(key):
    return f'{key[0]}.{key[1]}'


def _wavelength_text(wavelength):
    """'λ = 1532.5 nm' label of a wavelength [m]."""
    return f'λ = {float(wavelength) / METRES_PER_NM:.6g} nm'


def _wavelength_labels(result):
    """Frequency-combo labels of a fetched spectral result ([] while it is not fetched)."""
    if not isinstance(result, dict):
        return []
    return [_wavelength_text(wavelength) for wavelength in np.atleast_1d(result['wavelength'])]


def _dft_intensity_choices(result):
    """[(combo label, result key)] of a fetched dft_fields metadata result; an H_z monitor adds 'hz_intensity'."""
    if not isinstance(result, dict):
        return []
    quantity = result.get('quantity', 'intensity')
    choices = [(quantity, 'intensity')]
    if quantity == _DFT_QUANTITY['hz']:
        choices.append((_HZ_DFT_QUANTITY, 'hz_intensity'))
    return choices


def _spectrum_quantities(result):
    """{quantity: (values, unit)} of a spectral result: a flux spectrum's power (_FLUX_QUANTITY), else the xy series."""
    if result.get('kind') == 'flux_spectrum':
        return {_FLUX_QUANTITY: (result['power'], result.get('unit', 'a.u.'))}
    series_units = result.get('series_units') or {}
    return {name: (values, series_units.get(name, '')) for name, values in result['series'].items()}


def _spectrum_curve(result, quantity):
    """(wavelength [m], values float64, unit) of one quantity of a spectral result; None when it lacks the quantity."""
    if not isinstance(result, dict):
        return None
    entry = _spectrum_quantities(result).get(quantity)
    if entry is None:
        return None
    wavelength = result['wavelength'] if result.get('kind') == 'flux_spectrum' else result['x']
    return np.asarray(wavelength, dtype=np.float64), np.asarray(entry[0], dtype=np.float64), entry[1]


def _spectrum_ratio(curve, reference_curve):
    """values / reference values on identical wavelengths (NaN where the reference vanishes); None on a mismatch."""
    wavelength, values, _ = curve
    reference_wavelength, reference_values, _ = reference_curve
    if wavelength.shape != reference_wavelength.shape or not np.allclose(wavelength, reference_wavelength,
                                                                         rtol=1e-9, atol=0.0):
        return None
    return safe_ratio(values, reference_values)


def _ratio_note(reference_key, reference_curve, quantity, mismatched):
    """Status text under the Spectra ratio plot."""
    if reference_key is None:
        return ('Choose a reference for the ratio plot (there is no automatic empty-reference run: use a monitor of '
                'this model or rerun without the scatterer).')
    reference_label = _key_text(reference_key)
    if reference_curve is None:
        return f"{reference_label} has no {quantity or 'result'} to divide by."
    if mismatched:
        return f"Wavelengths differ from the reference for: {', '.join(mismatched)}"
    return f'{quantity} relative to {reference_label}.'


def _draw_polar(plot, angles, normalised_power):
    """Hand-drawn polar plot: rings at 0.25..1, spokes every 30 deg, the pattern r = normalised power."""
    plot.clear()
    grid_pen = pg.mkPen((175, 175, 175), width=1)
    theta = np.linspace(0.0, 2.0 * np.pi, 181)
    for radius in (0.25, 0.5, 0.75, 1.0):
        plot.plot(radius * np.cos(theta), radius * np.sin(theta), pen=grid_pen)
        ring_label = pg.TextItem(f'{radius:g}', color=(120, 120, 120), anchor=(0.0, 1.0))
        ring_label.setPos(radius * math.cos(math.radians(80.0)), radius * math.sin(math.radians(80.0)))
        plot.addItem(ring_label)
    for degrees in range(0, 360, 30):
        angle = math.radians(degrees)
        plot.plot([0.0, math.cos(angle)], [0.0, math.sin(angle)], pen=grid_pen)
        spoke_label = pg.TextItem(f'{degrees}°', color=(90, 90, 90), anchor=(0.5, 0.5))
        spoke_label.setPos(1.13 * math.cos(angle), 1.13 * math.sin(angle))
        plot.addItem(spoke_label)
    closed_angles = np.append(angles, angles[:1])
    closed_power = np.append(normalised_power, normalised_power[:1])
    plot.plot(closed_power * np.cos(closed_angles), closed_power * np.sin(closed_angles),
              pen=pg.mkPen(SERIES_COLORS[0], width=2))
    plot.setRange(xRange=(-1.25, 1.25), yRange=(-1.25, 1.25), padding=0.0)


def _iteration_axis(times):
    """(1-based sample index, axis label) when every sample shares one time (stationary iterations), else None;
    the rule of visualization._draw_timeseries."""
    times = np.asarray(times, dtype=np.float64)
    if times.size == 0 or (times.size > 1 and np.ptp(times) > 0):
        return None
    return np.arange(1, times.size + 1), f'Iteration (all at t = {_time_text(times[0])})'


def _frames_range(frames):
    """[min, max] of the finite values over every FieldData in frames (colour limits of a replay), None if none."""
    lows, highs = [], []
    for frame in frames:
        values = frame.values
        finite = np.isfinite(values)
        if not finite.all():   # NaN (blank) or infinite cells: a copy of the finite ones
            values = values[finite]
        if values.size:
            lows.append(float(values.min()))
            highs.append(float(values.max()))
    return np.array([min(lows), max(highs)]) if lows else None


def _final_value(result):
    """Last sample of a timeseries result dict (NaN when empty or missing)."""
    values = np.asarray(result['values'], dtype=np.float64) if isinstance(result, dict) else np.empty(0)
    return float(values[-1]) if values.size else math.nan


def _sector_number(values, index):
    """'%.6g' text of values[index] for the Parity sectors table ('' when missing, 'NaN' when not finite)."""
    try:
        value = float(values[index])
    except (IndexError, TypeError, ValueError):
        return ''
    return f'{value:.6g}' if math.isfinite(value) else 'NaN'


def _sweep_series_choices(result):
    """Series of one sweep value's result the Sweep tab offers ('' = the default curve of sweep_value_curve): every
    xy series, a flux spectrum's power and power_per_cw, else ''."""
    kind = result.get('kind') if isinstance(result, dict) else None
    if kind == 'xy':
        return list(result['series'])
    if kind == 'flux_spectrum':
        return ['power'] + ['power_per_cw'] * ('power_per_cw' in result)
    return ['']


def _time_text(seconds):
    time_scale, time_unit_text = time_unit(abs(float(seconds)) or 1e-15)
    return f'{float(seconds) / time_scale:.4g} {time_unit_text}'


def _error_text(error):
    """Message of a bridge.call failure (text or an object with .message)."""
    return str(getattr(error, 'message', error))


def _series_x(xy, series):
    """SI x of one series of an xy dict (its series_x entry, else the plot's x)."""
    return (xy.get('series_x') or {}).get(series, xy['x'])


def _built_cells_text(summary):
    """'412 × 210 cells' of a BuildSummary; with several meshes "mesh 'm' 412 × 210 cells; mesh 'h' 1,200 elements"."""
    mesh_records = getattr(summary, 'meshes', None) or []
    if len(mesh_records) > 1:   # S7.2: one size per built mesh
        return '; '.join(f"mesh '{record['name']}' " + (f"{record['elements']:,} elements"
                                                        if record.get('elements') is not None else
                                                        ' × '.join(map(str, record.get('shape') or ())) + ' cells')
                         for record in mesh_records)
    z_count = f' × {summary.nz}' if summary.nz > 1 else ''
    return f'{summary.nx} × {summary.ny}{z_count} cells'


class ResultsPanel(QtWidgets.QWidget):
    """Results dock: field display / snapshots / export and 1D result tabs; data arrives through the WorkerBridge."""
    availability_changed = QtCore.Signal(bool)   # has_results changed (the tree's 'Results' node follows it)
    # 3D preview cut planes: True = the shown session field or its cut changed (refetch), False = colours only
    field_view_changed = QtCore.Signal(bool)

    def __init__(self, document, bridge, canvas, parent=None):
        super().__init__(parent)
        self._document = document
        self._bridge = bridge
        self._canvas = canvas
        self._summary = None             # BuildSummary of the current session
        self._study_summary = None       # StudyFinished.summary of the last study
        self._session_parameters = None  # parameters of the worker session (fetched after Built), None until known
        self._built_inputs = (None, None)   # (bridge.results_spec, its _results_inputs) for the out-of-date check
        self._run_active = False        # a study of the current model runs (curve views show _RUN_NOTE)
        self._results = {}               # (physics, feature) -> fetched 1D result dict
        self._live_series = {}           # (physics, feature) -> live timeseries dict
        self._fft_cache = {}             # (physics, feature) -> (result dict, sample count, remove_mean, spectrum)
        self._derived_values = {}        # derived value name -> (_derived_definition, value, error text or None)
        self._built_derived = {}         # derived value name -> _derived_definition in the spec of the last build
        self._profile = None             # profile dict shown in the Line profile tab ('curve': line curve name)
        self._profile_curve_names = []   # line analytic curves parallel to the profile combo (after its first entry)
        # (fetch key, run summary, (fields, moment, mid_run)) of the last line-curve fetch: an edit that leaves the key (curve line,
        # field, plane, snapshot, interface, projection, sampling, parameters, run) unchanged redraws without a fetch
        self._profile_fetched = None
        self._field_choices = []         # [(physics, component)] parallel to the field combo (the chosen physics)
        self._field_infos = {}           # (physics, component) -> FieldInfo of every physics
        self._field_physics = []         # physics names parallel to the physics combo of the Field tab
        self._feature_kinds = []         # [(physics, feature, result_kind)] of the current build
        self._probe_keys, self._far_field_keys, self._dft_keys = [], [], []
        self._spectrum_keys = []         # flux spectra + xy results known to be versus wavelength, in feature order
        self._xy_table_keys = []         # 'table' and the other 'xy' results (unfetched xy included), in feature order
        self._wavelength_xy_keys = set() # xy keys whose fetched result was versus wavelength (kept until replaced)
        self._ratio_notes = {}           # Spectra quantity -> ratio note text of the last ratio plots
        self._sweep_plots = []           # [(label, xy or None)] of the last study's sweep (the Sweep view's plots)
        self._sweep_quantities = {}      # sweep plot label -> ('derived' | 'final' | 'series', name or
                                         #  'physics.feature') | ('table', ('physics.feature', column)) |
                                         #  ('result', ('physics.feature', series))
        self._sweep_value_results = {}   # 'result' sweep plot label -> {series: (swept value, result dict)}
        self._live_sweep = None          # SweepValueDone.sweep of the sweep still running (None: none / finished)
        self._derived_series_plots = []  # [(label, xy)] of the last run's derived_series (Derived values tab)
        self._sweep_field_value = None   # index of the sweep value whose kept snapshots the Field tab shows, or None
        self._snapshot_times = []
        self._snapshot_labels = []       # label per snapshot ('lambda = ... nm'), None for time snapshots
        self._show_next_live_frame = False
        self._follow_paused_by_view = False   # 'Follow live' unticked by a snapshot / DFT view: re-ticked on next run
        self._derived_pending = False
        self._derived_curve_plots = []   # analytic.derived_curve_plots of the last evaluation
        self._derived_curves_pending = False
        self._symmetric_override = None  # bool once the user toggled 'Symmetric limits'; None = FieldInfo.signed
        self._session_state_field = None # FieldData of show_selected_field; cleared by a build / study
        self._session_lost_reason = None # why the worker holds no session for these results (banner text), or None
        self._had_results = False        # has_results at the last availability_changed
        self._plot_views = []            # every ResultPlotView of the panel (_plot_view)
        self._refinement_level = None    # mesh refinement level being computed (counted by Built), None: no such run

        self._snapshot_timer = QtCore.QTimer(self)
        self._snapshot_timer.setSingleShot(True)
        self._snapshot_timer.setInterval(150)
        self._snapshot_timer.timeout.connect(self._fetch_snapshot)
        self._replay_frames = None       # FieldData per snapshot of _replay_key (fetched on Play), or None
        self._replay_key = None          # ('physics.component', _snapshot_plane()) of _replay_frames
        self._unfolded_frames = {}       # id(frame) -> (frame, unfolded copy) of _replay_frames (Unfold mirror planes)
        self._replay_range = None        # [min, max] over every replay frame: fixed colour limits, or None
        self._replay_token = 0           # bumped when the snapshots change: drops a frame fetch still pending
        self._play_timer = QtCore.QTimer(self)
        self._play_timer.timeout.connect(self._advance_replay)

        self._stale_label = selectable(QtWidgets.QLabel('Results out of date: the model changed after the last build '
                                                        '/ run.'))
        self._stale_label.setStyleSheet('background:#fff3cd; color:#664d03; padding:3px;')
        self._stale_label.setWordWrap(True)
        self._stale_label.hide()
        self._sweep_label = selectable(QtWidgets.QLabel())   # running sweep value / mesh refinement level
        self._sweep_label.setWordWrap(True)
        self._sweep_label.hide()
        self._level_label = selectable(QtWidgets.QLabel())   # mesh refinement: level / meshes the results come from
        self._level_label.setStyleSheet('background:#e7f0fb; color:#1a3a66; padding:3px;')
        self._level_label.setWordWrap(True)
        self._level_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self._level_label.hide()
        self._message_label = QtWidgets.QLabel()
        self._message_label.setWordWrap(True)
        self._message_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)   # error texts may contain '<...>'
        self._message_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self._tabs = QtWidgets.QTabWidget()
        self._probes_tab = self._build_probes_tab()
        self._spectra_tab = self._build_spectra_tab()
        self._far_field_tab = self._build_far_field_tab()
        self._dft_tab = self._build_dft_tab()
        self._xy_table_tab = self._build_xy_table_tab()
        self._derived_tab = self._build_derived_tab()
        self._derived_curves_tab = self._build_derived_curves_tab()
        self._sweep_tab = self._build_sweep_tab()
        self._sectors_tab = self._build_sectors_tab()
        self._convergence_tab = self._build_convergence_tab()
        self._estimation_tab = EstimationTab(document, self._plot_view(
            lambda: estimation_plots(estimation_report(self._study_summary)), curve_options=estimation_curve_options,
            note=lambda name, xy: self._run_note(), export_stem=lambda name: file_stem(name, 'estimation'),
            empty_text='run a study with parameter estimation first'))
        self._estimation_tab.message.connect(self._show_message)
        for widget, title in ((self._build_field_tab(), 'Field'), (self._probes_tab, 'Probes'),
                              (self._spectra_tab, 'Spectra'), (self._far_field_tab, 'Far field'),
                              (self._dft_tab, 'DFT'), (self._xy_table_tab, 'Curves & tables'),
                              (self._derived_tab, 'Derived values'), (self._derived_curves_tab, 'Derived curves'),
                              (self._sweep_tab, 'Sweep'), (self._sectors_tab, 'Parity sectors'),
                              (self._convergence_tab, 'Convergence'), (self._estimation_tab, 'Estimation'),
                              (self._build_profile_tab(), 'Line profile')):
            self._tabs.addTab(widget, title)
        self._update_tab_visibility()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._stale_label)
        layout.addWidget(self._sweep_label)
        layout.addWidget(self._level_label)
        layout.addWidget(self._tabs, 1)
        layout.addWidget(self._message_label)

        for signal, handler in ((bridge.built, self._on_built), (bridge.build_failed, self._on_build_failed),
                                (bridge.live_frame, self._on_live_frame),
                                (bridge.live_results, self._on_live_results),
                                (bridge.study_started, self._on_study_started),
                                (bridge.study_finished, self._on_study_finished),
                                (bridge.study_failed, self._on_study_failed),
                                (bridge.sweep_step, self._on_sweep_step),
                                (bridge.sweep_value_done, self._on_sweep_value_done)):
            signal.connect(self._current_only(handler))   # a job of a replaced model keeps running unseen
        document.replaced.connect(self._on_document_replaced)
        document.changed.connect(self._on_document_changed)
        canvas.profile_changed.connect(self._on_profile_changed)
        canvas.field_cut_requested.connect(self._on_canvas_cut_requested)
        self._schedule_derived_table()
        self._update_profile_choices()

    # ---- public API ----
    @property
    def has_results(self) -> bool:
        """True while a build or study of the current model left something to show."""
        return self._summary is not None or self._study_summary is not None

    def refresh(self):
        """Fetch every feature result of the current session (dft_fields: metadata only) via bridge.call."""
        if self._summary is None:
            return
        keys = tuple((physics, feature) for physics, feature, _ in self._summary.feature_results)
        if keys:
            self._call(lambda session, keys=keys: _fetch_feature_results(session, keys), self._on_feature_results)

    def show_selected_field(self):
        """Fetch the selected field of the current session and show it on the canvas."""
        self._stop_replay()
        selection = self._selected_field()
        if selection is None:
            self._show_message('No fields yet: build the model first.')
            return
        physics, component = selection
        plane = self._selected_plane()
        self._call(lambda session, physics=physics, component=component, plane=plane:
                   session.canvas_field(component, physics, plane=plane), self._show_session_field)

    def fetch_cut_planes(self, on_result, on_error, max_pixels) -> str:
        """3D preview: fetch the selected field of the current session on the x, y and z cuts (the chosen cut on its
        axis, the middle window cell on the others) -> on_result([FieldData]); '' when sent, else why not."""
        selection, chosen = self._selected_field(), self._selected_plane()
        if self._summary is None or self._bridge.results_spec is None or selection is None or chosen is None:
            return 'Field cut planes: build (and run) the 3D model, then choose the field in Results.'
        if self._bridge.busy:
            return 'Field cut planes: drawn again when the running job has finished.'
        physics, component = selection
        planes = [chosen if axis == chosen[0] else (axis, centres[len(centres) // 2])
                  for axis in 'xyz' for centres in (self._plane_centres.get(axis) or [],)
                  if centres or axis == chosen[0]]
        self._call(lambda session, physics=physics, component=component, planes=planes:
                   [session.display_field(component, physics, max_pixels=max_pixels, plane=plane) for plane in planes],
                   on_result, on_error=on_error)
        return ''

    def cut_plane_colors(self) -> tuple:
        """(colormap name, symmetric about 0) chosen in the Field tab; None = the field's own."""
        return self._colormap_combo.currentData(), self._symmetric_override

    def _selected_plane(self):
        """(axis, position [m]) of the cut-plane selectors for 3D models, None in 2D (Session.field plane)."""
        position = self._plane_combo.currentData() if self._plane_combo.count() else None
        return None if position is None else (self._plane_axis_combo.currentData(), float(position))

    def _line_curve_plane(self):
        """Default plane of a line curve without its own: the chosen z cut, None (mid-height z) for a side cut."""
        plane = self._selected_plane()
        return plane if plane is not None and plane[0] == 'z' else None

    def _populate_plane_combo(self, summary):
        """Cut-plane axes and positions of a 3D build (window cell centres; the previous axis kept, else z, at its
        middle cell); hidden in 2D."""
        previous_axis = self._plane_axis_combo.currentData() or 'z'
        self._plane_centres = {axis: list(getattr(summary, f'{axis}_centers', ()) or ()) for axis in 'xyz'}
        self._plane_axis_combo.blockSignals(True)
        try:
            self._plane_axis_combo.clear()
            if self._plane_centres['z']:
                for axis in 'xyz':
                    self._plane_axis_combo.addItem(_PLANE_AXIS_LABELS[axis], axis)
                self._plane_axis_combo.setCurrentIndex('xyz'.index(previous_axis))
        finally:
            self._plane_axis_combo.blockSignals(False)
        self._fill_plane_positions()
        visible = bool(self._plane_centres['z'])
        for widget in (self._plane_label, self._plane_row, self._export_volume_button):
            widget.setVisible(visible)

    def _fill_plane_positions(self):
        """Position combo of the chosen cut axis: its window cell centres, the middle one selected."""
        axis = self._plane_axis_combo.currentData()
        centres = self._plane_centres.get(axis, []) if axis is not None else []
        self._plane_combo.blockSignals(True)
        try:
            self._plane_combo.clear()
            if centres:
                scale, unit = length_unit(max(abs(value) for value in centres) or 1e-6)
                for position in centres:
                    self._plane_combo.addItem(f'{axis} = {position / scale:.4g} {unit}', position)
                self._plane_combo.setCurrentIndex(len(centres) // 2)
        finally:
            self._plane_combo.blockSignals(False)

    def _on_plane_axis_selected(self):
        self._fill_plane_positions()
        self._on_plane_selected()

    def _on_canvas_cut_requested(self, axis, position):
        """Canvas View / cut edited in the field view: choose that cut plane (the cell centre nearest position; None =
        the current one on that axis, else the middle) as if picked here."""
        current = self._selected_plane()
        if position is None and current is not None and current[0] == axis:
            return
        if self._set_plane_choice(axis, position) and self._selected_plane() != current:
            self._on_plane_selected()

    def _set_plane_choice(self, axis, position):
        """Set the cut-plane combos (signals blocked, nothing fetched) to axis at the cell centre nearest position
        (None = the middle); False when the build has no such axis."""
        centres = self._plane_centres.get(axis) or []
        axis_index = self._plane_axis_combo.findData(axis)
        if not centres or axis_index < 0:
            return False
        self._plane_axis_combo.blockSignals(True)
        try:
            self._plane_axis_combo.setCurrentIndex(axis_index)
        finally:
            self._plane_axis_combo.blockSignals(False)
        self._fill_plane_positions()
        if position is not None:
            self._plane_combo.blockSignals(True)
            try:
                self._plane_combo.setCurrentIndex(min(range(len(centres)),
                                                      key=lambda index: abs(centres[index] - position)))
            finally:
                self._plane_combo.blockSignals(False)
        return True

    def _on_plane_selected(self):
        """Cut plane chosen: show it in the snapshot shown (when one is), else in the session state."""
        self._update_snapshot_label()
        if self._snapshot_shown and self._snapshot_times:
            self._stop_replay()
            self._fetch_snapshot()
        elif self._summary is not None and self._selected_field() is not None:
            self.show_selected_field()
        if self._current_profile_curve() is not None:   # a line curve without its own plane follows a z cut
            self._on_profile_choice()

    def export_field_data(self):
        """Ask for a .npz / .csv path and write the field shown on the canvas (io.model_io.export_field); a resampled
        view of the unchanged session state is refetched on the native cells (session.field, native: the (r, angle)
        cells of a polar / spherical domain)."""
        field_data = self._canvas.current_field
        if field_data is None:
            self._show_message('No field shown: show a field first.')
            return
        path = ask_save_path(self, self._document, 'Export field data',
                             f'{file_stem(f"{field_data.physics}.{field_data.component}")}.npz', TABLE_FILE_TYPES)
        if not path:
            return
        if (field_data.native_shape is not None and field_data is self._session_state_field
                and not self._bridge.busy):
            physics, component, plane = field_data.physics, field_data.component, getattr(field_data, 'plane', None)
            self._call(lambda session, physics=physics, component=component, plane=plane:
                       session.field(component, physics, plane=plane, native=True),
                       lambda native_field, path=path: self._write_field_export(native_field, path),
                       on_error=lambda error: self._show_message(f'Export failed: {_error_text(error)}'))
            return
        self._write_field_export(field_data, path)

    def export_volume_data(self):
        """3D models: ask for a .npz path and write the selected field of the current session state on every window
        cell (Session.volume, io.model_io.export_volume)."""
        selection = self._selected_field()
        if selection is None or not self._plane_centres.get('z'):
            self._show_message('Volume export needs a built 3D model: build it and select a field first.')
            return
        if self._bridge.busy:
            self._show_message('Volume export reads the current state: wait until the running job has finished.')
            return
        physics, component = selection
        path = ask_save_path(self, self._document, 'Export volume data',
                             f'{file_stem(f"{physics}.{component}")}_volume.npz', TABLE_FILE_TYPES[:1])
        if not path:
            return
        self._call(lambda session, physics=physics, component=component: session.volume(component, physics),
                   lambda volume, path=path: self._write_volume_export(volume, path),
                   on_error=lambda error: self._show_message(f'Export failed: {_error_text(error)}'))

    def _report(self, text, error):
        """write_export's report: the panel message line."""
        self._show_message(text, error=error)

    def _write_volume_export(self, volume, path):
        if not write_export(lambda target: export_volume(volume, target), path, self._report):
            return
        shape = ' × '.join(str(count) for count in volume.values.shape)
        self._show_message(f'Volume data ({shape} cells) written to {path}', error=False)

    def _write_field_export(self, field_data, path):
        """export_field plus a status message; a display resampling (snapshot, live frame) is named in it."""
        if not write_export(lambda target: export_field(field_data, target), path, self._report):
            return
        if field_data.native_shape is None:
            self._show_message(f'Field data written to {path}', error=False)
            return
        n_window_x, n_window_y = field_data.native_shape
        self._show_message(f'Field data written to {path}: nearest-cell display resampling of the {n_window_x} × '
                           f'{n_window_y} window cells (flagged in the metadata). For native cells show the current '
                           f'state (not a snapshot or live frame) and export again.', error=False)

    def export_image(self):
        """Write the canvas figure as shown (canvas.plot_panel.export, matplotlib) to the Figure box's export file, or,
        without one, to a .pdf / .svg / .eps / .png path asked for."""
        panel = self._canvas.plot_panel
        path = panel.export_path()
        if path:
            suffixes = [suffix for _label, suffix in PLOT_FILE_TYPES]
            if os.path.splitext(path)[1].lower() not in suffixes:
                self._show_message(f"Export file {path}: the name must end in {', '.join(suffixes)}.")
                return
        else:
            path = self._ask_figure_path()
            if not path:
                return
        write_export(panel.export, path, self._report, 'Figure')

    def _ask_figure_path(self):
        """Save dialog of a canvas figure file ('' when cancelled)."""
        return ask_save_path(self, self._document, 'Export figure', f'{self._canvas.mode}.pdf', PLOT_FILE_TYPES)

    def _choose_figure_path(self):
        """The Figure box's Choose…: the picked file becomes its export file."""
        path = self._ask_figure_path()
        if path:
            remember_directory(path)
            self._canvas.plot_panel.set_export_path(path)

    # ---- tab construction ----
    def _plot_view(self, plots, **options):
        """ResultPlotView of this panel: exports stamped with _export_metadata, messages to the message line."""
        view = ResultPlotView(self._document, plots, metadata=self._export_metadata, **options)
        view.message.connect(self._show_message)
        self._plot_views.append(view)
        return view

    def _build_field_tab(self):
        tab = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(tab)
        self._field_physics_combo = QtWidgets.QComboBox()
        self._field_physics_combo.setToolTip('Physics whose fields the Field list shows')
        self._field_physics_combo.currentIndexChanged.connect(lambda index: self._on_field_physics_selected())
        form.addRow('Physics', self._field_physics_combo)
        self._field_combo = QtWidgets.QComboBox()
        self._field_combo.currentIndexChanged.connect(self._on_field_selected)
        form.addRow('Field', self._field_combo)
        self._plane_centres = {}   # 3D models: cut axis -> window cell centres [m] (BuildSummary x / y / z_centers)
        self._plane_axis_combo = QtWidgets.QComboBox()   # 3D models: axis the cut plane is normal to
        self._plane_axis_combo.setToolTip(rich_tooltip(
            '3D models: direction of the cut through the field.\n\n'
            '- Horizontal (z): the layout seen from above, at one height.\n'
            '- Front (y) and Side (x): vertical cuts through the layer stack.\n\n'
            'Snapshots show the stored cut nearest the one chosen here (the mid-height horizontal cut is always '
            'stored; vertical cuts only when listed in the stored cut planes of the study); live frames always show '
            'the mid-height cut. Line curves without their own '
            'height use the horizontal cut chosen here, or mid-height while a vertical cut is shown.'))
        self._plane_axis_combo.currentIndexChanged.connect(lambda index: self._on_plane_axis_selected())
        self._plane_combo = QtWidgets.QComboBox()   # 3D models: cut position of the shown field (session.field plane)
        self._plane_combo.setToolTip('3D models: position of the cut through the field (a cell centre)')
        self._plane_combo.currentIndexChanged.connect(lambda index: self._on_plane_selected())
        self._plane_label = QtWidgets.QLabel('Cut plane')
        self._plane_row = self._row(self._plane_axis_combo, self._plane_combo, None)
        form.addRow(self._plane_label, self._plane_row)
        self._plane_label.hide()
        self._plane_row.hide()
        self._follow_live = QtWidgets.QCheckBox('Follow live updates')
        self._follow_live.setChecked(True)
        self._follow_live.clicked.connect(self._on_follow_live_clicked)   # user clicks only, not setChecked
        show_button = QtWidgets.QPushButton('Show')
        show_button.clicked.connect(lambda checked=False: self.show_selected_field())
        form.addRow(self._row(self._follow_live, None, show_button))
        self._colormap_combo = QtWidgets.QComboBox()
        self._colormap_combo.addItem('Field default', None)
        for name in _COLORMAPS:
            self._colormap_combo.addItem(name, name)
        self._colormap_combo.currentIndexChanged.connect(self._on_colormap_selected)
        form.addRow('Colormap', self._colormap_combo)
        self._symmetric_check = QtWidgets.QCheckBox('Symmetric limits about 0')
        self._symmetric_check.toggled.connect(self._on_symmetric_toggled)
        form.addRow('', self._symmetric_check)
        self._shown_field = None
        self._snapshot_shown = False   # _shown_field is a snapshot's (a cut-plane change refetches the snapshot)
        self._unfold_check = QtWidgets.QCheckBox('Unfold mirror planes')
        self._unfold_check.setToolTip(rich_tooltip(
            'Show the whole structure: the field is mirrored across every mirror boundary of the domain, with the '
            'sign each field component takes under that mirror.\n\n'
            'Exports and line profiles then use the unfolded field too.'))
        self._unfold_check.toggled.connect(self._on_unfold_toggled)
        form.addRow('', self._unfold_check)
        self._show_mesh_check = QtWidgets.QCheckBox('Show mesh')
        self._show_mesh_check.setToolTip(rich_tooltip(
            "Draw the cell lines of the mesh the shown field lives on (its physics' mesh, as the model defines it "
            'now) over the field.\n\nTriangle meshes: the triangles the results were computed on. After mesh '
            'refinement: the mesh of the last level.'))
        self._show_mesh_check.toggled.connect(self._canvas.set_show_field_mesh)
        form.addRow('', self._show_mesh_check)
        self._sweep_value_indices = [None]   # sweep value index per Sweep value combo entry (None: the session)
        self._sweep_value_combo = QtWidgets.QComboBox()
        self._sweep_value_combo.setToolTip(rich_tooltip(
            'Sweeps that keep the fields of every value: the sweep value whose stored snapshots the slider shows.\n\n'
            'The first entry is the current results (the last value); the others are read from the kept snapshots.'))
        self._sweep_value_combo.currentIndexChanged.connect(lambda index: self._on_sweep_value_selected())
        self._sweep_value_label = QtWidgets.QLabel('Sweep value')
        form.addRow(self._sweep_value_label, self._sweep_value_combo)
        self._sweep_value_label.hide()
        self._sweep_value_combo.hide()
        self._snapshot_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._snapshot_slider.setEnabled(False)
        self._snapshot_slider.valueChanged.connect(self._on_snapshot_moved)
        self._snapshot_label = selectable(QtWidgets.QLabel('No snapshots'))
        form.addRow('Snapshot', self._snapshot_slider)
        form.addRow('', self._snapshot_label)
        self._play_button = QtWidgets.QPushButton('Play')
        self._play_button.setCheckable(True)
        self._play_button.setToolTip(rich_tooltip(
            'Replay the stored snapshots of the selected field in order, one frame per tick.\n\n'
            "Snapshots are evenly spaced in simulated time (the study's output interval), "
            'so the replay runs in linear time; only the last frame (the final state) can follow a shorter or, once '
            'the stored snapshots reach 1 GB, a longer gap.'))
        self._play_button.toggled.connect(self._on_play_toggled)
        self._fps_spin = QtWidgets.QSpinBox()
        self._fps_spin.setRange(1, 60)
        self._fps_spin.setValue(10)
        self._fps_spin.setSuffix(' fps')
        self._fps_spin.valueChanged.connect(self._on_fps_changed)
        self._loop_check = QtWidgets.QCheckBox('Loop')
        self._fixed_limits_check = QtWidgets.QCheckBox('Fixed colour limits')
        self._fixed_limits_check.setChecked(True)
        self._fixed_limits_check.setToolTip(rich_tooltip(
            'On: one colour scale over all replay frames, so field strengths compare between frames.\n\n'
            'Off: each frame scales to its own range, so weak fields (early or late in the run) stay visible.'))
        self._fixed_limits_check.toggled.connect(self._on_fixed_limits_toggled)
        self._replay_rate_label = QtWidgets.QLabel()
        form.addRow('Replay', self._row(self._play_button, self._fps_spin, self._loop_check, None))
        form.addRow('', self._row(self._fixed_limits_check, None, self._replay_rate_label))
        export_data = QtWidgets.QPushButton('Export data…')
        export_data.clicked.connect(lambda checked=False: self.export_field_data())
        self._export_volume_button = QtWidgets.QPushButton('Export volume…')   # 3D models only
        self._export_volume_button.setToolTip(rich_tooltip(
            '3D models: save the selected field of the current state in every cell of the model (not only the cut '
            'shown) as a NumPy archive (.npz) with the cell centres along x, y and z.\n\n'
            'Large models need a lot of memory: above 1 GiB of values the export is refused; export cut planes '
            'instead, or use the scripting interface with a larger limit.'))
        self._export_volume_button.clicked.connect(lambda checked=False: self.export_volume_data())
        self._export_volume_button.hide()
        form.addRow(self._row(export_data, self._export_volume_button, None))
        # the canvas figure's Axes / Labels / Exports tabs (Export plot = the figure as shown, matplotlib)
        figure_panel = self._canvas.plot_panel
        figure_panel.export_requested.connect(self.export_image)
        figure_panel.export_path_requested.connect(self._choose_figure_path)
        form.addRow(figure_panel)
        return tab

    def _build_probes_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        self._probe_list = self._feature_list()
        self._probe_list.itemSelectionChanged.connect(self._update_probe_views)
        self._probe_view = self._plot_view(
            self._probe_plots, overlays=self._probe_overlays,
            curve_options=lambda name, xy, series: POINT_OPTIONS if xy['x_label'] != 'Time' else {},
            note=lambda name, xy: ('Live preview (min / max per display bin)'
                                   if xy is not None and xy.get('preview') else ''),
            export_stem=lambda name: 'probes',
            extra_export_actions=lambda menu: self._add_raw_export_actions(menu, self._selected_probe_keys()),
            downsample=True)
        self._remove_mean_check = QtWidgets.QCheckBox('Remove mean before FFT')
        self._remove_mean_check.setChecked(True)
        self._remove_mean_check.toggled.connect(lambda checked: self._fft_view.request_redraw())
        self._fft_view = self._plot_view(
            self._fft_plots, overlays=self._fft_overlays,
            note=lambda name, xy: ('FFT of a decimated live series: available after the run'
                                   if self._probes_decimated() else ''),
            export_stem=lambda name: 'probe_spectra', leading_widgets=(self._remove_mean_check,), downsample=True)
        layout.addWidget(self._probe_list)
        layout.addWidget(self._splitter(self._probe_view, self._fft_view), 1)
        return tab

    def _build_spectra_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        self._spectrum_list = self._feature_list()
        self._spectrum_list.itemSelectionChanged.connect(self._update_spectra_views)
        self._spectrum_show_combo = QtWidgets.QComboBox()
        self._spectrum_show_combo.addItems(_SPECTRUM_SHOW_CHOICES)
        self._spectrum_show_combo.setToolTip(rich_tooltip(
            'Results: each selected result as computed.\n\n'
            'Ratio to reference: each selected result divided by the reference result at every wavelength (for '
            'example a transmission).\n\n'
            'The list next to the plot choice shows all curves or one of them.'))
        self._spectrum_show_combo.currentIndexChanged.connect(lambda index: self._on_spectrum_show_changed())
        self._reference_label = QtWidgets.QLabel('Reference')
        self._reference_combo = QtWidgets.QComboBox()
        self._reference_combo.setToolTip('Result the ratios divide by')
        self._reference_combo.currentIndexChanged.connect(lambda index: self._update_spectra_views())
        self._spectrum_view = self._plot_view(
            self._spectrum_plots, overlays=self._spectrum_overlays,
            curve_options=lambda name, xy, series: {} if self._spectrum_shows_ratios() else POINT_OPTIONS,
            note=lambda name, xy: self._run_note(self._ratio_notes.get(name, '') if self._spectrum_shows_ratios()
                                                 else ''),
            export_stem=lambda name: (f'{name}_ratio' if name else 'spectra_ratio') if self._spectrum_shows_ratios()
            else name or 'spectra',
            extra_export_actions=lambda menu: self._add_raw_export_actions(
                menu, self._selected_keys(self._spectrum_list, self._spectrum_keys)))
        self._spectrum_view.plot_combo.setToolTip(rich_tooltip(
            'Quantity plotted for each selected result that has it.\n\n'
            'For example the power through a flux monitor, or one series of a mode monitor or probe.'))
        layout.addWidget(self._spectrum_list)
        layout.addWidget(self._row(QtWidgets.QLabel('Show'), self._spectrum_show_combo, self._reference_label,
                                   self._reference_combo, None))
        layout.addWidget(self._spectrum_view, 1)
        self._on_spectrum_show_changed()
        return tab

    def _build_far_field_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        self._far_field_combo = QtWidgets.QComboBox()
        self._far_field_combo.currentIndexChanged.connect(lambda index: self._update_far_field_controls())
        self._far_frequency_combo = QtWidgets.QComboBox()
        self._far_frequency_combo.currentIndexChanged.connect(lambda index: self._draw_far_field())
        self._polar_plot = pg.PlotWidget()
        self._polar_plot.setAspectLocked(True)
        self._polar_plot.hideAxis('left')
        self._polar_plot.hideAxis('bottom')
        self._polar_plot.setTitle('Normalised far-field power (linear)')
        self._polar_plot.addLegend(offset=(10, 10))   # names the analytic patterns (grid lines are unnamed)
        self._polar_stale = False   # True: the polar plot waits for the tab to show (_draw_far_field)
        self._db_view = self._plot_view(
            self._far_field_plots, overlays=self._far_field_overlays, context=lambda name: self._far_field_key(),
            note=lambda name, xy: self._run_note(), export_stem=lambda name: f'{name}_far_field',
            extra_export_actions=lambda menu: self._add_raw_export_actions(
                menu, [key for key in (self._far_field_key(),) if key is not None]))
        layout.addWidget(self._row(self._far_field_combo, self._far_frequency_combo, None))
        layout.addWidget(self._splitter(self._polar_plot, self._db_view), 1)
        return tab

    def _build_dft_tab(self):
        tab = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(tab)
        self._dft_combo = QtWidgets.QComboBox()
        self._dft_combo.currentIndexChanged.connect(lambda index: self._update_dft_controls())
        self._dft_frequency_combo = QtWidgets.QComboBox()
        self._dft_intensity_combo = QtWidgets.QComboBox()
        self._dft_intensity_keys = []   # result keys of the intensity combo entries
        self._dft_show_button = QtWidgets.QPushButton('Show on canvas')
        self._dft_show_button.clicked.connect(lambda checked=False: self._show_dft())
        self._dft_note = QtWidgets.QLabel('Field intensity over the box at the chosen wavelength: electric field '
                                          'intensity, or for a magnetic field out of plane physics also the magnetic '
                                          'field intensity. Export it from the Field tab.')
        selectable(self._dft_note)
        self._dft_note.setToolTip(rich_tooltip('Electric field out of plane: |E_z|^2. Magnetic field out of plane: '
                                               '|E_x|^2 + |E_y|^2, or |H_z|^2. Fetched for one wavelength on demand.'))
        self._dft_note.setWordWrap(True)
        form.addRow('Monitor', self._dft_combo)
        form.addRow('Wavelength', self._dft_frequency_combo)   # the monitor's frequencies, listed as wavelengths
        form.addRow('Quantity', self._dft_intensity_combo)
        form.addRow('', self._dft_show_button)
        form.addRow(self._dft_note)
        return tab

    def _build_xy_table_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        self._xy_table_combo = QtWidgets.QComboBox()
        self._xy_table_combo.currentIndexChanged.connect(lambda index: self._update_xy_table_controls())
        self._table_x_combo = QtWidgets.QComboBox()   # tables: abscissa column of the plot ('' = row index)
        self._table_x_combo.setToolTip(rich_tooltip('Column plotted along the horizontal axis; the other number '
                                                    'columns are plotted against it'))
        self._table_x_combo.currentIndexChanged.connect(lambda index: self._update_xy_plot())
        self._table_x_columns = []   # parallel to _table_x_combo
        self._xy_view = self._plot_view(
            self._xy_plots, overlays=self._xy_overlays,
            context=lambda name: self._current_key(self._xy_table_combo, self._xy_table_keys),
            curve_options=lambda name, xy, series: POINT_OPTIONS if self._xy_is_table() else {},
            note=lambda name, xy: self._run_note(), export_stem=lambda name: name or 'curves',
            extra_export_actions=lambda menu: self._add_raw_export_actions(
                menu, [key for key in (self._current_key(self._xy_table_combo, self._xy_table_keys),)
                       if key is not None]),
            leading_widgets=(self._xy_table_combo, self._table_x_combo))
        self._xy_view.series_combo.setToolTip(rich_tooltip(
            'Plot every series of the chosen unit, or one of them alone.\n\n'
            'For a thermal noise spectrum over several materials, psd is the combined noise and psd_<material> the '
            'share of one material.'))
        self._result_table = QtWidgets.QTableWidget(0, 0)
        self._result_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._result_table_source = None   # the table result _result_table shows (refilled only when it changes)
        layout.addWidget(self._splitter(self._xy_view, self._result_table), 1)   # plot over the table (tables)
        return tab

    def _build_derived_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        self._derived_table = QtWidgets.QTableWidget(0, 7)
        self._derived_table.setHorizontalHeaderLabels(['Name', 'Expression', 'Operation', 'Value', 'Reference',
                                                       'Relative error', 'Tolerance'])
        self._derived_table.horizontalHeader().setStretchLastSection(True)
        self._derived_table.verticalHeader().hide()
        evaluate_button = QtWidgets.QPushButton('Evaluate')
        evaluate_button.clicked.connect(lambda checked=False: self._evaluate_derived())
        export_button = QtWidgets.QPushButton('Export table…')
        export_button.setToolTip(rich_tooltip('Save the table (values of the last Evaluate, references, relative '
                                              'errors and tolerances) with the model title and the parameter values '
                                              'of the run, as CSV by default.'))
        export_button.clicked.connect(lambda checked=False: self._export_derived_table())
        layout.addWidget(self._derived_table, 1)
        layout.addWidget(self._row(None, evaluate_button, export_button, None))
        self._derived_series_view = self._plot_view(
            lambda: self._derived_series_plots, curve_options=lambda name, xy, series: POINT_OPTIONS,
            note=lambda name, xy: self._run_note('Derived values (without a source) at every stored time or '
                                                 'wavelength of the last run, on its live state'),
            export_stem=lambda name: 'derived_series', empty_text='run a time-dependent or frequency-domain study')
        self._derived_series_view.hide()
        layout.addWidget(self._derived_series_view, 1)
        return tab

    def _build_derived_curves_tab(self):
        self._derived_view = self._plot_view(
            lambda: [(plot, xy) for plot, xy, _errors in self._derived_curve_plots],
            curve_options=self._derived_curve_options, note=self._derived_curve_note,
            export_stem=lambda name: name or 'derived_curves', empty_text='run the study first')
        self._derived_view.plot_combo.setToolTip('Plot shown: derived curves naming the same plot are drawn together')
        return self._derived_view

    def _build_sweep_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        self._sweep_info = selectable(QtWidgets.QLabel('No sweep in the last study.'))
        self._sweep_view = self._plot_view(
            lambda: self._sweep_plots, overlays=self._sweep_overlays, curve_options=self._sweep_curve_options,
            note=lambda name, xy: self._run_note(), export_stem=lambda name: name or 'sweep',
            empty_text='run a sweep first')
        self._sweep_view.plot_combo.setToolTip('Quantity plotted against the swept parameter')
        layout.addWidget(self._sweep_info)
        layout.addWidget(self._sweep_view, 1)
        return tab

    def _build_sectors_tab(self):
        """Parity sectors of the last study (summary parity_sectors): one row per sector run (status, its snapshots,
        its derived values) and the combined row."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        self._sectors_info = selectable(QtWidgets.QLabel('No parity sectors in the last study.'))
        self._sectors_info.setWordWrap(True)
        self._sectors_table = read_only_table((), (
            'Each row is one run of the drawn part with the mirror planes set to one combination of even and odd. '
            'The combined row is what the drawn sources give alone on the whole structure: powers, fluxes and '
            'absorbed powers are averaged over the runs and divided by their number, values linear in the field '
            '(a real or imaginary part) are averaged, and values without a valid combination (maximum, minimum, '
            'phase) show NaN.\n\n'
            'Mode and resonance studies list every run separately; nothing is combined.'))
        layout.addWidget(self._sectors_info)
        layout.addWidget(self._sectors_table, 1)
        return tab

    def _update_sectors_tab(self):
        """Fill the Parity sectors table from the study summary (empty without parity sectors)."""
        summary = self._study_summary or {}
        sectors = summary.get('parity_sectors')
        table = self._sectors_table
        table.setRowCount(0)
        if not sectors:
            self._sectors_info.setText('No parity sectors in the last study.')
            return
        names = list(sectors.get('derived') or {})
        table.setColumnCount(3 + len(names))
        table.setHorizontalHeaderLabels(['Sector', 'Status', 'Snapshots', *names])
        snapshot_sectors = list(sectors.get('snapshot_sectors') or [])
        labels = list(sectors.get('labels') or [])
        rows = [[label, status, str(snapshot_sectors.count(label)),
                 *(_sector_number(sectors['derived'][name], index) for name in names)]
                for index, (label, status) in enumerate(zip(labels, sectors.get('status') or []))]
        if sectors.get('combined'):
            derived = summary.get('derived') or {}
            rows.append(['combined', summary.get('status', ''), str(snapshot_sectors.count('')),
                         *(_sector_number([derived.get(name, math.nan)], 0) for name in names)])
        fill_table(table, rows)
        sweep = summary.get('sweep')
        sweep_text = (f"; sweep: the last value ({sweep['parameter']} = {float(sweep['values'][-1]):.6g}), every "
                      'value in the Sweep tab') if sweep and sweep.get('values') else ''
        self._sectors_info.setText(f"{len(labels)} sector runs over the mirror faces "
                                   f"{', '.join(sectors.get('faces') or [])}"
                                   f"{'' if sectors.get('combined') else ' (not combined)'}{sweep_text}")

    # ---- convergence tab (mesh refinement levels, S7.7) ----
    def _build_convergence_tab(self):
        """Mesh refinement of the last study (summary 'convergence'): stop message, one table row per level, the
        Richardson estimate, value / error plots versus the cell count and 'Keep adapted mesh'."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        self._convergence_info = selectable(QtWidgets.QLabel(_NO_CONVERGENCE))
        self._convergence_info.setWordWrap(True)
        self._convergence_info.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self._convergence_table = read_only_table((), _CONVERGENCE_TABLE_TIP)
        self._convergence_view = self._plot_view(
            self._convergence_plots, curve_options=self._convergence_curve_options,
            note=lambda name, xy: self._run_note(), export_stem=lambda name: file_stem(name, 'convergence'),
            empty_text='run a study with mesh refinement first')
        self._convergence_view.plot_combo.setToolTip('The watched derived value, or the relative error estimates, '
                                                     'versus the number of cells')
        self._richardson_label = selectable(QtWidgets.QLabel())
        self._richardson_label.setWordWrap(True)
        self._richardson_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self._keep_mesh_button = QtWidgets.QPushButton('Keep adapted mesh')
        self._keep_mesh_button.setToolTip(rich_tooltip(_KEEP_MESH_TIP))
        self._keep_mesh_button.setEnabled(False)
        self._keep_mesh_button.clicked.connect(lambda checked=False: self._keep_adapted_meshes())
        layout.addWidget(self._convergence_info)
        layout.addWidget(self._splitter(self._convergence_table, self._convergence_view), 1)
        layout.addWidget(self._richardson_label)
        layout.addWidget(self._row(self._keep_mesh_button, None))
        return tab

    def _convergence_report(self):
        """summary 'convergence' of the last study (mesh refinement levels), None without one."""
        return (self._study_summary or {}).get('convergence') or None

    def _update_convergence_tab(self):
        """Fill the Convergence tab and the level label from the study summary and hand the canvas the meshes of the
        last level ('Show mesh'); empty without mesh refinement."""
        report = self._convergence_report()
        levels = list((report or {}).get('levels') or [])
        last_level = int(levels[-1].get('level') or 0) if levels else 0
        table = self._convergence_table
        table.setRowCount(0)
        table.setColumnCount(0)
        self._keep_mesh_button.setEnabled(bool(report and last_level > 0 and report.get('adapted_meshes')))
        self._update_level_label(report, levels)
        self._canvas.set_results_meshes(*self._adapted_mesh_estimates(report, last_level))
        self._convergence_view.request_redraw()
        if report is None:
            self._convergence_info.setText(_NO_CONVERGENCE)
            self._convergence_info.setStyleSheet('')
            self._richardson_label.setText('')
            return
        target = str(report.get('target', ''))
        refined = set(report.get('meshes') or [])
        mesh_names = list(dict.fromkeys(name for record in levels for name in (record.get('cells') or {})))
        other_names = list(dict.fromkeys(name for record in levels for name in (record.get('derived') or {})
                                         if name != target))
        headers = ['Level', *(f"Mesh '{name}' {self._cells_word(name).lower()}" + ('' if name in refined else
                                                                                    ' (fixed)')
                              for name in mesh_names),
                   target, 'Relative change', *other_names, 'Study time', 'Status']
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        rows = []
        for row, record in enumerate(levels):
            cells, derived = record.get('cells') or {}, record.get('derived') or {}
            rows.append([str(record.get('level', row)),
                         *(f'{int(cells[name]):,}' if cells.get(name) is not None else '' for name in mesh_names),
                         table_number(record.get('value'), 10), table_number(record.get('relative_change'), 3),
                         *(table_number(derived.get(name)) for name in other_names),
                         format_duration(record.get('study_seconds')), str(record.get('status', ''))])
        fill_table(table, rows)
        method = {'adaptive': 'Adaptive', 'uniform': 'Uniform'}.get(report.get('method'), str(report.get('method')))
        meshes_text = ', '.join(f"'{name}'" for name in report.get('meshes') or [])
        self._convergence_info.setText(
            f"{method} refinement of mesh{'es' if len(refined) > 1 else ''} {meshes_text} watching {target} "
            f"(tolerance {table_number(report.get('tolerance'), 3)}): {report.get('message', '')}")
        self._convergence_info.setStyleSheet(f"color:{'#205020' if report.get('converged') else '#9a5b00'};")
        fit = report.get('richardson')
        if fit and fit.get('order_cells') is not None:
            fit_levels = list(fit.get('levels') or [])
            levels_text = f' over levels {fit_levels[0]}–{fit_levels[-1]}' if fit_levels else ''
            text = (f"Richardson extrapolation{levels_text}: {fit['extrapolated']:.10g} ± {fit['error_estimate']:.3g} "
                    f"(relative {table_number(fit.get('relative_error_estimate'), 3)}); observed order "
                    f"{table_number(fit.get('order'), 3)} in the cell size ({fit['order_cells']:.3g} in the cell "
                    'count).')
        elif fit:   # the last change was zero
            text = f"Richardson extrapolation: the value stopped changing at {fit['extrapolated']:.10g}."
        else:
            text = f"No Richardson extrapolation: {report.get('richardson_note') or 'it needs three levels'}."
        self._richardson_label.setText(text)

    def _update_level_label(self, report, levels):
        """Label above the tabs naming the mesh refinement level (and its meshes) the results come from; hidden
        without mesh refinement."""
        if not report or not levels:
            self._level_label.hide()
            return
        last = levels[-1]
        cells = '; '.join(f"mesh '{name}' {int(count):,} {self._cells_word(name).lower()}"
                          for name, count in (last.get('cells') or {}).items() if count is not None)
        stop_reason = report.get('stop_reason', '')
        state = ('converged' if report.get('converged') else
                 f"not converged ({_STOP_REASON_TEXTS.get(stop_reason, stop_reason)})")
        if (self._study_summary or {}).get('session_live', True):
            shown = f"the results shown are those of level {last.get('level')}"
        else:
            shown = (f"level {last.get('level')} was the last one computed, but its fields are gone (a later level "
                     'failed)')
        self._level_label.setText(f'Mesh refinement {state}: {shown}, on {cells}. Every level: Convergence tab.')
        self._level_label.show()

    def _cells_word(self, mesh_name):
        """'Elements' for a triangle mesh of the results' spec, else 'Cells'."""
        spec = self._bridge.results_spec or self._document.spec
        try:
            kind = spec.mesh_by_name(mesh_name).kind
        except KeyError:
            kind = 'tensor'
        return 'Cells' if kind == 'tensor' else 'Elements'

    def _adapted_mesh_estimates(self, report, last_level):
        """({mesh name: estimate}, level text) of the refined meshes the results were computed on when they are not
        the model's (last level > 0), for the canvas 'Show mesh'; ({}, '') otherwise or when not estimable."""
        spec = self._bridge.results_spec
        if not report or last_level <= 0 or spec is None:
            return {}, ''
        try:
            adapted = ModelSpec.from_dict({'meshes': copy.deepcopy(list(report.get('adapted_meshes') or []))}).meshes
            by_name = {mesh_spec.name: mesh_spec for mesh_spec in adapted}
            level_spec = spec.copy()
            level_spec.meshes = [by_name.get(mesh_spec.name, mesh_spec) for mesh_spec in level_spec.meshes]
            parameters = evaluate_parameters(level_spec.parameters)
            estimates = {name: estimate_mesh(level_spec, parameters, self._document.registry, name)
                         for name in by_name}
        except MESH_ESTIMATE_ERRORS:
            return {}, ''
        return estimates, f'mesh refinement level {last_level}'

    def _convergence_plots(self):
        """[(name, xy)] of the Convergence view: the watched value (+ its Richardson extrapolation) and the relative
        error estimates (+ the tolerance) versus the cell count of the refined meshes, log axes."""
        report = self._convergence_report()
        levels = list((report or {}).get('levels') or [])
        if not levels:
            return []
        target = str(report.get('target', 'value'))
        cells = np.array([float(record.get('refined_cells') or record.get('n_cells') or np.nan) for record in levels])
        values = np.array([np.nan if record.get('value') is None else float(record['value']) for record in levels])
        refined = list(report.get('meshes') or [])
        word = 'Elements' if refined and all(self._cells_word(name) == 'Elements' for name in refined) else 'Cells'
        x_label = f'{word} in the simulated region' + (' (refined meshes)' if len(refined) > 1 else '')
        fit = report.get('richardson') or {}
        extrapolated = float(fit['extrapolated']) if fit.get('order_cells') is not None else None
        value_curves = [(target, cells, values, '')]
        if extrapolated is not None:
            value_curves.append(('Richardson extrapolation', cells, np.full(cells.size, extrapolated), ''))
        changes = np.array([np.nan if record.get('relative_change') is None else float(record['relative_change'])
                            for record in levels])
        error_curves = [('relative change from the previous level', cells, changes)]
        if extrapolated:   # set and nonzero
            error_curves.append(('relative distance to the extrapolated value', cells,
                                 np.abs(values - extrapolated) / abs(extrapolated)))
        error_curves.append(('tolerance', cells, np.full(cells.size, float(report.get('tolerance') or np.nan))))
        with np.errstate(invalid='ignore'):   # log axis: zero (no change) and NaN are left out
            error_curves = [(name, x, np.where(y > 0.0, y, np.nan), '1') for name, x, y in error_curves]
        return [(f'{target} versus mesh size', curve_xy(value_curves, x_label, '1', target, log_x=True)),
                ('Relative error estimate', curve_xy(error_curves, x_label, '1', 'relative error', log_x=True,
                                                     log_y=True))]

    @staticmethod
    def _convergence_curve_options(name, xy, series):
        """Reference lines (Richardson extrapolation, tolerance) grey dotted; the level values with markers."""
        return {'pen': _REFERENCE_PEN} if series in _CONVERGENCE_REFERENCE_SERIES else POINT_OPTIONS

    def _keep_adapted_meshes(self):
        """'Keep adapted mesh': the last level's refined meshes become model meshes (document.keep_adapted_meshes)."""
        report = self._convergence_report()
        if report is None:
            return
        try:
            names = self._document.keep_adapted_meshes(report.get('adapted_meshes') or [])
        except USER_ERRORS as exc:
            self._show_message(f'Keep adapted mesh: {error_message(exc)}')
            return
        self._show_message(f"Adapted mesh kept in the model as {', '.join(names)}; the physics of the refined meshes "
                           'now run on it.', error=False)

    def _build_profile_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        self._profile_info = selectable(QtWidgets.QLabel(_PROFILE_HINT))
        self._profile_info.setWordWrap(True)
        self._profile_combo = QtWidgets.QComboBox()
        self._profile_combo.setToolTip(rich_tooltip('The line drawn on the field view, or the line of an analytic '
                                                    'curve (its field from the final state, with the analytic curve '
                                                    'dashed)'))
        self._profile_combo.currentIndexChanged.connect(self._on_profile_choice)
        draw_button = QtWidgets.QPushButton('Draw profile line')
        draw_button.clicked.connect(lambda checked=False: self._canvas.set_tool('profile'))
        self._profile_view = self._plot_view(
            self._profile_plots, overlays=self._profile_overlays,
            export_stem=lambda name: f"profile_{(self._profile or {}).get('curve') or 'drawn_line'}",
            extra_export_actions=lambda menu: menu.addAction('Raw profile…').triggered.connect(
                lambda checked=False: self._export_profile()),
            leading_widgets=(self._profile_combo, draw_button), empty_text='draw a profile line on a field first')
        layout.addWidget(self._profile_info)
        layout.addWidget(self._profile_view, 1)
        return tab

    @staticmethod
    def _row(*widgets):
        """Horizontal row of widgets; None inserts a stretch."""
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        for widget in widgets:
            if widget is None:
                layout.addStretch(1)
            else:
                layout.addWidget(widget)
        return row

    @staticmethod
    def _splitter(*widgets):
        """Vertical splitter of widgets, equal stretch."""
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        for widget in widgets:
            splitter.addWidget(widget)
        return splitter

    @staticmethod
    def _feature_list():
        widget = QtWidgets.QListWidget()
        widget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        widget.setMaximumHeight(90)
        widget.setToolTip('Select the results to plot. With none selected, all are plotted.')
        return widget

    # ---- list / combo helpers ----
    @staticmethod
    def _fill_list(widget, keys):
        widget.blockSignals(True)
        widget.clear()
        for key in keys:
            widget.addItem(_key_text(key))
        widget.blockSignals(False)

    @staticmethod
    def _fill_combo(combo, labels, select=None):
        """Refill a combo without firing signals; select = index to show (None: the previous one), the first item
        when out of range."""
        index = combo.currentIndex() if select is None else select
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(labels)
        if 0 <= index < len(labels):
            combo.setCurrentIndex(index)
        combo.blockSignals(False)

    @classmethod
    def _fill_key_combo(cls, combo, keys, current_key, leading_labels=()):
        """Refill a combo with leading_labels + 'physics.feature' of keys; current_key stays selected when listed."""
        cls._fill_combo(combo, list(leading_labels) + [_key_text(key) for key in keys],
                        select=len(leading_labels) + keys.index(current_key) if current_key in keys else 0)

    @staticmethod
    def _selected_keys(widget, keys):
        rows = sorted(widget.row(item) for item in widget.selectedItems())
        return [keys[row] for row in rows if row < len(keys)] or list(keys)

    @staticmethod
    def _current_key(combo, keys, offset=0):
        index = combo.currentIndex() - offset
        return keys[index] if 0 <= index < len(keys) else None

    def _current_result(self, combo, keys):
        """Fetched result (dict or error text) of the combo's current key; None before a fetch or with no key."""
        return self._results.get(self._current_key(combo, keys))

    def _selected_field(self):
        index = self._field_combo.currentIndex()
        return self._field_choices[index] if 0 <= index < len(self._field_choices) else None

    def _show_message(self, text, error=True):
        self._message_label.setStyleSheet(f"color:{'#a01010' if error else '#205020'};")
        self._message_label.setText(text)

    def _on_call_error(self, error):
        self._show_message(_error_text(error))

    def _current_only(self, handler):
        """Bridge-signal slot calling handler only for events of jobs submitted since the last document replacement."""
        return lambda *args: handler(*args) if self._bridge.event_is_current else None

    def _call(self, fn, on_result, on_error=None):
        """bridge.call whose result / error is dropped when the document was replaced meanwhile."""
        generation = self._bridge.generation

        def if_current(callback):
            return lambda value: callback(value) if self._bridge.generation == generation else None
        self._bridge.call(fn, on_result=if_current(on_result), on_error=if_current(on_error or self._on_call_error))

    # ---- curve views: shared ----
    def _redraw_views(self):
        for view in self._plot_views:
            view.request_redraw()

    def _run_note(self, text=''):
        """text, after _RUN_NOTE while a study runs (the plot then shows earlier results)."""
        return ' '.join(part for part in ((_RUN_NOTE if self._run_active else ''), text) if part)

    def _export_metadata(self):
        """(model title, parameters) of the session the results come from; None once the worker holds none (an
        export must not be stamped with the document's values)."""
        spec = self._bridge.results_spec
        if spec is None:
            return None
        return spec.title, dict(self._curve_parameters())

    def _add_raw_export_actions(self, menu, keys):
        """'Raw result: physics.feature…' actions writing the fetched result dict of each key (export_result)."""
        for key in keys:
            menu.addAction(f'Raw result: {_key_text(key)}…').triggered.connect(
                lambda checked=False, key=key: self._export_key(key))

    # ---- bridge signals ----
    def _on_built(self, summary):
        self._summary = summary
        built_spec = self._bridge.results_spec   # set by the bridge before it emits built
        self._built_derived = ({derived.name: _derived_definition(derived, _curves_by_name(built_spec))
                                for derived in built_spec.derived_values} if built_spec is not None else {})
        self._session_state_field = None
        self._fetch_session_parameters()
        self._populate_plane_combo(summary)
        self._populate_field_combo()
        self._fetch_mesh_cut_centres(show=False)   # 3D, several meshes: the shown physics' own cut positions
        self._reset_result_views(summary.feature_results)
        self._show_message(f'Built: {_built_cells_text(summary)}, {len(summary.fields)} fields, '
                           f'{len(summary.feature_results)} feature results', error=False)
        self._update_availability()
        if self._run_active and self._refinement_level is not None:   # the next mesh refinement level was built
            self._refinement_level += 1
            self._show_refinement_progress()

    def _show_refinement_progress(self):
        """'Mesh refinement level k of at most n: <cells>' while a mesh refinement run builds / runs level k (levels
        counted by Built: one build per level)."""
        spec = self._bridge.results_spec
        adaptation = spec.study.adaptation if spec is not None else None
        if self._refinement_level is None or adaptation is None or self._summary is None:
            return
        level = self._refinement_level
        self._sweep_label.setText(f"Mesh refinement level {level}{' (the model meshes)' if level == 0 else ''} of at "
                                  f'most {adaptation.max_levels}: {_built_cells_text(self._summary)}')
        self._sweep_label.show()

    def _fetch_session_parameters(self):
        """Parameters of the worker session (a swept model's Build uses the base value, not the last sweep value);
        a call refused because a sweep rebuilt meanwhile is ignored (StudyFinished fetches again)."""
        self._session_parameters = None

        def failed(error):   # other failures are shown: curves then use the built spec's parameters
            if 'session changed' not in str(error):
                self._show_message(f"Session parameters not fetched (curves use the built model's): "
                                   f'{_error_text(error)}')
        self._call(lambda session: {name: float(value) for name, value in session.parameters.items()},
                   self._on_session_parameters, on_error=failed)

    def _on_session_parameters(self, parameters):
        self._session_parameters = parameters
        self._schedule_derived_table()   # references, analytic curves and profiles follow the parameters
        self._update_all_result_views()
        if self._current_profile_curve() is not None:
            self._on_profile_choice()

    def _update_availability(self):
        if self.has_results != self._had_results:
            self._had_results = self.has_results
            self.availability_changed.emit(self._had_results)

    def _reset_result_views(self, feature_results):
        """Forget fetched / study results and refill the result lists for feature_results [(physics, feature, kind)]."""
        self._study_summary = None
        self._session_lost_reason = None
        self._results.clear()
        self._live_series.clear()
        self._fft_cache.clear()
        self._derived_values = {}
        self._feature_kinds = list(feature_results)
        by_kind = {}
        for physics, feature, result_kind in feature_results:
            by_kind.setdefault(result_kind, []).append((physics, feature))
        self._probe_keys = by_kind.get('timeseries', [])
        self._far_field_keys = by_kind.get('far_field', [])
        self._dft_keys = by_kind.get('dft_fields', [])
        self._fill_list(self._probe_list, self._probe_keys)
        self._fill_combo(self._far_field_combo, [_key_text(key) for key in self._far_field_keys])
        self._fill_combo(self._dft_combo, [_key_text(key) for key in self._dft_keys])
        self._sort_curve_results()
        self._update_sweep_value_choices()
        self._setup_snapshot_slider([])
        self._update_all_result_views()
        self._update_sweep_tab()
        self._update_derived_series()
        self._update_sectors_tab()
        self._update_convergence_tab()
        self._estimation_tab.show_report(estimation_report(self._study_summary))
        self._schedule_derived_table()
        self._schedule_derived_curves()
        self._update_out_of_date()
        self._update_tab_visibility()

    def _sort_curve_results(self):
        """Spectra = flux spectra + xy results known to be versus wavelength; Curves & tables = tables + the other xy
        results. Refills only a list whose keys changed (selection / reference kept otherwise)."""
        spectrum_keys, xy_table_keys = [], []
        for physics, feature, result_kind in self._feature_kinds:
            key = (physics, feature)
            if result_kind == 'flux_spectrum' or (result_kind == 'xy' and key in self._wavelength_xy_keys):
                spectrum_keys.append(key)
            elif result_kind in ('xy', 'table'):
                xy_table_keys.append(key)
        if spectrum_keys != self._spectrum_keys:
            reference_key = self._current_key(self._reference_combo, self._spectrum_keys, offset=1)
            self._spectrum_keys = spectrum_keys
            self._fill_list(self._spectrum_list, spectrum_keys)
            self._fill_key_combo(self._reference_combo, spectrum_keys, reference_key, ('(none)',))
        if xy_table_keys != self._xy_table_keys:
            current_key = self._current_key(self._xy_table_combo, self._xy_table_keys)
            self._xy_table_keys = xy_table_keys
            self._fill_key_combo(self._xy_table_combo, xy_table_keys, current_key)

    def _update_tab_visibility(self):
        """Show a result tab only when the build / study / model can fill it (Field and Line profile always)."""
        sweep = self._current_sweep()
        sectors = (self._study_summary or {}).get('parity_sectors')
        for tab, content in ((self._probes_tab, self._probe_keys), (self._spectra_tab, self._spectrum_keys),
                             (self._far_field_tab, self._far_field_keys), (self._dft_tab, self._dft_keys),
                             (self._xy_table_tab, self._xy_table_keys),
                             (self._derived_tab, self._document.spec.derived_values),
                             (self._derived_curves_tab, self._document.spec.derived_curves), (self._sweep_tab, sweep),
                             (self._sectors_tab, sectors), (self._convergence_tab, self._convergence_report()),
                             (self._estimation_tab, estimation_report(self._study_summary))):
            self._tabs.setTabVisible(self._tabs.indexOf(tab), bool(content))

    def _on_build_failed(self, failed):
        self._summary = None
        self._session_parameters = None
        self._field_choices, self._field_physics = [], []
        self._fill_combo(self._field_physics_combo, [])
        self._fill_combo(self._field_combo, [])
        self._session_lost_reason = _BUILD_FAILED_REASON
        self._set_run_active(False)
        self._schedule_derived_curves()
        self._update_out_of_date()
        self._show_message(f'Build failed: {_error_text(failed)}')
        self._update_availability()
        self.field_view_changed.emit(True)

    def _set_run_active(self, active):
        """Running state of the curve views' notes; they redraw when it changes."""
        if active != self._run_active:
            self._run_active = active
            self._redraw_views()

    def _on_follow_live_clicked(self, checked):
        self._follow_paused_by_view = False   # the user's own choice persists across runs

    def _pause_follow_live(self):
        """Untick 'Follow live' while a snapshot / DFT field is viewed; the next run re-ticks it."""
        if self._follow_live.isChecked():
            self._follow_live.setChecked(False)
            self._follow_paused_by_view = True

    def _on_study_started(self):
        self._stop_replay()
        self._session_state_field = None
        if self._follow_paused_by_view:   # else the canvas would keep the previous run's field
            self._follow_live.setChecked(True)
            self._follow_paused_by_view = False
        self._show_next_live_frame = self._follow_live.isChecked()
        self._set_run_active(True)
        self._show_message('Study running…', error=False)
        spec = self._bridge.results_spec   # mesh refinement: StudyStarted follows the Built of level 0
        self._refinement_level = 0 if spec is not None and spec.study.adaptation is not None else None
        self._show_refinement_progress()

    def _on_live_frame(self, field_data):
        if not self._follow_live.isChecked():
            return
        if (field_data.physics, field_data.component) != self._selected_field():
            return                     # frame of the previous live-field choice
        if self._canvas.mode == 'field' or self._show_next_live_frame:
            self._show_next_live_frame = False
            self._show_field(field_data)

    def _on_live_results(self, results):
        self._live_series.update(results)
        self._probe_view.request_redraw(live=True)   # throttled; only while the Probes tab is shown
        self._fft_view.request_redraw(live=True)

    def _on_study_finished(self, finished):
        summary = finished.summary
        self._study_summary = summary
        self._session_state_field = None   # a field shown mid-run is not the final state
        self._sweep_label.hide()       # shown per sweep value / refinement level (each rebuilds: not hidden on built)
        self._canvas.set_sweep_value()
        self._refinement_level = None
        self._derived_values = {name: (self._built_derived.get(name), value, None)   # the run spec's definitions
                                for name, value in (summary.get('derived') or {}).items()}
        self._live_sweep = None        # the finished table replaces the live one
        self._update_sweep_value_choices()
        self._setup_snapshot_slider(summary.get('snapshot_times') or [], summary.get('snapshot_labels') or [],
                                    summary.get('live_snapshot'))
        self._schedule_derived_table()
        self._update_sweep_tab()
        self._update_derived_series()
        self._update_sectors_tab()
        self._update_convergence_tab()
        self._estimation_tab.show_report(estimation_report(summary))
        self._update_tab_visibility()
        self._update_availability()
        session_live = summary.get('session_live', True)
        lost = ('the last mesh refinement level failed, only the convergence table is available'
                if summary.get('convergence') else 'the last sweep value failed, only the sweep table is available')
        if not session_live:
            self._session_lost_reason = lost
        self._set_run_active(False)
        self._schedule_derived_curves()
        self._update_out_of_date()
        self._show_message(study_summary_text(summary, format_duration(summary.get('wall_time')))
                           + ('' if session_live else f'; {lost}'), error=not session_live)
        if not session_live:
            self._session_parameters = None
            return                     # no session to fetch from: refresh / show_selected_field would only fail
        self._fetch_session_parameters()   # a sweep's session is its last value's
        self.refresh()
        if self._follow_live.isChecked():
            self.show_selected_field()
        else:   # the 3D preview's cut planes show the final state either way
            self.field_view_changed.emit(True)
        if self._current_profile_curve() is not None:   # the final state of the new run
            self._on_profile_choice()

    def _on_study_failed(self, message, traceback_text):
        self._session_state_field = None
        self._sweep_label.hide()
        self._canvas.set_sweep_value()
        self._refinement_level = None
        if self._live_sweep is not None:   # a failed sweep leaves no table to show
            self._live_sweep = None
            self._update_sweep_tab()
            self._update_tab_visibility()
        # a Compute whose build failed (BuildFailed came first): keep that banner and message
        build_failure = self._bridge.results_spec is None and self._session_lost_reason == _BUILD_FAILED_REASON
        if self._bridge.results_spec is None:   # cleared by the bridge before it emits: e.g. a GPU OOM closed it
            if not build_failure:
                self._session_lost_reason = 'the last run failed and closed its session, build again'
            self._session_parameters = None
        self._set_run_active(False)
        self._schedule_derived_curves()
        self._update_out_of_date()
        if not build_failure:
            self._show_message(f'Study failed: {message}')

    def _on_sweep_step(self, step):
        self._sweep_label.setText(f'Sweep {step.index + 1}/{step.count}: {step.parameter} = {float(step.value):.6g}')
        self._sweep_label.show()
        self._canvas.set_sweep_value(step.parameter, step.value)   # the geometry of this value until the study ends
        if step.index == 0 and self._live_sweep is not None:   # a new sweep: drop the previous run's partial table
            self._live_sweep = None
            self._update_sweep_tab()

    def _on_sweep_value_done(self, done):
        """Fill the Sweep tab with the values computed so far while the sweep runs."""
        self._live_sweep = done.sweep
        self._update_sweep_tab()
        self._update_tab_visibility()

    def _current_sweep(self):
        """Sweep table of the last study, else the one of the sweep still running, else None."""
        return (self._study_summary or {}).get('sweep') or self._live_sweep

    def _on_feature_results(self, results):
        errors = []
        for key, value in results.items():
            if isinstance(value, dict):
                self._results[key] = value
                if is_wavelength_xy(value):
                    self._wavelength_xy_keys.add(key)
                else:
                    self._wavelength_xy_keys.discard(key)
            else:
                errors.append(f'{_key_text(key)}: {value}')
        if errors:
            self._show_message('Some results failed: ' + '; '.join(errors))
        self._sort_curve_results()
        self._update_tab_visibility()
        self._update_all_result_views()

    def _update_all_result_views(self):
        """Refill the result-dependent controls, then redraw every curve view (coalesced; hidden views when shown)."""
        self._update_far_field_controls()
        self._update_dft_controls()
        self._update_xy_table_controls()
        self._redraw_views()

    # ---- document ----
    def _on_document_replaced(self):
        """New / opened model: forget every result of the old one (its worker session lives until the next build) and
        ignore its still running job and pending calls."""
        self._bridge.advance_generation()
        self._summary = None
        self._session_parameters = None
        self._field_choices, self._field_physics = [], []
        self._field_infos = {}
        self._fill_combo(self._field_physics_combo, [])
        self._fill_combo(self._field_combo, [])
        self._wavelength_xy_keys.clear()
        self._reset_result_views(())
        self._profile = None
        self._profile_info.setText(_PROFILE_HINT)
        self._profile_view.set_error('')
        self._profile_combo.blockSignals(True)
        self._profile_combo.setCurrentIndex(0)   # changed('all') follows and refills it
        self._profile_combo.blockSignals(False)
        self._sweep_label.hide()
        self._refinement_level = None
        self._live_sweep = None
        self._update_sweep_tab()
        self._set_run_active(False)
        self._show_message('', error=False)
        self._update_availability()
        self.field_view_changed.emit(True)

    def _on_document_changed(self, section, origin):
        self._update_out_of_date()
        if section in _DERIVED_TABLE_SECTIONS:   # rows: definitions, evaluation meshes, parameters (references)
            self._schedule_derived_table()
        if section in ('all', 'results'):   # analytic / derived curve definitions: redraw the views showing them
            self._update_profile_choices()
            self._update_all_result_views()
            self._schedule_derived_curves()

    def _update_out_of_date(self):
        # old plots / field stay visible after a failed build (results_spec None), so they count as results;
        # compared by content: a derived-value edit or an undo back to the built model is not stale
        shows_results = (self._summary is not None or bool(self._results) or bool(self._live_series)
                         or self._canvas.current_field is not None)
        built_spec = self._bridge.results_spec
        if built_spec is not None and self._built_inputs[0] is not built_spec:   # serialised once per build
            self._built_inputs = (built_spec, results_inputs(built_spec.to_dict()))
        stale = built_spec is None or self._built_inputs[1] != results_inputs(self._document.spec_dict())
        cause = (self._session_lost_reason if built_spec is None and self._session_lost_reason
                 else 'the model changed after the last build / run')
        self._stale_label.setText(f'Results out of date: {cause}.')
        self._stale_label.setVisible(shows_results and stale)

    # ---- field tab ----
    def _populate_field_combo(self):
        """Physics combo of the built fields (the previous choice kept, else the physics of the first primary field),
        then its field list."""
        previous = self._selected_field()
        self._field_infos = {(physics, info.name): info for physics, info in self._summary.fields}
        self._field_physics = list(dict.fromkeys(physics for physics, _info in self._summary.fields))
        if previous is not None and previous[0] in self._field_physics:
            physics = previous[0]
        else:
            physics = next((name for name, info in self._summary.fields if info.primary),
                           self._field_physics[0] if self._field_physics else None)
        self._fill_combo(self._field_physics_combo, [self._physics_text(name) for name in self._field_physics],
                         select=self._field_physics.index(physics) if physics in self._field_physics else 0)
        self._fill_field_list(previous)

    def _physics_text(self, name):
        """'name (type label)' of a physics of the built model (the name alone when its type is unknown), plus
        " – mesh 'm'" when the model has several meshes."""
        spec = self._bridge.results_spec or self._document.spec
        mesh_name = physics_mesh_name(spec, name) if several_meshes(spec) else ''
        mesh_text = f" – mesh '{mesh_name}'" if mesh_name else ''
        try:
            label = self._document.registry.physics_type(spec.physics_by_name(name).type).label
        except KeyError:
            return f'{name}{mesh_text}'
        return f'{name} ({label}){mesh_text}'

    def _fill_field_list(self, previous=None):
        """Field combo of the chosen physics (previous kept when listed, else its primary field), then apply it."""
        index = self._field_physics_combo.currentIndex()
        physics = self._field_physics[index] if 0 <= index < len(self._field_physics) else None
        fields = [(name, info) for name, info in self._summary.fields if name == physics]
        self._field_choices = [(name, info.name) for name, info in fields]
        spec = self._bridge.results_spec or self._document.spec
        mesh_name = physics_mesh_name(spec, physics) if physics is not None and several_meshes(spec) else ''
        self._field_combo.setToolTip(f"Fields of {physics} on its mesh '{mesh_name}'" if mesh_name else '')
        labels = [info.label + (f' [{info.unit}]' if info.unit and info.unit != '1' else '') for _name, info in fields]
        if previous in self._field_choices:
            select = self._field_choices.index(previous)
        else:
            select = next((position for position, (_name, info) in enumerate(fields) if info.primary), 0)
        self._fill_combo(self._field_combo, labels, select=select)
        self._apply_field_choice()

    def _on_field_physics_selected(self):
        self._fill_field_list()
        self._update_snapshot_label()
        if not self._fetch_mesh_cut_centres():
            self.show_selected_field()

    def _fetch_mesh_cut_centres(self, show=True) -> bool:
        """3D model with several meshes: fetch the cut positions of the chosen physics' mesh (Session.cut_centers),
        refill the cut selector and (show) show the field; False (nothing fetched) otherwise."""
        spec = self._bridge.results_spec
        selection = self._selected_field()
        if spec is None or not spec.is_3d or not several_meshes(spec) or selection is None or self._bridge.busy:
            return False
        physics = selection[0]
        self._call(lambda session, physics=physics: {axis: list(session.cut_centers(axis, physics)) for axis in 'xyz'},
                   lambda centres, show=show: self._on_mesh_cut_centres(centres, show))
        return True

    def _on_mesh_cut_centres(self, centres, show):
        """Cut positions of the shown physics' mesh: refill the position combo (the axis kept), then (show) show the
        field."""
        self._plane_centres = centres
        self._fill_plane_positions()
        if show and self._summary is not None and self._selected_field() is not None:
            self.show_selected_field()

    def _apply_field_choice(self):
        """Symmetric default and live field of the selected component (no fetch)."""
        selection = self._selected_field()
        if selection is None:
            return
        info = self._field_infos.get(selection)
        self._symmetric_override = None
        self._set_symmetric_check(info is not None and info.signed)
        self._canvas.set_symmetric(None)
        self._bridge.set_live_field(*selection)

    def _set_symmetric_check(self, flag):
        self._symmetric_check.blockSignals(True)
        self._symmetric_check.setChecked(bool(flag))
        self._symmetric_check.blockSignals(False)

    def _on_symmetric_toggled(self, checked):
        self._symmetric_override = bool(checked)
        self._canvas.set_symmetric(checked)
        self.field_view_changed.emit(False)

    def _show_field(self, field_data, color_range=None, snapshot=False):
        """Show a FieldData on the canvas (color_range: fixed colour limits of a replay; snapshot: it is a snapshot's,
        so a cut-plane change refetches the snapshot), unfolded over the mirror faces when that box is ticked; the
        symmetric box follows its FieldInfo.signed until the user sets it."""
        self._shown_field = field_data   # before unfolding: replay identity checks and the unfold toggle
        self._snapshot_shown = snapshot
        if self._unfold_check.isChecked():
            field_data = self._unfolded(field_data, snapshot)
        self._canvas.show_field(field_data, color_range)
        if self._symmetric_override is None:
            self._set_symmetric_check(field_data.info.signed)

    def _unfolded(self, field_data, snapshot):
        """field_data unfolded over the mirror faces (as it is when that fails, with a message); a replay frame's
        unfolded copy is kept for the frames' lifetime (each tick would mirror the whole array again)."""
        replay = snapshot and self._replay_frames is not None and any(frame is field_data
                                                                      for frame in self._replay_frames)
        cached = self._unfolded_frames.get(id(field_data)) if replay else None
        if cached is not None and cached[0] is field_data:
            return cached[1]
        spec = self._bridge.results_spec or self._document.spec
        try:
            unfolded = unfold_field(field_data, mirror_faces(spec, field_data.physics))
        except ValueError as exc:
            self._show_message(f'Not unfolded: {exc}')
            return field_data
        if replay:
            self._unfolded_frames[id(field_data)] = (field_data, unfolded)
        return unfolded

    def _on_unfold_toggled(self, checked):
        if self._shown_field is not None:
            self._show_field(self._shown_field, snapshot=self._snapshot_shown)

    def _show_session_field(self, field_data):
        """_show_field of the current session state (session.field), remembered so export can refetch native cells."""
        self._session_state_field = field_data
        self._show_field(field_data)
        self.field_view_changed.emit(True)

    def _on_field_selected(self, index):
        self._apply_field_choice()
        self._update_snapshot_label()
        self.show_selected_field()

    def _on_colormap_selected(self, index):
        try:
            self._canvas.set_colormap(self._colormap_combo.itemData(index))
        except ValueError as exc:
            self._show_message(str(exc))
        self.field_view_changed.emit(False)

    def _setup_snapshot_slider(self, times, labels=(), live=None):
        """Slider over the snapshots, set at `live` (index of the snapshot holding the live state, e.g. resonance 1),
        else the last."""
        self._stop_replay()
        self._replay_frames = self._replay_key = self._replay_range = None
        self._unfolded_frames.clear()
        self._replay_token += 1
        self._snapshot_times = [float(time) for time in times]
        self._snapshot_labels = list(labels)
        last = max(0, len(self._snapshot_times) - 1)
        self._snapshot_slider.blockSignals(True)
        self._snapshot_slider.setRange(0, last)
        self._snapshot_slider.setValue(live if live is not None and 0 <= live <= last else last)
        self._snapshot_slider.setEnabled(bool(self._snapshot_times))
        self._snapshot_slider.blockSignals(False)
        self._update_snapshot_label()
        self._update_replay_rate()

    def _update_snapshot_label(self):
        if not self._snapshot_times:
            self._snapshot_label.setText('No snapshots')
            return
        index = self._snapshot_slider.value()
        stored = (self._study_summary or {}).get('stored_fields') or []
        selection = self._selected_field()
        note = '' if selection is None or f'{selection[0]}.{selection[1]}' in stored else ' (field not stored)'
        if not note and self._selected_plane() is not None and self._snapshot_plane() is None:
            note = ' (this vertical cut is not stored: add it to the stored cut planes)'
        label = self._snapshot_labels[index] if index < len(self._snapshot_labels) else None
        moment = display_label(label) or f't = {_time_text(self._snapshot_times[index])}'
        self._snapshot_label.setText(f'{index + 1}/{len(self._snapshot_times)}: {moment}{note}')

    def _on_snapshot_moved(self, value):
        self._update_snapshot_label()
        if self._selected_replay_frames() is not None:   # cached by Play: shown at once, no worker call
            self._snapshot_timer.stop()
            self._show_replay_frame(value)
        else:
            self._snapshot_timer.start()

    def _stored_snapshot_key(self):
        """'physics.component' of the selected field when snapshots store it, else None (message shown when stored
        fields exist but not this one)."""
        selection = self._selected_field()
        if not self._snapshot_times or selection is None:
            return None
        key = f'{selection[0]}.{selection[1]}'
        stored = (self._study_summary or {}).get('stored_fields') or []
        if key not in stored:
            self._show_message(f"{key} is not stored in snapshots (stored: {', '.join(stored) or 'none'}); "
                               f"add it to the study's stored fields.")
            return None
        return key

    def _snapshot_plane(self):
        """Chosen cut plane for snapshots (each shows its held cut on that axis nearest it, _snapshot_field); None in
        2D or for a vertical axis without a stored cut (summary stored_cuts)."""
        plane = self._selected_plane()
        if plane is None or plane[0] == 'z':
            return plane
        stored = (self._study_summary or {}).get('stored_cuts') or []
        return plane if any(axis == plane[0] for axis, _ in stored) else None

    def _unstored_cut_refused(self):
        """True (message shown) when the chosen vertical cut has no stored cut on its axis: the snapshot is not
        shown and the cut-plane combos go back to the shown field's cut (canvas and combos stay in step)."""
        plane = self._selected_plane()
        if plane is None or self._snapshot_plane() is not None:
            return False
        shown_plane = getattr(self._shown_field, 'plane', None)
        if shown_plane is not None and shown_plane[0] != plane[0]:
            self._set_plane_choice(shown_plane[0], float(shown_plane[1]))
            self._update_snapshot_label()
        self._show_message(f"Snapshots hold no {plane[0]} cut: add one to the study's stored cut planes and run "
                           f"again; the current results (live state) can be cut anywhere.")
        return True

    def _fetch_snapshot(self):
        key = self._stored_snapshot_key()
        if key is None or self._unstored_cut_refused():
            return
        self._pause_follow_live()
        index, plane = self._snapshot_slider.value(), self._snapshot_plane()
        kept = self._kept_snapshots()
        if kept is not None:   # host data of a kept sweep value: no worker call
            try:
                self._show_field(_snapshot_field(kept[index], key, plane), snapshot=True)
            except (KeyError, IndexError) as exc:
                self._show_message(f'This snapshot of the sweep value holds no {key}: {error_message(exc)}')
            return
        self._call(lambda session, index=index, key=key, plane=plane:
                   _snapshot_field(session.snapshots[index], key, plane),
                   lambda field_data: self._show_field(field_data, snapshot=True))

    # ---- kept sweep snapshots (SweepSpec.keep_fields) ----
    def _update_sweep_value_choices(self):
        """Field tab 'Sweep value' combo, shown when the last study kept the snapshots of every sweep value: entry 0 =
        the session (the last value), then each value with kept snapshots; resets the choice to the session."""
        sweep = (self._study_summary or {}).get('sweep') or {}
        kept = sweep.get('snapshots') or []
        labels, self._sweep_value_indices = ['Current results (last value)'], [None]
        for index, (value, snapshots) in enumerate(zip(sweep.get('values') or [], kept)):
            if snapshots:
                labels.append(f"{sweep['parameter']} = {float(value):.6g}")
                self._sweep_value_indices.append(index)
        self._sweep_field_value = None
        self._fill_combo(self._sweep_value_combo, labels, select=0)
        shown = len(labels) > 1
        self._sweep_value_label.setVisible(shown)
        self._sweep_value_combo.setVisible(shown)

    def _on_sweep_value_selected(self):
        """Point the snapshot slider at the chosen sweep value's kept snapshots (or back at the session's) and show
        the selected one."""
        index = self._sweep_value_combo.currentIndex()
        self._sweep_field_value = (self._sweep_value_indices[index]
                                   if 0 <= index < len(self._sweep_value_indices) else None)
        kept = self._kept_snapshots()
        if kept is None:
            summary = self._study_summary or {}
            self._setup_snapshot_slider(summary.get('snapshot_times') or [], summary.get('snapshot_labels') or [],
                                        summary.get('live_snapshot'))
        else:
            flagged = [index for index, snapshot in enumerate(kept) if snapshot.live]
            self._setup_snapshot_slider([snapshot.time for snapshot in kept], [snapshot.label for snapshot in kept],
                                        flagged[-1] if flagged else None)
        self._fetch_snapshot()

    def _kept_snapshots(self):
        """Kept Snapshots of the sweep value chosen in the Field tab, or None (the session's snapshots)."""
        if self._sweep_field_value is None:
            return None
        kept = ((self._study_summary or {}).get('sweep') or {}).get('snapshots') or []
        return kept[self._sweep_field_value] if self._sweep_field_value < len(kept) else None

    # ---- snapshot replay ----
    def _selected_replay_frames(self):
        """Cached replay frames of the selected field (on the chosen cut plane), or None."""
        selection = self._selected_field()
        if (self._replay_frames is None or selection is None
                or self._replay_key != (f'{selection[0]}.{selection[1]}', self._snapshot_plane())):
            return None
        return self._replay_frames

    def _show_replay_frame(self, index):
        self._pause_follow_live()
        color_range = self._replay_range if self._fixed_limits_check.isChecked() else None
        self._show_field(self._replay_frames[index], color_range=color_range, snapshot=True)

    def _on_fixed_limits_toggled(self, checked):
        """Redraw a shown replay frame with / without the fixed colour limits."""
        frames = self._selected_replay_frames()
        index = self._snapshot_slider.value()
        if frames is not None and index < len(frames) and self._shown_field is frames[index]:
            self._show_replay_frame(index)

    def _update_replay_rate(self):
        """'= <simulated time> per second' of the replay: median snapshot spacing x fps ('' without a time axis)."""
        spacings = np.diff(self._snapshot_times)
        spacings = spacings[spacings > 0.0]
        text = '' if not spacings.size else f'= {_time_text(float(np.median(spacings)) * self._fps_spin.value())} per s'
        self._replay_rate_label.setText(text)

    def _set_play_button(self, playing):
        self._play_button.blockSignals(True)
        self._play_button.setChecked(playing)
        self._play_button.setText('Pause' if playing else 'Play')
        self._play_button.blockSignals(False)

    def _stop_replay(self):
        self._play_timer.stop()
        self._set_play_button(False)

    def _on_play_toggled(self, checked):
        if not checked:
            self._stop_replay()
            return
        key = self._stored_snapshot_key()
        if key is None or self._unstored_cut_refused():
            if not self._snapshot_times:
                self._show_message('No snapshots to replay: run a study with an output interval first.')
            self._stop_replay()
            return
        self._set_play_button(True)
        if self._selected_replay_frames() is not None:
            self._begin_playback()
            return
        token, plane = self._replay_token, self._snapshot_plane()
        kept = self._kept_snapshots()
        if kept is not None:   # host frames of a kept sweep value
            try:
                frames = [_snapshot_field(snapshot, key, plane) for snapshot in kept]
            except (KeyError, IndexError) as exc:
                self._on_replay_error(error_message(exc))
                return
            self._on_replay_frames(frames, (key, plane), token)
            return
        self._show_message('Loading replay frames…', error=False)
        self._call(lambda session, key=key, plane=plane: [_snapshot_field(snapshot, key, plane)
                                                          for snapshot in session.snapshots],
                   lambda frames, key=key, plane=plane, token=token: self._on_replay_frames(frames, (key, plane), token),
                   on_error=self._on_replay_error)

    def _on_replay_frames(self, frames, replay_key, token):
        if token != self._replay_token:   # the snapshots changed meanwhile
            return
        self._replay_frames = frames
        self._replay_key = replay_key
        self._unfolded_frames.clear()
        self._replay_range = _frames_range(frames)
        self._show_message(f'Replay: {len(frames)} frames of {replay_key[0]}.', error=False)
        if self._play_button.isChecked():
            self._begin_playback()

    def _on_replay_error(self, error):
        self._stop_replay()
        self._show_message(f'Replay failed: {_error_text(error)}')

    def _begin_playback(self):
        """Show the current frame (the first after the end) and start the frame timer."""
        index = self._snapshot_slider.value()
        if index > 0 and index >= self._snapshot_slider.maximum():
            self._snapshot_slider.setValue(0)   # _on_snapshot_moved shows it
        else:
            self._show_replay_frame(index)
        self._play_timer.start(round(1000 / self._fps_spin.value()))

    def _advance_replay(self):
        if self._selected_replay_frames() is None:
            self._stop_replay()
            return
        next_index = self._snapshot_slider.value() + 1
        if next_index > self._snapshot_slider.maximum():
            if not self._loop_check.isChecked():
                self._stop_replay()
                return
            next_index = 0
        self._snapshot_slider.setValue(next_index)

    def _on_fps_changed(self, fps):
        self._update_replay_rate()
        if self._play_timer.isActive():
            self._play_timer.setInterval(round(1000 / fps))

    # ---- raw result export ----
    def _export_key(self, key):
        """export_result of one fetched result dict with the session's model title / parameters; a table defaults to
        .csv; a decimated live preview is refused."""
        result = self._results.get(key)
        if result is None:
            result = self._live_series.get(key)
        if not isinstance(result, dict):
            self._show_message('No result to export yet: compute first.')
            return
        if result.get('decimation_stride', 1) > 1:
            self._show_message(f'{_key_text(key)} is a live preview (min / max per display bin), not the result: '
                               'export after the run.')
            return
        stem = file_stem(_key_text(key))
        if result.get('kind') == 'table':
            self._write_result(result, f'{stem}.csv', _CSV_FIRST_FILE_TYPES)
        else:
            self._write_result(result, f'{stem}.npz')

    def _write_result(self, result, default_name, file_types=TABLE_FILE_TYPES):
        path = ask_save_path(self, self._document, 'Export result', default_name, file_types)
        if not path:
            return
        metadata = self._export_metadata()
        if metadata is not None:   # Compare data and readers tell runs apart by these
            result = dict(result, model=metadata[0], parameters=metadata[1])
        write_export(lambda target: export_result(result, target), path, self._report, 'Result')

    # ---- probes tab ----
    def _timeseries(self, key):
        result = self._results.get(key)
        if isinstance(result, dict) and result.get('kind') == 'timeseries':
            return result
        return self._live_series.get(key)

    def _selected_probe_keys(self):
        return self._selected_keys(self._probe_list, self._probe_keys)

    def _probes_decimated(self):
        """True while a selected probe shows a decimated live preview (its FFT would alias)."""
        return any((self._timeseries(key) or {}).get('decimation_stride', 1) > 1 for key in self._selected_probe_keys())

    def _probe_spectrum(self, key, result):
        """probe_spectrum of a probe series, cached per (key, result object, sample count, remove-mean)."""
        remove_mean = self._remove_mean_check.isChecked()
        sample_count = np.size(result['values'])
        cached = self._fft_cache.get(key)
        if cached is not None and cached[0] is result and cached[1:3] == (sample_count, remove_mean):
            return cached[3]
        spectrum = probe_spectrum(result['t'], result['values'], remove_mean)
        self._fft_cache[key] = (result, sample_count, remove_mean, spectrum)
        return spectrum

    def _update_probe_views(self):
        self._probe_view.request_redraw()
        self._fft_view.request_redraw()

    def _probe_plots(self):
        """Plot of the selected timed probes vs time and one of the stationary-iteration probes vs the 1-based index
        ('Probe iterations' beside timed ones); unit groups by units; preview while a series is a decimated preview."""
        timed, iterated, iteration_labels, decimated_groups = [], [], set(), set()
        for key in self._selected_probe_keys():
            result = self._timeseries(key)
            if result is None or np.size(result['t']) == 0:
                continue
            iteration_axis = _iteration_axis(result['t'])
            if iteration_axis is not None:   # stationary iterations, all at one t: vs index with symbols, no FFT
                iteration_labels.add(iteration_axis[1])
                group, curves, x = 'iterated', iterated, iteration_axis[0]
            else:
                group, curves, x = 'timed', timed, result['t']
            curves.append((_key_text(key), x, result['values'], result.get('unit', '')))
            if result.get('decimation_stride', 1) > 1:
                decimated_groups.add(group)
        plots = []
        if timed:
            plots.append(('Probe values', curve_xy(timed, 'Time', 's', 'Value', preview='timed' in decimated_groups)))
        if iterated:
            x_label = iteration_labels.pop() if len(iteration_labels) == 1 else 'Iteration'
            plots.append(('Probe iterations' if timed else 'Probe values',
                          curve_xy(iterated, x_label, '', 'Value', preview='iterated' in decimated_groups)))
        return plots

    def _probe_overlays(self, plot, drawn):
        for series, color in drawn.drawn:
            key = split_key(series)
            self._draw_feature_curves(plot, key, self._timeseries(key), color, report=self._probe_view.add_error)

    def _fft_plots(self):
        """One plot of the FFT magnitude of every selected timed, non-decimated probe (log y by default)."""
        curves = []
        for key in self._selected_probe_keys():
            result = self._timeseries(key)
            if (result is None or np.size(result['t']) == 0 or _iteration_axis(result['t']) is not None
                    or result.get('decimation_stride', 1) > 1):
                continue
            spectrum = self._probe_spectrum(key, result)
            if spectrum is not None:
                unit = f"{result.get('unit') or ''} s".strip()   # |FFT| dt: [probe unit s], units with spaces
                curves.append((_key_text(key), spectrum[0], spectrum[1], unit))
        return [('Probe spectra', curve_xy(curves, 'Frequency', 'Hz', 'FFT magnitude', log_y=True))] if curves else []

    def _fft_overlays(self, plot, drawn):
        for series, color in drawn.drawn:
            key = split_key(series)
            self._draw_feature_curves(plot, key, self._timeseries(key), color, FFT_SERIES, any_series=False,
                                      frequency=_series_x(drawn.xy, series), report=self._fft_view.add_error)

    # ---- spectra tab ----
    def _spectrum_quantity_names(self, keys):
        """Quantities of the fetched spectral results of keys, in first-seen order."""
        names = []
        for key in keys:
            result = self._results.get(key)
            if isinstance(result, dict):
                names += [name for name in _spectrum_quantities(result) if name not in names]
        return names

    def _update_spectra_views(self):
        self._spectrum_view.request_redraw()

    def _spectrum_shows_ratios(self):
        """True when the Spectra Show choice is 'Ratio to reference'."""
        return self._spectrum_show_combo.currentIndex() == 1

    def _on_spectrum_show_changed(self):
        """Reference choice shown only with ratios, then redraw."""
        ratios = self._spectrum_shows_ratios()
        self._reference_label.setVisible(ratios)
        self._reference_combo.setVisible(ratios)
        self._update_spectra_views()

    def _spectrum_plots(self):
        """One plot per quantity: that quantity of every selected spectral result having it vs wavelength (Show
        'Ratio to reference': _ratio_plots, under the same plot names)."""
        if self._spectrum_shows_ratios():
            return self._ratio_plots()
        keys = self._selected_keys(self._spectrum_list, self._spectrum_keys)
        plots = []
        for quantity in self._spectrum_quantity_names(keys):
            curves = []
            for key in keys:
                curve = _spectrum_curve(self._results.get(key), quantity)
                if curve is not None:
                    curves.append((_key_text(key), *curve))
            plots.append((quantity, curve_xy(curves, 'Wavelength', 'm', quantity)))
        return plots

    def _spectrum_overlays(self, plot, drawn):
        """Analytic curves of each drawn result: its flux / series curves, with ratios its ratio curves."""
        ratios = self._spectrum_shows_ratios()
        for series, color in drawn.drawn:
            key = split_key(series)
            result = self._results.get(key)
            if ratios:
                target_series = RATIO_SERIES
            else:
                target_series = _FLUX_SERIES if result.get('kind') == 'flux_spectrum' else drawn.name
            self._draw_feature_curves(plot, key, result, color, target_series, drawn.x_scale,
                                      report=self._spectrum_view.add_error)

    def _ratio_plots(self):
        """One plot per quantity: each selected result's quantity / the reference's, on identical wavelengths; the
        notes (no reference, mismatched wavelengths) go to _ratio_notes."""
        keys = self._selected_keys(self._spectrum_list, self._spectrum_keys)
        reference_key = self._current_key(self._reference_combo, self._spectrum_keys, offset=1)
        plots, self._ratio_notes = [], {}
        for quantity in self._spectrum_quantity_names(keys):
            reference_curve = (None if reference_key is None
                               else _spectrum_curve(self._results.get(reference_key), quantity))
            curves, mismatched = [], []
            for key in keys:
                curve = _spectrum_curve(self._results.get(key), quantity)
                if curve is None or reference_curve is None or key == reference_key:
                    continue
                ratio = _spectrum_ratio(curve, reference_curve)
                if ratio is None:
                    mismatched.append(_key_text(key))
                else:
                    curves.append((_key_text(key), curve[0], ratio, '1'))
            self._ratio_notes[quantity] = _ratio_note(reference_key, reference_curve, quantity, mismatched)
            plots.append((quantity, curve_xy(curves, 'Wavelength', 'm', 'Ratio to reference')))
        return plots

    # ---- far-field tab ----
    def _far_field_key(self):
        return self._current_key(self._far_field_combo, self._far_field_keys)

    def _update_far_field_controls(self):
        result = self._current_result(self._far_field_combo, self._far_field_keys)
        self._fill_combo(self._far_frequency_combo, _wavelength_labels(result))
        self._draw_far_field()

    def _draw_far_field(self):
        """Polar plot (hand drawn) of the chosen far field and wavelength, then the dB view; a hidden tab draws the
        polar plot from the dB view's redraw once shown (_far_field_overlays)."""
        self._polar_plot.clear()
        self._db_view.request_redraw()
        self._polar_stale = not self._polar_plot.isVisible()
        if not self._polar_stale:
            self._draw_polar_far_field()

    def _draw_polar_far_field(self):
        self._polar_stale = False
        result = self._current_result(self._far_field_combo, self._far_field_keys)
        frequency_index = self._far_frequency_combo.currentIndex()
        if not isinstance(result, dict) or frequency_index < 0:
            return
        angles = np.asarray(result['angles'], dtype=np.float64)
        normalised, _db = normalised_db(np.atleast_2d(result['power'])[frequency_index], _FAR_FIELD_DB_FLOOR)
        _draw_polar(self._polar_plot, angles, normalised)
        wavelength = float(np.atleast_1d(result['wavelength'])[frequency_index])
        self._draw_feature_curves(   # analytic patterns normalised to their own peak, like the simulated one
            self._polar_plot, self._far_field_key(), result, SERIES_COLORS[0], wavelength=wavelength,
            transform=lambda angle, values: (normalised_db(values)[0] * np.cos(angle),
                                             normalised_db(values)[0] * np.sin(angle)),
            report=lambda text: None)   # the dB view reports the same curves

    def _far_field_plots(self):
        """The chosen far field at the chosen wavelength in dB below its peak vs angle (one plot)."""
        key = self._far_field_key()
        result = self._results.get(key)
        frequency_index = self._far_frequency_combo.currentIndex()
        if key is None:
            return []
        if not isinstance(result, dict) or frequency_index < 0:
            return [(_key_text(key), None)]
        _normalised, db = normalised_db(np.atleast_2d(result['power'])[frequency_index], _FAR_FIELD_DB_FLOOR)
        wavelength = float(np.atleast_1d(result['wavelength'])[frequency_index])
        xy = curve_xy([('normalised power', np.degrees(np.asarray(result['angles'], dtype=np.float64)), db, 'dB')],
                      'Angle from +x', 'deg', 'Normalised power', wavelength=wavelength)
        return [(f'{_key_text(key)}, {_wavelength_text(wavelength)}', xy)]

    def _far_field_overlays(self, plot, drawn):
        if self._polar_stale:   # changed while the tab was hidden
            self._draw_polar_far_field()
        key = self._far_field_key()
        self._draw_feature_curves(
            plot, key, self._results.get(key), SERIES_COLORS[0], wavelength=drawn.xy['wavelength'],
            transform=lambda angle, values: (np.degrees(angle), normalised_db(values, _FAR_FIELD_DB_FLOOR)[1]),
            report=self._db_view.add_error)

    # ---- DFT tab ----
    def _update_dft_controls(self):
        result = self._current_result(self._dft_combo, self._dft_keys)
        labels = _wavelength_labels(result)
        self._fill_combo(self._dft_frequency_combo, labels)
        intensity_choices = _dft_intensity_choices(result)
        self._dft_intensity_keys = [intensity_key for _, intensity_key in intensity_choices]
        self._fill_combo(self._dft_intensity_combo, [label for label, _ in intensity_choices])
        self._dft_show_button.setEnabled(bool(labels))

    def _show_dft(self):
        key = self._current_key(self._dft_combo, self._dft_keys)
        frequency_index = self._dft_frequency_combo.currentIndex()
        intensity_index = self._dft_intensity_combo.currentIndex()
        if key is None or frequency_index < 0 or not 0 <= intensity_index < len(self._dft_intensity_keys):
            self._show_message('No DFT result yet: compute first.')
            return
        self._pause_follow_live()
        physics, feature = key
        intensity_key = self._dft_intensity_keys[intensity_index]
        self._call(lambda session, physics=physics, feature=feature, index=frequency_index, key=intensity_key:
                   _dft_field_data(session, physics, feature, index, key), self._show_field)

    # ---- curves & tables tab ----
    def _xy_table_result(self):
        """Fetched result dict of the chosen 'xy' / 'table' feature, None before it is fetched."""
        result = self._current_result(self._xy_table_combo, self._xy_table_keys)
        return result if isinstance(result, dict) else None

    def _xy_is_table(self):
        result = self._xy_table_result()
        return result is not None and result.get('kind') == 'table'

    def _current_table_x_column(self):
        """Chosen abscissa column of a table result ('' = row index)."""
        index = self._table_x_combo.currentIndex()
        return self._table_x_columns[index] if 0 <= index < len(self._table_x_columns) else ''

    def _xy_view_data(self):
        """(key, fetched result, plotted xy dict): the chosen xy result, or table_xy of the chosen table over the
        chosen column; result / xy None before a fetch."""
        key = self._current_key(self._xy_table_combo, self._xy_table_keys)
        result = self._xy_table_result()
        if result is None or result.get('kind') != 'table':
            return key, result, result
        try:
            return key, result, table_xy(result, self._current_table_x_column())
        except ValueError:
            return key, result, None

    def _xy_plots(self):
        key, _result, xy = self._xy_view_data()
        return [] if key is None else [(_key_text(key), xy)]

    def _update_xy_table_controls(self):
        """Column choices of a chosen table (kept, else the x_column of its first curve, else the first number
        column), its rows in the table view, then the plot."""
        result = self._xy_table_result()
        is_table = result is not None and result.get('kind') == 'table'
        columns = list(numeric_table_columns(result)) if is_table else []
        key = self._current_key(self._xy_table_combo, self._xy_table_keys)
        curve_columns = [] if key is None else [curve.x_column for curve in self._analytic_curves(_key_text(key))]
        choices = [''] + columns
        if choices == self._table_x_columns:   # same columns: keep the choice
            preferred = self._current_table_x_column()
        else:
            preferred = next((column for column in curve_columns if column in choices),
                             columns[0] if len(columns) > 1 else '')
        self._table_x_columns = choices
        self._fill_combo(self._table_x_combo, ['(row)'] + columns, select=choices.index(preferred))
        self._table_x_combo.setVisible(is_table)
        self._result_table.setVisible(is_table)
        self._fill_result_table(result if is_table else None)
        self._update_xy_plot()

    def _update_xy_plot(self):
        """Plot box hidden for a table without number columns, then redraw."""
        _key, result, xy = self._xy_view_data()
        plot_visible = (xy is not None and bool(xy['series'])) or result is None or result.get('kind') != 'table'
        self._xy_view.panel.setVisible(plot_visible)
        self._xy_view.request_redraw()

    def _xy_overlays(self, plot, drawn):
        """Dashed analytic curves of each drawn series (a table's: of the chosen x_column) and, with the first unit
        group, the curves of no particular series in the first series colour."""
        key, result, _xy = self._xy_view_data()
        x_column = self._current_table_x_column() if result.get('kind') == 'table' else None
        report = self._xy_view.add_error
        for series, color in drawn.drawn:
            self._draw_feature_curves(plot, key, result, color, series, drawn.x_scale, any_series=False,
                                      x_column=x_column, report=report)
        if drawn.first_group:
            self._draw_feature_curves(plot, key, result, SERIES_COLORS[0], '', drawn.x_scale, x_column=x_column,
                                      report=report)

    def _fill_result_table(self, result):
        """Show a table result read-only under 'column [unit]' headers; None empties the view. Skipped while the
        view already shows this result (document edits refresh the controls, not the fetched rows)."""
        if result is self._result_table_source:
            return
        self._result_table_source = result
        columns = [] if result is None else [str(column) for column in result['columns']]
        units = [] if result is None else [str(unit) for unit in result.get('units') or [''] * len(columns)]
        rows = [] if result is None else result['rows']
        table = self._result_table
        table.clear()
        table.setColumnCount(len(columns))
        table.setRowCount(len(rows))
        table.setHorizontalHeaderLabels([column if unit in ('', '1') else f'{column} [{unit}]'
                                         for column, unit in zip(columns, units)])
        for row_index, row in enumerate(rows):
            for column_index, value in enumerate(row):
                table.setItem(row_index, column_index, QtWidgets.QTableWidgetItem(table_cell_text(value)))
        table.resizeColumnsToContents()

    # ---- derived values tab ----
    def _schedule_derived_table(self):
        if not self._derived_pending:
            self._derived_pending = True
            QtCore.QTimer.singleShot(0, self._rebuild_derived_table)

    def _derived_rows(self):
        """[(texts, tips, numbers)] per document derived value: the seven table texts, column -> tooltip and the export
        row (name, expression, operation, value, reference, relative error, tolerance; None where unknown)."""
        parameters = self._curve_parameters()
        curves = _curves_by_name(self._document.spec)
        rows = []
        for derived in self._document.spec.derived_values:
            definition = _derived_definition(derived, curves)
            stored = self._derived_values.get(derived.name)
            value, value_tip = math.nan, ''
            if stored is None or stored[0] != definition:   # never evaluated, or edited since
                value_text = ''
                value_tip = 'Not evaluated for this definition: press Evaluate.'
            elif not math.isfinite(float(stored[1])):
                value_text = 'failed'
                value_tip = stored[2] or 'Not a finite number. Any failure is logged as a warning in the Messages tab.'
            else:
                value = float(stored[1])
                value_text = f'{value:.6g}'
            reference, error = math.nan, math.nan
            reference_text, reference_tip, error_text = '', '', ''
            tolerance_text = '' if derived.tolerance is None else f'{float(derived.tolerance):.3g}'
            if derived.reference is not None:
                try:
                    reference = evaluate_scalar(derived.reference, parameters)
                except USER_ERRORS as exc:
                    reference_text, reference_tip = 'failed', error_message(exc)
                else:
                    reference_text = f'{reference:.6g}'
                    error = relative_error(value, reference)
                    error_text = f'{error:.3g}' if math.isfinite(error) else ''
                    if error_text and derived.tolerance is not None:   # the headless self-test gate
                        tolerance_text += ': passes' if error <= float(derived.tolerance) else ': fails'
            expression_text = definition[0] if derived.source is None else f'{definition[0]} of {derived.source}'
            texts = (derived.name, expression_text, _operation_text(derived, labelled=True), value_text,
                     reference_text, error_text, tolerance_text)
            numbers = (derived.name, expression_text, _operation_text(derived),
                       *(float(number) if math.isfinite(number) else None for number in (value, reference, error)),
                       None if derived.tolerance is None else float(derived.tolerance))
            mesh_tip = ''
            if derived.source is None and several_meshes(self._document.spec):   # S7.2 D3
                mesh_name = evaluation_mesh(self._document.spec, self._document.registry, derived.expression)
                mesh_tip = (f"Evaluated on mesh '{mesh_name}' (the mesh of the first physics the expression names); "
                            'fields of other meshes are interpolated onto it.')
            rows.append((texts, {1: mesh_tip, 3: value_tip, 4: reference_tip}, numbers))
        return rows

    def _export_derived_table(self):
        """The Derived values table as a 'table' result (.csv by default): the values of the last Evaluate."""
        rows = self._derived_rows()
        if not rows:
            self._show_message('No derived values to export: add one under Result settings.')
            return
        if not any(numbers[3] is not None for _texts, _tips, numbers in rows):
            self._show_message('No derived value evaluated for the current definitions: press Evaluate first.')
            return
        table = {'kind': 'table', 'columns': list(_DERIVED_EXPORT_COLUMNS),
                 'rows': [list(numbers) for _texts, _tips, numbers in rows]}
        self._write_result(table, 'derived_values.csv', _CSV_FIRST_FILE_TYPES)

    def _rebuild_derived_table(self):
        self._derived_pending = False
        rows = self._derived_rows()
        table = self._derived_table
        table.setRowCount(len(rows))
        for row, (texts, tips, _numbers) in enumerate(rows):
            for column, text in enumerate(texts):
                item = QtWidgets.QTableWidgetItem(text)
                item.setFlags(QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled)
                if tips.get(column):
                    item.setToolTip(tips[column])
                table.setItem(row, column, item)
        self._update_tab_visibility()   # Derived values follows the model's derived_values

    def _evaluate_derived(self):
        if self._summary is None:             # no session, or one built from a replaced model
            self._show_message('No results yet: build or compute first.')
            return
        derived_specs = copy.deepcopy(self._document.spec.derived_values)   # the document's current definitions
        curves = copy.deepcopy(self._document.spec.derived_curves)          # (curve sources)
        self._call(lambda session: _evaluate_derived_values(session, derived_specs, curves), self._on_derived_values)

    def _on_derived_values(self, values):
        self._derived_values = dict(values)
        self._rebuild_derived_table()

    # ---- derived curves tab ----
    def _schedule_derived_curves(self):
        if not self._derived_curves_pending:
            self._derived_curves_pending = True
            QtCore.QTimer.singleShot(0, self._evaluate_derived_curves)

    def _evaluate_derived_curves(self):
        """derived_curve_plots of the document's current derived curves on the worker; cleared without a session,
        skipped while a job runs (its Built / StudyFinished re-evaluates)."""
        self._derived_curves_pending = False
        curves = copy.deepcopy(self._document.spec.derived_curves)
        if not curves or self._summary is None or self._session_lost_reason is not None:
            self._on_derived_curves([])
            return
        if self._bridge.busy:
            return
        self._call(lambda session: _evaluate_derived_curves(session, curves), self._on_derived_curves)

    def _on_derived_curves(self, plots):
        """Keep the evaluated plots, name failing curves in the view's error line and redraw."""
        self._derived_curve_plots = list(plots)
        failures = [f'{name}: {error}' for _plot, _xy, errors in self._derived_curve_plots
                    for name, error in errors.items()]
        self._derived_view.set_error('Derived curves not drawn: ' + '; '.join(failures) if failures else '')
        self._derived_view.request_redraw()

    @staticmethod
    def _derived_curve_options(name, xy, series):
        """A derived curve's reference series grey dotted (off the series palette)."""
        return {'pen': _REFERENCE_PEN} if series in (xy.get('reference_series') or {}).values() else {}

    def _derived_curve_note(self, name, xy):
        """Run note, then the error (max / rms of curve_error) of each curve of the plot with a reference, judged
        against its self-test tolerance from the document."""
        tolerances = {curve.name: curve.tolerance for curve in self._document.spec.derived_curves}
        parts = []
        for curve_name, record in ((xy or {}).get('curve_error') or {}).items():
            if record.get('note'):
                parts.append(f"{curve_name}: {record['note']}")
                continue
            if not record.get('n_points'):
                parts.append(f'{curve_name}: no point to compare with its reference')
                continue
            text = f"{curve_name}: error to reference max {record['max']:.3g}, rms {record['rms']:.3g}"
            tolerance = tolerances.get(curve_name)
            if tolerance is not None:
                verdict = 'passes' if record['max'] <= float(tolerance) else 'fails'
                text += f' (tolerance {float(tolerance):.3g}: {verdict})'
            parts.append(text)
        return self._run_note('; '.join(parts))

    def _update_derived_series(self):
        """Derived values tab plot of the last run's derived_series (the source-free derived values along its time or
        wavelength axis); hidden without one."""
        xy = (self._study_summary or {}).get('derived_series')
        self._derived_series_plots = [] if not xy else [('Derived values along the study', {
            **xy, 'x': np.asarray(xy['x'], dtype=np.float64),
            'series': {name: np.asarray(values, dtype=np.float64) for name, values in xy['series'].items()}})]
        self._derived_series_view.setVisible(bool(self._derived_series_plots))
        self._derived_series_view.request_redraw()

    # ---- sweep tab ----
    def _update_sweep_tab(self):
        """Sweep plots of the last study, or of the values done so far of a running sweep (one per derived value,
        1D result series, derived series, final value and table column), kept for the Sweep view; references and
        analytic curves are drawn as its overlays."""
        sweep = self._current_sweep()
        self._sweep_plots, self._sweep_quantities, self._sweep_value_results = [], {}, {}
        errors = []
        if sweep:
            quantities = [(f'Derived value: {name}', ('derived', name)) for name in sweep.get('derived') or {}]
            result_series = {}   # key -> its series choices over every value (dft_fields metadata has no curve)
            for per_value in sweep.get('results') or []:
                for result_key, result in (per_value or {}).items():
                    if sweep_value_curve(result) is None:
                        continue
                    choices = result_series.setdefault(result_key, [])
                    choices += [series for series in _sweep_series_choices(result) if series not in choices]
            result_keys = list(result_series)
            for result_key, choices in result_series.items():
                quantities += [(f'Result: {result_key}' + (f' [{series}]' if len(choices) > 1 else ''),
                                ('result', (result_key, series))) for series in choices]
            series_names = dict.fromkeys(name for xy in sweep.get('derived_series') or [] if xy
                                         for name in xy['series'])
            axis = next((xy['x_label'].lower() for xy in sweep.get('derived_series') or [] if xy), '')
            quantities += [(f'Derived value vs {axis}: {name}', ('series', name)) for name in series_names]
            for result_key in result_keys:
                if any(((per_value or {}).get(result_key) or {}).get('kind') == 'timeseries'
                       for per_value in sweep.get('results') or []):
                    quantities.append((f'Final value vs {sweep["parameter"]}: {result_key}', ('final', result_key)))
            table_columns = {}   # result key -> its number columns over every value, in first-seen order
            for per_value in sweep.get('results') or []:
                for result_key, result in (per_value or {}).items():
                    if isinstance(result, dict) and result.get('kind') == 'table':
                        columns = table_columns.setdefault(result_key, [])
                        columns += [column for column in numeric_table_columns(result) if column not in columns]
            for result_key, columns in table_columns.items():
                quantities += [(f'Table column vs {sweep["parameter"]}: {result_key} {column} (sorted per value)',
                                ('table', (result_key, column))) for column in columns]
            for label, quantity in quantities:
                try:
                    xy = self._sweep_xy(sweep, label, *quantity)
                except ValueError as exc:
                    errors.append(f'{label}: {error_message(exc)}')
                    xy = None
                self._sweep_plots.append((label, xy))
                self._sweep_quantities[label] = quantity
            running = sweep is self._live_sweep   # _current_sweep prefers the finished table
            self._sweep_info.setText(f"Sweep over {sweep['parameter']}: {len(sweep['values'])} values"
                                     + (' done so far (running)' if running else ''))
        else:
            self._sweep_info.setText('No sweep in the last study.')
        self._sweep_view.set_error('; '.join(errors))
        self._sweep_view.request_redraw()

    def _sweep_xy(self, sweep, label, quantity_kind, name):
        """xy of one sweep quantity: scalars vs the swept parameter (derived, final), table bands '<column>_<k>', or
        one curve per swept value (result (key, series), derived series name; ValueError for a table the bands cannot
        be built from)."""
        parameter, parameter_values = sweep['parameter'], np.asarray(sweep['values'], dtype=np.float64)
        per_value_results = list(sweep.get('results') or [])[:parameter_values.size]
        per_value_results += [None] * (parameter_values.size - len(per_value_results))
        if quantity_kind == 'series':
            per_value_results = [{name: xy} if xy else None for xy in sweep.get('derived_series') or []]
            per_value_results += [None] * (parameter_values.size - len(per_value_results))
            quantity_kind, name = 'result', (name, name)
        if quantity_kind == 'table':
            result_key, column = name
            tables = [(per_value or {}).get(result_key) for per_value in per_value_results]
            tables = [table if isinstance(table, dict) and table.get('kind') == 'table' else None for table in tables]
            bands = sweep_table_xy(parameter, parameter_values, tables, column)
            return curve_xy([(series, bands['x'], values, bands['series_units'].get(series, ''))
                             for series, values in bands['series'].items()], parameter, '', f'{result_key} {column}')
        if quantity_kind in ('derived', 'final'):
            if quantity_kind == 'derived':
                scalars, series, unit = np.asarray(sweep['derived'][name], dtype=np.float64), name, ''
            else:   # last sample of each value's timeseries (NaN where a value has none)
                results = [(per_value or {}).get(name) for per_value in per_value_results]
                scalars = np.array([_final_value(result) for result in results], dtype=np.float64)
                series = f'{name} (final value)'
                unit = next((result.get('unit', '') for result in results if isinstance(result, dict)), '')
            finite = np.isfinite(scalars)
            return curve_xy([(series, parameter_values[finite], scalars[finite], unit)], parameter, '', series)
        result_key, result_series = name
        curves, value_results, axes = [], {}, None
        for value, per_value in zip(parameter_values, per_value_results):
            result = (per_value or {}).get(result_key)
            curve = sweep_value_curve(result, result_series)
            if curve is None:
                continue
            axes = axes or (curve['x_label'], curve['x_unit'], curve['y_label'])
            series = f'{parameter} = {value:.4g}'
            if series in value_results:   # repeated swept values stay separate curves
                series = f'{series} ({len(value_results) + 1})'
            value_results[series] = (float(value), result)
            curves.append((series, curve['x'], curve['y'], curve['unit']))
        self._sweep_value_results[label] = value_results
        return None if axes is None else curve_xy(curves, *axes)

    def _sweep_curve_options(self, name, xy, series):
        """Markers on the scalar / band plots and on one-point value curves (a pen-only one-point curve draws
        nothing)."""
        if (self._sweep_quantities.get(name, ('',))[0] not in ('result', 'series')
                or np.size(xy['series'][series]) < 2):
            return POINT_OPTIONS
        return {}

    def _sweep_overlays(self, plot, drawn):
        sweep = self._current_sweep()
        quantity = self._sweep_quantities.get(drawn.name)
        if not sweep or quantity is None:
            return
        quantity_kind, name = quantity
        report = self._sweep_view.add_error
        if quantity_kind in ('derived', 'final'):
            self._draw_sweep_curves(plot, sweep, name, report)
        elif quantity_kind == 'table':
            for series, color in drawn.drawn:
                self._draw_sweep_analytic(plot, series, color, report)
            self._draw_sweep_analytic(plot, '', SERIES_COLORS[0], report)
        elif quantity_kind == 'result' and self._analytic_curves(name[0]):   # each value's curves at its parameters
            curve_spec, key = self._curve_spec(), split_key(name[0])
            value_results = self._sweep_value_results.get(drawn.name, {})
            for position, (series, color) in enumerate(drawn.drawn):
                value, result = value_results[series]
                self._draw_sweep_value_curves(plot, key, result, name[1], color, drawn.x_scale,
                                              parameters_with(curve_spec, sweep['parameter'], value),
                                              legend=position == 0, report=report)

    # ---- analytic curves (document definitions, parameters of the session) ----
    def _analytic_curves(self, target):
        """The document's analytic curves on target ('physics.feature', 'sweep' or 'line')."""
        return [curve for curve in self._document.spec.analytic_curves if curve.target == target]

    def _curve_spec(self):
        """Spec of the last build (its parameters and sweep), else the document's."""
        return self._bridge.results_spec or self._document.spec

    def _curve_parameters(self):
        """Parameters of the worker session when fetched, else those of the study summary / last build."""
        if self._session_parameters is not None:
            return self._session_parameters
        return built_parameters(self._curve_spec(), self._study_summary)

    def _draw_feature_curves(self, plot, key, result, color, series='', x_scale=1.0, wavelength=None,
                             transform=None, any_series=True, frequency=None, x_column=None, parameters=None,
                             legend=True, report=None):
        """Dashed analytic curves on result key whose series is `series` (any_series: also those with series '',
        except on the ratio / FFT plot) and, unless None, whose x_column is x_column; x divided by x_scale after
        transform(x, y); parameters None = the session's; failures go to report(text)."""
        if key is None or not isinstance(result, dict):
            return
        for curve in self._analytic_curves(_key_text(key)):
            matches = curve.series == series or (any_series and curve.series == ''
                                                 and series not in (RATIO_SERIES, FFT_SERIES))
            if not matches or (x_column is not None and curve.x_column != x_column):
                continue
            try:
                x, y = feature_curve(curve, result, self._curve_parameters() if parameters is None else parameters,
                                     wavelength, frequency)
            except _CURVE_ERRORS as exc:
                report(f"Analytic curve '{curve.name}': {error_message(exc)}")
                continue
            if transform is not None:
                x, y = transform(x, y)
            plot.plot(np.asarray(x) / x_scale, y, pen=dashed_pen(color),
                      name=f'{curve.name} (analytic)' if legend else None, connect='finite')

    def _plot_sweep_curve(self, plot, label, expression, pen, report):
        """sweep_curve of expression on the Sweep plot; False (reported) when it fails."""
        try:
            x, y = sweep_curve(expression, self._curve_spec())
        except _CURVE_ERRORS as exc:
            report(f'{label}: {error_message(exc)}')
            return False
        plot.plot(x, y, pen=pen, name=label, connect='finite')
        return True

    def _draw_sweep_analytic(self, plot, series, color, report):
        """The sweep analytic curves of exactly this series, dashed in color."""
        for curve in self._analytic_curves('sweep'):
            if curve.series == series:
                self._plot_sweep_curve(plot, f'{curve.name} (analytic)', curve.expression, dashed_pen(color), report)

    def _draw_sweep_value_curves(self, plot, key, result, series, color, x_scale, parameters, legend, report):
        """Dashed curves targeting result key over one sweep value's result, in the coordinates of
        analytic.sweep_value_curve (series: the drawn xy / flux series, '' = its default)."""
        kind = result.get('kind') if isinstance(result, dict) else None
        options = {'parameters': parameters, 'legend': legend, 'report': report}
        if kind == 'flux_spectrum':
            self._draw_feature_curves(plot, key, result, color, series or _FLUX_SERIES, x_scale, **options)
        elif kind == 'timeseries':
            self._draw_feature_curves(plot, key, result, color, **options)
        elif kind == 'far_field':
            self._draw_feature_curves(
                plot, key, result, color, **options,
                transform=lambda angle, values: (np.degrees(angle), normalised_db(values, _FAR_FIELD_DB_FLOOR)[1]))
        elif kind == 'xy' and result['series']:
            self._draw_feature_curves(plot, key, result, color, series or next(iter(result['series'])), x_scale,
                                      **options)

    def _draw_sweep_curves(self, plot, sweep, name, report):
        """Grey dotted reference of derived value `name` (the document's expression on analytic.sweep_curve's points,
        else the summary's per value), then the sweep analytic curves of series name or '' dashed in the next series colours."""
        drawn_reference = False
        for derived in self._document.spec.derived_values:
            if derived.name == name and derived.reference is not None:
                expression = (derived.reference if isinstance(derived.reference, str)
                              else repr(float(derived.reference)))
                drawn_reference = (self._plot_sweep_curve(plot, f'{name} reference', expression, _REFERENCE_PEN,
                                                          report) or drawn_reference)
        stored = (sweep.get('reference') or {}).get(name)
        if not drawn_reference and stored is not None:   # definition removed or failing: the run's references
            values = np.asarray(sweep['values'], dtype=np.float64)
            order = np.argsort(values)
            plot.plot(values[order], np.asarray(stored, dtype=np.float64)[order], pen=_REFERENCE_PEN,
                      name=f'{name} reference (run)', connect='finite')
        analytic = [curve for curve in self._analytic_curves('sweep') if curve.series in ('', name)]
        for position, curve in enumerate(analytic, start=1):   # colour 0 is the simulated series
            self._plot_sweep_curve(plot, f'{curve.name} (analytic)', curve.expression,
                                   dashed_pen(SERIES_COLORS[position % len(SERIES_COLORS)]), report)

    # ---- line profile tab ----
    def _update_profile_choices(self):
        """Line profile combo: the drawn line, then the document's line curves (current choice kept by name; when the
        curve list changes while no line is drawn, the first curve)."""
        current = self._current_profile_curve()
        previous_names = self._profile_curve_names
        self._profile_curve_names = [curve.name for curve in self._analytic_curves('line')]
        labels = [_DRAWN_PROFILE] + [f'{curve.name}: {curve.field}' for curve in self._analytic_curves('line')]
        if current in self._profile_curve_names:
            index = self._profile_curve_names.index(current) + 1
        elif self._profile is None and self._profile_curve_names and self._profile_curve_names != previous_names:
            index = 1
        else:
            index = 0
        changed = index != self._profile_combo.currentIndex() or self._profile_combo.count() != len(labels)
        self._fill_combo(self._profile_combo, labels, select=index)
        if changed or current is not None:
            self._on_profile_choice()

    def _current_profile_curve(self):
        """Name of the line curve chosen in the Line profile combo, None for the drawn line."""
        index = self._profile_combo.currentIndex() - 1
        return self._profile_curve_names[index] if 0 <= index < len(self._profile_curve_names) else None

    def _on_profile_choice(self, *args):
        name = self._current_profile_curve()
        if name is None:
            if self._profile is not None and self._profile.get('curve') is None:
                self._plot_profile(self._profile)
            else:
                self._profile = None
                self._profile_view.set_error('')
                self._profile_view.request_redraw()
                self._profile_info.setText(_PROFILE_HINT)
            return
        if self._summary is None or self._bridge.results_spec is None:
            self._profile_info.setText(f'{name}: build or compute first (the profile reads the final state).')
            return
        curve = self._document.spec.analytic_curve(name)
        field_key = curve.field
        field_keys = line_curve_field_keys(curve)   # a projected interface curve: + the y component
        mid_run = self._run_active   # a field fetched while the study runs is not its final state
        if is_segment_3d(curve.line):   # the 3D-segment form: sampled from the volume, or the snapshot's profile
            if curve.snapshot is not None:
                labels = self._snapshot_labels + [None] * (len(self._snapshot_times) - len(self._snapshot_labels))
                try:
                    index = snapshot_index(curve.snapshot, labels[:len(self._snapshot_times)])
                except _CURVE_ERRORS as exc:
                    self._profile_info.setText(f'{name}: snapshot: {error_message(exc)}')
                    return
                moment = display_label(labels[index]) or f't = {_time_text(self._snapshot_times[index])}'
                self._call(lambda session, index=index, name=name: session.snapshots[index].segments.get(name),
                           lambda segment, name=name, moment=moment: self._show_segment_profile(name, segment, False,
                                                                                               moment))
                return
            try:
                start, end = line_endpoints(curve.line, self._curve_parameters())
            except _CURVE_ERRORS as exc:
                self._profile_view.set_error(f"Analytic curve '{name}': line: {error_message(exc)}")
                return
            self._call(lambda session, key=field_key, start=start, end=end, sampling=curve.sampling:
                       _session_segment_profile(session, key, start, end, sampling),
                       lambda segment, name=name: self._show_segment_profile(name, segment, mid_run))
            return
        if curve.snapshot is None:
            try:
                plane = line_plane(curve, self._curve_parameters())
            except _CURVE_ERRORS as exc:
                self._profile_view.set_error(f"Analytic curve '{name}': plane: {error_message(exc)}")
                return
            plane = self._line_curve_plane() if plane is None else plane   # always a z cut (or mid-height)
            # polar / spherical interface curves read the (r, angle) cells themselves (exact one-sided planes)
            native = bool(curve.interface) and self._document.spec.domain.coordinates != 'cartesian'
            fetch_key = self._profile_fetch_key(curve, plane)
            if self._reuse_profile_fetch(name, fetch_key):
                return
            self._call(lambda session, keys=field_keys, plane=plane, native=native:
                       [_session_field(session, key, plane, native) for key in keys],
                       lambda fields, name=name, fetch_key=fetch_key: self._show_fetched_profile(
                           fetch_key, name, fields, None, mid_run))
            return
        labels = self._snapshot_labels + [None] * (len(self._snapshot_times) - len(self._snapshot_labels))
        stored = (self._study_summary or {}).get('stored_fields') or []
        try:
            index = snapshot_index(curve.snapshot, labels[:len(self._snapshot_times)])
            missing = [key for key in field_keys if key not in stored]
            if missing:
                raise ValueError(f"{', '.join(missing)} not stored in snapshots (stored: "
                                 f"{', '.join(stored) or 'none'})")
        except _CURVE_ERRORS as exc:
            self._profile_info.setText(f'{name}: snapshot: {error_message(exc)}')
            return
        moment = display_label(labels[index]) or f't = {_time_text(self._snapshot_times[index])}'
        fetch_key = self._profile_fetch_key(curve, index)
        if self._reuse_profile_fetch(name, fetch_key):
            return
        self._call(lambda session, index=index, keys=field_keys: [session.snapshots[index].fields[key] for key in keys],
                   lambda fields, name=name, moment=moment, fetch_key=fetch_key: self._show_fetched_profile(
                       fetch_key, name, fields, moment, False))

    def _profile_fetch_key(self, curve, where):
        """What a line curve's field fetch depends on (where: its plane or snapshot index); None while a study runs
        (live state: always refetched)."""
        if self._run_active:
            return None
        return (curve.name, curve.field, repr(curve.line), curve.interface, curve.projection, curve.sampling,
                repr(where), repr(self._curve_parameters()), len(self._snapshot_times))

    def _reuse_profile_fetch(self, name, fetch_key) -> bool:
        """Redraw the last fetched fields of line curve name when fetch_key matches (an expression edit): no worker
        call; False when a fetch is needed."""
        cached = self._profile_fetched
        if fetch_key is None or cached is None or cached[0] != fetch_key or cached[1] is not self._summary:
            return False   # another key, or other results (the run summary object)
        fields, moment, mid_run = cached[2]
        self._show_curve_profile(name, fields, moment, mid_run)
        return True

    def _show_fetched_profile(self, fetch_key, name, fields, moment, mid_run):
        self._profile_fetched = None if fetch_key is None else (fetch_key, self._summary, (fields, moment, mid_run))
        self._show_curve_profile(name, fields, moment, mid_run)

    def _show_curve_profile(self, name, fields, moment=None, mid_run=False):
        """Plot the fetched fields (line_curve_field_keys order) along line curve `name` (moment: its snapshot text)
        with its analytic curve dashed."""
        if name != self._current_profile_curve():
            return                     # the choice changed while the field was fetched
        curve = self._document.spec.analytic_curve(name)
        parameters = self._curve_parameters()
        field_data, info = fields[0], fields[0].info
        try:
            start, end = line_endpoints(curve.line, parameters)
        except _CURVE_ERRORS as exc:
            self._profile_view.set_error(f"Analytic curve '{name}': line: {error_message(exc)}")
            return
        if curve.interface:   # one-sided samples of the interface selection (igor 1.8)
            try:
                distance, samples, _ = interface_line_profile(
                    self._document.spec, parameters, curve.interface, field_data, start, end,
                    partner_data=fields[-1] if curve.projection else None, projection=curve.projection)
            except _CURVE_ERRORS as exc:
                self._profile_view.set_error(f"Analytic curve '{name}': interface: {error_message(exc)}")
                return
            if curve.projection:
                info = projected_info(info, curve.projection)
        else:
            distance, samples, _, _ = line_profile(field_data.values, field_data.extent, start, end,
                                                   field_data.x_centers, field_data.y_centers)
        profile = {'distance': distance, 'values': samples, 'info': info, 'start': start, 'end': end,
                   'curve': name, 'plane': getattr(field_data, 'plane', None), 'snapshot': moment,
                   'resampled': field_data.native_shape is not None, 'mid_run': mid_run,
                   'mesh': self._field_mesh(field_data)}
        try:
            profile['analytic'] = line_curve(curve.expression, parameters, start, end,
                                             z=field_plane_height(field_data))
        except _CURVE_ERRORS as exc:
            profile['error'] = f"Analytic curve '{name}': {error_message(exc)}"
        self._plot_profile(profile)

    def _show_segment_profile(self, name, segment, mid_run, moment=None):
        """Plot a session.LineProfile of line curve `name` (3D segment; moment: its snapshot text) with its analytic
        curve dashed."""
        if name != self._current_profile_curve():
            return                     # the choice changed while the profile was fetched
        if segment is None:            # a snapshot of a run before this curve existed
            self._profile_info.setText(f'{name}: the snapshot holds no profile of this line (compute again)')
            return
        curve = self._document.spec.analytic_curve(name)
        profile = {'distance': segment.distance, 'values': segment.values, 'info': segment.info,
                   'start': segment.start, 'end': segment.end, 'curve': name, 'plane': None, 'snapshot': moment,
                   'axes': ('x', 'y', 'z'), 'resampled': False, 'mid_run': mid_run}
        try:
            profile['analytic'] = line_curve(curve.expression, self._curve_parameters(), segment.start, segment.end)
        except _CURVE_ERRORS as exc:
            profile['error'] = f"Analytic curve '{name}': {error_message(exc)}"
        self._plot_profile(profile)

    def _on_profile_changed(self, profile):
        if self._current_profile_curve() is not None:   # a drawn line replaces the chosen curve's line
            self._profile_combo.blockSignals(True)
            self._profile_combo.setCurrentIndex(0)
            self._profile_combo.blockSignals(False)
        field_data = self._canvas.current_field
        self._plot_profile(dict(profile, plane=getattr(field_data, 'plane', None),
                                axes=tuple(getattr(field_data, 'axes', ('x', 'y'))),   # start / end along them
                                mid_run=self._run_active and self._follow_live.isChecked(),
                                mesh=self._field_mesh(field_data)))

    def _field_mesh(self, field_data):
        """Mesh name of a FieldData of the results (FieldData.mesh, else its physics' mesh in the built spec); ''
        with one mesh or without a field."""
        spec = self._bridge.results_spec or self._document.spec
        if field_data is None or not several_meshes(spec):
            return ''
        return field_mesh_name(field_data, spec)

    def _plot_profile(self, profile):
        """Show a profile dict {'distance' [m], 'values', 'info', 'start', 'end', optional 'curve', 'analytic',
        'error', 'snapshot', 'plane', 'axes' (of start / end, default x, y), 'resampled', 'mid_run'} in the Line
        profile view."""
        self._profile = profile
        info = profile['info']
        length_scale, length_unit_text = length_unit(max(abs(value) for value in (*profile['start'], *profile['end']))
                                                     or 1e-6)
        start = ', '.join(f'{value / length_scale:.4g}' for value in profile['start'])
        end = ', '.join(f'{value / length_scale:.4g}' for value in profile['end'])
        axes = tuple(profile.get('axes') or ('x', 'y'))
        axes_text = '' if axes == ('x', 'y') else f" ({', '.join(axes)})"
        text = f'{info.label} from ({start}) to ({end}) {length_unit_text}{axes_text}'
        plane = profile.get('plane')
        if plane is not None:
            text += f' in the cut {plane[0]} = {float(plane[1]) / length_scale:.4g} {length_unit_text}'
        if profile.get('snapshot'):
            text += f" at snapshot {profile['snapshot']}"
        if profile.get('mesh'):
            text += f" on mesh '{profile['mesh']}'"
        if profile.get('mid_run'):
            text += ' (mid-run state, not the final one)'
        self._profile_info.setText(text)
        self._profile_view.set_error(profile.get('error', ''))
        self._profile_view.request_redraw()

    def _profile_extras(self):
        """Line, plane, snapshot and resampling metadata of the shown profile (exports)."""
        profile = self._profile
        extras = {'start': [float(value) for value in profile['start']],
                  'end': [float(value) for value in profile['end']], 'resampled': bool(profile.get('resampled'))}
        if profile.get('plane') is not None:
            extras[f"plane_{profile['plane'][0]}"] = float(profile['plane'][1])
        axes = tuple(profile.get('axes') or ('x', 'y'))
        if axes != ('x', 'y'):   # a side cut: start / end are (u, v) along these axes
            extras['axes'] = list(axes)
        if profile.get('snapshot'):
            extras['snapshot'] = str(profile['snapshot'])
        if profile.get('mesh'):
            extras['mesh'] = str(profile['mesh'])
        return extras

    def _profile_plots(self):
        if self._profile is None:
            return []
        info = self._profile['info']
        name = self._profile.get('curve') or 'drawn line'
        return [(name, curve_xy([(info.label, self._profile['distance'], self._profile['values'], info.unit)],
                                'Distance along the line', 'm', info.label, **self._profile_extras()))]

    def _profile_overlays(self, plot, drawn):
        analytic = self._profile.get('analytic') if self._profile is not None else None
        if analytic is not None:
            plot.plot(analytic[0], analytic[1], pen=dashed_pen(SERIES_COLORS[0], width=2),
                      name=f"{self._profile['curve']} (analytic)", connect='finite')

    def _export_profile(self):
        """export_result of the shown profile with its line, plane, snapshot and resampling flag."""
        if self._profile is None:
            self._show_message('No line profile yet: use "Draw profile line" on a field.')
            return
        info = self._profile['info']
        self._write_result({'kind': 'line_profile', 'distance': self._profile['distance'],
                            'values': self._profile['values'], 'component': info.name, 'unit': info.unit,
                            **self._profile_extras()}, f'profile_{file_stem(info.name)}.npz')
