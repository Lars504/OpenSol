"""Shared 'resonance_monitor' feature row of the time-stepped physics with a point probe (em_fdtd_ez / _hz / _3d,
elastic_waves): band / start Params, apply / result / validate callbacks over a physics' ResonanceProbe declaration,
and the point-trace memory items. Numerics in core.resonance; imports like model.definition_helpers."""
from dataclasses import dataclass, field, replace
from typing import Callable

from ..core.constants import C0
from ..core.memory_estimate import FLOAT32_BYTES, FLOAT64_BYTES
from ..core.resonance import (AMPLITUDE_FLOOR, BAND_KINDS, MATCH_TOLERANCE, MAX_SAMPLES, ResonanceMonitor,
                              analysis_host_bytes)
from .definition_helpers import study_duration, validate_points_inside
from .schema import FeatureType, Issue, Param, feature_path, resolve_values

FEATURE_NAME = 'resonance_monitor'
MAX_POINTS = 16                # probe points of one resonance_monitor (one joint pole fit)
MIN_WINDOW_PERIODS = 10        # analysed window / longest band period below which validation warns
PROBE_INITIAL_SAMPLES = 1024   # growable point traces (core.monitors.FieldProbe, acoustics AcousticProbe): doubled when full


@dataclass(frozen=True)
class ResonanceProbe:
    """What a physics declares for its resonance_monitor: the probe it records and how to time and label it."""
    component: Param             # 'component' Param of the recordable components
    add_probe: Callable          # (ctx, solver, feature_spec, values, selection) -> probe (get_data(), field_component)
    time_step: Callable          # (solver) -> dt [s] between the recorded samples, read when the result is computed
    waveforms: Callable          # (solver) -> the sources' temporal objects (a `duration` when pulsed; None allowed)
    unit: Callable               # (solver, component) -> unit of the recorded component
    source_waveforms: Callable   # (vctx, physics_spec) -> (pulse ends [s] of the enabled sources, names of cw ones)
    band: str = 'wavelength'     # core.resonance.BAND_KINDS: vacuum wavelengths [m] or frequencies [Hz]
    sample_delay_steps: dict = field(default_factory=dict)   # component -> steps its sample lags its recorded time
    validate_point: Callable = validate_points_inside
    pulse_end_text: str = 'the end 2 t0 of the latest gaussian / ricker source pulse'


_BAND_PARAMS = {
    'wavelength': (
        Param('wavelength_min', 'float', 400e-9, label='Shortest wavelength', unit='m', min=0.0, min_exclusive=True,
              tip='Shortest vacuum wavelength searched for resonances, entered in metres (4e-7 for 400 nm).\n\n'
                  'Keep the range inside the spectrum of the source pulse.',
              help='Vacuum wavelength. Keep the band inside the source spectrum; narrower = faster, better '
                   'conditioned fit.'),
        Param('wavelength_max', 'float', 800e-9, label='Longest wavelength', unit='m', min=0.0, min_exclusive=True,
              tip="Longest vacuum wavelength searched for resonances, entered in metres. It must be larger than "
                  "'Shortest wavelength'.",
              help='> wavelength_min.'),
    ),
    'frequency': (
        Param('frequency_min', 'float', 0.5e6, label='Lowest frequency', unit='Hz', min=0.0, min_exclusive=True,
              tip='Lowest frequency searched for resonances, in hertz (5e5 for 500 kHz).\n\n'
                  'Keep the range inside the spectrum of the source pulse.',
              help='Keep the band inside the source spectrum; narrower = faster, better conditioned fit.'),
        Param('frequency_max', 'float', 2e6, label='Highest frequency', unit='Hz', min=0.0, min_exclusive=True,
              tip="Highest frequency searched for resonances, in hertz. It must be larger than 'Lowest frequency'.",
              help='> frequency_min, below the Nyquist frequency 1 / (2 dt).'),
    ),
}
_BAND_UNITS = {'wavelength': 'm', 'frequency': 'Hz'}


def _start_param(pulse_end_text):
    return Param('start_time', 'float', None, label='Analysis start', unit='s', min=0.0, optional=True,
                 tip='Time from which the recorded signal is analysed, in seconds.\n\n'
                     'Leave empty to start when the last source pulse has ended, so only the free ring-down is fitted.',
                 help=f'Start t_s of the analysed window; empty = {pulse_end_text} (0 with only cw sources). The fit '
                      'assumes a free ring-down: a start inside a pulse adds spurious poles.')


def _description(band, note):
    wavelength_column = 'wavelength [m, vacuum C0 / frequency], ' if band == 'wavelength' else ''
    return (
        'Resonance finder (harmonic inversion, as Harminv / matrix pencil): records one component in the cell '
        f'containing each selected point (1 to {MAX_POINTS}; a coordinate on a cell edge goes to the upper cell) every '
        'step. Reading the result (CPU, ~1 s per point) fits the traces from t_s on jointly (shared poles, amplitudes '
        'per point) by damped sinusoids a exp(-gamma (t - t_s)) cos(omega (t - t_s) + phi), so a mode on a nodal line '
        "of one point is found through the others. Result 'table', one row per resonance in the band, strongest "
        f'first: resonance_number, {wavelength_column}frequency = omega / (2 pi) [Hz], frequency_imag = gamma / (2 '
        'pi) [Hz] (> 0 decays), q_factor = omega / (2 gamma), amplitude a [component unit] and phase phi [rad] at t_s '
        'where largest, error = |pole change between two pencil sizes| / omega. Dropped: error > '
        f'{MATCH_TOLERANCE:g}, amplitude < {AMPLITUDE_FLOOR:g} of the strongest at every point, growth > 2x over the '
        f'window. Extras: component, unit, dt [s], start_time, end_time [s], sample_interval [steps], n_samples (<= '
        f"{MAX_SAMPLES}), model_order, {band}_min, {band}_max, note ('' or why none was found), points, "
        'point_amplitudes / point_phases (per row, one per point). Q far above pi f T_window resolves while the '
        'ring-down stays above float32 noise (~1e-7). A mode with a node at every point is missed; a cw source adds '
        f'an undamped line.{" " + note if note else ""}')


def _make_apply(probe_declaration):
    band = probe_declaration.band

    def apply(ctx, solver, feature_spec, values, selection):
        """One probe per selected point (add_probe sees a one-point selection), fitted jointly."""
        names = list(selection.objects)
        probes = [probe_declaration.add_probe(ctx, solver, feature_spec, values,
                                              replace(selection, entities=[name], objects=[name]))
                  for name in names]
        delay = probe_declaration.sample_delay_steps.get(probes[0].field_component, 0.0) if probes else 0.0
        return ResonanceMonitor(probes, values[f'{band}_min'], values[f'{band}_max'], band=band,
                                start_time=values['start_time'], sample_delay_steps=delay, point_names=names)
    return apply


def _make_result(probe_declaration):
    def result(session, solver, monitor, feature_spec, options):
        return monitor.result(probe_declaration.waveforms(solver),
                              probe_declaration.unit(solver, monitor.field_component),
                              probe_declaration.time_step(solver))
    return result


def _make_validate(probe_declaration):
    band = probe_declaration.band
    unit = _BAND_UNITS[band]

    def validate(vctx, physics_spec, feature_spec, values):
        """Point inside, increasing band, cw sources, an analysed window of >= 10 periods of the lowest frequency."""
        issues = probe_declaration.validate_point(vctx, physics_spec, feature_spec, values)
        path = feature_path(physics_spec, feature_spec)
        lower, upper = values[f'{band}_min'], values[f'{band}_max']
        if not upper > lower:
            issues.append(Issue('error', path, f'{band}_max ({upper:.6g} {unit}) must be > {band}_min ({lower:.6g} '
                                               f'{unit})'))
            return issues
        pulse_ends, cw_names = probe_declaration.source_waveforms(vctx, physics_spec)
        if cw_names:
            issues.append(Issue('warning', path, f"cw source(s) {', '.join(cw_names)} keep driving the field, so the "
                                                 f'fit finds an undamped line at their {band} and the ring-down is '
                                                 'not free; use pulsed sources'))
        duration = study_duration(vctx) if vctx.study_type == 'time_dependent' else None
        if not isinstance(duration, (int, float)):
            return issues
        start = values['start_time'] if values['start_time'] is not None else max(pulse_ends, default=0.0)
        lowest_frequency = C0 / upper if band == 'wavelength' else lower
        periods = (duration - start) * lowest_frequency
        if periods < MIN_WINDOW_PERIODS:
            start_text = 'start_time' if values['start_time'] is not None else 'the end of the last source pulse'
            limit_name = f'{band}_max' if band == 'wavelength' else f'{band}_min'
            issues.append(Issue('warning', path, f'the analysed window from {start_text} ({start:.4g} s) to the end '
                                                 f'of the study ({duration:.4g} s) holds {max(periods, 0.0):.3g} '
                                                 f'periods of {limit_name}, fewer than {MIN_WINDOW_PERIODS}: the '
                                                 'resonances are poorly resolved or not found; run longer'))
        return issues
    return validate


def resonance_monitor_feature(probe_declaration, summary, description_note=''):
    """FeatureType 'resonance_monitor' of a time-stepped physics over its ResonanceProbe declaration."""
    band = probe_declaration.band
    if band not in BAND_KINDS:
        raise ValueError(f'band must be one of {BAND_KINDS}, got {band!r}')
    params = (probe_declaration.component, *_BAND_PARAMS[band], _start_param(probe_declaration.pulse_end_text))
    return FeatureType(FEATURE_NAME, 'Resonance finder', 'monitor', 'point', params=params, object_kinds=('point',),
                       max_entities=MAX_POINTS, result_kind='table', description=_description(band, description_note),
                       summary=summary, apply=_make_apply(probe_declaration),
                       result=_make_result(probe_declaration), validate=_make_validate(probe_declaration),
                       precisions=('single', 'double'))   # harmonic inversion fails on half-precision traces


def _trace_duration(context):
    """Resolved study duration (time_dependent) or max_duration (frequency_domain pulse, an upper bound) [s] or None."""
    study = context.spec.study
    try:
        params = context.registry.study_type(study.type).params
        values = resolve_values(params, study.settings if isinstance(study.settings, dict) else {}, context.parameters)
    except (KeyError, ValueError, TypeError):
        return None
    duration = values.get('duration', values.get('max_duration'))
    return duration if isinstance(duration, (int, float)) else None


def point_trace_items(context, mesh, trace_types, dt=None):
    """(build items, run items, host items) of the growable point traces (float32 device values, float64 host times)
    of the enabled features of trace_types over the study duration in steps of dt (None = mesh.dt, the EM step), plus
    one resonance analysis at a time."""
    traces = [feature for feature in context.physics_spec.features if feature.enabled and feature.type in trace_types]
    if not traces:
        return [], [], []
    duration = _trace_duration(context)
    dt = mesh.dt if dt is None else dt
    n_steps = int(duration / dt) + 1 if duration is not None and dt else 0
    capacity = PROBE_INITIAL_SAMPLES
    while capacity < n_steps:
        capacity *= 2
    points = {feature.name: _point_count(context, feature) for feature in traces}
    n_traces = sum(points.values())
    build = [(f'point traces ({n_traces} x float32, initial {PROBE_INITIAL_SAMPLES} samples)',
              n_traces * FLOAT32_BYTES * PROBE_INITIAL_SAMPLES)]
    run = [(f'point trace growth ({n_traces} x float32 to {capacity} samples)',
            n_traces * FLOAT32_BYTES * (capacity - PROBE_INITIAL_SAMPLES))]
    host = [(f'point trace times ({n_traces} x float64)', n_traces * FLOAT64_BYTES * capacity)]
    widest = max((points[feature.name] for feature in traces if feature.type == FEATURE_NAME), default=0)
    if widest:   # one analysis at a time: the monitor with the most points
        host.append((f'resonance analysis (harmonic inversion temporaries, {widest} point(s))',
                     analysis_host_bytes(n_steps, widest)))
    return build, run, host


def _point_count(context, feature):
    """Probe points of a point-trace feature: resonance_monitor one per selected point (named selections expanded),
    every other type 1."""
    if feature.type != FEATURE_NAME:
        return 1
    from .geometry import expand_entities
    try:
        return max(1, len(expand_entities(context.spec, feature.selection)))
    except (ValueError, KeyError, TypeError):   # a bad selection: validation reports it
        return 1
