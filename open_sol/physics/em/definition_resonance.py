"""EM resonance_monitor (A6) over the shared row of model.definition_resonance: the ResonanceProbe declarations of the
FDTD EM types (em_fdtd_ez / _hz here, em_fdtd_3d through em_resonance_probe), the EM source pulse ends, and register()
for the 'disk_resonator' template and the resonance self-tests (numerics in core.resonance)."""
from ...model.definition_helpers import validate_points_inside
from ...model.definition_resonance import ResonanceProbe
from ...model.definition_resonance import resonance_monitor_feature as shared_resonance_feature
from ...model.schema import Param
from .definition_common import (EZ_PROBE_COMPONENT_TIP, HZ_PROBE_COMPONENT_TIP, PROBE_COMPONENT_CHOICE_LABELS,
                                _pulse_end, _source_values, apply_field_probe)
from .selftests_resonance import DISK_RESONATOR_DESCRIPTION, DISK_RESONATOR_LABEL, disk_resonator_spec

_COMPONENT_HELP = {
    'ez': 'ez in V/m at the centre of the cell containing the point, or hx / hy in A/m on its +y / +x face (half a '
          'time step earlier; the reference time accounts for it).',
    'hz': 'hz in A/m at the centre of the cell containing the point (half a time step earlier; the reference time '
          'accounts for it), or ex / ey in V/m on its +y / +x face.',
}
_COMPONENT_TIPS = {'ez': EZ_PROBE_COMPONENT_TIP, 'hz': HZ_PROBE_COMPONENT_TIP}
_H_SAMPLE_DELAY_STEPS = 0.5   # H is stepped half a step before the recorded time of the probe sample
_DELAYED_COMPONENTS = {'ez': ('hx', 'hy'), 'hz': ('hz',)}
RESONANCE_SUMMARY = ('Finds the resonances of a structure that rings after a short pulse: their wavelengths, quality '
                     'factors and strengths, from the field recorded at one or more points. Put the points where the '
                     'modes are strong; several points also catch a mode that has a node at one of them.')


def em_source_waveforms(vctx, physics_spec):
    """(pulse ends 2 t0 [s] of the enabled gaussian / ricker sources, names of the enabled cw sources)."""
    sources = _source_values(vctx, physics_spec, vctx.registry.physics_type(physics_spec.type))
    pulse_ends = [end for _, values in sources if (end := _pulse_end(values)) is not None]
    cw_names = [feature.name for feature, values in sources if values.get('waveform') == 'cw']
    return pulse_ends, cw_names


def _engine_time_step(solver):
    return float(solver.engine.dt)


def _source_temporals(solver):
    return [getattr(source, 'temporal', None) for source in solver.sources]


def _renderer_unit(solver, component):
    return solver.renderer().field_info(component).unit


def em_resonance_probe(component, delayed_components, add_probe, validate_point=validate_points_inside):
    """ResonanceProbe of an FDTD EM physics: vacuum-wavelength band, engine dt, the sources' temporals, renderer units;
    the H components in delayed_components are sampled half a step before their recorded time."""
    return ResonanceProbe(component=component, add_probe=add_probe, time_step=_engine_time_step,
                          waveforms=_source_temporals, unit=_renderer_unit, source_waveforms=em_source_waveforms,
                          band='wavelength',
                          sample_delay_steps={name: _H_SAMPLE_DELAY_STEPS for name in delayed_components},
                          validate_point=validate_point)


def _component_param(polarization):
    choices = tuple(choice for choice, _ in PROBE_COMPONENT_CHOICE_LABELS[polarization])
    return Param('component', 'choice', choices[0], label='Field component', choices=choices,
                 choice_labels=PROBE_COMPONENT_CHOICE_LABELS[polarization], tip=_COMPONENT_TIPS[polarization],
                 help=f'Recorded field: {_COMPONENT_HELP[polarization]}')


def resonance_monitor_feature(polarization):
    """FeatureType 'resonance_monitor' of em_fdtd_<polarization>."""
    probe = em_resonance_probe(_component_param(polarization), _DELAYED_COMPONENTS[polarization], apply_field_probe)
    return shared_resonance_feature(probe, RESONANCE_SUMMARY)


def _selftest_cases():
    """The resonance SelfTestCases (imported on the first registry.selftests(), not by get_registry)."""
    from .selftests_resonance import RESONANCE_SELFTEST_CASES
    return RESONANCE_SELFTEST_CASES


def register(registry):
    """Register the 'disk_resonator' template and the resonance selftest loader (after the FDTD types)."""
    registry.register_template(
        'disk_resonator', DISK_RESONATOR_LABEL, DISK_RESONATOR_DESCRIPTION, disk_resonator_spec,
        summary='A silicon disk in air rings after a short light pulse near 1550 nm. The resonance finder lists the '
                'whispering-gallery resonances with their quality factors.')
    registry.register_selftests(_selftest_cases)
