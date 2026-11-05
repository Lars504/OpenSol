"""Validation of em_fdtd_ez / _hz in a frequency_domain study (harmonic FDTD, harmonic_fdtd.py): one shared pulsed
waveform, study wavelengths inside its band (accumulator memory: memory_estimates, E8)."""
from ...core.constants import C0
from ...model.schema import Issue
from .definition_common import FREQUENCY_DOMAIN_STUDY, _source_band, _source_values, _study_wavelengths

HARMONIC_FDTD_TEXT = ('In a frequency_domain study it runs one pulse and gives phasor fields per study wavelength '
                      '(harmonic FDTD, see that study): every enabled source must share one gaussian or ricker waveform '
                      '(same wavelength and bandwidth), the study wavelengths should lie inside its band, and no '
                      'coupling may read this physics.')


def make_validate_harmonic_fdtd(validate_physics):
    """validate_physics plus harmonic_fdtd_issues in a frequency_domain study."""
    def validate(vctx, physics_spec, settings_values):
        issues = list(validate_physics(vctx, physics_spec, settings_values))
        if vctx.study_type == FREQUENCY_DOMAIN_STUDY:
            issues.extend(harmonic_fdtd_issues(vctx, physics_spec))
        return issues
    return validate


def _waveform_signature(values):
    """(waveform, wavelength, bandwidth | None) that must agree between the sources of one physics."""
    waveform = values.get('waveform')
    return waveform, values['wavelength'], values.get('bandwidth') if waveform == 'gaussian' else None


def harmonic_fdtd_issues(vctx, physics_spec):
    """Errors for no source, mixed waveforms or cw; warnings for study wavelengths outside the source band (amplitude
    < 1e-2 of its peak)."""
    path = ('physics', physics_spec.name)
    physics_type = vctx.registry.physics_type(physics_spec.type)
    sources = _source_values(vctx, physics_spec, physics_type)
    if not sources:
        return [Issue('error', path, 'a frequency_domain study runs an FDTD physics as one pulse, normalised by its '
                                     'source spectrum: add an enabled source (gaussian or ricker waveform)')]
    signatures = {}
    for feature, values in sources:
        signatures.setdefault(_waveform_signature(values), []).append(feature.name)
    if len(signatures) > 1:
        groups = '; '.join(f"{', '.join(names)}: {waveform} at {wavelength:.4g} m"
                           + (f' (bandwidth {bandwidth:g})' if bandwidth is not None else '')
                           for (waveform, wavelength, bandwidth), names in signatures.items())
        return [Issue('error', path, f'in a frequency_domain study every enabled source of an FDTD physics must share '
                                     f'one waveform, wavelength and bandwidth (the response is divided by its '
                                     f'spectrum); found {groups}')]
    first_values = sources[0][1]
    if first_values.get('waveform') == 'cw':
        return [Issue('error', path, "a frequency_domain study needs gaussian or ricker FDTD sources: a 'cw' source "
                                     'never decays, so the pulse cannot end; use waveform gaussian (bandwidth covering '
                                     'the study wavelengths) or ricker')]
    issues = []
    try:
        wavelengths = _study_wavelengths(vctx.spec, vctx.parameters, vctx.registry)
    except ValueError:   # reported at ('study',)
        return issues
    band = _source_band(first_values, None)
    if band is not None and wavelengths:
        f_lo, f_hi = band
        outside = [wavelength for wavelength in wavelengths if not f_lo <= C0 / wavelength <= f_hi]
        if outside:
            issues.append(Issue('warning', path, f'{len(outside)} of {len(wavelengths)} study wavelengths (between '
                                                 f'{min(outside):.4g} and {max(outside):.4g} m) lie outside the '
                                                 f'source spectrum ({C0 / f_hi:.4g} to '
                                                 f"{f'{C0 / f_lo:.4g}' if f_lo > 0 else 'inf'} m at 1e-2 of its peak "
                                                 'amplitude), so their harmonic fields are mostly noise; move the '
                                                 'source wavelength or widen its bandwidth'))
    return issues   # accumulator memory: memory_estimates.fdtd_memory_estimate (E8 run items, budget check)
