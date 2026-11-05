"""Validation of the 'nk_fit' dispersion entries (A7): each material's fit as an Issue and the model's optical band
against its fit range (CPU; the fits are cached in materials.fitted_library_poles)."""
import math

from ...model.schema import PERMITTIVITY_PROPERTY_NAMES, Issue
from .definition_common import EM_FAMILY, FREQUENCY_DOMAIN_STUDY, _study_wavelengths
from .material_fit import ERROR_WARNING_LEVEL, LOSS_WARNING_LEVEL, NK_FIT_MODEL, nk_fit_settings

_RICKER_SIGNIFICANT = (0.40, 1.77)   # f / f_peak where the Ricker amplitude spectrum falls to 1/e of its peak
# band params of spectral monitors (dft / flux / mode / far-field) and of resonance_monitor
_MONITOR_BOUNDS = ('wavelength_start', 'wavelength_stop', 'wavelength_min', 'wavelength_max')


def nk_fit_issues(vctx):
    """Per material with an nk_fit entry: an error when it does not expand, else the fit summary (info; warning above
    the error levels) and a warning when the model band (model_band) leaves the fit range."""
    from ...materials import effective_material_data, fitted_library_poles   # lazy: materials imports physics.em
    issues, band = [], None
    for material in vctx.spec.materials:
        dispersion = material.dispersion if isinstance(material.dispersion, (list, tuple)) else ()
        entries = [pole for pole in dispersion if isinstance(pole, dict) and pole.get('model') == NK_FIT_MODEL]
        if not entries:
            continue
        path = ('material', material.name)
        try:
            effective_material_data(material.library, material.properties, material.dispersion)
            n_lorentz, start, stop = nk_fit_settings(entries[0])
            fit = fitted_library_poles(material.library, n_lorentz, start, stop)
        except ValueError as exc:
            issues.append(Issue('error', path, f'nk_fit: {exc}'))
            continue
        user_properties = material.properties if isinstance(material.properties, dict) else {}
        set_permittivity = [name for name in PERMITTIVITY_PROPERTY_NAMES if user_properties.get(name) is not None]
        if set_permittivity and fit.drude is None:   # with a Drude pole _material_issues already warns
            issues.append(Issue('warning', path, f"{', '.join(set_permittivity)} replace the fitted eps_inf "
                                                 f'({fit.eps_inf:.4g}) as background permittivity, so the material no '
                                                 'longer follows the table; leave them unset'))
        range_text = f'{start * 1e9:.4g}-{stop * 1e9:.4g} nm'
        if fit.relative_error > ERROR_WARNING_LEVEL or fit.loss_error > LOSS_WARNING_LEVEL:
            issues.append(Issue('warning', path, f"nk_fit of '{material.library}' over {range_text}: "
                                                 f"{fit.summary_text()}; raise n_lorentz or narrow the fit range"))
        else:
            issues.append(Issue('info', path, f"nk_fit of '{material.library}' over {range_text}: "
                                              f"{fit.summary_text()}"))
        if band is None:
            band = model_band(vctx) or ()
        if band and (band[0] < start * (1.0 - 1e-6) or band[1] > stop * (1.0 + 1e-6)):
            issues.append(Issue('warning', path,
                                f'the EM physics use vacuum wavelengths {band[0] * 1e9:.4g}-{band[1] * 1e9:.4g} nm '
                                f'(study wavelengths, wavelength settings, source spectra to 1/e, monitor ranges) but '
                                f'the nk_fit covers {range_text}: outside it the fitted poles extrapolate the table; '
                                f'widen wavelength_start / wavelength_stop'))
    return issues


def model_band(vctx):
    """(shortest, longest) vacuum wavelength [m] of the EM physics, or None: frequency_domain study wavelengths, the
    wavelength settings (other studies), source spectra to 1/e amplitude and monitor bands (spectral, resonance)."""
    wavelengths = []
    frequency_domain = vctx.study_type == FREQUENCY_DOMAIN_STUDY
    if frequency_domain:
        try:
            wavelengths += _study_wavelengths(vctx.spec, vctx.parameters, vctx.registry)
        except ValueError:   # reported at ('study',)
            pass
    for physics in vctx.spec.physics:
        try:
            physics_type = vctx.registry.physics_type(physics.type)
        except KeyError:
            continue
        if physics_type.family != EM_FAMILY:
            continue
        settings = vctx.values(physics_type.settings, physics.settings if isinstance(physics.settings, dict) else {})
        if settings and not frequency_domain and _positive(settings.get('wavelength')):
            wavelengths.append(settings['wavelength'])
        for feature in physics.features:
            if not feature.enabled or not isinstance(feature.params, dict):
                continue
            try:
                feature_type = physics_type.feature_type(feature.type)
            except KeyError:
                continue
            if feature_type.category not in ('source', 'monitor'):
                continue   # e.g. a gain line, phase-change reference or effective_index wavelength: no band
            values = vctx.values(feature_type.params, feature.params)
            if values:
                wavelengths += _feature_wavelengths(feature_type.category, values)
    wavelengths = [wavelength for wavelength in wavelengths if _positive(wavelength)]
    return (min(wavelengths), max(wavelengths)) if wavelengths else None


def _feature_wavelengths(category, values):
    """Vacuum wavelengths [m] bounding a feature's band: a source spectrum to 1/e amplitude (gaussian f0 (1 +-
    bandwidth), ricker 0.40-1.77 f_peak, cw f0), a monitor's wavelength_start / _stop or wavelength_min / _max."""
    if category == 'monitor':
        return [values[key] for key in _MONITOR_BOUNDS if _positive(values.get(key))]
    wavelengths = []
    wavelength = values.get('wavelength')
    if _positive(wavelength):
        waveform = values.get('waveform')
        if waveform == 'gaussian' and _positive(values.get('bandwidth')):
            low, high = 1.0 - values['bandwidth'], 1.0 + values['bandwidth']
        elif waveform == 'ricker':
            low, high = _RICKER_SIGNIFICANT
        else:
            low = high = 1.0
        wavelengths.append(wavelength / high)
        if low > 0.0:
            wavelengths.append(wavelength / low)
    return wavelengths


def _positive(value):
    """True for a finite real number > 0 (bools excluded)."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            and value > 0.0)
