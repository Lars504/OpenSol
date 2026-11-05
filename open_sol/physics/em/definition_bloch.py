"""Bloch phase of the FDTD types em_fdtd_ez / _hz (A6; the engines step a second, imaginary field set) and of the FDFD
types em_fdfd_ez / _hz (symmetry 2.2; phased wrap entries of the operator): the bloch_kx / bloch_ky settings, the
BoundaryConfig with bloch faces when k != 0 and their validation."""
from ...model.schema import Issue, Param, feature_path
from .definition_common import _q_accum_referenced, domain_requirements, quasi_static_domain_requirements
from .selftests_bloch import (BAND_DIAGRAM_DESCRIPTION, BAND_DIAGRAM_LABEL, FDFD_BAND_DIAGRAM_DESCRIPTION,
                              FDFD_BAND_DIAGRAM_LABEL, bloch_band_diagram_spec, fdfd_band_diagram_spec)

# Feature types that drive only the real field set (each period image repeats them with the Bloch phase) and the
# mode monitor, whose port mode is solved without the Bloch phase.
_REAL_SET_SOURCES = ('point_source', 'plane_wave_tfsf', 'mode_source')
_K_ZERO_PORTS = ('mode_source', 'mode_monitor')

_BLOCH_HELP = (
    'Bloch wave vector along {axis} in rad/m, used only when bc_{axis}_lo and bc_{axis}_hi are periodic: the fields '
    'repeat with the phase exp(-j k{axis} L) per period L (the meshed period; exp(+j omega t), so k{axis} > 0 for a '
    'wave travelling toward +{axis}); k{axis} and k{axis} + 2 pi / L are the same state. 0 = plain periodic. Any k != '
    '0 steps a second (imaginary) field set: field memory and time per step double. Line sources and gaussian beams '
    'carry exp(-j k.r) along the line, so a line along {axis} launches an oblique plane wave at sin(theta) = k{axis} / '
    '(n k0) from its normal (k0 = 2 pi / vacuum wavelength): at a fixed k{axis} the angle changes with frequency across '
    'a pulse, and frequencies below c k{axis} / n are evanescent. The calibrated amplitude holds at normal incidence: '
    'at angle theta an E_z sheet radiates amplitude / cos(theta), an in-plane sheet |E| = amplitude. Point, plane-wave '
    'box and mode sources drive only the real field set (every period image repeats them with the Bloch phase: good '
    'for band diagrams, where a point source and a resonance monitor per swept k give the bands). DFT-based monitors '
    '(flux, field map, far field, mode monitor) accumulate F_re + j F_im, i.e. the response at +k for every positive '
    'frequency; probes, field plots, expressions and couplings read the real set only. Absorbed power '
    '(track_absorption, q_accum, em_heat_joule) is refused. Typical: k{axis} = k0 sin(theta) of the incidence angle '
    'in vacuum, e.g. an expression 2*pi/lambda0*sin(theta) swept over theta.')
_BLOCH_TIP = ('Phase step of the wave from one period to the next along {axis}, in radians per metre, for periodic '
              'edges.\n\n'
              '0 means no phase step, as for light at normal incidence. For light arriving at an angle theta in '
              'vacuum use 2 pi sin(theta) / wavelength. Any other value doubles memory and time per step, and the '
              'absorbed power result is then not available.')


_FDFD_BLOCH_HELP = (
    'Bloch wave vector along {axis} in rad/m, used only when bc_{axis}_lo and bc_{axis}_hi are periodic: the solution '
    'is the Bloch wave F(x + L) = F(x) exp(-j k{axis} L) per period L (the meshed period; exp(+j omega t), so k{axis} > '
    '0 for a wave travelling toward +{axis}); k{axis} and k{axis} + 2 pi / L are the same state. 0 = plain periodic. '
    'The phase sits on the wrap entries of the operator: same unknowns and solve cost as k = 0. Every source is '
    'repeated in each period image with that phase. Line sources and gaussian beams carry exp(-j k.r) along the line (r '
    '= cell centres from the grid lo corner, as em_fdtd_*), so a line spanning the period launches an oblique plane '
    'wave at sin(theta) = k{axis} / (n k0) from its normal (k0 = 2 pi / vacuum wavelength); the calibrated amplitude '
    'holds at normal incidence: at angle theta an E_z sheet radiates amplitude / cos(theta), an in-plane sheet |E| = '
    'amplitude. A plane_wave_tfsf with bloch_ky != 0 becomes the oblique plane wave exp(-j (k_x x + ky y)), k_x from '
    'the discrete dispersion of the background (amplitude = E_z, or the in-plane |E|; ky at or beyond the background '
    'wavenumber is refused: evanescent; graded y cells are refused, graded x cells are fine); with bloch_kx it stays '
    'a +x wave repeated per period image. Mode sources, '
    'mode monitors and mode-power objectives on a line spanning a whole Bloch period are refused (the port mode would '
    'need the phase); port lines across the Bloch axis, or shorter than the period, are fine. Every result (absorbed '
    'power and em_heat_joule, probes, flux, far field, adjoint gradients) is that of the Bloch wave at +k. Band '
    'diagrams: an eigenfrequency study swept over k gives the band frequencies per k (template fdfd_band_diagram). '
    'Typical: k{axis} = k0 sin(theta) of the incidence angle in vacuum, e.g. an expression 2*pi/lambda0*sin(theta).')
_FDFD_BLOCH_TIP = ('Phase step of the wave from one period to the next along {axis}, in radians per metre, for '
                   'periodic edges.\n\n'
                   '0 means no phase step, as for light at normal incidence. For light arriving at an angle theta in '
                   'vacuum use 2 pi sin(theta) / wavelength; swept in an eigenfrequency study it gives a band diagram. '
                   'It costs nothing extra.')


def bloch_params(help_text=_BLOCH_HELP, tip=_BLOCH_TIP):
    """bloch_kx / bloch_ky settings (visible with periodic faces on that axis); defaults: the FDTD texts."""
    return tuple(Param(f'bloch_k{axis}', 'float', 0.0, label=f'Bloch wave vector {axis}', unit='rad/m',
                       visible_if=((f'bc_{axis}_lo', ('periodic',)),),
                       help=help_text.format(axis=axis), tip=tip.format(axis=axis))
                 for axis in ('x', 'y'))


def fdfd_bloch_params():
    """bloch_kx / bloch_ky settings of the FDFD types."""
    return bloch_params(_FDFD_BLOCH_HELP, _FDFD_BLOCH_TIP)


def bloch_active_axes(settings_values):
    """Axes with periodic faces and a numeric Bloch k != 0."""
    return tuple(axis for axis in ('x', 'y')
                 if settings_values[f'bc_{axis}_lo'] == 'periodic' and settings_values[f'bc_{axis}_hi'] == 'periodic'
                 and isinstance(settings_values.get(f'bloch_k{axis}'), (int, float))
                 and settings_values[f'bloch_k{axis}'] != 0.0)


def fdtd_domain_requirements(settings_values):
    """domain_requirements with bloch faces and bloch_kx / bloch_ky on the periodic axes whose k != 0."""
    return _with_bloch_faces(domain_requirements(settings_values), settings_values)


def fdfd_domain_requirements(settings_values):
    """quasi_static_domain_requirements with bloch faces and bloch_kx / bloch_ky on the periodic axes whose k != 0."""
    return _with_bloch_faces(quasi_static_domain_requirements(settings_values), settings_values)


def _with_bloch_faces(requirements, settings_values):
    """`requirements` with its boundary_config's periodic faces turned bloch where k != 0 (validated again)."""
    config = requirements.boundary_config
    for axis in bloch_active_axes(settings_values):
        setattr(config, f'{axis}_lo', 'bloch')
        setattr(config, f'{axis}_hi', 'bloch')
        setattr(config, f'bloch_k{axis}', float(settings_values[f'bloch_k{axis}']))
    config._validate()
    return requirements


def bloch_issues(vctx, physics_spec, settings_values):
    """Bloch k on a non-periodic axis (warning, ignored), absorbed power with k != 0 (error), features that drive only
    the real field set (info) or solve their port mode at k = 0 (warning)."""
    path = ('physics', physics_spec.name)
    issues = _ignored_k_issues(path, settings_values)
    if not bloch_active_axes(settings_values):
        return issues
    if settings_values['track_absorption'] or _q_accum_referenced(vctx.spec, physics_spec.name, vctx.registry):
        issues.append(Issue('error', path, 'absorbed power (track_absorption, q_accum, absorbed_power, em_heat_joule, '
                                           'em_semiconductor_generation) '
                                           'is not available with a Bloch wave vector k != 0: the complex field mixes '
                                           'the +k and -k responses; set bloch_kx / bloch_ky to 0'))
    for feature in physics_spec.features:
        if not feature.enabled:
            continue
        if feature.type in _REAL_SET_SOURCES:
            issues.append(Issue('info', feature_path(physics_spec, feature),
                                f"with Bloch k != 0 the {feature.type} drives the real field set only: every period "
                                'image repeats it with the Bloch phase (a phased array, not an oblique plane wave; '
                                'use a line source across the periodic axis for oblique incidence)'))
        if feature.type in _K_ZERO_PORTS:
            issues.append(Issue('warning', feature_path(physics_spec, feature),
                                f'the {feature.type} solves its port mode without the Bloch phase, so a port line '
                                'along a Bloch axis is wrong for k != 0'))
    return issues


def _ignored_k_issues(path, settings_values):
    """Warnings for a Bloch k set on an axis whose faces are not periodic (it is ignored)."""
    issues = []
    for axis in ('x', 'y'):
        k = settings_values.get(f'bloch_k{axis}')
        if isinstance(k, (int, float)) and k != 0.0 and settings_values[f'bc_{axis}_lo'] != 'periodic':
            issues.append(Issue('warning', path, f'bloch_k{axis} = {k:.4g} rad/m is ignored: bc_{axis}_lo / '
                                                 f'bc_{axis}_hi are not periodic'))
    return issues


# FDFD feature types whose line is a port (slab mode solved on it): refused on a line spanning a whole Bloch period.
FDFD_PORT_FEATURES = ('mode_source', 'mode_monitor')


def is_fdfd_port_feature(feature):
    """Whether an FDFD feature spec solves a port mode on its line (mode source / monitor, mode-power objective)."""
    return feature.type in FDFD_PORT_FEATURES or (feature.type == 'adjoint_objective'
                                                  and feature.params.get('quantity', 'flux') != 'flux')


def fdfd_bloch_issues(vctx, physics_spec, settings_values):
    """FDFD: Bloch k on a non-periodic axis (warning); with k != 0 infos for point sources and a +x TFSF along a Bloch
    x axis (repeated per period image, no oblique wave). Port lines spanning a period: definition_fdfd."""
    path = ('physics', physics_spec.name)
    issues = _ignored_k_issues(path, settings_values)
    axes = bloch_active_axes(settings_values)
    for feature in physics_spec.features:
        if not feature.enabled:
            continue
        if feature.type == 'point_source' and axes:
            issues.append(Issue('info', feature_path(physics_spec, feature),
                                'with Bloch k != 0 every period image repeats this point source with the Bloch '
                                'phase (a phased array; use a line source across the periodic axis for an oblique '
                                'plane wave)'))
        if feature.type == 'plane_wave_tfsf' and 'x' in axes:
            issues.append(Issue('info', feature_path(physics_spec, feature),
                                'with bloch_kx != 0 the +x TFSF plane wave stays at normal incidence and every period '
                                'image along x repeats it with the Bloch phase (bloch_ky tilts it instead)'))
    return issues


def with_bloch_validation(validate_physics):
    """validate(vctx, physics_spec, settings_values) of `validate_physics` plus bloch_issues."""
    def validate(vctx, physics_spec, settings_values):
        return list(validate_physics(vctx, physics_spec, settings_values)) + bloch_issues(vctx, physics_spec,
                                                                                         settings_values)
    return validate


def _selftest_cases():
    """The Bloch SelfTestCases (imported on the first registry.selftests(), not by get_registry)."""
    from .selftests_bloch import BLOCH_SELFTEST_CASES
    return BLOCH_SELFTEST_CASES


def register(registry):
    """Register the 'bloch_band_diagram' (FDTD, needs resonance_monitor) and 'fdfd_band_diagram' (FDFD eigenfrequency)
    templates and the Bloch selftest loader (FDTD and FDFD)."""
    registry.register_template(
        'bloch_band_diagram', BAND_DIAGRAM_LABEL, BAND_DIAGRAM_DESCRIPTION, bloch_band_diagram_spec,
        summary='Band diagram of a photonic crystal of dielectric rods: one unit cell, the phase step between cells '
                'swept from the centre to the edge of the first zone, and the resonance finder listing the band '
                'frequencies at each step.')
    registry.register_template(
        'fdfd_band_diagram', FDFD_BAND_DIAGRAM_LABEL, FDFD_BAND_DIAGRAM_DESCRIPTION, fdfd_band_diagram_spec,
        summary='Band diagram of the same photonic crystal solved directly: one unit cell, the phase step between '
                'cells swept across the first zone, and the eigenfrequency study listing the band frequencies at '
                'each step.')
    registry.register_selftests(_selftest_cases)
