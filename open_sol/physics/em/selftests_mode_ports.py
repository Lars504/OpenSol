"""Model-run self-tests of the mode source and mode monitor (Fresnel plane-wave ports, silicon slab, directional
coupler FDTD vs FDFD) and the 'directional_coupler' template (em_fdtd_hz); no physics imports, every spec is built
fresh on each call."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import PERIODIC_Y, analytic_slab_index, float_list, glass_half_space, vertical_line
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           StudySpec)

_PHYSICS_NAME = 'em'
_PHYSICS_TYPES = {'ez': 'em_fdtd_ez', 'hz': 'em_fdtd_hz'}
_SOURCE = 'source'
_UPSTREAM = 'upstream'
_DOWNSTREAM = 'downstream'

# em_hz_mode_port_fresnel (mode_port_fresnel_case, also wrapped by selftests_backend): y-periodic strip 5 x 0.1 um, CPML along x, uniform 10 nm mesh, window
# cell k spanning [10 k, 10 (k + 1)] nm along x. Vertical lines at cell centres span y 0..0.1 um, all 10 rows of the
# periodic axis (a periodic solve: mode 1 is the uniform plane wave, n_eff = sqrt(eps_r)): upstream monitor x = 0.505 um
# (cell 50), source x = 1.005 um (cell 100: total field from cell 100 on, interface on its lo face at 1.0 um),
# downstream monitor x = 3.505 um (cell 350, measuring face 3.51 um). The 'glass' run fills x >= 2.5 um (cells 250..,
# continued into the x_hi PML through the clamped centres) with eps_r = 4. Run budget per run: 530 x 10 cells, ~2300
# steps, 2 monitors x 5 wavelengths.
_FRESNEL_SIZE = (5e-6, 0.1e-6)            # m
_FRESNEL_CELL = 10e-9                     # m
# s: the pulse ends by ~20 fs and has passed both monitors by ~32 fs; above the validation unit-slip limit 2 t0 +
# diagonal n_max / C0 = 52.5 fs of the glass run
_FRESNEL_DURATION = 55e-15
_FRESNEL_LINES = ((_UPSTREAM, 0.505e-6), (_SOURCE, 1.005e-6), (_DOWNSTREAM, 3.505e-6))   # (feature, x [m])
_FRESNEL_INTERFACE = 2.5e-6               # m
_FRESNEL_SOURCE = {'mode_number': 1, 'direction': 'positive', 'waveform': 'gaussian', 'wavelength': 1e-6,
                   'bandwidth': 0.5, 'power': 1.0}
_FRESNEL_SPECTRAL = {'wavelength_start': 0.9e-6, 'wavelength_stop': 1.1e-6, 'n_wavelengths': 5}
_FRESNEL_REPORTED_WAVELENGTH = 1e-6       # m, 'measured' / 'expected' are the reflectance here
_GLASS_EPS_R = 4.0
_GLASS_INDEX = math.sqrt(_GLASS_EPS_R)
_REFLECTANCE = ((_GLASS_INDEX - 1.0) / (_GLASS_INDEX + 1.0)) ** 2   # 1/9
_TRANSMITTANCE = 4.0 * _GLASS_INDEX / (_GLASS_INDEX + 1.0) ** 2      # 8/9
_FRESNEL_TRANSFER_TOLERANCE = 0.01        # absolute power fraction
_FRESNEL_LEAK_TOLERANCE = 1e-3            # power fraction of a direction that should carry nothing

# em_mode_port_slab / em_hz_mode_port_slab: 4 x 2 um all-CPML box, uniform 20 nm mesh, window cell k spanning
# [20 k, 20 (k + 1)] nm. A silicon slab fills y in [0.86, 1.14) um = rows 43..56 (edges on cell faces, so the grid slab
# is exactly 0.28 um thick, centred at y = 1 um) along the whole x range and into the x PML; silica everywhere else.
# Vertical lines at cell centres span y 0..2 um, so both ends lie on cpml window edges and every solve continues through
# the 15 PML rows on each side: upstream monitor x = 0.51 um (cell 25), source x = 1.01 um (cell 50), downstream monitor
# x = 3.01 um (cell 150). Run budget: 230 x 130 cells, ~2350 steps, 2 monitors x 5 wavelengths.
_SLAB_SIZE = (4e-6, 2e-6)                 # m
_SLAB_CELL = 20e-9                        # m
# s: pulse 2 t0 ~ 49 fs, group delay to the downstream monitor < 35 fs, plus tail; validation limit 2 t0 + diagonal
# n_max / C0 = 101 fs
_SLAB_DURATION = 110e-15
_SLAB_Y_RANGE = (0.86e-6, 1.14e-6)        # m
_SLAB_THICKNESS = 0.28e-6                 # m
_SLAB_LINES = ((_UPSTREAM, 0.51e-6), (_SOURCE, 1.01e-6), (_DOWNSTREAM, 3.01e-6))   # (feature, x [m])
_SILICON_EPS_R = 12.08
_SILICA_EPS_R = 2.085
_SLAB_SOURCE = {'mode_number': 1, 'direction': 'positive', 'waveform': 'gaussian', 'wavelength': 1.55e-6,
                'bandwidth': 0.3, 'power': 1.0}
_SLAB_SPECTRAL = {'wavelength_start': 1.5e-6, 'wavelength_stop': 1.6e-6, 'n_wavelengths': 5}
_SLAB_REPORTED_WAVELENGTH = 1.55e-6       # m, 'measured' / 'expected' are n_eff here
_SLAB_FORWARD_TOLERANCE = 0.02            # absolute power fraction
_SLAB_LEAK_TOLERANCE = 2e-3               # power fraction of a direction that should carry nothing
_SLAB_INDEX_TOLERANCE = 0.01              # relative n_eff

# directional_coupler template (em_fdtd_hz): uniform 20 nm mesh, window cell k spanning [20 k, 20 (k + 1)] nm; strip
# edges lie on cell faces (the S-bend polygons aside) and lines at cell centres. Strip A (input / through) fills y in
# [0.6, 0.9) um over the whole length; strip B fills y in [1.1, 1.4) um (0.2 um gap) for x in [4.5, 8.5) um and bends
# away through cosine S-bends (3 um long, 0.8 um offset) to y in [1.9, 2.2) um at both ends. The strip-A lines
# (reflection x = 0.41 um, source x = 0.81 um, through x = 12.51 um) span y from 0 (cpml window edge: the solve
# continues through the PML) to 1.39 um (rows up to 69: a metal end at 1.40 um, 0.5 um from strip A and from the
# separated strip B, where the TM0 tail is ~4e-3 of its peak, below the 1e-2 truncation warning); the cross line
# (x = 12.51 um) spans 1.41 um (row 70, metal end at 1.40 um) to 2.8 um (cpml window edge, 0.6 um above strip B). Run
# budget: 680 x 170 cells, ~6400 steps, 3 monitors x 21 wavelengths.
DIRECTIONAL_COUPLER_LABEL = 'Directional coupler (in-plane electric field)'
DIRECTIONAL_COUPLER_DESCRIPTION = ('EM (H_z polarisation): two 0.3 um silicon strips in silica with a 4 um long, '
                                   '0.2 um gap coupling section and cosine S-bends (uniform 20 nm mesh); a 1.55 um '
                                   'gaussian mode source in strip A and mode monitors give the through, cross and '
                                   'reflected power fractions (|S|^2) and phases, 1.45-1.65 um.')
_COUPLER_SIZE = (13e-6, 2.8e-6)           # m
_COUPLER_STRIP_A = (0.6e-6, 0.9e-6)       # m, y range
_COUPLER_STRIP_B_NEAR = (1.1e-6, 1.4e-6)  # m, y range along the coupling section
_COUPLER_STRIP_B_FAR = (1.9e-6, 2.2e-6)   # m, y range at the ports
_COUPLER_BREAKS = (1.5e-6, 4.5e-6, 8.5e-6, 11.5e-6)   # m, x: bend in, coupling section, bend out
_COUPLER_BEND_SEGMENTS = 16
_COUPLER_LINES = (('reflection', 0.41e-6, 0.0, 1.39e-6), (_SOURCE, 0.81e-6, 0.0, 1.39e-6),
                  ('through', 12.51e-6, 0.0, 1.39e-6), ('cross', 12.51e-6, 1.41e-6, 2.8e-6))   # (feature, x, y0, y1)
_COUPLER_SOURCE = {'mode_number': 1, 'direction': 'positive', 'waveform': 'gaussian', 'wavelength': 1.55e-6,
                   'bandwidth': 0.3, 'power': 1.0}
_COUPLER_SPECTRAL = {'wavelength_start': 1.45e-6, 'wavelength_stop': 1.65e-6, 'n_wavelengths': 21}
_COUPLER_DURATION = 300e-15               # s: pulse ~50 fs plus the group delay over ~12 um of strip
# em_hz_directional_coupler_fdtd_vs_fdfd: the template geometry and source in em_fdtd_hz with its monitors at the FDFD
# wavelengths and a shorter run, and the same geometry in em_fdfd_hz (frequency_domain study at 1.5, 1.55, 1.6 um, FDFD
# unknowns 680 x 170); port fractions compared at those wavelengths.
_COUPLER_FDFD_SOURCE = {'mode_number': 1, 'direction': 'positive', 'power': 1.0, 'phase': 0.0}
_COUPLER_FDFD_STUDY = {'wavelength_start': 1.5e-6, 'wavelength_stop': 1.6e-6, 'n_wavelengths': 3}
# s: pulse 2 t0 ~ 49 fs plus the group delay ~170 fs over ~12 um of strip, plus tail; validation limit 2 t0 + diagonal
# n_max / C0 = 203 fs
_COUPLER_CHECK_DURATION = 260e-15
_COUPLER_REPORTED_WAVELENGTH = 1.55e-6    # m, 'measured' / 'expected' are the split ratio cross / (through + cross)
_COUPLER_PORTS = (('through', 'fraction_positive'), ('cross', 'fraction_positive'),
                  ('reflection', 'fraction_negative'))
_COUPLER_AGREEMENT_TOLERANCE = 0.02       # absolute power fraction, FDTD vs FDFD


# ---- specs ----
def _port_feature(feature_name, line_name, source_params, spectral):
    """mode_source feature for _SOURCE, else a mode_monitor (mode 1) on the line."""
    if feature_name == _SOURCE:
        return FeatureSpec(feature_name, 'mode_source', [line_name], dict(source_params))
    return FeatureSpec(feature_name, 'mode_monitor', [line_name], dict(spectral, mode_number=1))


def _port_lines(lines, y_end, source_params, spectral):
    """(geometry, features): a vertical line '<feature>_line' spanning y 0..y_end at each (feature, x) of lines, with
    the mode source on _SOURCE and mode monitors on the others."""
    geometry, features = [], []
    for feature_name, x in lines:
        line_name = f'{feature_name}_line'
        geometry.append(vertical_line(line_name, x, 0.0, y_end))
        features.append(_port_feature(feature_name, line_name, source_params, spectral))
    return geometry, features


def _fresnel_spec(polarization, glass):
    """y-periodic strip with the mode source and the two mode monitors; glass adds the eps_r = 4 half-space."""
    size_x, size_y = _FRESNEL_SIZE
    geometry, features = _port_lines(_FRESNEL_LINES, size_y, _FRESNEL_SOURCE, _FRESNEL_SPECTRAL)
    materials = []
    if glass:
        half_space, glass_material = glass_half_space(_FRESNEL_INTERFACE, size_x, size_y, _GLASS_EPS_R)
        geometry.append(half_space)
        materials.append(glass_material)
    physics_type = _PHYSICS_TYPES[polarization]
    return ModelSpec(
        title=f"{physics_type} mode ports: {'vacuum / glass interface' if glass else 'vacuum'}",
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=geometry,
        materials=materials,
        physics=[PhysicsSpec(_PHYSICS_NAME, physics_type, settings=dict(PERIODIC_Y), features=features)],
        mesh=MeshSpec(mode='uniform', dx=_FRESNEL_CELL),
        study=StudySpec(type='time_dependent', settings={'duration': _FRESNEL_DURATION}))


def _fresnel_specs(polarization):
    """'vacuum' then 'glass': same mode source and monitors."""
    return {'vacuum': _fresnel_spec(polarization, False), 'glass': _fresnel_spec(polarization, True)}


def slab_waveguide_spec(physics_type, source_params, spectral, study):
    """Silicon slab in silica, all-CPML, mode source and two mode monitors (spectral settings) spanning the window
    height; shared with selftests_fdfd."""
    size_x, size_y = _SLAB_SIZE
    y_lo, y_hi = _SLAB_Y_RANGE
    geometry, features = _port_lines(_SLAB_LINES, size_y, source_params, spectral)
    return ModelSpec(
        title=f'{physics_type} mode ports: silicon slab waveguide',
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=[GeometrySpec('slab', 'rectangle', {'x0': 0.0, 'y0': y_lo, 'x1': size_x, 'y1': y_hi}), *geometry],
        materials=[MaterialSpec('silica', properties={'eps_r': _SILICA_EPS_R}, selection=['all']),
                   MaterialSpec('silicon', properties={'eps_r': _SILICON_EPS_R}, selection=['slab'])],
        physics=[PhysicsSpec(_PHYSICS_NAME, physics_type, features=features)],
        mesh=MeshSpec(mode='uniform', dx=_SLAB_CELL),
        study=study)


def _slab_specs(polarization):
    """'run': the FDTD slab waveguide with the gaussian mode source, time_dependent."""
    return {'run': slab_waveguide_spec(_PHYSICS_TYPES[polarization], _SLAB_SOURCE, _SLAB_SPECTRAL,
                                       StudySpec(type='time_dependent', settings={'duration': _SLAB_DURATION}))}


def _rectangle(name, x0, y_range, x1):
    return GeometrySpec(name, 'rectangle', {'x0': x0, 'y0': y_range[0], 'x1': x1, 'y1': y_range[1]})


def _s_bend_points(x_start, x_end, start_range, end_range, n_segments):
    """Polygon vertices [[x, y], ...] of a strip moving from the y range start_range at x_start to end_range at x_end
    along a cosine S-bend centreline, edges offset perpendicular to it, rounded to 0.1 nm."""
    # centreline y_start + (y_end - y_start) (1 - cos(pi s)) / 2, s = (x - x_start) / length; upper edge left to right,
    # then the lower edge back
    y_start, y_end = 0.5 * sum(start_range), 0.5 * sum(end_range)
    half_width = 0.5 * (start_range[1] - start_range[0])
    length = x_end - x_start
    upper, lower = [], []
    for index in range(n_segments + 1):
        s = index / n_segments
        x = x_start + s * length
        y = y_start + 0.5 * (y_end - y_start) * (1.0 - math.cos(math.pi * s))
        angle = math.atan(0.5 * math.pi * (y_end - y_start) / length * math.sin(math.pi * s))
        offset_x, offset_y = half_width * math.sin(angle), half_width * math.cos(angle)
        upper.append([round(x - offset_x, 10), round(y + offset_y, 10)])
        lower.append([round(x + offset_x, 10), round(y - offset_y, 10)])
    return upper + lower[::-1]


def directional_coupler_spec(physics_type='em_fdtd_hz', source_params=_COUPLER_SOURCE, spectral=_COUPLER_SPECTRAL,
                             study=None):
    """Template 'directional_coupler' (defaults: em_fdtd_hz, time_dependent): a mode source in strip A of a two-strip
    silicon coupler in silica; mode monitors on the reflected (upstream), through and cross ports."""
    size_x, size_y = _COUPLER_SIZE
    bend_in_start, section_start, section_end, bend_out_end = _COUPLER_BREAKS
    strips = [
        _rectangle('strip_a', 0.0, _COUPLER_STRIP_A, size_x),
        _rectangle('strip_b_in', 0.0, _COUPLER_STRIP_B_FAR, bend_in_start),
        GeometrySpec('bend_in', 'polygon', {'points': _s_bend_points(
            bend_in_start, section_start, _COUPLER_STRIP_B_FAR, _COUPLER_STRIP_B_NEAR, _COUPLER_BEND_SEGMENTS)}),
        _rectangle('strip_b_coupler', section_start, _COUPLER_STRIP_B_NEAR, section_end),
        GeometrySpec('bend_out', 'polygon', {'points': _s_bend_points(
            section_end, bend_out_end, _COUPLER_STRIP_B_NEAR, _COUPLER_STRIP_B_FAR, _COUPLER_BEND_SEGMENTS)}),
        _rectangle('strip_b_out', bend_out_end, _COUPLER_STRIP_B_FAR, size_x),
    ]
    lines, features = [], []
    for feature_name, x, y_start, y_end in _COUPLER_LINES:
        line_name = f'{feature_name}_line'
        lines.append(vertical_line(line_name, x, y_start, y_end))
        features.append(_port_feature(feature_name, line_name, source_params, spectral))
    if study is None:
        study = StudySpec(type='time_dependent', settings={'duration': _COUPLER_DURATION})
    return ModelSpec(
        title='Directional coupler (H_z polarisation)',
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=strips + lines,
        materials=[MaterialSpec('cladding', library='silica', selection=['all']),
                   MaterialSpec('silicon', library='silicon', selection=[strip.name for strip in strips])],
        physics=[PhysicsSpec(_PHYSICS_NAME, physics_type, features=features)],
        mesh=MeshSpec(mode='uniform', dx=20e-9),
        study=study)


def _coupler_specs():
    """'fdtd': the directional_coupler template with its monitors at the FDFD wavelengths, run for
    _COUPLER_CHECK_DURATION; 'fdfd': its geometry in em_fdfd_hz at 1.5, 1.55, 1.6 um."""
    return {'fdtd': directional_coupler_spec(
                'em_fdtd_hz', _COUPLER_SOURCE, _COUPLER_FDFD_STUDY,
                StudySpec(type='time_dependent', settings={'duration': _COUPLER_CHECK_DURATION})),
            'fdfd': directional_coupler_spec('em_fdfd_hz', _COUPLER_FDFD_SOURCE, {},
                                             StudySpec(type='frequency_domain', settings=dict(_COUPLER_FDFD_STUDY)))}


# ---- analytic slab modes ----
def _slab_fundamental_index(wavelength, polarization):
    """n_eff of the fundamental (even) mode of the silicon / silica slab at the vacuum wavelength."""
    return analytic_slab_index(2.0 * math.pi / wavelength, _SLAB_THICKNESS, _SILICON_EPS_R, _SILICA_EPS_R,
                               polarization)


# ---- checks ----
def _mode_spectra(results, feature):
    """(wavelengths [m], fraction_positive, fraction_negative, Re n_eff) float64 arrays of a source-normalised mode
    monitor; ValueError when it carries no fractions."""
    result = results.result(_PHYSICS_NAME, feature)
    if not result.get('normalized'):
        raise ValueError(f"mode monitor '{feature}' has no source-normalised fractions (the physics needs exactly one "
                         'enabled mode source)')
    series = result['series']
    return tuple(np.asarray(values, dtype=np.float64) for values in (
        result['wavelength'], series['fraction_positive'], series['fraction_negative'], result['n_eff']))


def _check_fresnel(results_by_name):
    """Mode power fractions of both runs vs 1 (vacuum transmission), 1/9 (reflection), 8/9 (transmission) and 0 (every
    other direction); error in tolerance units."""
    spectra = {(run, feature): _mode_spectra(results_by_name[run], feature)
               for run in ('vacuum', 'glass') for feature in (_UPSTREAM, _DOWNSTREAM)}
    # (run, monitor, direction index 0 = positive / 1 = negative) -> expected fraction; the others should vanish
    expected = {('vacuum', _DOWNSTREAM, 0): 1.0, ('glass', _DOWNSTREAM, 0): _TRANSMITTANCE,
                ('glass', _UPSTREAM, 1): _REFLECTANCE}
    transfer_errors, leaks, details = [], [], {}
    for (run, feature), (_, positive, negative, _) in spectra.items():
        for direction_index, (direction, fraction) in enumerate((('positive', positive), ('negative', negative))):
            target = expected.get((run, feature, direction_index))
            if target is None:
                leaks.append(np.abs(fraction))
            else:
                transfer_errors.append(np.abs(fraction - target))
            details[f'{run}_{feature}_fraction_{direction}'] = float_list(fraction)
    err_transfer = float(np.max(transfer_errors))   # NaN propagates
    err_leak = float(np.max(leaks))
    error = float(np.max([err_transfer / _FRESNEL_TRANSFER_TOLERANCE, err_leak / _FRESNEL_LEAK_TOLERANCE]))
    wavelengths, _, reflectance, _ = spectra['glass', _UPSTREAM]
    reported = int(np.argmin(np.abs(wavelengths - _FRESNEL_REPORTED_WAVELENGTH)))
    return {'measured': float(reflectance[reported]), 'expected': _REFLECTANCE, 'error': error, 'tolerance': 1.0,
            'details': {'wavelengths': float_list(wavelengths), 'err_transfer': err_transfer, 'err_leak': err_leak,
                        'reflectance': _REFLECTANCE, 'transmittance': _TRANSMITTANCE, **details}}


def _check_slab(results_by_name, polarization):
    """Downstream fraction_positive vs 1, the other fractions vs 0 and the downstream n_eff vs the analytic slab
    relation; error in tolerance units."""
    run = results_by_name['run']
    wavelengths, forward, reflected, n_eff = _mode_spectra(run, _DOWNSTREAM)
    _, upstream_forward, upstream_backward, _ = _mode_spectra(run, _UPSTREAM)
    n_analytic = np.array([_slab_fundamental_index(wavelength, polarization) for wavelength in wavelengths])
    err_forward = float(np.max(np.abs(forward - 1.0)))   # NaN propagates
    err_leak = float(np.max(np.abs([reflected, upstream_forward, upstream_backward])))
    err_index = float(np.max(np.abs(n_eff / n_analytic - 1.0)))
    error = float(np.max([err_forward / _SLAB_FORWARD_TOLERANCE, err_leak / _SLAB_LEAK_TOLERANCE,
                          err_index / _SLAB_INDEX_TOLERANCE]))
    reported = int(np.argmin(np.abs(wavelengths - _SLAB_REPORTED_WAVELENGTH)))
    return {'measured': float(n_eff[reported]), 'expected': float(n_analytic[reported]), 'error': error,
            'tolerance': 1.0,
            'details': {'wavelengths': float_list(wavelengths), 'n_eff': float_list(n_eff),
                        'n_eff_analytic': float_list(n_analytic), 'downstream_fraction_positive': float_list(forward),
                        'downstream_fraction_negative': float_list(reflected),
                        'upstream_fraction_positive': float_list(upstream_forward),
                        'upstream_fraction_negative': float_list(upstream_backward), 'err_forward': err_forward,
                        'err_leak': err_leak, 'err_index': err_index}}


def _port_fraction(results, feature, series_name):
    """(ascending wavelengths [m], fraction series) float64 of a source-normalised mode monitor."""
    wavelengths, positive, negative, _ = _mode_spectra(results, feature)
    fraction = positive if series_name == 'fraction_positive' else negative
    order = np.argsort(wavelengths)
    return wavelengths[order], fraction[order]


def _check_coupler(results_by_name):
    """FDTD port fractions (interpolated in wavelength) vs the FDFD ones at the three FDFD wavelengths; error in units
    of _COUPLER_AGREEMENT_TOLERANCE."""
    differences, details, split = {}, {}, {}
    for run in ('fdtd', 'fdfd'):
        fractions = {}
        for feature, series_name in _COUPLER_PORTS:
            wavelengths, fraction = _port_fraction(results_by_name[run], feature, series_name)
            fractions[feature] = (wavelengths, fraction)
            details[f'{run}_{feature}_{series_name}'] = float_list(fraction)
        details[f'{run}_wavelengths'] = float_list(fractions['through'][0])
        split[run] = fractions
    fdfd_wavelengths = split['fdfd']['through'][0]
    for feature, _ in _COUPLER_PORTS:
        fdtd_wavelengths, fdtd_fraction = split['fdtd'][feature]
        interpolated = np.interp(fdfd_wavelengths, fdtd_wavelengths, fdtd_fraction)
        differences[feature] = float(np.max(np.abs(interpolated - split['fdfd'][feature][1])))   # NaN propagates

    def split_ratio(run):
        wavelengths, through = split[run]['through']
        cross = split[run]['cross'][1]
        ratio = cross / (through + cross)
        return float(np.interp(_COUPLER_REPORTED_WAVELENGTH, wavelengths, ratio))

    error = max(differences.values()) / _COUPLER_AGREEMENT_TOLERANCE
    return {'measured': split_ratio('fdtd'), 'expected': split_ratio('fdfd'), 'error': error, 'tolerance': 1.0,
            'details': {'largest_difference': differences, **details}}


# polarisation -> the wording of its cases
_CASE_TEXT = {
    'ez': {'prefix': 'em', 'label': 'EM', 'polarization': 'E_z polarisation', 'fresnel_note': '',
           'relation': 'kappa tan(kappa d / 2) = gamma (E_z and dE_z/dy continuous)', 'slab_note': ''},
    'hz': {'prefix': 'em_hz', 'label': 'EM H_z', 'polarization': 'H_z polarisation',
           'fresnel_note': '; the E_y face on the interface takes the mean permittivity (1 + 4) / 2, the correct '
                           'average for an E tangential to it',
           'relation': 'kappa tan(kappa d / 2) = (n1^2 / n2^2) gamma (H_z and dH_z/dy / eps continuous)',
           'slab_note': '; H_z: the interface faces take the mean permittivity, which keeps the continuity of the '
                        'tangential E_x first-order exact'},
}


def mode_port_fresnel_case(polarization):
    """SelfTestCase of the plane-wave mode ports in a y-periodic strip (vacuum and vacuum / glass runs) in
    polarisation 'ez' or 'hz'."""
    text = _CASE_TEXT[polarization]
    return SelfTestCase(
        name=f"{text['prefix']}_mode_port_fresnel",
        label=f"{text['label']} mode ports: plane wave and Fresnel",
        description='A gaussian mode source (1 um, bandwidth 0.5, 1 W/m, direction positive) spans the period of a '
                    f"y-periodic strip (CPML along x, 10 nm mesh, {text['polarization']}), so its periodic mode "
                    'solve gives the uniform plane wave; mode monitors 0.5 um upstream and 2.5 um downstream report '
                    'source-normalised mode power fractions at 5 wavelengths from 0.9 to 1.1 um. Run vacuum: '
                    'downstream fraction_positive = 1 and every other fraction = 0 (upstream fraction_negative is the '
                    'backward leakage of the one-sided source, downstream fraction_negative the CPML reflection). Run '
                    'glass: an eps_r = 4 half-space from 1.5 um downstream of the source gives upstream '
                    'fraction_negative = R = ((n - 1) / (n + 1))^2 = 1/9 and downstream fraction_positive = T = 4 n / '
                    '(n + 1)^2 = 8/9 (n = 2, the downstream mode solved in the glass), the other fractions 0. error = '
                    'max(transfer error / 0.01, leak / 1e-3), tolerance 1, with transfer error = the largest '
                    '|fraction - (1, 1/9 or 8/9)| and leak = the largest fraction that should vanish, over both runs '
                    'and all wavelengths; measured / expected = R at 1 um. Budget: transfer 0.01 for the discrete '
                    'Fresnel coefficients at >= 45 cells per wavelength in the glass (O((k dx)^2) < 1e-3; the '
                    'projection uses the FDTD-consistent mode impedance, so the half-cell E / H offset of a flux sum '
                    f"does not enter){text['fresnel_note']} and CPML reflection; leak 1e-3 for the one-sided launch "
                    'away from 1 um (the incident phase step k h across the half cell before the line is exact only '
                    'at the source wavelength: (Delta k h / 2)^2 < 1e-5 here) and normal-incidence CPML reflections '
                    f'(< 1e-4 of the power). Run length {_FRESNEL_DURATION * 1e15:g} fs, not yet confirmed by a run.',
        specs=lambda: _fresnel_specs(polarization),
        check=_check_fresnel)


def mode_port_slab_case(polarization):
    """SelfTestCase of the silicon slab waveguide (directionality, transmission and n_eff) in polarisation 'ez' or
    'hz'."""
    text = _CASE_TEXT[polarization]
    return SelfTestCase(
        name=f"{text['prefix']}_mode_port_slab",
        label=f"{text['label']} mode ports: silicon slab waveguide",
        description='A gaussian mode source (1.55 um, bandwidth 0.3, mode 1, direction positive, 1 W/m) launches the '
                    'fundamental mode of a 0.28 um silicon slab (eps_r 12.08) in silica (eps_r 2.085) along +x in a 4 '
                    f"x 2 um all-CPML box (20 nm mesh, {text['polarization']}); the source and two mode monitors (0.5 "
                    'um upstream, 2 um downstream) span the window height, so every mode solve continues through the '
                    'y PML. At 5 wavelengths from 1.5 to 1.6 um the downstream fraction_positive must be 1 (lossless '
                    'straight guide), every other fraction 0 (upstream: the backward leakage of the source; downstream '
                    'fraction_negative: CPML reflection of the guided mode), and the downstream n_eff must match '
                    f"the fundamental even mode of the symmetric slab, {text['relation']}, kappa = k0 sqrt(n1^2 - "
                    'n_eff^2), gamma = k0 sqrt(n_eff^2 - n2^2), solved by bisection on kappa d / 2 in (0, pi / 2). '
                    'error = max(|fraction_positive - 1| / 0.02, leak / 2e-3, |n_eff / n_analytic - 1| / 0.01), '
                    'tolerance 1; measured / expected = n_eff at 1.55 um. Budget: forward 0.02 for the source mode '
                    'solved at 1.55 um only (up to 3.3% off centre the impedance mismatch costs (Delta n_eff / 2 '
                    'n_eff)^2 ~ 1e-4 and the profile mismatch < 2e-3, which goes into radiation the projection '
                    'rejects) and CPML; leak 2e-3 for that off-centre backward leakage and the guided-mode CPML '
                    'reflection; n_eff 0.01 for the FDTD-consistent mode against the continuum: discrete dispersion '
                    'along x ((k dx)^2 / 24 ~ 0.25%), transverse discretisation at 14 cells across the slab (~0.1%'
                    f"{text['slab_note']}) and the numerical frequency (~0.01%). Run length "
                    f'{_SLAB_DURATION * 1e15:g} fs, not yet confirmed by a run.',
        specs=lambda: _slab_specs(polarization),
        check=lambda results_by_name: _check_slab(results_by_name, polarization))


COUPLER_AGREEMENT_CASE = SelfTestCase(
    name='em_hz_directional_coupler_fdtd_vs_fdfd',
    label='EM H_z directional coupler: FDTD vs FDFD split',
    description="The 'directional_coupler' template geometry (em_fdtd_hz, gaussian mode source at 1.55 um; its "
                'monitors set to the FDFD wavelengths 1.5, 1.55, 1.6 um and the run to '
                f'{_COUPLER_CHECK_DURATION * 1e15:g} fs instead of the template\'s 1.45-1.65 um and '
                f'{_COUPLER_DURATION * 1e15:g} fs) against the same geometry in em_fdfd_hz (frequency_domain study at '
                '1.5, 1.55 and 1.6 um, mode_source phase 0, 1 W/m), so the split ratio of the template is not an '
                'estimate but agrees between the two engines on the same grid. At the FDFD wavelengths the through '
                'and cross fraction_positive and the reflection fraction_negative must agree within 0.02 '
                '(absolute). Budget: the FDTD launch is exact at 1.55 um only (off-centre profile / impedance '
                'mismatch of the 1.55 um mode < 2e-3 at 1.5 / 1.6 um), the FDTD temporal dispersion shifts the '
                'frequency by (omega dt)^2 / 24 ~ 1.3e-4 (< 1e-3 in the fractions), the CPML / DFT truncation (the '
                'pulse and its ~170 fs group delay end by ~220 fs) stays below 1e-3; the spatial grid, the port '
                'modes and the materials are shared. error = largest difference / 0.02, tolerance 1; measured / '
                'expected = cross / (through + cross) at 1.55 um of FDTD / FDFD. Shorter FDTD run not yet '
                'confirmed by a run.',
    specs=_coupler_specs,
    check=_check_coupler)

# E_z plane-wave ports: examples/em/analytic_lorentz_halfspace (R) and analytic_matched_absorber_mu_sigma_m (R ~ 0, T);
# the H_z surface plasmon: examples/em/analytic_spp_drude.
MODE_PORT_SELFTEST_CASES = (
    mode_port_fresnel_case('hz'),
    mode_port_slab_case('ez'),
    mode_port_slab_case('hz'),
    COUPLER_AGREEMENT_CASE,
)
