"""Bloch-phase self-tests: oblique Fresnel transmission of both polarisations (the H_z one through the Brewster angle)
in the complex-field FDTD (A6) and the phased-wrap FDFD (symmetry 2.2), the FDFD oblique TFSF plane wave and empty-
lattice eigenfrequencies, and the 'bloch_band_diagram' (FDTD resonance_monitor) / 'fdfd_band_diagram' (eigenfrequency)
templates (k sweep). CPU only, every spec is built fresh on each call."""
import math

import numpy as np

from ...core.constants import C0, ETA0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import float_list, glass_half_space, vertical_line
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, ParameterSpec,
                           PhysicsSpec, StudySpec, SweepSpec)

_PHYSICS_NAME = 'em'
_TRANSMITTED_FLUX = 'transmitted'

# em_hz_brewster_oblique: y-periodic strip (Bloch ky), CPML along x, 10 nm cells; a vertical line source at x = 1 um
# launches the oblique wave toward +x, glass (n = 1.5) fills x >= 2.5 um (into the x_hi PML), the flux line sits at
# x = 3.5 um. em_oblique_fresnel is the same strip transposed (x-periodic, Bloch kx, propagation along +y), so the
# x-wrap branches of the kernels run with sin != 0 too. k puts the Brewster angle atan(1.5) = 56.3 deg at 1 um, so the
# angle sweeps 48.6-60.9 deg over the monitored 0.9-1.05 um (sin(theta) = k lambda / (2 pi)); cutoff (grazing) at
# 2 pi / k = 1.20 um.
# The E_z sheet radiates ~1 / cos(theta_i) there and the vacuum reference rings at the cutoff (decay ~t^-1/2), which
# leaks into its truncated DFT (worst at 1.05 um; H_z sheets radiate no such peak): bandwidth 0.08 puts the pulse
# spectrum at the cutoff at 3e-3 of its peak (0.15 gave 0.2), the monitored band stays above 0.39.
# The period (10 cells) admits only the zeroth diffraction order.
_OBLIQUE_EPS_R = 2.25
_OBLIQUE_INDEX = math.sqrt(_OBLIQUE_EPS_R)
_BREWSTER_WAVELENGTH = 1e-6                                                   # m
_OBLIQUE_KY = 2.0 * math.pi * math.sin(math.atan(_OBLIQUE_INDEX)) / _BREWSTER_WAVELENGTH   # rad/m
_OBLIQUE_SIZE = (5e-6, 0.1e-6)                                                # m
_OBLIQUE_PULSE = {'waveform': 'gaussian', 'wavelength': 0.97e-6, 'bandwidth': 0.08, 'amplitude': 1.0}
_OBLIQUE_SPECTRAL = {'wavelength_start': 0.9e-6, 'wavelength_stop': 1.05e-6, 'n_wavelengths': 7}
# s: pulse 2 t0 ~ 116 fs, slowest monitored wave (0.49 c along x) ~ 17 fs; the reference's cutoff ringing (~t^-1/2)
# leaks ~sqrt(300 / 200) = 1.22x more into the truncated DFT than at 300 fs (run once to check the 1.05 um error)
_OBLIQUE_DURATION = 200e-15
_OBLIQUE_TOLERANCE = 0.015        # absolute, worst wavelength
_BLOCH_AXIS = {'ez': 'x', 'hz': 'y'}   # periodic (Bloch) axis of each case; the wave travels along the other one
_OBLIQUE_TEXT = {
    'ez': {'name': 'em_oblique_fresnel', 'label': 'EM oblique Fresnel transmission (Bloch k)', 'pol': 's',
           'field': 'E_z', 'formula': 'r_s = (cos(theta_i) - n cos(theta_t)) / (cos(theta_i) + n cos(theta_t))'},
    'hz': {'name': 'em_hz_brewster_oblique', 'label': 'EM H_z oblique Fresnel transmission through Brewster',
           'pol': 'p', 'field': 'in-plane E',
           'formula': 'r_p = (n cos(theta_i) - cos(theta_t)) / (n cos(theta_i) + cos(theta_t)), zero at the Brewster '
                      'angle atan(n) = 56.3 deg (1 um), where T = 1'},
}

# 'bloch_band_diagram' template: E_z bands of the square lattice of dielectric rods (eps_r 8.9, r = 0.2 a, a = 0.5 um;
# Joannopoulos et al., Photonic Crystals ch. 5: TM gap 0.32-0.44 in omega a / (2 pi c), i.e. 1.14-1.56 um) along
# Gamma-X: one unit cell, every face periodic, bloch_kx = k_fraction * 2 pi / a swept over 0..0.5.
BAND_DIAGRAM_LABEL = 'Photonic crystal band diagram'
BAND_DIAGRAM_DESCRIPTION = ('EM (E_z): one unit cell of a square lattice of dielectric rods (eps_r 8.9, r = 0.2 a, '
                            'a = 0.5 um), Bloch-periodic with bloch_kx = k_fraction 2 pi / a swept over 0-0.5 '
                            '(Gamma-X); a point-source pulse and a resonance finder per k give the band frequencies '
                            '(TM gap ~1.14-1.56 um).')
_LATTICE = 0.5e-6                 # m, lattice constant a
_ROD_RADIUS = 0.2 * _LATTICE
_ROD_EPS_R = 8.9
_K_FRACTIONS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)   # bloch_kx in units of 2 pi / a
# 'fdfd_band_diagram' template: the same rod cell in em_fdfd_ez, an eigenfrequency study (6 resonances nearest 1.2 um,
# between the first two TM bands) per swept k; 25 x 25 cells, so each solve is instantaneous.
FDFD_BAND_DIAGRAM_LABEL = 'Photonic crystal band diagram (frequency domain)'
FDFD_BAND_DIAGRAM_DESCRIPTION = ('EM (FDFD, E_z): the rod unit cell of bloch_band_diagram (eps_r 8.9, r = 0.2 a, a = '
                                 '0.5 um), Bloch-periodic with bloch_kx = k_fraction 2 pi / a swept over 0-0.5 '
                                 '(Gamma-X); an eigenfrequency study per k lists the 6 band frequencies nearest 1.2 um '
                                 '(resonance_table), without time stepping or a resonance fit.')
_FDFD_BAND_TARGET_WAVELENGTH = 1.2e-6   # m
_FDFD_BAND_RESONANCES = 6

# FDFD oblique cases (fdfd_oblique_fresnel_case): the FDTD strips, solved at the monitored wavelengths.
_FDFD_OBLIQUE_TOLERANCE = 0.01    # absolute, worst wavelength
# em_fdfd_oblique_tfsf: empty vacuum strip 3 x 0.1 um (y-periodic, Bloch ky = _OBLIQUE_KY: 56.3 deg at 1 um), CPML along
# x, 10 nm cells; a full-period TFSF box over x 1-2 um (cells [100, 201)) and vertical flux lines at cell edges 0.5 /
# 1.5 / 2.5 um (cells 50, 150, 250). The discrete oblique wave makes the box exactly transparent: the scattered field
# outside is round-off, the inside flux A^2 L sin(k_x dx) / (2 eta0 k0 dx) differs from A^2 L cos(theta) / (2 eta0) by
# ~(k dx)^2 / 6 ~ 5e-4. Unknowns 330 x 10.
_TFSF_SIZE = (3e-6, 0.1e-6)                    # m
_TFSF_BOX_X = (1e-6, 2e-6)                     # m
_TFSF_AMPLITUDE = 1.0                          # V/m: E_z (ez) / in-plane |E| (hz)
_TFSF_FLUX_LINES = {'upstream': 0.5e-6, 'inside': 1.5e-6, 'downstream': 2.5e-6}   # m
_TFSF_TOLERANCE = 2e-3                         # relative inside flux; scattered / inside flux
# em_fdfd_bloch_empty_lattice: square cell a = 0.5 um of uniform glass (eps_r 2.25), every face Bloch-periodic with
# (kx, ky) = (0.3, 0.15) 2 pi / a, 25 nm cells (20 x 20 unknowns = the 400 grid harmonics); the 3 resonances nearest 1
# um are the empty-lattice bands |q| = 0.716, 0.901, 1.101 (2 pi / a) of the discrete dispersion (exact: no PML, no
# dispersion, so the eigenproblem is linear).
_EMPTY_LATTICE = 0.5e-6                        # m
_EMPTY_LATTICE_CELL = 25e-9                    # m
_LATTICE_K = (0.3 * 2.0 * math.pi / _EMPTY_LATTICE, 0.15 * 2.0 * math.pi / _EMPTY_LATTICE)   # rad/m
_LATTICE_TARGET_WAVELENGTH = 1e-6              # m
_LATTICE_N_RESONANCES = 3
_LATTICE_TOLERANCE = 1e-6                      # relative wavelength (ARPACK, float32 cell widths)
_BANDS_TABLE = 'bands'


def _oblique_spec(physics_type, run_name, bloch_axis):
    """'reference' (vacuum) or 'device' (glass from 2.5 um along the propagation axis) oblique strip, periodic along
    bloch_axis ('y': Bloch ky, propagation +x; 'x': the transposed strip, Bloch kx, propagation +y); FDTD: a pulse in a
    time_dependent study, FDFD: the monitored wavelengths in a frequency_domain study."""
    harmonic = physics_type.startswith('em_fdfd')
    length, period = _OBLIQUE_SIZE

    def placed(along, across):   # (propagation coordinate, periodic coordinate) -> (x, y)
        return (along, across) if bloch_axis == 'y' else (across, along)

    def across_line(position):   # line spanning the period at `position` along the propagation axis
        (x0, y0), (x1, y1) = placed(position, 0.0), placed(position, period)
        return {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1}
    size_x, size_y = placed(length, period)
    settings = {f'bc_{bloch_axis}_lo': 'periodic', f'bc_{bloch_axis}_hi': 'periodic',
                f'bloch_k{bloch_axis}': _OBLIQUE_KY}
    spec = ModelSpec(
        title=f'Self-test: oblique Fresnel transmission ({run_name})',
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=[GeometrySpec('source_line', 'line', across_line(1e-6)),
                  GeometrySpec('flux_line', 'line', across_line(3.5e-6))],
        physics=[PhysicsSpec(_PHYSICS_NAME, physics_type, settings=settings, features=[
            FeatureSpec('source', 'line_source', ['source_line'],
                        {'amplitude': 1.0, 'phase': 0.0} if harmonic else dict(_OBLIQUE_PULSE)),
            FeatureSpec(_TRANSMITTED_FLUX, 'flux_monitor', ['flux_line'], {} if harmonic else dict(_OBLIQUE_SPECTRAL))])],
        mesh=MeshSpec(mode='uniform', dx=10e-9),
        study=(StudySpec(type='frequency_domain', settings=dict(_OBLIQUE_SPECTRAL)) if harmonic else
               StudySpec(type='time_dependent', settings={'duration': _OBLIQUE_DURATION})))
    if run_name == 'device':
        if bloch_axis == 'y':
            half_space, glass = glass_half_space(2.5e-6, size_x, size_y, _OBLIQUE_EPS_R)
        else:
            half_space = GeometrySpec('half_space', 'rectangle', {'x0': 0.0, 'y0': 2.5e-6, 'x1': size_x,
                                                                  'y1': size_y})
            glass = MaterialSpec('glass', properties={'eps_r': _OBLIQUE_EPS_R}, selection=['half_space'])
        spec.geometry.append(half_space)
        spec.materials.append(glass)
    return spec


def oblique_transmission(wavelengths, polarization):
    """Fresnel power transmission vacuum -> n = _OBLIQUE_INDEX at sin(theta_i) = ky lambda / (2 pi), s ('ez') or p
    ('hz') polarisation; float64 array."""
    sin_incident = _OBLIQUE_KY * np.asarray(wavelengths, dtype=np.float64) / (2.0 * math.pi)
    cos_incident = np.sqrt(1.0 - sin_incident ** 2)
    cos_transmitted = np.sqrt(1.0 - (sin_incident / _OBLIQUE_INDEX) ** 2)
    n = _OBLIQUE_INDEX
    if polarization == 'ez':
        reflection = (cos_incident - n * cos_transmitted) / (cos_incident + n * cos_transmitted)
    else:
        reflection = (n * cos_incident - cos_transmitted) / (n * cos_incident + cos_transmitted)
    return 1.0 - reflection ** 2


def _check_oblique(results_by_name, polarization, tolerance):
    """T = device flux / reference flux per wavelength vs the Fresnel T(theta(lambda)); measured = T at the wavelength
    closest to 1 um (the Brewster angle for 'hz'), error = the worst |T - T_Fresnel|."""
    reference = results_by_name['reference'].result(_PHYSICS_NAME, _TRANSMITTED_FLUX)
    device = results_by_name['device'].result(_PHYSICS_NAME, _TRANSMITTED_FLUX)
    wavelengths = np.asarray(reference['wavelength'], dtype=np.float64)
    reference_power = np.asarray(reference['power'], dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        transmission = np.asarray(device['power'], dtype=np.float64) / reference_power
    expected = oblique_transmission(wavelengths, polarization)
    deviation = np.abs(transmission - expected)
    worst = int(np.argmax(deviation))   # a NaN counts as the largest deviation
    reported = int(np.argmin(np.abs(wavelengths - _BREWSTER_WAVELENGTH)))
    angles = np.degrees(np.arcsin(_OBLIQUE_KY * wavelengths / (2.0 * math.pi)))
    return {'measured': float(transmission[reported]), 'expected': float(expected[reported]),
            'error': float(deviation[worst]), 'tolerance': tolerance,
            'details': {'wavelengths': float_list(wavelengths), 'incidence_angles_deg': float_list(angles),
                        'transmission': float_list(transmission), 'fresnel': float_list(expected),
                        'worst_wavelength': float(wavelengths[worst]), 'bloch_k': _OBLIQUE_KY,
                        'reference_power': float_list(reference_power)}}


def oblique_fresnel_case(physics_type, polarization):
    """SelfTestCase of the oblique (Bloch ky) Fresnel transmission in polarisation 'ez' (s) or 'hz' (p, Brewster)."""
    text = _OBLIQUE_TEXT[polarization]
    bloch_axis = _BLOCH_AXIS[polarization]
    normal_axis = 'y' if bloch_axis == 'x' else 'x'
    return SelfTestCase(
        name=text['name'],
        label=text['label'],
        description=f'A gaussian pulse (0.97 um, bandwidth 0.08) from a line source spanning a {bloch_axis}-periodic '
                    f'strip with Bloch wave vector k{bloch_axis} = {_OBLIQUE_KY:.6g} rad/m (complex-field FDTD: the '
                    f'sheet carries exp(-j k{bloch_axis} {bloch_axis}), the {bloch_axis} wraps the Bloch phase; 10 nm '
                    f'mesh, CPML along {normal_axis}) crosses a vacuum / glass interface normal to {normal_axis} '
                    f"(eps_r = 2.25, n = 1.5) at oblique incidence, {text['pol']}-polarised ({text['field']}). "
                    'At fixed k the angle follows sin(theta_i) = k lambda / (2 pi): 48.6-60.9 deg over the 7 '
                    'monitored wavelengths 0.9-1.05 um. T = flux 1 um behind the interface / flux of a vacuum '
                    'reference run with the same source must equal 1 - |r|^2, '
                    f"{text['formula']}, cos(theta_t) = sqrt(1 - sin(theta_i)^2 / n^2); error = max |T - T_Fresnel|, "
                    'measured / expected = T at 1 um. Tolerance 0.015: discrete dispersion at >= 60 cells per '
                    'wavelength in the glass (the discrete angle differs from the continuum one by ~(k dx)^2), the '
                    'half-cell E / H offset of the discrete flux, CPML reflection of the oblique waves, and the '
                    f'truncated ({_OBLIQUE_DURATION * 1e15:g} fs) DFT of the reference, which rings at the 1.2 um '
                    'grazing cutoff (the narrow pulse keeps its spectrum there at 3e-3 of the peak; worst at 1.05 '
                    f'um). {_OBLIQUE_DURATION * 1e15:g} fs not yet confirmed by a run.',
        specs=lambda: {run_name: _oblique_spec(physics_type, run_name, bloch_axis)
                       for run_name in ('reference', 'device')},
        check=lambda results_by_name: _check_oblique(results_by_name, polarization, _OBLIQUE_TOLERANCE),
        heavy=True)


def fdfd_oblique_fresnel_case(polarization):
    """SelfTestCase of the FDFD oblique Fresnel transmission on the strips of oblique_fresnel_case (frequency_domain)."""
    text = _OBLIQUE_TEXT[polarization]
    bloch_axis = _BLOCH_AXIS[polarization]
    normal_axis = 'y' if bloch_axis == 'x' else 'x'
    physics_type = f'em_fdfd_{polarization}'
    return SelfTestCase(
        name=text['name'].replace('em_', 'em_fdfd_', 1),
        label=text['label'].replace('EM ', 'EM FDFD ', 1),
        description=f'The strip of {text["name"]} in {physics_type}: a line source spanning the {bloch_axis}-periodic '
                    f'strip with Bloch wave vector k{bloch_axis} = {_OBLIQUE_KY:.6g} rad/m (the sheet carries exp(-j '
                    f'k{bloch_axis} {bloch_axis}), the operator wrap entries carry exp(-j k{bloch_axis} L); 10 nm '
                    'mesh, CPML '
                    f'along {normal_axis}) crosses a vacuum / glass interface (eps_r = 2.25, n = 1.5) at oblique '
                    f"incidence, {text['pol']}-polarised ({text['field']}), solved at the 7 wavelengths 0.9-1.05 um "
                    '(frequency_domain; sin(theta_i) = k lambda / (2 pi): 48.6-60.9 deg). T = device flux 1 um behind '
                    'the interface / flux of the vacuum reference run must equal 1 - |r|^2, '
                    f"{text['formula']}; error = max |T - T_Fresnel|, measured / expected = T at 1 um. Tolerance "
                    '0.01: discrete dispersion at >= 60 cells per wavelength in the glass (the discrete angle and the '
                    'face-averaged interface permittivity differ from the continuum at ~(k dx)^2 / first order).',
        specs=lambda: {run_name: _oblique_spec(physics_type, run_name, bloch_axis)
                       for run_name in ('reference', 'device')},
        check=lambda results_by_name: _check_oblique(results_by_name, polarization, _FDFD_OBLIQUE_TOLERANCE))


def _oblique_tfsf_spec(polarization):
    """Empty y-periodic vacuum strip (Bloch ky, CPML along x), a full-period TFSF box over x 1-2 um and three flux lines
    (upstream, inside, downstream) at the stationary wavelength 1 um."""
    length, period = _TFSF_SIZE
    geometry = [GeometrySpec('tfsf_box', 'rectangle', {'x0': _TFSF_BOX_X[0], 'y0': 0.0, 'x1': _TFSF_BOX_X[1],
                                                       'y1': period})]
    features = [FeatureSpec('plane_wave', 'plane_wave_tfsf', ['tfsf_box'], {'amplitude': _TFSF_AMPLITUDE, 'phase': 0.0})]
    for name, position in _TFSF_FLUX_LINES.items():
        geometry.append(vertical_line(f'{name}_line', position, 0.0, period))
        features.append(FeatureSpec(name, 'flux_monitor', [f'{name}_line']))
    settings = {'bc_y_lo': 'periodic', 'bc_y_hi': 'periodic', 'bloch_ky': _OBLIQUE_KY,
                'wavelength': _BREWSTER_WAVELENGTH}
    return ModelSpec(
        title=f'Self-test: oblique FDFD TFSF plane wave ({polarization})',
        domain=DomainSpec(size_x=length, size_y=period),
        geometry=geometry,
        physics=[PhysicsSpec(_PHYSICS_NAME, f'em_fdfd_{polarization}', settings=settings, features=features)],
        mesh=MeshSpec(mode='uniform', dx=10e-9),
        study=StudySpec(type='stationary'))


def _check_oblique_tfsf(results_by_name):
    """Inside flux vs A^2 L cos(theta) / (2 eta0) and the scattered (upstream / downstream) fluxes over it, per run."""
    sin_incident = _OBLIQUE_KY * _BREWSTER_WAVELENGTH / (2.0 * math.pi)
    expected = _TFSF_AMPLITUDE ** 2 * _TFSF_SIZE[1] * math.sqrt(1.0 - sin_incident ** 2) / (2.0 * float(ETA0))
    errors, details = [], {'expected_inside_power': expected, 'incidence_angle_deg': math.degrees(math.asin(
        sin_incident))}
    for polarization, results in results_by_name.items():
        powers = {name: float(np.asarray(results.result(_PHYSICS_NAME, name)['power'], dtype=np.float64)[0])
                  for name in _TFSF_FLUX_LINES}
        inside = powers['inside']
        errors += [abs(inside / expected - 1.0), abs(powers['upstream']) / abs(inside),
                   abs(powers['downstream']) / abs(inside)]
        details[f'{polarization}_powers'] = powers
    return {'measured': max(errors), 'expected': 0.0, 'error': max(errors), 'tolerance': _TFSF_TOLERANCE,
            'details': details}


def _empty_lattice_spec(polarization):
    """Fully Bloch-periodic square cell of uniform glass (eps_r 2.25) with a resonance table: eigenfrequency study."""
    periodic = {f'bc_{face}': 'periodic' for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi')}
    settings = {**periodic, 'bloch_kx': _LATTICE_K[0], 'bloch_ky': _LATTICE_K[1],
                'wavelength': _LATTICE_TARGET_WAVELENGTH}
    return ModelSpec(
        title=f'Self-test: FDFD Bloch empty lattice ({polarization})',
        domain=DomainSpec(size_x=_EMPTY_LATTICE, size_y=_EMPTY_LATTICE),
        materials=[MaterialSpec('glass', properties={'eps_r': _OBLIQUE_EPS_R}, selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS_NAME, f'em_fdfd_{polarization}', settings=settings,
                             features=[FeatureSpec(_BANDS_TABLE, 'resonance_table')])],
        mesh=MeshSpec(mode='uniform', dx=_EMPTY_LATTICE_CELL),
        study=StudySpec(type='eigenfrequency', settings={'n_resonances': _LATTICE_N_RESONANCES}))


def empty_lattice_wavelengths():
    """Vacuum wavelengths [m] of the _LATTICE_N_RESONANCES discrete empty-lattice bands nearest the target: eps k^2 =
    sum over axes (2 / dx)^2 sin^2(q dx / 2), q = k_Bloch + 2 pi m / L for each of the N grid harmonics m."""
    cell = float(np.float32(_EMPTY_LATTICE_CELL))   # the float32 grid widths the operator uses
    n_cells = int(round(_EMPTY_LATTICE / _EMPTY_LATTICE_CELL))
    period = n_cells * cell
    orders = np.arange(n_cells) - n_cells // 2
    q_x = _LATTICE_K[0] + 2.0 * math.pi * orders[:, None] / period
    q_y = _LATTICE_K[1] + 2.0 * math.pi * orders[None, :] / period
    wavenumber_squared = ((2.0 / cell) ** 2 * (np.sin(0.5 * q_x * cell) ** 2 + np.sin(0.5 * q_y * cell) ** 2)
                          / _OBLIQUE_EPS_R)
    omegas = C0 * np.sqrt(wavenumber_squared.ravel())
    omega_target = 2.0 * math.pi * C0 / _LATTICE_TARGET_WAVELENGTH
    nearest = omegas[np.argsort(np.abs(omegas - omega_target))[:_LATTICE_N_RESONANCES]]
    return 2.0 * math.pi * C0 / nearest


def _check_empty_lattice(results_by_name):
    """Max relative deviation of the tabulated wavelengths (rank order) from empty_lattice_wavelengths, per run."""
    expected = empty_lattice_wavelengths()
    errors, details = [], {'expected_wavelengths': float_list(expected)}
    for polarization, results in results_by_name.items():
        rows = results.result(_PHYSICS_NAME, _BANDS_TABLE)['rows']
        measured = np.array([row[1] for row in rows[:expected.size]], dtype=np.float64)
        if measured.size != expected.size:
            errors.append(math.inf)
        else:
            errors.append(float(np.max(np.abs(measured / expected - 1.0))))
        details[f'{polarization}_wavelengths'] = float_list(measured)
    return {'measured': max(errors), 'expected': 0.0, 'error': max(errors), 'tolerance': _LATTICE_TOLERANCE,
            'details': details}


def bloch_band_diagram_spec():
    """Template 'bloch_band_diagram': E_z unit cell of the rod lattice, k sweep along Gamma-X, resonance finder."""
    centre = 0.5 * _LATTICE
    periodic = {f'bc_{face}': 'periodic' for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi')}
    return ModelSpec(
        title=BAND_DIAGRAM_LABEL,
        parameters=[ParameterSpec('a', _LATTICE, 'lattice constant [m]'),
                    ParameterSpec('k_fraction', _K_FRACTIONS[-1], 'Bloch kx in units of 2 pi / a (Gamma-X: 0-0.5)')],
        domain=DomainSpec(size_x=_LATTICE, size_y=_LATTICE),
        geometry=[GeometrySpec('rod', 'circle', {'cx': centre, 'cy': centre, 'r': _ROD_RADIUS}),
                  GeometrySpec('source_point', 'point', {'x': 0.12e-6, 'y': 0.33e-6}),
                  GeometrySpec('monitor_point', 'point', {'x': 0.37e-6, 'y': 0.16e-6})],
        materials=[MaterialSpec('rod', properties={'eps_r': _ROD_EPS_R}, selection=['rod'])],
        physics=[PhysicsSpec(_PHYSICS_NAME, 'em_fdtd_ez', settings={**periodic, 'bloch_kx': 'k_fraction * 2 * pi / a'},
                             features=[
            FeatureSpec('source', 'point_source', ['source_point'],
                        {'waveform': 'gaussian', 'wavelength': 1.2e-6, 'bandwidth': 0.6}),
            FeatureSpec('bands', 'resonance_monitor', ['monitor_point'],
                        {'component': 'ez', 'wavelength_min': 0.9e-6, 'wavelength_max': 3e-6})])],
        mesh=MeshSpec(mode='uniform', dx=20e-9),
        study=StudySpec(type='time_dependent', settings={'duration': 2e-12},
                        sweep=SweepSpec('k_fraction', list(_K_FRACTIONS))))


def fdfd_band_diagram_spec():
    """Template 'fdfd_band_diagram': the rod cell of bloch_band_diagram in em_fdfd_ez, k sweep, eigenfrequency study."""
    centre = 0.5 * _LATTICE
    periodic = {f'bc_{face}': 'periodic' for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi')}
    return ModelSpec(
        title=FDFD_BAND_DIAGRAM_LABEL,
        parameters=[ParameterSpec('a', _LATTICE, 'lattice constant [m]'),
                    ParameterSpec('k_fraction', _K_FRACTIONS[-1], 'Bloch kx in units of 2 pi / a (Gamma-X: 0-0.5)')],
        domain=DomainSpec(size_x=_LATTICE, size_y=_LATTICE),
        geometry=[GeometrySpec('rod', 'circle', {'cx': centre, 'cy': centre, 'r': _ROD_RADIUS})],
        materials=[MaterialSpec('rod', properties={'eps_r': _ROD_EPS_R}, selection=['rod'])],
        physics=[PhysicsSpec(_PHYSICS_NAME, 'em_fdfd_ez',
                             settings={**periodic, 'bloch_kx': 'k_fraction * 2 * pi / a',
                                       'wavelength': _FDFD_BAND_TARGET_WAVELENGTH},
                             features=[FeatureSpec(_BANDS_TABLE, 'resonance_table')])],
        mesh=MeshSpec(mode='uniform', dx=20e-9),
        study=StudySpec(type='eigenfrequency', settings={'n_resonances': _FDFD_BAND_RESONANCES},
                        sweep=SweepSpec('k_fraction', list(_K_FRACTIONS))))


# FDTD oblique_fresnel_case runs only inside selftests_backend em_fdtd_cpu_gpu_media (hz); standalone they repeat
# examples/em/analytic_bloch_fresnel_brewster (T_s, T_p, R_p_excess) and analytic_bloch_bragg_bands (FDTD bloch_kx wrap).
BLOCH_SELFTEST_CASES = (
    fdfd_oblique_fresnel_case('ez'),
    fdfd_oblique_fresnel_case('hz'),
    SelfTestCase(
        name='em_fdfd_oblique_tfsf',
        label='EM FDFD oblique TFSF plane wave (Bloch ky)',
        description='An empty vacuum strip 3 x 0.1 um, y-periodic with Bloch ky = '
                    f'{_OBLIQUE_KY:.6g} rad/m (incidence 56.3 deg at 1 um), CPML along x, 10 nm cells, holds a '
                    'full-period plane_wave_tfsf box over x 1-2 um (amplitude 1 V/m: E_z in em_fdfd_ez, the in-plane '
                    '|E| in em_fdfd_hz), solved at 1 um (stationary). The oblique incident wave exp(-j (k_x x + ky '
                    'y)) with k_x from the discrete dispersion must make the box transparent: flux through x = 1.5 um '
                    '(inside) = A^2 L cos(theta) / (2 eta0), flux through x = 0.5 / 2.5 um (scattered field) ~ 0. '
                    'error = the worst of |inside / expected - 1| and |scattered| / inside over both polarisations; '
                    'tolerance 2e-3 (discrete dispersion ~(k dx)^2 / 6 ~ 5e-4; the scattered fluxes are round-off).',
        specs=lambda: {polarization: _oblique_tfsf_spec(polarization) for polarization in ('ez', 'hz')},
        check=_check_oblique_tfsf),
    SelfTestCase(
        name='em_fdfd_bloch_empty_lattice',
        label='EM FDFD Bloch empty-lattice bands',
        description='Eigenfrequency study of a square cell a = 0.5 um of uniform glass (eps_r 2.25), every face '
                    'Bloch-periodic with (kx, ky) = (0.3, 0.15) 2 pi / a, 25 nm cells, in em_fdfd_ez and '
                    'em_fdfd_hz: the 3 resonances nearest 1 um must be the discrete empty-lattice bands eps k^2 = '
                    '(2 / dx)^2 [sin^2(q_x dx / 2) + sin^2(q_y dx / 2)], q = k + 2 pi m / a over the 20 x 20 grid '
                    'harmonics, ranked by |omega - omega_target| (|q| = 0.716, 0.901, 1.101 in 2 pi / a). Pins the '
                    'phase and sign of the wrap entries on both axes in the eigen path. error = max relative '
                    'wavelength deviation, tolerance 1e-6 (no PML, no dispersion: the problem is exactly linear).',
        specs=lambda: {polarization: _empty_lattice_spec(polarization) for polarization in ('ez', 'hz')},
        check=_check_empty_lattice),
)
