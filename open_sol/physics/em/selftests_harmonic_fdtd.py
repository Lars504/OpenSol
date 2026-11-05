"""Self-tests of the harmonic FDTD (frequency_domain study of em_fdtd_*): em_fdtd_ez against em_fdfd_ez on one model,
an em_fdtd_hz slab against the Airy transmission off the pulse carrier; the 'cylinder_harmonic_fdtd' template."""
import cmath
import math

import numpy as np

from ...core.constants import C0, MU0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import float_list
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, ParameterSpec,
                           PhysicsSpec, StudySpec)

# em_harmonic_fdtd_vs_fdfd: y-periodic strip 4 x 0.5 um, CPML along x, uniform 20 nm mesh (50 cells per vacuum
# wavelength, 25 in the cylinder), window cell k spanning [20 k, 20 (k + 1)] nm. A vertical line source at the
# centre of column 25 (x = 0.51 um) spans the period, so it launches a plane wave along +x onto a periodic row of
# dielectric cylinders (eps_r 4, r = 0.15 um, axis at (2.0, 0.25) um). Period 0.5 um < lambda / n_max: no diffraction
# order and no guided-mode resonance of the row at 1 um, so the response is smooth and the pulse rings down fast.
# Two runs share the mesh, the rasterised cylinder and a frequency_domain study at 1 um: em_fdtd_ez with a gaussian
# pulse centred at 1 um (the FDTD sheet calibration is exact at f0 only) and auto_shutoff 1e-6, em_fdfd_ez with the
# phase-0 line source. FDFD unknowns (200 + 30) x 25 = 5,750.
_RUNS = ('fdtd', 'fdfd')
_PHYSICS_NAME = 'em'
_PHYSICS_TYPES = {'fdtd': 'em_fdtd_ez', 'fdfd': 'em_fdfd_ez'}
_WAVELENGTH = 1e-6                      # m, vacuum: study wavelength = FDTD pulse centre
_CELL = 20e-9                           # m
_SIZE = (4e-6, 0.5e-6)                  # m
_SOURCE_X = 25.5 * _CELL                # m, centre of column 25
_CYLINDER = {'cx': 2.0e-6, 'cy': 0.25e-6, 'r': 0.15e-6}
_CYLINDER_EPS_R = 4.0
_AMPLITUDE = 1.0                        # V/m radiated to each side of the line
_BANDWIDTH = 0.5
_AUTO_SHUTOFF = 1e-6                    # field energy fraction ending the FDTD pulse (amplitude ~1e-3)
_ABS_TOLERANCE = 0.03                   # max | |E_fdtd| - |E_fdfd| | / max |E_fdfd|
_COMPLEX_TOLERANCE = 0.05               # max |E_fdtd - E_fdfd| / max |E_fdfd| (phase drift of numerical dispersion)

# em_hz_harmonic_fdtd_slab: y-periodic strip 4 x 0.1 um, CPML along x, uniform 12.5 nm mesh, line source at the centre
# of column 40 (x = 0.50625 um); the H_z line source launches an E_y / H_z plane-wave pulse (1 um, bandwidth 0.3)
# along +x onto a slab (eps_r 4, n = 2) over x = 1.5..2.0 um (40 whole cells; its edges on E_y faces, which take the
# face mean). One frequency_domain study of the three wavelengths 0.9, 1.0, 1.1 um (the source spectrum is 0.87 / 1 /
# 0.91 of its peak there, so a wrong S(omega) normalisation is off by 15 / 0 / 10 %) stores hz_re / hz_im per
# wavelength; H_z is read on the one-cell column at x = 3.0 um; a vacuum reference run gives the incident phasor there.
_SLAB_RUNS = ('device', 'reference')
_SLAB_PHYSICS_TYPE = 'em_fdtd_hz'
_SLAB_CELL = 12.5e-9                    # m
_SLAB_SOURCE_X = 40.5 * _SLAB_CELL      # m, centre of column 40
_SLAB_SIZE = (4e-6, 0.1e-6)             # m
_SLAB_X = (1.5e-6, 2.0e-6)              # m
_SLAB_EPS_R = 4.0
_SLAB_BANDWIDTH = 0.3
_SLAB_WAVELENGTHS = (0.9e-6, 1.0e-6, 1.1e-6)   # m: the study wavelengths
_READ_X = 3.0e-6                        # m, lo edge of the one-cell read column
_COURANT_FACTOR = 0.99                  # em_fdtd_* default: dt = factor dx / (C0 sqrt(2)) on the uniform mesh
_READ_FIELDS = {'re': 'hz_re', 'im': 'hz_im'}   # stored harmonic fields read on the column
_SLAB_T_TOLERANCE = 0.01                # max |t - t_Airy| (complex, H_z behind the slab / incident)
_INCIDENT_TOLERANCE = 0.01              # max |eta0 |H_z incident| / amplitude - 1|

# 'cylinder_harmonic_fdtd' template: dielectric cylinder (eps_r 4, r = 0.3 um) at the centre of a 3 x 3 um CPML box,
# 20 nm mesh (21 cells per wavelength in the cylinder at 0.85 um), TFSF plane-wave pulse (1 um, bandwidth 0.5) on the
# box x, y = 1.0..2.0 um; 8 study wavelengths 0.85..1.2 um inside the pulse band (accumulators 8 x 8 B x 22,500 cells).
CYLINDER_HARMONIC_LABEL = 'Dielectric cylinder, time-domain spectrum'
CYLINDER_HARMONIC_DESCRIPTION = (
    'EM FDTD (E_z) in a frequency_domain study (harmonic FDTD): one plane-wave pulse (TFSF box, 1 um, bandwidth 0.5) '
    'scatters off a dielectric cylinder (eps_r 4, radius r_cyl = 0.3 um) and gives the complex E_z phasor of every '
    'study wavelength 0.85-1.2 um (8 values, one snapshot each) from a single time-domain run; switch the physics to '
    'em_fdfd_ez for the same maps by one frequency-domain solve per wavelength.')
CYLINDER_HARMONIC_SUMMARY = ('A light pulse scatters off a dielectric cylinder of refractive index 2; one time-domain '
                             'run gives the steady-state field pattern at eight wavelengths between 850 and 1200 nm.')
_TEMPLATE_DOMAIN = 3e-6                 # m, square
_TEMPLATE_CELL = 20e-9                  # m
_TEMPLATE_TFSF = (1.0e-6, 2.0e-6)       # m, box edges in x and y
_TEMPLATE_RADIUS = 0.3e-6               # m


def _harmonic_spec(run):
    """The cylinder-row strip in em_fdtd_ez ('fdtd', gaussian pulse) or em_fdfd_ez ('fdfd', phase 0), frequency_domain
    study at _WAVELENGTH."""
    size_x, size_y = _SIZE
    source_values = ({'waveform': 'gaussian', 'wavelength': _WAVELENGTH, 'bandwidth': _BANDWIDTH,
                      'amplitude': _AMPLITUDE} if run == 'fdtd' else {'amplitude': _AMPLITUDE, 'phase': 0.0})
    physics_type = _PHYSICS_TYPES[run]
    return ModelSpec(
        title=f'Self-test: {physics_type} cylinder row, frequency domain at 1 um',
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=[GeometrySpec('source_line', 'line', {'x0': _SOURCE_X, 'y0': 0.0, 'x1': _SOURCE_X, 'y1': size_y}),
                  GeometrySpec('cylinder', 'circle', dict(_CYLINDER))],
        materials=[MaterialSpec('dielectric', properties={'eps_r': _CYLINDER_EPS_R}, selection=['cylinder'])],
        physics=[PhysicsSpec(_PHYSICS_NAME, physics_type, settings={'bc_y_lo': 'periodic', 'bc_y_hi': 'periodic'},
                             features=[FeatureSpec('source', 'line_source', ['source_line'], source_values)])],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type='frequency_domain', settings={'wavelengths': [_WAVELENGTH],
                                                            'auto_shutoff': _AUTO_SHUTOFF}))


def _harmonic_specs():
    """'fdtd' and 'fdfd': the same strip, mesh and study."""
    return {run: _harmonic_spec(run) for run in _RUNS}


def cylinder_harmonic_fdtd_spec():
    """Template 'cylinder_harmonic_fdtd': TFSF pulse on a dielectric cylinder in em_fdtd_ez, frequency_domain study at
    8 wavelengths (harmonic FDTD)."""
    centre = 0.5 * _TEMPLATE_DOMAIN
    lo, hi = _TEMPLATE_TFSF
    return ModelSpec(
        title=CYLINDER_HARMONIC_LABEL,
        parameters=[ParameterSpec('r_cyl', _TEMPLATE_RADIUS, 'cylinder radius [m]')],
        domain=DomainSpec(size_x=_TEMPLATE_DOMAIN, size_y=_TEMPLATE_DOMAIN),
        geometry=[GeometrySpec('cylinder', 'circle', {'cx': centre, 'cy': centre, 'r': 'r_cyl'}),
                  GeometrySpec('tfsf_box', 'rectangle', {'x0': lo, 'y0': lo, 'x1': hi, 'y1': hi})],
        materials=[MaterialSpec('dielectric', properties={'eps_r': _CYLINDER_EPS_R}, selection=['cylinder'])],
        physics=[PhysicsSpec(_PHYSICS_NAME, 'em_fdtd_ez', features=[
            FeatureSpec('plane_wave', 'plane_wave_tfsf', ['tfsf_box'],
                        {'waveform': 'gaussian', 'wavelength': _WAVELENGTH, 'bandwidth': _BANDWIDTH,
                         'amplitude': _AMPLITUDE})])],
        mesh=MeshSpec(mode='uniform', dx=_TEMPLATE_CELL),
        study=StudySpec(type='frequency_domain', settings={'wavelength_start': 0.85e-6, 'wavelength_stop': 1.2e-6,
                                                            'n_wavelengths': 8}))


def _complex_ez(results, run):
    """complex128 E_z phasor (nx, ny) of the window from the ez_re / ez_im output fields."""
    real = np.asarray(results.field('ez_re').values, dtype=np.float64)
    imaginary = np.asarray(results.field('ez_im').values, dtype=np.float64)
    if real.shape != imaginary.shape:
        raise ValueError(f'{run}: ez_re {real.shape} and ez_im {imaginary.shape} differ in shape')
    return real + 1j * imaginary


def _check_harmonic_vs_fdfd(results_by_name):
    """|E_z| and the complex E_z of the harmonic FDTD vs the FDFD over the whole window, relative to max |E_z (FDFD)|;
    measured = error = the worst deviation in tolerance units."""
    fdtd, fdfd = (_complex_ez(results_by_name[run], run) for run in _RUNS)
    if fdtd.shape != fdfd.shape:
        raise ValueError(f'the FDTD window {fdtd.shape} and the FDFD window {fdfd.shape} differ: the meshes must agree')
    scale = float(np.max(np.abs(fdfd)))
    err_abs = float(np.max(np.abs(np.abs(fdtd) - np.abs(fdfd)))) / scale if scale > 0.0 else math.nan
    err_complex = float(np.max(np.abs(fdtd - fdfd))) / scale if scale > 0.0 else math.nan
    normalized_errors = {'abs': err_abs / _ABS_TOLERANCE, 'complex': err_complex / _COMPLEX_TOLERANCE}
    worst_item = max(normalized_errors, key=lambda item: math.inf if math.isnan(normalized_errors[item])
                     else normalized_errors[item])
    worst = normalized_errors[worst_item]
    mean_phase = float(np.angle(np.sum(fdtd * np.conj(fdfd))))   # ~0 when the source signs agree, ~pi if flipped
    details = {'err_abs': err_abs, 'err_complex': err_complex, 'fdfd_max_abs': scale,
               'fdtd_max_abs': float(np.max(np.abs(fdtd))), 'mean_phase_difference_rad': mean_phase,
               'normalized_errors': normalized_errors, 'worst_item': worst_item}
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': 1.0, 'details': details}


def _slab_spec(run):
    """The hz strip with ('device') or without ('reference') the slab, one frequency_domain study of _SLAB_WAVELENGTHS
    storing hz_re / hz_im per wavelength."""
    size_x, size_y = _SLAB_SIZE
    geometry = [GeometrySpec('source_line', 'line', {'x0': _SLAB_SOURCE_X, 'y0': 0.0, 'x1': _SLAB_SOURCE_X,
                                                     'y1': size_y})]
    materials = []
    if run == 'device':
        geometry.append(GeometrySpec('slab', 'rectangle', {'x0': _SLAB_X[0], 'y0': 0.0, 'x1': _SLAB_X[1],
                                                           'y1': size_y}))
        materials.append(MaterialSpec('dielectric', properties={'eps_r': _SLAB_EPS_R}, selection=['slab']))
    return ModelSpec(
        title=f'Self-test: {_SLAB_PHYSICS_TYPE} slab, harmonic FDTD ({run})',
        domain=DomainSpec(size_x=size_x, size_y=size_y), geometry=geometry, materials=materials,
        physics=[PhysicsSpec(_PHYSICS_NAME, _SLAB_PHYSICS_TYPE,
                             settings={'bc_y_lo': 'periodic', 'bc_y_hi': 'periodic',
                                       'courant_factor': _COURANT_FACTOR},
                             features=[FeatureSpec('source', 'line_source', ['source_line'],
                                                   {'waveform': 'gaussian', 'wavelength': _WAVELENGTH,
                                                    'bandwidth': _SLAB_BANDWIDTH, 'amplitude': _AMPLITUDE})])],
        mesh=MeshSpec(mode='uniform', dx=_SLAB_CELL),
        study=StudySpec(type='frequency_domain',
                        settings={'wavelengths': list(_SLAB_WAVELENGTHS), 'auto_shutoff': _AUTO_SHUTOFF},
                        stored_fields=[f'{_PHYSICS_NAME}.{name}' for name in _READ_FIELDS.values()]))


def _discrete_wavenumber(wavelength, index):
    """k [rad/m] of the 1D Yee dispersion sin(k dx / 2) = index dx sin(omega dt / 2) / (C0 dt) on the slab strip
    mesh."""
    dt = _COURANT_FACTOR * _SLAB_CELL / (C0 * math.sqrt(2.0))
    omega = 2.0 * math.pi * C0 / wavelength
    return 2.0 / _SLAB_CELL * math.asin(index * _SLAB_CELL * math.sin(0.5 * omega * dt) / (C0 * dt))


def airy_slab_transmission(wavelength):
    """Complex field transmission of the slab at normal incidence, referred to the incident wave at the same point
    behind it: t12 t21 e^{-j k1 d} / (1 - r21^2 e^{-2 j k1 d}) e^{+j k0 d} (exp(+j omega t), discrete Yee k; the
    discrete impedance is exactly eta0 / n, so the interface coefficients are the continuum ones)."""
    index = math.sqrt(_SLAB_EPS_R)
    thickness = _SLAB_X[1] - _SLAB_X[0]
    k_slab, k_vacuum = _discrete_wavenumber(wavelength, index), _discrete_wavenumber(wavelength, 1.0)
    r_inner = (index - 1.0) / (index + 1.0)
    t_in, t_out = 2.0 / (1.0 + index), 2.0 * index / (1.0 + index)
    round_trip = cmath.exp(-2j * k_slab * thickness)
    return (t_in * t_out * cmath.exp(-1j * k_slab * thickness) / (1.0 - r_inner ** 2 * round_trip)
            * cmath.exp(1j * k_vacuum * thickness))


def _column_mean(field_data):
    """Mean of a snapshot field over the window cells whose x centre lies in [_READ_X, _READ_X + _SLAB_CELL)."""
    values = np.asarray(field_data.values, dtype=np.float64)
    x_centers = field_data.x_centers
    if x_centers is None:   # uniform cells over the extent
        x0, x1 = field_data.extent[0], field_data.extent[1]
        x_centers = x0 + (np.arange(values.shape[0]) + 0.5) * (x1 - x0) / values.shape[0]
    column = (np.asarray(x_centers) >= _READ_X) & (np.asarray(x_centers) < _READ_X + _SLAB_CELL)
    if np.count_nonzero(column) != 1:
        raise ValueError(f'expected one read column at x = {_READ_X:g} m, found {np.count_nonzero(column)}')
    return float(np.mean(values[column]))


def _read_phasors(results, run):
    """complex128 H_z on the read column per _SLAB_WAVELENGTHS entry, from the snapshot of that solved wavelength."""
    solved = [float(value) for value in results.summary().get('wavelengths') or []]
    phasors = []
    for wavelength in _SLAB_WAVELENGTHS:
        index = next((k for k, value in enumerate(solved) if abs(value / wavelength - 1.0) < 1e-9), None)
        if index is None:
            raise ValueError(f'{run}: wavelength {wavelength:g} m not among the solved {solved}')
        real, imaginary = (_column_mean(results.snapshot_field(index, _READ_FIELDS[part])) for part in ('re', 'im'))
        phasors.append(complex(real, imaginary))
    return np.asarray(phasors, dtype=np.complex128)


def _check_slab_airy(results_by_name):
    """t = H_z(device) / H_z(reference) behind the slab vs airy_slab_transmission, and eta0 |H_z(reference)| vs the
    source amplitude, at every swept wavelength; measured = error = the worse in tolerance units."""
    device, reference = (_read_phasors(results_by_name[run], run) for run in _SLAB_RUNS)
    with np.errstate(divide='ignore', invalid='ignore'):
        transmission = device / reference
    expected = np.array([airy_slab_transmission(wavelength) for wavelength in _SLAB_WAVELENGTHS])
    incident_ratio = MU0 * C0 * np.abs(reference) / _AMPLITUDE
    err_t = float(np.max(np.abs(transmission - expected)))   # NaN propagates
    err_incident = float(np.max(np.abs(incident_ratio - 1.0)))
    normalized_errors = {'transmission': err_t / _SLAB_T_TOLERANCE, 'incident': err_incident / _INCIDENT_TOLERANCE}
    worst = max(math.inf if math.isnan(value) else value for value in normalized_errors.values())
    details = {'wavelengths': float_list(_SLAB_WAVELENGTHS), 'err_transmission': err_t,
               'err_incident': err_incident, 'normalized_errors': normalized_errors,
               't_abs': float_list(np.abs(transmission)), 't_phase_rad': float_list(np.angle(transmission)),
               't_airy_abs': float_list(np.abs(expected)), 't_airy_phase_rad': float_list(np.angle(expected)),
               'incident_ratio': float_list(incident_ratio)}
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': 1.0, 'details': details}


HARMONIC_FDTD_SELFTEST_CASES = (
    SelfTestCase(
        name='em_harmonic_fdtd_vs_fdfd',
        label='EM harmonic FDTD vs FDFD',
        description='A vertical line source (amplitude 1 V/m) spans the period of a y-periodic strip (4 x 0.5 um, CPML '
                    'along x, 20 nm mesh) and lights a periodic row of dielectric cylinders (eps_r 4, r = 0.15 um; '
                    'period below lambda / n, so no diffraction order or guided resonance). Two runs with one '
                    'frequency_domain study at 1 um: em_fdtd_ez with a gaussian pulse centred at 1 um (bandwidth 0.5; '
                    'harmonic FDTD: window DFT divided by the source spectrum, auto_shutoff 1e-6) and em_fdfd_ez with '
                    'the phase-0 line source. Over every window cell, max | |E_fdtd| - |E_fdfd| | / max |E_fdfd| must '
                    'stay within 0.03 and max |E_fdtd - E_fdfd| / max |E_fdfd| within 0.05; measured = error = the '
                    'worse of the two in tolerance units, expected 0, tolerance 1. The complex comparison is the '
                    'absolute cross-check of both source conventions (a flipped FDFD source sign gives an error near '
                    '2, mean_phase_difference_rad near pi). Budget: FDTD numerical dispersion against the FDFD stencil '
                    '(O(dx^2) at a fixed Courant number, ~8e-4 in k at 50 cells per wavelength: a phase lag ~1.3e-2 '
                    'rad over the 3.5 um path, more inside the cylinder at 25 cells per wavelength), the different '
                    'CPML stretches (real kappa folded vs complex), the truncated ring-down (energy 1e-6 of its peak) '
                    'and complex64 DFT sums; the sheet calibration is exact for both at the pulse centre; the '
                    'staircased cylinder is shared by both runs. 20 nm mesh not yet confirmed by a run.',
        specs=_harmonic_specs,
        check=_check_harmonic_vs_fdfd),
    SelfTestCase(
        name='em_hz_harmonic_fdtd_slab',
        label='EM H_z harmonic FDTD slab vs Airy',
        description='em_fdtd_hz in a frequency_domain study (harmonic FDTD): a vertical line source (amplitude 1 V/m) '
                    'spans the period of a y-periodic strip (4 x 0.1 um, CPML along x, 12.5 nm mesh) and sends an '
                    'E_y / H_z gaussian pulse (1 um, bandwidth 0.3) onto a 0.5 um slab (eps_r 4, n = 2, x = 1.5..2.0 '
                    'um). One frequency_domain study of three wavelengths (0.9, 1.0, 1.1 um from the one pulse: off '
                    'the carrier the source spectrum S(omega) is 0.87 / 0.91 of its peak, so the normalisation by S '
                    'at each frequency is checked where it matters) stores hz_re / hz_im per wavelength, read on a '
                    'one-cell column 1 um behind the slab (snapshot of each solved wavelength); a vacuum reference '
                    'run gives the incident phasor there. t = device / reference must match '
                    'the Airy transmission t12 t21 e^{-j k1 d} / (1 - r21^2 e^{-2 j k1 d}) e^{+j k0 d} with the '
                    'discrete Yee wavenumbers (|t|^2 = 0.81 / 1 / 0.86; the discrete impedance is exactly eta0 / n) '
                    'within 0.01 (complex), and eta0 |H_z| of the reference the amplitude within 0.01; measured = '
                    'error = the worse of the two in tolerance units, expected 0, tolerance 1. Budget: the face-mean '
                    'eps on the two slab-edge E_y faces (O(dx^2) interface shift), CPML reflection, the truncated '
                    'ring-down (auto_shutoff 1e-6), complex64 DFT sums, the sheet calibration away from f0 (vacuum: '
                    'cos(kappa / 2) changes by 2e-4), float32 snapshot values. Multi-wavelength study (replacing a '
                    'sweep of the single wavelength) not yet confirmed by a run.',
        specs=lambda: {run: _slab_spec(run) for run in _SLAB_RUNS},
        check=_check_slab_airy),
)
