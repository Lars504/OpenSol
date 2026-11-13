"""Self-tests of the em_fdtd_3d plane wave, mode ports and far field: normal-incidence Fresnel on a dielectric
half-space and on a Drude film (periodic pole faces), mode launch / overlap transmission in a straight 3D waveguide,
and the Hertzian dipole far-field pattern."""
import math

import numpy as np

from ...core.constants import C0, ETA0
from ...model.schema import SelfTestCase, SelfTestVariant
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           RefinementSpec, StudySpec)
from .selftests import (DEFAULT_COURANT, DIPOLE_TOLERANCE, SELFTEST_DURATION, cubic_time_step, dipole_flux_errors,
                        dipole_selftest_spec)

_PHYSICS_NAME = 'em'
_PHYSICS_TYPE = 'em_fdtd_3d'          # literal (selftests never import the definition)

# ---- plane wave on a dielectric half-space (periodic x, y: a one-sided TFSF plane) ----
_PW_CELL = 40e-9                      # m: 25 cells per vacuum wavelength at 1 um, ~17 in the n = 1.5 medium
_PW_PERIOD = 4 * _PW_CELL             # m transverse period (x and y)
_PW_LENGTH = 2.0e-6                   # m along z (50 cells) + 10 CPML cells per z face
_PW_SOURCE_Z = 0.4e-6                 # m upstream TFSF face
_PW_INTERFACE_Z = 1.2e-6              # m start of the half-space (runs into the z_hi CPML)
_PW_EPS = 2.25                        # n = 1.5: R = 0.04, T = 0.96
_PW_PLANES = {'reflected': 0.2e-6, 'incident': 0.8e-6, 'transmitted': 1.6e-6}   # flux planes z [m]
_PW_PULSE = {'direction': '+z', 'polarization': 'x', 'waveform': 'gaussian', 'wavelength': 1.0e-6, 'bandwidth': 0.3,
             'amplitude': 1.0}
_PW_BAND = {'wavelength_start': 0.95e-6, 'wavelength_stop': 1.05e-6, 'n_wavelengths': 3}
# s: pulse 2 t0 ~32 fs (envelope e^-20 there), + ~6 fs to the interface and back past the reflected plane; the
# validation bound 2 t0 + diagonal x n_max / c is ~42 fs
_PW_DURATION = 45e-15
_PW_TOLERANCE = 5e-3

# ---- Drude film on the same column (pole face weights on the periodic wrap faces) ----
_DF_COURANT = 0.9                     # the leap-frog Drude needs eps_inf >= courant^2 + (omega_p dt)^2 / 4 (0.85 here)
_DF_OMEGA_P = 6.0e15                  # rad/s: eps ~ -9 - 1j at 1 um (skin depth ~50 nm)
_DF_GAMMA = 2.0e14                    # rad/s
_DF_FILM = (1.2e-6, 1.28e-6)          # m: 2 cells in vacuum, both interfaces on H nodes

# ---- closed TFSF box on a graded z axis (vacuum: every scattered-field flux is leakage) ----
_GT_BOX_Z = (0.4e-6, 1.6e-6)          # m closed total-field box along z, both faces on 40 nm cells
_GT_FINE_Z = (0.8e-6, 1.2e-6)         # m refined to _GT_FINE_CELL along z only (ramps inside the box)
_GT_FINE_CELL = 20e-9                 # m: ratio 2 still grades inside the box; sets dt (finest dz)
_GT_PLANES = {'upstream': 0.2e-6, 'incident': 0.6e-6, 'downstream': 1.8e-6}   # flux planes z [m]
_GT_TOLERANCE = 1e-6                  # scattered / incident power

# ---- closed TFSF box in vacuum, CPML on every face (side patches, -d direction, other polarisation axes) ----
# the scattered field is zero by construction: the CPML only meets a leak (its reflection is second order), so a thin
# layer and a window 2 cells beyond the flux box do
_CB_SIZE = 0.96e-6                    # m cube window (24 cells of _PW_CELL) + _CB_PML_CELLS CPML cells per face
_CB_PML_CELLS = 4
_CB_BOX = (0.28e-6, 0.68e-6)          # m total-field box along every axis (cell edges)
_CB_FLUX = (0.08e-6, 0.88e-6)         # m scattered-field flux box around it
_CB_RUNS = {'minus_y_ez': ('-y', 'z'), 'plus_x_ey': ('+x', 'y')}   # run: (direction, polarization)
_CB_DURATION = 50e-15                 # s: the pulse (2 t0 ~32 fs) has crossed the box
_CB_TOLERANCE = 1e-6                  # scattered / incident power, as _GT_TOLERANCE

# ---- straight waveguide: mode source -> mode monitors / flux plane ----
# the graded quantities rest on the port mode being the discrete Yee eigenmode (beta_d = (2/D) arcsin(K D/2)), so they
# are resolution-independent up to the PML stretch, not dispersion errors
_WG_CELL = 1.0e-6 / 30.0              # m: 15 cells per core wavelength at 1 um, core 15 x 9 cells
_WG_SIZE = (1.5e-6, 1.5e-6, 1.2e-6)   # m window (45 x 45 x 36 cells) + 10 CPML cells per face
_WG_CORE = (0.5e-6, 0.3e-6)           # m core width (x) and height (y): mode 1 is the non-degenerate E_x mode
_WG_EPS = 4.0                         # core n = 2 in vacuum
_WG_SOURCE_Z = 9 * _WG_CELL           # 0.3 um
_WG_UPSTREAM_Z = 5 * _WG_CELL         # 0.167 um: mode monitor in the scattered field, backward leakage of the launch
_WG_DOWNSTREAM_Z = 27 * _WG_CELL      # 0.9 um: mode monitor and flux plane downstream
# the dielectric guide's fundamental has no cutoff, so the wider spectrum keeps clear of the metal-guide cutoff gotcha
_WG_SOURCE = {'normal': 'z', 'position': _WG_SOURCE_Z, 'mode_number': 1, 'direction': 'positive',
              'waveform': 'gaussian', 'wavelength': 1.0e-6, 'bandwidth': 0.3, 'power': 1.0}
_WG_BAND = {'wavelength_start': 0.98e-6, 'wavelength_stop': 1.02e-6, 'n_wavelengths': 3}
_WG_DURATION = 55e-15                 # s: pulse 2 t0 ~32 fs plus the group delay; validation bound ~48 fs
_WG_TOLERANCE = 0.02

# ---- dipole far field and flux box (one run: the far field shares the flux box's DFT box) ----
_FF_MOMENT = 1e-12                    # A m (= selftests._MOMENT, dipole_vacuum_spec)
_FF_ANGLES = 36
_FF_MIN_PATTERN = 0.25                # compare only where cos^2(angle) >= this (away from the nulls along z)
_HALF_STORAGE_VARIANT = 'half_storage'
_HALF_STORAGE_TOLERANCE = 0.06        # the 1-2 % discretisation error plus the fp16 storage noise (~1e-3 per field)
_HALF_STORAGE_NODE = f'{_HALF_STORAGE_VARIANT}_{_PHYSICS_NAME}_stepper'   # node name of api._variant_spec


def _column_spec(title, layer_z, material, courant=DEFAULT_COURANT, direction='+z'):
    """Periodic 4 x 4 cell column along z: a one-sided TFSF plane wave (+z, E_x) onto the layer box between heights
    layer_z (filled with `material`), flux planes _PW_PLANES; direction '-z' mirrors every height (z -> L - z)."""
    period = _PW_PERIOD

    def height(z):
        return z if direction == '+z' else _PW_LENGTH - z
    bc = {f'bc_{axis}_{side}': 'periodic' for axis in ('x', 'y') for side in ('lo', 'hi')}
    flux_features = [FeatureSpec(name, 'flux_plane', [], {'normal': 'z', 'position': height(z), **_PW_BAND})
                     for name, z in _PW_PLANES.items()]
    source_z, layer_z = sorted((height(_PW_SOURCE_Z), height(_PW_LENGTH))), sorted(height(z) for z in layer_z)
    return ModelSpec(
        title=title,
        domain=DomainSpec(size_x=period, size_y=period, size_z=_PW_LENGTH),
        geometry=[GeometrySpec('tfsf_region', 'box', {'x0': 0.0, 'y0': 0.0, 'z0': source_z[0], 'x1': period,
                                                      'y1': period, 'z1': source_z[1]}),
                  GeometrySpec('layer', 'box', {'x0': 0.0, 'y0': 0.0, 'z0': layer_z[0], 'x1': period, 'y1': period,
                                                'z1': layer_z[1]})],
        materials=[material],
        physics=[PhysicsSpec(_PHYSICS_NAME, _PHYSICS_TYPE, settings={'pml_cells': 10, 'courant_factor': courant, **bc},
                             features=[FeatureSpec('plane_wave', 'plane_wave_source', ['tfsf_region'],
                                                   {**_PW_PULSE, 'direction': direction}),
                                       *flux_features])],
        mesh=MeshSpec(mode='uniform', dx=_PW_CELL),
        study=StudySpec(type='time_dependent', settings={'duration': _PW_DURATION}))


def plane_wave_half_space_spec(direction='+z'):
    """Periodic 4 x 4 cell column along z: a one-sided TFSF plane wave (+z, E_x; '-z' mirrored) onto an eps 2.25
    half-space."""
    return _column_spec('Self-test: 3D plane wave on a dielectric half-space', (_PW_INTERFACE_Z, _PW_LENGTH),
                        MaterialSpec('dielectric', properties={'eps_r': _PW_EPS}, selection=['layer']),
                        direction=direction)


def _pw_time_step(courant=DEFAULT_COURANT):
    """3D CFL time step [s] of the uniform cubic plane-wave grid."""
    return cubic_time_step(_PW_CELL, courant)


def _half_kappa(frequency, index):
    """kappa / 2 (kappa = numerical k x cell) of a normal-incidence plane wave in a medium of refractive `index`."""
    dt = _pw_time_step()
    return np.arcsin(np.sin(np.pi * np.asarray(frequency, dtype=np.float64) * dt) * index * _PW_CELL / (C0 * dt))


def _discrete_incident_power(frequency):
    """cw power [W] of the unit plane wave through the period as the flux planes read it: A^2 / (2 eta0) x area x
    cos(kappa/2) cos(omega dt/2) x (cos(kappa0/2) / cos(kappa/2))^2, the last factor the aux soft source (exact at f0)."""
    half_omega_dt = np.pi * np.asarray(frequency, dtype=np.float64) * _pw_time_step()
    half_kappa = _half_kappa(frequency, 1.0)
    carrier_half_kappa = _half_kappa(C0 / _PW_PULSE['wavelength'], 1.0)
    amplitude = _PW_PULSE['amplitude']
    return (amplitude ** 2 / (2.0 * float(ETA0)) * _PW_PERIOD ** 2 * np.cos(carrier_half_kappa) ** 2
            / np.cos(half_kappa) * np.cos(half_omega_dt))


def _discrete_fresnel_reflectance(frequency):
    """Yee normal-incidence R of a vacuum / eps interface on an H node: (sin((kappa2 - kappa1)/2) / sin((kappa2 +
    kappa1)/2))^2 (-> ((n-1)/(n+1))^2 for fine cells); T = 1 - R for the flux-plane power."""
    vacuum, medium = _half_kappa(frequency, 1.0), _half_kappa(frequency, math.sqrt(_PW_EPS))
    return (np.sin(medium - vacuum) / np.sin(medium + vacuum)) ** 2


def _check_fresnel(results_by_name):
    """R, T from the flux planes (incident = incident-plane flux + reflected power; the '-z' run's fluxes negated) vs
    the discrete Fresnel R, 1 - R, and the incident power vs the discrete plane-wave flux; worst absolute deviation
    over both runs."""
    worst, details = 0.0, {}
    for run, sign in (('run', 1.0), ('minus_z', -1.0)):
        results = results_by_name[run]
        flux = {name: sign * np.asarray(results.result(_PHYSICS_NAME, name).get('power_per_cw'), dtype=np.float64)
                for name in _PW_PLANES}
        frequency = np.asarray(results.result(_PHYSICS_NAME, 'incident')['frequency'], dtype=np.float64)
        incident = flux['incident'] - flux['reflected']
        reflectance, transmittance = -flux['reflected'] / incident, flux['transmitted'] / incident
        expected_r = _discrete_fresnel_reflectance(frequency)
        expected_t = 1.0 - expected_r
        incident_ratio = incident / _discrete_incident_power(frequency)
        deviations = np.concatenate((np.abs(reflectance - expected_r), np.abs(transmittance - expected_t),
                                     np.abs(incident_ratio - 1.0)))
        deviations = np.where(np.isfinite(deviations), deviations, np.inf)
        worst = max(worst, float(np.max(deviations)))
        details[run] = {'reflectance': reflectance.tolist(), 'transmittance': transmittance.tolist(),
                        'incident_power_ratio': incident_ratio.tolist()}
        if run == 'run':
            measured, expected = float(transmittance[1]), float(expected_t[1])
            details.update(wavelengths=(C0 / frequency).tolist(), expected_reflectance=expected_r.tolist(),
                           expected_transmittance=expected_t.tolist())
    index = math.sqrt(_PW_EPS)
    details['continuum_reflectance'] = ((index - 1.0) / (index + 1.0)) ** 2
    return {'measured': measured, 'expected': expected, 'error': worst, 'tolerance': _PW_TOLERANCE,
            'details': details}


def drude_film_spec():
    """The periodic column with a 2-cell Drude film (eps_inf 1) in vacuum instead of the half-space, courant 0.9."""
    metal = MaterialSpec('drude_metal', properties={}, selection=['layer'], dispersion=[
        {'model': 'drude', 'omega_p': _DF_OMEGA_P, 'gamma': _DF_GAMMA, 'eps_inf': 1.0}])
    return _column_spec('Self-test: 3D plane wave on a Drude film (periodic column)', _DF_FILM, metal, _DF_COURANT)


def _discrete_film_amplitudes(frequency):
    """(r, t) complex128 of the film on the 1D Yee line at normal incidence: E at the cell centres, interfaces on H
    nodes, where the two media's solutions agree on both neighbouring cells; leap-frog Drude chi at the grid frequency."""
    dt = _pw_time_step(_DF_COURANT)
    half_omega_dt = np.pi * np.asarray(frequency, dtype=np.float64) * dt
    grid_omega = 2.0 * np.sin(half_omega_dt) / dt
    chi = -_DF_OMEGA_P ** 2 / (grid_omega ** 2 - 1j * _DF_GAMMA * grid_omega * np.cos(half_omega_dt))
    kappa_vacuum = 2.0 * np.arcsin(grid_omega * _PW_CELL / (2.0 * C0)).astype(np.complex128)
    kappa_film = 2.0 * np.arcsin(grid_omega * _PW_CELL * np.sqrt(1.0 + chi) / (2.0 * C0))
    n_film = int(round((_DF_FILM[1] - _DF_FILM[0]) / _PW_CELL))
    reflection, transmission = [], []
    for k0, k1 in zip(kappa_vacuum, kappa_film):
        # unknowns (r, A, B, t): E = e^{-j k0 m} + r e^{j k0 m} before (m < 0), A e^{-j k1 m} + B e^{j k1 m} in the
        # film (0 <= m < n), t e^{-j k0 m} after; matched at m = -1, 0 and m = n - 1, n
        matrix = np.array([
            [np.exp(-1j * k0), -np.exp(1j * k1), -np.exp(-1j * k1), 0.0],
            [1.0, -1.0, -1.0, 0.0],
            [0.0, np.exp(-1j * k1 * (n_film - 1)), np.exp(1j * k1 * (n_film - 1)), -np.exp(-1j * k0 * (n_film - 1))],
            [0.0, np.exp(-1j * k1 * n_film), np.exp(1j * k1 * n_film), -np.exp(-1j * k0 * n_film)]],
            dtype=np.complex128)
        r, _, _, t = np.linalg.solve(matrix, np.array([-np.exp(1j * k0), -1.0, 0.0, 0.0], dtype=np.complex128))
        reflection.append(r)
        transmission.append(t)
    return np.array(reflection), np.array(transmission)


def _check_drude_film(results_by_name):
    """R, T of the film from the flux planes (as _check_fresnel) vs |r|^2, |t|^2 of the discrete 1D film; worst
    absolute deviation."""
    results = results_by_name['run']
    flux = {name: np.asarray(results.result(_PHYSICS_NAME, name).get('power_per_cw'), dtype=np.float64)
            for name in _PW_PLANES}
    frequency = np.asarray(results.result(_PHYSICS_NAME, 'incident')['frequency'], dtype=np.float64)
    incident = flux['incident'] - flux['reflected']
    reflectance, transmittance = -flux['reflected'] / incident, flux['transmitted'] / incident
    r, t = _discrete_film_amplitudes(frequency)
    expected_r, expected_t = np.abs(r) ** 2, np.abs(t) ** 2
    deviations = np.concatenate((np.abs(reflectance - expected_r), np.abs(transmittance - expected_t)))
    deviations = np.where(np.isfinite(deviations), deviations, np.inf)
    return {'measured': float(transmittance[1]), 'expected': float(expected_t[1]), 'error': float(np.max(deviations)),
            'tolerance': _PW_TOLERANCE,
            'details': {'wavelengths': (C0 / frequency).tolist(), 'reflectance': reflectance.tolist(),
                        'transmittance': transmittance.tolist(), 'expected_reflectance': expected_r.tolist(),
                        'expected_transmittance': expected_t.tolist(),
                        'absorptance': (1.0 - reflectance - transmittance).tolist()}}


def graded_tfsf_box_spec():
    """The periodic vacuum column with a closed TFSF box (z _GT_BOX_Z) whose middle is refined along z only."""
    period = _PW_PERIOD
    bc = {f'bc_{axis}_{side}': 'periodic' for axis in ('x', 'y') for side in ('lo', 'hi')}
    flux_features = [FeatureSpec(name, 'flux_plane', [], {'normal': 'z', 'position': z, **_PW_BAND})
                     for name, z in _GT_PLANES.items()]
    return ModelSpec(
        title='Self-test: 3D TFSF box on a graded z axis',
        domain=DomainSpec(size_x=period, size_y=period, size_z=_PW_LENGTH),
        geometry=[GeometrySpec('tfsf_region', 'box', {'x0': 0.0, 'y0': 0.0, 'z0': _GT_BOX_Z[0], 'x1': period,
                                                      'y1': period, 'z1': _GT_BOX_Z[1]}),
                  GeometrySpec('fine', 'box', {'x0': 0.0, 'y0': 0.0, 'z0': _GT_FINE_Z[0], 'x1': period, 'y1': period,
                                               'z1': _GT_FINE_Z[1]})],
        physics=[PhysicsSpec(_PHYSICS_NAME, _PHYSICS_TYPE, settings={'pml_cells': 10, **bc},
                             features=[FeatureSpec('plane_wave', 'plane_wave_source', ['tfsf_region'], dict(_PW_PULSE)),
                                       *flux_features])],
        mesh=MeshSpec(mode='nonuniform', dx=_PW_CELL,
                      refinements=[RefinementSpec('fine', _PW_CELL, 0.0, dx_z=_GT_FINE_CELL)]),
        study=StudySpec(type='time_dependent', settings={'duration': _PW_DURATION}))


def _check_graded_tfsf_box(results_by_name):
    """Largest scattered-field flux (upstream / downstream of the closed box) over the total-field flux inside it, per
    wavelength; raises unless the build has more z cells than the uniform column (graded z active)."""
    results = results_by_name['run']
    build = results.summary().get('build') or {}
    uniform_nz = round(_PW_LENGTH / _PW_CELL) + 2 * 10
    if not int(build.get('nz', 0)) > uniform_nz:
        raise ValueError(f"expected a graded z axis (nz > {uniform_nz}), got build nz {build.get('nz')}")
    flux = {name: np.asarray(results.result(_PHYSICS_NAME, name).get('power_per_cw'), dtype=np.float64)
            for name in _GT_PLANES}
    leakage = np.maximum(np.abs(flux['upstream']), np.abs(flux['downstream'])) / flux['incident']
    leakage = np.where(np.isfinite(leakage) & (flux['incident'] > 0.0), leakage, np.inf)
    frequency = np.asarray(results.result(_PHYSICS_NAME, 'incident')['frequency'], dtype=np.float64)
    return {'measured': float(np.max(leakage)), 'expected': 0.0, 'error': float(np.max(leakage)),
            'tolerance': _GT_TOLERANCE,
            'details': {'wavelengths': (C0 / frequency).tolist(), 'leakage': leakage.tolist(),
                        'incident_power_per_cw': flux['incident'].tolist(), 'nz': int(build['nz']),
                        'dx_min': build.get('dx_min')}}


def closed_tfsf_box_spec(direction, polarization):
    """Vacuum cube, CPML on every face: a TFSF box closed on all six sides inside a scattered-field flux box."""
    def box(name, bounds):
        low, high = bounds
        return GeometrySpec(name, 'box', {'x0': low, 'y0': low, 'z0': low, 'x1': high, 'y1': high, 'z1': high})
    return ModelSpec(
        title=f'Self-test: closed 3D TFSF box ({direction}, E_{polarization})',
        domain=DomainSpec(size_x=_CB_SIZE, size_y=_CB_SIZE, size_z=_CB_SIZE),
        geometry=[box('tfsf_region', _CB_BOX), box('flux_surface', _CB_FLUX)],
        physics=[PhysicsSpec(_PHYSICS_NAME, _PHYSICS_TYPE, settings={'pml_cells': _CB_PML_CELLS}, features=[
            FeatureSpec('plane_wave', 'plane_wave_source', ['tfsf_region'],
                        {**_PW_PULSE, 'direction': direction, 'polarization': polarization}),
            FeatureSpec('scattered', 'flux_box', ['flux_surface'], dict(_PW_BAND))])],
        mesh=MeshSpec(mode='uniform', dx=_PW_CELL),
        study=StudySpec(type='time_dependent', settings={'duration': _CB_DURATION}))


def _check_closed_tfsf_box(results_by_name):
    """Largest |net scattered-field flux| out of the box around the closed TFSF box over the incident power through the
    box cross-section (the discrete plane-wave flux of _discrete_incident_power on that area), per run and wavelength."""
    area_ratio = ((_CB_BOX[1] - _CB_BOX[0]) / _PW_PERIOD) ** 2
    leakages = {}
    for run in _CB_RUNS:
        result = results_by_name[run].result(_PHYSICS_NAME, 'scattered')
        frequency = np.asarray(result['frequency'], dtype=np.float64)
        scattered = np.asarray(result.get('power_per_cw'), dtype=np.float64)
        leakage = np.abs(scattered) / (_discrete_incident_power(frequency) * area_ratio)
        leakages[run] = np.where(np.isfinite(leakage), leakage, np.inf)
    error = max(float(np.max(values)) for values in leakages.values())
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': _CB_TOLERANCE,
            'details': {'wavelengths': (C0 / frequency).tolist(),
                        'leakage': {run: values.tolist() for run, values in leakages.items()},
                        'runs': {run: {'direction': direction, 'polarization': polarization}
                                 for run, (direction, polarization) in _CB_RUNS.items()}}}


def waveguide_mode_spec():
    """Straight n = 2 rectangular core along z in vacuum: mode source, upstream / downstream mode monitors, flux plane."""
    size_x, size_y, size_z = _WG_SIZE
    width, height = _WG_CORE
    core = {'x0': 0.5 * (size_x - width), 'x1': 0.5 * (size_x + width), 'y0': 0.5 * (size_y - height),
            'y1': 0.5 * (size_y + height), 'z0': 0.0, 'z1': size_z}
    return ModelSpec(
        title='Self-test: 3D waveguide mode launch and transmission',
        domain=DomainSpec(size_x=size_x, size_y=size_y, size_z=size_z),
        geometry=[GeometrySpec('core', 'box', core)],
        materials=[MaterialSpec('core_material', properties={'eps_r': _WG_EPS}, selection=['core'])],
        physics=[PhysicsSpec(_PHYSICS_NAME, _PHYSICS_TYPE, settings={'pml_cells': 10}, features=[
            FeatureSpec('launch', 'mode_source', [], dict(_WG_SOURCE)),
            FeatureSpec('upstream', 'mode_monitor', [], {'normal': 'z', 'position': _WG_UPSTREAM_Z, 'mode_number': 1,
                                                         **_WG_BAND}),
            FeatureSpec('downstream', 'mode_monitor', [], {'normal': 'z', 'position': _WG_DOWNSTREAM_Z,
                                                           'mode_number': 1, **_WG_BAND}),
            FeatureSpec('through', 'flux_plane', [], {'normal': 'z', 'position': _WG_DOWNSTREAM_Z, **_WG_BAND})])],
        mesh=MeshSpec(mode='uniform', dx=_WG_CELL),
        study=StudySpec(type='time_dependent', settings={'duration': _WG_DURATION}))


def _check_waveguide(results_by_name):
    """At the source wavelength: forward mode fraction downstream and flux / source power ~ 1, backward leakage
    upstream ~ 0; error = the worst of |fraction - 1|, |flux / P - 1| and the leakage."""
    results = results_by_name['run']
    downstream = results.result(_PHYSICS_NAME, 'downstream')['series']
    upstream = results.result(_PHYSICS_NAME, 'upstream')['series']
    flux = np.asarray(results.result(_PHYSICS_NAME, 'through').get('power_per_cw'), dtype=np.float64)
    centre = _WG_BAND['n_wavelengths'] // 2
    forward = float(np.asarray(downstream['fraction_positive'])[centre])
    leakage = float(np.asarray(upstream['fraction_negative'])[centre])
    flux_ratio = float(flux[centre]) / _WG_SOURCE['power']
    deviations = [abs(forward - 1.0), abs(flux_ratio - 1.0), abs(leakage)]
    error = max(deviation if math.isfinite(deviation) else math.inf for deviation in deviations)
    return {'measured': forward, 'expected': 1.0, 'error': error, 'tolerance': _WG_TOLERANCE,
            'details': {'forward_fraction': np.asarray(downstream['fraction_positive']).tolist(),
                        'backward_fraction_downstream': np.asarray(downstream['fraction_negative']).tolist(),
                        'backward_leakage_upstream': np.asarray(upstream['fraction_negative']).tolist(),
                        'flux_per_source_power': (flux / _WG_SOURCE['power']).tolist()}}


def dipole_far_field_spec():
    """The dipole self-test model plus a far-field box on the flux box outline (x-z cut, the flux box wavelengths)."""
    spec = dipole_selftest_spec()
    spec.physics[0].features.append(FeatureSpec('far_field', 'far_field', ['flux_surface'], {
        'wavelength_start': 0.9e-6, 'wavelength_stop': 1.1e-6, 'n_wavelengths': 3, 'cut_plane': 'xz',
        'n_angles': _FF_ANGLES}))
    return spec


def _check_far_field(results_by_name):
    """Worst relative error of the far-field power_per_cw [W/sr] in the x-z plane vs eta0 k^2 (I l)^2 cos^2(angle) /
    (32 pi^2) (angle from +x, where cos^2 >= 0.25) and of the flux box power vs eta0 k^2 (I l)^2 / (12 pi); raises
    when the half_storage variant's stepper did not run half_storage."""
    results = results_by_name['run']
    roles = ((results.summary().get('solvers') or {}).get(_PHYSICS_NAME) or {}).get('roles') or {}
    stepper = roles.get('stepper') or {}
    if stepper.get('solver') == _HALF_STORAGE_NODE and stepper.get('precision') != 'half_storage':
        raise ValueError(f"expected the stepper node {_HALF_STORAGE_NODE!r} to run half_storage, summary 'solvers' "
                         f"records {stepper.get('precision')!r}")
    flux_error, flux_details = dipole_flux_errors(results)
    result = results.result(_PHYSICS_NAME, 'far_field')
    wavelengths = np.asarray(result['wavelength'], dtype=np.float64)
    angles = np.asarray(result['angles'], dtype=np.float64)
    measured = np.asarray(result.get('power_per_cw', np.full((wavelengths.size, angles.size), np.nan)),
                          dtype=np.float64)
    wavenumbers = 2.0 * math.pi / wavelengths
    pattern = np.cos(angles) ** 2
    expected = float(ETA0) * wavenumbers[:, None] ** 2 * _FF_MOMENT ** 2 * pattern[None, :] / (32.0 * math.pi ** 2)
    used = np.broadcast_to(pattern[None, :] >= _FF_MIN_PATTERN, expected.shape)
    errors = np.where(used, np.abs(measured / expected - 1.0), 0.0)
    errors = np.where(np.isfinite(errors), errors, np.inf)
    worst = np.unravel_index(int(np.argmax(errors)), errors.shape)
    far_field_error = float(errors[worst])
    if flux_error > far_field_error:
        flux_worst = int(np.argmax(flux_details['relative_errors']))
        measured_value, expected_value = flux_details['power_per_cw'][flux_worst], flux_details['expected'][flux_worst]
    else:
        measured_value, expected_value = float(measured[worst]), float(expected[worst])
    return {'measured': measured_value, 'expected': expected_value, 'error': max(far_field_error, flux_error),
            'tolerance': DIPOLE_TOLERANCE,
            'details': {'far_field_error': far_field_error, 'flux_error': flux_error,
                        'stepper_precision': stepper.get('precision'), 'flux': flux_details,
                        'wavelengths': wavelengths.tolist(), 'angles_deg': np.degrees(angles).tolist(),
                        'worst_relative_error_per_wavelength': errors.max(axis=1).tolist(),
                        'broadside_power_per_cw': measured[:, 0].tolist(),
                        'broadside_expected': expected[:, 0].tolist()}}


SOURCE_CASES = (
    SelfTestCase(
        name='em3d_plane_wave_fresnel',
        label='EM 3D: TFSF plane wave on a dielectric half-space (Fresnel)',
        description='em_fdtd_3d, x / y periodic 4 x 4 cell column (40 nm), CPML along z: one-sided TFSF plane wave '
                    '(+z, E_x, gaussian 1 um, bandwidth 0.3) onto an eps 2.25 half-space. Flux planes in the scattered '
                    'field (reflected), in the total field before the interface and in the medium give R and T '
                    'against the discrete Yee Fresnel R = (sin((kappa2-kappa1)/2) / sin((kappa2+kappa1)/2))^2, T = 1 - '
                    'R (continuum 0.04 / 0.96; the grid adds ~2e-3 at 0.95 um) at 0.95-1.05 um, and the incident '
                    'power against the discrete plane-wave flux A^2 / (2 eta0) x area x cos(kappa/2) cos(omega dt/2) '
                    'x (cos(kappa0/2) / cos(kappa/2))^2 (soft source exact at f0); error = worst absolute deviation, '
                    'tolerance 5e-3 (the carrier-matched 1D Mur ends leave < 1e-3; a TFSF sign, face or timing error, '
                    'a wrong aux calibration or periodic flux weights give >= 1e-2). Run minus_z: the same column '
                    'mirrored (z -> L - z, -z wave, one-sided box open onto z_lo), its fluxes negated, against the '
                    f'same references (the -d aux offsets and patch indices). {_PW_DURATION * 1e15:g} fs runs. Not yet '
                    'confirmed by a run.',
        specs=lambda: {'run': plane_wave_half_space_spec(), 'minus_z': plane_wave_half_space_spec('-z')},
        check=_check_fresnel),
    SelfTestCase(
        name='em3d_drude_film_fresnel',
        label='EM 3D: TFSF plane wave on a Drude film (periodic column, pole face weights)',
        description='The em3d_plane_wave_fresnel column (x / y periodic, 4 x 4 cells of 40 nm, courant_factor 0.9) with '
                    'a 2-cell (80 nm) Drude film in vacuum instead of the half-space (omega_p 6e15 rad/s, gamma 2e14 '
                    'rad/s, eps_inf 1: eps ~ -9 - 1j at 1 um). R and T from the flux planes at 0.95-1.05 um vs |r|^2, '
                    '|t|^2 of the same film on the 1D Yee line (E at cell centres, interfaces on H nodes) with the '
                    'leap-frog Drude chi = -omega_p^2 / (W^2 - j gamma W cos(omega dt/2)), W = (2/dt) sin(omega '
                    'dt/2): exact for the discretisation; error = worst absolute deviation, tolerance 5e-3 (TFSF '
                    'leakage < 1e-3). The film spans the period, so every E_x face (also the periodic wrap face) '
                    'carries the full pole; half a pole on the wrap faces (a quarter of them) shifts R and T by a '
                    f'few 1e-2 (hand estimate). Not yet run at {_PW_DURATION * 1e15:g} fs.',
        specs=lambda: {'run': drude_film_spec()},
        check=_check_drude_film),
    SelfTestCase(
        name='em3d_tfsf_graded_leakage',
        label='EM 3D: closed TFSF box on a graded mesh leaks nothing',
        description=(f'The em3d_plane_wave_fresnel column (x / y periodic, 40 nm cells, CPML along z) in vacuum with a '
                     f'closed TFSF box z = {_GT_BOX_Z[0] * 1e6:g}-{_GT_BOX_Z[1] * 1e6:g} um (+z, E_x, gaussian 1 um) on '
                     f'mesh nonuniform: z refined to {_GT_FINE_CELL * 1e9:g} nm in {_GT_FINE_Z[0] * 1e6:g}-'
                     f'{_GT_FINE_Z[1] * 1e6:g} um (dx_z only), so the cells grade inside the box while both faces sit '
                     'on base cells. The 1D incident line runs on the box cells\' own z widths (solver_plan S7.1 '
                     'lift), so it is the 3D update of the transversely uniform wave and the scattered field stays '
                     'zero: error = max over 0.95-1.05 um of the flux through the planes before (0.2 um) and after '
                     '(1.8 um) the box over the flux inside it (0.6 um), tolerance 1e-6 (float32 rounding < 1e-9; an '
                     'incident line on one spacing or face spacings taken as cell widths leak far more, hand '
                     'estimate). Raises unless the build has more z cells than the uniform column. Not yet confirmed '
                     f'by a run (nor at {_GT_FINE_CELL * 1e9:g} nm / {_PW_DURATION * 1e15:g} fs).'),
        specs=lambda: {'run': graded_tfsf_box_spec()},
        check=_check_graded_tfsf_box),
    SelfTestCase(
        name='em3d_tfsf_closed_box',
        label='EM 3D: TFSF box closed on all six sides leaks nothing',
        description=(f'em_fdtd_3d vacuum cube {_CB_SIZE * 1e6:g} um (40 nm cells, {_CB_PML_CELLS} CPML cells on every '
                     'face: it only meets a leak): a TFSF box '
                     f'{_CB_BOX[0] * 1e6:g}-{_CB_BOX[1] * 1e6:g} um on every axis (gaussian 1 um, bandwidth 0.3) inside '
                     f'a scattered-field flux box {_CB_FLUX[0] * 1e6:g}-{_CB_FLUX[1] * 1e6:g} um. Runs minus_y_ez (-y, '
                     'E_z) and plus_x_ey (+x, E_y): the transverse side patches, a -d direction and two polarisation '
                     'axes the periodic columns never reach. The 1D incident line is the 3D update of the uniform wave, '
                     'so the scattered field is exactly zero: error = max over runs and 0.95-1.05 um of |net flux out '
                     'of the flux box| over the incident power through the box cross-section, tolerance 1e-6 (float32 '
                     'rounding; a wrong side-patch sign, index, Levi-Civita sign or -d offset leaks >= 1e-3, hand '
                     'estimate). Not yet confirmed by a run (nor with the thin CPML / tight window).'),
        specs=lambda: {run: closed_tfsf_box_spec(direction, polarization)
                       for run, (direction, polarization) in _CB_RUNS.items()},
        check=_check_closed_tfsf_box),
    SelfTestCase(
        name='em3d_mode_source_transmission',
        label='EM 3D: waveguide mode launch and mode-overlap transmission',
        description=(f'em_fdtd_3d, n = 2 core 0.5 x 0.3 um along z in vacuum ({_WG_CELL * 1e9:.1f} nm cells, window '
                     f'{_WG_SIZE[0] * 1e6:g} x {_WG_SIZE[1] * 1e6:g} x {_WG_SIZE[2] * 1e6:g} um, CPML everywhere): mode '
                     f"source (mode 1, 1 W, gaussian 1 um, bandwidth {_WG_SOURCE['bandwidth']:g}) at z = "
                     f'{_WG_SOURCE_Z * 1e6:.3g} um, {_WG_DURATION * 1e15:g} fs. At 1 um: forward fraction of a mode '
                     f'monitor at z = {_WG_DOWNSTREAM_Z * 1e6:.3g} um and a flux plane there / 1 W ~ 1, backward '
                     f'fraction of a mode monitor in the scattered field at z = {_WG_UPSTREAM_Z * 1e6:.3g} um ~ 0; '
                     'error = the worst of the three deviations, tolerance 2e-2 (the cross-section mode is the '
                     'discrete 3D eigenmode up to the PML stretch, so the cell size enters only there; a wrong '
                     'correction sign, half-step phase, flux factor or overlap formula shows at >= 5e-2). Not yet run '
                     'with these cells, bandwidth and window.'),
        specs=lambda: {'run': waveguide_mode_spec()},
        check=_check_waveguide),
    SelfTestCase(
        name='em3d_far_field_dipole',
        label='EM 3D: dipole radiated power and far-field pattern (flux box, near-to-far transform)',
        description=('The dipole_radiation_3d model (z dipole I l = 1e-12 A m, gaussian 1 um, bandwidth 0.5, at the '
                     'centre of a 1.2 um vacuum cube, 40 nm cells, 10 CPML cells per face; '
                     f'{SELFTEST_DURATION * 1e15:g} fs) with a far-field box on the flux box outline (0.2-1.0 um; '
                     'both read one DFT box). Flux box power_per_cw at 0.9, 1.0, 1.1 um vs the Hertzian-dipole cw '
                     'power eta0 k^2 (I l)^2 / (12 pi), and the far-field power_per_cw in the x-z plane at 36 angles '
                     'vs eta0 k^2 (I l)^2 cos^2(angle) / (32 pi^2), compared where cos^2 >= 0.25; error = the worst '
                     f'relative error of both, tolerance {DIPOLE_TOLERANCE:g} (lattice dispersion at 22-28 cells per '
                     'wavelength and the face quadrature give ~1-2 %; a wrong current density (volume), flux sign / '
                     'face pairing, S(omega) normalisation, CPML reflection, current sign or phase reference shows '
                     f'at >= 10 %). Variant {_HALF_STORAGE_VARIANT}: the same run on a half_storage stepper (fp16 '
                     'fields, float32 arithmetic; field scale from the dipole moment, H stored as eta0 H), tolerance '
                     f'{_HALF_STORAGE_TOLERANCE:g} (plus the fp16 storage noise); a wrong stored-unit scale on the '
                     'dipole, the curl factors or the flux readout is off by orders of magnitude, a broken fp16 '
                     "conversion gives NaN or zero power; raises unless summary 'solvers' records half_storage on "
                     f'the variant node. Not yet run at {SELFTEST_DURATION * 1e15:g} fs, nor the far field on '
                     'half_storage.'),
        specs=lambda: {'run': dipole_far_field_spec()},
        check=_check_far_field,
        variants=(SelfTestVariant(_HALF_STORAGE_VARIANT, tolerance=_HALF_STORAGE_TOLERANCE,
                                  precision='half_storage', roles=('stepper',)),)),
)
