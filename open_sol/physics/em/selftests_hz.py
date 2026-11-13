"""Analytic H_z-polarisation EM self-tests (plane wave, Fresnel, dielectric and Drude Mie cylinders) and the
'gold_nanodisk_hz' template; the plane-wave, Fresnel and Drude cylinder cases are polarisation-generic
(`plane_wave_case`, `fresnel_case`, `mie_drude_cylinder_case`, also registered for E_z by selftests.py). CPU only,
every spec is built fresh on each call."""
import math

import numpy as np

from ...core.constants import C0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import PERIODIC_Y, float_list, glass_half_space, vertical_line, worst_item
from ...model.spec import (DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec,
                           ParameterSpec, PhysicsSpec, RefinementSpec, StudySpec)

_PHYSICS_NAME = 'em'
_PHYSICS_TYPE = 'em_fdtd_hz'
_PROBE_FEATURE = 'probe'
_TRANSMITTED_FLUX = 'transmitted'
_INCIDENT_FLUX = 'incident'
_BOX_SIDES = ('left', 'right', 'bottom', 'top')
_STEADY_PERIODS = 3       # trailing periods searched for the steady peak |probe value|

# plane_wave_case / fresnel_case (E_z and H_z polarisation): wording and probed tangential E per polarisation
_ANALYTIC_CASE_TEXT = {
    'ez': {'name': 'em_', 'label': 'EM ', 'title': '', 'component': 'ez', 'field': 'E_z',
           'launch': ') and launches a plane wave along +x; the peak', 'impedance_note': '', 'mesh_note': '',
           'interface_note': ''},
    'hz': {'name': 'em_hz_', 'label': 'EM H_z ', 'title': 'H_z ', 'component': 'ey', 'field': 'E_y',
           'launch': ', H_z polarisation); its E_y sheet current on the hi face of the line cells launches a plane wave '
                     'along +x, and the peak',
           'impedance_note': ' of the face permittivity', 'mesh_note': ', H_z polarisation',
           'interface_note': 'The E_y face on the interface takes the mean permittivity (1 + 4) / 2, the correct '
                             'average for an E tangential to the interface. '},
}

# em_plane_wave_amplitude / em_hz_plane_wave_amplitude
_PLANE_WAVE_WAVELENGTH = 1e-6     # m, vacuum
_PLANE_WAVE_AMPLITUDE = 1.0       # V/m of tangential E radiated to each side of the line
_PLANE_WAVE_MEDIUM_EPS_R = 4.0    # 'medium' run: eps_r everywhere (PML included), local impedance eta0 / 2
_PLANE_WAVE_RUNS = ('vacuum', 'medium')
_PLANE_WAVE_TOLERANCE = 0.02      # relative, worst run
_PLANE_WAVE_PERIODS = 24          # run length [periods]: window from 21, ramp residual <= exp(-17/3) after the transit
_PLANE_WAVE_STRIP = 0.04e-6       # m, y period (2 cells: the field is y-invariant)

# em_fresnel_transmission / em_hz_fresnel_transmission (+ _graded)
# s: the pulse (ends at 19 fs) passes the flux line by ~40 fs; above the validation unit-slip limit 2 t0 + diagonal
# n_max / C0 = 52.5 fs of the glass run. The 'fresnel_interface' template keeps _FRESNEL_TEMPLATE_DURATION.
_FRESNEL_DURATION = 55e-15
_FRESNEL_TEMPLATE_DURATION = 100e-15
_GLASS_EPS_R = 4.0
_GLASS_INDEX = math.sqrt(_GLASS_EPS_R)
_FRESNEL_TRANSMISSION = 4.0 * _GLASS_INDEX / (1.0 + _GLASS_INDEX) ** 2   # 4 n1 n2 / (n1 + n2)^2 = 8/9 (n1 = 1)
_FRESNEL_TOLERANCE = 0.015        # absolute, worst wavelength

# Graded Fresnel variants (em_fresnel_transmission_graded, em_hz_fresnel_transmission_graded,
# em_fdfd_fresnel_transmission_graded): `graded_interface_mesh` refines x only, so inv_dx != inv_dy around the interface.
GRADED_INTERFACE_CELL = 4e-9      # m, x cell at the interface (FDFD graded Fresnel)
_FDTD_GRADED_INTERFACE_CELL = 5e-9   # m, the FDTD graded Fresnel cases: larger dt, still graded (< 6 nm)
GRADED_INTERFACE_REACH = 0.3e-6   # m, x distance over which the cells blend back to the base cell
_GRADED_DX_MIN_FRACTION = 0.6     # a graded build must reach dx_min <= this x the base cell


def graded_fresnel_text(interface_cell):
    """Description of the graded Fresnel mesh with interface_cell [m] at the interface line."""
    return (f"graded mesh: mode 'nonuniform' (base 10 nm) with a vertical refinement line on the interface x = 2.5 um "
            f'(dx {interface_cell * 1e9:g} nm, margin {GRADED_INTERFACE_REACH * 1e6:g} um), so the x cells blend '
            f'linearly from {interface_cell * 1e9:g} nm at the interface to 10 nm {GRADED_INTERFACE_REACH * 1e6:g} um '
            'away while y keeps 10 nm (unequal inv_dx / inv_dy); the source and the flux line sit on 10 nm cells, so T '
            'pins the graded update and the interface on fine cells (source calibration and flux read-out cancel in '
            f"the ratio); the check raises unless every run's build reached dx_min <= {_GRADED_DX_MIN_FRACTION:g} x 10 "
            'nm (no silent uniform fallback)')


GRADED_FRESNEL_TEXT = graded_fresnel_text(GRADED_INTERFACE_CELL)

# em_hz_mie_cylinder: 10 nm cells, window cell k spans [10 k, 10 (k + 1)] nm. The TFSF box (0.7, 1.7) um covers H_z
# cells [70, 171) (an edge exactly on a cell edge takes one extra cell). The flux lines at 0.505 / 1.895 um (columns /
# rows 50 and 189) measure through the hi faces at 0.51 / 1.90 um and span cells [51, 190) (0.515 .. 1.895 um): a
# closed box in the scattered field. The reference incident line at x = 1.2 um spans cells [80, 160) (0.805 .. 1.595
# um) inside the total field. The vacuum reference needs only the TFSF box and the incident line: it runs on a
# _MIE_REFERENCE_SIZE domain with every shape shifted by -_MIE_REFERENCE_SHIFT (a whole number of cells), so the box-
# relative aux line, dt and the incident flux are those of the full domain.
_MIE_SIZE = 2.4e-6                # m, square all-CPML domain
_MIE_REFERENCE_SIZE = 1.4e-6      # m, the reference run's domain
_MIE_REFERENCE_SHIFT = 0.5e-6     # m, x and y shift of the reference run's shapes (50 cells)
_MIE_CELL = 10e-9                 # m
# s: the pulse ends at 19 fs and crosses the box in < 10 fs; the low-Q (Q < 10) cylinder resonances ring down within
# ~30 fs after it
_MIE_DURATION = 80e-15
_MIE_CENTER = 1.2e-6              # m, cylinder axis at (center, center)
_MIE_RADIUS = 0.2e-6              # m, 20 cells
_MIE_EPS_R = 2.25                 # m = 1.5
_MIE_TFSF_BOX = (0.7e-6, 0.7e-6, 1.7e-6, 1.7e-6)
_MIE_PULSE = {'waveform': 'gaussian', 'wavelength': 1e-6, 'bandwidth': 0.5, 'amplitude': 1.0}
_MIE_SPECTRAL = {'wavelength_start': 0.9e-6, 'wavelength_stop': 1.1e-6, 'n_wavelengths': 3}
_MIE_INCIDENT_LINE = (1.2e-6, 0.805e-6, 1.2e-6, 1.595e-6)
_MIE_REFERENCE_LENGTH = 0.8e-6    # m, the 80 cells of the incident line
_MIE_BOX_PREFIX = 'flux_'
_MIE_BOX_LINES = (0.505e-6, 1.895e-6)    # m, line positions (cell centres)
_MIE_BOX_SPAN = (0.515e-6, 1.895e-6)     # m
_MIE_TOLERANCE = 0.08             # relative, worst wavelength
_MIE_GRADED_CELL = 5e-9           # m, em_hz_mie_cylinder_graded: refinement around the cylinder
_MIE_GRADED_MARGIN = 0.05e-6      # m

# em_hz_mie_drude_cylinder / em_mie_drude_cylinder (mie_drude_cylinder_case, H_z and E_z polarisation): 2.5 nm cells,
# window cell k spans [2.5 k, 2.5 (k + 1)] nm. The TFSF box (0.35, 0.85) um covers the out-of-plane field's cells
# [140, 341) (its x1 / y1 lie exactly on a cell edge, which takes one extra cell). Flux lines sit at
# cell centres and measure through the hi face of their cell. The outer box (scattered field) has faces at 0.30 / 0.90
# um: lines in columns / rows 119 and 359 spanning cells [120, 360), 20 cells outside the TFSF box (19 on the hi
# sides). The inner box (total field) has faces at 0.50 / 0.70 um: lines in columns / rows 199 and 279 spanning cells
# [200, 280), 60 cells inside the TFSF box (61 on the hi sides). The cylinder [0.55, 0.65] um (cells [220, 260)) sits
# 20 cells inside the inner box. The reference incident line sits in column 239 (hi face 0.6 um) and spans cells
# [160, 320) inside the total field. The pulse spectrum exp(-((f - f0) / (0.3 f0))^2) (f0 = C0 / 650 nm) leaves < 1e-6
# of its energy outside the monitored band 0.36-3.6 um (|f - f0| > 2.6 * 0.3 f0), which the Parseval energy check
# needs; the dipole resonance of the cylinder near 222 nm lies far outside the pulse. Run budget: about 510^2 cells
# (device) + 310^2 (reference), ~10k steps and 9 flux monitors x 241 wavelengths over both runs. The vacuum reference
# runs on a _DRUDE_REFERENCE_SIZE domain with every shape shifted by -_DRUDE_REFERENCE_SHIFT (as for _MIE_REFERENCE_*).
_DRUDE_SIZE = 1.2e-6              # m, square all-CPML domain
_DRUDE_REFERENCE_SIZE = 0.7e-6    # m, the reference run's domain
_DRUDE_REFERENCE_SHIFT = 0.25e-6  # m, x and y shift of the reference run's shapes (100 cells of 2.5 nm, 50 of 5 nm)
_DRUDE_CELL = 2.5e-9              # m
# s: the pulse (tau = 2.3 fs, centred at 4.5 tau) ends by ~21 fs and crosses the box in ~4 fs; 40 fs more is ~10
# Drude damping times (100 fs gave energy errors 4e-4 of the 4% tolerance)
_DRUDE_DURATION = 60e-15
_DRUDE_CENTER = 0.6e-6            # m, cylinder axis at (center, center)
_DRUDE_RADIUS = 50e-9             # m, 20 cells
_DRUDE_OMEGA_P = 1.2e16           # rad/s
_DRUDE_GAMMA = 1e14               # rad/s
_DRUDE_EPS_INF = 1.0
_DRUDE_TFSF_BOX = (0.35e-6, 0.35e-6, 0.85e-6, 0.85e-6)
_DRUDE_PULSE = {'waveform': 'gaussian', 'wavelength': 650e-9, 'bandwidth': 0.3, 'amplitude': 1.0}
_DRUDE_SPECTRAL = {'wavelength_start': 0.36e-6, 'wavelength_stop': 3.6e-6, 'n_wavelengths': 241}
_DRUDE_INCIDENT_LINE = (0.59875e-6, 0.40125e-6, 0.59875e-6, 0.79875e-6)
_DRUDE_REFERENCE_LENGTH = 0.4e-6  # m, the 160 cells of the incident line
_DRUDE_OUTER_PREFIX = 'out_'
_DRUDE_OUTER_LINES = (0.29875e-6, 0.89875e-6)   # m, line positions (cell centres)
_DRUDE_OUTER_SPAN = (0.30125e-6, 0.89875e-6)    # m
_DRUDE_INNER_PREFIX = 'in_'
_DRUDE_INNER_LINES = (0.49875e-6, 0.69875e-6)   # m, line positions (cell centres)
_DRUDE_INNER_SPAN = (0.50125e-6, 0.69875e-6)    # m
_ABSORBED_POWER_VALUE = 'absorbed_line_power'   # derived value [W/m], mean over the whole run
_DRUDE_COMPARED_BAND = (0.6e-6, 1.2e-6)         # m, Mie comparison band (off the dipole resonance)
_DRUDE_REPORTED_WAVELENGTH = 0.8e-6             # m, 'measured' / 'expected' are Q_abs here
_DRUDE_SCATTERING_TOLERANCE = 0.10              # relative, worst compared wavelength
_DRUDE_ABSORPTION_TOLERANCE = 0.15              # relative, worst compared wavelength
_DRUDE_ENERGY_TOLERANCE = 0.04                  # relative

GOLD_NANODISK_HZ_LABEL = 'Gold nanodisk (in-plane electric field)'
GOLD_NANODISK_HZ_DESCRIPTION = ('EM (H_z polarisation): TFSF plane wave with in-plane E on a Drude gold disk (uniform '
                                '2 nm mesh, 200 fs); near-field DFT of |E|^2 shows the particle plasmon, plus the '
                                'far-field pattern, 400-900 nm.')


# ---- specs ----
def cw_duration(wavelength, periods):
    """Run length [s] of `periods` periods of the vacuum wavelength."""
    return periods * wavelength / C0


def _plane_wave_spec(physics_type, polarization, run_name):
    """y-periodic strip (2 cells), cw line source spanning the period at x = 1 um, tangential E probe at x = 3.01 um;
    'medium' fills every cell with eps_r = _PLANE_WAVE_MEDIUM_EPS_R."""
    text = _ANALYTIC_CASE_TEXT[polarization]
    materials = ([MaterialSpec('medium', properties={'eps_r': _PLANE_WAVE_MEDIUM_EPS_R}, selection=['all'])]
                 if run_name == 'medium' else [])
    return ModelSpec(
        title=f"Self-test: {text['title']}plane-wave amplitude ({run_name})",
        domain=DomainSpec(size_x=4e-6, size_y=_PLANE_WAVE_STRIP),
        geometry=[vertical_line('source_line', 1e-6, 0.0, _PLANE_WAVE_STRIP),
                  GeometrySpec('probe_point', 'point', {'x': 3.01e-6, 'y': 0.03e-6})],
        materials=materials,
        physics=[PhysicsSpec(_PHYSICS_NAME, physics_type, settings=dict(PERIODIC_Y), features=[
            FeatureSpec('source', 'line_source', ['source_line'],
                        {'waveform': 'cw', 'wavelength': _PLANE_WAVE_WAVELENGTH, 'amplitude': _PLANE_WAVE_AMPLITUDE}),
            FeatureSpec(_PROBE_FEATURE, 'field_probe', ['probe_point'], {'component': text['component']})])],
        mesh=MeshSpec(mode='uniform', dx=20e-9),
        study=StudySpec(type='time_dependent',
                        settings={'duration': cw_duration(_PLANE_WAVE_WAVELENGTH, _PLANE_WAVE_PERIODS)}))


def _fresnel_reference_spec(physics_type, polarization, duration=_FRESNEL_TEMPLATE_DURATION):
    """Vacuum y-periodic strip: gaussian line source at x = 1 um, flux line at x = 3.5 um (0.9-1.1 um), run for
    duration [s]."""
    return ModelSpec(
        title=f"{_ANALYTIC_CASE_TEXT[polarization]['title']}Fresnel transmission: vacuum reference",
        domain=DomainSpec(size_x=5e-6, size_y=0.1e-6),
        geometry=[vertical_line('source_line', 1e-6, 0.0, 0.1e-6),
                  vertical_line('flux_line', 3.5e-6, 0.0, 0.1e-6)],
        physics=[PhysicsSpec(_PHYSICS_NAME, physics_type, settings=dict(PERIODIC_Y), features=[
            FeatureSpec('source', 'line_source', ['source_line'],
                        {'waveform': 'gaussian', 'wavelength': 1e-6, 'bandwidth': 0.5, 'amplitude': 1.0}),
            FeatureSpec(_TRANSMITTED_FLUX, 'flux_monitor', ['flux_line'],
                        {'wavelength_start': 0.9e-6, 'wavelength_stop': 1.1e-6, 'n_wavelengths': 5})])],
        mesh=MeshSpec(mode='uniform', dx=10e-9),
        study=StudySpec(type='time_dependent', settings={'duration': duration}))


def fresnel_device_spec(physics_type, polarization, duration=_FRESNEL_TEMPLATE_DURATION):
    """The vacuum reference plus a glass (eps_r = 4) half-space from x = 2.5 um, which continues into the x_hi PML
    through the clamped cell centres (E_z, default duration: the 'fresnel_interface' template)."""
    spec = _fresnel_reference_spec(physics_type, polarization, duration)
    spec.title = f"{_ANALYTIC_CASE_TEXT[polarization]['title']}Fresnel transmission at a dielectric interface"
    half_space, glass = glass_half_space(2.5e-6, 5e-6, 0.1e-6, _GLASS_EPS_R)
    spec.geometry.append(half_space)
    spec.materials.append(glass)
    return spec


def _fresnel_specs(physics_type, polarization):
    """'reference' (vacuum) then 'device' (fresnel_device_spec): same source and flux line, _FRESNEL_DURATION."""
    return {'reference': _fresnel_reference_spec(physics_type, polarization, _FRESNEL_DURATION),
            'device': fresnel_device_spec(physics_type, polarization, _FRESNEL_DURATION)}


def graded_interface_mesh(spec, interface_x, interface_cell=GRADED_INTERFACE_CELL):
    """Switch spec in place to mode 'nonuniform' graded in x toward the vertical refinement line 'refine_line' at
    interface_x (interface_cell there, the old dx as base); returns spec."""
    spec.title = f'{spec.title}, graded mesh'
    spec.geometry.append(vertical_line('refine_line', interface_x, 0.0, spec.domain.size_y))
    spec.mesh = MeshSpec(mode='nonuniform', dx=spec.mesh.dx, refinements=[
        RefinementSpec('refine_line', interface_cell, GRADED_INTERFACE_REACH)])
    return spec


def require_graded(results_by_name, base_cell):
    """results_by_name; ValueError unless every run's build reached dx_min <= _GRADED_DX_MIN_FRACTION x base_cell."""
    for run_name, results in results_by_name.items():
        dx_min = float((results.summary().get('build') or {}).get('dx_min', math.nan))
        if not dx_min <= _GRADED_DX_MIN_FRACTION * base_cell:
            raise ValueError(f"run '{run_name}': expected a graded mesh with dx_min <= "
                             f'{_GRADED_DX_MIN_FRACTION * base_cell:.3g} m, got dx_min {dx_min:.4g} m')
    return results_by_name


def _cylinder(center, radius):
    """Circle 'cylinder' with its axis at (center, center)."""
    return GeometrySpec('cylinder', 'circle', {'cx': center, 'cy': center, 'r': radius})


def _incident_flux(line, spectral):
    """(geometry, features) of the reference run's vertical flux line (x0, y0, x1, y1) named 'incident'."""
    x0, y0, x1, y1 = line
    return ([GeometrySpec('incident_line', 'line', {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})],
            [FeatureSpec(_INCIDENT_FLUX, 'flux_monitor', ['incident_line'], dict(spectral))])


def _flux_box(prefix, line_positions, span, spectral):
    """(geometry, features) of the closed flux box '<prefix>left/right/bottom/top': vertical lines at x =
    line_positions and horizontal ones at y = line_positions, each spanning span = (start, end)."""
    lo, hi = line_positions
    start, end = span
    endpoints = {'left': (lo, start, lo, end), 'right': (hi, start, hi, end),
                 'bottom': (start, lo, end, lo), 'top': (start, hi, end, hi)}
    geometry, features = [], []
    for side in _BOX_SIDES:
        x0, y0, x1, y1 = endpoints[side]
        feature_name = prefix + side
        line_name = f'{feature_name}_line'
        geometry.append(GeometrySpec(line_name, 'line', {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1}))
        features.append(FeatureSpec(feature_name, 'flux_monitor', [line_name], dict(spectral)))
    return geometry, features


def _tfsf_spec(physics_type, title, size, cell_size, duration, tfsf_box, pulse, geometry, features, materials=(),
               settings=None, derived_values=()):
    """All-CPML size x size domain on a uniform mesh, lit by a +x TFSF plane wave on the rectangle tfsf_box, with one
    physics 'em' of physics_type."""
    x0, y0, x1, y1 = tfsf_box
    return ModelSpec(
        title=title,
        domain=DomainSpec(size_x=size, size_y=size),
        geometry=[GeometrySpec('tfsf_box', 'rectangle', {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1}), *geometry],
        materials=list(materials),
        physics=[PhysicsSpec(_PHYSICS_NAME, physics_type, settings={} if settings is None else dict(settings),
                             features=[FeatureSpec('plane_wave', 'plane_wave_tfsf', ['tfsf_box'], dict(pulse)),
                                       *features])],
        mesh=MeshSpec(mode='uniform', dx=cell_size),
        study=StudySpec(type='time_dependent', settings={'duration': duration}),
        derived_values=list(derived_values))


def _shifted(coordinates, shift):
    """coordinates (x0, y0, x1, y1) [m] each moved by -shift."""
    return tuple(value - shift for value in coordinates)


def _mie_specs():
    """'reference' (vacuum, incident flux line; small shifted domain) then 'device' (glass cylinder inside a closed
    scattered-field box)."""
    incident_geometry, incident_features = _incident_flux(_shifted(_MIE_INCIDENT_LINE, _MIE_REFERENCE_SHIFT),
                                                          _MIE_SPECTRAL)
    reference = _tfsf_spec(_PHYSICS_TYPE, 'H_z Mie cylinder: vacuum reference', _MIE_REFERENCE_SIZE, _MIE_CELL,
                           _MIE_DURATION, _shifted(_MIE_TFSF_BOX, _MIE_REFERENCE_SHIFT), _MIE_PULSE, incident_geometry,
                           incident_features)
    box_geometry, box_features = _flux_box(_MIE_BOX_PREFIX, _MIE_BOX_LINES, _MIE_BOX_SPAN, _MIE_SPECTRAL)
    device = _tfsf_spec(_PHYSICS_TYPE, 'H_z Mie cylinder: dielectric', _MIE_SIZE, _MIE_CELL, _MIE_DURATION,
                        _MIE_TFSF_BOX, _MIE_PULSE, [_cylinder(_MIE_CENTER, _MIE_RADIUS), *box_geometry], box_features,
                        materials=[MaterialSpec('glass', properties={'eps_r': _MIE_EPS_R}, selection=['cylinder'])])
    return {'reference': reference, 'device': device}


def _graded_mie_specs():
    """_mie_specs on a graded mesh refined to _MIE_GRADED_CELL around the cylinder (drawn in the reference run too, at
    its shifted position)."""
    specs = _mie_specs()
    if not any(shape.name == 'cylinder' for shape in specs['reference'].geometry):
        specs['reference'].geometry.append(_cylinder(_MIE_CENTER - _MIE_REFERENCE_SHIFT, _MIE_RADIUS))
    for spec in specs.values():
        spec.title = f'{spec.title}, graded mesh'
        spec.mesh = MeshSpec(mode='nonuniform', dx=_MIE_CELL, refinements=[
            RefinementSpec('cylinder', _MIE_GRADED_CELL, _MIE_GRADED_MARGIN)])
    return specs


def _drude_specs(physics_type, cell):
    """'reference' (vacuum, incident flux line; small shifted domain) then 'device' (Drude cylinder inside an inner
    total-field and an outer scattered-field flux box, absorption tracked, derived absorbed_line_power over the
    cylinder) of physics_type on uniform cells of size cell [m]."""
    incident_geometry, incident_features = _incident_flux(_shifted(_DRUDE_INCIDENT_LINE, _DRUDE_REFERENCE_SHIFT),
                                                          _DRUDE_SPECTRAL)
    reference = _tfsf_spec(physics_type, f'{physics_type} Drude cylinder: vacuum reference', _DRUDE_REFERENCE_SIZE,
                           cell, _DRUDE_DURATION, _shifted(_DRUDE_TFSF_BOX, _DRUDE_REFERENCE_SHIFT), _DRUDE_PULSE,
                           incident_geometry, incident_features)
    outer_geometry, outer_features = _flux_box(_DRUDE_OUTER_PREFIX, _DRUDE_OUTER_LINES, _DRUDE_OUTER_SPAN,
                                               _DRUDE_SPECTRAL)
    inner_geometry, inner_features = _flux_box(_DRUDE_INNER_PREFIX, _DRUDE_INNER_LINES, _DRUDE_INNER_SPAN,
                                               _DRUDE_SPECTRAL)
    metal = MaterialSpec('drude_metal', properties={}, selection=['cylinder'], dispersion=[
        {'model': 'drude', 'omega_p': _DRUDE_OMEGA_P, 'gamma': _DRUDE_GAMMA, 'eps_inf': _DRUDE_EPS_INF}])
    absorbed_power = DerivedValueSpec(_ABSORBED_POWER_VALUE, f'{_PHYSICS_NAME}_absorbed_power', ['cylinder'],
                                      'integral')
    device = _tfsf_spec(physics_type, f'{physics_type} Drude cylinder: Mie and energy balance', _DRUDE_SIZE,
                        cell, _DRUDE_DURATION, _DRUDE_TFSF_BOX, _DRUDE_PULSE,
                        [_cylinder(_DRUDE_CENTER, _DRUDE_RADIUS), *outer_geometry, *inner_geometry],
                        outer_features + inner_features, materials=[metal], settings={'track_absorption': True},
                        derived_values=[absorbed_power])
    return {'reference': reference, 'device': device}


def gold_nanodisk_hz_spec():
    """Template 'gold_nanodisk_hz': the built-in gold_nanodisk model in H_z polarisation, run for 200 fs so the Drude
    ring-down exp(-gamma t / 2) is down to ~4e-5."""
    spectral = {'wavelength_start': 400e-9, 'wavelength_stop': 900e-9, 'n_wavelengths': 11}
    return ModelSpec(
        title='Gold nanodisk scattering (H_z polarisation)',
        parameters=[ParameterSpec('r_disk', '50*nm', 'disk radius')],
        domain=DomainSpec(size_x=1e-6, size_y=1e-6),
        geometry=[GeometrySpec('disk', 'circle', {'cx': 0.5e-6, 'cy': 0.5e-6, 'r': 'r_disk'}),
                  GeometrySpec('tfsf_box', 'rectangle', {'x0': 0.35e-6, 'y0': 0.35e-6, 'x1': 0.65e-6, 'y1': 0.65e-6}),
                  GeometrySpec('near_field_box', 'rectangle',
                               {'x0': 0.4e-6, 'y0': 0.4e-6, 'x1': 0.6e-6, 'y1': 0.6e-6}),
                  GeometrySpec('far_field_box', 'rectangle',
                               {'x0': 0.25e-6, 'y0': 0.25e-6, 'x1': 0.75e-6, 'y1': 0.75e-6})],
        materials=[MaterialSpec('gold', library='gold', selection=['disk'])],
        physics=[PhysicsSpec(_PHYSICS_NAME, _PHYSICS_TYPE, features=[
            FeatureSpec('plane_wave', 'plane_wave_tfsf', ['tfsf_box'],
                        {'waveform': 'gaussian', 'wavelength': 550e-9, 'bandwidth': 0.8, 'amplitude': 1.0}),
            FeatureSpec('near_field', 'dft_monitor', ['near_field_box'], dict(spectral)),
            FeatureSpec('far_field', 'far_field', ['far_field_box'], dict(spectral, n_angles=360))])],
        mesh=MeshSpec(mode='uniform', dx=2e-9),
        study=StudySpec(type='time_dependent', settings={'duration': 200e-15}))


# ---- Bessel functions and the cylinder Mie series (CPU) ----
def _bessel_tables(max_order, inner_argument, outer_argument):
    """(J_n(inner), J_n(outer), Y_n(outer) for n = 0..max_order) from scipy.special; inner may be complex, outer is
    real and > 0."""
    import scipy.special
    orders = np.arange(max_order + 1)
    return (scipy.special.jv(orders, inner_argument), scipy.special.jv(orders, outer_argument),
            scipy.special.yv(orders, outer_argument))


def _bessel_derivatives(values, argument):
    """Z_n'(z) = Z_(n-1)(z) - (n / z) Z_n(z) for n = 0..N from Z_0..Z_N (N >= 1), with Z_(-1) = -Z_1."""
    previous = np.concatenate(([-values[1]], values[:-1]))
    return previous - np.arange(values.size) / argument * values


def _mie_efficiencies(size_parameter, relative_index, e_along_axis):
    """(Q_sca, Q_ext) of an infinite cylinder at normal incidence, Bohren & Huffman 8.4: case I (b_n, E
    along the axis) when e_along_axis else case II (a_n, H along the axis); exp(-i omega t) convention (lossy m has
    Im m > 0), orders n = 0..ceil(x + 4 x^(1/3) + 2)."""
    x = float(size_parameter)
    m = complex(relative_index)
    max_order = int(math.ceil(x + 4.0 * x ** (1.0 / 3.0) + 2.0))
    j_inner, j_outer, y_outer = _bessel_tables(max_order, m * x, x)
    dj_inner = _bessel_derivatives(j_inner, m * x)
    dj_outer = _bessel_derivatives(j_outer, x)
    hankel_outer = j_outer + 1j * y_outer
    dhankel_outer = dj_outer + 1j * _bessel_derivatives(y_outer, x)
    if e_along_axis:
        coefficients = ((j_inner * dj_outer - m * dj_inner * j_outer)
                        / (j_inner * dhankel_outer - m * dj_inner * hankel_outer))
    else:
        coefficients = ((m * j_inner * dj_outer - j_outer * dj_inner)
                        / (m * j_inner * dhankel_outer - hankel_outer * dj_inner))
    multiplicity = np.where(np.arange(max_order + 1) == 0, 1.0, 2.0)   # orders n and -n contribute alike
    q_sca = 2.0 / x * float(np.sum(multiplicity * np.abs(coefficients) ** 2))
    q_ext = 2.0 / x * float(np.sum(multiplicity * coefficients).real)
    return q_sca, q_ext


def _cylinder_efficiencies(wavelengths, radius, relative_index, e_along_axis):
    """(Q_sca, Q_abs) float64 arrays at each vacuum wavelength [m]; relative_index(wavelength) gives m in the
    exp(-i omega t) convention, e_along_axis picks the Mie case (see _mie_efficiencies)."""
    q_sca = np.empty(wavelengths.size)
    q_abs = np.empty(wavelengths.size)
    for index, wavelength in enumerate(wavelengths):
        scattering, extinction = _mie_efficiencies(2.0 * math.pi * radius / wavelength, relative_index(wavelength),
                                                   e_along_axis)
        q_sca[index] = scattering
        q_abs[index] = extinction - scattering
    return q_sca, q_abs


def _glass_relative_index(wavelength):
    """Index m = sqrt(eps_r) of the dielectric Mie cylinder (non-dispersive)."""
    return math.sqrt(_MIE_EPS_R)


def _drude_relative_index(wavelength):
    """m = sqrt(eps_inf - omega_p^2 / (omega^2 + i gamma omega)) (principal root, Im m >= 0): the Bohren & Huffman
    exp(-i omega t) form, the complex conjugate of the solver's exp(+j omega t) Drude permittivity."""
    omega = 2.0 * math.pi * C0 / wavelength
    permittivity = _DRUDE_EPS_INF - _DRUDE_OMEGA_P ** 2 / (omega ** 2 + 1j * _DRUDE_GAMMA * omega)
    return complex(np.sqrt(permittivity))


# ---- checks ----
def steady_peak(results, wavelength, physics_name=_PHYSICS_NAME):
    """(peak |value| of physics_name's 'probe' feature over the last _STEADY_PERIODS periods before its last sample,
    window start [s], samples in the window)."""
    probe = results.result(physics_name, _PROBE_FEATURE)
    times = np.asarray(probe['t'], dtype=np.float64)
    values = np.asarray(probe['values'], dtype=np.float64)
    if times.size == 0:
        raise ValueError('the field probe recorded no samples')
    window_start = times[-1] - _STEADY_PERIODS * wavelength / C0
    in_window = times >= window_start
    return float(np.max(np.abs(values[in_window]))), float(window_start), int(np.count_nonzero(in_window))


def _check_plane_wave(results_by_name):
    """Steady peak |E| at the probe vs the line-source amplitude per run; measured = the worst run (relative
    error, NaN counts as the worst)."""
    peaks, errors, details = {}, {}, {'unit': 'V/m'}
    for run_name in _PLANE_WAVE_RUNS:
        peak, window_start, window_samples = steady_peak(results_by_name[run_name], _PLANE_WAVE_WAVELENGTH)
        peaks[run_name] = peak
        errors[run_name] = abs(peak - _PLANE_WAVE_AMPLITUDE) / _PLANE_WAVE_AMPLITUDE
        details.update({f'{run_name}_peak': peak, f'{run_name}_window_start': window_start,
                        f'{run_name}_window_samples': window_samples})
    worst, _ = worst_item(errors)
    details['worst_run'] = worst
    return {'measured': peaks[worst], 'expected': _PLANE_WAVE_AMPLITUDE, 'error': errors[worst],
            'tolerance': _PLANE_WAVE_TOLERANCE, 'details': details}


def _check_fresnel(results_by_name):
    """T = device flux / reference flux per wavelength vs 4 n1 n2 / (n1 + n2)^2; measured = the worst wavelength."""
    reference = results_by_name['reference'].result(_PHYSICS_NAME, _TRANSMITTED_FLUX)
    device = results_by_name['device'].result(_PHYSICS_NAME, _TRANSMITTED_FLUX)
    reference_power = np.asarray(reference['power'], dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        transmission = np.asarray(device['power'], dtype=np.float64) / reference_power
    deviation = np.abs(transmission - _FRESNEL_TRANSMISSION)
    worst = int(np.argmax(deviation))   # a NaN counts as the largest deviation
    return {'measured': float(transmission[worst]), 'expected': _FRESNEL_TRANSMISSION,
            'error': float(deviation[worst]), 'tolerance': _FRESNEL_TOLERANCE,
            'details': {'wavelengths': float_list(reference['wavelength']),
                        'transmission': float_list(transmission),
                        'reference_power': float_list(reference_power)}}


def _incident_intensity(results, reference_length):
    """(vacuum wavelengths [m], incident spectral power per unit length I = P_inc / reference_length) of a reference
    run, host float64."""
    incident = results.result(_PHYSICS_NAME, _INCIDENT_FLUX)
    return (np.asarray(incident['wavelength'], dtype=np.float64),
            np.asarray(incident['power'], dtype=np.float64) / reference_length)


def _net_outflow(results, prefix):
    """Outward spectral flux (right - left + top - bottom) of the closed flux box '<prefix>left/right/bottom/top'."""
    power = {side: np.asarray(results.result(_PHYSICS_NAME, prefix + side)['power'], dtype=np.float64)
             for side in _BOX_SIDES}
    return power['right'] - power['left'] + power['top'] - power['bottom']


def _check_mie_cylinder(results_by_name):
    """Q_sca = outward scattered flux / (I 2a) per wavelength vs the case II series; measured = the worst wavelength."""
    wavelengths, intensity = _incident_intensity(results_by_name['reference'], _MIE_REFERENCE_LENGTH)
    q_sca, _ = _cylinder_efficiencies(wavelengths, _MIE_RADIUS, _glass_relative_index, False)
    with np.errstate(divide='ignore', invalid='ignore'):
        q_sca_measured = (_net_outflow(results_by_name['device'], _MIE_BOX_PREFIX)
                          / (intensity * 2.0 * _MIE_RADIUS))
        deviation = np.abs(q_sca_measured / q_sca - 1.0)
    worst = int(np.argmax(deviation))   # a NaN counts as the largest deviation
    return {'measured': float(q_sca_measured[worst]), 'expected': float(q_sca[worst]),
            'error': float(deviation[worst]), 'tolerance': _MIE_TOLERANCE,
            'details': {'wavelengths': float_list(wavelengths), 'Q_sca_meas': float_list(q_sca_measured),
                        'Q_sca': float_list(q_sca)}}


def _check_drude_cylinder(results_by_name, e_along_axis):
    """Q_sca / Q_abs of the outer / inner flux boxes vs the Mie series (case I when e_along_axis, else case II) over
    _DRUDE_COMPARED_BAND, and the Parseval energy of the inner box vs the J.E loss integrated over the cylinder; error
    in tolerance units."""
    device = results_by_name['device']
    wavelengths, intensity = _incident_intensity(results_by_name['reference'], _DRUDE_REFERENCE_LENGTH)
    scattered_power = _net_outflow(device, _DRUDE_OUTER_PREFIX)
    absorbed_power = -_net_outflow(device, _DRUDE_INNER_PREFIX)
    compared = (wavelengths >= _DRUDE_COMPARED_BAND[0]) & (wavelengths <= _DRUDE_COMPARED_BAND[1])
    compared_wavelengths = wavelengths[compared]
    q_sca, q_abs = _cylinder_efficiencies(compared_wavelengths, _DRUDE_RADIUS, _drude_relative_index, e_along_axis)
    with np.errstate(divide='ignore', invalid='ignore'):
        geometric_power = intensity[compared] * 2.0 * _DRUDE_RADIUS
        q_sca_measured = scattered_power[compared] / geometric_power
        q_abs_measured = absorbed_power[compared] / geometric_power
        err_sca = float(np.max(np.abs(q_sca_measured / q_sca - 1.0)))   # NaN propagates
        err_abs = float(np.max(np.abs(q_abs_measured / q_abs - 1.0)))
    # Parseval: the DFT spectra carry exp(-i omega t) dt, so the energy per unit length is (1/pi) int_0^inf P d omega
    angular_frequency = 2.0 * math.pi * C0 / wavelengths
    order = np.argsort(angular_frequency)
    omega_sorted, absorbed_sorted = angular_frequency[order], absorbed_power[order]
    energy_flux = float(0.5 * np.sum((absorbed_sorted[1:] + absorbed_sorted[:-1]) * np.diff(omega_sorted))) / math.pi
    summary = device.summary()
    absorbed_line_power = float(summary['derived'][_ABSORBED_POWER_VALUE])   # float() also reads 'nan'
    loss_window = float(summary['t_end'])   # the accumulation window is the whole run
    energy_loss = absorbed_line_power * loss_window
    with np.errstate(divide='ignore', invalid='ignore'):
        err_energy = float(abs(np.float64(energy_loss) / np.float64(energy_flux) - 1.0))
    error = float(np.max([err_sca / _DRUDE_SCATTERING_TOLERANCE, err_abs / _DRUDE_ABSORPTION_TOLERANCE,
                          err_energy / _DRUDE_ENERGY_TOLERANCE]))   # NaN propagates
    reported = int(np.argmin(np.abs(compared_wavelengths - _DRUDE_REPORTED_WAVELENGTH)))
    return {'measured': float(q_abs_measured[reported]), 'expected': float(q_abs[reported]),
            'error': error, 'tolerance': 1.0,
            'details': {'wavelengths': float_list(compared_wavelengths),
                        'reported_wavelength': float(compared_wavelengths[reported]),
                        'Q_sca_meas': float_list(q_sca_measured), 'Q_sca': float_list(q_sca),
                        'Q_abs_meas': float_list(q_abs_measured), 'Q_abs': float_list(q_abs),
                        'E_flux': energy_flux, 'E_q': energy_loss, 'energy_unit': 'J/m',
                        'absorbed_line_power': absorbed_line_power, 'loss_window': loss_window,
                        'err_sca': err_sca, 'err_abs': err_abs, 'err_energy': err_energy}}


_MIE_SERIES_TEXT = ("Bohren & Huffman case II (H parallel to the axis) series Q_sca = (2/x)(|a_0|^2 + 2 sum |a_n|^2), "
                    "x = 2 pi a / lambda, a_n = [m J_n(mx) J_n'(x) - J_n(x) J_n'(mx)] / [m J_n(mx) H_n'(x) - H_n(x) "
                    "J_n'(mx)], H_n = J_n + i Y_n, n = 0..ceil(x + 4 x^(1/3) + 2)")
_MIE_SERIES_TEXT_E_ALONG_AXIS = ("Bohren & Huffman case I (E parallel to the axis) series Q_sca = (2/x)(|b_0|^2 + 2 "
                                 "sum |b_n|^2), x = 2 pi a / lambda, b_n = [J_n(mx) J_n'(x) - m J_n'(mx) J_n(x)] / "
                                 "[J_n(mx) H_n'(x) - m J_n'(mx) H_n(x)], H_n = J_n + i Y_n, n = 0..ceil(x + 4 x^(1/3) "
                                 "+ 2)")
_BESSEL_TEXT = 'Bessel functions come from scipy.special.'
# polarisation -> the wording of its Drude cylinder case (mie_drude_cylinder_case)
_DRUDE_CASE_TEXT = {
    'ez': {'name': 'em_mie_drude_cylinder', 'label': 'EM Drude cylinder: Mie and energy balance',
           'incident': 'E along the axis', 'axis_field': 'E',
           'series': f'{_MIE_SERIES_TEXT_E_ALONG_AXIS} and Q_ext = (2/x) Re(b_0 + 2 sum b_n)',
           'band_note': 'no surface-plasmon resonance exists with E along the axis',
           'budget': 'Budget: Q_sca 10% / Q_abs 15% for the metal surface staircased at 20 cells per radius (E_z is '
                     'tangential to it, so no corner enhancement, but the ~25 nm skin depth at 0.8 um spans only ~10 '
                     'cells), the O((omega dt)^2) < 0.1% error of the leap-frog Drude update and CPML < 0.5%; energy '
                     '4% for the discrete Poynting / J.E mismatch O((omega dt)^2) and the Parseval quadrature over '
                     '241 wavelengths.'},
    'hz': {'name': 'em_hz_mie_drude_cylinder', 'label': 'EM H_z Drude cylinder: Mie and energy balance',
           'incident': 'E in plane', 'axis_field': 'H',
           'series': f'{_MIE_SERIES_TEXT} and Q_ext = (2/x) Re(a_0 + 2 sum a_n)',
           'band_note': 'off the dipole resonance near 222 nm',
           'budget': 'Budget: Q_sca 10% / Q_abs 15% for the metal surface staircased at 20 cells per radius in H_z '
                     'polarisation (corner field enhancement raises the skin-layer loss), arithmetic face averaging '
                     'of the normal E, the O((omega dt)^2) < 0.1% error of the leap-frog Drude update and CPML < '
                     '0.5%; energy 4% for the discrete Poynting / J.E mismatch O((omega dt)^2) and the Parseval '
                     'quadrature over 241 wavelengths.'},
}


def mie_drude_cylinder_case(physics_type, polarization, cell=_DRUDE_CELL):
    """SelfTestCase of the Drude cylinder (Mie Q_sca / Q_abs over 0.6-1.2 um and the J.E energy balance) run with
    physics_type in polarisation 'ez' (E along the axis, Mie case I) or 'hz' (H along the axis, case II); a cell other
    than _DRUDE_CELL names the case '<name>_<cell>nm' (its budget is the 2.5 nm one: for backend parity only)."""
    if polarization not in _DRUDE_CASE_TEXT:
        raise ValueError(f"polarization must be one of {sorted(_DRUDE_CASE_TEXT)}, got {polarization!r}")
    text = _DRUDE_CASE_TEXT[polarization]
    e_along_axis = polarization == 'ez'
    coarse = cell != _DRUDE_CELL
    return SelfTestCase(
        name=f"{text['name']}_{cell * 1e9:g}nm" if coarse else text['name'],
        label=f"{text['label']} ({cell * 1e9:g} nm)" if coarse else text['label'],
        description=f"A +x gaussian TFSF plane wave (650 nm, bandwidth 0.3, {text['incident']}) hits a Drude cylinder "
                    f"(r = 50 nm, omega_p 1.2e16 rad/s, gamma 1e14 rad/s, eps_inf 1) with {text['axis_field']} along "
                    f'its axis in a 1.2 x 1.2 um all-CPML box ({cell * 1e9:g} nm mesh, {_DRUDE_DURATION * 1e15:g} fs, '
                    'track_absorption). Flux lines at '
                    '241 wavelengths (0.36-3.6 um) form an outer closed box in the scattered field and an inner one '
                    'in the total field: Q_sca = outward flux of the outer box / (I 2a) and Q_abs = inward flux of '
                    'the inner box / (I 2a), with I = incident flux / 0.4 um of a vacuum reference run (a 0.7 x 0.7 '
                    'um domain holding only the shifted TFSF box and incident line: the box-relative incident wave '
                    'is the same). At the '
                    f"wavelengths in 0.6-1.2 um ({text['band_note']}) they must match the {text['series']}, Q_abs = "
                    'Q_ext - Q_sca, with m = sqrt(eps_inf - omega_p^2 / (omega^2 + i gamma omega)): Bohren & '
                    "Huffman's exp(-i omega t) convention, the complex conjugate of the solver's exp(+j omega t) "
                    'permittivity; the efficiencies do not depend on the convention once m is taken in theirs. '
                    'Energy balance: the absorbed energy per unit length from the inner box by Parseval, (1/pi) '
                    'times the integral of P_abs over omega (the pulse carries < 1e-6 of its energy outside the '
                    'band), must equal the derived value absorbed_line_power (J.E loss integrated over the cylinder '
                    'cells only, so a loss leak into vacuum neighbours shows) times the run time t_end (the '
                    'accumulation window). error = max(err_sca / 0.10, err_abs / 0.15, err_energy / 0.04) with the '
                    'worst relative errors, tolerance 1; measured / expected = Q_abs at the wavelength closest to '
                    f"0.8 um. {text['budget']} {_BESSEL_TEXT}"
                    + (' The budget assumes 2.5 nm cells.' if coarse else ' Small reference domain not yet confirmed '
                       'by a run.'),
        specs=lambda: _drude_specs(physics_type, cell),
        check=lambda results_by_name: _check_drude_cylinder(results_by_name, e_along_axis))


def plane_wave_case(physics_type, polarization):
    """SelfTestCase of the cw line-source plane-wave amplitude (vacuum and eps_r = 4 runs) in polarisation ez / hz."""
    text = _ANALYTIC_CASE_TEXT[polarization]
    field = text['field']
    return SelfTestCase(
        name=f"{text['name']}plane_wave_amplitude",
        label=f"{text['label']}plane-wave amplitude",
        description='A cw line source (vacuum wavelength 1 um, amplitude 1 V/m) spans the period of a y-periodic '
                    f"strip (2 cells of 20 nm, CPML along x{text['launch']} |{field}| over the last 3 of "
                    f'{_PLANE_WAVE_PERIODS} periods at a '
                    f'probe 2 um downstream must equal the amplitude, |{field}| = 1 V/m. Two runs: vacuum, and eps_r '
                    '= 4 in every cell (PML included), where the calibration must use the local impedance eta0 / 2'
                    f"{text['impedance_note']} (eta0 would give 2 V/m); error = the larger relative error. Tolerance "
                    '0.02: ramp residual at the window start after the 2 / 4 period transit exp(-19/3) / exp(-17/3) '
                    '~ 2e-3 / 3.5e-3, the source calibration includes the discrete launch '
                    'factor cos(k dx/2) (0.2% / 0.8% at 50 / 25 cells per wavelength), CPML reflection < 0.5% (its '
                    'vacuum-optimum grading is ~2x too strong at n = 2), peak sampling at ~70 steps per period '
                    '~0.1%. 24 periods / 2-cell strip not yet confirmed by a run.',
        specs=lambda: {run_name: _plane_wave_spec(physics_type, polarization, run_name)
                       for run_name in _PLANE_WAVE_RUNS},
        check=_check_plane_wave)


def fresnel_case(physics_type, polarization, graded, heavy=False):
    """SelfTestCase of the vacuum / glass Fresnel transmission in polarisation 'ez' / 'hz', on the uniform 10 nm mesh
    or (graded) on graded_interface_mesh; heavy = SelfTestCase.heavy."""
    text = _ANALYTIC_CASE_TEXT[polarization]
    name = f"{text['name']}fresnel_transmission"
    label = f"{text['label']}Fresnel transmission"
    if graded:
        return SelfTestCase(
            name=f'{name}_graded',
            label=f'{label} on a graded mesh',
            description=f'{name} (both runs) on a {graded_fresnel_text(_FDTD_GRADED_INTERFACE_CELL)}. T must still '
                        'equal 8/9; tolerance 0.015 as there (the finer interface cells only reduce the discrete '
                        'error). 5 nm interface cell and 55 fs not yet confirmed by a run.',
            specs=lambda: {run_name: graded_interface_mesh(spec, 2.5e-6, _FDTD_GRADED_INTERFACE_CELL)
                           for run_name, spec in _fresnel_specs(physics_type, polarization).items()},
            check=lambda results_by_name: _check_fresnel(require_graded(results_by_name, 10e-9)),
            heavy=heavy)
    return SelfTestCase(
        name=name,
        label=label,
        description='A gaussian pulse (1 um, bandwidth 0.5) from a line source spanning a y-periodic strip (10 nm '
                    f"mesh{text['mesh_note']}) crosses a vacuum / glass interface (eps_r = 4, n = 2) at normal "
                    'incidence. T = flux 1 um behind the interface divided by the flux of a vacuum reference run with '
                    'the same source, at 5 wavelengths from 0.9 to 1.1 um, must equal 4 n1 n2 / (n1 + n2)^2 = 8/9; '
                    f"error = max |T - 8/9|. {text['interface_note']}Tolerance 0.015: discrete dispersion at >= 45 "
                    'cells per wavelength in the glass, the half-cell E / H offset of the discrete flux (~0.2%) and '
                    f'CPML reflection in the dielectric. Run length {_FRESNEL_DURATION * 1e15:g} fs (the pulse has '
                    'passed the flux line by ~40 fs), not yet confirmed by a run.',
        specs=lambda: _fresnel_specs(physics_type, polarization),
        check=_check_fresnel,
        heavy=heavy)


# Not registered on its own: graded by selftests_backend em_fdtd_cpu_gpu_media (its own check on both backends).
HZ_MIE_CYLINDER_CASE = SelfTestCase(
    name='em_hz_mie_cylinder',
    label='EM H_z dielectric cylinder: Mie scattering',
    description='A +x gaussian TFSF plane wave (1 um, bandwidth 0.5, E in plane) hits a dielectric cylinder (r = '
                '0.2 um, eps_r = 2.25, m = 1.5) with H along its axis in a 2.4 x 2.4 um all-CPML box (10 nm mesh, '
                f'{_MIE_DURATION * 1e15:g} fs: the low-Q resonances ring down within ~30 fs after the pulse). '
                'Q_sca = outward flux of a closed box of four flux lines in the scattered field / (I 2a), '
                'with I = incident flux / 0.8 um of a vacuum reference run (a 1.4 x 1.4 um domain holding only '
                'the shifted TFSF box and incident line), at 3 wavelengths from 0.9 to 1.1 um '
                f'must match the {_MIE_SERIES_TEXT}; error = max relative deviation. Tolerance 0.08: the '
                'circle staircased at 20 cells per radius, arithmetic face averaging of eps where the E normal to '
                'the surface needs the harmonic mean, discrete dispersion at >= 60 cells per wavelength in the '
                f'cylinder, TFSF leakage and CPML reflection. {_BESSEL_TEXT} {_MIE_DURATION * 1e15:g} fs and the '
                'small reference not yet confirmed by a run.',
    specs=_mie_specs,
    check=_check_mie_cylinder,
    heavy=True)


# The uniform-mesh H_z Fresnel case is the normal-incidence point (theta 0) of the example
# em/analytic_bloch_fresnel_brewster (T_p, R_p_excess) and is graded by selftests_backend em_fdtd_cpu_gpu.
HZ_SELFTEST_CASES = (
    plane_wave_case(_PHYSICS_TYPE, 'hz'),
    fresnel_case(_PHYSICS_TYPE, 'hz', graded=True),
    SelfTestCase(
        name='em_hz_mie_cylinder_graded',
        label='EM H_z dielectric cylinder: Mie scattering on a graded mesh',
        description=f"em_hz_mie_cylinder (both runs) on a graded mesh: mode 'nonuniform', base {_MIE_CELL * 1e9:g} nm, "
                    f'the cylinder (drawn in the shifted vacuum reference too, without material, so both runs share '
                    'dt) refined to '
                    f'{_MIE_GRADED_CELL * 1e9:g} nm with a {_MIE_GRADED_MARGIN * 1e6:g} um margin, so x and y grade '
                    'inside the TFSF box while its edges and the flux lines stay on base cells: the plane_wave_tfsf '
                    'incident line runs on the graded x widths and the side corrections use the local spacings '
                    '(solver_plan S7.1 lift; a mismatch would leak incident light into the scattered-field box). Same '
                    f'check and tolerance; raises unless both builds reached dx_min <= {_GRADED_DX_MIN_FRACTION:g} x '
                    'the base cell. Not yet confirmed by a run.',
        specs=_graded_mie_specs,
        check=lambda results_by_name: _check_mie_cylinder(require_graded(results_by_name, _MIE_CELL))),
    mie_drude_cylinder_case(_PHYSICS_TYPE, 'hz'),
)
