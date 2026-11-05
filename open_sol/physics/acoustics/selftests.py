"""Analytic elastic / acoustic validation cases (plane P and S waves in a solid, normal reflection at a fluid-fluid
interface, a thermoelastic slab, rigid-walled plate resonances, axisymmetric cylinder modes and hoop stress, the
half_storage stepper, monoclinic c16 / c26 plane waves) and the 'photoacoustic_disc' template spec, registered by definition.register; the spec factories
return fresh objects on every call."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import solver_role
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           SolverSpec, StudySpec)

ELASTIC_TYPE_NAME = 'elastic_waves'   # = definition.PHYSICS_TYPE_NAME (definition imports this module)
PHYSICS_NAME = 'acoustics'
STRIP_CELLS_Y = 4                     # periodic strip width (cells): every wave stays plane
STRIP_SETTINGS = {'x': {'bc_y_lo': 'periodic', 'bc_y_hi': 'periodic', 'pml_cells': 20},    # waves along x
                  'y': {'bc_x_lo': 'periodic', 'bc_x_hi': 'periodic', 'pml_cells': 20}}    # waves along y
RICKER_FREQUENCY = 1e6                # Hz; the Ricker wavelet peaks (value 1) at t0 = 4 / f

# elastic_wave_speeds: a traction sheet at SHEET_X launches plane P (normal traction) or S (tangential) waves; probes
# at NEAR_X / FAR_X (cell centres) give the speed from the cross-correlation delay and the amplitude T / (2 rho c).
# Run p_wave_y repeats the P wave along a y strip of an orthotropic crystal (c22 != c11: pins c22, d/dy, y-CPML).
SOLID = {'rho': 2200.0, 'sound_speed': 5960.0, 'shear_speed': 3760.0}   # silica-like
ORTHOTROPIC = {'rho': 2200.0, 'c11': 78.15e9, 'c12': 16e9, 'c22': 46.9e9, 'c66': 31.1e9}   # c22 / rho = 4617^2
SPEED_CELL = 1e-4                     # m: ~37 cells per S wavelength at the Ricker peak (positions stay on whole cells)
SPEED_LENGTH = 50e-3                  # m
SHEET_X = 10e-3
NEAR_X = 20e-3 + 0.5 * SPEED_CELL
FAR_X = 40e-3 + 0.5 * SPEED_CELL
SHEET_TRACTION = 1e6                  # Pa
# s per run: the pulse (+-1.5 us around its peak) has passed FAR_X (peaks 9.0 / 10.5 / 12.0 us); the cut tail holds
# only CPML echoes
SPEED_DURATIONS = {'p_wave': 12.5e-6, 'p_wave_y': 13e-6, 's_wave': 14e-6}
SPEED_TOLERANCE = 1e-2

# acoustic_normal_reflection: plane pressure pulse in fluid 1 hits fluid 2 at INTERFACE_X at normal incidence.
FLUID_1 = {'rho': 1000.0, 'sound_speed': 1480.0}
FLUID_2 = {'rho': 2000.0, 'sound_speed': 2500.0}
REFLECTION_CELL = 2.5e-5              # m: ~60 cells per wavelength in fluid 1 at the Ricker peak
REFLECTION_LENGTH = 20e-3
REFLECTION_SHEET_X = 5e-3
INCIDENT_X = 8e-3 + 0.5 * REFLECTION_CELL
INTERFACE_X = 12e-3                   # a cell edge
TRANSMITTED_X = 15e-3 + 0.5 * REFLECTION_CELL
# s: peaks incident 6.0, transmitted 9.9, reflected 11.4 us (+-1.5 us); the x_lo CPML echo of the left-going pulse
# (12.8 us, <= 1 %) comes after the reflected peak, the x_hi echo of the transmitted one after the end
REFLECTION_DURATION = 13e-6
REFLECTION_TOLERANCE = 1e-2

# acoustic_thermoelastic_slab: a heating pulse q = Q exp(-((t - delay) / width)^2) on SLAB_X0..SLAB_X1 of a fluid with
# Grueneisen parameter GRUNEISEN; right of the slab p(x, t) = (G Q / 2) [F(t - (x - x1) / c) - F(t - (x - x0) / c)],
# F the time integral of the envelope, plateau G Q sqrt(pi) width / 2 = G H / 2.
SLAB_FLUID = {'rho': 1000.0, 'sound_speed': 1500.0, 'gruneisen': 0.11}
SLAB_CELL = 1.25e-5
SLAB_LENGTH = 20e-3                   # end echoes reach the probe after SLAB_DURATION
SLAB_X0, SLAB_X1 = 10e-3, 12e-3       # plateau (x1 - x0) / c = 1.33 us, several envelope widths
SLAB_PROBE_X = 14e-3 + 0.5 * SLAB_CELL
HEATING_POWER = 1e11                  # W/m^3
PULSE_WIDTH = 2e-7                    # s
PULSE_DELAY = 6e-7                    # s
SLAB_DURATION = 5e-6                  # s: the trailing front has passed the probe by ~3.8 us
SLAB_TOLERANCE = 2e-2

# acoustic_plate_resonances: a solid rectangle with rigid (slip) walls on the grid edges rings after a force pulse;
# P modes phi = cos(p pi x / Lx) cos(q pi y / Ly), S modes psi = sin sin, L = cells x PLATE_CELL. In the band: S(1,1)
# ~633 kHz, P(1,0) ~802 kHz, S(2,1) ~913 kHz; P(0,1) / S(1,2) (~1.07 / 1.08 MHz) lie above it.
PLATE_SOLID = {'rho': 2700.0, 'sound_speed': 6420.0, 'shear_speed': 3040.0}   # aluminium-like
PLATE_CELL = 1e-4                     # m
PLATE_CELLS = (40, 30)
PLATE_MODES = (('S', 1, 1), ('P', 1, 0), ('S', 2, 1))
PLATE_BAND = (0.55e6, 0.95e6)         # Hz
PLATE_FORCE_CELL = (10, 21)           # off the nodal lines of the three modes
PLATE_PROBE_CELL = (28, 6)            # vx on the +x face of this cell
PLATE_RICKER_FREQUENCY = 0.8e6        # Hz
# s: the window after the pulse end 2 t0 = 10 us is 40 us = 22 periods of the band floor (validation warns below 10)
# and ~4.4 / delta-f of the closest pair (S(2,1) - P(1,0)); ~55 decimated samples (every ~63 steps after the filter)
PLATE_DURATION = 50e-6
PLATE_COURANT = 0.99                  # the check reads dt from the table
PLATE_TOLERANCE = 1e-3

# elastic_axisymmetric_cylinder: geometry axisymmetric, PLATE_SOLID filling r < a = CYLINDER_CELLS cells with a slip
# wall at r = a, periodic z strip of STRIP_CELLS_Y cells; a ring traction sheet through the strip excites only
# z-uniform modes. Run 'radial' (normal traction, probe vx): breathing modes u_r ~ J1(k r); run 'axial' (tangential
# traction, probe vy): axial shear modes u_z ~ J0(k r); both have J1(k a) = 0 and f = c k / (2 pi), c = sound_speed /
# shear_speed; run radial carries a second monitor 'hoop_resonances' on the hoop stress stt in the same probe cell (the
# breathing modes again). Run entries: (traction direction, sheet cell, probe cell, Ricker f [Hz], band [Hz]);
# CYLINDER_MONITORS: mode label -> (run, resonance_monitor feature, probe component).
CYLINDER_CELL = 1e-4                  # m
CYLINDER_CELLS = 48                   # a = 4.8 mm
BESSEL_J1_ZEROS = (3.831705970207512, 7.015586669815619)
CYLINDER_RUNS = {'radial': ('x', 14, 33, 1.1e6, (0.6e6, 1.8e6)),     # ~816 kHz, ~1.49 MHz (2.17 MHz above)
                 'axial': ('y', 2, 46, 0.55e6, (0.3e6, 0.85e6))}    # ~386, ~707 kHz (1.03 MHz above)
CYLINDER_MONITORS = {'radial': ('radial', 'resonances', 'vx'), 'hoop': ('radial', 'hoop_resonances', 'stt'),
                     'axial': ('axial', 'resonances', 'vy')}
# s: the axial window after its pulse end 2 t0 = 14.5 us holds 10 periods of its 0.3 MHz band floor (validation warns
# below 10) and ~10 / delta-f (mode spacings >= 320 kHz); radial 24 periods
CYLINDER_DURATION = 48e-6
CYLINDER_DISCRETE_TOLERANCE = 1e-3    # vs the exact discrete modes of the scheme
CYLINDER_CONTINUUM_TOLERANCE = 3e-3   # vs the Bessel modes (O((k dx)^2) dispersion ~1e-3)

# elastic_axisymmetric_hoop_stress: geometry axisymmetric, PLATE_SOLID with HOOP_GRUENEISEN filling r < HOOP_CELLS
# cells (slip wall at r = a, periodic z strip) heated uniformly by a heat_pulse_source: nothing moves (the ring-volume
# forms balance a uniform stress exactly), so s_rr = s_zz = s_tt = -G H and pressure = G H, H = q0 x the integral of the
# envelope up to the sample time.
HOOP_CELLS = 16
HOOP_GRUENEISEN = 2.0
HOOP_POWER = 1e12                     # W/m^3 (G H ~ 0.35 MPa)
HOOP_PULSE_WIDTH = 1e-7               # s
HOOP_PULSE_DELAY = 3e-7               # s
HOOP_DURATION = 1e-6                  # s: the pulse ends at 7.5e-7 s
HOOP_PROBE_CELL = 5
HOOP_TOLERANCE = 1e-4                 # float32 sums of ~100 increments and the radius-row rounding

# elastic_waves_half_storage: run p_wave of elastic_wave_speeds and run slab of acoustic_thermoelastic_slab on an
# explicit_kernels stepper node at precision half_storage; each error over HALF_STORAGE_FACTOR x its single tolerance.
HALF_STORAGE_FACTOR = 2.0

# elastic_monoclinic_plane_waves (piezoelectric plan stage 3): MONOCLINIC_SOLID (positive definite with c16, c26 != 0).
# Ring runs (x_ring / y_ring): fully periodic boxes of RING_CELLS along the axis (4 across) at courant 0.99, x + y
# traction sheets, a resonance_monitor; plane modes k = 2 pi n / L along the axis, discrete symbol sk^2 [[c11, c16 a],
# [c16 a, c66]] (x; y: [[c66, c26 a], [c26 a, c22]]), sk = (2 / h) sin(k h / 2), a = cos(k h / 2) (the corner / centre
# averages), leap-frog sin(omega dt / 2) = dt omega_c / 2, Q = inf. Launch runs (x_launch / y_launch): absorbing strips,
# a tangential Ricker sheet, the cross-polarised probe (x: force y, probe vx; y: force x, probe vy) against the exact
# plane waves sum_m e_m (e_m . f) T s(t - d / c_m) / (2 rho c_m): its sign follows c16 / c26 (the ring spectrum only
# sees their squares).
MONOCLINIC_SOLID = {'rho': 4700.0, 'c11': 2.0e11, 'c12': 0.7e11, 'c22': 1.6e11, 'c66': 0.6e11, 'c16': 0.2e11,
                    'c26': -0.15e11}
RING_CELL = 1e-5                      # m
RING_CELLS = 64                       # L = 0.64 mm: qS / qP n = 1 at ~5.5 / ~9.2-10.3 MHz
RING_SHEET_CELL, RING_PROBE_CELL = 7, 37
RING_COURANT = 0.99
RING_RICKER_FREQUENCY = 1.2e7         # Hz
RING_BAND = (4e6, 2.4e7)              # Hz: n = 1, 2 of both branches (qS n = 3, 4 also inside)
RING_MODES = (1, 2)
# s: the window after the pulse end 2 t0 = 0.67 us holds 21 periods of the 4 MHz band floor and ~3.4 / delta-f of the
# closest pair (x qP1 / qS2, 0.63 MHz)
RING_DURATION = 6e-6
RING_DISCRETE_TOLERANCE = 1e-4
RING_CONTINUUM_TOLERANCE = 1e-3       # n = 1 only: O((k h)^2) ~4e-4 at 64 cells per wavelength
RING_DT_TOLERANCE = 1e-5              # float32 coefficients
LAUNCH_CELL = 2.5e-5                  # m: ~140 cells per qS wavelength at the Ricker peak
LAUNCH_LENGTH = 20e-3                 # the x_hi CPML echo (20 mm more path) arrives after LAUNCH_DURATION
LAUNCH_SHEET_CELL, LAUNCH_PROBE_CELL = 200, 400
LAUNCH_DURATION = 7e-6                # s: the qS pulse (peak ~5.4 us) passes the probe by ~6.9 us
LAUNCH_TOLERANCE = 1e-2               # dispersion at >= 56 cells per wavelength up to 2.5 x the Ricker peak
MONOCLINIC_RUNS = {'x': ('c16', 'y', 'vx'), 'y': ('c26', 'x', 'vy')}   # axis: (term, launch force, launch probe)


def _strip_spec(title, cell, length, materials, geometry, features, duration, axis='x'):
    """Uniform-mesh strip along axis (length), periodic across it (STRIP_CELLS_Y cells), absorbing ends,
    time-dependent study."""
    width = STRIP_CELLS_Y * cell
    size_x, size_y = (length, width) if axis == 'x' else (width, length)
    return ModelSpec(
        title=title,
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=list(geometry),
        materials=list(materials),
        physics=[PhysicsSpec(PHYSICS_NAME, ELASTIC_TYPE_NAME, settings=dict(STRIP_SETTINGS[axis]),
                             features=list(features))],
        mesh=MeshSpec(mode='uniform', dx=cell),
        study=StudySpec(type='time_dependent', settings={'duration': duration}))


def _sheet(name, position, cell, axis='x'):
    """Line across a strip along axis at that coordinate."""
    width = STRIP_CELLS_Y * cell
    corners = ({'x0': position, 'y0': 0.0, 'x1': position, 'y1': width} if axis == 'x'
               else {'x0': 0.0, 'y0': position, 'x1': width, 'y1': position})
    return GeometrySpec(name, 'line', corners)


def _point(name, position, cell, axis='x'):
    along, across = ('x', 'y') if axis == 'x' else ('y', 'x')
    return GeometrySpec(name, 'point', {along: position, across: 1.5 * cell})


def _ricker_sheet(direction, traction=SHEET_TRACTION):
    return {'waveform': 'ricker', 'frequency': RICKER_FREQUENCY, 'traction': traction, 'direction': direction}


# ---- specs ----
def _wave_speed_runs():
    """'p_wave': normal traction, probes record vx; 's_wave': tangential traction, probes record vy (x strips of
    SOLID); 'p_wave_y': normal traction on a y strip of ORTHOTROPIC, probes record vy."""
    runs = {}
    for run, material, axis, direction, component in (('p_wave', SOLID, 'x', 'x', 'vx'),
                                                      ('s_wave', SOLID, 'x', 'y', 'vy'),
                                                      ('p_wave_y', ORTHOTROPIC, 'y', 'y', 'vy')):
        runs[run] = _strip_spec(
            f'Plane {run[0].upper()} wave along {axis} in an elastic solid', SPEED_CELL, SPEED_LENGTH,
            [MaterialSpec('solid', properties=dict(material), selection=['all'])],
            [_sheet('sheet', SHEET_X, SPEED_CELL, axis), _point('near_point', NEAR_X, SPEED_CELL, axis),
             _point('far_point', FAR_X, SPEED_CELL, axis)],
            [FeatureSpec('sheet', 'traction_line_source', ['sheet'], _ricker_sheet(direction)),
             FeatureSpec('near', 'field_probe', ['near_point'], {'component': component}),
             FeatureSpec('far', 'field_probe', ['far_point'], {'component': component})],
            SPEED_DURATIONS[run], axis)
    return runs


def _expected_wave_speeds():
    """{run: (rho, exact speed)} of the elastic_wave_speeds runs."""
    return {'p_wave': (SOLID['rho'], SOLID['sound_speed']), 's_wave': (SOLID['rho'], SOLID['shear_speed']),
            'p_wave_y': (ORTHOTROPIC['rho'], math.sqrt(ORTHOTROPIC['c22'] / ORTHOTROPIC['rho']))}


def _reflection_runs():
    rectangle = GeometrySpec('dense_region', 'rectangle', {'x0': INTERFACE_X, 'y0': 0.0, 'x1': REFLECTION_LENGTH,
                                                           'y1': STRIP_CELLS_Y * REFLECTION_CELL})
    return {'run': _strip_spec(
        'Normal reflection at a fluid-fluid interface', REFLECTION_CELL, REFLECTION_LENGTH,
        [MaterialSpec('fluid_1', properties=dict(FLUID_1), selection=['all']),
         MaterialSpec('fluid_2', properties=dict(FLUID_2), selection=['dense_region'])],
        [rectangle, _sheet('sheet', REFLECTION_SHEET_X, REFLECTION_CELL),
         _point('incident_point', INCIDENT_X, REFLECTION_CELL),
         _point('transmitted_point', TRANSMITTED_X, REFLECTION_CELL)],
        [FeatureSpec('sheet', 'traction_line_source', ['sheet'], _ricker_sheet('x')),
         FeatureSpec('incident', 'field_probe', ['incident_point'], {'component': 'pressure'}),
         FeatureSpec('transmitted', 'field_probe', ['transmitted_point'], {'component': 'pressure'})],
        REFLECTION_DURATION)}


def _thermoelastic_runs():
    slab = GeometrySpec('slab', 'rectangle', {'x0': SLAB_X0, 'y0': 0.0, 'x1': SLAB_X1, 'y1': STRIP_CELLS_Y * SLAB_CELL})
    return {'run': _strip_spec(
        'Thermoelastic pressure of a heated slab', SLAB_CELL, SLAB_LENGTH,
        [MaterialSpec('fluid', properties=dict(SLAB_FLUID), selection=['all'])],
        [slab, _point('probe_point', SLAB_PROBE_X, SLAB_CELL)],
        [FeatureSpec('heating', 'heat_pulse_source', ['slab'],
                     {'power_density': HEATING_POWER, 'pulse_width': PULSE_WIDTH, 'delay': PULSE_DELAY}),
         FeatureSpec('probe', 'field_probe', ['probe_point'], {'component': 'pressure'})],
        SLAB_DURATION)}


def _plate_runs():
    """'run': PLATE_SOLID filling a box of rigid faces, a point force pulse at 30 deg, a resonance_monitor on vx."""
    size_x, size_y = (cells * PLATE_CELL for cells in PLATE_CELLS)
    force_x, force_y = ((index + 0.5) * PLATE_CELL for index in PLATE_FORCE_CELL)
    probe_x, probe_y = ((index + 0.5) * PLATE_CELL for index in PLATE_PROBE_CELL)
    return {'run': ModelSpec(
        title='Self-test: elastic plate resonances',
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=[GeometrySpec('force_point', 'point', {'x': force_x, 'y': force_y}),
                  GeometrySpec('probe_point', 'point', {'x': probe_x, 'y': probe_y})],
        materials=[MaterialSpec('solid', properties=dict(PLATE_SOLID), selection=['all'])],
        physics=[PhysicsSpec(PHYSICS_NAME, ELASTIC_TYPE_NAME,
                             settings={**{f'bc_{face}': 'rigid' for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi')},
                                       'courant': PLATE_COURANT},
                             features=[FeatureSpec('force', 'force_source', ['force_point'],
                                                   {'waveform': 'ricker', 'frequency': PLATE_RICKER_FREQUENCY,
                                                    'force': 1.0, 'angle': 30.0}),
                                       FeatureSpec('resonances', 'resonance_monitor', ['probe_point'],
                                                   {'component': 'vx', 'frequency_min': PLATE_BAND[0],
                                                    'frequency_max': PLATE_BAND[1]})])],
        mesh=MeshSpec(mode='uniform', dx=PLATE_CELL),
        study=StudySpec(type='time_dependent', settings={'duration': PLATE_DURATION}))}


def _cylinder_runs():
    """'radial' / 'axial': the (r, z) cylinder strip with a vertical ring traction sheet and the CYLINDER_MONITORS
    resonance_monitors of that run (radial: vx and stt in one probe cell)."""
    radius, height = CYLINDER_CELLS * CYLINDER_CELL, STRIP_CELLS_Y * CYLINDER_CELL
    settings = {'geometry': 'axisymmetric', 'bc_x_lo': 'rigid', 'bc_x_hi': 'rigid', 'bc_y_lo': 'periodic',
                'bc_y_hi': 'periodic'}
    runs = {}
    for run, (direction, sheet_cell, probe_cell, frequency, band) in CYLINDER_RUNS.items():
        sheet_r, probe_r = ((cell + 0.5) * CYLINDER_CELL for cell in (sheet_cell, probe_cell))
        monitors = [FeatureSpec(feature, 'resonance_monitor', ['probe_point'],
                                {'component': component, 'frequency_min': band[0], 'frequency_max': band[1]})
                    for monitor_run, feature, component in CYLINDER_MONITORS.values() if monitor_run == run]
        runs[run] = ModelSpec(
            title=f'Self-test: axisymmetric cylinder, {run} modes',
            domain=DomainSpec(size_x=radius, size_y=height),
            geometry=[GeometrySpec('sheet', 'line', {'x0': sheet_r, 'y0': 0.0, 'x1': sheet_r, 'y1': height}),
                      GeometrySpec('probe_point', 'point', {'x': probe_r, 'y': 1.5 * CYLINDER_CELL})],
            materials=[MaterialSpec('solid', properties=dict(PLATE_SOLID), selection=['all'])],
            physics=[PhysicsSpec(PHYSICS_NAME, ELASTIC_TYPE_NAME, settings=dict(settings), features=[
                FeatureSpec('sheet', 'traction_line_source', ['sheet'],
                            {'waveform': 'ricker', 'frequency': frequency, 'traction': SHEET_TRACTION,
                             'direction': direction}),
                *monitors])],
            mesh=MeshSpec(mode='uniform', dx=CYLINDER_CELL),
            study=StudySpec(type='time_dependent', settings={'duration': CYLINDER_DURATION}))
    return runs


def _hoop_runs():
    """'uniform': the (r, z) strip of HOOP_CELLS cells heated everywhere, probes on stt and pressure, stt stored."""
    radius, height = HOOP_CELLS * CYLINDER_CELL, STRIP_CELLS_Y * CYLINDER_CELL
    probe = GeometrySpec('probe_point', 'point', {'x': (HOOP_PROBE_CELL + 0.5) * CYLINDER_CELL,
                                                  'y': 1.5 * CYLINDER_CELL})
    return {'uniform': ModelSpec(
        title='Self-test: axisymmetric hoop stress of a uniformly heated cylinder',
        domain=DomainSpec(size_x=radius, size_y=height),
        geometry=[probe],
        materials=[MaterialSpec('solid', properties={**PLATE_SOLID, 'gruneisen': HOOP_GRUENEISEN}, selection=['all'])],
        physics=[PhysicsSpec(PHYSICS_NAME, ELASTIC_TYPE_NAME,
                             settings={'geometry': 'axisymmetric', 'bc_x_lo': 'rigid', 'bc_x_hi': 'rigid',
                                       'bc_y_lo': 'periodic', 'bc_y_hi': 'periodic'},
                             features=[FeatureSpec('heating', 'heat_pulse_source', ['all'],
                                                   {'power_density': HOOP_POWER, 'pulse_width': HOOP_PULSE_WIDTH,
                                                    'delay': HOOP_PULSE_DELAY}),
                                       FeatureSpec('hoop_probe', 'field_probe', ['probe_point'], {'component': 'stt'}),
                                       FeatureSpec('pressure_probe', 'field_probe', ['probe_point'],
                                                   {'component': 'pressure'})])],
        mesh=MeshSpec(mode='uniform', dx=CYLINDER_CELL),
        study=StudySpec(type='time_dependent', settings={'duration': HOOP_DURATION},
                        stored_fields=[f'{PHYSICS_NAME}.stt']))}


def _half_storage_runs():
    """'p_wave' (elastic_wave_speeds) and 'slab' (acoustic_thermoelastic_slab) on a half_storage stepper node."""
    runs = {'p_wave': _wave_speed_runs()['p_wave'], 'slab': _thermoelastic_runs()['run']}
    for spec in runs.values():
        spec.title += ', half-precision field storage'
        spec.solvers = [SolverSpec('fp16_fields', 'explicit_kernels', {'precision': 'half_storage'})]
        spec.physics[0].solvers = {'stepper': 'fp16_fields'}
    return runs


def _monoclinic_runs():
    """'x_ring' / 'y_ring': periodic boxes of MONOCLINIC_SOLID with x and y Ricker sheets and a resonance_monitor;
    'x_launch' / 'y_launch': absorbing strips with one tangential sheet and the cross-polarised probe."""
    material = [MaterialSpec('crystal', properties=dict(MONOCLINIC_SOLID), selection=['all'])]
    runs = {}
    for axis, (_, force, component) in MONOCLINIC_RUNS.items():
        length, width = RING_CELLS * RING_CELL, STRIP_CELLS_Y * RING_CELL
        size_x, size_y = (length, width) if axis == 'x' else (width, length)
        ring_component = 'vx' if axis == 'x' else 'vy'
        runs[f'{axis}_ring'] = ModelSpec(
            title=f'Self-test: monoclinic plane modes along {axis} (periodic box)',
            domain=DomainSpec(size_x=size_x, size_y=size_y),
            geometry=[_sheet('sheet', (RING_SHEET_CELL + 0.5) * RING_CELL, RING_CELL, axis),
                      _point('probe_point', (RING_PROBE_CELL + 0.5) * RING_CELL, RING_CELL, axis)],
            materials=list(material),
            physics=[PhysicsSpec(PHYSICS_NAME, ELASTIC_TYPE_NAME,
                                 settings={**{f'bc_{face}': 'periodic' for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi')},
                                           'courant': RING_COURANT},
                                 features=[*(FeatureSpec(f'sheet_{direction}', 'traction_line_source', ['sheet'],
                                                         {'waveform': 'ricker', 'frequency': RING_RICKER_FREQUENCY,
                                                          'traction': SHEET_TRACTION, 'direction': direction})
                                             for direction in ('x', 'y')),
                                           FeatureSpec('resonances', 'resonance_monitor', ['probe_point'],
                                                       {'component': ring_component, 'frequency_min': RING_BAND[0],
                                                        'frequency_max': RING_BAND[1]})])],
            mesh=MeshSpec(mode='uniform', dx=RING_CELL),
            study=StudySpec(type='time_dependent', settings={'duration': RING_DURATION}))
        runs[f'{axis}_launch'] = _strip_spec(
            f'Self-test: monoclinic cross-polarised plane waves along {axis}', LAUNCH_CELL, LAUNCH_LENGTH, material,
            [_sheet('sheet', (LAUNCH_SHEET_CELL + 0.5) * LAUNCH_CELL, LAUNCH_CELL, axis),
             _point('probe_point', (LAUNCH_PROBE_CELL + 0.5) * LAUNCH_CELL, LAUNCH_CELL, axis)],
            [FeatureSpec('sheet', 'traction_line_source', ['sheet'], _ricker_sheet(force)),
             FeatureSpec('probe', 'field_probe', ['probe_point'], {'component': component})],
            LAUNCH_DURATION, axis)
    return runs


PHOTOACOUSTIC_DESCRIPTION = (
    'elastic_waves (time_dependent): a 1 mm absorbing disc in water heated by a 100 ns pulse (heat_pulse_source, '
    'Grueneisen 0.2): the thermoelastic pressure (initial p0 = Grueneisen x absorbed energy density, ~35 kPa) leaves '
    'the disc as an N-shaped pulse recorded by a pressure probe 2 mm away.')
PHOTOACOUSTIC_SUMMARY = ('Photoacoustics: a short heating pulse in a small absorbing disc in water launches a pressure '
                         'wave, recorded by a probe nearby.')


def photoacoustic_disc_spec():
    """Template 'photoacoustic_disc': water, an absorbing disc heated by a gaussian pulse, a pressure probe."""
    size = 6e-3
    return ModelSpec(
        title='Photoacoustic disc in water',
        description=PHOTOACOUSTIC_DESCRIPTION,
        domain=DomainSpec(size_x=size, size_y=size),
        geometry=[GeometrySpec('absorber', 'circle', {'cx': 0.5 * size, 'cy': 0.5 * size, 'r': 0.5e-3}),
                  GeometrySpec('hydrophone_point', 'point', {'x': 0.5 * size + 2e-3, 'y': 0.5 * size})],
        materials=[MaterialSpec('water', properties={'rho': 1000.0, 'sound_speed': 1480.0, 'gruneisen': 0.11},
                                selection=['all']),
                   MaterialSpec('absorber', properties={'rho': 1000.0, 'sound_speed': 1500.0, 'gruneisen': 0.2},
                                selection=['absorber'])],
        physics=[PhysicsSpec(PHYSICS_NAME, ELASTIC_TYPE_NAME, features=[
            FeatureSpec('laser_pulse', 'heat_pulse_source', ['absorber'],
                        {'power_density': 1e12, 'pulse_width': 1e-7}),
            FeatureSpec('hydrophone', 'field_probe', ['hydrophone_point'], {'component': 'pressure'})])],
        mesh=MeshSpec(mode='uniform', dx=2e-5),
        study=StudySpec(type='time_dependent', settings={'duration': 4e-6, 'output_interval': 5e-7}))


# ---- checks (CPU, on Results-like objects) ----
def _probe(results, feature):
    data = results.result(PHYSICS_NAME, feature)
    return np.asarray(data['t'], dtype=np.float64), np.asarray(data['values'], dtype=np.float64)


def _delay(times, first, second):
    """Delay [s] of `second` behind `first` (equally sampled): cross-correlation peak with parabolic refinement."""
    correlation = np.correlate(second, first, mode='full')
    peak = int(np.argmax(correlation))
    shift = 0.0
    if 0 < peak < correlation.size - 1:
        left, centre, right = correlation[peak - 1:peak + 2]
        curvature = left - 2.0 * centre + right
        shift = 0.5 * (left - right) / curvature if curvature != 0.0 else 0.0
    return (peak - (first.size - 1) + shift) * float(times[1] - times[0])


def _check_wave_speeds(results_by_name):
    """Speeds from the near / far delay and the near peak velocity against sqrt(c/rho) and T / (2 rho c)."""
    return _wave_speed_result(results_by_name, tuple(_expected_wave_speeds()))


def _wave_speed_result(results_by_name, runs):
    """_check_wave_speeds over the named runs (p_wave among them)."""
    errors, details = {}, {}
    expected_speeds = _expected_wave_speeds()
    for run in runs:
        rho, speed = expected_speeds[run]
        times, near = _probe(results_by_name[run], 'near')
        _, far = _probe(results_by_name[run], 'far')
        if near.size < 3 or near.shape != far.shape:
            return {'measured': math.nan, 'expected': speed, 'error': math.inf, 'tolerance': SPEED_TOLERANCE,
                    'details': {'run': run, 'n_samples': [int(near.size), int(far.size)]}}
        delay = _delay(times, near, far)
        measured_speed = (FAR_X - NEAR_X) / delay if delay > 0.0 else math.nan
        peak_velocity = float(np.max(np.abs(near)))
        expected_velocity = SHEET_TRACTION / (2.0 * rho * speed)
        errors[f'{run}_speed'] = abs(measured_speed - speed) / speed
        errors[f'{run}_amplitude'] = abs(peak_velocity - expected_velocity) / expected_velocity
        details[run] = {'speed_m_per_s': measured_speed, 'expected_speed_m_per_s': speed,
                        'peak_velocity_m_per_s': peak_velocity, 'expected_peak_m_per_s': expected_velocity}
    error = max(errors.values())   # NaN propagates
    return {'measured': details['p_wave']['speed_m_per_s'], 'expected': SOLID['sound_speed'],
            'error': error if math.isfinite(error) else math.inf, 'tolerance': SPEED_TOLERANCE,
            'details': {'errors': errors, **details}}


def _signed_peak(values):
    """The value of largest magnitude (sign kept)."""
    return float(values[int(np.argmax(np.abs(values)))]) if values.size else math.nan


def _check_normal_reflection(results_by_name):
    """Reflected / incident and transmitted / incident peak pressures against (Z2 - Z1) / (Z2 + Z1), 2 Z2 / (Z2 + Z1)."""
    results = results_by_name['run']
    times, incident_trace = _probe(results, 'incident')
    transmitted_times, transmitted_trace = _probe(results, 'transmitted')
    c1 = FLUID_1['sound_speed']
    t0 = 4.0 / RICKER_FREQUENCY
    incident_arrival = t0 + (INCIDENT_X - REFLECTION_SHEET_X) / c1
    reflected_arrival = t0 + (2.0 * INTERFACE_X - REFLECTION_SHEET_X - INCIDENT_X) / c1
    split = 0.5 * (incident_arrival + reflected_arrival)
    incident = _signed_peak(incident_trace[times < split])
    reflected = _signed_peak(incident_trace[times >= split])
    transmitted = _signed_peak(transmitted_trace)
    impedance_1 = FLUID_1['rho'] * c1
    impedance_2 = FLUID_2['rho'] * FLUID_2['sound_speed']
    expected_r = (impedance_2 - impedance_1) / (impedance_2 + impedance_1)
    expected_t = 2.0 * impedance_2 / (impedance_2 + impedance_1)
    measured_r, measured_t = reflected / incident, transmitted / incident
    errors = {'reflection': abs(measured_r - expected_r) / abs(expected_r),
              'transmission': abs(measured_t - expected_t) / expected_t}
    error = max(errors.values())
    return {'measured': measured_r, 'expected': expected_r, 'error': error if math.isfinite(error) else math.inf,
            'tolerance': REFLECTION_TOLERANCE,
            'details': {'errors': errors, 'transmission': measured_t, 'expected_transmission': expected_t,
                        'incident_peak_Pa': incident, 'reflected_peak_Pa': reflected,
                        'transmitted_peak_Pa': transmitted, 'n_samples': int(times.size),
                        'transmitted_samples': int(transmitted_times.size)}}


def _thermoelastic_pressure(times):
    """Exact right-going pressure [Pa] at SLAB_PROBE_X of the heated slab in the uniform fluid."""
    from math import erf
    c = SLAB_FLUID['sound_speed']
    amplitude = 0.5 * SLAB_FLUID['gruneisen'] * HEATING_POWER * 0.5 * math.sqrt(math.pi) * PULSE_WIDTH

    def envelope_integral(t):   # (1 + erf) part of the integral of the envelope
        return np.array([1.0 + erf((value - PULSE_DELAY) / PULSE_WIDTH) for value in np.atleast_1d(t)])

    return amplitude * (envelope_integral(times - (SLAB_PROBE_X - SLAB_X1) / c)
                        - envelope_integral(times - (SLAB_PROBE_X - SLAB_X0) / c))


def _check_thermoelastic_slab(results_by_name):
    """max |p - p_exact| over the probe trace and |max p - G H / 2|, both relative to G H / 2."""
    times, pressure = _probe(results_by_name['run'], 'probe')
    plateau = 0.5 * SLAB_FLUID['gruneisen'] * HEATING_POWER * math.sqrt(math.pi) * PULSE_WIDTH
    if pressure.size == 0:
        return {'measured': math.nan, 'expected': plateau, 'error': math.inf, 'tolerance': SLAB_TOLERANCE,
                'details': {'n_samples': 0}}
    exact = _thermoelastic_pressure(times)
    peak = float(np.max(pressure))
    errors = {'waveform': float(np.max(np.abs(pressure - exact))) / plateau, 'plateau': abs(peak - plateau) / plateau}
    error = max(errors.values())
    return {'measured': peak, 'expected': plateau, 'error': error if math.isfinite(error) else math.inf,
            'tolerance': SLAB_TOLERANCE,
            'details': {'errors': errors, 'n_samples': int(times.size), 'absorbed_energy_J_per_m3':
                        HEATING_POWER * math.sqrt(math.pi) * PULSE_WIDTH}}


def _discrete_plate_frequency(kind, p, q, dt):
    """Frequency [Hz] of plate mode (p, q) from the lossless staggered leap-frog: sin(omega dt / 2) = (dt / 2) c K,
    K^2 = sum (2 / dx)^2 sin^2(p pi / (2 n)), c = sound_speed (P) or shear_speed (S)."""
    speed = PLATE_SOLID['sound_speed' if kind == 'P' else 'shear_speed']
    wavenumber_squared = sum((2.0 / PLATE_CELL) ** 2 * math.sin(index * math.pi / (2 * cells)) ** 2
                             for index, cells in zip((p, q), PLATE_CELLS))
    return 2.0 * math.asin(0.5 * dt * speed * math.sqrt(wavenumber_squared)) / (2.0 * math.pi * dt)


def _resonance_table(results, feature='resonances'):
    """(table, frequencies (float64 array), nearest) of that resonance_monitor; nearest(f) = (frequency, 1/Q) of the
    row closest to f, None without rows."""
    table = results.result(PHYSICS_NAME, feature)
    columns = list(table['columns'])
    rows = [list(row) for row in table['rows']]
    frequencies = np.asarray([row[columns.index('frequency')] for row in rows], dtype=np.float64)

    def nearest(frequency):
        if not frequencies.size:
            return None
        row = rows[int(np.argmin(np.abs(frequencies - frequency)))]
        measured = row[columns.index('frequency')]
        return measured, 2.0 * abs(row[columns.index('frequency_imag')]) / measured
    return table, frequencies, nearest


def _check_plate_resonances(results_by_name):
    """Frequency vs the discrete recursion and 1/Q (~0, lossless) of each plate mode; worst of both."""
    table, frequencies, nearest = _resonance_table(results_by_name['run'])
    errors, per_mode = [], {}
    for kind, p, q in PLATE_MODES:
        label = f'{kind}({p},{q})'
        expected = _discrete_plate_frequency(kind, p, q, float(table['dt']))
        found = nearest(expected)
        if found is None:
            errors.append(math.inf)
            per_mode[label] = None
            continue
        measured, inverse_q = found
        mode_error = max(abs(measured / expected - 1.0), inverse_q)
        errors.append(mode_error)
        per_mode[label] = {'frequency': measured, 'expected_frequency': expected, 'inverse_Q': inverse_q,
                           'error': mode_error}
    worst = int(np.argmax(errors))
    return {'measured': float(frequencies.size), 'expected': float(len(PLATE_MODES)), 'error': float(errors[worst]),
            'tolerance': PLATE_TOLERANCE,
            'details': {'worst_mode': '{}({},{})'.format(*PLATE_MODES[worst]), 'modes': per_mode,
                        'n_rows': int(frequencies.size), 'dt': table.get('dt'),
                        'sample_interval': table.get('sample_interval'), 'n_samples': table.get('n_samples'),
                        'note': table.get('note')}}


def _discrete_cylinder_frequencies(run, dt):
    """Lowest len(BESSEL_J1_ZEROS) nonzero mode frequencies [Hz] of the staggered (r, z) scheme on z-uniform fields:
    K v = omega^2 M v of the ring-volume energy (radial / hoop: v_r on the free faces, e_rr and e_tt = (v_lo + v_hi) /
    (2 r_c) per cell; axial: v_z at the centres, corner shear (outer corner pinned)), then sin(omega dt / 2) = dt
    omega_c / 2."""
    n, width = CYLINDER_CELLS, CYLINDER_CELL
    rho = PLATE_SOLID['rho']
    shear_modulus = rho * PLATE_SOLID['shear_speed'] ** 2
    c11 = rho * PLATE_SOLID['sound_speed'] ** 2
    c12 = c11 - 2.0 * shear_modulus
    centres = (np.arange(n) + 0.5) * width
    faces = (np.arange(n) + 1.0) * width
    if run != 'axial':
        free = n - 1                                  # v_r faces 0..n-2; face n-1 is the wall
        radial_strain, hoop_strain = np.zeros((n, free)), np.zeros((n, free))
        for cell in range(n):
            for face, sign in ((cell, 1.0), (cell - 1, -1.0)):
                if 0 <= face < free:
                    radial_strain[cell, face] = sign / width
                    hoop_strain[cell, face] = 0.5 / centres[cell]
        weight = centres * width
        stiffness = ((radial_strain.T * weight) @ (c11 * radial_strain + c12 * hoop_strain)
                     + (hoop_strain.T * weight) @ (c12 * radial_strain + c11 * hoop_strain))
        mass = rho * faces[:free] * width
    else:
        corners = n - 1                               # corner n-1 on the wall carries no shear
        shear_strain = np.zeros((corners, n))
        for corner in range(corners):
            shear_strain[corner, corner], shear_strain[corner, corner + 1] = -1.0 / width, 1.0 / width
        stiffness = (shear_strain.T * (faces[:corners] * width)) @ (shear_modulus * shear_strain)
        mass = rho * centres * width
    scale = 1.0 / np.sqrt(mass)
    omega_squared = np.linalg.eigvalsh(stiffness * scale[:, None] * scale[None, :])
    omega = np.sqrt(omega_squared[omega_squared > 1e-6 * omega_squared.max()])   # drops the rigid axial translation
    return 2.0 * np.arcsin(0.5 * dt * omega[:len(BESSEL_J1_ZEROS)]) / (2.0 * math.pi * dt)


def _check_cylinder_modes(results_by_name):
    """Per mode the worst of |f / f_discrete - 1| and 1/Q over CYLINDER_DISCRETE_TOLERANCE and |f / f_Bessel - 1|
    over CYLINDER_CONTINUUM_TOLERANCE (error <= 1 passes)."""
    radius = CYLINDER_CELLS * CYLINDER_CELL
    errors, per_mode = [], {}
    for mode_label, (run, feature, _) in CYLINDER_MONITORS.items():
        table, _, nearest = _resonance_table(results_by_name[run], feature)
        speed = PLATE_SOLID['shear_speed' if run == 'axial' else 'sound_speed']
        for index, (zero, discrete) in enumerate(zip(BESSEL_J1_ZEROS,
                                                     _discrete_cylinder_frequencies(run, float(table['dt'])))):
            label = f'{mode_label}_{index + 1}'
            continuum = speed * zero / (2.0 * math.pi * radius)
            found = nearest(discrete)
            if found is None:
                errors.append(math.inf)
                per_mode[label] = None
                continue
            measured, inverse_q = found
            discrete_error = max(abs(measured / discrete - 1.0), inverse_q)
            continuum_error = abs(measured / continuum - 1.0)
            errors.append(max(discrete_error / CYLINDER_DISCRETE_TOLERANCE,
                              continuum_error / CYLINDER_CONTINUUM_TOLERANCE))
            per_mode[label] = {'frequency': measured, 'discrete_frequency': float(discrete),
                               'bessel_frequency': continuum, 'inverse_Q': inverse_q,
                               'discrete_error': discrete_error, 'continuum_error': continuum_error}
    worst = max(errors)
    first = per_mode.get('radial_1') or {}
    return {'measured': first.get('frequency', math.nan), 'expected': first.get('bessel_frequency', math.nan),
            'error': worst if math.isfinite(worst) else math.inf, 'tolerance': 1.0,
            'details': {'modes': per_mode, 'discrete_tolerance': CYLINDER_DISCRETE_TOLERANCE,
                        'continuum_tolerance': CYLINDER_CONTINUUM_TOLERANCE}}


def _absorbed_energy_density(time):
    """H [J/m^3] of the hoop-stress heating pulse up to `time`: q0 w sqrt(pi) / 2 (erf((t - d) / w) + erf(d / w))."""
    return 0.5 * HOOP_POWER * HOOP_PULSE_WIDTH * math.sqrt(math.pi) * (
        math.erf((time - HOOP_PULSE_DELAY) / HOOP_PULSE_WIDTH) + math.erf(HOOP_PULSE_DELAY / HOOP_PULSE_WIDTH))


def _check_hoop_stress(results_by_name):
    """Stored stt field (every cell), the last stt and pressure probe samples against -G H and G H, relative to G H."""
    results = results_by_name['uniform']
    times, hoop = _probe(results, 'hoop_probe')
    _, pressure = _probe(results, 'pressure_probe')
    if hoop.size == 0 or pressure.size == 0:
        return {'measured': math.nan, 'expected': math.nan, 'error': math.inf, 'tolerance': HOOP_TOLERANCE,
                'details': {'n_samples': [int(hoop.size), int(pressure.size)]}}
    expected = HOOP_GRUENEISEN * _absorbed_energy_density(float(times[-1]))
    field = np.asarray(results.field(f'{PHYSICS_NAME}.stt').values, dtype=np.float64)
    errors = {'stt_field': float(np.max(np.abs(field + expected))) / expected,
              'stt_probe': abs(float(hoop[-1]) + expected) / expected,
              'pressure_probe': abs(float(pressure[-1]) - expected) / expected}
    error = max(errors.values())
    return {'measured': float(hoop[-1]), 'expected': -expected, 'error': error if math.isfinite(error) else math.inf,
            'tolerance': HOOP_TOLERANCE,
            'details': {'errors': errors, 'field_shape': list(field.shape), 'final_time_s': float(times[-1])}}


def _check_half_storage(results_by_name):
    """elastic_wave_speeds on p_wave and acoustic_thermoelastic_slab on slab, each error over HALF_STORAGE_FACTOR x
    its tolerance (passes at <= 1); raises unless summary 'solvers' records the half_storage stepper of both runs."""
    for run, results in results_by_name.items():
        precision = solver_role(results, PHYSICS_NAME, 'stepper').get('precision')
        if precision != 'half_storage':
            raise ValueError(f"run {run}: expected the stepper to run half_storage, summary 'solvers' records "
                             f'{precision!r}')
    parts = {'p_wave': _wave_speed_result(results_by_name, ('p_wave',)),
             'slab': _check_thermoelastic_slab({'run': results_by_name['slab']})}
    errors = {run: part['error'] / (HALF_STORAGE_FACTOR * part['tolerance']) for run, part in parts.items()}
    error = max(errors.values())
    return {'measured': parts['p_wave']['measured'], 'expected': parts['p_wave']['expected'],
            'error': error if math.isfinite(error) else math.inf, 'tolerance': 1.0,
            'details': {'normalised_errors': errors, **{run: part['details'] for run, part in parts.items()}}}


def _monoclinic_tensor(axis, averaging=1.0):
    """2 x 2 acoustic tensor [Pa] of MONOCLINIC_SOLID for waves along axis in (vx, vy), the shear coupling times
    averaging (the discrete cos(k h / 2))."""
    solid = MONOCLINIC_SOLID
    if axis == 'x':
        return np.array([[solid['c11'], averaging * solid['c16']], [averaging * solid['c16'], solid['c66']]])
    return np.array([[solid['c66'], averaging * solid['c26']], [averaging * solid['c26'], solid['c22']]])


def _monoclinic_time_step():
    """The build's dt at RING_COURANT on square cells: courant h / (sqrt(S / rho) sqrt(2)), S the stability bound with
    |c16|, |c26| (definition.stability_stiffness, weights 1/2)."""
    solid = MONOCLINIC_SOLID
    c11, c12, c22, c66 = (solid[name] for name in ('c11', 'c12', 'c22', 'c66'))
    c16, c26 = abs(solid['c16']), abs(solid['c26'])
    p = 0.5 * (c11 + c66) + c16
    q = 0.5 * (c66 + c22) + c26
    r = 0.5 * (c16 + abs(c12 + c66) + c26)
    bound = max(0.5 * (p + q) + math.hypot(0.5 * (p - q), r), c11, c22)
    return RING_COURANT * RING_CELL / (math.sqrt(bound / solid['rho']) * math.sqrt(2.0))


def _ring_frequencies(axis, mode, dt):
    """((qS, qP) discrete leap-frog, (qS, qP) continuum Christoffel) frequencies [Hz] of ring mode `mode`."""
    rho, length = MONOCLINIC_SOLID['rho'], RING_CELLS * RING_CELL
    half_phase = mode * math.pi / RING_CELLS   # k h / 2
    symbol = (2.0 / RING_CELL * math.sin(half_phase)) ** 2
    omega = np.sqrt(np.linalg.eigvalsh(_monoclinic_tensor(axis, math.cos(half_phase))) * symbol / rho)
    discrete = 2.0 * np.arcsin(0.5 * dt * omega) / (2.0 * math.pi * dt)
    continuum = mode * np.sqrt(np.linalg.eigvalsh(_monoclinic_tensor(axis)) / rho) / length
    return discrete, continuum


def _ricker(times):
    """s(t) of the Ricker sheet waveform (peak 1 at 4 / f)."""
    argument = math.pi * RICKER_FREQUENCY * (np.asarray(times, dtype=np.float64) - 4.0 / RICKER_FREQUENCY)
    return (1.0 - 2.0 * argument ** 2) * np.exp(-argument ** 2)


def _launch_exact(axis, times):
    """Exact cross-polarised probe trace [m/s]: sum over the two plane modes of e_p (e . f) T s(t - d / c) / (2 rho c),
    d from the sheet (tangential: the cell-centre faces of its line) to the probe (the hi face of its cell)."""
    _, force, component = MONOCLINIC_RUNS[axis]
    rho = MONOCLINIC_SOLID['rho']
    distance = (LAUNCH_PROBE_CELL + 1.0 - LAUNCH_SHEET_CELL - 0.5) * LAUNCH_CELL
    force_vector = np.array([1.0, 0.0]) if force == 'x' else np.array([0.0, 1.0])
    probe_index = 0 if component == 'vx' else 1
    values, vectors = np.linalg.eigh(_monoclinic_tensor(axis))
    trace = np.zeros(np.shape(times))
    for value, vector in zip(values, vectors.T):
        speed = math.sqrt(value / rho)
        trace += (vector[probe_index] * float(vector @ force_vector) * SHEET_TRACTION / (2.0 * rho * speed)
                  * _ricker(np.asarray(times) - distance / speed))
    return trace


def _check_monoclinic(results_by_name):
    """Ring runs: per mode and branch max(|f / f_discrete - 1|, 1/Q) / RING_DISCRETE_TOLERANCE (n = 1 also |f /
    f_Christoffel - 1| / RING_CONTINUUM_TOLERANCE), |dt / dt_expected - 1| / RING_DT_TOLERANCE; launch runs max |v -
    exact| / max |exact| / LAUNCH_TOLERANCE; error = the worst (tolerance 1)."""
    errors, details = {}, {}
    expected_dt = _monoclinic_time_step()
    for axis in MONOCLINIC_RUNS:
        table, _, nearest = _resonance_table(results_by_name[f'{axis}_ring'])
        dt = float(table['dt'])
        errors[f'{axis}_ring_dt'] = abs(dt / expected_dt - 1.0) / RING_DT_TOLERANCE
        details[f'{axis}_ring_dt'] = {'dt': dt, 'expected_dt': expected_dt}
        for mode in RING_MODES:
            discrete, continuum = _ring_frequencies(axis, mode, dt)
            for branch, expected, christoffel in zip(('qS', 'qP'), discrete, continuum):
                label = f'{axis}_{branch}{mode}'
                found = nearest(float(expected))
                if found is None:
                    errors[label] = math.inf
                    continue
                measured, inverse_q = found
                discrete_error = max(abs(measured / expected - 1.0), inverse_q)
                continuum_error = abs(measured / christoffel - 1.0)
                errors[label] = discrete_error / RING_DISCRETE_TOLERANCE
                if mode == 1:
                    errors[label] = max(errors[label], continuum_error / RING_CONTINUUM_TOLERANCE)
                details[label] = {'frequency': measured, 'discrete_frequency': float(expected),
                                  'christoffel_frequency': float(christoffel), 'inverse_Q': inverse_q,
                                  'discrete_error': discrete_error, 'continuum_error': continuum_error}
        times, trace = _probe(results_by_name[f'{axis}_launch'], 'probe')
        exact = _launch_exact(axis, times)
        scale = float(np.max(np.abs(exact))) if exact.size else 0.0
        launch_error = float(np.max(np.abs(trace - exact))) / scale if scale > 0.0 and trace.size > 10 else math.inf
        errors[f'{axis}_launch'] = launch_error / LAUNCH_TOLERANCE
        details[f'{axis}_launch'] = {'relative_error': launch_error, 'peak_m_per_s': _signed_peak(trace),
                                     'expected_peak_m_per_s': _signed_peak(exact), 'n_samples': int(trace.size)}
    worst = max(errors, key=errors.get)
    first = details.get('x_qP1') or {}
    return {'measured': first.get('frequency', math.nan), 'expected': first.get('christoffel_frequency', math.nan),
            'error': errors[worst] if math.isfinite(errors[worst]) else math.inf, 'tolerance': 1.0,
            'details': {'errors': errors, 'worst': worst, 'runs': details}}


_STRIP_TEXT = (f'a strip periodic along y ({STRIP_CELLS_Y} cells, so every wave stays plane) with absorbing x faces '
               '(CPML of 20 cells inside the domain)')
# ELASTIC_CASES: every case (selftests_backend wraps them in its cpu-vs-gpu parity cases); SELFTEST_CASES registers
# only those no example checks: the reflection is examples/acoustics/acoustic_impedance_step_sweep, the thermoelastic
# slab photoacoustic_beer_lambert + axisymmetric_heated_core_hoop_stress, the plate resonances
# plate_thickness_resonances + rigid_duct_modes, the uniform hoop stress axisymmetric_heated_core_hoop_stress.
_COVERED_BY_EXAMPLES = frozenset({'acoustic_normal_reflection', 'acoustic_thermoelastic_slab',
                                  'acoustic_plate_resonances', 'elastic_axisymmetric_hoop_stress'})
ELASTIC_CASES = (
    SelfTestCase(
        name='elastic_wave_speeds', label='Elastic waves: plane P and S wave speed and amplitude',
        description=(f"Two runs (p_wave, s_wave) in {_STRIP_TEXT} of an isotropic solid (rho {SOLID['rho']:g} kg/m^3, sound_speed "
                     f"{SOLID['sound_speed']:g}, shear_speed {SOLID['shear_speed']:g} m/s, uniform "
                     f'{SPEED_CELL * 1e6:g} um cells): a traction_line_source across the strip at x = '
                     f'{SHEET_X * 1e3:g} mm (Ricker {RICKER_FREQUENCY:g} Hz, traction {SHEET_TRACTION:g} Pa) normal '
                     '(run p_wave, probes record vx) or tangential (run s_wave, vy) to the line; run p_wave_y repeats '
                     'p_wave rotated (a strip along y, periodic x faces, absorbing y faces, normal traction on a '
                     f"horizontal sheet at y = {SHEET_X * 1e3:g} mm, probes record vy) in an orthotropic crystal (rho "
                     f"{ORTHOTROPIC['rho']:g}, c11 {ORTHOTROPIC['c11']:g}, c12 {ORTHOTROPIC['c12']:g}, c22 "
                     f"{ORTHOTROPIC['c22']:g}, c66 {ORTHOTROPIC['c66']:g} Pa). Exact: the waves travel at sqrt(c11 / "
                     'rho), sqrt(c66 / rho) and sqrt(c22 / rho) with peak velocity T / (2 rho c) (the sheet sends half '
                     'the traction each way). Measured: the speed from the cross-correlation delay between the probes '
                     f'at {NEAR_X * 1e3:g} and {FAR_X * 1e3:g} mm along the strip, and the near peak velocity, each run '
                     'ending once its pulse has passed the far probe. error = the largest of the six relative errors; '
                     'numerical dispersion at >= 15 cells per S wavelength (up to 2.5x the Ricker peak) stays below '
                     f'~5e-3 (O((k dx)^2); 1.3e-3 at half the cell size); tolerance {SPEED_TOLERANCE:g}. Pins c11, c22 '
                     '(a swapped c11 / c22 is 29 % off), c66, the d/dx and d/dy stencils, the x and y CPML ends, the '
                     'face density and the sheet weights. Cell size and durations not yet confirmed by a run.'),
        specs=_wave_speed_runs, check=_check_wave_speeds),
    SelfTestCase(
        name='acoustic_normal_reflection', label='Acoustic waves: normal reflection at a fluid-fluid interface',
        description=(f"Plane pressure pulse (normal traction sheet, Ricker {RICKER_FREQUENCY:g} Hz) in {_STRIP_TEXT}: "
                     f"fluid 1 (rho {FLUID_1['rho']:g}, sound_speed {FLUID_1['sound_speed']:g}) meets fluid 2 (rho "
                     f"{FLUID_2['rho']:g}, sound_speed {FLUID_2['sound_speed']:g}) at x = {INTERFACE_X * 1e3:g} mm "
                     f'(a cell edge; {REFLECTION_CELL * 1e6:g} um cells). Exact: pressure reflection (Z2 - Z1) / (Z2 + '
                     'Z1) and transmission 2 Z2 / (Z2 + Z1), Z = rho c. Measured: the signed peak pressures of the '
                     'reflected and transmitted pulses over the incident one (probe windows split halfway between the '
                     'incident and reflected arrivals). The interface face takes the mean density of its two cells '
                     '(the exact mass of its dual cell), so the error is O((k dx)^2) (~1.5e-3 expected; 3.7e-4 at '
                     f'half the cell size); tolerance {REFLECTION_TOLERANCE:g} on both ratios. Layout (sheet '
                     f'{REFLECTION_SHEET_X * 1e3:g} mm, probes in the cells right of '
                     f'{(INCIDENT_X - 0.5 * REFLECTION_CELL) * 1e3:g} / {(TRANSMITTED_X - 0.5 * REFLECTION_CELL) * 1e3:g} mm, '
                     f'{REFLECTION_DURATION * 1e6:g} us) not yet confirmed by a run.'),
        specs=_reflection_runs, check=_check_normal_reflection),
    SelfTestCase(
        name='acoustic_thermoelastic_slab', label='Acoustic waves: thermoelastic pressure of a heated slab',
        description=(f'A heat_pulse_source Q = {HEATING_POWER:g} W/m^3 x exp(-((t - {PULSE_DELAY:g}) / '
                     f'{PULSE_WIDTH:g})^2) on the slab {SLAB_X0 * 1e3:g}..{SLAB_X1 * 1e3:g} mm of a fluid (rho '
                     f"{SLAB_FLUID['rho']:g}, sound_speed {SLAB_FLUID['sound_speed']:g}, gruneisen "
                     f"{SLAB_FLUID['gruneisen']:g}) in {_STRIP_TEXT}, {SLAB_CELL * 1e6:g} um cells. Exact (1D wave "
                     'equation with the source gruneisen dq/dt): right of the slab p = (G Q / 2) [F(t - (x - x1) / c) '
                     '- F(t - (x - x0) / c)], F the time integral of the envelope, with the plateau G H / 2, H = Q '
                     'sqrt(pi) width the absorbed energy density. error = max(max |p - p_exact| over the probe trace, '
                     f'|max p - G H / 2|) / (G H / 2); tolerance {SLAB_TOLERANCE:g} (dispersion of the ~12-cell '
                     f'rise). Pins the sign and size of the thermoelastic source and the probe times. Probe at '
                     f'{(SLAB_PROBE_X - 0.5 * SLAB_CELL) * 1e3:g} mm (cell right of it), {SLAB_DURATION * 1e6:g} us; '
                     'the 2 mm slab and this layout not yet confirmed by a run.'),
        specs=_thermoelastic_runs, check=_check_thermoelastic_slab),
    SelfTestCase(
        name='acoustic_plate_resonances', label='Elastic waves: resonance finder on a rigid-walled plate',
        description=(f"A {PLATE_CELLS[0]} x {PLATE_CELLS[1]}-cell ({PLATE_CELL * 1e3:g} mm) isotropic solid (rho "
                     f"{PLATE_SOLID['rho']:g}, sound_speed {PLATE_SOLID['sound_speed']:g}, shear_speed "
                     f"{PLATE_SOLID['shear_speed']:g} m/s, lossless) with rigid (slip) walls on all four grid edges "
                     f'rings after a point force (Ricker {PLATE_RICKER_FREQUENCY:g} Hz, 30 deg) for '
                     f'{PLATE_DURATION * 1e6:g} us at courant {PLATE_COURANT:g} (not yet confirmed by a run); a '
                     'resonance_monitor on vx (frequency band '
                     f'{PLATE_BAND[0]:g}-{PLATE_BAND[1]:g} Hz) finds S(1,1) ~633 kHz, P(1,0) ~802 kHz and S(2,1) ~913 '
                     'kHz. Exact: slip walls separate P modes (potential cos cos) and S modes (sin sin) with '
                     'k = (p pi / Lx, q pi / Ly), L the grid size; the staggered leap-frog gives sin(omega dt / 2) = '
                     '(dt / 2) c K, K^2 = sum (2/dx)^2 sin^2(p pi / (2 n)), c = sound_speed or shear_speed, Q = inf. '
                     'error = the worst of |f / f_exact - 1| and 1/Q = 2 |frequency_imag| / f, tolerance '
                     f'{PLATE_TOLERANCE:g}: float32 coefficients and fit noise leave ~1e-6; a wall half a cell off '
                     'moves f by ~1 %, a P / S mix-up by ~2x. Pins the frequency band and the acoustic dt of the '
                     'shared resonance finder.'),
        specs=_plate_runs, check=_check_plate_resonances),
    SelfTestCase(
        name='elastic_axisymmetric_cylinder', label='Elastic waves: axisymmetric (r, z) modes of a solid cylinder',
        description=(f'geometry axisymmetric: the solid of acoustic_plate_resonances fills r < a = '
                     f'{CYLINDER_CELLS * CYLINDER_CELL * 1e3:g} mm ({CYLINDER_CELLS} uniform '
                     f'{CYLINDER_CELL * 1e3:g} mm cells, x_lo the axis, a rigid (slip) wall at r = a) in a z strip of '
                     f'{STRIP_CELLS_Y} periodic cells; a vertical traction_line_source through the strip (a ring '
                     'sheet, Ricker pulse) excites only z-uniform modes, resonance_monitors fit the ring-down over '
                     f'{CYLINDER_DURATION * 1e6:g} us. Run radial: normal traction, the breathing modes u_r ~ J1(k r) '
                     '(hoop stress, c12 hoop coupling, (1/r) d(r s_rr)/dr - s_tt / r) on two monitors in one probe '
                     'cell, vx (modes radial) and the stt component (modes hoop: the same modes in the hoop stress); '
                     'run axial: tangential traction, vy probe, the axial shear modes u_z ~ J0(k r) ((1/r) d(r '
                     's_rz)/dr). Exact: J1(k a) = '
                     '0 (k a = 3.8317, 7.0156), f = c k / (2 pi) with c = sound_speed / shear_speed; and the exact '
                     'discrete modes of the scheme (the ring-volume generalised eigenproblem in numpy, leap-frog '
                     'sin(omega dt / 2) = dt omega_c / 2). error = the worst over the 2 + 2 + 2 modes of max(|f / '
                     f'f_discrete - 1|, 1/Q) / {CYLINDER_DISCRETE_TOLERANCE:g} and |f / f_Bessel - 1| / '
                     f'{CYLINDER_CONTINUUM_TOLERANCE:g}, tolerance 1. A missing hoop term or a wrong axis weight moves '
                     'the modes by percents, a non-adjoint pair grows (1/Q). The merged hoop monitor and the '
                     'duration not yet confirmed by a run.'),
        specs=_cylinder_runs, check=_check_cylinder_modes),
    SelfTestCase(
        name='elastic_axisymmetric_hoop_stress', label='Elastic waves: hoop stress of a uniformly heated cylinder',
        description=(f'geometry axisymmetric: the solid of acoustic_plate_resonances with gruneisen {HOOP_GRUENEISEN:g} '
                     f'fills r < a = {HOOP_CELLS} cells of {CYLINDER_CELL * 1e3:g} mm (rigid slip wall at r = a, '
                     f'{STRIP_CELLS_Y}-cell periodic z strip); a heat_pulse_source on every cell (q0 '
                     f'{HOOP_POWER:g} W/m^3, width {HOOP_PULSE_WIDTH:g} s, delay {HOOP_PULSE_DELAY:g} s) for '
                     f'{HOOP_DURATION * 1e6:g} us, probes on stt and pressure in cell {HOOP_PROBE_CELL}, the stt '
                     'field stored. Exact: a uniform stress is an equilibrium of the ring-volume forms ((1/r) d(r '
                     's_rr)/dr - s_tt / r = 0 discretely), so nothing moves and s_tt = s_rr = s_zz = -G H, pressure '
                     '= G H, H = q0 w sqrt(pi) / 2 (erf((t - d) / w) + erf(d / w)) at the last sample. error = the '
                     'worst of max |stt + G H| over the final stt field, |stt + G H| and |p - G H| of the last probe '
                     f'samples, over G H; tolerance {HOOP_TOLERANCE:g}. Pins the stt output field and probe component '
                     '(sign, value, cell), the hoop part of the thermoelastic source and the axisymmetric pressure.'),
        specs=_hoop_runs, check=_check_hoop_stress),
    SelfTestCase(
        name='elastic_waves_half_storage', label='Elastic waves: half-precision field storage',
        description=('Run p_wave of elastic_wave_speeds (traction sheet, vx probes: velocity scatter, velocity '
                     'scale) and the run of acoustic_thermoelastic_slab (heat_pulse_source, pressure probe: stress '
                     'scale) with an explicit_kernels stepper node at precision half_storage (fp16 fields, stress '
                     'stored / a power of two from the peak source stress, velocity / that over the power of two of '
                     'the largest impedance). error = the worst of each run\'s error over '
                     f'{HALF_STORAGE_FACTOR:g} x its single-precision tolerance ({SPEED_TOLERANCE:g}, '
                     f'{SLAB_TOLERANCE:g}: the fp16 noise floor is ~1e-3), tolerance 1; raises unless summary '
                     "'solvers' records the stepper precision half_storage."),
        specs=_half_storage_runs, check=_check_half_storage),
    SelfTestCase(
        name='elastic_monoclinic_plane_waves', label='Elastic waves: monoclinic c16 / c26 plane waves',
        description=(f"A monoclinic solid (rho {MONOCLINIC_SOLID['rho']:g}, c11 {MONOCLINIC_SOLID['c11']:g}, c12 "
                     f"{MONOCLINIC_SOLID['c12']:g}, c22 {MONOCLINIC_SOLID['c22']:g}, c66 {MONOCLINIC_SOLID['c66']:g}, "
                     f"c16 {MONOCLINIC_SOLID['c16']:g}, c26 {MONOCLINIC_SOLID['c26']:g} Pa). Runs x_ring / y_ring: "
                     f'fully periodic boxes of {RING_CELLS} x {STRIP_CELLS_Y} ({STRIP_CELLS_Y} x {RING_CELLS}) cells of '
                     f'{RING_CELL * 1e6:g} um at courant {RING_COURANT:g}, x and y Ricker sheets '
                     f'({RING_RICKER_FREQUENCY:g} Hz) across, a resonance_monitor on vx (vy) for '
                     f'{RING_DURATION * 1e6:g} us; plane modes k = 2 pi n / L, n = 1, 2, of both branches. Exact: the '
                     'discrete symbol (2 / h)^2 sin^2(k h / 2) [[c11, c16 a], [c16 a, c66]] (y: [[c66, c26 a], [c26 a, '
                     'c22]]), a = cos(k h / 2) from the corner / centre averages of the shear coupling, leap-frog '
                     'sin(omega dt / 2) = dt omega_c / 2, Q = inf (the averages are an adjoint pair); n = 1 against '
                     'the continuum Christoffel speeds f = n c / L; dt = courant h / (sqrt(S / rho) sqrt(2)), S the '
                     'stability bound with |c16|, |c26|. Runs x_launch / y_launch: strips of '
                     f'{LAUNCH_CELL * 1e6:g} um cells (absorbing ends) with one tangential Ricker sheet '
                     f'({RICKER_FREQUENCY:g} Hz; x: force y, probe vx; y: force x, probe vy) '
                     f'{LAUNCH_PROBE_CELL - LAUNCH_SHEET_CELL} cells from the probe: the cross-polarised trace sum_m e_m,p (e_m . f) T s(t - d / c_m) / (2 rho '
                     'c_m) exists only through c16 / c26 and follows their sign (the ring spectrum sees only their '
                     'squares). error = the worst of max(|f / f_discrete - 1|, 1/Q) / '
                     f'{RING_DISCRETE_TOLERANCE:g}, |f / f_Christoffel - 1| / {RING_CONTINUUM_TOLERANCE:g} (n = 1), '
                     f'|dt / dt_expected - 1| / {RING_DT_TOLERANCE:g} and max |v - exact| / max |exact| / '
                     f'{LAUNCH_TOLERANCE:g}; tolerance 1. Pins the c16 / c26 stress terms (size, sign, averaging), '
                     'their energy conservation, stability at courant 0.99 and the dt bound. Ring duration, launch '
                     f'distance and {LAUNCH_DURATION * 1e6:g} us not yet confirmed by a run.'),
        specs=_monoclinic_runs, check=_check_monoclinic),
)
SELFTEST_CASES = tuple(case for case in ELASTIC_CASES if case.name not in _COVERED_BY_EXAMPLES)
