"""Self-tests of pressure_acoustics (frequency domain) and its 'rigid_disk_scattering' template: plane waves in a duct on
triangles and tetrahedra (impedance, PML, sound-soft and vibrating ends) against the closed forms, the scattered-field
background in an empty axisymmetric domain and its total-field twin against 2 p0 cos(k z), and the thin rigid disk
<Delta p> / p0 -> (8 / 3 pi) k a. CPU backend, each case repeated on the gpu (variant 'gpu': back_end auto = cuDSS
mixed, B1c); tolerances are first estimates (not yet run). pressure_acoustics_host_direct: the PML duct on the MUMPS /
PARDISO back ends against SuperLU (skipped without python-mumps / pydiso)."""
import math

import numpy as np

from ...model.schema import SelfTestCase, SelfTestVariant
from ...model.selftest_helpers import (available_host_direct, normalized_result, skip_without_host_direct, solver_role,
                                       with_direct_node)
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           RefinementSpec, SolverDefaultsSpec, StudySpec)

TYPE_NAME = 'pressure_acoustics'   # = definition_harmonic.TYPE_NAME (selftests never import definitions)
PHYSICS = 'pressure_acoustics'
_STUDY = 'frequency_response'      # = model.study_frequency_response.STUDY_NAME
_AIR_DENSITY = 1.2                 # kg/m^3
_AIR_SPEED = 343.0                 # m/s
_AIR = {'rho': _AIR_DENSITY, 'sound_speed': _AIR_SPEED}
# B1c: every case again on the gpu, where the 'linear' role's back_end auto resolves to cudss_mixed (with nvmath; else
# SuperLU): the same graded tolerance (checks return errors in tolerance units, tolerance 1)
_GPU_VARIANTS = (SelfTestVariant('gpu', tolerance=1.0, backend='gpu'),)


def _wavenumber(frequency):
    return 2.0 * math.pi * frequency / _AIR_SPEED


def _model(title, domain, geometry, features, mesh, settings, study, stored=('p_abs',)):
    """One pressure_acoustics physics in air on `mesh`, cpu backend, a frequency_response study."""
    return ModelSpec(
        title=title, domain=domain, geometry=list(geometry),
        materials=[MaterialSpec('air', properties=dict(_AIR), selection=['all'])],
        physics=[PhysicsSpec(PHYSICS, TYPE_NAME, settings=dict(settings), features=list(features))],
        meshes=[mesh], solver_defaults=SolverDefaultsSpec(backend='cpu'),
        study=StudySpec(type=_STUDY, settings=dict(study), stored_fields=[f'{PHYSICS}.{name}' for name in stored]))


def _probes(points):
    """(point geometry, frequency_probe features) of (x, y[, z]) tuples."""
    geometry, features = [], []
    for index, point in enumerate(points):
        params = dict(zip(('x', 'y', 'z'), point))
        geometry.append(GeometrySpec(f'p{index}', 'point', params))
        features.append(FeatureSpec(f'probe{index}', 'frequency_probe', [f'p{index}']))
    return geometry, features


def _probe_values(results, count, component='p', frequency_index=0):
    """Complex probe values probe0 .. probe<count - 1> of one frequency."""
    values = []
    for index in range(count):
        series = results.result(PHYSICS, f'probe{index}')['series']
        values.append(complex(series[f'{component}_re'][frequency_index], series[f'{component}_im'][frequency_index]))
    return np.array(values)


def _graded(errors, tolerances, details):
    """measured / error = the worst error in tolerance units (tolerance 1), recording errors and tolerances."""
    return normalized_result({item: errors[item] / tolerances[item] for item in errors}, details, errors=errors,
                             tolerances=tolerances)


# ---- 1. / 2. plane-wave duct (triangles, tetrahedra) ----
# x in [0, L], cross-section W (x W in 3D), sound-hard side walls: a 1D field. Drive on x_lo: prescribed pressure p0
# (runs impedance / pml / soft) or an inward normal velocity v0 (run velocity); x_hi: impedance rho c (exact
# absorber at normal incidence: p0 exp(-j k x)), a PML (two-wave fit: |backward / forward|), sound soft (p0 sin(k (L -
# x)) / sin(k L)). P2 at ~23 elements per wavelength: dispersion ~(k h)^4 / 1440 ~ 4e-6 per radian.
_DUCT_LENGTH = 0.5           # m
_DUCT_WIDTH = 0.05           # m
_DUCT_FREQUENCY = 600.0      # Hz: lambda 0.572 m, k L = 5.50 (sin(k L) = -0.71: no soft-end resonance)
_DUCT_ELEMENT = 0.025        # m
_DUCT_PRESSURE = 1.0         # Pa
_DUCT_VELOCITY = 1e-3        # m/s
_DUCT_PROBE_X = tuple(0.05 * index for index in range(1, 10))   # m
_DUCT_PML_CELLS = 12         # x the element size: 0.3 m = 0.52 lambda
_DUCT_PML_LAYERS = 16        # structured ring layers (triangles)
_DUCT_PML_REFLECTION = 1e-5
_DUCT_RUNS = ('impedance', 'pml', 'soft', 'velocity')
_DUCT_FIELD_TOLERANCE = 1e-3       # relative to the peak of the closed form
_DUCT_REFLECTION_TOLERANCE = 1e-3  # |backward / forward| of the PML run (triangles: 16 ring layers)
_DUCT_REFLECTION_TOLERANCE_3D = 3e-3   # ... tetrahedra (12 Kuhn cells across the shell: a coarser profile)
_DUCT_FORWARD_TOLERANCE = 2e-3     # |forward / p0 - 1| of the PML run (x 3 on tetrahedra)


def _duct_spec(run, three_d):
    settings = {'discretization': 'quadratic'}
    if run == 'velocity':
        features = [FeatureSpec('drive', 'vibrating_boundary', ['x_lo'], {'quantity': 'velocity',
                                                                          'velocity': _DUCT_VELOCITY})]
    else:
        features = [FeatureSpec('drive', 'prescribed_pressure', ['x_lo'], {'pressure': _DUCT_PRESSURE})]
    if run in ('impedance', 'velocity'):
        features.append(FeatureSpec('end', 'impedance', ['x_hi']))
    elif run == 'soft':
        features.append(FeatureSpec('end', 'sound_soft', ['x_hi']))
    else:
        settings.update(pml_x_hi=True, pml_cells=_DUCT_PML_CELLS, pml_reflection=_DUCT_PML_REFLECTION)
    middle = 0.5 * _DUCT_WIDTH
    geometry, probes = _probes([(x, middle, middle) if three_d else (x, middle) for x in _DUCT_PROBE_X])
    if three_d:
        mesh = MeshSpec(name='fluid', kind='tetrahedra', mode='uniform', dx=_DUCT_ELEMENT, mesher='builtin')
    else:
        mesh = MeshSpec(name='fluid', kind='triangles', mode='uniform', dx=_DUCT_ELEMENT, pml_layers=_DUCT_PML_LAYERS)
    domain = DomainSpec(size_x=_DUCT_LENGTH, size_y=_DUCT_WIDTH, size_z=_DUCT_WIDTH if three_d else None)
    return _model(f"Self-test: plane-wave duct, {run} end ({'tetrahedra' if three_d else 'triangles'})", domain,
                  geometry, features + probes, mesh, settings, {'frequencies': [_DUCT_FREQUENCY]})


def _duct_specs(three_d):
    return {run: _duct_spec(run, three_d) for run in _DUCT_RUNS}


def _duct_expected(run, positions):
    k = _wavenumber(_DUCT_FREQUENCY)
    positions = np.asarray(positions, dtype=np.float64)
    if run == 'soft':
        return _DUCT_PRESSURE * np.sin(k * (_DUCT_LENGTH - positions)) / math.sin(k * _DUCT_LENGTH)
    amplitude = _AIR_DENSITY * _AIR_SPEED * _DUCT_VELOCITY if run == 'velocity' else _DUCT_PRESSURE
    return amplitude * np.exp(-1j * k * positions)


def _check_duct(results_by_name, three_d=False):
    errors, tolerances, details = {}, {}, {}
    positions = np.array(_DUCT_PROBE_X)
    k = _wavenumber(_DUCT_FREQUENCY)
    for run, results in results_by_name.items():
        measured = _probe_values(results, len(positions))
        if run == 'pml':
            basis = np.column_stack([np.exp(-1j * k * positions), np.exp(1j * k * positions)])
            (forward, backward), *_ = np.linalg.lstsq(basis, measured, rcond=None)
            errors['pml reflection'] = float(abs(backward / forward))
            tolerances['pml reflection'] = _DUCT_REFLECTION_TOLERANCE_3D if three_d else _DUCT_REFLECTION_TOLERANCE
            errors['pml forward amplitude'] = float(abs(forward / _DUCT_PRESSURE - 1.0))
            tolerances['pml forward amplitude'] = _DUCT_FORWARD_TOLERANCE * (3.0 if three_d else 1.0)
            details['pml'] = {'forward': [forward.real, forward.imag], 'backward': [backward.real, backward.imag]}
            continue
        expected = _duct_expected(run, positions)
        errors[f'{run} field'] = float(np.max(np.abs(measured - expected)) / np.max(np.abs(expected)))
        tolerances[f'{run} field'] = _DUCT_FIELD_TOLERANCE
        details[run] = {'measured_abs': np.abs(measured).tolist(), 'expected_abs': np.abs(expected).tolist()}
    return _graded(errors, tolerances, details)


_DUCT_DESCRIPTION = (
    f'frequency_response at {_DUCT_FREQUENCY:g} Hz of air (rho {_AIR_DENSITY:g}, c {_AIR_SPEED:g}) in a '
    f'{_DUCT_LENGTH:g} m duct of width {_DUCT_WIDTH:g} m (sound-hard sides), P2 at {_DUCT_ELEMENT:g} m; runs: '
    f'prescribed pressure {_DUCT_PRESSURE:g} Pa on x_lo with x_hi impedance rho c (p0 exp(-j k x)), sound soft (p0 sin(k '
    f'(L - x)) / sin(k L)) or a PML of {_DUCT_PML_CELLS} element sizes (R {_DUCT_PML_REFLECTION:g}); an inward normal '
    f'velocity {_DUCT_VELOCITY:g} m/s on x_lo with the impedance end (rho c v0 exp(-j k x)). Probes at x = 0.05 .. 0.45 '
    f'm: the field runs within {_DUCT_FIELD_TOLERANCE:g} of the peak; the PML run fitted to forward + backward waves, '
    f'|backward / forward| <= {_DUCT_REFLECTION_TOLERANCE:g} and |forward / p0 - 1| <= {_DUCT_FORWARD_TOLERANCE:g}. '
    'measured = error = the worst item in tolerance units, tolerance 1 (first estimates, not yet run).')


# ---- pressure_acoustics_host_direct: the 'pml' duct (complex symmetric) at two frequencies, a scipy_splu node (the
# reference) and one node per installed host direct back end (literals: the selftest rule); the second frequency
# refactorises on the kept analysis (fem.harmonic previous=) ----
_HOST_DIRECT_BACK_ENDS = ('mumps_mixed', 'pardiso')
_HOST_DIRECT_REFERENCE = 'scipy_splu'
_HOST_DIRECT_RUN = 'pml'
_HOST_DIRECT_FREQUENCIES = (_DUCT_FREQUENCY, 1.1 * _DUCT_FREQUENCY)
_HOST_DIRECT_TOLERANCE = 1e-8      # relative to max |SuperLU probe| (double solves; mixed: GMRES to 1e-12)


def _host_direct_specs():
    """The 'pml' duct at _HOST_DIRECT_FREQUENCIES per back end (reference first); {} without python-mumps / pydiso."""
    back_ends = available_host_direct(_HOST_DIRECT_BACK_ENDS)
    if not back_ends:
        return {}
    specs = {}
    for back_end in (_HOST_DIRECT_REFERENCE, *back_ends):
        spec = _duct_spec(_HOST_DIRECT_RUN, False)
        spec.study.settings['frequencies'] = list(_HOST_DIRECT_FREQUENCIES)
        specs[back_end] = with_direct_node(spec, back_end, physics_names=(PHYSICS,))
    return specs


def _check_host_direct(results_by_name):
    """Per back end and frequency: max |p - p_SuperLU| / max |p_SuperLU| over the probes; the linear record must show
    that back end with a symmetric (LDL^T) factor and a reused analysis (second frequency)."""
    if not results_by_name:
        return skip_without_host_direct(_HOST_DIRECT_BACK_ENDS)
    reference = results_by_name[_HOST_DIRECT_REFERENCE]
    count = len(_DUCT_PROBE_X)
    errors, tolerances, details = {}, {}, {}
    for back_end, results in results_by_name.items():
        if back_end == _HOST_DIRECT_REFERENCE:
            continue
        for index, frequency in enumerate(_HOST_DIRECT_FREQUENCIES):
            expected = _probe_values(reference, count, frequency_index=index)
            measured = _probe_values(results, count, frequency_index=index)
            item = f'{back_end} {frequency:g} Hz'
            errors[item] = float(np.max(np.abs(measured - expected)) / np.max(np.abs(expected)))
            tolerances[item] = _HOST_DIRECT_TOLERANCE
        counters = solver_role(results, PHYSICS).get('counters') or {}
        record_ok = (counters.get('back_end') == back_end
                     and int(counters.get('host_direct_symmetric_factors') or 0) > 0
                     and int(counters.get('host_direct_reused_analyses') or 0) > 0)
        errors[f'{back_end} record'] = 0.0 if record_ok else 1.0
        tolerances[f'{back_end} record'] = 0.5
        details[back_end] = {'counters': dict(counters)}
    return _graded(errors, tolerances, details)


# ---- 3. scattered-field background in an empty axisymmetric domain, and its total-field twin ----
# (r, z) in [0, 0.1] x [0, 0.2] m of air at 2 kHz, rigid floor y_lo. scattered: background standing wave 2 p0 cos(k z)
# (direction -y, rigid plane 0), PML on x_hi and y_hi: no contrast, no obstacle and a zero floor flux, so p_s = 0 and the
# probes read the exact p_b. total: no background, no PML, prescribed 2 p0 cos(k Z) on y_hi, sound-hard x_hi: the 1D
# standing wave on P2 at ~34 elements per wavelength.
_EMPTY_RADIUS = 0.1          # m
_EMPTY_HEIGHT = 0.2          # m
_EMPTY_FREQUENCY = 2000.0    # Hz: lambda 0.1715 m, k Z = 7.33
_EMPTY_ELEMENT = 0.005       # m
_EMPTY_AMPLITUDE = 1.0       # Pa (p0)
_EMPTY_PROBES = tuple((r, z) for r in (0.02, 0.05, 0.08) for z in (0.03, 0.09, 0.15))
_EMPTY_TOLERANCE = 1e-4      # relative to 2 p0
_EMPTY_SCATTERED_TOLERANCE = 1e-10   # |p_s| / 2 p0 of the scattered run (exactly 0: no source)


def _empty_spec(run):
    settings = {'geometry': 'axisymmetric', 'discretization': 'quadratic'}
    geometry, features = _probes(_EMPTY_PROBES)
    if run == 'scattered':
        settings.update(pml_x_hi=True, pml_y_hi=True)
        features.append(FeatureSpec('incident', 'background_pressure_field', [], {
            'wave': 'standing_wave', 'amplitude': _EMPTY_AMPLITUDE, 'direction': '-y', 'plane_position': 0.0,
            'background_density': _AIR_DENSITY, 'background_sound_speed': _AIR_SPEED}))
    else:
        top = 2.0 * _EMPTY_AMPLITUDE * math.cos(_wavenumber(_EMPTY_FREQUENCY) * _EMPTY_HEIGHT)
        features.append(FeatureSpec('top', 'prescribed_pressure', ['y_hi'], {'pressure': top}))
    mesh = MeshSpec(name='fluid', kind='triangles', mode='uniform', dx=_EMPTY_ELEMENT)
    return _model(f'Self-test: empty axisymmetric domain, {run} field (pressure_acoustics)',
                  DomainSpec(size_x=_EMPTY_RADIUS, size_y=_EMPTY_HEIGHT), geometry, features, mesh, settings,
                  {'frequencies': [_EMPTY_FREQUENCY]})


def _empty_specs():
    return {'scattered': _empty_spec('scattered'), 'total': _empty_spec('total')}


def _check_empty(results_by_name):
    heights = np.array([z for _, z in _EMPTY_PROBES])
    expected = 2.0 * _EMPTY_AMPLITUDE * np.cos(_wavenumber(_EMPTY_FREQUENCY) * heights)
    scale = 2.0 * _EMPTY_AMPLITUDE
    errors, tolerances, details = {}, {}, {}
    for run, results in results_by_name.items():
        measured = _probe_values(results, len(_EMPTY_PROBES))
        errors[f'{run} p'] = float(np.max(np.abs(measured - expected)) / scale)
        tolerances[f'{run} p'] = _EMPTY_TOLERANCE
        if run == 'scattered':
            scattered = _probe_values(results, len(_EMPTY_PROBES), 'p_s')
            errors['scattered p_s'] = float(np.max(np.abs(scattered)) / scale)
            tolerances['scattered p_s'] = _EMPTY_SCATTERED_TOLERANCE
        details[run] = {'measured_re': measured.real.tolist(), 'expected': expected.tolist()}
    return _graded(errors, tolerances, details)


_EMPTY_DESCRIPTION = (
    f'Axisymmetric (r, z) air domain {_EMPTY_RADIUS:g} x {_EMPTY_HEIGHT:g} m at {_EMPTY_FREQUENCY:g} Hz, rigid floor '
    f'y_lo, P2 at {_EMPTY_ELEMENT:g} m (new_modules_plan S5, Basiri E2 check i). scattered: background standing wave 2 '
    'p0 cos(k z) (p0 1 Pa, direction -y, rigid plane 0) with PMLs on x_hi and y_hi: no contrast, no obstacle and a zero '
    'floor flux, so the scattered field must vanish (probes p_s / 2 p0 <= '
    f'{_EMPTY_SCATTERED_TOLERANCE:g}: no spurious PML-interface or face source) and the total probes read 2 p0 cos(k z); '
    'total: no background, 2 p0 cos(k Z) held on y_hi, sound-hard x_hi: the total-field P2 solution against 2 p0 cos(k '
    f'z). Nine probes each within {_EMPTY_TOLERANCE:g} of 2 p0. measured = error = the worst item in tolerance units, '
    'tolerance 1 (first estimates, not yet run).')


# ---- 4. thin rigid disk in a plane wave (axisymmetric) ----
# Disk r < a, thickness t << a at mid-height, a plane wave p0 exp(-j k z) travelling -z, PMLs on x_hi, y_lo, y_hi
# (free space, no floor). Low ka: the incompressible near field of a broadside disk, Delta p(r) = (4 / pi) j k p0
# sqrt(a^2 - r^2), so <Delta p> / p0 = (8 / 3 pi) k a (Basiri E2 check ii) up to O((k a)^2) and the thickness
# correction ~(t / a) ln(a / t) of the square rim (+ the incident gradient across t, (3 pi / 8) t / a): t / a = 0.01
# (the template) measured +3.1 % / +3.3 % at k a 0.05 / 0.1, so the self-test disk is 10x thinner (~0.4 % expected).
_DISK_RADIUS = 1e-3          # m
_DISK_THICKNESS = 1e-5       # m (template)
_DISK_TEST_THICKNESS = 1e-6  # m (self-test)
_DISK_DOMAIN = (0.02, 0.04)  # m: (r_max, z_max), 20 a / 40 a
_DISK_ELEMENT = 4e-3         # m: h_max (lambda / 15.7 at the highest frequency); along the disk: t, h_min t / 2
_DISK_KA = (0.05, 0.1)
_DISK_PML_CELLS = 6
_DISK_PML_REFLECTION = 1e-3
_DISK_TOLERANCE = 0.015      # relative <Delta p> / p0 against (8 / 3 pi) k a
_DISK_AMPLITUDE = 1.0        # Pa


def _disk_frequencies(products=_DISK_KA):
    return [ka * _AIR_SPEED / (2.0 * math.pi * _DISK_RADIUS) for ka in products]


def _disk_model(title, study, stored, thickness=_DISK_THICKNESS):
    """The axisymmetric thin-disk scattering model (selftest case and template), refined to `thickness` on the disk."""
    centre = 0.5 * _DISK_DOMAIN[1]
    disk = GeometrySpec('disk', 'rectangle', {'x0': 0.0, 'y0': centre - 0.5 * thickness, 'x1': _DISK_RADIUS,
                                              'y1': centre + 0.5 * thickness})
    features = [
        FeatureSpec('incident', 'background_pressure_field', [], {
            'wave': 'plane_wave', 'amplitude': _DISK_AMPLITUDE, 'direction': '-y',
            'background_density': _AIR_DENSITY, 'background_sound_speed': _AIR_SPEED}),
        FeatureSpec('rigid', 'rigid_obstacle', ['disk']),
        FeatureSpec('top', 'surface_pressure', ['disk'], {'side': '+y'}),
        FeatureSpec('bottom', 'surface_pressure', ['disk'], {'side': '-y'})]
    settings = {'geometry': 'axisymmetric', 'discretization': 'quadratic', 'pml_x_hi': True, 'pml_y_lo': True,
                'pml_y_hi': True, 'pml_cells': _DISK_PML_CELLS, 'pml_reflection': _DISK_PML_REFLECTION}
    mesh = MeshSpec(name='fluid', kind='triangles', mode='nonuniform', dx=_DISK_ELEMENT, h_min=0.5 * thickness,
                    refinements=[RefinementSpec('disk', thickness)])
    return _model(title, DomainSpec(size_x=_DISK_DOMAIN[0], size_y=_DISK_DOMAIN[1]), [disk], features, mesh, settings,
                  study, stored)


def _disk_specs():
    return {'run': _disk_model('Self-test: thin rigid disk in a plane wave (pressure_acoustics)',
                               {'frequencies': _disk_frequencies()}, ('p_abs',), _DISK_TEST_THICKNESS)}


def _check_disk(results_by_name):
    results = results_by_name['run']
    top, bottom = (results.result(PHYSICS, name)['series'] for name in ('top', 'bottom'))
    errors, tolerances, details = {}, {}, {}
    for index, ka in enumerate(_DISK_KA):
        difference = (complex(top['p_mean_re'][index], top['p_mean_im'][index])
                      - complex(bottom['p_mean_re'][index], bottom['p_mean_im'][index]))
        measured = abs(difference) / _DISK_AMPLITUDE
        expected = 8.0 / (3.0 * math.pi) * ka
        errors[f'ka {ka:g}'] = abs(measured / expected - 1.0)
        tolerances[f'ka {ka:g}'] = _DISK_TOLERANCE
        details[f'ka {ka:g}'] = {'measured': measured, 'expected': expected}
    return _graded(errors, tolerances, details)


_DISK_DESCRIPTION = (
    f'Axisymmetric thin rigid disk (radius {_DISK_RADIUS:g} m, thickness {_DISK_TEST_THICKNESS:g} m, a '
    f'rigid_obstacle) at mid-height of a {_DISK_DOMAIN[0]:g} x {_DISK_DOMAIN[1]:g} m air domain with PMLs on x_hi, y_lo, y_hi (free '
    'space), background plane wave p0 exp(-j k z) travelling -z (scattered field: the obstacle source plus the PMLs); '
    'surface_pressure on the disk top (side +y) and bottom (-y): |<p_top> - <p_bottom>| / p0 against the low-frequency '
    f"broadside disk (8 / 3 pi) k a at k a = {', '.join(f'{ka:g}' for ka in _DISK_KA)} within {_DISK_TOLERANCE:g} "
    '(relative; O((k a)^2) and the ~(t / a) ln(a / t) thickness correction below 1 %: +3 % at t / a = 0.01; Basiri E2 '
    f'check ii). Mesh refined to the thickness along the disk, h_max {_DISK_ELEMENT:g} m. measured = error = the worst '
    'item in tolerance units, tolerance 1.')

# ---- template ----
RIGID_DISK_LABEL = 'Rigid disk scattering (pressure acoustics)'
RIGID_DISK_DESCRIPTION = (
    'Axisymmetric pressure_acoustics: a thin rigid disk in a plane wave (scattered-field background, PMLs), the mean '
    'pressure on its top and bottom faces from k a = 0.05 to 0.3 (log sweep); the low-frequency pressure difference is '
    '(8 / 3 pi) k a p0.')
RIGID_DISK_SUMMARY = ('A thin rigid disk hit by a sound wave: the pressure on its two faces and the sound it scatters, '
                      'over a range of frequencies.')


def rigid_disk_spec():
    """Template 'rigid_disk_scattering': the selftest disk on a 6-point log sweep of k a from 0.05 to 0.3."""
    low, high = _disk_frequencies((0.05, 0.3))
    return _disk_model('Rigid disk scattering', {'frequency_start': low, 'frequency_stop': high, 'n_frequencies': 6,
                                                 'log_spacing': True}, ('p_abs', 'p_s_abs', 'p_re'))


HARMONIC_SELFTEST_CASES = (
    SelfTestCase(
        name='pressure_acoustics_duct',
        label='Pressure acoustics: plane-wave duct on triangles (impedance, PML, soft and vibrating ends)',
        description=_DUCT_DESCRIPTION, specs=lambda: _duct_specs(False), check=_check_duct, variants=_GPU_VARIANTS),
    SelfTestCase(
        name='pressure_acoustics_host_direct',
        label='Pressure acoustics: PML duct on MUMPS / PARDISO linear solvers vs SuperLU',
        description=(f"The 'pml' run of pressure_acoustics_duct (triangles, cpu) at {_HOST_DIRECT_FREQUENCIES[0]:g} and "
                     f'{_HOST_DIRECT_FREQUENCIES[1]:g} Hz with a sparse_direct node pinned to scipy_splu (the '
                     'reference) and one per installed back end of mumps_mixed, pardiso: fem.harmonic factorises the '
                     'complex symmetric A(omega) by factorize_host(symmetric=True, previous=) as a library LDL^T '
                     '(mixed: complex64 factor + float64 GMRES refinement) and refactorises the second frequency on '
                     'the kept analysis. Per back end and frequency max |p - p_SuperLU| / max |p_SuperLU| over the '
                     f'probes within {_HOST_DIRECT_TOLERANCE:g}; the linear record must show that back end with a '
                     'symmetric factor and a reused analysis. measured = error = the worst item in tolerance units, '
                     'tolerance 1; passes as skipped without python-mumps / pydiso. Not run yet.'),
        specs=_host_direct_specs, check=_check_host_direct),
    SelfTestCase(
        name='pressure_acoustics_duct_3d',
        label='Pressure acoustics: plane-wave duct on tetrahedra (impedance, PML, soft and vibrating ends)',
        description=_DUCT_DESCRIPTION.replace('width', 'square cross-section side') + ' Tetrahedra: the builtin Kuhn '
                    f'split, a PML shell; PML tolerances {_DUCT_REFLECTION_TOLERANCE_3D:g} (reflection) and 3 x the '
                    'forward one.',
        specs=lambda: _duct_specs(True), check=lambda results_by_name: _check_duct(results_by_name, True),
        variants=_GPU_VARIANTS),
    SelfTestCase(
        name='pressure_acoustics_background_empty',
        label='Pressure acoustics: background standing wave in an empty axisymmetric domain',
        description=_EMPTY_DESCRIPTION, specs=_empty_specs, check=_check_empty, variants=_GPU_VARIANTS),
    SelfTestCase(
        name='pressure_acoustics_rigid_disk',
        label='Pressure acoustics: pressure difference across a thin rigid disk in a plane wave',
        description=_DISK_DESCRIPTION, specs=_disk_specs, check=_check_disk, heavy=True, variants=_GPU_VARIANTS),
)
