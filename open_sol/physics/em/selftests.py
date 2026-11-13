"""Analytic E_z-polarisation EM self-test cases (line-current Hankel field and far-field isotropy; the plane-wave,
Fresnel and Drude Mie cylinder cases come from selftests_hz's polarisation-generic factories), the wall-plane case of
both FDTD polarisations (wall helpers shared with selftests_fdfd) and the 'fresnel_interface' template; CPU only, every
spec is built fresh on each call."""
import math

import numpy as np

from ...core.constants import C0, MU0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import PERIODIC_Y, vertical_line, worst_item
# re-exported for the sibling em selftests modules (imported from .selftests)
from ...model.selftest_helpers import normalized_result, tolerance_units_result  # noqa: F401
from ...model.spec import DomainSpec, FeatureSpec, GeometrySpec, MeshSpec, ModelSpec, PhysicsSpec, StudySpec
from .selftests_hz import (cw_duration, fresnel_case, fresnel_device_spec, mie_drude_cylinder_case, plane_wave_case,
                           steady_peak)

_PHYSICS_NAME = 'em'
_PHYSICS_TYPE = 'em_fdtd_ez'
_PROBE_FEATURE = 'probe'

# em_line_current_hankel
_HANKEL_WAVELENGTH = 1e-6                 # m, vacuum
_HANKEL_CURRENT = 1e-3                    # A
_HANKEL_SOURCE_XY = (2.01e-6, 2.01e-6)    # m, a cell centre of the 20 nm mesh
_HANKEL_PROBE_XY = (3.01e-6, 2.01e-6)     # m, a cell centre 50 cells to the right
_HANKEL_RADIUS = math.dist(_HANKEL_SOURCE_XY, _HANKEL_PROBE_XY)   # 1.0e-6 m
_HANKEL_TOLERANCE = 0.03          # relative
_HANKEL_PERIODS = 24              # run length [periods]: window from 21, ramp residual exp(-20/3) after the transit

# em_line_current_hankel's far-field check: a far-field box (1.5, 2.5) um, cells [75, 126) on the 20 nm mesh,
# around the source cell 100; one DFT wavelength (the cw wavelength), 72 angles.
_FAR_FIELD_FEATURE = 'far_field'
_FAR_FIELD_BOX = (1.5e-6, 1.5e-6, 2.5e-6, 2.5e-6)   # m (x0, y0, x1, y1)
_FAR_FIELD_N_ANGLES = 72
_FAR_FIELD_ISOTROPY_TOLERANCE = 0.05   # max |P(phi) / mean P - 1|

# em_wall_reflection_phase (shared with selftests_fdfd): y-periodic strip 3 x 0.2 um, uniform 20 nm mesh (50 cells per
# wavelength), window cell k spanning [20 k, 20 (k + 1)] nm. A line source in column s radiates A to each side: ez E_z
# at the cell centre (s + 0.5) dx, hz E_y on its hi face (s + 1) dx. The -x wave reflects with r (E) off a wall plane
# x_w, so the probe 2 um downstream sees a travelling wave of amplitude A |1 + r e^{-2 j k d}|, d = x_s - x_w. The wall
# images are exact on the lattice; s puts d near 3 cells (r = -1) or near the quarter-wave node (r = +1), where half a
# cell of wall offset moves the amplitude by >= 11%.
_WALL_WAVELENGTH = 1e-6           # m, vacuum
_WALL_CELL = 20e-9                # m
_WALL_SIZE = (3e-6, 0.2e-6)       # m
_WALL_PROBE_XY = (2.01e-6, 0.11e-6)   # m, centre of cell (100, 5)
_WALL_AMPLITUDE = 1.0             # V/m radiated to each side of the line
_WALL_TOLERANCE = 0.02            # relative, worst run
_WALL_PERIODS = 22                # FDTD run length [periods]: window from 19, ramp residual exp(-17/3) after transit
_WALL_HZ_PHYSICS = 'em_hz'        # FDTD case: the em_fdtd_hz physics beside the em_fdtd_ez one ('em') of a wall run
_WALL_SOURCE_COLUMN = {-1: 3, 1: 10}          # by E reflection coefficient
_WALL_SOURCE_OFFSET = {'ez': 0.5, 'hz': 1.0}  # source plane from the lo edge of its column [cells]
_WALL_COMPONENT = {'ez': 'ez', 'hz': 'ey'}    # probed tangential E
_PEC_BLOCK_COLUMNS = 10           # pec_domain runs: cpml x_lo, PEC over window columns [0, 10), source in column 13
# (polarization, bc_x_lo) -> (wall plane [cells from the window lo edge], E reflection coefficient): E_z pec at the
# outermost centre, H_z pmc at the outermost centre, E_z pmc / H_z pec on its hi face, both mirrors on the window edge
# (their ghost cell is padded outside the window).
_WALL_PLANES = {('ez', 'pec'): (0.5, -1), ('ez', 'pmc'): (1.0, 1), ('ez', 'symmetric'): (0.0, 1),
                ('ez', 'antisymmetric'): (0.0, -1), ('hz', 'pec'): (1.0, -1), ('hz', 'pmc'): (0.5, 1),
                ('hz', 'symmetric'): (0.0, 1), ('hz', 'antisymmetric'): (0.0, -1)}
# pec_domain: E_z = 0 at the last PEC cell centre, H_z polarisation: E_y = 0 on its hi face
_PEC_DOMAIN_PLANES = {'ez': _PEC_BLOCK_COLUMNS - 0.5, 'hz': float(_PEC_BLOCK_COLUMNS)}
_WALL_RUNS = tuple((polarization, wall) for polarization in ('ez', 'hz')
                   for wall in ('pec', 'pmc', 'symmetric', 'antisymmetric', 'pec_domain'))


# ---- specs ----
def fresnel_interface_spec():
    """Template 'fresnel_interface': the E_z device run of em_fresnel_transmission (vacuum / glass half-space)."""
    return fresnel_device_spec(_PHYSICS_TYPE, 'ez')


def _hankel_specs():
    """'run': 4 x 4 um all-CPML box, cw filament current in its centre cell, E_z probe 1 um (50 cells) away and a
    far-field box around the source at the cw wavelength."""
    source_x, source_y = _HANKEL_SOURCE_XY
    probe_x, probe_y = _HANKEL_PROBE_XY
    x0, y0, x1, y1 = _FAR_FIELD_BOX
    spec = ModelSpec(
        title='Self-test: line-current Hankel field and far-field isotropy',
        domain=DomainSpec(size_x=4e-6, size_y=4e-6),
        geometry=[GeometrySpec('source_point', 'point', {'x': source_x, 'y': source_y}),
                  GeometrySpec('probe_point', 'point', {'x': probe_x, 'y': probe_y}),
                  GeometrySpec('far_field_box', 'rectangle', {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})],
        physics=[PhysicsSpec(_PHYSICS_NAME, _PHYSICS_TYPE, features=[
            FeatureSpec('source', 'point_source', ['source_point'],
                        {'waveform': 'cw', 'wavelength': _HANKEL_WAVELENGTH, 'current': _HANKEL_CURRENT}),
            FeatureSpec(_PROBE_FEATURE, 'field_probe', ['probe_point'], {'component': 'ez'}),
            FeatureSpec(_FAR_FIELD_FEATURE, 'far_field', ['far_field_box'],
                        {'wavelength_start': _HANKEL_WAVELENGTH, 'wavelength_stop': _HANKEL_WAVELENGTH,
                         'n_wavelengths': 1, 'n_angles': _FAR_FIELD_N_ANGLES})])],
        mesh=MeshSpec(mode='uniform', dx=20e-9),
        study=StudySpec(type='time_dependent',
                        settings={'duration': cw_duration(_HANKEL_WAVELENGTH, _HANKEL_PERIODS)}))
    return {'run': spec}


def wall_layout(polarization, wall):
    """(wall plane [cells from the window lo edge], E reflection coefficient, source column) of a wall run; wall is a
    bc_x_lo choice or 'pec_domain'."""
    if wall == 'pec_domain':
        return _PEC_DOMAIN_PLANES[polarization], -1, _PEC_BLOCK_COLUMNS + 3
    plane, reflection = _WALL_PLANES[(polarization, wall)]
    return plane, reflection, _WALL_SOURCE_COLUMN[reflection]


def wall_run_name(polarization, wall):
    return f'{polarization}_{wall}'


def wall_reflection_spec(physics_type, polarization, wall, source_values, extra_settings, study):
    """y-periodic strip with bc_x_lo = wall (or cpml and a pec_domain block over columns [0, 10)), a line source in
    the column of wall_layout and a probe of the tangential E 2 um downstream."""
    size_x, size_y = _WALL_SIZE
    _, _, column = wall_layout(polarization, wall)
    source_x = (column + 0.5) * _WALL_CELL
    probe_x, probe_y = _WALL_PROBE_XY
    settings = {**PERIODIC_Y, **extra_settings}
    geometry = [vertical_line('source_line', source_x, 0.0, size_y),
                GeometrySpec('probe_point', 'point', {'x': probe_x, 'y': probe_y})]
    features = [FeatureSpec('source', 'line_source', ['source_line'], dict(source_values)),
                FeatureSpec(_PROBE_FEATURE, 'field_probe', ['probe_point'],
                            {'component': _WALL_COMPONENT[polarization]})]
    if wall == 'pec_domain':
        geometry.append(GeometrySpec('pec_block', 'rectangle',
                                     {'x0': 0.0, 'y0': 0.0, 'x1': _PEC_BLOCK_COLUMNS * _WALL_CELL, 'y1': size_y}))
        features.append(FeatureSpec('pec', 'pec_domain', ['pec_block']))
    else:
        settings['bc_x_lo'] = wall
    return ModelSpec(
        title=f'Self-test: {physics_type} reflection off a {wall} wall',
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=geometry,
        physics=[PhysicsSpec(_PHYSICS_NAME, physics_type, settings=settings, features=features)],
        mesh=MeshSpec(mode='uniform', dx=_WALL_CELL),
        study=study)


def _wall_specs():
    """'<wall>' per wall of _WALL_RUNS: one model with the em_fdtd_ez physics 'em' and the em_fdtd_hz physics
    _WALL_HZ_PHYSICS (same wall, source column and probe point; independent fields), cw source, _WALL_PERIODS periods."""
    source_values = {'waveform': 'cw', 'wavelength': _WALL_WAVELENGTH, 'amplitude': _WALL_AMPLITUDE}
    specs = {}
    for wall in dict.fromkeys(wall for _, wall in _WALL_RUNS):
        study = StudySpec(type='time_dependent', settings={'duration': cw_duration(_WALL_WAVELENGTH, _WALL_PERIODS)})
        spec = wall_reflection_spec(_PHYSICS_TYPE, 'ez', wall, source_values, {}, study)
        hz_physics = wall_reflection_spec('em_fdtd_hz', 'hz', wall, source_values, {}, study).physics[0]
        if wall_layout('ez', wall)[2] != wall_layout('hz', wall)[2]:
            raise ValueError(f"wall '{wall}': the ez and hz runs need one source column to share a model")
        hz_physics.name = _WALL_HZ_PHYSICS
        spec.physics.append(hz_physics)
        spec.title = f'Self-test: em_fdtd_ez and em_fdtd_hz reflection off a {wall} wall'
        specs[wall] = spec
    return specs


# ---- case records (shared by the em selftests modules) ----
# ---- checks ----
def _hankel0_modulus(argument):
    """|H0^(2)(x)| at x = argument (scipy.special.hankel2)."""
    import scipy.special
    return float(abs(scipy.special.hankel2(0, argument)))


def _check_hankel_and_far_field(results_by_name):
    """Steady peak |E_z| at r vs omega mu0 I / 4 |H0^(2)(k r)| of a 2D line current and the far-field isotropy of the
    same run; error = max(hankel error / its tolerance, isotropy error / its tolerance), tolerance 1; measured /
    expected = the peak |E_z| and the Hankel field."""
    measured, window_start, window_samples = steady_peak(results_by_name['run'], _HANKEL_WAVELENGTH)
    omega = 2.0 * math.pi * C0 / _HANKEL_WAVELENGTH
    k_r = 2.0 * math.pi * _HANKEL_RADIUS / _HANKEL_WAVELENGTH
    hankel_modulus = _hankel0_modulus(k_r)
    expected = omega * MU0 * _HANKEL_CURRENT / 4.0 * hankel_modulus
    hankel_error = abs(measured - expected) / expected
    isotropy = _check_far_field_isotropy(results_by_name)
    errors = (hankel_error / _HANKEL_TOLERANCE, isotropy['error'] / _FAR_FIELD_ISOTROPY_TOLERANCE)
    error = max(errors) if all(math.isfinite(value) for value in errors) else math.nan
    return {'measured': measured, 'expected': expected, 'error': error, 'tolerance': 1.0,
            'details': {'unit': 'V/m', 'radius': _HANKEL_RADIUS, 'k_r': k_r, 'hankel_modulus': hankel_modulus,
                        'window_start': window_start, 'window_samples': window_samples,
                        'hankel_error': hankel_error, 'hankel_tolerance': _HANKEL_TOLERANCE,
                        'far_field_relative': isotropy['measured'], 'far_field_error': isotropy['error'],
                        'far_field_tolerance': _FAR_FIELD_ISOTROPY_TOLERANCE, 'far_field': isotropy['details']}}


def _check_far_field_isotropy(results_by_name):
    """Far-field power per angle over its mean vs 1 (a line current radiates isotropically); worst angle."""
    far_field = results_by_name['run'].result(_PHYSICS_NAME, _FAR_FIELD_FEATURE)
    angles = np.asarray(far_field['angles'], dtype=np.float64)
    power = np.asarray(far_field['power'], dtype=np.float64)
    if power.shape != (1, angles.size):
        raise ValueError(f'far field: expected one wavelength x {angles.size} angles, got shape {power.shape}')
    power = power[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        relative = power / float(np.mean(power))
    deviation = np.abs(relative - 1.0)
    worst = int(np.argmax(deviation))   # a NaN counts as the largest deviation
    return {'measured': float(relative[worst]), 'expected': 1.0, 'error': float(deviation[worst]),
            'tolerance': _FAR_FIELD_ISOTROPY_TOLERANCE,
            'details': {'worst_angle': float(angles[worst]), 'mean_power': float(np.mean(power)),
                        'unit': far_field.get('unit'), 'relative_min': float(np.min(relative)),
                        'relative_max': float(np.max(relative))}}


def wall_expected_amplitude(polarization, wall):
    """A |1 + r e^{-2 j k d}| downstream of the line source of a wall run (vacuum k, d = source plane - wall plane)."""
    plane, reflection, column = wall_layout(polarization, wall)
    distance = (column + _WALL_SOURCE_OFFSET[polarization] - plane) * _WALL_CELL
    round_trip_phase = 4.0 * math.pi * distance / _WALL_WAVELENGTH
    return _WALL_AMPLITUDE * abs(1.0 + reflection * complex(math.cos(round_trip_phase), -math.sin(round_trip_phase)))


def wall_reflection_result(measured_by_run):
    """Check dict of {wall run name: downstream |E|} vs wall_expected_amplitude; measured / expected / error of the
    worst run (relative, NaN counts as the worst)."""
    expected, errors = {}, {}
    for name, measured in measured_by_run.items():
        polarization, wall = name.split('_', 1)
        expected[name] = wall_expected_amplitude(polarization, wall)
        errors[name] = abs(measured / expected[name] - 1.0)
    worst, _ = worst_item(errors)
    return {'measured': measured_by_run[worst], 'expected': expected[worst], 'error': errors[worst],
            'tolerance': _WALL_TOLERANCE,
            'details': {'unit': 'V/m', 'worst_run': worst, 'measured_by_run': dict(measured_by_run),
                        'expected_by_run': expected, 'relative_errors': errors}}


def _check_wall_reflection(results_by_name):
    """Steady peak |E| downstream per polarisation and wall vs wall_expected_amplitude."""
    physics_names = {'ez': _PHYSICS_NAME, 'hz': _WALL_HZ_PHYSICS}
    return wall_reflection_result({wall_run_name(polarization, wall): steady_peak(
        results_by_name[wall], _WALL_WAVELENGTH, physics_names[polarization])[0] for polarization, wall in _WALL_RUNS})


WALL_REFLECTION_TEXT = ('A line source (vacuum wavelength 1 um, 1 V/m to each side) spans the period of a y-periodic '
                        'strip (20 nm mesh, 50 cells per wavelength) a few cells in front of a wall on its x_lo side; '
                        'the -x wave reflects with r = -1 (PEC-like) or +1 (PMC-like) off the wall plane x_w, so the '
                        'probe of the tangential E 2 um downstream sees a travelling wave of amplitude |1 + r e^{-2 j '
                        'k d}|, d = source plane - x_w (E_z at the line-cell centre, E_y on its hi face). Expected '
                        'planes: E_z pec at the outermost cell centre, H_z pmc at the outermost cell centre, E_z pmc and '
                        'H_z pec on the hi face of the outermost cell, both mirrors (symmetric = PMC, antisymmetric = '
                        'PEC) on the window edge (x = 0; their ghost cell is padded outside), a pec_domain block (cpml x_lo) at its last cell centre (E_z) or on its hi face (H_z). d '
                        'is ~3 cells for r = -1 and ~10 cells (near the quarter-wave node) for r = +1, so half a cell '
                        'of wall offset or a wrong parity moves the amplitude by >= 11%. error = the worst relative '
                        'error over the runs, tolerance 0.02: the wall images are exact on the lattice, so only the '
                        'source calibration (< 0.5%), CPML reflection (< 0.5%), the discrete wavenumber (< 0.1%) and '
                        'the peak sampling / collocation (~0.2%) remain.')


# Not registered on its own: graded by selftests_backend em_fdtd_cpu_gpu (its own check on both backends) and the
# example em/analytic_two_source_array (Hankel near field and far-field pattern).
LINE_CURRENT_HANKEL_CASE = SelfTestCase(
    name='em_line_current_hankel',
    label='EM line-current Hankel field and far-field isotropy',
    description='A cw filament current I = 1 mA (vacuum wavelength 1 um) at the centre of a 4 x 4 um CPML box '
                f'(20 nm mesh); the peak |E_z| over the last 3 of {_HANKEL_PERIODS} periods at r = 1 um must match '
                'the 2D line-current field omega mu0 I / 4 |H0^(2)(k r)| with k r = 2 pi (relative error, '
                'tolerance 0.03: discrete Green function and dispersion at 50 cells per wavelength ~1%, CPML '
                'corner reflections and the ramp residual exp(-20/3) ~ 1.3e-3 at the window start after the '
                'transit). The same run carries a far_field monitor on the box (1.5, 2.5) um around the source '
                'at the cw wavelength, 72 angles: a line current in vacuum radiates the same power in every '
                'direction, so the far-field power per angle over its mean must be 1 (the pattern is in a.u., so '
                'only its shape is checked; worst |P / mean - 1|, tolerance 0.05: E / H face averaging onto the '
                'contour (cos(k dx/2), 0.2%), the trapezoid contour, lattice anisotropy of the discrete dispersion '
                'and CPML corner reflections; a wrong sign or a missing equivalent current makes the pattern '
                'strongly directional). error = max(Hankel error / 0.03, isotropy error / 0.05), tolerance 1; '
                'measured / expected = the peak |E_z| and the Hankel field. |H0^(2)| comes from scipy.special. '
                f'{_HANKEL_PERIODS} periods not yet confirmed by a run.',
    specs=_hankel_specs,
    check=_check_hankel_and_far_field)


# The uniform-mesh E_z Fresnel case is the normal-incidence point (theta 0) of the example
# em/analytic_bloch_fresnel_brewster (T_s).
SELFTEST_CASES = (
    plane_wave_case(_PHYSICS_TYPE, 'ez'),
    fresnel_case(_PHYSICS_TYPE, 'ez', graded=True),
    SelfTestCase(
        name='em_wall_reflection_phase',
        label='EM wall planes and mirror parities (FDTD)',
        description=f'{WALL_REFLECTION_TEXT} Five runs, bc_x_lo pec, pmc, symmetric, antisymmetric and a pec_domain '
                    'block, each one model with an em_fdtd_ez and an em_fdtd_hz physics (same wall, source column and '
                    f'probe point; the fields are independent); peak |E| over the last 3 of {_WALL_PERIODS} periods '
                    '(ramp residual exp(-17/3) ~ 3.5e-3 at the window start after the ~2 period transit). Paired '
                    f'models and {_WALL_PERIODS} periods not yet confirmed by a run.',
        specs=_wall_specs,
        check=_check_wall_reflection),
    mie_drude_cylinder_case(_PHYSICS_TYPE, 'ez'),
)
