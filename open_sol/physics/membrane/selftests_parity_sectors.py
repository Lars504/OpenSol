"""Self-tests of membrane parity sectors (experiment plan F3): a driven half rectangle, the half tunnelling guide with two
incident modes of both parities, and the half plate rectangle's eigen sectors against the full models."""
import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import scaled_worst_result
from ...model.spec import (DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec,
                           PhysicsSpec, StudySpec)
from .selftests import (_FILM, _FREQUENCY_COLUMN, _MEMBRANE_TYPE, _MIRROR_MODES, _PHYSICS, _RECTANGLE, _SUPPORTED_MODES,
                        _SUPPORTED_TOLERANCE, _TABLE, _TRIANGLE_MESH, _TUNNEL_AMPLITUDE, _TUNNEL_H, _mirror_specs,
                        _probe_phasors, _solver_defaults, _supported_frequencies, _tunnel_spec)

# ---- membrane_parity_sectors_driven: the clamped template rectangle driven by a point force off the x mirror line at
# three frequencies between its resonances; full model on a mirror_x mesh vs the half x <= a / 2 (symmetry on x_hi)
# with parity_sectors on the same half mesh: the drive-point probe (combined sector result), the probe at a point of
# the other half (mean of the signed sector probes at its image), the |w|^2 integral (quadratic total) and the
# combined snapshot. Same mesh halves, so only the solver round-off differs (first-estimate tolerances). The two display
# rasters have different pixel grids (pixel = sqrt(area / max_pixels)); the point force's log spike at the drive node
# makes a raw raster peak depend on how near a pixel centre lands, so the snapshot is compared pixel-wise (full raster
# bilinear at the half's centres) outside _DRIVEN_SPIKE_RADIUS of the drive point and its image.
_DRIVEN_H = _RECTANGLE[0] / 60.0
_DRIVEN_FREQUENCIES = [4.0e5, 6.0e5, 7.8e5]   # Hz: below f_11 = 495 kHz, between f_11 and f_21, f_21 and f_12
_DRIVEN_FORCE = 1e-6                          # N
_DRIVEN_POINTS = {'drive': (0.3, 0.4), 'image': (0.2, 0.6), 'drive_image': (0.7, 0.4), 'far': (0.8, 0.6)}   # x / a, y / b
_DRIVEN_TOLERANCE = 1e-6
_DRIVEN_SNAPSHOT_TOLERANCE = 1e-3             # combined display raster vs the full one, max |difference| / max |full|
_DRIVEN_SPIKE_RADIUS = 4.0 * _DRIVEN_H        # pixels this close to the drive point or its image are not compared


def _driven_spec(half):
    """The rectangle driven at 'drive'; half: x <= a / 2, symmetry on x_hi, parity_sectors (probes at the drive point
    and at 'image', the mirror image of the full model's 'far' probe point)."""
    size_x, size_y = _RECTANGLE
    names = ('drive', 'image') if half else tuple(_DRIVEN_POINTS)
    geometry = [GeometrySpec(name, 'point', {'x': _DRIVEN_POINTS[name][0] * size_x,
                                             'y': _DRIVEN_POINTS[name][1] * size_y}) for name in names]
    features = [FeatureSpec('frame', 'clamped', ['x_lo', 'y_lo', 'y_hi'] + ([] if half else ['x_hi'])),
                FeatureSpec('drive_force', 'point_force', ['drive'], {'force': _DRIVEN_FORCE}),
                FeatureSpec('drive_probe', 'field_probe', ['drive']),
                FeatureSpec('image_probe', 'field_probe', ['image']) if half
                else FeatureSpec('far_probe', 'field_probe', ['far'])]
    if half:
        features.append(FeatureSpec('mirror', 'symmetry', ['x_hi']))
    return ModelSpec(
        title=f"Self-test: driven membrane rectangle ({'half, parity sectors' if half else 'full'})",
        domain=DomainSpec(size_x=0.5 * size_x if half else size_x, size_y=size_y), geometry=geometry,
        materials=[MaterialSpec('film', properties=dict(_FILM), selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS, _MEMBRANE_TYPE, features=features)],
        meshes=[MeshSpec(name=_TRIANGLE_MESH, kind='triangles', mode='uniform', dx=_DRIVEN_H, mirror_x=not half)],
        solver_defaults=_solver_defaults('cpu'),
        study=StudySpec(type='frequency_response', settings={'frequencies': list(_DRIVEN_FREQUENCIES),
                                                             'parity_sectors': half}),
        derived_values=[DerivedValueSpec('energy', 'w_re*w_re + w_im*w_im', ['all'], 'integral')])


def _driven_specs():
    return {'full': _driven_spec(False), 'half': _driven_spec(True)}


def _sector_sign(label) -> float:
    """-1 for a sector whose (one) mirror face is antisymmetric, else +1."""
    return -1.0 if 'antisymmetric' in label else 1.0


def _phasor(series, name, index=None):
    values = (np.asarray(series[f'{name}_re'], dtype=np.float64)
              + 1j * np.asarray(series[f'{name}_im'], dtype=np.float64))
    return values if index is None else complex(values[index])


def _pixel_centres(field):
    """(x (nx,), y (ny,)) pixel centres [m] of a FieldData raster: its x / y_centers, else uniform over its extent."""
    n_x, n_y = np.shape(field.values)
    x_min, x_max, y_min, y_max = field.extent
    x_centres = (np.asarray(field.x_centers, dtype=np.float64) if field.x_centers is not None
                 else x_min + (np.arange(n_x) + 0.5) * ((x_max - x_min) / n_x))
    y_centres = (np.asarray(field.y_centers, dtype=np.float64) if field.y_centers is not None
                 else y_min + (np.arange(n_y) + 0.5) * ((y_max - y_min) / n_y))
    return x_centres, y_centres


def _snapshot_field_error(field_full, field_half):
    """Max |half raster - full raster (bilinear at the half's pixel centres)| / max |full there| over the pixels farther
    than _DRIVEN_SPIKE_RADIUS from the drive point and its image (pixels outside the full raster's centres skipped)."""
    from scipy.interpolate import RegularGridInterpolator
    full_values = np.asarray(field_full.values, dtype=np.float64)
    half_values = np.asarray(field_half.values, dtype=np.float64)
    interpolate = RegularGridInterpolator(_pixel_centres(field_full), full_values, bounds_error=False,
                                          fill_value=np.nan)
    grid_x, grid_y = np.meshgrid(*_pixel_centres(field_half), indexing='ij')
    reference = interpolate(np.column_stack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
    size_x, size_y = _RECTANGLE
    compared = np.isfinite(reference) & np.isfinite(half_values)
    for name in ('drive', 'drive_image'):
        spike_x, spike_y = _DRIVEN_POINTS[name][0] * size_x, _DRIVEN_POINTS[name][1] * size_y
        compared &= np.hypot(grid_x - spike_x, grid_y - spike_y) > _DRIVEN_SPIKE_RADIUS
    if not compared.any():
        raise ValueError('no pixel of the half raster lies inside the full raster away from the drive spike')
    return float(np.max(np.abs(half_values[compared] - reference[compared])) / np.max(np.abs(reference[compared])))


def _check_driven(results_by_name):
    full, half = results_by_name['full'], results_by_name['half']
    _, full_drive = _probe_phasors(full, 'drive_probe')
    _, full_far = _probe_phasors(full, 'far_probe')
    half_drive = _phasor(half.sector_result(_PHYSICS, 'drive_probe', combined=True)['series'], 'w')
    labels = half.summary()['parity_sectors']['labels']
    sectors = half.sector_result(_PHYSICS, 'image_probe')
    half_far = np.mean([_sector_sign(label) * _phasor(result['series'], 'w')
                        for label, result in zip(labels, sectors)], axis=0)
    energy_full = float(full.summary()['derived']['energy'])
    energy_half = float(half.summary()['derived']['energy'])
    index = len(_DRIVEN_FREQUENCIES) - 1   # the combined snapshot of the last frequency (combined snapshots first)
    field_full = full.snapshot_field(index, 'w_re')
    field_half = half.snapshot_field(index, 'w_re')
    peak_full = float(np.nanmax(np.abs(np.asarray(field_full.values, dtype=np.float64))))
    peak_half = float(np.nanmax(np.abs(np.asarray(field_half.values, dtype=np.float64))))
    extent_error = float(np.max(np.abs(np.subtract(field_half.extent, field_full.extent)))) / _RECTANGLE[0]
    errors = {'drive_probe': float(np.max(np.abs(half_drive - full_drive))) / float(np.max(np.abs(full_drive))),
              'far_probe': float(np.max(np.abs(half_far - full_far))) / float(np.max(np.abs(full_far))),
              'energy': abs(energy_half / energy_full - 1.0),
              'snapshot_extent': extent_error, 'snapshot_field': _snapshot_field_error(field_full, field_half),
              'sectors': abs(len(labels) - 2)}
    tolerances = {'drive_probe': _DRIVEN_TOLERANCE, 'far_probe': _DRIVEN_TOLERANCE, 'energy': _DRIVEN_TOLERANCE,
                  'snapshot_extent': 1e-9, 'snapshot_field': _DRIVEN_SNAPSHOT_TOLERANCE, 'sectors': 0.5}
    return scaled_worst_result(errors, tolerances, labels=labels, energy_full=energy_full, energy_half=energy_half,
                               raster_peak_full=peak_full, raster_peak_half=peak_half,
                               snapshot_label=half.summary().get('snapshot_labels', [None])[index])


# ---- membrane_parity_sectors_port: the tunnelling guide of membrane_port_tunnelling at 8.1 MHz with incident modes 1
# (even about the centre line) and 2 (odd, phase 90 deg) entering together: the full model (10 port modes, mirror_x
# mesh) vs the half x <= W1 / 2 (symmetry on x_hi, 5 port modes = the unfolded modes 1..10) with parity_sectors: the
# combined outgoing amplitudes of both ports (unfolded numbering) to 1e-3 of A, the power fractions to 1e-3 and the
# combined power balance T + R = 1 to 1e-6 (first estimates).
_PORT_TOLERANCE = 1e-3
_PORT_BALANCE_TOLERANCE = 1e-6
_PORT_MODES = 10


def _port_specs():
    specs = {'full': _tunnel_spec('filter'), 'half': _tunnel_spec('filter', half=True)}
    for name, spec in specs.items():
        count = _PORT_MODES if name == 'full' else _PORT_MODES // 2
        for feature in spec.physics[0].features:
            if feature.name == 'in_port':
                feature.params = {'mode_count': count, 'incident_modes': [1.0, 2.0],
                                  'incident_amplitudes': [_TUNNEL_AMPLITUDE, _TUNNEL_AMPLITUDE],
                                  'incident_phases': [0.0, 90.0]}
            elif feature.name == 'out_port':
                feature.params = {'mode_count': count}
        spec.study.settings['snapshots'] = 'last'
        spec.title += ', incident modes 1 and 2'
    specs['full'].meshes = [MeshSpec(name=_TRIANGLE_MESH, kind='triangles', mode='uniform', dx=_TUNNEL_H,
                                     mirror_x=True)]
    specs['half'].study.settings['parity_sectors'] = True
    return specs


def _check_port(results_by_name):
    full, half = results_by_name['full'], results_by_name['half']
    errors = {'amplitude': 0.0, 'power_fraction': 0.0}
    balance, details = 0.0, {}
    for port in ('in_port', 'out_port'):
        reference = full.result(_PHYSICS, port)['series']
        combined = half.sector_result(_PHYSICS, port, combined=True)
        series = combined['series']
        for number in range(1, _PORT_MODES + 1):
            name = f'mode_{number}'
            errors['amplitude'] = max(errors['amplitude'], abs(_phasor(series, name, 0) - _phasor(reference, name, 0))
                                      / _TUNNEL_AMPLITUDE)
            fraction = float(np.asarray(series[f'power_fraction_{number}'])[0])
            errors['power_fraction'] = max(errors['power_fraction'], abs(
                fraction - float(np.asarray(reference[f'power_fraction_{number}'])[0])))
            balance += fraction
        details[port] = {'sector_modes': combined.get('sector_modes'),
                         'fractions': [float(np.asarray(series[f'power_fraction_{number}'])[0])
                                       for number in range(1, _PORT_MODES + 1)]}
    errors['balance'] = abs(balance - 1.0)
    errors['numbering'] = float(details['in_port']['sector_modes'] != list(range(1, _PORT_MODES + 1)))
    tolerances = {'amplitude': _PORT_TOLERANCE, 'power_fraction': _PORT_TOLERANCE,
                  'balance': _PORT_BALANCE_TOLERANCE, 'numbering': 0.5}
    return scaled_worst_result(errors, tolerances, **details)


# ---- membrane_parity_sectors_eigen: the half supported plate rectangle of membrane_plate_mirror_half (symmetry on x_hi)
# with parity_sectors: the even and the odd sector each give their _MIRROR_MODES lowest modes; together the lowest
# _SUPPORTED_MODES are those of the full rectangle (closed form, its tolerance), every snapshot label carries its
# parity tag and n_resonances sums the sectors.
def _eigen_specs():
    half = _mirror_specs()['rectangle_even']
    half.title = 'Self-test: half supported plate rectangle, parity sectors'
    half.study.settings['parity_sectors'] = True
    return {'half': half}


def _check_eigen(results_by_name):
    half = results_by_name['half']
    tables = [table for table in half.sector_result(_PHYSICS, _TABLE) if table is not None]
    measured = sorted(float(row[_FREQUENCY_COLUMN]) for table in tables for row in table['rows'])
    expected = _supported_frequencies()
    if len(measured) < len(expected):
        raise ValueError(f'expected {len(expected)} resonances over the sectors, got {len(measured)}')
    summary = half.summary()
    labels = summary.get('snapshot_labels') or []
    errors = {'frequency': max(abs(value / reference - 1.0) for value, reference in zip(measured, expected)),
              'n_resonances': abs(int(summary.get('n_resonances') or 0) - 2 * _MIRROR_MODES),
              'untagged_labels': sum(1 for label in labels if '[x_hi ' not in str(label)),
              'sectors': abs(len(tables) - 2)}
    tolerances = {'frequency': _SUPPORTED_TOLERANCE, 'n_resonances': 0.5, 'untagged_labels': 0.5, 'sectors': 0.5}
    return scaled_worst_result(errors, tolerances, measured_hz=measured, expected_hz=expected, labels=labels)


MEMBRANE_PARITY_SECTOR_CASES = (
    SelfTestCase(
        'membrane_parity_sectors_driven', 'Membrane: parity sectors of a driven half rectangle',
        'The clamped template rectangle (P2 h = a / 60) driven by a point force at (0.3 a, 0.4 b) at 400, 600 and 780 '
        'kHz: the full model on a mirror_x mesh against the half x <= a / 2 with a symmetry face on x_hi and the '
        'frequency_response setting parity_sectors (two sector runs on the same half mesh): the combined drive-point '
        'probe, the probe at (0.8 a, 0.6 b) from the signed sector probes at its image, the |w|^2 integral (quadratic '
        f'total) to {_DRIVEN_TOLERANCE:g}, the combined snapshot\'s unfolded extent and its raster against the full one '
        '(full raster bilinear at the half\'s pixel centres, pixels within 4 h of the drive point or its image left '
        f'out: the point-force log spike) to {_DRIVEN_SNAPSHOT_TOLERANCE:g} of the compared peak. First estimates.',
        _driven_specs, _check_driven),
    SelfTestCase(
        'membrane_parity_sectors_port', 'Membrane: parity sectors of the tunnelling guide with two incident modes',
        'The membrane_port_tunnelling guide at 8.1 MHz with incident modes 1 and 2 (A each, phases 0 / 90 deg) at the '
        'input port: the full model (10 port modes, mirror_x mesh) against the half guide (symmetry on x_hi, 5 modes '
        'per port = the unfolded modes 1..10) with parity_sectors (mode 1 driven at 2 A in the even sector, mode 2 in '
        f'the odd one): combined outgoing amplitudes of both ports ({_PORT_TOLERANCE:g} of A), power fractions '
        f'({_PORT_TOLERANCE:g}), the combined T + R = 1 ({_PORT_BALANCE_TOLERANCE:g}) and the unfolded numbering 1..10. '
        'First estimates.',
        _port_specs, _check_port),
    SelfTestCase(
        'membrane_parity_sectors_eigen', 'Membrane: eigen parity sectors of the half plate rectangle',
        'The half simply supported plate rectangle of membrane_plate_mirror_half with the eigenfrequency setting '
        f'parity_sectors: the {_MIRROR_MODES} modes of each sector together give the {_SUPPORTED_MODES} lowest of the '
        f'full rectangle (closed form, {_SUPPORTED_TOLERANCE:g}), n_resonances = {2 * _MIRROR_MODES} and every '
        "snapshot label tagged '[x_hi symmetric]' / '[x_hi antisymmetric]'.",
        _eigen_specs, _check_eigen),
)
