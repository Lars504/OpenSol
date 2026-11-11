"""Self-tests of the em_fdtd_3d mirror faces (symmetry 2.5): a dipole pair in a full vacuum cube against one dipole in
its x_lo-mirrored half (flux box, unfolded z-cut fields, unfolded far field), and a TFSF plane-wave box opening onto
an x_lo / y_lo mirror plane against the full box (scattered flux box, a mid-pulse z-cut snapshot)."""
import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import max_relative_error, worst_result
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           StudySpec)

_PHYSICS_NAME = 'em'
_PHYSICS_TYPE = 'em_fdtd_3d'          # literal (selftests never import the definition)
# Dipole pair: uniform 40 nm cells, 8 CPML cells per cpml face. Full cube 0.96 um (24 cells per axis), symmetric about
# the cell edge x = 0.48 um (edge 12): z dipoles (gaussian 1 um) at x = 0.30 / 0.66 um (cell centres 7 / 16), y = z =
# 0.48 um (the z face on edge 12, the y cell 12), the image dipole with moment +-1e-12 A m (symmetric: in phase,
# antisymmetric: opposite); flux box x, y, z in [0.12, 0.84] um. Half: x size 0.48 um with the mirror on x_lo (ghost
# cell outside x = 0), one dipole at x = 0.18 um (cell 4), flux box x in [0, 0.36] um (touching the plane:
# mirror_factor 2). The half run is the exact even / odd restriction of the full one (same dt and steps) for any cube
# size and duration, so flux and fields agree to float32 round-off. Cells 40^3 (full) and 21 x 40 x 40 (half), ~330
# steps per run.
_CELL = 40e-9                         # m
_PAIR_SIZE = 0.96e-6                  # m, full cube
_PAIR_PLANE = 0.48e-6                 # m, x of the full model's symmetry plane
_PAIR_BOX = (0.12e-6, 0.84e-6)        # m, full box on every axis (half: x from 0 to 0.36 um)
_PAIR_PML_CELLS = 8
# s: pulse 2 t0 = 19.1 fs + the validation's domain transit (5.5 fs, full cube); the end-of-run z cuts are less residual
_PAIR_DURATION = 25e-15
_DIPOLE_OFFSET = 0.18e-6              # m, dipoles at plane -+ it
_MOMENT = 1e-12                       # A m
_PULSE = {'dipole_axis': 'z', 'waveform': 'gaussian', 'wavelength': 1.0e-6, 'bandwidth': 0.5}
_BAND = {'wavelength_start': 0.9e-6, 'wavelength_stop': 1.1e-6, 'n_wavelengths': 3}
# TFSF mirror cube (below): 1.2 um (30 cells, 10 CPML cells per face), symmetric about the cell edge 0.6 um
_SIZE = 1.2e-6                        # m, full cube
_PLANE = 0.6e-6                       # m, coordinate of the full model's symmetry plane
_BOX = (0.2e-6, 1.0e-6)               # m, full flux box on every axis
_TOLERANCE = 1e-4                     # relative (float32 stepping, spectra summed in another order)
# mirror type -> the z-cut fields compared after unfolding (exact ones: E_z / E_y unstaggered across x, and H_z, whose
# plane face is pinned to 0 by the symmetric mirror)
_FIELDS = {'symmetric': ('ez', 'hz'), 'antisymmetric': ('ez', 'ey')}
# far fields of the flux box in two cut planes with x components (the half: its box currents plus their x images, the
# plane face dropped; the full box is the half box and its image, so the patterns agree to round-off)
_FAR_FIELD_PLANES = ('xz', 'xy')
_FAR_FIELD_ANGLES = 72


def _mirror_spec(mirror, half):
    """Full dipole-pair cube, or (half) its half with an x_lo mirror of type `mirror`."""
    centre = 0.5 * _PAIR_SIZE
    lo, hi = _PAIR_BOX
    offset = 0.0 if half else _PAIR_PLANE
    dipoles = [('dipole_plus', offset + _DIPOLE_OFFSET, _MOMENT)]
    if not half:
        dipoles.append(('dipole_minus', offset - _DIPOLE_OFFSET, _MOMENT if mirror == 'symmetric' else -_MOMENT))
    geometry = [GeometrySpec('flux_surface', 'box', {'x0': 0.0 if half else lo, 'y0': lo, 'z0': lo,
                                                     'x1': hi - _PAIR_PLANE if half else hi, 'y1': hi, 'z1': hi})]
    features = [FeatureSpec('radiated_power', 'flux_box', ['flux_surface'], dict(_BAND))]
    features += [FeatureSpec(f'far_field_{plane}', 'far_field', ['flux_surface'],
                             dict(_BAND, cut_plane=plane, n_angles=_FAR_FIELD_ANGLES)) for plane in _FAR_FIELD_PLANES]
    for name, x, moment in dipoles:
        geometry.append(GeometrySpec(f'{name}_point', 'point', {'x': x, 'y': centre, 'z': centre}))
        features.append(FeatureSpec(name, 'dipole_source', [f'{name}_point'], dict(_PULSE, current_moment=moment)))
    settings = {'pml_cells': _PAIR_PML_CELLS}
    if half:
        settings['bc_x_lo'] = mirror
    variant = f'half, {mirror} x_lo' if half else 'full'
    return ModelSpec(
        title=f'Self-test: em_fdtd_3d mirror ({variant})',
        domain=DomainSpec(size_x=_PAIR_PLANE if half else _PAIR_SIZE, size_y=_PAIR_SIZE, size_z=_PAIR_SIZE),
        geometry=geometry,
        physics=[PhysicsSpec(_PHYSICS_NAME, _PHYSICS_TYPE, settings=settings, features=features)],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type='time_dependent', settings={'duration': _PAIR_DURATION}))


def _mirror_specs():
    return {f'{mirror}_{variant}': _mirror_spec(mirror, variant == 'half')
            for mirror in _FIELDS for variant in ('full', 'half')}


def _check_mirror(results_by_name):
    """Per mirror type: flux box power (half x mirror_factor 2) vs full, unfolded mid-height fields and far-field
    patterns (both cut planes) vs full."""
    errors, details = {}, {}
    for mirror, components in _FIELDS.items():
        full, half = results_by_name[f'{mirror}_full'], results_by_name[f'{mirror}_half']
        half_flux, full_flux = (results.result(_PHYSICS_NAME, 'radiated_power') for results in (half, full))
        errors[f'{mirror}_flux'] = max_relative_error(half_flux['power'], full_flux['power'])
        errors[f'{mirror}_mirror_factor'] = abs(float(half_flux['mirror_factor']) - 2.0)
        details[f'{mirror}_flux_power'] = np.asarray(full_flux['power'], dtype=np.float64).tolist()
        for component in components:
            name = f'{_PHYSICS_NAME}.{component}'
            errors[f'{mirror}_{component}'] = max_relative_error(half.field(name, unfold=True).values,
                                                                 full.field(name).values)
        for plane in _FAR_FIELD_PLANES:
            feature = f'far_field_{plane}'
            errors[f'{mirror}_{feature}'] = max_relative_error(half.result(_PHYSICS_NAME, feature)['power'],
                                                               full.result(_PHYSICS_NAME, feature)['power'])
    return worst_result(errors, _TOLERANCE, **details)


# ---- TFSF box opening onto a mirror plane ----
# Uniform 40 nm cells, 10 CPML cells per cpml face, full 1.2 um cube symmetric about the cell edge 0.6 um of the
# mirrored axis, with a +z plane wave polarised along x (gaussian 1 um, amplitude 1 V/m) on the TFSF box [0.3, 0.9]
# um^3 around an eps 4 cube [0.48, 0.72] um^3, and a flux box [0.2, 1.0] um^3 in the scattered field. The x plane is
# normal to the polarisation (E_x even: antisymmetric), the y plane to the third axis (symmetric). One full run (both
# mirrored axes compare against it). Half: the mirrored axis 0.6 um long with that mirror on its lo face; the TFSF and
# flux boxes reach the plane (the TFSF box opens there, the flux box doubles). The half is the exact restriction of
# the full run, up to float32 rounding (fused multiply-adds round mirror images differently: spectra agree to ~1e-7),
# for any duration (the flux spectra of both share its truncation). Fields are compared in the 12 fs snapshot (the
# pulse peak, injected at t0 = 9.5 fs, has just crossed the mid-height cut): a field after the pulse has left is a
# residual ~4e-4 of the 1 V/m pulse (its E_x / E_y far less), where that rounding reaches 1e-2.
_TFSF_MIRRORS = {'x': 'antisymmetric', 'y': 'symmetric'}
# mirrored axis -> mid-height z-cut snapshot fields compared on the half window (the full one's upper half)
_TFSF_FIELDS = {'x': ('ez', 'ey'), 'y': ('ex', 'ez')}
_TFSF_FULL_FIELDS = ('ex', 'ey', 'ez')   # stored by the full run: every half's components
_TFSF_BOX = (0.3e-6, 0.9e-6)          # m, every axis
_TFSF_SCATTERER = (0.48e-6, 0.72e-6)  # m, every axis
_TFSF_EPS = 4.0
_TFSF_PULSE = {'direction': '+z', 'polarization': 'x', 'waveform': 'gaussian', 'wavelength': 1.0e-6,
               'bandwidth': 0.5, 'amplitude': 1.0}
_TFSF_DURATION = 24e-15               # s: the pulse (2 t0 ~19 fs) has crossed the box
_TFSF_OUTPUT_INTERVAL = 12e-15        # s: snapshots at 12 fs and the final 24 fs
_TFSF_FIELD_TIME = 12e-15             # s, compared snapshot


def _tfsf_box(name, bounds, axis, half):
    """Box object over bounds on every axis; along the mirrored axis (half) from the plane, shifted to the half."""
    corners = {}
    for coordinate in 'xyz':
        low, high = bounds
        if half and coordinate == axis:
            low, high = max(low - _PLANE, 0.0), high - _PLANE
        corners[f'{coordinate}0'], corners[f'{coordinate}1'] = low, high
    return GeometrySpec(name, 'box', corners)


def _tfsf_mirror_spec(axis):
    """Full TFSF cube model (axis None), or its half with the _TFSF_MIRRORS mirror on the lo face of `axis`."""
    half = axis is not None
    sizes = {coordinate: _PLANE if half and coordinate == axis else _SIZE for coordinate in 'xyz'}
    settings = {'pml_cells': 10}
    if half:
        settings[f'bc_{axis}_lo'] = _TFSF_MIRRORS[axis]
    features = [FeatureSpec('plane_wave', 'plane_wave_source', ['tfsf_region'], dict(_TFSF_PULSE)),
                FeatureSpec('scattered_power', 'flux_box', ['flux_surface'], dict(_BAND))]
    variant = f'half, {_TFSF_MIRRORS[axis]} {axis}_lo' if half else 'full'
    stored = _TFSF_FIELDS[axis] if half else _TFSF_FULL_FIELDS
    return ModelSpec(
        title=f'Self-test: em_fdtd_3d TFSF box onto a mirror ({variant})',
        domain=DomainSpec(size_x=sizes['x'], size_y=sizes['y'], size_z=sizes['z']),
        geometry=[_tfsf_box('tfsf_region', _TFSF_BOX, axis, half),
                  _tfsf_box('scatterer', _TFSF_SCATTERER, axis, half),
                  _tfsf_box('flux_surface', _BOX, axis, half)],
        materials=[MaterialSpec('glass', properties={'eps_r': _TFSF_EPS}, selection=['scatterer'])],
        physics=[PhysicsSpec(_PHYSICS_NAME, _PHYSICS_TYPE, settings=settings, features=features)],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type='time_dependent', settings={'duration': _TFSF_DURATION,
                                                         'output_interval': _TFSF_OUTPUT_INTERVAL},
                        stored_fields=[f'{_PHYSICS_NAME}.{component}' for component in stored]))


def _tfsf_mirror_specs():
    return {'full': _tfsf_mirror_spec(None), **{f'{axis}_half': _tfsf_mirror_spec(axis) for axis in _TFSF_MIRRORS}}


def _snapshot_near(results, time):
    """Index of the snapshot nearest `time` [s] (summary snapshot_times)."""
    times = results.summary()['snapshot_times']
    return min(range(len(times)), key=lambda index: abs(float(times[index]) - time))


def _check_tfsf_mirror(results_by_name):
    """Per mirrored axis: scattered flux box power (half x mirror_factor 2) and the 12 fs mid-height z-cut snapshot
    fields of the half vs the (one) full run's on the half window."""
    errors, details = {}, {}
    full = results_by_name['full']
    for axis, components in _TFSF_FIELDS.items():
        half = results_by_name[f'{axis}_half']
        half_flux, full_flux = (results.result(_PHYSICS_NAME, 'scattered_power') for results in (half, full))
        errors[f'{axis}_flux'] = max_relative_error(half_flux['power'], full_flux['power'])
        errors[f'{axis}_mirror_factor'] = abs(float(half_flux['mirror_factor']) - 2.0)
        details[f'{axis}_scattered_power'] = np.asarray(full_flux['power'], dtype=np.float64).tolist()
        full_index, half_index = (_snapshot_near(results, _TFSF_FIELD_TIME) for results in (full, half))
        details[f'{axis}_field_times'] = [float(results.summary()['snapshot_times'][index])
                                          for results, index in ((full, full_index), (half, half_index))]
        axis_index = 'xy'.index(axis)
        for component in components:
            name = f'{_PHYSICS_NAME}.{component}'
            half_values = np.asarray(half.snapshot_field(half_index, name).values)
            full_values = np.asarray(full.snapshot_field(full_index, name).values)
            restricted = np.take(full_values, np.arange(full_values.shape[axis_index] - half_values.shape[axis_index],
                                                        full_values.shape[axis_index]), axis=axis_index)
            errors[f'{axis}_{component}'] = max_relative_error(half_values, restricted)
    return worst_result(errors, _TOLERANCE, **details)


MIRROR_3D_CASES = (
    SelfTestCase(
        name='em3d_mirror_dipole_pair',
        label='EM 3D: mirror faces, half vs full domain (FDTD)',
        description=f'em_fdtd_3d vacuum cube {_PAIR_SIZE * 1e6:g} um (40 nm cells, {_PAIR_PML_CELLS} CPML cells per '
                    f'face, {_PAIR_DURATION * 1e15:g} fs) with two z dipoles at x = {_PAIR_PLANE * 1e6:g} -+ 0.18 um '
                    '(gaussian 1 um, 1e-12 A m; the image in phase for the symmetric, opposite for the antisymmetric '
                    f'case) against its half {_PAIR_PLANE * 1e6:g} um wide with that mirror on x_lo and one dipole: the '
                    'half run is the exact restriction of the full one. The flux box power (0.9-1.1 um) of the half '
                    'box touching the plane, doubled (mirror_factor 2), equals the full box power; the mid-height z '
                    'cut of E_z and H_z (symmetric) / E_z and E_y (antisymmetric) unfolded by '
                    'Results.field(unfold=True) equals the full one; the far-field patterns of the box (x-z and x-y '
                    'cuts, 72 angles: the half box currents plus their x images, its plane face dropped) equal the '
                    f'full box patterns. error = the largest relative deviation, tolerance {_TOLERANCE:g} (float32 '
                    'round-off); a wrong ghost row or image sign shows up at O(1). Not yet run with this cube and '
                    'duration.',
        specs=_mirror_specs,
        check=_check_mirror),
    SelfTestCase(
        name='em3d_tfsf_mirror',
        label='EM 3D: TFSF box opening onto a mirror plane, half vs full (FDTD)',
        description='em_fdtd_3d vacuum cube 1.2 um (40 nm cells, 10 CPML cells per face): a +z plane wave polarised '
                    'along x (gaussian 1 um, 1 V/m) on the TFSF box [0.3, 0.9] um^3 around an eps 4 cube [0.48, 0.72] '
                    'um^3, scattered power on the flux box [0.2, 1.0] um^3 (0.9-1.1 um), against its half 0.6 um '
                    'along x with an antisymmetric x_lo mirror (plane normal to the polarisation) and, separately, '
                    'along y with a symmetric y_lo mirror (plane normal to the third axis): the TFSF box opens onto '
                    'the plane (core.mirror.plane_wave_mirror_issue), its image completes it, so the half run is '
                    'the exact restriction of the full one. The doubled scattered flux (mirror_factor 2) equals the '
                    'full one, and the 12 fs mid-height z-cut snapshot of E_z, E_y (x mirror) / E_x, E_z (y mirror) '
                    'equals the full one on the half window (the end-of-run field is a residual far below the pulse, '
                    'where float32 round-off is not negligible). One full run (storing E_x, E_y, E_z) serves both '
                    f'halves; {_TFSF_DURATION * 1e15:g} fs runs. error = the largest relative deviation, '
                    f'tolerance {_TOLERANCE:g} (float32 round-off); a missing or doubled box face on the plane shows '
                    'up as the incident wave leaking into the scattered field, at O(1). Not yet run with the shared '
                    'full run and this duration.',
        specs=_tfsf_mirror_specs,
        check=_check_tfsf_mirror),
)
