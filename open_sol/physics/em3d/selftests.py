"""Self-test helpers and template of em_fdtd_3d: a z dipole pulse in a vacuum CPML cube, its radiated cw power through a
closed flux box against the analytic eta0 k^2 (I l)^2 / (12 pi) (graded-mesh case here; the uniform-mesh check runs
in em3d_far_field_dipole, selftests_sources3d)."""
import math

import numpy as np

from ...core.constants import C0, ETA0
from ...model.schema import SelfTestCase
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MeshSpec, ModelSpec, PhysicsSpec, RefinementSpec,
                           StudySpec)

_PHYSICS_NAME = 'em'
_PHYSICS_TYPE = 'em_fdtd_3d'          # literal (selftests never import the definition)
DEFAULT_COURANT = 0.99                # default courant_factor (fraction of the 3D CFL limit)
_FLUX_FEATURE = 'radiated_power'
_CELL = 40e-9                         # m: 25 cells per wavelength at 1 um
_SIZE = 1.2e-6                        # m cube (30 cells) + 10 CPML cells per face
_PML_CELLS = 10
_MOMENT = 1e-12                       # A m
_PULSE = {'dipole_axis': 'z', 'waveform': 'gaussian', 'wavelength': 1.0e-6, 'bandwidth': 0.5,
          'current_moment': _MOMENT}
_BAND = (0.9e-6, 1.1e-6, 3)           # m, m, count of the flux box wavelengths
_BOX = (0.2e-6, 1.0e-6)               # m: box faces 10 cells from the dipole, 5 from the CPML
_DURATION = 40e-15                    # s, template: pulse 2 t0 ~19 fs plus the transit out of the box
# s, self-test runs: 2 t0 = 19.1 fs + the validation's domain transit 6.9 fs = 26 fs; the DFT end term of the dipole's
# leftover static charge does not depend on the duration
SELFTEST_DURATION = 28e-15
DIPOLE_TOLERANCE = 0.05
_GRADED_CELL = 25e-9                  # m: dipole-point refinement (x, y and z bands; 8 cells over the 0.2 um band)
_GRADED_REACH = 0.1e-6                # m: refinement margin around the point (inside the flux box)

DIPOLE_TEMPLATE_KEY = 'dipole_radiation_3d'
DIPOLE_TEMPLATE_LABEL = 'Dipole radiation (3D)'
DIPOLE_TEMPLATE_DESCRIPTION = (
    'em_fdtd_3d: a z-oriented electric dipole (1e-12 A m, gaussian pulse at 1 um) in a 1.2 um vacuum cube with CPML on '
    'all six faces; a flux box around it gives the radiated cw power, analytically eta0 k^2 (I l)^2 / (12 pi).')


def dipole_vacuum_spec():
    """Template 'dipole_radiation_3d' (the self-tests shorten its duration: dipole_selftest_spec)."""
    centre = 0.5 * _SIZE
    lo, hi = _BOX
    return ModelSpec(
        title='Dipole radiation (3D)',
        description=DIPOLE_TEMPLATE_DESCRIPTION,
        domain=DomainSpec(size_x=_SIZE, size_y=_SIZE, size_z=_SIZE),
        geometry=[GeometrySpec('dipole_point', 'point', {'x': centre, 'y': centre, 'z': centre}),
                  GeometrySpec('flux_surface', 'box', {'x0': lo, 'y0': lo, 'z0': lo, 'x1': hi, 'y1': hi, 'z1': hi})],
        physics=[PhysicsSpec(_PHYSICS_NAME, _PHYSICS_TYPE, settings={'pml_cells': _PML_CELLS}, features=[
            FeatureSpec('dipole', 'dipole_source', ['dipole_point'], dict(_PULSE)),
            FeatureSpec(_FLUX_FEATURE, 'flux_box', ['flux_surface'],
                        {'wavelength_start': _BAND[0], 'wavelength_stop': _BAND[1], 'n_wavelengths': _BAND[2]})])],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type='time_dependent', settings={'duration': _DURATION}))


def dipole_selftest_spec():
    """The dipole template run for SELFTEST_DURATION."""
    spec = dipole_vacuum_spec()
    spec.study.settings['duration'] = SELFTEST_DURATION
    return spec


def _graded_dipole_spec():
    """The dipole self-test model on a graded mesh: the dipole point refined to _GRADED_CELL on x, y and z (graded z)."""
    spec = dipole_selftest_spec()
    spec.title = 'Self-test: dipole radiation (3D), graded mesh'
    spec.mesh = MeshSpec(mode='nonuniform', dx=_CELL,
                         refinements=[RefinementSpec('dipole_point', _GRADED_CELL, _GRADED_REACH)])
    return spec


def _check_graded_dipole_power(results_by_name):
    """Flux box power vs the closed form (dipole_flux_errors), plus a raise unless the build has more z cells than the
    uniform grid and dx_min ~ _GRADED_CELL (graded z active)."""
    results = results_by_name['run']
    build = results.summary().get('build') or {}
    uniform_nz = round(_SIZE / _CELL) + 2 * _PML_CELLS
    if not (int(build.get('nz', 0)) > uniform_nz and abs(float(build.get('dx_min', 0.0)) / _GRADED_CELL - 1.0) < 0.05):
        raise ValueError(f'expected a graded z axis (nz > {uniform_nz}, dx_min {_GRADED_CELL:g} m), got build '
                         f"nz {build.get('nz')}, dx_min {build.get('dx_min')}")
    error, details = dipole_flux_errors(results)
    worst = int(np.argmax(details['relative_errors']))
    details['power_au'] = np.asarray(results.result(_PHYSICS_NAME, _FLUX_FEATURE)['power'], dtype=np.float64).tolist()
    details.update(nz=int(build['nz']), dx_min=float(build['dx_min']))
    return {'measured': details['power_per_cw'][worst], 'expected': details['expected'][worst], 'error': error,
            'tolerance': DIPOLE_TOLERANCE, 'details': details}


def cubic_time_step(cell, courant=DEFAULT_COURANT):
    """3D CFL time step [s] courant x cell / (c0 sqrt 3) of a uniform cubic grid."""
    return courant * cell / (C0 * math.sqrt(3.0))


def dipole_flux_errors(results, power_ratio=1.0):
    """(worst relative error, details) of the 'radiated_power' flux box power_per_cw against power_ratio x eta0 k0^2
    (I l)^2 / (12 pi) of the 1e-12 A m dipole."""
    result = results.result(_PHYSICS_NAME, _FLUX_FEATURE)
    wavelengths = np.asarray(result['wavelength'], dtype=np.float64)
    measured = np.asarray(result.get('power_per_cw', np.full(wavelengths.shape, np.nan)), dtype=np.float64)
    expected = power_ratio * float(ETA0) * (2.0 * math.pi / wavelengths) ** 2 * _MOMENT ** 2 / (12.0 * math.pi)
    errors = np.abs(measured / expected - 1.0)
    errors = np.where(np.isfinite(errors), errors, np.inf)
    return float(np.max(errors)), {'wavelengths': wavelengths.tolist(), 'power_per_cw': measured.tolist(),
                                   'expected': expected.tolist(), 'relative_errors': errors.tolist()}


SELFTEST_CASES = (
    SelfTestCase(
        name='em3d_dipole_radiated_power_graded',
        label='EM 3D: dipole radiated power on a graded mesh (graded z)',
        description=(f'The dipole_radiation_3d model ({SELFTEST_DURATION * 1e15:g} fs) with mesh nonuniform: the '
                     f'dipole point refined to {_GRADED_CELL * 1e9:g} nm (margin {_GRADED_REACH * 1e6:g} um) on x, y '
                     f'and z, {_CELL * 1e9:g} nm at the flux box and the CPML: unequal cell widths along z at the '
                     'dipole face (J = I l / V), the graded z ramp and graded-z dt. power_per_cw of the flux box '
                     '(0.9-1.1 um) vs the Hertzian-dipole eta0 k^2 (I l)^2 / (12 pi), worst relative error, tolerance '
                     f'{DIPOLE_TOLERANCE:g} (lattice dispersion and box quadrature ~1-2 %). The check raises unless the '
                     'build has more z cells than the uniform grid and dx_min is the refinement size within 5 % '
                     f'(segment rescaling). Not yet run at {_GRADED_CELL * 1e9:g} nm / {SELFTEST_DURATION * 1e15:g} '
                     'fs.'),
        specs=lambda: {'run': _graded_dipole_spec()},
        check=_check_graded_dipole_power),
)
DIPOLE_TEMPLATE = (DIPOLE_TEMPLATE_KEY, DIPOLE_TEMPLATE_LABEL, DIPOLE_TEMPLATE_DESCRIPTION, dipole_vacuum_spec,
                   'A tiny antenna in empty space and the power it radiates.')
