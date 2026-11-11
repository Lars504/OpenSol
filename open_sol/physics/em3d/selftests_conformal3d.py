"""Self-tests of the em_fdtd_3d conformal PEC (solver_defaults.conformal_boundaries, solver_plan S7.6): the TM101
resonance of a vacuum sphere cut out of a PEC block from the host operator of the conformal coefficients (closed form,
largest eigenvalue against the leap-frog limit) and the FDTD ring-down against that operator (gpu and cpu backends)."""
import math

import numpy as np

from ...core.constants import C0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import failed_count_result
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MeshSpec, ModelSpec, PhysicsSpec, SolverDefaultsSpec,
                           StudySpec)

_PHYSICS_NAME = 'em'
_PHYSICS_TYPE = 'em_fdtd_3d'          # literal (selftests never import the definition)
_RESONANCE_FEATURE = 'resonances'
_CELL = 50e-9                         # m
_CELLS = 28                           # per axis; pec on all six faces (the block fills the wall layers anyway)
_SIZE = _CELLS * _CELL
_RADIUS = 11.37 * _CELL
_CENTRE = tuple(offset * _CELL for offset in (14.0 + 0.31, 14.0 + 0.17, 14.0 - 0.23))   # off the cell centres
_COURANT = 0.99
# TM101 of a PEC sphere: (x j1(x))' = 0 at x = k R (tan x = x / (1 - x^2)); triply degenerate, split by the grid
_TM101_ROOT = 2.743707269992
_BAND = (1.15e-6, 1.50e-6)            # m: TM101 ~1.302 um; TM201 ~0.923 um and TE101 ~0.795 um lie outside
_PULSE = {'dipole_axis': 'z', 'waveform': 'gaussian', 'wavelength': 1.30e-6, 'bandwidth': 0.3,
          'current_moment': 1e-12}
_SOURCE_OFFSET = (0.06, -0.04, 0.05)  # x radius from the centre: a z dipole near it rings mostly the z-like mode
_PROBE_OFFSET = (0.11, 0.08, -0.07)
_DURATION = 0.5e-12                   # s: ~115 periods of TM101, ~120 decimated samples
# mean TM101 omega vs k R = 2.7437: measured 9.5e-3 = Yee dispersion ~2e-3 plus the stability floor, which at courant
# 0.99 caps 3409 of the 4888 cut faces (weights ~<= 1); staircase 5.2e-2, so a lost correction still fails
_OPERATOR_TOLERANCE = 1.5e-2
_RING_DOWN_TOLERANCE = 1e-3           # FDTD frequency vs the leap-frog frequency of the host operator (same arrays)
_LARGEST_TOLERANCE = 1e-4             # ARPACK tolerance of the largest eigenvalue


def _cavity_spec(conformal, backend='gpu'):
    """Vacuum sphere in a PEC block (pec_domain 'metal' = block minus sphere), z dipole pulse, resonance_monitor."""
    points = {name: dict(zip('xyz', (centre + offset * _RADIUS for centre, offset in zip(_CENTRE, offsets))))
              for name, offsets in (('source_point', _SOURCE_OFFSET), ('probe_point', _PROBE_OFFSET))}
    settings = {f'bc_{axis}_{side}': 'pec' for axis in 'xyz' for side in ('lo', 'hi')}
    return ModelSpec(
        title=f'Self-test: conformal PEC sphere cavity (3D, conformal {conformal}, {backend})',
        domain=DomainSpec(size_x=_SIZE, size_y=_SIZE, size_z=_SIZE),
        geometry=[GeometrySpec('block', 'box', {'x0': 0.0, 'y0': 0.0, 'z0': 0.0, 'x1': _SIZE, 'y1': _SIZE,
                                                'z1': _SIZE}),
                  GeometrySpec('cavity', 'sphere', {'cx': _CENTRE[0], 'cy': _CENTRE[1], 'cz': _CENTRE[2],
                                                    'r': _RADIUS}),
                  GeometrySpec('metal', 'difference', {'base': 'block', 'subtract': ['cavity']}),
                  *(GeometrySpec(name, 'point', values) for name, values in points.items())],
        physics=[PhysicsSpec(_PHYSICS_NAME, _PHYSICS_TYPE, settings=settings, features=[
            FeatureSpec('metal', 'pec_domain', ['metal']),
            FeatureSpec('dipole', 'dipole_source', ['source_point'], dict(_PULSE)),
            FeatureSpec(_RESONANCE_FEATURE, 'resonance_monitor', ['probe_point'],
                        {'component': 'ez', 'wavelength_min': _BAND[0], 'wavelength_max': _BAND[1]})])],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        solver_defaults=SolverDefaultsSpec(backend=backend, conformal_boundaries=conformal),
        study=StudySpec(type='time_dependent', settings={'duration': _DURATION}))


def _cavity_dt():
    """Engine dt of the uniform cavity mesh (float32 widths, core.grid3d rule: fraction of the 3D CFL limit)."""
    cell = float(np.float32(_CELL))
    return float(np.float32(_COURANT / (C0 * math.sqrt(3.0) / cell)))


def _cavity_coefficients(dt, conformal):
    """(pec bool (n, n, n) as the build masks it at the float32 centres, widths, ConformalPEC3D or None): the outline
    test of the 3D geometry locator, physics.em3d.conformal_pec3d on the whole grid (no padding with pec faces)."""
    from ...model.boundary_locator import GeometryLocator
    from ...model.geometry import rasterize_box, resolve_geometry
    from .conformal_pec3d import build_conformal_pec_3d
    shapes = resolve_geometry(_cavity_spec(conformal), {})
    centres = (np.arange(_CELLS) + 0.5) * _CELL
    centres32 = centres.astype(np.float32)
    window, mask = rasterize_box('metal', shapes, centres32, centres32, centres32, np)
    pec = np.zeros((_CELLS,) * 3, dtype=bool)
    pec[window] = mask
    widths = np.full(_CELLS, float(np.float32(_CELL)))
    if not conformal:
        return pec, widths, None
    inside = GeometryLocator(shapes, [], dimensions=3).inside(['metal'])
    coefficients = build_conformal_pec_3d((widths,) * 3, (centres,) * 3, (slice(0, _CELLS),) * 3, (0, 0, 0), pec, pec,
                                          inside, dt)
    return pec, widths, coefficients


def _operator(pec, widths, coefficients):
    """Sparse C0^2 curl_h curl_e (omega^2 = its eigenvalues) on the live E faces of the pec-walled grid (operators3d:
    not between two PEC cells; the corrected H faces weighted 1 + extra, the engine arrays); host float64."""
    from .boundaries3d import BoundaryConfig3D
    from .operators3d import curl_curl_operator, live_unknowns_3d, weighted_curl_e, yee_operators_3d
    config = BoundaryConfig3D({f'{axis}_{side}': 'pec' for axis in 'xyz' for side in ('lo', 'hi')})
    ops = yee_operators_3d((widths,) * 3, config, (float(widths[0]),) * 3)
    live = live_unknowns_3d(pec.shape, config, tuple(slice(0, count) for count in pec.shape), pec)
    curl_e = None if coefficients is None else weighted_curl_e(ops, coefficients.face_cell, coefficients.face_axis,
                                                                coefficients.face_extra)
    return (C0 ** 2) * curl_curl_operator(ops, live, curl_e=curl_e).tocsc()


def _eigenvalues(dt, conformal, with_largest=True):
    """(the three eigenvalues [1/s^2] nearest TM101, the largest one (None unless with_largest), ConformalPEC3D or
    None) of the host operator."""
    from scipy.sparse.linalg import ArpackNoConvergence, eigs
    pec, widths, coefficients = _cavity_coefficients(dt, conformal)
    operator = _operator(pec, widths, coefficients)
    target = (C0 * _TM101_ROOT / _RADIUS) ** 2
    nearest = np.sort(np.real(eigs(operator, k=3, sigma=target, which='LM', return_eigenvectors=False)))
    if not with_largest:
        return nearest, None, coefficients
    try:
        largest = eigs(operator, k=1, which='LM', ncv=64, maxiter=20000, tol=_LARGEST_TOLERANCE,
                       return_eigenvectors=False)
    except ArpackNoConvergence as exc:   # the Ritz values found so far
        largest = exc.eigenvalues
    largest = float(np.max(np.abs(largest))) if np.size(largest) else math.inf
    return nearest, largest, coefficients


def _operator_checks(dt):
    """(checks, details, {(conformal, dt): the three TM101 eigenvalues}) of the host operators: mean TM101 omega of the
    conformal one vs c x 2.7437 / R (staircase in the details) and its largest eigenvalue times dt^2 / 4 below 1."""
    exact = C0 * _TM101_ROOT / _RADIUS
    checks, details, nearest_by_key = {}, {}, {}
    for conformal in (True, False):   # the staircase largest eigenvalue is never graded: not solved
        nearest, largest, coefficients = _eigenvalues(dt, conformal, with_largest=conformal)
        nearest_by_key[(conformal, dt)] = nearest
        omegas = np.sqrt(np.maximum(nearest, 0.0))
        error = abs(float(np.mean(omegas)) / exact - 1.0)
        label = 'conformal' if conformal else 'staircase'
        details[label] = {'tm101_error': error, 'split': float((omegas[-1] - omegas[0]) / np.mean(omegas))}
        if conformal:
            details[label].update(largest_eigenvalue_dt2_over_4=largest * dt ** 2 / 4.0,
                                  coefficients=coefficients.summary())
            checks['TM101_conformal'] = (error, _OPERATOR_TOLERANCE)
            checks['stability'] = (largest * dt ** 2 / 4.0, 1.0)
    return checks, details, nearest_by_key


def _leapfrog_frequency(eigenvalue, dt):
    """Frequency [Hz] of a spatial eigenvalue omega^2 under the leap-frog: sin(omega dt / 2) = dt sqrt(lambda) / 2."""
    return math.asin(min(1.0, 0.5 * dt * math.sqrt(max(eigenvalue, 0.0)))) / (math.pi * dt)


_RUNS = {'on': (True, 'gpu'), 'off': (False, 'gpu'), 'cpu_on': (True, 'cpu')}


def _cavity_specs():
    return {name: _cavity_spec(conformal, backend) for name, (conformal, backend) in _RUNS.items()}


def _check_cavity(results_by_name):
    """The operator checks (_operator_checks at the cavity dt), then per run present (gpu runs are skipped without a
    GPU): the ring-down frequency vs the leap-frog frequency of the nearest of the three TM101 eigenvalues of the host
    operator of the same coefficients (an engine check of the sparse correction kernels on both backends; the operator
    solves are reused when the run's dt equals the cavity dt); the closed-form error in the details."""
    exact_frequency = C0 * _TM101_ROOT / (2.0 * math.pi * _RADIUS)
    checks, operator_details, cached = _operator_checks(_cavity_dt())
    details = {'operator': operator_details}
    ring_down_checked = False
    for run, (conformal, backend) in _RUNS.items():
        if run not in results_by_name:
            continue
        table = results_by_name[run].result(_PHYSICS_NAME, _RESONANCE_FEATURE)
        columns = list(table['columns'])
        found = [float(row[columns.index('frequency')]) for row in table['rows']]
        dt = float(table['dt'])
        if (conformal, dt) not in cached:
            cached[(conformal, dt)] = _eigenvalues(dt, conformal, with_largest=False)[0]
        predicted = [_leapfrog_frequency(float(value), dt) for value in cached[(conformal, dt)]]
        error = min((abs(value / reference - 1.0) for value in found for reference in predicted), default=math.inf)
        checks[run] = (error, _RING_DOWN_TOLERANCE)
        ring_down_checked = True
        measured = min(found, key=lambda value: abs(value / predicted[0] - 1.0)) if found else math.nan
        details[run] = {'backend': backend, 'found': found, 'predicted': predicted, 'error': error,
                        'closed_form_error': abs(measured / exact_frequency - 1.0)}
    if not ring_down_checked:
        checks['runs'] = (math.inf, 0.0)
    return failed_count_result(checks, details)


_CAVITY_TEXT = (f'vacuum sphere of radius {_RADIUS * 1e9:g} nm (11.37 cells of {_CELL * 1e9:g} nm, centre (0.31, 0.17, '
                '-0.23) cells off a cell centre) cut out of a PEC block (pec_domain, a box minus the sphere; pec on '
                f'all six faces of the {_CELLS}^3 grid)')
CONFORMAL_3D_CASES = (
    SelfTestCase(
        name='em3d_conformal_pec_cavity',
        label='EM 3D conformal PEC: sphere cavity operator and ring-down',
        description=f'Operator (CPU, physics/em3d/conformal_pec3d, scipy sparse eigs): the E-field operator C0^2 D G '
                    f'of a {_CAVITY_TEXT}, with the Dey-Mittra H-face weights the engine gets. The mean omega of the '
                    f'three TM101 eigenvalues within {_OPERATOR_TOLERANCE:g} of c x 2.7437 / R ((x j1)\' = 0; '
                    'measured 9.5e-3: Yee dispersion ~2e-3 at 11.4 cells plus the stability floor, which at courant '
                    '0.99 caps ~70 % of the cut faces; staircase 5.2e-2) and the largest eigenvalue times dt^2 '
                    '/ 4 below 1 at courant 0.99 on the whole grid (the build tightens its floor by the same '
                    'eigenvalue on a 4-cell band around the cut faces; this checks that the band suffices); '
                    'staircase errors and the triplet split in the details. Ring-downs: em_fdtd_3d '
                    f'({_DURATION * 1e12:g} ps, z dipole pulse near the centre, resonance_monitor E_z), '
                    'conformal_boundaries on (gpu and cpu backend) and off (gpu). The TM101 frequency equals the '
                    'leap-frog frequency asin(dt sqrt(lambda) / 2) / (pi dt) of the nearest TM101 eigenvalue of the '
                    f'same operators within {_RING_DOWN_TOLERANCE:g} (an engine check of the sparse correction kernel '
                    'and its numpy twin; without a GPU only the cpu run is checked). Errors against the closed form '
                    'in the details. error = number of failed sub-checks and runs, tolerance 0. Not yet run at '
                    f'{_DURATION * 1e12:g} ps (the grid splits the triplet; a split above ~2e-3 blended by the fit '
                    'would show here).',
        specs=_cavity_specs,
        check=_check_cavity,
        heavy=True),
)
